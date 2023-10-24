"""
Updated on 10/10/2023

@author: adgrube
"""

import torch
import torch.nn as nn
from torch_sparse import spspmm, transpose
from torch.autograd import grad
import torch_geometric.utils as pygu
import graph_utils as gu
from attention import (SparseNodeEdgeAttentionLayer, 
                       SparseTwoCliqueAttentionLayer)
from utils import MLP, spmm
from data import ROOT_DIR


class ODEBlock(nn.Module):
    def __init__(self, odefunc_class, opt, data):
        super(ODEBlock, self).__init__()

        # Set odefunc for block and range for integration
        self.opt = opt
        self.odefunc = odefunc_class(opt, data)
        if opt['whole_trajectory']:
            tRange = torch.linspace(0, opt['final_time'], 501)
            self.register_buffer('tRange', tRange)
        else:
            self.register_buffer('tRange', torch.tensor([0, opt['final_time']]))

        # Option for adjoint backprop
        if opt['adjoint_backprop']:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # Option for different train and test integrators
        self.train_integrator = odeint
        self.test_integrator = odeint  #TODO not currently used
        self.set_tol()

    # Sets the initial condition for the integration
    def set_x0(self, x0):
        q0, p0 = x0
        self.odefunc.x0 = (q0.clone().detach(), p0.clone().detach())

    # Sets the tolerances of the integrator
    def set_tol(self):
        self.atol = self.opt['tol_scale'] * 1e-7
        self.rtol = self.opt['tol_scale'] * 1e-9
        if self.opt['adjoint_backprop']:
            self.atol_adjoint = self.opt['adjoint_tol_scale'] * 1e-7
            self.rtol_adjoint = self.opt['adjoint_tol_scale'] * 1e-9

    # Resets the tolerances of the integrator to default
    def reset_tol(self):
        self.atol = 1e-7
        self.rtol = 1e-9
        self.atol_adjoint = 1e-7
        self.rtol_adjoint = 1e-9

    # Integrates the RHS defined by odefunc  
    def forward(self, x):
        q, p = x
        t = self.tRange.type_as(q) # ???

        # Compute attention mats
        if (self.opt['constant_attention'] 
            and not self.opt['bracket'] == 'node'):
            self.odefunc.A0, self.odefunc.A1, self.odefunc.A2 = \
                self.odefunc.att_later_twoCl(q, self.odefunc.edge_index, 
                                             self.odefunc.d0_index, 
                                             self.odefunc.twoClique_index, 
                                             self.odefunc.d1_index)
            if not self.opt['consistent_A2']:
                self.odefunc.A0, self.odefunc.A1 = self.odefunc.att_layer_NE(
                    q, self.odefunc.edge_index, self.odefunc.d0_index)

        # Define integrator and function
        integrator = self.train_integrator if self.training \
                                           else self.test_integrator
        func =  self.odefunc
        x = (q, p)

        # Do the integration
        if self.opt['adjoint_backprop'] and self.training:
            x_init_and_final = integrator(func, x, t,
                method=self.opt['method'],
                options={'step_size': self.opt['step_size']},
                adjoint_method=self.opt['adjoint_method'],
                # adjoint_options={'step_size': self.opt['adjoint_step_size']},
                adjoint_options=dict(step_size=self.opt['adjoint_step_size'], 
                                     max_num_steps=self.opt['max_num_steps']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            x_init_and_final = integrator(func, x, t,
                method=self.opt['method'],
                # options={'step_size': self.opt['step_size']},
                options=dict(step_size=self.opt['step_size'], 
                             max_num_steps=self.opt['max_num_steps']),
                atol=self.atol,
                rtol=self.rtol)

        if self.opt['whole_trajectory']:
            # return q and p over whole trajectory
            return (x_init_and_final[0], x_init_and_final[1])
        else:
            # return q and p at final time
            return (x_init_and_final[0][1], x_init_and_final[1][1])


class ODEFunc(nn.Module):
    def __init__(self, opt, data):
        super().__init__()
        self.opt = opt
        self.nfe = 0

        # making sure hidden_dim is input dim when no encoder/decoder
        if opt['no_encoder_decoder']:
            opt['hidden_dim'] = data.x.shape[-1]

        # For checking the energy/entropy values
        if opt['verbose']:
            self.verbListE = []
            self.verbListS = []
        
        # This is a fudge factor which does not affect property preservation
        if opt['alpha_multiplier']:
            self.alpha = nn.Parameter(torch.ones(1))

        # Choice of whether to augment G with self-loops
        if opt['add_self_loops']:
            data.edge_index = pygu.add_self_loops(data.edge_index)[0]

        # Computing sparse mats d0, d1 (or load precomputed)
        loc = f"{ROOT_DIR}/data/{opt['dataset']}" + \
              f"/precomputed-SL_{opt['add_self_loops']}" + \
              f"lcc_{opt['use_lcc']}_dtype_{opt['dtype']}"
        try: 
            mats = torch.load(f'{loc}.pt')
            self.edge_index, self.nodes_to_edge = \
                mats['edge_index'], mats['nodes_to_edge']
            self.d0_index, self.d0_vals, self.d0_shape = \
                mats['d0_index'], mats['d0_vals'], mats['d0_shape']
            self.twoClique_index, self.nodes_to_twoCliques = \
                mats['twoClique_index'], mats['nodes_to_twoCliques']
            self.d1_index, self.d1_vals, self.d1_shape = \
                mats['d1_index'], mats['d1_vals'], mats['d1_shape']
            self.d0t_index, self.d0t_vals = transpose(
                self.d0_index, self.d0_vals, *self.d0_shape)
            self.d1t_index, self.d1t_vals = transpose(
                self.d1_index, self.d1_vals, *self.d1_shape)
        except:
            self.edge_index, self.nodes_to_edge = \
                gu.get_unique_edges(data.edge_index)
            self.d0_index, self.d0_vals, self.d0_shape = \
                gu.build_sparse_d0(self.edge_index)
            self.twoClique_index, self.nodes_to_twoCliques = \
                gu.get_2_cliques(self.edge_index)
            self.d1_index, self.d1_vals, self.d1_shape = \
                gu.build_sparse_d1(self.twoClique_index, self.nodes_to_edge)
            self.d0t_index, self.d0t_vals = transpose(
                self.d0_index, self.d0_vals, *self.d0_shape)
            self.d1t_index, self.d1t_vals = transpose(
                self.d1_index, self.d1_vals, *self.d1_shape) 
            dictionary = {'edge_index': self.edge_index, 
                          'nodes_to_edge': self.nodes_to_edge,
                          'd0_index': self.d0_index, 'd0_vals': self.d0_vals,
                          'd0_shape': self.d0_shape,
                          'twoClique_index': self.twoClique_index,
                          'nodes_to_twoCliques': self.nodes_to_twoCliques,
                          'd1_index': self.d1_index, 'd1_vals': self.d1_vals,
                          'd1_shape': self.d1_shape}
            torch.save(dictionary, f'{loc}.pt')
        
        # Defining layers that generate the attention matrices
        if not opt['bracket'] == 'node':
            self.att_later_twoCl = SparseTwoCliqueAttentionLayer(opt)
            if not opt['consistent_A2']:
                self.att_layer_NE = SparseNodeEdgeAttentionLayer(opt)
        else:
            n_edges, n_nodes = self.d0_shape
            dim = (n_edges + n_nodes) * opt['hidden_dim']
            channels = [dim for i in range(5)]
            self.net = MLP(channels, False)

        # Learnable E,S for metriplectic bracket
        if opt['bracket'] == 'metriplectic':
            # For convenience
            self.d0d0t_index, self.d0d0t_vals = \
                spspmm(self.d0_index, self.d0_vals, self.d0t_index, 
                       self.d0t_vals, *self.d0_shape, self.d0_shape[0])
            self.d1td1_index, self.d1td1_vals = \
                spspmm(self.d1t_index, self.d1t_vals, self.d1_index, 
                       self.d1_vals, *self.d1_shape[::-1], self.d1_shape[1])

            # Build E,S shallow MLPs
            qChannels = [opt['hidden_dim'], opt['ES_mlp_width'], 1]
            pChannels = [opt['hidden_dim'], opt['ES_mlp_width'], 1]
            self.fE   = MLP(qChannels, batch_norm=False)
            self.fS   = MLP(qChannels, batch_norm=False)
            self.gE   = MLP(pChannels, batch_norm=False)
            self.gS   = MLP(pChannels, batch_norm=False)

    # Computes the RHS of the bracket system
    def forward(self, t, x):
        self.nfe += 1
        q, p = x

        # This seems to help for some reason
        if self.opt['alpha_multiplier']:
            alpha = torch.sigmoid(self.alpha)
        else:
            alpha = 1.

        # Computing the attention matrices
        if not self.opt['bracket'] == 'node':
            if not self.opt['constant_attention']:
                A0, A1, A2 = self.att_later_twoCl(
                    q, self.edge_index, self.d0_index, 
                    self.twoClique_index, self.d1_index)
                if not self.opt['consistent_A2']:
                    A0, A1 = self.att_layer_NE(
                        q, self.edge_index, self.d0_index)
            else:
                A0 = self.A0
                A1 = self.A1
                A2 = self.A2

            # THIS CANNOT BE IN-PLACE (or autograd complains)
            A0 = A0.unsqueeze(-1)
            A1 = A1.unsqueeze(-1)
            A2 = A2.unsqueeze(-1)

        # For debugging (makes everything more sensitive to stepsize)
        # TODO: moving this before the computation would be more efficient
        if self.opt['no_attention']:
            A0 = torch.ones_like(A0)
            A1 = torch.ones_like(A1)
            A2 = torch.ones_like(A2)

        if self.opt['bracket'] == 'hamiltonian':
            # Hamiltonian system
            qPart = -alpha * spmm(self.d0t_index, self.d0t_vals,
                                  *self.d0_shape[::-1], p) / A0
            pPart =  alpha * spmm(self.d0_index, self.d0_vals, 
                                  *self.d0_shape, q / A0)

        elif self.opt['bracket'] in ['gradient', 'gradient_q_only']:
            # Gradient system
            d0ainvq   = spmm(self.d0_index, self.d0_vals,
                             *self.d0_shape, q / A0)
            qPart     = -alpha * spmm(self.d0t_index, self.d0t_vals,
                                      *self.d0_shape[::-1], A1 * d0ainvq) / A0
            if self.opt['bracket'] == 'gradient_q_only':
                pPart = -alpha * p
            else:
                # Compute 1-form Laplacian
                d1ainvp   = spmm(self.d1_index, self.d1_vals,
                                *self.d1_shape, p / A1)
                d1ad1Part = spmm(self.d1t_index, self.d1t_vals, 
                                *self.d1_shape[::-1], A2 * d1ainvp) / A1
                a0invd0tp = spmm(self.d0t_index, self.d0t_vals,
                                *self.d0_shape[::-1], p) / A0
                d0d0aPart = spmm(self.d0_index, self.d0_vals, 
                                *self.d0_shape, a0invd0tp)
                pPart     = -alpha * (d1ad1Part + d0d0aPart)

        elif self.opt['bracket'] == 'double':
            # Double bracket system
            qPart1  = spmm(self.d0t_index, self.d0t_vals,
                           *self.d0_shape[::-1], p) / A0
            pPart1  = spmm(self.d0_index, self.d0_vals,
                           *self.d0_shape, q / A0)
            qPart2  = spmm(self.d0t_index, self.d0t_vals,
                           *self.d0_shape[::-1], A1 * pPart1) / A0
            pPart2  = spmm(self.d0_index, self.d0_vals, 
                           *self.d0_shape, qPart1)
            qPart   = alpha * -qPart1 - qPart2
            pPart   = alpha *  pPart1 - pPart2

        elif self.opt['bracket'] == 'metriplectic':
            # Metriplectic system
            fq      = self.fE(torch.sum(q, axis=-2))
            d0d0tp  = spmm(self.d0d0t_index, self.d0d0t_vals, 
                           self.d0_shape[0], self.d0_shape[0], p)
            gd0d0tp = self.gE(torch.sum(d0d0tp, axis=-2))
            d1td1p  = spmm(self.d1td1_index, self.d1td1_vals,
                           self.d1_shape[1], self.d1_shape[1], p)
            hd1td1p = self.gS(torch.sum(d1td1p, axis=-2))
        
            # Gradients of energy/entropy with respect to features
            dfq      = grad(fq, inputs=q, 
                            grad_outputs=torch.ones_like(fq))[0]
            dgd0d0tp = grad(gd0d0tp, inputs=p, 
                            grad_outputs=torch.ones_like(gd0d0tp))[0]
            dhd1td1p = grad(hd1td1p, inputs=p, 
                            grad_outputs=torch.ones_like(hd1td1p))[0]

            # Equations to solve
            qPart  = -spmm(self.d0t_index, self.d0t_vals, 
                          *self.d0_shape[::-1], dgd0d0tp) / A0
            pPart1 = spmm(self.d0_index, self.d0_vals, 
                          *self.d0_shape, dfq / A0)
            tmp    = spmm(self.d1_index, self.d1_vals, 
                          *self.d1_shape, dhd1td1p)
            pPart2 = spmm(self.d1t_index, self.d1t_vals, 
                          *self.d1_shape[::-1], A2 * tmp)
            pPart  = pPart1 + pPart2

        elif self.opt['bracket'] == 'node':
            qp  = torch.cat((q,p), axis=-2)
            out = self.net(qp.view(qp.shape[0],-1)).view(qp.shape)
            qPart, pPart = torch.split(out, self.d0_shape, dim=-2)

        else:
            raise RuntimeError('not an implemented bracket formalism!')

        # Check the energy/entropy values along the integration.
        if self.opt['verbose']:
            if self.opt['bracket'] != 'metriplectic':
                E = 0.5 * (torch.sum(q**2) + torch.sum(p**2))
                print(f'the energy is {E.item()}')
                self.verbListE.append(E.item())
            else:
                E = fq + gd0d0tp
                S = hd1td1p 
                print(f'energy is {E.item()} and entropy is {S.item()}')
                self.verbListE.append(E.item())
                self.verbListS.append(S.item())
                # print(E)
                # print(S)

        return (qPart, pPart)
    