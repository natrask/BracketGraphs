"""
Updated on 10/10/2023

@author: adgrube
"""

import os
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter_add
import torch.nn as nn
# from torch_scatter import scatter, segment_csr, gather_csr
# from torch_geometric.utils.num_nodes import maybe_num_nodes


# Builds a simple MLP network from a list of widths (tanh activation).
def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
               nn.Sequential(nn.Linear(channels[i-1], channels[i]), 
               nn.Tanh() if i != len(channels)-1 else nn.Identity(),
               nn.BatchNorm1d(channels[i]) if batch_norm==True \
                   else nn.Identity())
        for i in range(1, len(channels))
    ])


# Centered squareplus activation
def squareplus(x):
    out = x - x.max()
    # out = out.exp()
    return (out + torch.sqrt(out ** 2 + 4)) / 2


# Prints the number of parameters in the model
def print_model_params(model, verbose=False):
    total_num_params = 0
    if verbose:
        print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'name is {name} and its shape is {list(param.data.shape)}')
            total_num_params += param.numel()
    print(f"Model has a total of {total_num_params} params")


# Runs vanilla NODE (rhs) with FC network
class NodeNet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = MLP(channels, False)

    def forward(self, t, x):
        q, p = x
        cat  = torch.cat((q,p), axis=-2)
        return self.net(cat.flatten()).view(cat.shape)


# https://github.com/twitter-research/graph-neural-pde
def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, 
                               weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, 
                                   weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, 
                                   weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, 
                                weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, 
                                  weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))
    

# Function for computing RHS of DP ODEs
def DoublePendulum(t, x, params):
    th1, th2, dth1, dth2 = x
    l1, l2, m1, m2, g, k1, k2 = params
    
    delth  = th1 - th2
    alpha  = k1 * dth1
    beta   = k2 * dth2
    gamma1 = 2. * alpha - 2.*beta * torch.cos(delth)
    gamma2 = 2. * alpha * torch.cos(delth) - 2. * (m1+m2) * beta / m2
   
    num1 = (m2 * l1 * dth1**2 * torch.sin(2. * delth) 
            + 2. * m2 * l2 * dth2**2 * torch.sin(delth) 
            + 2. * g * m2 * torch.cos(th2) * torch.sin(delth) 
            + 2. * g * m1 * torch.sin(th1) + gamma1)
    den1 = -2. * l1 * ( m1 + m2 * torch.sin(delth)**2 )
   
    num2 = (m2 * l2 * dth2**2 * torch.sin(2. * delth) 
            + 2. * (m1 + m2) * l1 * dth1**2 * torch.sin(delth) 
            + 2. * (m1 + m2) * g * torch.cos(th1) * torch.sin(delth) + gamma2)
    den2 = 2. * l2 * ( m1 + m2 * torch.sin(delth)**2 )
   
    return torch.stack([dth1, dth2, num1/den1, num2/den2], dim=0)


# Wrapper for torchdiffeq to play nice with DoublePendulum
class Lambda(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
    def forward(self, t, x):
        return DoublePendulum(t, x, self.params)
    

# Counter of forward and backward passes.
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.cnt = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.cnt += 1

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.sum / self.cnt

    def get_value(self):
        return self.val
  

# This is just the usual spmm with an assertion check commented out
# So, it applies to more general dense tensors.
def spmm(index: Tensor, value: Tensor, m: int, n: int,
         matrix: Tensor) -> Tensor:
    """Matrix product of sparse matrix with dense tensor.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix, either of
            floating-point or integer type. Does not work for boolean and
            complex number data types.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix of same type as
            :obj:`value`.

    :rtype: :class:`Tensor`
    """

    # assert n == matrix.size(-2)

    row, col = index[0], index[1]
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    out = scatter_add(out, row, dim=-2, dim_size=m)

    return out


# # https://twitter.com/jon_barron/status/1387167648669048833?s=12
# # @torch.jit.script
# def squareplus(src: Tensor, index: Optional[Tensor], 
#                ptr: Optional[Tensor] = None, 
#                num_nodes: Optional[int] = None) -> Tensor:
#     """Computes a sparsely evaluated softmax.
#     Given a value tensor :attr:`src`, this function first groups the values
#     along the first dimension based on the indices specified in :attr:`index`,
#     and then proceeds to compute the softmax individually for each group.

#     Args:
#         src (Tensor): The source tensor.
#         index (LongTensor): The indices of elements for applying the softmax.
#         ptr (LongTensor, optional): If given, computes the softmax based on
#             sorted inputs in CSR representation. (default: :obj:`None`)
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

#     :rtype: :class:`Tensor`
#     """
#     out = src - src.max()
#     # out = out.exp()
#     out = (out + torch.sqrt(out ** 2 + 4)) / 2

#     if ptr is not None:
#         out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
#     elif index is not None:
#         N = maybe_num_nodes(index, num_nodes)
#         out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
#     else:
#         raise NotImplementedError

#     return out / (out_sum + 1e-16)