"""
Updated on 10/10/2023

@author: adgrube
"""

import torch
import torch.nn as nn
from torch_scatter import scatter
from utils import squareplus
from graph_utils import *   #for unit test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SparseNodeEdgeAttentionLayer(nn.Module):
    """
    A module which computes the sparse attention matrices A1, A0,
    in the usual way with (symmetrized) scaled-dot product attention.

    Attributes
    ----------
    in_dim: int
        number of nodal features
    hidden_dim: int
        number of transformed nodal features
    attention_dim: int
        dimension of attentional embedding (defaults to out_features)
    heads: int
        number of attention heads (defaults to 1)
    opt: dict
        dictionary with options

    Methods
    -------
    init_weights
        Initializes embedding weights to constant
    forward
        Computes the sparse attention matrices
    """

    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt['hidden_dim']
        self.opt = opt

        # Set attention embedding dimension and number of heads
        try:
            self.h = int(opt['heads'])
            self.attention_dim = opt['attention_ratio'] * self.h
        except KeyError:
            self.attention_dim = opt['hidden_dim']
            self.h = 1

        assert self.attention_dim % self.h == 0, 'Number of heads ({}) must' \
            'be a factor of the dimension size ({})'.format(
            self.h, self.attention_dim)
        
        # Initialize embedding layers and remainder after division into heads 
        self.d_k = self.attention_dim // self.h

        self.Q = nn.Linear(self.hidden_dim, self.attention_dim)
        self.init_weights(self.Q)

        self.K = nn.Linear(self.hidden_dim, self.attention_dim)
        self.init_weights(self.K)

        if self.opt['attention_type'] == "exp_kernel":
            self.output_var = nn.Parameter(torch.ones(1))
            self.lengthscale = nn.Parameter(torch.ones(1))


    def init_weights(self, m):
        """
        Initializes weights of a network m
        """
        if type(m) == nn.Linear:
            # nn.init.xavier_uniform_(m.weight, gain=1.414)
            m.bias.data.fill_(0.01)
            nn.init.constant_(m.weight, 1e-5)


    def forward(self, x, edge_index, d0_index):
        """
        x is [nNodes, nFeatures]
        """
        # Embed to get [nNodes, att_dim]
        q = self.Q(x)
        k = self.K(x)

        # split into h heads to get [nodes, heads, att_dim']
        q = q.view(-1, self.h, self.d_k)
        k = k.view(-1, self.h, self.d_k)

        # Compute products [nEdges, heads, att_dim']
        q_i = q[edge_index[0, :], :, :]
        k_j = k[edge_index[1, :], :, :]
        if not self.opt['no_symmetrize']:
            q_j = q[edge_index[1, :], :, :]
            k_i = k[edge_index[0, :], :, :]

        # Compute pre-attention coeffs [nEdges, heads]
        if self.opt['attention_type'] == 'scaled_dot':
            if not self.opt['no_symmetrize']:
                pre = 0.5*torch.sum(q_i * k_j + q_j * k_i, dim=-1)
            else:
                pre = torch.sum(q_i * k_j, dim=-1)

        elif self.opt['attention_type'] == 'exp_kernel':
            if not self.opt['no_symmetrize']:
                arg = torch.sum((q_i - k_j + q_j - k_i)**2, dim=1
                                ) / 2*self.lengthscale**2
            else:
                arg = torch.sum((q_i - k_j)**2, dim=1) / 2*self.lengthscale**2
            pre = self.output_var**2 * torch.exp(-arg)

        elif self.opt['attention_type'] == 'cosine_sim':
            cos = nn.CosineSimilarity(dim=1, eps=1e-5)
            if not self.opt['no_symmetrize']:
                pre = cos(q_i, k_j) + cos(q_j, k_i)
            else:
                pre = cos(q_i, k_j)

        elif self.opt['attention_type'] == 'pearson':
            cos = nn.CosineSimilarity(dim=1, eps=1e-5)
            q_i_mu = torch.mean(q_i, dim=1, keepdim=True)
            k_j_mu = torch.mean(k_j, dim=1, keepdim=True)
            q_i = q_i - q_i_mu
            k_j = k_j - k_j_mu
            if not self.opt['no_symmetrize']:
                q_j_mu = torch.mean(q_j, dim=1, keepdim=True)
                k_i_mu = torch.mean(k_i, dim=1, keepdim=True)
                q_j = q_j - q_j_mu
                k_i = k_i - k_i_mu                
                pre = cos(q_i, k_j) + cos(q_j, k_i)
            else:
                pre = cos(q_i, k_j)

        # Need to average over heads in this step
        pre = torch.mean(pre, dim=-1)

        # Top degree is always f(pre)
        if self.opt['use_squareplus']:
            diagA1 = squareplus(pre)
        else:
            diagA1 = torch.exp(pre)

        '''For A0, each edge score belongs to two nodes with global indices
        contained in the column indices of d0.  So, we first duplicate
        the edge scores.'''
        all_ones = torch.ones(diagA1.shape[0], 2).to(device)
        edge_score_per_node = (diagA1.view(-1,1) * all_ones).view(-1)

        # Then, we scatter them into A0.
        diagA0 = scatter(edge_score_per_node, d0_index[1], dim=0)

        return diagA0, diagA1
    

class SparseTwoCliqueAttentionLayer(nn.Module):
    """
    A module which computes the sparse attention matrices A2, A1, A0,
    starting with 2-cliques and bootstrapping down.  The 2-clique attention
    is created by symmetrizing the contraction of the tensor product 
    (Qx_i) \otimes (Kx_j) \otimes (Vx_k) against the identity tensor.

    Attributes
    ----------
    in_dim: int
        number of nodal features
    hidden_dim: int
        number of transformed nodal features
    attention_dim: int
        dimension of attentional embedding (defaults to out_features)
    heads: int
        number of attention heads (defaults to 1)
    opt: dict
        dictionary with options

    Methods
    -------
    init_weights
        Initializes embedding weights to constant
    forward
        Computes the sparse attention matrices
    """

    def __init__(self, opt):
        super().__init__()
        self.hidden_dim = opt['hidden_dim']
        self.opt = opt

        # Set attention embedding dimension and number of heads
        try:
            self.h = int(opt['heads'])
            self.attention_dim = opt['attention_ratio'] * self.h
        except KeyError:
            self.attention_dim = opt['hidden_dim']
            self.h = 1

        assert self.attention_dim % self.h == 0, 'Number of heads ({}) must'\
            'be a factor of the dimension size ({})'.format(
            self.h, self.attention_dim)
        
        # Initialize embedding layers and remainder after division into heads 
        self.d_k = self.attention_dim // self.h

        self.Q = nn.Linear(self.hidden_dim, self.attention_dim)
        self.init_weights(self.Q)

        self.K = nn.Linear(self.hidden_dim, self.attention_dim)
        self.init_weights(self.K)
    
        self.V = nn.Linear(self.hidden_dim, self.attention_dim)
        self.init_weights(self.V)


    def init_weights(self, m):
        """
        Initializes weights of a network m
        """
        if type(m) == nn.Linear:
            # nn.init.xavier_uniform_(m.weight, gain=1.414)
            m.bias.data.fill_(0.01)
            nn.init.constant_(m.weight, 1e-5)


    def forward(self, x, edge_index, d0_index, twoClique_index, d1_index):
        """
        x is [nNodes, nFeatures]
        """
        # Embed to get [nNodes, att_dim]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # split into h heads to get [nNodes, heads, att_dim']
        q = q.view(-1, self.h, self.d_k)
        k = k.view(-1, self.h, self.d_k)
        v = v.view(-1, self.h, self.d_k)

        # Compute products [n2cliques, heads, att_dim']
        q_i = q[twoClique_index[0, :], :, :]
        k_j = k[twoClique_index[1, :], :, :]
        v_k = v[twoClique_index[2, :], :, :]

        # Compute pre-attention coeffs [n2cliques, heads]
        if not self.opt['no_symmetrize']:
            q_j = q[twoClique_index[1, :], :, :]
            q_k = q[twoClique_index[2, :], :, :]
            k_i = k[twoClique_index[0, :], :, :]
            k_k = k[twoClique_index[2, :], :, :]
            v_i = v[twoClique_index[0, :], :, :]
            v_j = v[twoClique_index[1, :], :, :]
            pre = 1/6*torch.sum(q_i*k_j*v_k + q_j*k_k*v_i + q_k*k_i*v_j
                              + q_j*k_i*v_k + q_k*k_j*v_i + q_i*k_k*v_j, 
                                dim=-1)
        else:
            pre = torch.sum(q_i * k_j * v_k, dim=-1)

        # Need to average over heads in this step
        pre = torch.mean(pre, dim=-1)

        # Compute attention scores.  Top degree is always f(pre).
        if self.opt['use_squareplus']:
            diagA2 = squareplus(pre)
        else:
            diagA2 = torch.exp(pre)

        '''Now we need to associate each global edge with a sum on adjacent
        2-cliques.  To do this, we first triplicate the 2-clique scores,
        since each 2-clique has three edges.'''
        all_ones = torch.ones(diagA2.shape[0], 3).to(device)
        clique_score_per_edge = (diagA2.view(-1,1) * all_ones).view(-1)

        '''Since d1cols contains the three global edge indices associated to 
        each 2-clique, we just scatter the values into the global edges.
        dim_size is passed because not all edges necessarily have 
        a clique score.'''
        diagA1 = scatter(clique_score_per_edge, d1_index[1], dim=0, 
                         dim_size=edge_index.shape[1])

        # Computing A0 is now identical to before.
        all_ones = torch.ones(diagA1.shape[0], 2).to(device)
        edge_score_per_node = (diagA1.view(-1,1) * all_ones).view(-1)
        diagA0 = scatter(edge_score_per_node, d0_index[1], dim=0)

        return diagA0, diagA1, diagA2


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test on toy graph
    options = {'no_symmetrize': False, 'use_squareplus': False,
               'attention_ratio': 6, 'heads': 3, 'hidden_dim': 6}

    edge_index = torch.LongTensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 0, 4], 
                                  [1, 3, 0, 2, 3, 1, 3, 1, 2, 0, 4, 0]]
                                  ).to(device)
    unique_edge_index, nodes_to_edge = get_unique_edges(edge_index)
    d0idx, d0vals, d0shape = build_sparse_d0(unique_edge_index)
    twoClique_index, nodes_to_twoCliques = get_2_cliques(unique_edge_index)
    d1idx, d1vals, d1shape = build_sparse_d1(twoClique_index, nodes_to_edge)
    feat = torch.rand([5, 6]).to(device)
    layer = SparseNodeEdgeAttentionLayer(options).to(device)
    A0, A1 = layer(feat, unique_edge_index, d0idx)
    layer = SparseTwoCliqueAttentionLayer(options).to(device)
    A0p, A1p, A2p = layer(feat, unique_edge_index,
                          d0idx, twoClique_index, d1idx)
    print('toy graph passed!')

    # Test on CORA
    options = {'no_symmetrize': False, 'use_squareplus': False,
               'attention_ratio': 8, 'heads': 3, 'hidden_dim': 3703}

    # path = os.path.join(DATA_PATH, ds)
    dataset = pyg.datasets.Planetoid(f'{os.getcwd()}/data', 'Citeseer')
    data = dataset.data.to(device)
    unique_edge_index, nodes_to_edge = get_unique_edges(data.edge_index)
    d0idx, d0vals, d0shape = build_sparse_d0(unique_edge_index)
    twoClique_index, nodes_to_twoCliques = get_2_cliques(unique_edge_index)
    d1idx, d1vals, d1shape = build_sparse_d1(twoClique_index, nodes_to_edge)
    layer = SparseNodeEdgeAttentionLayer(options).to(device)
    A0, A1 = layer(data.x, unique_edge_index, d0idx)
    layer = SparseTwoCliqueAttentionLayer(options).to(device)
    A0p, A1p, A2p = layer(data.x, unique_edge_index, 
                          d0idx, twoClique_index, d1idx)
    print('Cora passed!')