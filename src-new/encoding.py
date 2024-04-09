"""
Updated on 10/10/2023

@author: adgrube
"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, to_undirected
from utils import MLP

# Message passing MLP node encoder of the form 
# MP(x)_i = psi(x_i, \sum_j phi(x_i,x_j))
# Always uses an undirected graph.  Defaults to adding self-loops.
class MPNodeEncoder(MessagePassing):
    def __init__(self, channels_list_in, channels_list_out, self_loops=True):
        super().__init__(aggr="mean", flow="source_to_target", node_dim=-2)
        self.self_loops = self_loops
        self.mlp_in = MLP(channels_list_in, batch_norm=False)
        self.mlp_out = MLP(channels_list_out, batch_norm=False)

    def forward(self, x, edge_index):
        # Propagate calls message, aggregate, update
        edge_index = to_undirected(edge_index)

        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index)

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i, x_j have shape [E, in_channels]
        tmp = torch.cat([x_i, x_j], dim=-1)
        inside = self.mlp_in(tmp)
        tmp = torch.cat([x_i, inside], dim=-1)

        return self.mlp_out(tmp)