"""
Updated on 10/10/2023

@author: adgrube
"""

import torch.nn as nn
import torch.nn.functional as F

from ODE_block import ODEBlock, ODEFunc
from utils import Meter, MLP, spmm
from encoding import MPNodeEncoder

class GNN(nn.Module):
    def __init__(self, opt, dataset):
        super(GNN, self).__init__()

        # Define the odeblock which handles the integration
        self.odeblock = ODEBlock(ODEFunc, opt, dataset.data)
        self.opt = opt
        self.edge_index = dataset.data.edge_index

        # For counting function evaluations
        self.fm = Meter()
        self.bm = Meter()

        # Parameters for encoding/decoding
        n_node_F = dataset.data.x.shape[-1]  # assumes features in last index
        hidden_dim  = opt['hidden_dim']
        e_width = opt['encoding_width']
        d_width = opt['decoding_width']
        e_in_channels  = [2*n_node_F, e_width, n_node_F]
        e_out_channels = [2*n_node_F, e_width, hidden_dim]
        d_in_channels  = [2*hidden_dim, d_width, hidden_dim]
        try:  # if classification problem
            out_F = dataset.num_classes
        except AttributeError: 
            out_F = dataset.data.y.shape[-1]
        d_out_channels = [2*hidden_dim, d_width, out_F]

        # Define encoders.  Either linear or message-passing.
        if opt['no_encoder_decoder']:
            self.enc_nodes = nn.Identity()
            self.enc_edges = nn.Identity()
        elif opt['linear_encoder']:
            self.enc_nodes = nn.Linear(n_node_F, hidden_dim)
            self.enc_edges = (nn.Identity() if opt['no_edge_encoder'] else
                              nn.Linear(n_node_F, hidden_dim)) # node_F = edge_F
        else:
            self.enc_nodes = MPNodeEncoder(e_in_channels, e_out_channels,
                                           self_loops=True)
            # TODO: add message passing edge encoding
            self.enc_edges = (nn.Identity() if opt['no_edge_encoder'] else 
                              MLP([n_node_F, e_width, hidden_dim], 
                                   batch_norm=False))

        # Define decoders.  Either linear or message-passing
        if opt['no_encoder_decoder']:
            self.dec_nodes = nn.Identity()
            self.dec_edges = nn.Identity()        
        elif opt['linear_decoder']:
            self.dec_nodes = nn.Linear(hidden_dim, out_F)
            self.dec_edges = (nn.Identity() if opt['no_edge_decoder'] else
                              nn.Linear(hidden_dim, out_F)) # node_F = edge_F
        else:
            self.dec_nodes = MPNodeEncoder(d_in_channels, d_out_channels,
                                        self_loops=True)
            # TODO: add message passing edge encoding
            self.dec_edges = (nn.Identity() if opt['no_edge_decoder'] else
                              MLP([hidden_dim, d_width, out_F], 
                                   batch_norm=False))

    # Retrieves the result of the nfe count
    def getNFE(self):
        return self.odeblock.odefunc.nfe

    # Sets nfe counter to 0
    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0

    # Given the state x, encode->solve->decode to get x'
    def forward(self, x):
        q, p = x

        # Do initial dropout
        if self.opt['pre_encoder_dropout'] != 0:
            q = F.dropout(q, self.opt['pre_encoder_dropout'],
                          training=self.training)
            if self.opt['dropout_edges']:
                p = F.dropout(p, self.opt['pre_encoder_dropout'],
                            training=self.training)
            
        # Pass through encoders
        if self.opt['linear_encoder'] or self.opt['no_encoder_decoder']:
            q = self.enc_nodes(q)
            p = self.enc_edges(p)
        else:
            q = self.enc_nodes(q, self.edge_index)
            p = self.enc_edges(p)

        if self.opt['no_edge_encoder']:
            d0_index = self.odeblock.odefunc.d0_index
            d0_vals  = self.odeblock.odefunc.d0_vals
            d0_shape = self.odeblock.odefunc.d0_shape
            p = spmm(d0_index, d0_vals, *d0_shape, q)

        x = (q, p)

        # Set IC for latent ODE
        self.odeblock.set_x0(x)

        # Integrate the latent ODE
        (qq, pp) = self.odeblock(x)

        # # BS term testing
        # if self.opt['activate_latent']:
        #     qq = F.relu(qq)
        #     pp = F.relu(pp)

        # Dropout
        if self.opt['pre_decoder_dropout'] != 0:
            qq = F.dropout(qq, self.opt['pre_decoder_dropout'], 
                           training=self.training)
            if self.opt['dropout_edges']:
                pp = F.dropout(pp, self.opt['pre_decoder_dropout'], 
                            training=self.training)

        # Decode to get node/edge predictions
        if self.opt['linear_decoder'] or self.opt['no_encoder_decoder']:
            qq = self.dec_nodes(qq)
            pp = self.dec_edges(pp)
        else:
            qq = self.dec_nodes(qq, self.edge_index)
            pp = self.dec_edges(pp)
        
        return (qq, pp)