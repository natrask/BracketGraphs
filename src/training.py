"""
Updated on 10/10/2023

@author: adgrube
"""

import torch
import torch.nn as nn
from torch_sparse import spmm


def train(model, optimizer, data, classification=True):
    model.train()
    optimizer.zero_grad()

    # Set node and edge features
    node_F = data.x
    edge_F = data.edge_x
    node_F.requires_grad = True
    edge_F.requires_grad = True
    
    # Compute nodal predictions
    out = model((node_F, edge_F))[0]

    # Evaluate loss (right now on nodes only)
    if classification:
        loss = nn.CrossEntropyLoss()(out.squeeze()[data.train_mask], 
                                    data.y.squeeze()[data.train_mask])
    else:
        # loss = nn.MSELoss()(out, data.y)
        loss = torch.mean(torch.abs(out - data.y))
    
    # Evaluate forward NFE
    model.fm.update(model.getNFE())
    model.resetNFE()

    # Update params
    loss.backward()
    optimizer.step()

    # Evaluate backward NFE
    model.bm.update(model.getNFE())
    model.resetNFE()

    return loss.item()


# @torch.no_grad() -- need grads for metriplectic.
def test_node_classifier(model, data, opt=None):  
    # opt required for runtime polymorphism
    model.eval()
    
    # Set node and edge features
    node_F = data.x
    edge_F = data.edge_x

    logits, accs = model((node_F, edge_F))[0], []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
# TODO: this is not implemented yet!!
def test_node_regressor(model, data):
    model.eval()
    
    # Set node and edge features
    node_F = data.x
    edge_F = data.edge_x

    logits, accs = model((node_F, edge_F))[0], []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs