"""
Updated on 10/10/2023

@author: adgrube
"""

import os
import torch
from torchdiffeq import odeint
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import (Planetoid, Amazon, Coauthor, WebKB,
                                      WikipediaNetwork, Actor)
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from utils import Lambda, DoublePendulum

import numpy as np

# Pathing stuff I don't really understand
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = f'{ROOT_DIR}/data'


# Builds the 1-trajectory double pendulum dataset with optional history
def get_DP_dataset(device='cpu', T=50, nSteps=501, history=1,
                   params=torch.tensor([1, 0.9, 1, 1, 1, 0.1, 0.1])):
    
    params = params.to(device)
    
    # Edge index and d0 by hand
    unique_edge_index = torch.LongTensor([[0,0,1],[1,2,2]]).to(device)
    # d0 = torch.tensor([[-1.,1.,0.],[-1.,0.,1.],[0.,-1.,1.]]).to(device)

    # Parameters for simulation
    s0 = torch.tensor([1.0, torch.pi/2., 0., 0.]).to(device)
    t  = torch.linspace(0., T, nSteps+history).to(device)

    # Compute solution to DP system
    with torch.no_grad():
        sol = odeint(Lambda(params).to(device), s0, t, method='dopri5')
    
    # Post-processing to extract positions at nodes
    l1, l2  = params[:2]
    th1     = sol[:,0]
    th2     = sol[:,1]
    x1      =  l1 * torch.sin(th1)
    y1      = -l1 * torch.cos(th1)
    x2      =  x1 + l2 * torch.sin(th2)
    y2      =  y1 - l2 * torch.cos(th2)
    x0      =  torch.zeros_like(x1)
    y0      =  torch.zeros_like(y1)

    # Defining node features as positions 
    # qData is [nSteps, nNodes, nCoords]
    X       = torch.cat((x0.unsqueeze(1), x1.unsqueeze(1), x2.unsqueeze(1)), 1)
    Y       = torch.cat((y0.unsqueeze(1), y1.unsqueeze(1), y2.unsqueeze(1)), 1)
    XY      = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), -1)

    if history == 0:
        # Set up to run whole trajectory from initial condition
        data = Data(x=XY[0], edge_index=unique_edge_index, y=XY).to(device)
    
    else:
        # Set up to predict *history* timesteps ahead
        q_data = torch.zeros([nSteps, 3, 2*history])
        for i in range(nSteps):
            if history != 1:
                q_data[i] = XY[i:i+history].permute(1,0,2
                                                    ).reshape(XY.shape[1], -1)
            else:
                q_data[i] = XY[i]

        # Defining shifts forward by *history* timesteps as targets
        y_data  = torch.cat((X[history:].unsqueeze(-1), 
                            Y[history:].unsqueeze(-1)), -1)
        
        # Creating Data object
        data = Data(x=q_data, edge_index=unique_edge_index,
                    y=y_data).to(device)

    # Need to make fake Dataset object...
    class DataWrapper():
        def __init__(self, data):
            self.data = data

    return DataWrapper(data)


### The remainder is lightly modified from the GRAND codebase

def get_dataset(opt: dict, data_dir):
    ds = opt['dataset']
    path = os.path.join(data_dir, ds)
    if ds in ['Cora', 'Citeseer', 'Pubmed']:
        # transform = T.Compose([LargestConnectedComponent(), T.GDC()])
        # transform = T.GDC()
        dataset = Planetoid(path, ds)
    elif ds in ['Computers', 'Photo']:
        dataset = Amazon(path, ds)
    elif ds == 'CoauthorCS':
        dataset = Coauthor(path, 'CS')
    elif ds in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=path, name=ds, transform=T.NormalizeFeatures())
    elif ds in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=path, name=ds, 
                                   transform=T.NormalizeFeatures())
    elif ds == 'film':
        dataset = Actor(root=path, transform=T.NormalizeFeatures())
    else:
        raise Exception('Unknown dataset.')
    
    # https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html?highlight=data    
    if opt['use_lcc']:
        lcc = get_largest_connected_component(dataset)

        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]

        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))

        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data
    
    train_mask_exists = True
    try:
        dataset.data.train_mask
    except AttributeError:
        train_mask_exists = False

    if (opt['use_lcc'] or not train_mask_exists):
        dataset.data = set_train_val_test_split(123, dataset.data,
                         num_development=5000 if ds == "CoauthorCS" else 1500)
        
    return dataset


def set_train_val_test_split(seed: int, data: Data, 
                             num_development: int = 1500,
                             num_per_class: int = 20) -> Data:
    import numpy as np
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]
    development_idx = rnd_state.choice(num_nodes, num_development,
                                       replace=False)
    test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

    train_idx = []
    for c in range(data.y.max() + 1):
        class_idx = development_idx[np.where(
                                    data.y[development_idx].cpu() == c)[0]]
        train_idx.extend(rnd_state.choice(class_idx, 
                                          num_per_class, replace=False))

    val_idx = [i for i in development_idx if i not in train_idx]

    def get_mask(idx):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = 1
        return mask

    data.train_mask = get_mask(train_idx)
    data.val_mask = get_mask(val_idx)
    data.test_mask = get_mask(test_idx)

    return data


def get_component(dataset: InMemoryDataset, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset.data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes
                     and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def get_largest_connected_component(dataset: InMemoryDataset) -> np.ndarray:
    remaining_nodes = set(range(dataset.data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


# from torch_geometric.data.datapipes import functional_transform
# from torch_geometric.transforms import BaseTransform
# from torch_geometric.utils import to_scipy_sparse_matrix

# # Lightly modified from pyg source code
# @functional_transform('largest_connected_component')
# class LargestConnectedComponent(BaseTransform):
#     r"""Selects the subgraph that corresponds to the
#     largest connected components in the graph
#     (functional name: :obj:`largest_connected_components`).

#     Args:
#         num_components (int, optional): Number of largest components to keep
#             (default: :obj:`1`)
#         connection (str, optional): Type of connection to use for directed
#             graphs, can be either :obj:`'strong'` or :obj:`'weak'`.
#             Nodes `i` and `j` are strongly connected if a path
#             exists both from `i` to `j` and from `j` to `i`. A directed graph
#             is weakly connected if replacing all of its directed edges with
#             undirected edges produces a connected (undirected) graph.
#             (default: :obj:`'weak'`)
#     """
#     def __init__(self, num_components: int = 1, connection: str = 'weak'):
#         assert connection in ['strong', 'weak'], 'Unknown connection type'
#         self.num_components = num_components
#         self.connection = connection

#     def forward(self, data: Data) -> Data:
#         import numpy as np
#         import scipy.sparse as sp

#         # Added this
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

#         num_components, component = sp.csgraph.connected_components(
#             adj, connection=self.connection)

#         if num_components <= self.num_components:
#             return data

#         _, count = np.unique(component, return_counts=True)
#         subset = np.in1d(component, count.argsort()[-self.num_components:])

#         # Added to_device call
#         return data.subgraph(torch.from_numpy(subset).bool().to(device))

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}({self.num_components})'