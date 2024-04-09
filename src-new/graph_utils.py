"""
Updated on 10/10/2023

@author: adgrube
"""

import torch
import torch_geometric as pyg
from torch_sparse import spmm, spspmm, transpose
import os

# Pathing stuff I don't really understand
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = f'{ROOT_DIR}/data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_unique_edges(edge_index):
    """Removes duplicate half-edges specified as row-col indices.
       Constructs a dictionary associating (i,j), i<j, with global edges.

    Parameters
    ----------
    edge_index: torch.LongTensor, shape [2, nHalfEdges]
        the source and target nodes of the graph half-edges.

    Returns
    -------
    torch.LongTensor, shape [2, nEdges]
        the unique source and target nodes of the edges, ordered i<j.
    dict
        dictionary associating (i,j) pairs i<j with global edge indices.
    """

    iListU = []; jListU = []
    iList, jList = edge_index.tolist()
    for (i,j) in zip(iList, jList):
        if j>=i:
            iListU.append(i); jListU.append(j)
    unique_edge_index = torch.LongTensor([iListU, jListU]).to(device)
    unique_edge_index = pyg.utils.sort_edge_index(unique_edge_index)
    keys = list(zip(unique_edge_index[0].tolist(), 
                    unique_edge_index[1].tolist()))
    vals = range(unique_edge_index.shape[1])

    return unique_edge_index, dict(zip(keys, vals))


def build_sparse_d0(unique_edge_index):
    """Computes a sparse graph gradient.

    Parameters
    ----------
    unique_edge_index: torch.LongTensor, shape [2, nEdges]
        the unique source and target nodes of the edges, ordered i<j.

    Returns
    -------
    d0idx: torch.LongTensor, shape [2, 2*nEdges]
        the row and column indices of the entries in d0 .
    d0vals: torch.FloatTensor, length 2*nEdges
        the values of the entries in d0.
    d0shape: list, length 2
        the [row, col] shape of d0
    """

    nRows   = unique_edge_index.shape[1]
    nCols   = torch.max(unique_edge_index).item()+1
    temp    = torch.LongTensor(range(nRows)).view(-1,1).to(device)
    temp2   = torch.ones(nRows, 2, dtype=torch.int64).to(device)
    d0rows  = (temp * temp2).view(-1)
    d0cols  = unique_edge_index.t().contiguous().view(-1)
    d0vals  = torch.ones(d0cols.shape[0]).to(device)
    d0vals[::2] = -1
    d0idx   = torch.LongTensor([d0rows.tolist(), d0cols.tolist()]).to(device)
    d0shape = [nRows, nCols]
    
    return d0idx, d0vals, d0shape


def get_2_cliques(unique_edge_index):
    """Construct tensor of 2-cliques, ordered i<j<k.

    Parameters
    ----------
    unique_edge_index: torch.LongTensor, shape [2, nEdges]
        the unique source and target nodes of the edges, ordered i<j.

    Returns
    -------
    torch.LongTensor, shape [3, n2cliques]
        tensor containing nodes of the 2-cliques, ordered i<j<k.
    dict
        dictionary associating (i,j,k) triples i<j<k with global 
        2-clique indices
    """

    ijkList = []
    iList, jList = unique_edge_index.tolist()
    for (i,j) in zip(iList, jList):
        jkEdgeMask = unique_edge_index[0] == j     #where source node is j
        jkEdges    = unique_edge_index[:,jkEdgeMask]  #all edges starting at j
        ikEdgeMask = unique_edge_index[0] == i
        ikEdges    = unique_edge_index[:,ikEdgeMask]
        for k in jkEdges[1].tolist():  #all targets with source j
            if (i,k) in zip(ikEdges[0].tolist(), ikEdges[1].tolist()):
                ijkList.append([i,j,k])
    twoClique_index = torch.LongTensor(ijkList).t().to(device)
    if ijkList != []:  #if there are actually 2-cliques
        keys = list(zip(twoClique_index[0].tolist(), 
                        twoClique_index[1].tolist(),
                        twoClique_index[2].tolist()))
        vals = range(len(ijkList))
    else:
        twoClique_index = torch.LongTensor([[],[],[]]).to(device)
        keys = vals = []

    return twoClique_index, dict(zip(keys, vals))


def build_sparse_d1(twoClique_index, nodes_to_edge):
    """Computes a sparse graph curl.

    Parameters
    ----------
    twoClique_index: torch.LongTensor, shape [3, n2cliques]
        tensor containing nodes of the 2-cliques, ordered i<j<k.

    Returns
    -------
    d1idx: torch.LongTensor, shape [2, 3*n2cliques]
        the row and column indices of the entries in d1.
    d1vals: torch.FloatTensor, length 3*nEdges
        the values of the entries in d1.
    d1shape: list, length 2
        the [row, col] shape of d1
    """

    nRows   = twoClique_index.shape[1]
    nCols   = len(nodes_to_edge)
    temp    = torch.LongTensor(range(nRows)).view(-1,1).to(device)
    temp2   = torch.ones(nRows, 3, dtype=torch.int64).to(device)
    d1rows  = (temp * temp2).view(-1)
    col_idx = torch.zeros(nRows, 3, dtype=torch.int64).to(device)
    for (i, clq) in enumerate(twoClique_index.t()):
        col_idx[i][0] = nodes_to_edge[(clq[0].item(), clq[1].item())]
        col_idx[i][1] = nodes_to_edge[(clq[1].item(), clq[2].item())]
        col_idx[i][2] = nodes_to_edge[(clq[0].item(), clq[2].item())]
    d1cols  =  col_idx.view(-1)
    d1vals  = torch.ones(d1cols.shape[0]).to(device)
    d1vals[2::3] = -1
    d1idx   = torch.LongTensor([d1rows.tolist(), d1cols.tolist()]).to(device)
    d1shape = [nRows, nCols]

    return d1idx, d1vals, d1shape




if __name__ == '__main__':
    # Test on toy graph
    edge_index = torch.LongTensor([[0, 0, 1, 1, 1, 2, 2, 3, 3, 3], 
                                  [1, 3, 0, 2, 3, 1, 3, 1, 2, 3]]).to(device)
    unique_byhand = torch.LongTensor([[0, 0, 1, 1, 2, 3], 
                                      [1, 3, 2, 3, 3, 3]]).to(device)
    unique_edge_index, nodes_to_edge = get_unique_edges(edge_index)
    d0idx, d0vals, d0shape = build_sparse_d0(unique_edge_index)
    twoClique_index, nodes_to_twoCliques = get_2_cliques(unique_edge_index)
    d1idx, d1vals, d1shape = build_sparse_d1(twoClique_index, nodes_to_edge)
    if torch.allclose(unique_edge_index, unique_byhand): print('edges match!')
    feat = torch.rand([4, 5]).to(device)
    d0x = spmm(d0idx, d0vals, *d0shape, feat)
    d1d0x = spmm(d1idx, d1vals, *d1shape, d0x)
    if torch.norm(d1d0x) == 0: print('curl of grad is 0!')

    # Test on CORA
    path = os.path.join(DATA_PATH, 'Citeseer')
    dataset = pyg.datasets.Planetoid(f'{os.getcwd()}/data', 'Citeseer')
    data = dataset.data.to(device)
    data.edge_index = pyg.utils.add_self_loops(data.edge_index)[0]
    unique_edge_index, nodes_to_edge = get_unique_edges(data.edge_index)
    d0idx, d0vals, d0shape = build_sparse_d0(unique_edge_index)
    twoClique_index, nodes_to_twoCliques = get_2_cliques(unique_edge_index)
    d1idx, d1vals, d1shape = build_sparse_d1(twoClique_index, nodes_to_edge)
    d0x = spmm(d0idx, d0vals, *d0shape, data.x)
    d1d0x = spmm(d1idx, d1vals, *d1shape, d0x)
    if torch.norm(d1d0x) == 0: print('curl of grad is 0!')
    # extra check for correctness of \Delta_0
    lapidx, lapvals = pyg.utils.get_laplacian(data.edge_index, 
                                              num_nodes=d0shape[1])
    d0tidx, d0tvals = transpose(d0idx, d0vals, *d0shape)
    lApidx, lApvals = spspmm(d0tidx, d0tvals, d0idx, d0vals, 
                            d0shape[1], d0shape[0], d0shape[1])
    vec1 = spmm(lapidx, lapvals, d0shape[1], d0shape[1], data.x)
    vec2 = spmm(lApidx, lApvals, d0shape[1], d0shape[1], data.x)
    if torch.norm(vec1-vec2) == 0: print('laplacian works!')