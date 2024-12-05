import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from scipy.sparse.csgraph import shortest_path
import torch.nn as nn
import dgl.nn as dglnn



class Discriminator(torch.nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()

        self.d = torch.nn.Linear(in_features, in_features, bias=True)
        self.wd = torch.nn.Linear(in_features, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    '''
    def weight_init(self, m):
        if isinstance(m, Parameter):
            torch.nn.init.xavier_uniform_(m.weight.data)

        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:                
                m.bias.data.uniform_(-stdv, stdv)
    '''

    def forward(self, ft):
        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)

        return self.sigmoid(prob)


def split_nodes(degrees, k):
    degrees = degrees.cpu().numpy()
    head = np.where(degrees >= k)[0]
    tail = np.where(degrees < k)[0]
    head = torch.tensor(head)
    tail = torch.tensor(tail)

    return head, tail


def link_dropout(src, dst, head, tail, k=5):
    assert len(src) == len(dst), "src and dst must have the same length"
    num_links = np.random.randint(k, size=len(np.unique(src)))
    num_links += 1
    src_t = []
    dst_t = []
    head = np.array(head)
    tail = np.array(tail)
    for i in range(len(head)):
        node = head[i]
        indices = np.where(src == node)[0]  
        links = dst[indices]  
        if len(links) > 0:
            num_to_keep = min(num_links[head[i]], len(links))
            chosen_indices = np.random.choice(len(links), num_to_keep, replace=False)
            for idx in chosen_indices:
                src_t.append(node)
                dst_t.append(links[idx])
                src_t.append(links[idx])
                dst_t.append(node)

    for i in range(len(tail)):
        node = tail[i]
        indices = np.where(src == node)[0] 
        links = dst[indices]   
        for j in range(len(links)):
            src_t.append(node)
            dst_t.append(links[j])
            src_t.append(links[j])
            dst_t.append(node)

    assert len(src_t) == len(dst_t), "link dropout error!"
    del num_links, indices, links, num_to_keep, node, idx

    return np.array(src_t), np.array(dst_t)


def neighbors(fringe, A, outgoing=True):
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)
    return res


def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    nodes = [src.tolist(), dst.tolist()]
    dists = [0, 0]
    visited = set(nodes)
    fringe = set(nodes)
    for dist in range(1, num_hops+1):
        if not directed:
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited  
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(list(fringe), max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe) 
    subgraph = A[nodes, :][:, nodes]

    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y


def de_node_labeling(adj, src, dst, max_dist=3):
    src, dst = (dst, src) if src > dst else (src, dst)

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()
