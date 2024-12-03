from ..base import BaseSampler
from SORCL.data import io, csr
from SORCL.model.module.propagation.LightGCN_Propagation import *
from SORCL.dataloading.sample.find_path import *
from SORCL.model import SORCL

import torch
import os.path as osp
import numpy as np
from scipy.sparse import csr_matrix
import torch.nn.functional as F

# 从边索引角度采样正例pos，1src-1pos（一条边两端的节点）
class ObservedEdges_Sampler(BaseSampler):

    def __init__(self, config, data):
        if 'indptr' in data:
            indptr = data['indptr']
            indices = data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            data.update({'indptr': indptr, 'indices': indices})
        src_indices = csr.get_src_indices(indptr)
        E_src = torch.LongTensor(src_indices)
        E_dst = torch.LongTensor(indices)
        self.E_src = E_src
        self.E_dst = E_dst

    def __call__(self, eid):
        src = self.E_src[eid]
        pos = self.E_dst[eid]
        return src, pos



# 从节点角度采样正例pos，1src-1pos
class NodeBased_ObservedEdges_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        # ensure sample indices（保证是采样节点）:
        assert 'str_num_total_samples' in config and config['str_num_total_samples'] in ['num_nodes', 'num_users']
        
        if 'indptr' in data:
            indptr = data['indptr']
            indices = data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            data.update({'indptr': indptr, 'indices': indices})
        self.indptr = indptr
        self.indices = indices
        # indptr2trans, indices2trans = csr.get_undirected(self.indptr, self.indices)
        # self.info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        # self.all_degrees = csr.get_degrees(indptr2trans)  # 所有节点的度数
    
    def __call__(self, nid):
        src = nid
        pos = []
        for u in src.numpy():
            nei = csr.get_neighbors(self.indptr, self.indices, u)   # 获取当前nid的所有邻居nei
            if len(nei) == 0:
                pos.append(u)
                continue
            else:
                idx = torch.randint(0, len(nei), (1,)).item()  # 从邻居nei里随机选一个作为pos
                pos.append(nei[idx])

        pos = torch.LongTensor(np.array(pos))

        return src, pos
