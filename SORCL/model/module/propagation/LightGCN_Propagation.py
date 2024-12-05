# from XGCN.model.LightGCN import get_lightgcn_out_emb
from SORCL.data import io, csr

from scipy.sparse.csgraph import dijkstra
import scipy.sparse
import numpy as np
import torch
import os.path as osp
import yaml
from tqdm import tqdm


class LightGCN_Propagation:

    def __init__(self, config, data):
        self.config = config
        self.data = data
        
        data_root = self.config['data_root']
        
        if 'undi_indptr' in self.data:
            indptr = self.data['undi_indptr']
            indices = self.data['undi_indices']
        else:
            # 导入训练集边的相关信息
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            indptr, indices = csr.get_undirected(indptr, indices)
            self.data.update({'undi_indptr': indptr, 'undi_indices': indices})

        E_src = csr.get_src_indices(indptr)
        E_dst = indices
        self.E_src = E_src
        self.E_dst = E_dst


        self.num_nodes = len(indptr) - 1
        self.all_degrees = csr.get_degrees(indptr)  # 所有节点的度数（入度角度出发）
        self.degree_dict = {node_idx: self.all_degrees[node_idx] for node_idx in range(self.num_nodes)}   # 和self.all_degrees等价,，只是为了下面获取节点不同权重计算方便
        if 'loss_weight' in self.data:
            self.loss_weight = self.data['loss_weight']
        else:
            self.loss_weight = get_sample_weight(self.degree_dict, self.config['beta_class'])
            self.data.update({'loss_weight': self.loss_weight})

        d_src = self.all_degrees[E_src]
        d_dst = self.all_degrees[E_dst]


        edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()

        del d_src, d_dst
            
        if hasattr(torch, 'sparse_csr_tensor'):
            print("## use torch.sparse_csr_tensor")
            self.A = torch.sparse_csr_tensor(
                torch.LongTensor(np.array(indptr, dtype=np.int64)),
                torch.LongTensor(E_dst),
                edge_weights,
                (self.num_nodes, self.num_nodes)
            )
        else:
            print("## use torch.sparse_coo_tensor")
            del indptr
            E_src = torch.IntTensor(E_src)
            E_dst = torch.IntTensor(E_dst)
            E = torch.cat([E_src, E_dst]).reshape(2, -1)
            del E_src, E_dst
            self.A = torch.sparse_coo_tensor(
                E, edge_weights, 
                (self.num_nodes, self.num_nodes)
            )


    def __call__(self, X):
        # X是初始emb table
        X_out_h = get_lightgcn_out_emb(
            self.A.to(self.config['emb_table_device']), X, self.config['num_gcn_layers'],
            stack_layers=self.config['stack_layers']
        )
        self.head, self.tail = split_nodes(degrees=self.all_degrees, k=self.config['tail_k'])  # 划分头节点和尾节点
        tail_A = link_dropout(self.E_src, self.E_dst, self.all_degrees, self.num_nodes, self.head)  # 在原来的图上制造假的尾节点，tail_A为新图

        X_out_t = get_lightgcn_out_emb(
            tail_A.to(self.config['emb_table_device']), X, self.config['num_gcn_layers'],
            stack_layers=self.config['stack_layers']
        )

        return X_out_h, X_out_t

    def __getitem__(self, item):
        # 通过索引获取节点度
        deg = self.degree_dict[item]
        return deg


def split_nodes(degrees, k):
    # 划分头节点和尾节点
    head = np.array([node for node, degree in enumerate(degrees) if degree >= k])
    tail = np.array([node for node, degree in enumerate(degrees) if degree < k])

    return head, tail

def link_dropout(src, dst, degrees, num_nodes, head):
    src = np.array(src)
    dst = np.array(dst)
    # Get the number of links to add for each index
    for i in range(head.shape[0]):
        # 查找 head[i] 的节点对应的行中非零元素的索引
        indices = np.where(src == head[i])[0]

        # 保留待删除索引对应的dst值（因为还要随机加回去）
        # dst_dict = dict(zip(indices, dst[indices]))
        # 随机生成新索引（删除所有连接，再从删除的连接中随机加回去几个）
        # num_links = np.random.randint(min(k, len(indices)), size=head.shape[0])+1
        # new_indices = np.random.choice(indices, num_links[i], replace=False)

        # 删！
        src = np.delete(src, indices)
        dst = np.delete(dst, indices)

        # 恢复几个
        # src = np.insert(src, new_indices, [head[i]*len(new_indices)])
        # for j in range(len(new_indices)):
        #     dst = np.insert(dst, new_indices[j], dst_dict[new_indices[j]])
    tail_indptr = csr.get_indptr_indices(src)

    d_src = degrees[src]
    d_dst = degrees[dst]
    edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()

    del d_src, d_dst
    tail_A = torch.sparse_csr_tensor(
        torch.LongTensor(np.array(tail_indptr, dtype=np.int64)),
        torch.LongTensor(dst),
        edge_weights,
        (num_nodes, num_nodes)
    )

    return tail_A


# 针对长尾分布的loss自适应权重模块
def get_sample_weight(degree_dict, beta):
    dic = {}
    degrees = degree_dict.values()
    print("loading weight...")
    for c in tqdm(degrees):
        if c in dic:
            dic[c] += 1
        else:
            dic[c] = 1
    weight_tensor = torch.tensor(list(dic.values()), dtype=torch.float)
    effective_num = 1.0 - torch.pow(beta, weight_tensor)
    weight = (1 - beta) / effective_num
    weight = weight / weight.sum() * len(dic.keys())

    indexed_dict = {key: index for index, key in enumerate(dic)}
    index_list = [indexed_dict[value] for value in degree_dict.values()]
    # 主打一个对一个的索引
    weight = weight[torch.tensor(index_list)]

    return weight


def get_lightgcn_out_emb(A, base_emb_table, num_gcn_layers, stack_layers=True):
    if not stack_layers:
        X = base_emb_table
        for _ in range(num_gcn_layers):
            X = A @ X
        return X
    else:
        X_list = [base_emb_table]
        for _ in range(num_gcn_layers):
            X = A @ X_list[-1]
            X_list.append(X)
        X_out = torch.stack(X_list, dim=1).mean(dim=1)
    return X_out

