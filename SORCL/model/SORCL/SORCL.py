from SORCL.data import io, csr
from SORCL.model.base import BaseEmbeddingModel
from SORCL.model.module import *
from .module import *

import torch
import torch.nn as nn
import dgl
import dgl.dataloading as dgldl
import dgl.nn as dglnn
import os.path as osp
import numpy as np
import scipy.sparse as ssp



class SORCL(BaseEmbeddingModel):

    def __init__(self, config):
        super().__init__(config)
        self.device = self.config['device']

        self.emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.emb_table.weight.shape, dtype=torch.float32)

        self.disc = Discriminator(self.emb_table.weight.shape[1]).to(self.device)
        self.criterion = torch.nn.BCELoss()

        data_root = self.config['data_root']
        indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))

        if self.graph_type == 'user-item':
            E_src = indices
            E_dst = csr.get_src_indices(indptr)
        else:
           
            undi_indptr, undi_indices = csr.get_undirected(indptr, indices)
            E_src = csr.get_src_indices(undi_indptr)
            E_dst = undi_indices

        edge_weight = torch.ones(E_dst.shape[0], dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight, (E_src, E_dst)),
            shape=(self.info['num_nodes'], self.info['num_nodes'])
        )
        del indices, indptr, edge_weight

        self.g = dgl.graph((E_src, E_dst)).to(self.device) 
        self.all_degrees = self.g.out_degrees()

        if self.config['train_num_layer_sample'] == '[]':
            block_sampler = dgldl.MultiLayerFullNeighborSampler(1)
        else:
            layer_sample = eval(self.config['train_num_layer_sample'])
            assert len(layer_sample) == 1
            block_sampler = dgldl.MultiLayerNeighborSampler(layer_sample)
        self.node_collator = dgldl.NodeCollator(
            self.g, self.g.nodes(), block_sampler
        )

        if self.config['use_uniform_weight']:
            self.fn_msg = dgl.function.copy_u('h', 'm') 
            self.fn_reduce = dgl.function.mean(msg='m', out='h')  
        else:
            print("## use lightgcn edge weights")
            E_src, E_dst = self.g.edges()
            d_src = self.all_degrees[E_src]
            d_dst = self.all_degrees[E_dst]
            edge_weights = 1 / (d_src * d_dst).sqrt()

            self.g.edata['ew'] = edge_weights.to(self.device)  
            self.fn_msg = dgl.function.u_mul_e('h', 'ew', 'm')  
            self.fn_reduce = dgl.function.sum(msg='m', out='h')  


        self.head, self.tail = split_nodes(degrees=self.all_degrees, k=self.config['tail_k'])
        E_src_t, E_dst_t = link_dropout(E_src, E_dst, self.head, self.tail, 5)
        self.g_t = dgl.graph((E_src_t, E_dst_t)).to(self.device)

        self.node_collator_t = dgldl.NodeCollator(
            self.g_t, self.g_t.nodes(), block_sampler
        )

        if self.config['use_uniform_weight']:
            self.fn_msg_t = dgl.function.copy_u('h', 'm')
            self.fn_reduce_t = dgl.function.mean(msg='m', out='h')  
        else:
            print("## use lightgcn edge weights")
            self.all_degrees_t = self.g_t.out_degrees()
            E_src_t, E_dst_t = self.g_t.edges()
            d_src_t = self.all_degrees_t[E_src_t]
            d_dst_t = self.all_degrees_t[E_dst_t]
            edge_weights_t = 1 / (d_src_t * d_dst_t).sqrt()

            self.g_t.edata['ew'] = edge_weights_t.to(self.device)  #

        self.optimizers = {}
        if not self.config['freeze_emb']:
            if self.config['use_sparse']:
                self.optimizers['emb_table-SparseAdam'] = torch.optim.SparseAdam(
                    [{'params': list(self.emb_table.parameters()),
                      'lr': self.config['emb_lr']}]
                )
            else:
                self.optimizers['emb_table-Adam'] = torch.optim.Adam(
                    [{'params': self.emb_table.parameters(),
                      'lr': self.config['emb_lr']}]
                )
        self.optimizers['optimizer_D'] = torch.optim.Adam([
            {'params': self.disc.parameters(), 'lr': self.config['D_lr'], 'weight_decay': self.config['D_lamda']}
        ])
        

    def _get_user_output_emb(self, graph, users=None):
        if users is None:
            with graph.local_scope():
                graph.srcdata['h'] = self.emb_table.weight
                graph.update_all(self.fn_msg, self.fn_reduce)
                if self.graph_type == 'user-item':
                    aggregated_item_emb = graph.dstdata['h'][:self.num_users]
                else:
                    aggregated_item_emb = graph.dstdata['h']

            if self.graph_type == 'user-item':
                user_self_emb = self.emb_table.weight[:self.num_users]
            else:
                user_self_emb = self.emb_table.weight
        else:
            input_items, _, blocks = self.node_collator.collate(users.to(self.device))
            block = blocks[0]

            with block.local_scope():
                block.srcdata['h'] = self.emb_table(input_items.to(self.device))
                block.update_all(self.fn_msg, self.fn_reduce)  # 更新节点特征
                aggregated_item_emb = block.dstdata['h']  # 聚合邻居节点后的节点表示

        user_self_emb = self.emb_table(users.to(self.device))  # 节点本身特征

        theta = self.config['theta']
        user_output_emb = theta * user_self_emb + (1 - theta) * aggregated_item_emb

        return user_output_emb


    def _get_user_output_emb_t(self, graph, users=None):
        if users is None:
            with graph.local_scope():
                graph.srcdata['h'] = self.emb_table.weight
                graph.update_all(self.fn_msg, self.fn_reduce)
                if self.graph_type == 'user-item':
                    aggregated_item_emb = graph.dstdata['h'][:self.num_users]
                else:
                    aggregated_item_emb = graph.dstdata['h']

            if self.graph_type == 'user-item':
                user_self_emb = self.emb_table.weight[:self.num_users]
            else:
                user_self_emb = self.emb_table.weight
        else:
            input_items, _, blocks = self.node_collator_t.collate(users.to(self.device))
            block = blocks[0]

            with block.local_scope():
                block.srcdata['h'] = self.emb_table(input_items.to(self.device))
                block.update_all(self.fn_msg, self.fn_reduce)  
                aggregated_item_emb = block.dstdata['h']  

            user_self_emb = self.emb_table(users.to(self.device))  

        theta = self.config['theta']
        user_output_emb = theta * user_self_emb + (1 - theta) * aggregated_item_emb

        return user_output_emb


    def forward_and_backward(self, epoch, batch_data):

        optimizer = self.optimizers['emb_table-Adam']
        optimizer.zero_grad()

        ((src, pos, neg),) = batch_data

        src_emb = self._get_user_output_emb(graph=self.g, users=src.to(self.device))
        pos_emb = self.emb_table(pos.to(self.device))
        neg_emb = self.emb_table(neg.to(self.device))

        loss = cosine_contrastive_loss(src_emb, pos_emb, neg_emb,
                                       self.config['margin'], self.config['neg_weight'])

        src_emb_t = self._get_user_output_emb_t(graph=self.g_t, users=src.to(self.device))
        pos_emb_t = self.emb_table(pos.to(self.device))
        neg_emb_t = self.emb_table(neg.to(self.device))

        loss_t = cosine_contrastive_loss(src_emb_t, pos_emb_t, neg_emb_t,
                                         self.config['margin'], self.config['neg_weight'])

        task_loss = (loss + loss_t) / 2

        prob_t = self.disc(src_emb_t)
        t_labels = torch.full((len(src), 1), 0.0, device=self.device)
        errorG = self.criterion(prob_t, t_labels)
        L_d = errorG / 2

        hop_pos, hop_neg = self.get_hop_node(batch_data)
        hop_pos_emb = self.emb_table(hop_pos.to(self.device))
        hop_neg_emb = self.emb_table(hop_neg.to(self.device))
        hop_loss = cosine_contrastive_loss(src_emb, hop_pos_emb, hop_neg_emb,
                                       self.config['margin'], 1)

        L_all = task_loss - (self.config['eta'] * L_d) + (self.config['eta2'] * hop_loss)

        rw = self.config['L2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1 / 2 * (1 / len(src)) * ((src_emb ** 2).sum() + (pos_emb ** 2).sum() + (neg_emb ** 2).sum())
            L_all += rw * L2_reg_loss

        L_all.backward()
        optimizer.step()

        return L_all.item(), task_loss.item()


    def train_disc(self, epoch, batch):

        self.disc.train()
        ((src, pos, neg),) = batch
        optimizer_D = self.optimizers['optimizer_D']
        optimizer_D.zero_grad()

        src_emb = self._get_user_output_emb(graph=self.g, users=src.to(self.device))
        src_emb_t = self._get_user_output_emb_t(graph=self.g_t, users=src.to(self.device))
        # print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

        prob_h = self.disc(src_emb)
        h_labels = torch.full((len(src), 1), 1.0, device=self.device, dtype=torch.float32)
        errorD = self.criterion(prob_h, h_labels)

        prob_t = self.disc(src_emb_t)
        t_labels = torch.full((len(src), 1), 0.0, device=self.device, dtype=torch.float32)
        errorG = self.criterion(prob_t, t_labels)

        L_d = (errorD + errorG) / 2

        L_d.backward()
        optimizer_D.step()

        return L_d.item()


    def get_hop_node(self, batch_data):

        ((src, pos, neg),) = batch_data
        max_node_list = []
        min_node_list = []
        
        for src, pos in zip(src, pos):
            nodes, subgraph, dists, _, _ = k_hop_subgraph(src, pos, 1, self.A, 1.0, 20, None, 1, False, None)
            label = de_node_labeling(subgraph, 0, 1) 
            max_node = nodes[label[torch.argmax(label)]]
            min_node = nodes[label[torch.argmin(label)]]
            max_node_list.append(max_node)
            min_node_list.append(min_node)
        '''
        for src, pos in zip(src, pos):
            nodes, subgraph, dists, _, _ = k_hop_subgraph(src, pos, 1, self.A, 1.0, 20, None, 1, False, None)
            label = de_node_labeling(subgraph, 0, 1)  

            filtered_indices = [i for i, node in enumerate(nodes) if node not in (src, pos)]
            filtered_labels = label[filtered_indices]
            filtered_nodes = [nodes[i] for i in filtered_indices]

            max_node = filtered_nodes[torch.argmax(filtered_labels)]
            min_node = filtered_nodes[torch.argmin(filtered_labels)]

            max_node_list.append(max_node)
            min_node_list.append(min_node)
        '''
        return torch.tensor(max_node_list), torch.tensor(min_node_list)


    @torch.no_grad()
    def infer_out_emb_table(self):
        if self.graph_type == 'user-item':
            self.out_emb_table[:self.num_users] = self._get_user_output_emb().cpu()
            self.out_emb_table[self.num_users:] = self.emb_table.weight[self.num_users:].cpu()
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.out_emb_table = self.emb_table.weight
            self.target_emb_table = self.out_emb_table

    def save(self, root=None):
        self._save_optimizers(root)
        self._save_emb_table(root)
        self._save_out_emb_table(root)

    def load(self, root=None):
        self._load_optimizers(root)
        self._load_emb_table(root)
        self._load_out_emb_table(root)
