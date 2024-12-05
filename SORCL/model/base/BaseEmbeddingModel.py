from .BaseModel import BaseModel
import SORCL
from SORCL.model.module import dot_product
from SORCL.model.module.mask_neighbor_score import mask_neighbor_score, mask_neighbor_score_user_item
from SORCL.utils.utils import ensure_dir, print_dict, get_formatted_results
from SORCL.utils.metric import argtopk

from SORCL.data import io, csr
from SORCL.model.module.propagation.LightGCN_Propagation import split_nodes

import torch
import numpy as np
import os.path as osp
from tqdm import tqdm

from time import sleep


class BaseEmbeddingModel(BaseModel):
    
    def __init__(self, config):
        print_dict(config)
        self.config = config
        self.data = {}
        self._init_BaseEmbeddingModel()
    
    def _init_BaseEmbeddingModel(self):
        self.data_root = self.config['data_root']
        assert osp.exists(self.data_root)
        
        self.results_root = self.config['results_root']
        ensure_dir(self.results_root)
        io.save_yaml(osp.join(self.results_root, 'config.yaml'), self.config)
        
        self.model_root = osp.join(self.results_root, 'model')
        ensure_dir(self.model_root)
        
        self.info = io.load_yaml(osp.join(self.data_root, 'info.yaml'))
        self.graph_type = self.info['graph_type']
        if self.graph_type == 'user-item':
            self.num_users = self.info['num_users']
        else:
            self.num_nodes = self.info['num_nodes']
        
        self.indptr = None
        self.indices = None
        self.head = None
        self.tail = None

    # 构造数据加载器和训练器
    def fit(self):
        config = self.config
        data = self.data
        model = self
        # 构造train data loader，dataloader每个batch为(src,pos,neg)
        train_dl = SORCL.create_DataLoader(config, data)

        # 构造训练器，将data和 model丢进去就可以开始训练了
        self.trainer = SORCL.create_Trainer(
            config, data, model, train_dl
        )
        self.trainer.train()
        
        if self.config['use_validation_for_early_stop']:
            self.load()

    # 计算测试集评价指标
    def test(self, test_config=None):
        if test_config is None:
            test_config = self.config
        
        test_evaluator = SORCL.create_test_Evaluator(
            config=test_config, data=self.data, model=self
        )
        results, rec_users = test_evaluator.eval(desc='test')
        
        results['formatted'] = get_formatted_results(results)
        return results, rec_users

    
    def on_val_begin(self):
        self.infer_out_emb_table()
    
    @torch.no_grad()
    def infer_out_emb_table(self):
        # self.out_emb_table = ...
        # return self.out_emb_table
        raise NotImplementedError
    
    def _eval_a_batch(self, batch_data, eval_type):
        return {
            'whole_graph_multi_pos': self._eval_whole_graph_multi_pos,
            'whole_graph_one_pos': self._eval_whole_graph_one_pos,
            'one_pos_k_neg': self._eval_one_pos_k_neg
        }[eval_type](batch_data)
    
    def _backward(self, loss):
        for opt in self.optimizers:
            self.optimizers[opt].zero_grad()
        loss.backward()
        for opt in self.optimizers:
            self.optimizers[opt].step()
    
    def save(self, root=None):
        raise NotImplementedError
    
    def load(self, root=None):
        raise NotImplementedError
    
    def _save_out_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        if not hasattr(self, 'out_emb_table'):
            self.infer_out_emb_table()
        torch.save(self.out_emb_table, osp.join(root, 'out_emb_table.pt'))

    def _save_optimizers(self, root=None):
        if root is None:
            root = self.model_root
        for opt in self.optimizers:
            torch.save(self.optimizers[opt].state_dict(),
                       osp.join(root, opt + '-state_dict.pt'))

    def _save_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        torch.save(self.emb_table.state_dict(), osp.join(root, 'emb_table-state_dict.pt'))
    
    def _load_out_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        self.out_emb_table = torch.load(osp.join(root, 'out_emb_table.pt'))
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users']:]
        else:
            self.target_emb_table = self.out_emb_table
    
    def _load_optimizers(self, root=None):
        if root is None:
            root = self.model_root
        for opt in self.optimizers:
            state_dict = torch.load(osp.join(root, opt + '-state_dict.pt'))
            self.optimizers[opt].load_state_dict(state_dict)
    
    def _load_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        state_dict = torch.load(osp.join(root, 'emb_table-state_dict.pt'))
        self.emb_table.load_state_dict(state_dict)
    
    @torch.no_grad()
    def _eval_whole_graph_multi_pos(self, batch_data):
        src, _ = batch_data
        
        all_target_score = self.infer_all_target_score(src, mask_nei=True)

        return all_target_score
    
    @torch.no_grad()
    def _eval_whole_graph_one_pos(self, batch_data):
        src, pos = batch_data

        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        pos_score = np.empty((len(src),), dtype=np.float32)
        for i in range(len(src)):
            pos_score[i] = all_target_score[i][pos[i]]
        pos_neg_score = np.concatenate((pos_score.reshape(-1, 1), all_target_score), axis=-1)
        
        return pos_neg_score
    
    @torch.no_grad()
    def _eval_one_pos_k_neg(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self.out_emb_table[src]
        pos_emb = self.target_emb_table[pos]
        neg_emb = self.target_emb_table[neg]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        pos_neg_score = torch.cat((pos_score.view(-1, 1), neg_score), dim=-1).cpu().numpy()
        return pos_neg_score


    def infer_all_target_score(self, src, mask_nei=True):
        # 计算src和整个图节点的点积
        src_emb = self.out_emb_table[src]

        all_target_score = (src_emb @ self.target_emb_table.t()).cpu().numpy()

        # 将邻居节点分数设置为很小
        if mask_nei:
            self.mask_neighbor_score(src, all_target_score)

        return all_target_score

    '''
    # 度中心近似度
    def _lookup_degreeSim(self, u, v):
        # u是当前用户，v是还未推荐用户
        DC_u = self.all_degrees[u]
        DC_v = self.all_degrees[v]

        sim = (DC_v - DC_u) / (self.num_nodes - 1)  # sim越大，节点度差异越大
        return sim

    # 用户之间的相似分数
    def _lookup_rel(self, rel_matrix, rec_current, u):
        rel = rel_matrix[rec_current][u]
        return rel

    def _mmr(self, lambda_score, user, rec_unranked, rel_matrix):
        mmr = 0
        rec_user = None
        for d in rec_unranked:
            sim = self._lookup_degreeSim(user, d)  # 度差异
            rel = self._lookup_rel(rel_matrix, user, d)  # 相似分数
            mmr_current = (1 - lambda_score) * rel + lambda_score * sim
            # argmax mmr
            if mmr_current > mmr:
                mmr = mmr_current
                rec_user = d
            else:
                continue

        if rec_user == None:
            print("mmr error")
            print(rec_unranked)

        return mmr, rec_user


    def rank_by_degree(self, src, all_target_score, max_k=50, k=20):
 
        # 结合节点度差异进行重排
        # src:当前batch内节点ID: ndarray(256,)
        # all_target_score:用户之间的点积分数: ndarray(256, 9498)
        # max_k:重排后的最大长度

        # tail_batch = [node for node in src if node in self.tail]  # 当前batch中为tail的节点ID
        # tail_index = [index for index, node in enumerate(src) if node in self.tail]  # 当前batch中为tail的节点的索引
        src_index = [index for index, node in enumerate(src)]
        reranking_matrix = np.zeros((all_target_score.shape[0], all_target_score.shape[1]))  # (256, 9498)


        for u in tqdm(src_index, position=0):
            sleep(0.01)
            u_selected = []
            u_unranked = list(argtopk(all_target_score[u], max_k))  # 前max_k大的元素索引

            while u_unranked:
                # 设置一个衰减的lambda_score
                num = len([u for u in u_selected[:k] if u not in u_unranked[:k]])
                lambda_score = 0.5 - num / (2 * k)  # 这样写准确率降的很离谱

                mmr_score, rec_u = self._mmr(
                    lambda_score,
                    u,
                    u_unranked,
                    all_target_score
                )

                u_selected.append(rec_u)
                reranking_matrix[u][rec_u] = mmr_score
                u_unranked.remove(rec_u)

        return reranking_matrix
    '''

    def mask_neighbor_score(self, src, all_target_score):
        if self.indptr is None:
            # 读取训练集indptr，indices获取邻居信息进行mask
            self._prepare_train_graph_for_mask()
            
        if self.graph_type == 'user-item':
            mask_neighbor_score_user_item(self.indptr, self.indices,
                src, all_target_score, self.num_users
            )
        else:
            # 跳转另一个函数
            mask_neighbor_score(self.indptr, self.indices,
                src, all_target_score
            )
    
    def _prepare_train_graph_for_mask(self):
        if 'indptr' in self.data:
            self.indptr = self.data['indptr']
            self.indices = self.data['indices']
        else:
            self.indptr = io.load_pickle(osp.join(self.data_root, 'indptr.pkl'))
            self.indices = io.load_pickle(osp.join(self.data_root, 'indices.pkl'))

        # 为了重排新加的
        self.all_degrees = csr.get_degrees(self.indptr)  # 所有节点的度数
        # self.head, self.tail = split_nodes(degrees=self.all_degrees, k=self.config['tail_k'])
        # self.head_dic = dict(zip(self.head, self.all_degrees[self.head]))
        # self.tail_dic = dict(zip(self.tail, self.all_degrees[self.tail]))


    def save_emb_as_txt(self, filename='out_emb_table.txt', fmt='%.6f'):
        np.savetxt(fname=filename, X=self.out_emb_table.cpu().numpy(), fmt=fmt)

    def infer_target_score(self, src, target):
        src_emb = self.out_emb_table[src]
        target_emb = self.out_emb_table[target]
        target_score = dot_product(src_emb, target_emb).cpu().numpy()
        return target_score
    
    def infer_topk(self, k, src, mask_nei=True):
        all_target_score = self.infer_all_target_score(src, mask_nei)
        score, node = torch.topk(all_target_score, k, dim=-1)
        return score, node




