from .BaseEmbeddingModel import BaseEmbeddingModel
from ..module import init_emb_table
from ..module import dot_product, bpr_loss, bce_loss
from SORCL.dataloading.create import prepare_gnn_graph
from SORCL.utils.utils import id_map

import torch
import dgl
import os.path as osp


class BaseGNN(BaseEmbeddingModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.graph_device = self.config['graph_device']
        self.emb_table_device = self.config['emb_table_device']
        self.gnn_device = self.config['gnn_device']
        self.out_emb_table_device = self.config['out_emb_table_device']
        
        self.forward_mode = self.config['forward_mode']
        if self.forward_mode == 'full_graph':
            assert len(set([self.graph_device,
                            self.emb_table_device,
                            self.gnn_device])) == 1
        # else:
        #     assert self.forward_mode == 'sample'
        self.g = prepare_gnn_graph(self.config, self.data)
        
        self.optimizers = {}
        
        self.emb_table = init_emb_table(self.config, self.info['num_nodes'])
        
        if not self.config['freeze_emb']:
            if self.config['use_sparse']:
                assert self.forward_mode != 'full_graph'
                self.optimizers['emb_table-SparseAdam'] = torch.optim.SparseAdam(
                    [{'params':list(self.emb_table.parameters()), 
                      'lr': self.config['emb_lr']}]
                )
            else:
                self.optimizers['emb_table-Adam'] = torch.optim.Adam(
                    [{'params': self.emb_table.parameters(),
                      'lr': self.config['emb_lr']}]
                )
        
        self._create_gnn()
        
    def _create_gnn(self):
        # self.gnn = ...
        # self.optimizers['gnn_optimizer'] = torch.optim.Adam(
        #     [{'params': self.gnn.parameters(),
        #       'lr': self.config['gnn_lr']}]
        # )
        raise NotImplementedError

    def _save_gnn(self, root=None):
        if root is None:
            root = self.model_root
        torch.save(self.gnn.state_dict(), osp.join(root, 'gnn-state_dict.pt'))

    def _load_gnn(self, root=None):
        if root is None:
            root = self.model_root
        state_dict = torch.load(osp.join(root, 'gnn-state_dict.pt'))
        self.gnn.load_state_dict(state_dict)

    def save(self, root=None):
        self._save_out_emb_table(root)
        self._save_emb_table(root)
        self._save_gnn(root)
        self._save_optimizers(root)
    
    def load(self, root=None):
        self._load_out_emb_table(root)
        self._load_emb_table(root)
        self._load_gnn(root)
        self._load_optimizers(root)
    
    def forward_and_backward(self, batch_data):
        if self.config['forward_mode'] == 'full_graph':
            ((src, pos, neg), ) = batch_data
            
            input_emb = self.emb_table.weight
            
            output_emb = self.gnn(self.g, input_emb)
            
            src_emb = output_emb[src]
            pos_emb = output_emb[pos]
            neg_emb = output_emb[neg]
        elif self.config['forward_mode'] == 'sub_graph':
            subg, (src, pos, neg) = batch_data
            
            input_emb = self.emb_table(subg.ndata[dgl.NID].to(self.emb_table_device))
            
            output_emb = self.gnn(subg, input_emb.to(self.gnn_device))
            
            src_emb = output_emb[src]
            pos_emb = output_emb[pos]
            neg_emb = output_emb[neg]
        else:
            ((src, pos, neg), ), (input_nid, output_nid, blocks, idx_mapping) = batch_data
            
            input_emb = self.emb_table(input_nid.to(self.emb_table_device))
            
            blocks = [block.to(self.gnn_device) for block in blocks]
            
            output_emb = self.gnn(blocks, input_emb.to(self.gnn_device))
            
            src_emb = output_emb[id_map(src, idx_mapping)]
            pos_emb = output_emb[id_map(pos, idx_mapping)]
            neg_emb = output_emb[id_map(neg, idx_mapping)]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss_type = self.config['loss_type']
        if loss_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif loss_type == 'bce':
            loss = bce_loss(pos_score, neg_score)
        else:
            assert 0
        
        rw = self.config['L2_reg_weight']
        if rw > 0:
            if 'use_ego_emb_L2_reg' in self.config and self.config['use_ego_emb_L2_reg']:
                src_ego_emb = self.emb_table(src.to(self.emb_table_device))
                pos_ego_emb = self.emb_table(pos.to(self.emb_table_device))
                neg_ego_emb = self.emb_table(neg.to(self.emb_table_device))
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_ego_emb**2).sum() + (pos_ego_emb**2).sum() + (neg_ego_emb**2).sum()
                )
            else:
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
                )
            loss += rw * L2_reg_loss
        
        self._backward(loss)
        return loss.item()
    
    @torch.no_grad()
    def infer_out_emb_table(self):
        if self.forward_mode == 'full_graph':
            # full graph infer
            input_emb = self.emb_table.weight
            self.out_emb_table = self.gnn(self.g, input_emb).to(self.out_emb_table_device)
        else:
            # infer on blocks
            self.out_emb_table = self._block_infer_out_emb_table()
            
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table
        
        return self.out_emb_table
    
    @torch.no_grad()
    def _block_infer_out_emb_table(self):
        X = torch.empty(size=self.emb_table.weight.shape, dtype=torch.float32,
                        device=self.out_emb_table_device)
        
        num_layer_sample = eval(self.config['infer_num_layer_sample'])
        
        for i in range(self.config['num_gcn_layers']):
            
            if len(num_layer_sample) == 0:
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            else:
                sampler = dgl.dataloading.NeighborSampler([num_layer_sample[i],])
            
            dl = dgl.dataloading.DataLoader(
                self.g, torch.arange(self.info['num_nodes']).to(self.g.device),
                sampler, batch_size=8192, shuffle=False
            )
            for input_nodes, output_nodes, blocks in dl:
                if i == 0:
                    input_emb = self.emb_table(input_nodes.to(self.emb_table_device))
                else:
                    input_emb = X[input_nodes]
                
                block = blocks[0].to(self.gnn_device)
                
                if hasattr(self.gnn, 'get_layer_output'):
                    output_emb = self.gnn.get_layer_output(block, input_emb, i)
                else:
                    gnn_layer = self.gnn.gnn_layers[i]
                    output_emb = gnn_layer(block, input_emb.to(self.gnn_device))
                
                X[output_nodes] = output_emb.to(self.out_emb_table_device)
        
        out_emb_table = X
        return out_emb_table
