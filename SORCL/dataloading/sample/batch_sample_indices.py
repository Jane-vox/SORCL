from ..base import BatchSampleIndicesGenerator
from ..utils import EpochReducer, BatchIndexer
from SORCL.data import io,csr

import numpy as np
import torch
import os.path as osp


class SampleIndicesWithReplacement(BatchSampleIndicesGenerator):
    
    def __init__(self, config, data):
        self.config = config
        info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        indptr = io.load_pickle(osp.join(self.config['data_root'], 'indptr.pkl'))
        indices = io.load_pickle(osp.join(self.config['data_root'], 'indices.pkl'))
        
        if 'str_num_total_samples' in config:
            s = config['str_num_total_samples']    # 'num_edges'
            assert s in ['num_edges', 'num_nodes', 'num_users']
            num_total_samples = info[s]  # number of samples indices
        else:
            num_total_samples = info['num_edges']
        
        batch_size = config['train_batch_size']
        ratio = config['epoch_sample_ratio']
        
        self.num_total_samples = num_total_samples  # 边的数量
        self.batch_size = batch_size
        self.num_batch_per_epoch = int(int(self.num_total_samples * ratio) / self.batch_size)

    
    def __len__(self):
        return self.num_batch_per_epoch
    
    def __getitem__(self, batch_idx):
        batch_sample_indices = self.get_batch_sample_indices(batch_idx)
        return batch_sample_indices

    def on_epoch_start(self):
        # do nothing
        pass


    # 生成随机索引
    def get_batch_sample_indices(self, batch_idx):
        batch_sample_indices = torch.randint(0, self.num_total_samples, (self.batch_size,))
        return batch_sample_indices


class SampleIndicesWithoutReplacement(BatchSampleIndicesGenerator):
    
    def __init__(self, config, data):
        info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        
        if 'str_num_total_samples' in config:
            s = config['str_num_total_samples']
            assert s in ['num_edges', 'num_nodes', 'num_users']
            num_total_samples = info[s]  # number of samples indices
        else:
            num_total_samples = info['num_edges']
        
        sample_indices = np.arange(num_total_samples)
        batch_size = config['train_batch_size'], 
        ratio = config['epoch_sample_ratio']
        
        self.epoch_reducer = EpochReducer(sample_indices, ratio)
        self.batch_indexer = None
        self.batch_size = batch_size
        
    def __len__(self):
        if self.batch_indexer is None:
            return None
        else:
            num_batch_per_epoch = len(self.batch_indexer)
            return num_batch_per_epoch
    
    def __getitem__(self, batch_idx):
        batch_sample_indices = self.get_batch_sample_indices(batch_idx)
        return batch_sample_indices
    
    def on_epoch_start(self):
        epoch_sample_indices = self.epoch_reducer.get_sample_indices_for_this_epoch()
        self.batch_indexer = BatchIndexer(epoch_sample_indices, self.batch_size)
    
    def get_batch_sample_indices(self, batch_idx):
        batch_sample_indices = self.batch_indexer[batch_idx]
        return torch.LongTensor(batch_sample_indices)
