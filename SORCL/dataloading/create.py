from .dataloading import *
from SORCL.dataloading.sample import SampleIndicesWithReplacement, SampleIndicesWithoutReplacement
from SORCL.dataloading.sample import ObservedEdges_Sampler, NodeBased_ObservedEdges_Sampler
from SORCL.dataloading.sample import RandomNeg_Sampler, StrictNeg_Sampler
from SORCL.data import io, csr

import dgl
import os.path as osp


def prepare_gnn_graph(config, data):
    if 'gnn_graph' in data:
        g = data['gnn_graph']
    else:
        if 'indptr' in data:
            indptr = data['indptr']
            indices = data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            data['indptr'] = indptr
            data['indices'] = indices
        indptr, indices = csr.get_undirected(indptr, indices)
        g = csr.to_dgl(indptr, indices)
        if 'graph_device' in config:
            g = g.to(config['graph_device'])
        data['gnn_graph'] = g
    return g


def create_BlockSampler(config, data):
    num_layer_sample = eval(config['train_num_layer_sample'])
    # e.g. num_layer_sample = [5, 10]: sample 5 nodes for the first layer, 10 for the second layer
    # num_layer_sample = []: do not sample for each layer
    if len(num_layer_sample) == 0:
        block_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            config['num_gcn_layers']
        )
    else:
        assert config['num_gcn_layers'] == len(num_layer_sample)
        block_sampler = dgl.dataloading.NeighborSampler(num_layer_sample)
        
    data['block_sampler'] = block_sampler
    
    return block_sampler


def create_DataLoader(config, data):
    if config['model'] == 'Node2vec':
        train_dl = data['pyg_node2vec_train_dl']
    elif config['model'] == 'GensimNode2vec':
        train_dl = None
    elif 'forward_mode' in config and config['forward_mode'] == 'sub_graph':
        train_dl = create_ClusterDataLoader(config, data)
    else:
        # 这个
        dataset = create_Dataset(config, data)
        train_dl = DataLoader(dataset, config['num_workers'])
    return train_dl

# 先创建dataset，选择一个
def create_Dataset(config, data):
    Dataset_type = config['Dataset_type']
    dataset = {
        'BlockDataset': create_BlockDataset,
        'NodeListDataset': create_NodeListDataset,
    }[Dataset_type](config, data)
    return dataset


def create_NodeListDataset(config, data):
    NodeListDataset_type = config['NodeListDataset_type']
    dataset = {
        'LinkDataset': create_LinkDataset,
    }[NodeListDataset_type](config, data)
    return dataset


# -----------最终构造dataset在这------------
def create_LinkDataset(config, data):
    pos_sampler = {
        'ObservedEdges_Sampler': ObservedEdges_Sampler,
        'NodeBased_ObservedEdges_Sampler': NodeBased_ObservedEdges_Sampler,
    }[config['pos_sampler']](config, data)
    
    neg_sampler = {
        'RandomNeg_Sampler': RandomNeg_Sampler,
        'StrictNeg_Sampler': StrictNeg_Sampler,
    }[config['neg_sampler']](config, data)
    
    batch_sample_indices_generator = {
        'SampleIndicesWithReplacement': SampleIndicesWithReplacement,
        'SampleIndicesWithoutReplacement': SampleIndicesWithoutReplacement,
    }[config['BatchSampleIndicesGenerator_type']](config, data)
    
    dataset = LinkDataset(pos_sampler, neg_sampler, batch_sample_indices_generator)
    return dataset


def create_BlockDataset(config, data):
    node_list_dataset = create_NodeListDataset(config, data)
    block_sampler = create_BlockSampler(config, data)
    g = prepare_gnn_graph(config, data)
    dataset = BlockDataset(g, block_sampler, node_list_dataset)
    return dataset


def create_ClusterDataLoader(config, data):
    if 'indptr' in data:
        indptr = data['indptr']
        indices = data['indices']
    else:
        data_root = config['data_root']
        indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
        data['indptr'] = indptr
        data['indices'] = indices
    indptr, indices = csr.get_undirected(indptr, indices)
    g = csr.to_dgl(indptr, indices)
    
    dl = ClusterDataLoader(g,
        partition_cache_filepath=config['partition_cache_filepath'],
        num_parts=config['num_parts'],
        group_size=config['group_size'],
        batch_size=config['train_batch_size'],
        num_workers=config['num_workers'],
        subgraph_device=config['subgraph_device']
    )
    return dl
