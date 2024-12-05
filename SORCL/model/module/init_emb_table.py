import torch


def init_emb_table(config, num_nodes=None, return_tensor=False):
    if 'emb_table_device' in config:
        device = config['emb_table_device']
    else:
        device = config['device']

    # 有初始化emb表
    if 'from_pretrained' in config and config['from_pretrained']:
        file_pretrained_emb = config['file_pretrained_emb']
        print('## load pretrained embedding:', file_pretrained_emb)
        emb_table = torch.load(file_pretrained_emb, map_location=device)
    # 没有初始化emb表，正态分布初始化节点嵌入
    else:
        emb_table = torch.empty(size=(num_nodes, config['emb_dim']), dtype=torch.float32, device=device)
        if 'init_method' not in config:
            torch.nn.init.normal_(emb_table, mean=0.0, std=config['emb_init_std'])
        else:
            torch.nn.init.xavier_uniform_(emb_table)

    
    freeze_emb = bool('freeze_emb' in config and config['freeze_emb'])   # 是否冻结嵌入表
    use_sparse = bool('use_sparse' in config and config['use_sparse'])   # 是否使用稀疏向量存储

    if return_tensor:
        # 输入参数为True
        assert not use_sparse
        emb_table.requires_grad = not freeze_emb   # freeze_emb 为 True，不允许其梯度更新
    else:
        emb_table = torch.nn.Embedding.from_pretrained(
            emb_table, freeze=freeze_emb, sparse=use_sparse
        )
    
    return emb_table
