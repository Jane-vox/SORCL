import pickle
import numpy as np
from scipy.sparse.csgraph import dijkstra
from tqdm import tqdm
import argparse
from SORCL.data import io,csr
import os.path as osp
from scipy.sparse import csr_matrix

from SORCL.model.module.propagation import LightGCN_Propagation

def build_adj(opt, verbose = True):
    data_root = "D:/Pycharm Community/PyCharm Community Edition 2023.2.1/Projects/XGCN_library-dev/dataset/instance_pokec"
    indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))

    propagation = LightGCN_Propagation

    if verbose:
        print("create adj matrix")




if __name__ == '__main__':
    # info = io.load_yaml('D:\Pycharm Community\PyCharm Community Edition 2023.2.1\Projects\XGCN_library-dev\dataset\instance_pokec\info.yaml')
    # print(info)
    # num_nodes = info['num_nodes']
    # num_edges = info['num_edges']
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', type=int, default=0,
                        help='Filter out unreliable edges below threshold.')
    opt = parser.parse_args()
    adj = build_adj(opt, verbose=True)




