from SORCL.data.csr import get_neighbors

import numpy as np
import numba


@numba.jit(nopython=True)
def mask_neighbor_score(indptr, indices, src, all_target_score):
    # 将训练集中存在的好友关系对应设置为无穷大
    for i, u in enumerate(src):
        nei_target_id = get_neighbors(indptr, indices, u)
        all_target_score[i][nei_target_id] = -np.inf


@numba.jit(nopython=True)
def mask_neighbor_score_user_item(indptr, indices, src, all_target_score, num_users):
    for i, u in enumerate(src):
        nei_target_id = get_neighbors(indptr, indices, u)
        all_target_score[i][nei_target_id - num_users] = -np.inf
