import networkx as nx
from SORCL.utils.utils import combine_dict_list_and_calc_mean


import numpy as np
import numba
from numba.typed import Dict
from numba.core import types
import json
import math

@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, 2 + 300))
    if length > len(ndcg_weights):
        ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights[:length]


@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, 1 + 300)
    if length > len(mrr_weights):
        mrr_weights = 1 / np.arange(2, length + 2)
    return mrr_weights[:length]


# 每个列表的排名是通过将列表的第一个元素与其余元素进行比较并计算有多少元素大于第一个元素来确定的
@numba.jit(nopython=True)
def get_rank(A):
    rank = np.empty(len(A), dtype=np.int32)
    for i in range(len(A)):
        a = A[i]
        key = a[0]
        r = 0
        for j in range(1, len(a)):
            if a[j] > key:
                r += 1
        rank[i] = r  # 表示当前用户，有多少个用户分数大于pos（推荐列表）
    return rank

# 验证集评估计算
def one_pos_metrics(S):
    # S的每一行表示一个user与其他节点之间的分数
    num_samples = S.shape[0]
    num_scores = S.shape[1]

    # add small noises
    S += np.random.uniform(low=-1e-6, high=1e-6, size=S.shape)   #  S(256, 1632723)
    # sorted_S = np.argsort(S, axis=1)[:, ::-1]
    # score_list100 = sorted_S[:, :100]
    # ILAD100 = np.mean(np.mean(1 - score_list100, axis=1))


    rank = get_rank(S)

    top5 = rank < 5
    top10 = rank < 10
    top20 = rank < 20
    top50 = rank < 50

    results = {
        "auc": (num_scores - 1 - rank).mean() / (num_scores - 1)
    }

    # ndcg
    w = _get_ndcg_weights(num_scores)
    results.update({
        # "ndcg": w[rank].sum() / num_samples,
        # "n1": top1.mean(),
        "n5": w[rank[top5]].sum() / num_samples,
        "n10": w[rank[top10]].sum() / num_samples,
        "n20": w[rank[top20]].sum() / num_samples,
        "n50": w[rank[top50]].sum() / num_samples,
    })

    # recall
    results.update({
        "r5": top5.mean(),
        "r10": top10.mean(),
        "r20": top20.mean(),
        "r50": top50.mean(),
    })


    # mrr
    # w_mrr = _get_mrr_weights(num_scores)
    # results.update({
    #     # "mrr": w[rank].sum() / num_samples,
    #     "m20": w_mrr[rank[top20]].sum() / num_samples,
    #     "m50": w_mrr[rank[top50]].sum() / num_samples,
    #     "m100": w_mrr[rank[top100]].sum() / num_samples,
    #     "m300": w_mrr[rank[top300]].sum() / num_samples,
    # })

    return results

def multi_pos_whole_graph_metrics(src, pos: list, all_target_score, all_degrees, num_nodes):
    results_dict_list, topk_users = multi_pos_metrics(src, pos, all_target_score, all_degrees, num_nodes)
    results = combine_dict_list_and_calc_mean(results_dict_list)
    return results, topk_users


# 查找输入数组 a 中前 k 个最大元素的索引
# @numba.jit(nopython=True)
def argtopk(a, k):
    if k == 1:
        # 只获取最大值top-1，直接用np函数得到
        return np.array([np.argmax(a)])
    else:
        # 取top-k
        ind = np.argpartition(a, -k)[-k:]  # 查找数组a中k个最大元素的索引
        return ind[np.argsort(a[ind])][::-1]  # 对这k个索引对应值进行降序排列后得到排序后的索引列表


def _lookup_degreeSim(all_degrees, u, v, num_nodes):
    # u是当前用户，v是还未推荐用户
    DC_u = all_degrees[u] / (num_nodes-1)
    DC_v = all_degrees[v] / (num_nodes-1)

    sim = DC_v - DC_u
    return sim

def _centrality_list(centrality, user_list, u):
    v_centrality_list = [centrality[str(i)] for i in user_list]
    u_centr = centrality[str(u)]
    return u_centr, v_centrality_list


def _get_centralitySim(u_centr, v_centr):
    sim = (v_centr-u_centr) * math.pow(10,0)
    return sim

# def _rerank(rec_unranked, user, rel_matrix):
#     new_topk_id = []
#     L_s = list(rec_unranked)  #未重排的用户列表
#     while L_s:
#         mmr = 0
#         lambda_score = 0.0
#         rel_sum = 0
#         centr_sum = 0
#
#         u_centr, v_centrality_list = _centrality_list(rec_unranked, user) # 被推荐用户的centr
#         for index, v in enumerate(L_s):
#             centr = _get_centralitySim(u_centr, v_centrality_list[index])   # 介数中心性差异
#             rel = rel_matrix[v]  # 相似分数
#             mmr_current = (1 - lambda_score) * rel + lambda_score * centr
#             rel_sum += rel
#             centr_sum += centr
#             # argmax mmr
#             if mmr_current > mmr:
#                 mmr = mmr_current
#                 new_topk_id.append(v)
#                 L_s.remove(v)
#                 # 动态调整lambda_score,如果前面选的集合相关性rel较高，后续选择则会偏向可达性centr
#
#                 lambda_score = lambda_score + (rel_sum-centr_sum)/(rel_sum+centr_sum)  # 还是按照个数设计，不用绝对值
#             else:
#                 continue
#
#     return np.array(new_topk_id)

def _mmr(lambda_score, user, rec_unranked, rel_matrix, all_degrees, num_nodes, centrality):
    mmr = 0
    rec_user = None
    u_centr, v_centrality_list = _centrality_list(centrality, rec_unranked, user)  # 中心用户的centr
    for index, d in enumerate(rec_unranked):
        # sim = _lookup_degreeSim(all_degrees, user, d, num_nodes)  # 度差异
        sim = _get_centralitySim(u_centr, v_centrality_list[index])   # 介数中心性差异
        rel = rel_matrix[d]  # 相似分数
        mmr_current = (1 - lambda_score) * rel + lambda_score * sim
        # argmax mmr
        if mmr_current > mmr:
            mmr = mmr_current
            rec_user = d
        else:
            continue
    if rec_user==None:
        rec_user = rec_unranked[0]

    assert rec_user is not None, "mmr error!"

    return mmr, rec_user


def rank_by_degree(L_s, u, rel, degrees, num_nodes, k, centrality):
    '''
    ！重要：重排记得修改介数中心性文件
    topk_id = rank_by_degree(topk_id, src[i], all_target_score[i], all_degrees, num_nodes, max_k)
    功能；对用户u的推荐用户列表进行重排
    L_s: 根据相似度得到的，用户u的推荐用户列表
    u: 用户u
    rel: 用户u和其他所有用户的相似度分数
    degrees:所有节点的度
    num_nodes: 节点总数
    k: 重排的top-k
    '''
    new_topk_id = []
    L_s = list(L_s)
    topk_id = L_s[:k]
    while L_s:
        num = len([u for u in new_topk_id[:k] if u not in topk_id])
        lambda_score = max(0.4 - num / (2 * k), 0)

        mmr_score, rec_u = _mmr(
            lambda_score,
            u,
            L_s,
            rel,
            degrees,
            num_nodes, 
            centrality
        )
        new_topk_id.append(rec_u)
        L_s.remove(rec_u)

    assert len(L_s) == 0, "rank_by_degree error!"

    return np.array(new_topk_id)


# @numba.jit(nopython=True, parallel=True)
def multi_pos_metrics(src, pos_list, all_target_score, all_degrees, num_nodes):
    results_dict_list = [
        Dict.empty(key_type=types.unicode_type, value_type=types.float32)
        for _ in range(len(pos_list))
    ]

    # 随机加点噪声
    all_target_score += np.random.uniform(low=-1e-6, high=1e-6, size=all_target_score.shape)  # shape(256, 4039)

    topk_list = [5, 10, 20, 50]
    max_k = topk_list[-1]
    topk_users = []

    ndcg_weights = 1 / np.log2(np.arange(2, max_k + 2))
    with open('./dataset/deezer/betweenness_centrality.json', 'r') as f:
        centrality = json.load(f)
    # 获取pred_label列表
    # for循环里是一个用户的评价指标计算，k最大为50
    for i in range(len(pos_list)):
        pos = pos_list[i]
        pos_set = set(list(pos))
        # 当前用户的前50个推荐分数的用户ID
        topk_id = argtopk(all_target_score[i], max_k)   # (50,)
        ####-------------重排--------------####
        topk_id = rank_by_degree(topk_id, src[i], all_target_score[i], all_degrees, num_nodes, max_k, centrality)
        topk_users.append(topk_id)  # len(topk_users[0])==50

        ground_truth_label = np.zeros(max_k)
        ground_truth_label[:len(pos)] = 1  # gt标签列表

        # 看topk_id里的id是否在pos里，在则pred里对应索引位置为1
        pred_label = np.zeros(max_k)   # (300,)
        for j in range(max_k):
            v = topk_id[j]  # 通过索引获取用户ID
            if v in pos_set:
                pred_label[j] = 1

        # 下面评价指标计算就是比较 pred_label列表 和 ground_truth_label列表
        results_dict = {}

        # calc recall
        for k in topk_list:
            results_dict['r' + str(k)] = pred_label[:k].sum() / ground_truth_label.sum()

        # calc ndcg
        s = pred_label * ndcg_weights
        truth_s = ground_truth_label * ndcg_weights
        for k in topk_list:
            results_dict['n' + str(k)] = s[:k].sum() / truth_s[:k].sum()

        for key in results_dict:
            results_dict_list[i][key] = results_dict[key]

    return results_dict_list, topk_users
