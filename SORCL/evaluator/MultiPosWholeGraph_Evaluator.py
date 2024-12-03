from SORCL.dataloading.eval import MultiPosWholeGraph_EvalDataLoader
from SORCL.utils.metric import multi_pos_whole_graph_metrics
from SORCL.utils.utils import combine_dict_list_and_calc_mean

from tqdm import tqdm
import os.path as osp
from SORCL.data import io, csr


class MultiPosWholeGraph_Evaluator:

    def __init__(self, config, model, file_eval_set, batch_size):
        self.model = model
        self.config = config
        self.data = {}
        self.eval_dl = MultiPosWholeGraph_EvalDataLoader(
            file_eval_set, batch_size
        )
        # self.centr = load_json()  # 读取本地存储的节点的介数中心性json
        if 'all_degrees' in self.data:
            self.all_degrees = self.data['all_degrees']
        else:
            self.indptr = io.load_pickle(osp.join(self.config['data_root'], 'indptr.pkl'))
            self.indices = io.load_pickle(osp.join(self.config['data_root'], 'indices.pkl'))
            self.all_degrees = csr.get_degrees(self.indptr)  # 所有节点的度数
            del self.indptr, self.indices
        self.info = io.load_yaml(osp.join(self.config['data_root'], 'info.yaml'))
        if self.info['graph_type'] == 'user-item':
            self.num_users = self.info['num_users']
        else:
            self.num_nodes = self.info['num_nodes']


    def eval(self, desc='eval'):
        batch_results_list = []
        batch_results_weights = []
        rec_users = []  # 存储所有推荐用户
        num_samples = self.eval_dl.num_samples()

        if (hasattr(self.model, 'infer_out_emb_table')) and not (
                hasattr(self.model, 'out_emb_table') and self.model.out_emb_table is not None):
            # 读取最后训练得到的embedding矩阵： self.target_emb_table
            self.model.infer_out_emb_table()

        for batch_data in tqdm(self.eval_dl, desc=desc):
            src, pos = batch_data
            num_batch_samples = len(src)

            # 计算batch内用户和所有用户的点积，设置邻居分数为很小
            all_target_score = self.model._eval_a_batch(
                batch_data, eval_type='whole_graph_multi_pos'
            )

            batch_results, topk_users = multi_pos_whole_graph_metrics(src, pos, all_target_score, self.all_degrees, self.num_nodes)

            batch_results_list.append(batch_results)
            rec_users.extend(zip(src, topk_users))
            batch_results_weights.append(num_batch_samples / num_samples)

        results = combine_dict_list_and_calc_mean(batch_results_list, batch_results_weights)

        return results, rec_users
