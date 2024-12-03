import SORCL
from SORCL.data import io
from SORCL.utils.parse_arguments import parse_arguments
from SORCL.utils.utils import set_random_seed

import os.path as osp
import time



def main():

    config_file_root = './config/'
    config = io.load_yaml(osp.join(config_file_root, 'SORCL-config.yaml'))

    set_random_seed(config['seed'])


    # 各种初始化+propagation部分
    model = SORCL.create_model(config)

    # 训练部分
    model.fit()

    # 测试集评价指标计算
    test_results, rec_users = model.test()
    print("test:", test_results)
    io.save_json(osp.join(config['results_root'], 'test_results.json'), test_results)

    model_root = osp.join(config['results_root'], 'model')
    rec_path = osp.join(model_root, 'rec_users.txt')

    # 把推荐加入的边保存到新的txt文件中
    with open(rec_path, 'w') as file:
        for data in rec_users:
            user_id, recommendations = data
            for recommendation in recommendations:
                file.write(f'{user_id}\t{recommendation}')
                file.write('\n')



if __name__ == "__main__":
    main()



