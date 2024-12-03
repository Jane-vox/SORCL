from torch.nn import Embedding
import os

from .TrainTracer import TrainTracer
import SORCL
from SORCL.data import io
from SORCL.utils.Timer import Timer
from SORCL.utils.utils import get_formatted_results
from SORCL.model.SORCL.module import *


import numpy as np
import os.path as osp
from tqdm import tqdm
import torch
from time import sleep
 
# log_writer = SummaryWriter()


def create_Trainer(config, data, model, train_dl):
    trainer = Trainer(
        config, data, model, train_dl,
    )
    return trainer


class Trainer:
    
    def __init__(self, config, data, model, train_dl):
        self.config = config
        self.data = data
        self.model = model
        self.train_dl = train_dl
        
        self.epochs = self.config['epochs']
        self.results_root = self.config['results_root']
        # self.w_bpr_uniform = [1,1,1]
        
        self.do_val = self.config['use_validation_for_early_stop']
        if self.do_val:
            self.val_method = SORCL.create_val_Evaluator(self.config, self.data, self.model)
            self.val_freq = self.config['val_freq']
            # 记录训练过程各值
            self.train_tracer = TrainTracer(
                data, model,
                key_score_metric=self.config['key_score_metric'],
                convergence_threshold=self.config['convergence_threshold'],
                results_root=self.results_root
            )
        
        self.timer = Timer(record_root=self.results_root)


    # 调用：train → _train_loop → _train_an_epoch
    def train(self):
        self.timer.start("train")
        if hasattr(self.model, 'on_train_begin'):
            self.model.on_train_begin()
        
        try:
            self._train_loop()
        except KeyboardInterrupt:
            pass

        if not self.do_val:
            self.model.save()
        
        if hasattr(self.model, 'on_train_end'):
            self.model.infer_out_emb_table()
            self.model.on_train_end()
            
        self.timer.end("train")
        self.timer.save_record()
    
    def _train_loop(self):
        # 训练200个epoch
        for epoch in range(self.epochs):
            self.data['epoch'] = epoch

            # 训练过程中的验证集指标计算
            if self.do_val and (epoch % self.val_freq == 0):
                self.timer.start("val")
                if hasattr(self.model, 'on_val_begin'):
                    self.model.on_val_begin()
                
                results = self.val_method.eval(desc='val')
                
                if hasattr(self.model, 'on_val_end'):
                    self.model.on_val_end()
                self.timer.end("val")
                
                print("val:", results)
                results.update({"loss": np.nan if epoch == 0 else epoch_loss})
                # 训练过程中验证集指标记录
                is_converged = self.train_tracer.check_and_save(epoch, results)
                if is_converged:
                    break

            # 开始训练epoch
            self.timer.start("epoch")
            if hasattr(self.model, 'on_epoch_begin'):

                self.model.on_epoch_begin()
            
            if hasattr(self.train_dl, 'subgraph_dl'):
                print("###----")
                with self.train_dl.subgraph_dl.enable_cpu_affinity():
                    epoch_loss = self._train_an_epoch()
            else:
                # 训练一个epoch
                epoch_loss = self._train_an_epoch()

            if hasattr(self.model, 'on_epoch_end'):
                self.model.on_epoch_end()
            self.timer.end("epoch")
        
    def _train_an_epoch(self):
        epoch = self.data['epoch'] + 1
        print('epoch {0}'.format(epoch))

        if hasattr(self.model, 'train_an_epoch'):
            epoch_loss = self.model.train_an_epoch()
        else:
            loss_list = []
            task_loss_list = []
            disc_loss_list = []

            for batch_data in tqdm(self.train_dl, desc='train'):
                self.timer.start("batch")

                disc_loss = self.model.train_disc(epoch, batch_data)
                disc_loss_list.append(disc_loss)

                loss, task_loss = self.model.forward_and_backward(epoch, batch_data)
                loss_list.append(loss)
                task_loss_list.append(task_loss)

                self.timer.end("batch")
                self.timer.save_record()

            epoch_loss = np.array(loss_list).mean()
            print("All loss: ", epoch_loss)

            task_loss = np.array(task_loss_list).mean()
            print("Task loss: ", task_loss)

            disc_loss = np.array(disc_loss_list).mean()
            print("Disc loss: ", disc_loss)

        return epoch_loss

