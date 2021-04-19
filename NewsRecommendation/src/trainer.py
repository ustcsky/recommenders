# coding:utf-8
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from metrics import AUC, MRR, nDCG
import numpy as np
from dataset import TestUserDataset
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, args, model, loader):
        self.device = torch.device("cuda")
        self.args = args
        self.model = model
        self.train_loader = loader.train_loader
        self.test_news_loader = loader.test_news_loader
        self.optimizer = torch.optim.Adam(model.model.parameters(), lr=args.lr,
                                          betas=args.betas, eps=args.epsilon)
        self.loss_list = []
        self.epoch = 0
        self.batch_num = 0

        if args.load is not None:
            checkpoint = torch.load(args.load, map_location=self.args.device)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch']
            self.batch_num = checkpoint['batch_num']
            self.loss_list = checkpoint['loss_list']
            print('Loading optimizer from', args.load)
            print('Resume from epoch', self.epoch)
        if not os.path.exists('../../result'):
            os.mkdir('../../result')
        if self.args.save is None:
            dir_name = self.args.model + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            dir_name = self.args.save
        self.save_dir = os.path.join('../../result', dir_name)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        if not os.path.exists(os.path.join(self.save_dir, 'checkpoints')):
            os.mkdir(os.path.join(self.save_dir, 'checkpoints'))

    def train(self):
        self.model.train()
        print('[Epoch ' + str(self.epoch + 1) + ']')
        cnt_test = 0
        # print('/', len(self.train_loader)) ##
        with tqdm(total=len(self.train_loader), desc='Training') as p:
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # if self.args.model == 'TANR':
                #     predict, topic_loss = self.model(batch)
                # else:
                predict = self.model(batch) # 前向传播
                # if cnt_test == 0:
                #     print('predict:', predict)
                loss = torch.stack([x[0] for x in -F.log_softmax(predict, dim=1)]).mean() # 计算loss
                # if cnt_test == 0:
                #     # print('-F.log_softmax(predict, dim=1):', -F.log_softmax(predict, dim=1))
                #     print('loss:', loss)
                #     cnt_test += 1
                # if self.args.model == 'TANR':
                #     loss += self.args.topic_weight * topic_loss
                self.loss_list.append(loss.item())
                if i % 200 == 0 or i + 1 == len(self.train_loader):
                    tqdm.write('Loss: ' + str(np.mean(self.loss_list)))
                    # if self.args.model == 'TANR':
                    #     tqdm.write('\tTopic_loss: ' + str(topic_loss.item()))
                loss.backward() # 反向传播
                self.optimizer.step() # 更新参数
                self.batch_num += 1
                p.update(1)
        self.epoch += 1
        self.save()

    def test(self):
        with torch.no_grad():
            self.model.eval()
            AUC_list, MRR_list, nDCG5_list, nDCG10_list = [], [], [], []
            total_news_id = {}
            with tqdm(total=len(self.test_news_loader), desc='Generating test news vector') as p:
                for i, batch in enumerate(self.test_news_loader):
                    news_id_batch, news_batch = batch
                    news_vector = self.model.model.get_news_vector(news_batch)
                    if i == 0:
                        total_test_news = news_vector
                    else:
                        total_test_news = torch.cat([total_test_news, news_vector])
                    for news_id in news_id_batch:
                        total_news_id[news_id] = len(total_news_id)
                    p.update(1)
            test_user_loader = DataLoader(
                TestUserDataset(self.args, total_test_news, total_news_id),
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.n_threads,
                pin_memory=False
            )
            with tqdm(total=len(test_user_loader), desc='Generating test user vector') as p:
                for i, batch in enumerate(test_user_loader):
                    user_vector = self.model.model.get_user_vector(batch)
                    if i == 0:
                        total_test_user = user_vector
                    else:
                        total_test_user = torch.cat([total_test_user, user_vector])
                    p.update(1)
            behaviors = pd.read_table(os.path.join(self.args.data_dir, 'test_behaviors.csv'), na_filter=False,
                                      usecols=[2, 3], names=['impressions', 'y_true'], header=0)
            with tqdm(total=len(behaviors), desc='Predicting') as p:
                for row in behaviors.itertuples():
                    user_vector = total_test_user[row.Index]
                    tmp = row.impressions.split(' ')
                    news_vector = torch.cat([total_test_news[total_news_id[i]].unsqueeze(dim=0) for i in tmp])
                    y_true = [int(x) for x in row.y_true.split(' ')]
                    predict = torch.matmul(news_vector, user_vector).tolist()
                    AUC_list.append(AUC(y_true, predict))
                    MRR_list.append(MRR(y_true, predict))
                    nDCG5_list.append(nDCG(y_true, predict, 5))
                    nDCG10_list.append(nDCG(y_true, predict, 10))
                    p.update(1)
            print('AUC:', np.mean(AUC_list))
            print('MRR:', np.mean(MRR_list))
            print('nDCG@5:', np.mean(nDCG5_list))
            print('nDCG@10:', np.mean(nDCG10_list))

    def save(self):
        print('Saving model...')
        checkpoint = {
            'model': self.model.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'batch_num': self.batch_num,
            'loss_list': self.loss_list
        }
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoints', 'epoch' + str(self.epoch) + '.pt'))
        print('Model saved as', os.path.join(self.save_dir, 'checkpoints', 'epoch' + str(self.epoch) + '.pt'))

    def terminate(self):
        if not self.args.train:
            self.test()
            return True
        else:
            return self.epoch >= self.args.epochs

    def plot_loss(self):
        print('Plotting loss figure...')
        plt.plot([_ for _ in range(self.batch_num)], self.loss_list)
        plt.savefig(os.path.join(self.save_dir, 'loss.png'))