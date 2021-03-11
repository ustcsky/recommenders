# coding:utf8
import os
# from PIL import Image
from torch.utils import data
import numpy as np
import csv

class LoadMind:
    def __init__(self,large = True, train=True):
        """
        主要目标： 获取数据集的地址，获取新闻和用户数据
        """
        self.large = large
        self.train = train
        self.this_file_dir = os.path.dirname(os.path.abspath(__file__))
        if self.large:
            self.root_large = 'large'
        else:
            self.root_large = 'small'
        if self.train:
            self.root_train = 'train'
        else:
            self.root_train = 'dev'
        self.root_news = self.this_file_dir + '/data/MIND' + self.root_large + '_' + self.root_train + '/news.tsv'
        self.root_behaviors = self.this_file_dir + '/data/MIND' + self.root_large + '_' + self.root_train + '/behaviors.tsv'


    def get_dataset_count(self):
        with open(self.root_news, 'r') as f_news:
            lines_news = f_news.readlines()
        with open(self.root_behaviors, 'r') as f_behaviors:
            lines_behaviors = f_behaviors.readlines()
        cnt_news = 0
        for line in lines_news:
            # print(line.strip())
            cnt_news += 1
        print('news numbers:', cnt_news)

        cnt_behaviors = 0
        for line in lines_behaviors:
            # print(line.strip())
            cnt_behaviors += 1
        print('behaviors numbers:', cnt_behaviors)

    def get_yq_data(self):
        yq_list = {'finance':0, 'news':0, 'middleeast':0, 'northamerica':0}
        root_news_yq = self.this_file_dir + '/data/MIND' + self.root_large + '_' + self.root_train + '/news_yq.csv'
        with open(root_news_yq, 'w') as f_yq:
            writer = csv.writer(f_yq)
            cnt = 0
            # 使用csv库读tsv文件
            with open(self.root_news, 'r') as f_news:
                lines_news = csv.reader(f_news, delimiter='\t')
                for line in lines_news:
                    if line[1] in list(yq_list.keys()):
                        yq_list[line[1]] += 1
                        writer.writerow(line)
                        cnt += 1
        print(self.root_news,'行数:', cnt)
        print(yq_list)

    def __getitem__(self, index):
        """
        一次返回one 数据
        """
        data_path = self.datas[index]
        if self.test:
            label = int(self.datas[index].split(',')[-2].split(',')[-1])
        else:
            label = 1 if 'yq' in data_path.split(',')[-1] else 0
        data = open(data_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.datas)


datas = LoadMind(large = True)
# datas.get_news_count()
datas.get_yq_data()