# -*- coding: UTF-8 -*-
## Global settings and imports
import sys
sys.path.append("../../")
import os
import pickle

data_path = os.path.abspath(os.path.join(os.getcwd(), "../../data/lstur_data"))
print(data_path)

train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'lstur.yaml')
print(train_news_file)

# 查看npy内容
import numpy as np
a = np.load(wordEmb_file) ## 路径, embedding.npy
print(a, a.size, a.shape) # 9308400, (31028, 300)
print(a[1]) # array([ 2.7204e-01, -6.2030e-02, -1.8840e-01,  2.3225e-02, -1.8158e-02,..., -1.8317e-01,  1.3230e-01])



# 查看pkl文件内容, word_dict.pkl
with open(wordDict_file, 'rb') as f:
    line = pickle.load(f)
    print(line) # {'the': 1, 'brands': 2, 'queen': 3,..., 'reflecting': 31026, 'dominic': 31027}


