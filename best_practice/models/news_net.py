# coding:utf8
from torch import nn
from basic_module import BasicModule
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
# % matplotlib inline
import numpy as np

class FM_Layer(nn.Module):
    def __init__(self, n=10, k=3):
        '''
        n: 输入维度
        k: factor的维度
        '''
        super(FM_Layer, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)   # 前两项线性层
        self.V = nn.Parameter(torch.randn(self.n, self.k))   # 交互矩阵

    def fm_layer(self, x):
        '''
         :输入:  x 为一个n维向量
        :返回:  实数值
         '''
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        output = linear_part + torch.sum(0.5 * interaction_part_2 - interaction_part_1)
        return output
    def forward(self, x):
        return self.fm_layer(x)

class NewsNet(BasicModule):
    def __init__(self, num_classes=2):
        super(NewsNet, self).__init__()
        self.model_name = 'news_net'
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text = "Here is the sentence I want embeddings for."
        text1 = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
        marked_text = "[CLS] " + text + " [SEP]"
        print(marked_text)
        tokenized_text = tokenizer.tokenize(marked_text)
        print(tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        for tup in zip(tokenized_text, indexed_tokens):
            print(tup)
        self.emb1 = indexed_tokens
        # fm = FM_Layer(10, 5)
        # x = torch.randn(1, 10)
        # output = fm(x)
        fm = FM_Layer(self.topic.features + self.hot.features + self.burst.features, 100, self.emb2.size() + 1 + 1)
        x = torch.cat(self.topic.features,
                      self.hot.features,
                      self.burst.features)
        output = fm(x)
        self.emb2 = output
        self.classifier = torch.cat(
            self.emb1,
            self.emb2,
        )

    def load_para(word_index, max_features):
        EMBEDDING_FILE = '../embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        def get_coefs(word, *arr):
            return word, np.asarray(arr, dtype='float32')

        embeddings_index = dict(
            get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

        all_embs = np.stack(embeddings_index.values())
        emb_mean, emb_std = -0.0053247833, 0.49346462
        embed_size = all_embs.shape[1]
        print(emb_mean, emb_std, "para")

        # word_index = tokenizer.word_index
        nb_words = min(max_features, len(word_index))
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, i in word_index.items():
            if i >= max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
