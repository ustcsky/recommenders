# coding:utf-8
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P

class Model(nn.Module):
    def __init__(self, args, word_embedding):
        super(Model, self).__init__()
        print('Making model...')

        args.cpu = False # ??
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        print('args.cpu:', args.cpu)
        args.device = self.device
        print('args.device:', args.device)
        self.n_GPUs = args.n_GPUs
        print('self.n_GPUs:', self.n_GPUs)

        module = import_module('model.' + args.model.upper())
        self.model = module.make_model(args, word_embedding).to(self.device)
        print('self.device:', self.device)
        if args.load is not None:
            print('Loading model from', args.load)
            checkpoint = torch.load(args.load, map_location=self.device) # 提取网络结构
            self.model.load_state_dict(checkpoint['model']) # 加载网络权重参数
        else:
            print('Finish making model ' + args.model)

    def forward(self, batch):
        if self.train:
            if self.n_GPUs > 1 and not self.cpu:
                return P.data_parallel(self.model, batch, range(self.n_GPUs))
            else:
                return self.model(batch)