# coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from option import args
from preprocess import preprocess_data
from loader import Loader
from model import Model
from trainer import Trainer
import numpy as np

torch.manual_seed(args.seed)

def main():
    print('Start')
    if args.pre:
        preprocess_data(args)
    else:
        print('Skip data preprocessing')
    try:
        word_embedding = torch.from_numpy(
            np.load(os.path.join(args.data_dir, 'word_embedding.npy'))).float()
    except FileNotFoundError:
        word_embedding = None
    device = torch.device("cuda")
    my_model = Model(args, word_embedding).to(device)
    # print('my_model.device:', my_model.device)

    my_loader = Loader(args)
    my_trainer = Trainer(args, my_model, my_loader)
    while not my_trainer.terminate():
        my_trainer.train()
        my_trainer.test()
    my_trainer.plot_loss()
    print('End')


if __name__ == '__main__':
    main()
