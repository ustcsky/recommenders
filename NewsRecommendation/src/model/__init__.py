from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P

class Model(nn.Module):
    def __init__(self, args, word_embedding):
        super(Model, self).__init__()
        print('Making model...')

        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        args.device = self.device
        self.n_GPUs = args.n_GPUs

        module = import_module('model.' + args.model.upper())
        self.model = module.make_model(args, word_embedding).to(self.device)
        if args.load is not None:
            print('Loading model from', args.load)
            checkpoint = torch.load(args.load, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
        else:
            print('Finish making model ' + args.model)

    def forward(self, batch):
        if self.train:
            if self.n_GPUs > 1 and not self.cpu:
                return P.data_parallel(self.model, batch, range(self.n_GPUs))
            else:
                return self.model(batch)