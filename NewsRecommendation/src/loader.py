from dataset import MyDataset, TestNewsDataset
from torch.utils.data import DataLoader

class Loader:
    def __init__(self, args):
        self.train_loader = None
        if args.train:
            self.train_loader = DataLoader(
                MyDataset(args, 'train'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_threads,
                pin_memory=not args.cpu,
                drop_last=True
            )
        self.test_news_loader = DataLoader(
            TestNewsDataset(args),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=not args.cpu,
            drop_last=False
        )