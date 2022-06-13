from dataloader import Dataset
from holder import Holder


class Args:
    def __init__(self):
        self.DEVICE = "cuda:7"
        self.EPOCH = 200
        self.FEATS_IDX = 0
        self.HIDDEN_DIM = 128
        self.RNN_LAYER = 2


if __name__ == '__main__':
    args = Args()
    dataset = Dataset(conv_init_file="./data/guanpian/conv_init.csv",
                      parallel_info_file="./data/guanpian/parallel_info.json", device=args.DEVICE)
    holder = Holder(args=args, dataset=dataset)
    for epoch in range(args.EPOCH + 1):
        holder.train(args)
        holder.eval(args)
