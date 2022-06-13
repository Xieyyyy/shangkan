import numpy as np
import torch
import torch.optim as op

from model import Model


class Holder():
    def __init__(self, args, dataset):
        self.args = args
        self.DEVICE = args.DEVICE
        self.dataset = dataset
        self.model = Model(self.dataset, args).to(args.DEVICE)
        self.optimizer = op.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.000001)
        self.lr_sch = op.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9, verbose=True)

    def train(self, args):
        self.model.train()
        train_loss = []
        train_mape = []
        train_rmse = []
        for iter, (x, y) in enumerate(self.dataset.train_loader):
            x = torch.Tensor(x[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            y = torch.Tensor(y[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            pred = self.model(x, self.dataset.adj_mx)

    @staticmethod
    def masked_mse(preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = (preds - labels) ** 2
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    @staticmethod
    def masked_rmse(preds, labels, null_val=np.nan):
        return torch.sqrt(Holder.masked_mse(preds=preds, labels=labels, null_val=null_val))

    @staticmethod
    def masked_mae(preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels)
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)

    @staticmethod
    def masked_mape(preds, labels, null_val=np.nan):
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels != null_val)
        mask = mask.float()
        mask /= torch.mean((mask))
        mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        loss = torch.abs(preds - labels) / labels
        loss = loss * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        return torch.mean(loss)
