import numpy as np
import torch
import torch.optim as op

from model import Model


class Holder():
    def __init__(self, args, dataset):
        '''
        整体训练管理类
        :param args:传入参数
        :param dataset: 传入数据集
        '''
        self.args = args
        self.DEVICE = args.DEVICE
        self.dataset = dataset
        self.model = Model(self.dataset, args).to(args.DEVICE)
        self.optimizer = op.Adam(self.model.parameters(), lr=0.00001, weight_decay=0.000001)
        self.lr_sch = op.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9, verbose=True)

    def train(self, args):
        self.model.train()
        train_loss = []
        for iter, (x, y) in enumerate(self.dataset.train_loader):
            self.optimizer.zero_grad()
            x = torch.Tensor(x[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            y = torch.Tensor(y[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            pred, z_mu, z_sigma = self.model(x, self.dataset.adj_mx)
            pred = self.dataset.inv_normalization(pred, dim=self.args.FEATS_IDX)
            real = self.dataset.inv_normalization(y, dim=self.args.FEATS_IDX)
            loss = self.masked_mse(pred, real, 0.0) + self.latent_loss(z_mu, z_sigma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            train_loss.append(loss.item())
            log = 'Train Loss: {:.4f}'
            if iter % 200 == 0:
                print(log.format(np.mean(train_loss)), flush=True)

    def eval(self, args):
        self.model.eval()
        test_mae = []
        test_mape = []
        test_rmse = []
        for iter, (x, y) in enumerate(self.dataset.test_loader):
            x = torch.Tensor(x[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            y = torch.Tensor(y[..., args.FEATS_IDX]).unsqueeze(-1).to(args.DEVICE)
            with torch.no_grad():
                pred, _, _ = self.model(x, self.dataset.adj_mx)
            pred = self.dataset.inv_normalization(pred, dim=self.args.FEATS_IDX)
            real = self.dataset.inv_normalization(y, dim=self.args.FEATS_IDX)
            mae = self.masked_mse(pred, real, 0.0)
            rmse = self.masked_rmse(pred, real, 0.0)
            mape = self.masked_mape(pred, real, 0.0)
            test_mae.append(mae.item())
            test_mape.append(mape.item())
            test_rmse.append(rmse.item())

        log = 'Test Loss: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(np.mean(test_mae), np.mean(test_mape), np.mean(test_rmse)), flush=True)

    @staticmethod
    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

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
