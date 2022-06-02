import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


class Dataset(object):
    def __init__(self, conv_init_file, parallel_info_file, num_node=500, start_year=2014, end_year=2020,
                 train_ratio=0.8, num_features=2):
        self.conv_init_file = pd.read_csv(conv_init_file).fillna(0)
        with open(parallel_info_file, "rb") as f:
            self.parallel_info = json.load(f)
        self.time_span = end_year - start_year + 1
        self.num_node = num_node
        self.train_ratio = train_ratio
        self.num_features = num_features
        self.max_values = np.asarray(
            [self.conv_init_file['measure_inner'].max(), self.conv_init_file['inner_dfm'].max()])
        self._normalization()
        self.interval_ring_dict = self._generate_interval_ring_dict()
        self.features_list, self.targets_list = self._generate_all_features()



    def _generate_interval_ring_dict(self):
        '''
        生成一个记录interval对应的ring的字典
        :return:
        '''
        interval_ring_dict = {}
        for interval_id in self.conv_init_file['interval_id'].unique():
            interval_ring_dict[int(interval_id)] = []
            for ring_idx in self.conv_init_file[self.conv_init_file['interval_id'] == interval_id]['ring_num'].unique():
                interval_ring_dict[int(interval_id)].append(int(ring_idx))
        return interval_ring_dict

    def _normalization(self):
        '''
        数据标准化
        :return:
        '''
        self.conv_init_file['measure_inner'], self.conv_init_file['inner_dfm'] = self.conv_init_file['measure_inner'] / \
                                                                                 self.max_values[0], \
                                                                                 self.conv_init_file[
                                                                                     'inner_dfm'] / self.max_values[0]

    def _generate_all_features(self):
        '''
        汇总所有的记录
        :return:
        '''

        features_list = [self._construct_features_tensor(line_idx)[0] for line_idx in
                         self.parallel_info.keys()]
        target_list = [self._construct_features_tensor(line_idx)[1] for line_idx in
                       self.parallel_info.keys()]
        return features_list, target_list

    def _construct_features_tensor(self, line_idx):
        '''
        根据ring的顺序生成特征
        :param order:
        :return: [temporal,num_ring,feature_size]
        '''

        def filter_effect_ring():
            '''
            针对一个线路过滤掉没有记录的ring
            :return:
            '''
            return [idx for idx in self.parallel_info[str(line_idx)] if idx in self.interval_ring_dict.keys()]

        features = []
        interval_list = filter_effect_ring()
        for interval_idx in interval_list:
            current_interval_data = self.conv_init_file[self.conv_init_file['interval_id'] == interval_idx]
            current_measuer_innner = torch.Tensor(current_interval_data['measure_inner'].values)
            current_inner_dfm = torch.Tensor(current_interval_data['inner_dfm'].values)
            current_data = torch.stack([current_measuer_innner, current_inner_dfm])  # [2,1197]
            features.append(current_data.transpose(0, 1).view(self.time_span, -1, self.num_features))

        return torch.cat(features, dim=1)

    @staticmethod
    def construct_adj_mx(num_node):
        '''
        根据特征tensor生成邻接矩阵
        :param features:
        :return:
        '''
        adj = torch.zeros([num_node, num_node])
        for i in range(adj.shape[0]):
            if i == (adj.shape[0] - 1):
                adj[i, i] = 1
                adj[i, i - 1] = 1
            elif i == 0:
                adj[i, i] = 1
                adj[i, i + 1] = 1
            else:
                adj[i, i] = 1
                adj[i, i - 1] = 1
                adj[i, i + 1] = 1
        return adj

    @staticmethod
    def sym_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()
