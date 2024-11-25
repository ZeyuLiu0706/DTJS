import numpy as np
import pandas as pd
import torch
import os, torch, fnmatch, random
import numpy as np
import torch.nn.functional as F

from LibMTL.dataset import AbsDataset

class AliExpressDataset(AbsDataset):
    def __init__(self,
                 path: str,
                 task_name: list = ['CTR', 'CTCVR'],
                 augmentation: bool = False):
        super(AliExpressDataset, self).__init__(path=path, 
                                        task_name=task_name,
                                        augmentation=augmentation)
        data = pd.read_csv(path).to_numpy()[:, 1:]
        self.categorical_data = data[:, :16].astype(np.int64)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1
    def __len__(self):
        return self.labels.shape[0]
    def _get_data_labels(self, idx):
        # print("c",self.categorical_data.shape) 
        # print("n",self.numerical_data.shape)
        categorical_tensor = torch.from_numpy(self.categorical_data[idx])
        numerical_tensor = torch.from_numpy(self.numerical_data[idx])
        labels={}
        for task in self.task_name:
            if task == 'CTR':
                label=self.labels[idx][0]
            elif task == 'CTCVR':
                label=self.labels[idx][1]
            # labels[task]=torch.from_numpy(label).float()
            labels[task]=torch.tensor(label,dtype=float)
        # print("c",self.categorical_data[idx].shape) 
        # print("n",self.numerical_data[idx].shape)
        combined_data=torch.cat((categorical_tensor, numerical_tensor))
        return combined_data, labels


# import numpy as np
# import pandas as pd
# import torch

# class AliExpressDataset(torch.utils.data.Dataset):
#     """
#     AliExpress Dataset
#     This is a dataset gathered from real-world traffic logs of the search system in AliExpress
#     Reference:
#         https://tianchi.aliyun.com/dataset/dataDetail?dataId=74690
#         Li, Pengcheng, et al. Improving multi-scenario learning to rank in e-commerce by exploiting task relationships in the label space. CIKM 2020.
#     """

#     def __init__(self, dataset_path):
#         data = pd.read_csv(dataset_path).to_numpy()[:, 1:]
#         self.categorical_data = data[:, :16].astype(np.int)
#         self.numerical_data = data[:, 16: -2].astype(np.float32)
#         self.labels = data[:, -2:].astype(np.float32)
#         self.numerical_num = self.numerical_data.shape[1]
#         self.field_dims = np.max(self.categorical_data, axis=0) + 1

#     def __len__(self):
#         return self.labels.shape[0]

#     def __getitem__(self, index):
#         print(self.labels[index])
#         return self.categorical_data[index], self.numerical_data[index], self.labels[index]