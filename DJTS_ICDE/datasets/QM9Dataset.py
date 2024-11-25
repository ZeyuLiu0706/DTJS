import torch, os
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Batch

class QM9Dataset(Dataset):
    def __init__(self, dataset, 
                 target: list=[0, 1, 2, 3, 5, 6, 12, 13, 14, 15, 11]):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.__getitem__(idx)
        label = {}
        for tn in self.target:
            label[str(tn)] = data.y[:, tn]
        return data, label

