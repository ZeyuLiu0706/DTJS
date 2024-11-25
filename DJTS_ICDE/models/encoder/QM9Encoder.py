import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set

class QM9Encoder(nn.Module):
    def __init__(self, num_features, dim):
        super(QM9Encoder, self).__init__()

        self.lin0 = torch.nn.Linear(num_features, dim)

        net = nn.Sequential(nn.Linear(5, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        
        # gru不支持二次求导
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)
        
    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
    
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        # out = self.lin2(out)
        return out  # .view(-1)