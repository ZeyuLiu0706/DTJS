import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()

        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class AliDecoder(nn.Module):
    def __init__(self):
        super(AliDecoder, self).__init__()
        bottom_mlp_dims = [512, 256]
        tower_mlp_dims = [128, 64]
        dropout = 0.2
        # self.task_num = task_num
        # self.net = nn.Linear(input_num, output_num)
        self.tower = MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout)


    def forward(self, x):
        # print("shape x:",x.shape)
        results = torch.sigmoid(self.tower(x).squeeze(1))
        return results
        # return self.net(x)