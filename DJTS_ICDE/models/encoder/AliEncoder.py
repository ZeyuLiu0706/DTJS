import torch
# from .layers import EmbeddingLayer, MultiLayerPerceptron
import numpy as np
import warnings

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
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


class AliEncoder(torch.nn.Module):
    """
    A pytorch implementation of Shared-Bottom Model.
    """

    def __init__(self):
        super().__init__()
        categorical_field_dims = [9,4,7,2,20,7,50,8,8,2,2,2,2,2,2,2]
        numerical_num = 63
        embed_dim = 128
        bottom_mlp_dims = [512, 256]
        dropout = 0.2
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        # self.task_num = task_num
        self.bottom = MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False)
        # self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])

    # def forward(self, categorical_x, numerical_x):
    def forward(self, x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        # print("x",x.shape) 64,79
        warnings.filterwarnings("ignore", category=UserWarning)
        categorical_x = x[:,:16]
        numerical_x = x[:,16:]
        # categorical_x = torch.tensor(categorical_x, dtype=int).clone().detach().requires_grad_()
        categorical_x = torch.tensor(categorical_x, dtype=int)
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        fea = self.bottom(emb)
        return fea    
        # results = [torch.sigmoid(self.tower[i](fea).squeeze(1)) for i in range(self.task_num)]
        # return results
