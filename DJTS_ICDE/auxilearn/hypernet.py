from abc import abstractmethod

from torch import nn
from torch.nn.utils import weight_norm
import torch
import torch.nn.functional as F
import os


class Naive_hyper(nn.Module):
    def __init__(self, data_num, task_num):
        super(Naive_hyper, self).__init__()
        self.weights = nn.Embedding(data_num, task_num)
        self.nolinear = nn.Softplus()
    
    def forward(self, losses, sample_id):
        current_weight = self.nolinear(self.weights(sample_id))
        final_loss = ( (current_weight*losses).mean(0) ).sum()
        return final_loss

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

class HyperNet(nn.Module):
    def __init__(self, main_task, input_dim):
        super().__init__()
        self.main_task = main_task
        self.input_dim = input_dim

    def forward(self, losses, outputs=None, labels=None, data=None):
        pass

    def _init_weights(self):
        pass

    def get_weights(self):
        return list(self.parameters())


class MonoHyperNet(HyperNet):
    """Monotonic Hypernets

    """
    def __init__(self, main_task, input_dim, clamp_bias=False):
        super().__init__(main_task=main_task, input_dim=input_dim)
        self.clamp_bias = clamp_bias

    def get_weights(self):
        return list(self.parameters())

    @abstractmethod
    def clamp(self):
        pass

class MonoJoint(MonoHyperNet):

    def __init__(self,main_task,input_dim,device,nonlinearity=None,bias=True,dropout_rate=0.,weight_normalization=True,K = 2,init_lower= 0.0, init_upper=1.0):
        super().__init__(main_task=main_task, input_dim=input_dim)

        self.device = device
        self.nonlinearity = nonlinearity if nonlinearity is not None else nn.Softplus()
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_normalization = weight_normalization
        self.direction_a = nn.Parameter(torch.rand(1,input_dim).to(torch.device(device=self.device)), requires_grad=True)
        self.magnitude_b = nn.Parameter(torch.rand(1,input_dim).to(torch.device(device=self.device)), requires_grad=True)
        
        self.layers = []
        self.net = nn.Sequential(*self.layers)
    
    def norm_loss1(self,losses):
        ###### this performs bad
        m = losses.mean(dim=0,keepdim = True)
        std = losses.std(0, keepdim=True)
        return (losses - m) / (std + 1e-6)

    @staticmethod
    def _init_layer(layer, init_lower, init_upper):
        b = init_upper if init_upper is not None else 1.
        a = init_lower if init_lower is not None else 0.
        if isinstance(layer, nn.Linear):
            layer.weight = nn.init.uniform_(layer.weight, b=b, a=a)
            if layer.bias is not None:
                layer.bias = nn.init.constant_(layer.bias, 0.)
    
    def get_loss_weight(self, dir_matrix, mag_matrix):
        dir_matrix=torch.unsqueeze(torch.stack(dir_matrix), dim=0)
        mag_matrix=torch.unsqueeze(torch.stack(mag_matrix), dim=0)

        dir = torch.sigmoid(dir_matrix*self.direction_a)
        mag = torch.sigmoid(mag_matrix*self.magnitude_b)
        return (dir+mag)


    def forward(self, dir_matrix, mag_matrix, losses, to_train = True):
        # detached_losses = losses.detach()
        # print('task num:',self.input_dim)
        inter_schedule = self.get_loss_weight(dir_matrix,mag_matrix)
        losses = torch.stack(losses)
        if not to_train:
            dir_matrix=torch.unsqueeze(torch.stack(dir_matrix), dim=0)
            mag_matrix=torch.unsqueeze(torch.stack(mag_matrix), dim=0)
            return ((inter_schedule*losses).sum())
        else:
            detached_mask = inter_schedule.detach()
            # return ((detached_mask*losses).sum()).mean()
            return ((detached_mask*losses).sum())


    def clamp(self):
        for l in self.net:
            if isinstance(l, nn.Linear):
                if self.weight_normalization:
                    l.weight_v.data.clamp_(0)
                    l.weight_g.data.clamp_(0)
                else:
                    l.weight.data.clamp_(0)

                if l.bias is not None and self.clamp_bias:
                    l.bias.data.clamp_(0)

