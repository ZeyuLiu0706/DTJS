import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss.abstract_loss import AbsLoss

class MSELoss(AbsLoss):
    r"""The Mean Squared Error (MSE) loss function.
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.mean_loss_fn = nn.MSELoss()
        self.sample_loss = nn.MSELoss(reduction='none')
        
    def compute_loss(self, pred, gt):
        loss = self.sample_loss(pred,gt)
        return loss
    
    def mean_loss(self, pred, gt):
        r"""
        """
        loss = self.mean_loss_fn(pred, gt)
        return loss