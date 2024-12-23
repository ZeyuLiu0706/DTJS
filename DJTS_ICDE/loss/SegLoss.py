import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from abstract_loss import AbsLoss
# from LibMTL.loss.abstract_loss import AbsLoss
from loss.abstract_loss import AbsLoss

class SegLoss(AbsLoss):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')
        self.test_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())
    
    def mean_loss(self, pred, gt):
        return self.test_loss_fn(pred, gt.long())