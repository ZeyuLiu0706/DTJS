import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric
from LibMTL.loss import AbsLoss

class QM9Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self, std, scale: bool = True, metric_name: list = ['MAE']):
        all_metric_info = {'MAE': 0}
        super(QM9Metric, self).__init__(metric_name, all_metric_info)

        self.std = std
        self.scale = scale
    
    def update_fun(self, pred, gt):
        r"""
        """
        if self.scale:
            abs_err = torch.abs(
                pred * (self.std).to(pred.device) - gt * (self.std).to(pred.device)
            ).view(pred.size()[0], -1).sum(-1)
        else:
            abs_err = torch.abs(
                pred.to(pred.device) - gt.to(pred.device)
            ).view(pred.size()[0], -1).sum(-1)
        self.record.append(abs_err.cpu().numpy())


    def score_fun(self):
        r"""
        """
        records = np.concatenate(self.record)
        return records.mean()
        # return {'MAE': records.mean()}
