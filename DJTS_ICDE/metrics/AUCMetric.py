import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import cupy
from sklearn.metrics import roc_auc_score

from LibMTL.metric import AbsMetric


class AUCMetric(AbsMetric):
    r"""Calculate the accuracy.
    """

    def __init__(self, metric_name: list = ['AUC']):
        all_metric_info = {'AUC': 1}
        super(AUCMetric, self).__init__(metric_name, all_metric_info)
        self.record = 0

    def update_fun(self, pred, gt):
        r"""
        """
        if not isinstance(gt,list):
            gt=gt.tolist()
            pred=pred.tolist()
        self.record=roc_auc_score(gt,pred)

        '''
        y_true_gpu = cupy.asarray(gt)
        y_score_gpu = cupy.asarray(pred)
        #
        auc_gpu = cupy.asarray(roc_auc_score(y_true_gpu.get(), y_score_gpu.get()))

        self.record = cupy.asnumpy(auc_gpu)
        # self.record = roc_auc_score(gt.cpu().numpy(), pred.cpu().numpy())
        '''

    def score_fun(self):
        if isinstance(self.record, list):
            self.record = 0
        # return {'AUC': self.record}
        return self.record


