import torch
from torch import nn


class ConfidenceLoss(nn.Module):
    def __init__(self, weighted=True):
        super().__init__()
        self.weighted = weighted

    def forward(self, conf_pred, conf_true):
        if self.weighted:
            h = conf_pred.shape[0]
            r = conf_true.sum().item()
            p1 = (h - r) / h
            p2 = r / h
            if r == 0:
                weights = torch.ones_like(conf_true)
            else:
                weights = (p1 / r) * conf_true + (1 - conf_true) * (p2 / h)
            return nn.functional.binary_cross_entropy(conf_pred, conf_true, weights)
        else:
            return nn.functional.binary_cross_entropy(conf_pred, conf_true)
