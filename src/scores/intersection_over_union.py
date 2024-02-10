import torch
import torch.nn as nn


class intersection_over_union(nn.Module):
    def __init__(self):
        super(intersection_over_union, self).__init__()

    @staticmethod
    def forward(inputs, targets):

        inputs = torch.sigmoid(inputs)

        inputs = (inputs > 0.5).int()
        targets = (targets > 0.5).int()

        intersection = (
            (inputs & targets).int().sum((1, 2))
        )  # Will be zero if Truth=0 or Prediction=0
        union = (inputs | targets).int().sum((1, 2))  # Will be zzero if both are 0

        SMOOTH = 1e-8

        iou = (intersection + SMOOTH) / (union + SMOOTH)

        return iou.mean()
