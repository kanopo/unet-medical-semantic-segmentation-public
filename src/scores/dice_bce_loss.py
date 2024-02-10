import torch
import torch.nn as nn


class dice_bce_loss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.smooth = smooth

    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError(
                "size mismatch, {} != {}".format(outputs.size(), targets.size())
            )

        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)

        intersection = (outputs * targets).sum()
        dice = (
            2.0
            * (intersection + self.smooth)
            / (targets.sum() + outputs.sum() + self.smooth)
        )

        loss -= self.dice_weight * torch.log(dice)  # try with 1- dice

        return loss
