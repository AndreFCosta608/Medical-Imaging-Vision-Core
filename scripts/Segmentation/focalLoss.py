import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        logits = logits.float()
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()
