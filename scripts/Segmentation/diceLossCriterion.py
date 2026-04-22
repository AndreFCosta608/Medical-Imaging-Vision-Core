import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.float()  # [B, C, H, W]
        probs = torch.softmax(logits, dim=1)
        preds = probs[:, 1, :, :]  # [B, H, W]

        if targets.ndim == 4:
            targets = targets.squeeze(1)  # [B, H, W]
        targets = (targets == 1).float()  # binariza se necessário

        intersection = (preds * targets).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
