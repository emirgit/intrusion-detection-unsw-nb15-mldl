import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    It down-weights well-classified examples and focuses on hard ones.
    """
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() == 1: # Binary case (logits)
            ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        else: # Multi-class case (logits)
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
