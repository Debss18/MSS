import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.itd_loss import ITDLoss

class CompositeLoss(nn.Module):
    """
    For 4-stem MSS:
      est, target: (B, S=4, C=2, T)
    Loss = L1(est, target) + lambda_itd * ITD(est, target)
    """
    def __init__(self, lambda_itd=0.1):
        super().__init__()
        self.lambda_itd = lambda_itd
        self.recon = nn.L1Loss()
        self.itd = ITDLoss()

    def forward(self, est, target):
        # est, target are expected as (B, S, C, T)
        assert est.shape == target.shape, \
            f"Shape mismatch: est {est.shape}, target {target.shape}"

        loss_recon = self.recon(est, target)
        loss_itd = self.itd(est, target)

        return loss_recon + self.lambda_itd * loss_itd
    