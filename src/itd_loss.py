import torch
import torch.nn as nn
import torch.nn.functional as F

class ITDLoss(nn.Module):
    def __init__(self, max_lag=200, eps=1e-8):
        super().__init__()
        self.max_lag = max_lag
        self.eps = eps

    def gcc_phat(self, x, y):
        X = torch.fft.rfft(x)
        Y = torch.fft.rfft(y)
        R = X * torch.conj(Y)
        R = R / (torch.abs(R) + self.eps)

        cc = torch.fft.irfft(R)
        cc = torch.roll(cc, shifts=cc.shape[-1] // 2, dims=-1)

        mid = cc.shape[-1] // 2
        window = cc[..., mid - self.max_lag : mid + self.max_lag]

        itd = torch.argmax(window, dim=-1) - self.max_lag
        return itd.float()

    def forward(self, est, target):
        """
        Supports:
            est,target: (B, S, 2, T) or (B, 2, T)
        """

        # Allow either 3D or 4D input
        if est.dim() == 3:
            est = est.unsqueeze(1)       # (B,2,T) → (B,1,2,T)
            target = target.unsqueeze(1)

        assert est.dim() == 4 and target.dim() == 4, \
            f"Expected 4D or 3D (expanded), got est {est.shape}, target {target.shape}"

        B, S, C, T = est.shape
        assert C == 2, f"Expected 2 channels, got {C}"

        # Flatten batch × stems
        est_L = est[:, :, 0, :].reshape(B * S, T)
        est_R = est[:, :, 1, :].reshape(B * S, T)
        tgt_L = target[:, :, 0, :].reshape(B * S, T)
        tgt_R = target[:, :, 1, :].reshape(B * S, T)

        itd_est = self.gcc_phat(est_L, est_R)
        itd_tgt = self.gcc_phat(tgt_L, tgt_R)

        return F.l1_loss(itd_est, itd_tgt)
    