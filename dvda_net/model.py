"""DVDA-Net model: dual views, shared encoder, attention branches, classifier + domain head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

from .config import NUM_CLASSES, DEVICE


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class NERM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if not self.training:
            return x
        s_input = torch.mean(torch.abs(x))
        s_noise = torch.clamp(0.1 * s_input + 1e-8, max=0.05)
        noise = torch.randn_like(x) * s_noise
        return x + noise


class WTRM(nn.Module):
    def __init__(self, wavelet='db1', mask_prob=0.05):
        super().__init__()
        self.wavelet = wavelet
        self.mask_prob = mask_prob

    def forward(self, x):
        if not self.training:
            return x

        B, D = x.shape
        x_np = x.detach().cpu().numpy()
        x_wtrm = np.zeros_like(x_np)

        for i in range(B):
            signal = x_np[i]
            try:
                cA, cD = pywt.dwt(signal, self.wavelet, mode='symmetric')

                if np.random.rand() < self.mask_prob * 0.5:
                    cA = cA * np.random.uniform(0.9, 1.1)

                mask_suppress = np.random.rand(len(cD)) < self.mask_prob
                cD[mask_suppress] *= 0.5

                reconstructed = pywt.idwt(cA, cD, self.wavelet, mode='symmetric')

                if len(reconstructed) > D:
                    reconstructed = reconstructed[:D]
                elif len(reconstructed) < D:
                    reconstructed = np.pad(reconstructed, (0, D - len(reconstructed)))

                x_wtrm[i] = reconstructed
            except Exception:
                x_wtrm[i] = signal

        return torch.tensor(x_wtrm, dtype=x.dtype, device=x.device)


class SharedFeatureExtractor(nn.Module):
    def __init__(self, in_dim, model_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, model_dim)
        )

    def forward(self, x):
        return self.net(x)


class ViewEncoder(nn.Module):
    def __init__(self, model_dim=128, num_heads=4, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.attention = nn.MultiheadAttention(model_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.alpha * attn_out)
        x = self.norm2(x + self.alpha * self.ffn(x))
        return x.squeeze(1)


class DVDANet(nn.Module):
    def __init__(self, in_dim, use_dual_view=True, model_dim=128):
        super().__init__()
        self.use_dual_view = use_dual_view

        self.nerm = NERM()
        self.wtrm = WTRM()
        self.shared_encoder = SharedFeatureExtractor(in_dim, model_dim)

        if use_dual_view:
            self.nerm_encoder = ViewEncoder(model_dim)
            self.wtrm_encoder = ViewEncoder(model_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES)
        )

        self.domain_disc = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=0.0, return_z=False):
        if self.use_dual_view:
            x_nerm = self.nerm(x)
            x_wtrm = self.wtrm(x)

            z_nerm = self.shared_encoder(x_nerm)
            z_wtrm = self.shared_encoder(x_wtrm)

            z_nerm = self.nerm_encoder(z_nerm)
            z_wtrm = self.wtrm_encoder(z_wtrm)

            z_fused = (z_nerm + z_wtrm) / 2.0
            z_main = z_fused
        else:
            z_main = self.shared_encoder(x)

        y = self.classifier(z_main)

        z_rev = GradReverse.apply(z_main, alpha)
        d = self.domain_disc(z_rev)

        if return_z:
            if self.use_dual_view:
                return y, d, z_main, z_nerm, z_wtrm
            return y, d, z_main
        return y, d

