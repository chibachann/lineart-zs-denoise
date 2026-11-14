"""画質の指標。PSNR は公式 Colab と同じ式（全画素・全チャンネルの平均二乗誤差）で計算する。"""
from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity


def psnr(clean: torch.Tensor, pred: torch.Tensor) -> float:
    """値域 [0,1] の PSNR（dB）。"""
    mse = torch.mean((clean - pred) ** 2).item()
    return 10 * math.log10(1.0 / mse)


def ssim(clean: torch.Tensor, pred: torch.Tensor) -> float:
    """RGB の SSIM（skimage、data_range=1）。"""
    a = clean.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    b = pred.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
