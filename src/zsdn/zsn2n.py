"""Zero-Shot Noise2Noise（Mansour & Heckel, CVPR 2023）の実装。

公式 Colab ノートブックの設定に合わせている:
  - ネット: 3×3 畳み込み 2 層 + 1×1 畳み込み、中間 48 チャンネル、LeakyReLU(0.2)。約 2.2 万パラメータ
  - 損失: 残差損失（対称）+ 一貫性損失
  - 学習: Adam, lr=1e-3, 2000 反復, 1500 反復目で lr を 0.5 倍
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZSN2NNet(nn.Module):
    """ノイズ成分（残差）を推定する小さな畳み込みネット。出力をノイズ画像から引くと除去結果になる。"""

    def __init__(self, n_chan: int, chan_embed: int = 48):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv3(x)


def pair_downsampler(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2×2 のマスごとに、右上＋左下の平均（D1）と左上＋右下の平均（D2）の 2 枚に間引く。

    img: (B, C, H, W)。H, W は偶数を想定。出力はそれぞれ (B, C, H/2, W/2)。
    """
    c = img.shape[1]
    k1 = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    k2 = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    out1 = F.conv2d(img, k1, stride=2, groups=c)
    out2 = F.conv2d(img, k2, stride=2, groups=c)
    return out1, out2


def zsn2n_loss(model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """ZS-N2N の損失 = 残差損失 + 一貫性損失（どちらも 2 枚について対称に平均）。"""
    noisy1, noisy2 = pair_downsampler(noisy)
    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)
    # 残差損失: 片方を除去した結果が、もう片方のノイズ画像に近づくように
    loss_res = 0.5 * (F.mse_loss(noisy1, pred2) + F.mse_loss(noisy2, pred1))
    # 一貫性損失: 「除去してから間引く」と「間引いてから除去する」が一致するように
    denoised1, denoised2 = pair_downsampler(noisy - model(noisy))
    loss_cons = 0.5 * (F.mse_loss(pred1, denoised1) + F.mse_loss(pred2, denoised2))
    return loss_res + loss_cons


@dataclass
class ZSN2NConfig:
    """学習の設定。既定値は公式 Colab と同じ。"""

    chan_embed: int = 48
    iters: int = 2000
    lr: float = 1e-3
    step_size: int = 1500
    gamma: float = 0.5


def denoise_zsn2n(noisy: torch.Tensor, cfg: ZSN2NConfig, init_seed: int, device: str) -> tuple[torch.Tensor, float]:
    """1 枚のノイズ画像だけでネットを学習し、除去結果と所要秒数を返す。

    noisy: (1, C, H, W)、値域 [0,1] 付近。戻り値の画像は [0,1] に切り詰めて CPU に置く。
    """
    torch.manual_seed(init_seed)
    noisy = noisy.to(device)
    model = ZSN2NNet(noisy.shape[1], cfg.chan_embed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        loss = zsn2n_loss(model, noisy)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        out = torch.clamp(noisy - model(noisy), 0, 1)
    if device == "mps":
        torch.mps.synchronize()
    return out.cpu(), time.perf_counter() - t0
