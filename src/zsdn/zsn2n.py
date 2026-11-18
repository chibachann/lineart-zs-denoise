"""Zero-Shot Noise2Noise（Mansour & Heckel, CVPR 2023）の実装。

公式 Colab ノートブックの設定に合わせている:
  - ネット: 3×3 畳み込み 2 層 + 1×1 畳み込み、中間 48 チャンネル、LeakyReLU(0.2)。約 2.2 万パラメータ
  - 損失: 残差損失（対称）+ 一貫性損失
  - 学習: Adam, lr=1e-3, 2000 反復, 1500 反復目で lr を 0.5 倍

部品ごとの効き方調査のため、間引きペアの作り方・損失関数・ネットワーク構造・残差学習を
ZSN2NConfig で 1 つずつ切り替えられるようにしてある。既定値のままなら元の実装と同じ結果になる。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZSN2NNet(nn.Module):
    """ノイズ成分（残差）を推定する小さな畳み込みネット。出力をノイズ画像から引くと除去結果になる。

    既定（depth=2, kernel=3, act="lrelu"）は公式どおり 3×3 畳み込み 2 層 + 1×1 畳み込み。
    depth は 3×3（kernel×kernel）の層の数、act="none" で活性化なし（ネット全体が線形フィルタになる）。
    """

    def __init__(self, n_chan: int, chan_embed: int = 48, depth: int = 2, kernel: int = 3, act: str = "lrelu"):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if act == "lrelu" else nn.Identity()
        # 層を作る順番は元の実装（conv1 → conv2 → conv3）と同じにして、同じシードなら同じ初期値になるようにする
        chans = [n_chan] + [chan_embed] * depth
        self.convs = nn.ModuleList(nn.Conv2d(a, b, kernel, padding=kernel // 2) for a, b in zip(chans, chans[1:]))
        self.last = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = self.act(conv(x))
        return self.last(x)


# 間引きペアの作り方: 2×2 のマスの中で、D1 と D2 にそれぞれどの画素をどれだけ混ぜるか
PAIR_KERNELS = {
    "diag": ([[0.0, 0.5], [0.5, 0.0]], [[0.5, 0.0], [0.0, 0.5]]),    # 公式: 右上＋左下 / 左上＋右下
    "horiz": ([[0.5, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.5, 0.5]]),   # 上の 2 画素 / 下の 2 画素（上下に 1px ずれ）
    "vert": ([[0.5, 0.0], [0.5, 0.0]], [[0.0, 0.5], [0.0, 0.5]]),    # 左の 2 画素 / 右の 2 画素（左右に 1px ずれ）
    "single": ([[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]),  # 平均しない: 左上 1 画素 / 右下 1 画素
}


def pair_downsampler(img: torch.Tensor, kind: str = "diag") -> tuple[torch.Tensor, torch.Tensor]:
    """2×2 のマスごとに画素を混ぜて、半分の大きさの画像 2 枚（D1, D2）に間引く。

    既定（diag）は公式どおり、右上＋左下の平均（D1）と左上＋右下の平均（D2）。
    img: (B, C, H, W)。H, W は偶数を想定。出力はそれぞれ (B, C, H/2, W/2)。
    """
    c = img.shape[1]
    a, b = PAIR_KERNELS[kind]
    k1 = torch.tensor([[a]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    k2 = torch.tensor([[b]], device=img.device, dtype=img.dtype).repeat(c, 1, 1, 1)
    out1 = F.conv2d(img, k1, stride=2, groups=c)
    out2 = F.conv2d(img, k2, stride=2, groups=c)
    return out1, out2


def zsn2n_loss(model: nn.Module, noisy: torch.Tensor, cfg: "ZSN2NConfig | None" = None) -> torch.Tensor:
    """ZS-N2N の損失 = 残差損失 + 一貫性損失（どちらも 2 枚について対称に平均）。

    cfg で、間引き方（pair）・誤差の測り方（loss: mse / l1）・一貫性損失の重み（cons_weight）・
    残差学習の有無（residual）を切り替えられる。cfg=None なら公式どおり。
    """
    cfg = cfg or ZSN2NConfig()
    err = F.mse_loss if cfg.loss == "mse" else F.l1_loss
    noisy1, noisy2 = pair_downsampler(noisy, cfg.pair)
    pred1 = apply_model(model, noisy1, cfg.residual)
    pred2 = apply_model(model, noisy2, cfg.residual)
    # 残差損失: 片方を除去した結果が、もう片方のノイズ画像に近づくように
    loss_res = 0.5 * (err(noisy1, pred2) + err(noisy2, pred1))
    if cfg.cons_weight == 0:
        return loss_res
    # 一貫性損失: 「除去してから間引く」と「間引いてから除去する」が一致するように
    denoised1, denoised2 = pair_downsampler(apply_model(model, noisy, cfg.residual), cfg.pair)
    loss_cons = 0.5 * (err(pred1, denoised1) + err(pred2, denoised2))
    return loss_res + cfg.cons_weight * loss_cons


def apply_model(model: nn.Module, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
    """除去結果を返す。residual=True ならネットはノイズを推定し、それを引く（公式）。False ならネットが直接きれいな画像を出す。"""
    return x - model(x) if residual else model(x)


@dataclass
class ZSN2NConfig:
    """学習の設定。既定値は公式 Colab と同じ。下の 4 グループは部品ごとの効き方調査で 1 つずつ変える。"""

    chan_embed: int = 48
    iters: int = 2000
    lr: float = 1e-3
    step_size: int = 1500
    gamma: float = 0.5
    # 間引きペアの作り方（PAIR_KERNELS のキー）。flip=True なら左右反転した画像を処理して元に戻す（向きの非対称の診断用）
    pair: str = "diag"
    flip: bool = False
    # 損失関数: 誤差の測り方と、一貫性損失の重み（0 なら残差損失だけ）
    loss: str = "mse"
    cons_weight: float = 1.0
    # ネットワーク構造: 3×3 層の数・畳み込みの大きさ・活性化（chan_embed は上）
    depth: int = 2
    kernel: int = 3
    act: str = "lrelu"
    # 残差学習: True ならノイズを推定して引く（公式）、False ならきれいな画像を直接出す
    residual: bool = True


def denoise_zsn2n(noisy: torch.Tensor, cfg: ZSN2NConfig, init_seed: int, device: str) -> tuple[torch.Tensor, float]:
    """1 枚のノイズ画像だけでネットを学習し、除去結果と所要秒数を返す。

    noisy: (1, C, H, W)、値域 [0,1] 付近。戻り値の画像は [0,1] に切り詰めて CPU に置く。
    """
    torch.manual_seed(init_seed)
    noisy = noisy.to(device)
    if cfg.flip:
        noisy = torch.flip(noisy, dims=[-1])
    model = ZSN2NNet(noisy.shape[1], cfg.chan_embed, cfg.depth, cfg.kernel, cfg.act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.gamma)
    t0 = time.perf_counter()
    for _ in range(cfg.iters):
        loss = zsn2n_loss(model, noisy, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        out = torch.clamp(apply_model(model, noisy, cfg.residual), 0, 1)
        if cfg.flip:
            out = torch.flip(out, dims=[-1])
    if device == "mps":
        torch.mps.synchronize()
    return out.cpu(), time.perf_counter() - t0
