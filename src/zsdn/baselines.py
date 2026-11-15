"""比較用の古典的なノイズ除去（学習なし）。E1 で ZS-N2N と並べる。

BM3D と NLM はノイズの強さ σ を知っている前提（オラクル）で動かす。ZS-N2N は σ を使わないので、
この 2 つには有利な条件であることに注意。
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy import ndimage


def _to_np(x: torch.Tensor) -> np.ndarray:
    """(1,C,H,W) のテンソル → (H,W) または (H,W,C) の numpy 配列。"""
    a = x[0].detach().cpu().numpy()
    return a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)


def _to_t(a: np.ndarray) -> torch.Tensor:
    """_to_np の逆。値は [0,1] に切り詰める。"""
    a = np.clip(a, 0, 1).astype(np.float32)
    a = a[None] if a.ndim == 2 else a.transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(a))[None]


def denoise_classic(noisy: torch.Tensor, name: str, sigma: float) -> tuple[torch.Tensor, float]:
    """古典手法でノイズを除く。sigma は 0〜255 スケールの標準偏差。戻り値は (出力, 秒)。"""
    a = _to_np(noisy)
    s = sigma / 255.0
    t0 = time.perf_counter()
    if name == "median3":  # 3×3 のメディアンフィルタ
        out = ndimage.median_filter(a, size=3 if a.ndim == 2 else (3, 3, 1))
    elif name == "nlm":  # Non-local means（skimage。h は σ の 0.8 倍、skimage の例に近い設定）
        from skimage.restoration import denoise_nl_means
        out = denoise_nl_means(a, h=0.8 * s, sigma=s, patch_size=5, patch_distance=6, fast_mode=True,
                               channel_axis=None if a.ndim == 2 else -1)
    elif name == "bm3d":  # BM3D（Tampere 大学の公式実装、非商用ライセンス）
        import bm3d
        out = bm3d.bm3d(a, sigma_psd=s)
    elif name == "identity":  # 何もしない（ノイズ画像そのもの）
        out = a
    else:
        raise ValueError(f"未知の手法: {name}")
    return _to_t(np.asarray(out)), time.perf_counter() - t0
