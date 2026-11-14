"""画像の読み込みと、ノイズ画像の作成・保存（全手法で同じノイズを使うため）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMG_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg"}


def list_images(folder: str | Path) -> list[Path]:
    """フォルダ内の画像を名前の自然順（1, 2, …, 10）で返す。"""
    files = [p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS]

    def key(p: Path):
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return (int(digits) if digits else 0, p.stem)

    return sorted(files, key=key)


def load_image(path: str | Path, crop: int | None = 256) -> torch.Tensor:
    """RGB 画像を (1, 3, H, W)、値域 [0,1] の float32 で読む。crop を指定すると中央を正方形に切り出す。"""
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if crop:
        h, w = img.shape[:2]
        top, left = (h - crop) // 2, (w - crop) // 2
        img = img[top : top + crop, left : left + crop]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()


def add_noise(clean: torch.Tensor, kind: str, level: float, seed: int) -> torch.Tensor:
    """公式 Colab と同じノイズを、固定シードで加える。

    kind="gauss": level は 0〜255 のスケールの標準偏差。加えたあと [0,1] に切り詰める
    kind="poiss": level は λ。poisson(λ·x)/λ
    """
    g = torch.Generator().manual_seed(seed)
    if kind == "gauss":
        noise = torch.randn(clean.shape, generator=g) * (level / 255.0)
        return torch.clamp(clean + noise, 0, 1)
    if kind == "poiss":
        return torch.poisson(level * clean, generator=g) / level
    raise ValueError(f"未知のノイズの種類: {kind}")


def cached_noisy(clean: torch.Tensor, kind: str, level: float, seed: int, cache_path: Path) -> torch.Tensor:
    """ノイズ画像をキャッシュから読む。なければ作って保存する（同じ条件なら全手法で同じノイズになる）。"""
    if cache_path.exists():
        return torch.load(cache_path)
    noisy = add_noise(clean, kind, level, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noisy, cache_path)
    return noisy
