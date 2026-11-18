"""部品ごとの効き方調査の比較図: 条件ごとに、細い線の出力を同じ場所で切り出して並べた PNG を作る。

使い方:
    uv run python -m zsdn.figs_e2 results/e2_components/<hash> results/e1_classic/<hash> docs/figs/e2_montage.png
行 = 正解・ノイズ画像・各条件・BM3D、列 = 線の向き（太さ 1px、黒い線、σ=25）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from zsdn.lineart import LineSpec, render

ANGLES = (0, 22.5, 45, 90, 135)
ROWS = ["clean", "noisy", "base", "pair_horiz", "pair_vert", "pair_single", "pair_flip", "loss_res_only", "loss_l1", "net_deep", "net_linear", "direct", "bm3d"]
        
CROP, SCALE = 48, 3  # 中央 48px を 3 倍に拡大（画素の並びが見えるように最近傍で）


def load_crop(path: Path) -> np.ndarray:
    """PNG を読み、中央 CROP px 四方を切り出す。"""
    a = np.asarray(Image.open(path).convert("L"))
    c = a.shape[0] // 2
    return a[c - CROP // 2:c + CROP // 2, c - CROP // 2:c + CROP // 2]


def main() -> None:
    e2, e1c, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    from zsdn.report_e2 import VARIANTS
    cell, pad, left, top = CROP * SCALE, 6, 230, 34
    W = left + len(ANGLES) * (cell + pad)
    H = top + len(ROWS) * (cell + pad)
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", 18)
    for j, a in enumerate(ANGLES):
        draw.text((left + j * (cell + pad) + cell // 2 - 20, 6), f"{a:g}°", fill=0, font=font)
    for i, r in enumerate(ROWS):
        y = top + i * (cell + pad)
        label = {"clean": "正解", "noisy": "ノイズ画像（σ=25）"}.get(r, VARIANTS.get(r, r))
        draw.text((8, y + cell // 2 - 12), label, fill=0, font=font)
        for j, a in enumerate(ANGLES):
            spec = LineSpec(1, a, 256, 16.0, 1.0, True)
            tag = f"{spec.name}_g25_n0_i0"
            if r == "clean":
                crop = (render(spec)[0][0, 0].numpy() * 255).round().astype(np.uint8)
                c = crop.shape[0] // 2
                crop = crop[c - CROP // 2:c + CROP // 2, c - CROP // 2:c + CROP // 2]
            elif r == "noisy":
                import torch
                n = torch.load(Path("data/noisy/lineart") / f"{spec.name}_gauss25_noclip_s0.pt")
                crop = (n[0, 0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                c = crop.shape[0] // 2
                crop = crop[c - CROP // 2:c + CROP // 2, c - CROP // 2:c + CROP // 2]
            else:
                p = (e1c if r == "bm3d" else e2) / "out" / f"{r}_{tag}.png"
                if not p.exists():
                    continue
                crop = load_crop(p)
            tile = Image.fromarray(crop).resize((cell, cell), Image.NEAREST)
            img.paste(tile, (left + j * (cell + pad), y))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    print(f"→ {out} ({W}×{H})")


if __name__ == "__main__":
    main()
