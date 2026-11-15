"""合成線画ベンチマーク（E1）の画像を作る。

1 枚の画像には、同じ太さ・同じ向きの平行な直線を等間隔に何本も描く。
白い紙（1.0）に黒い墨（0.0）で、画素の中を 8×8 に分けて数える方法でアンチエイリアスする。
線の中心線までの距離（dist）も一緒に返し、評価の正解に使う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LineSpec:
    """1 枚分の設定。width は線の太さ（px）、angle は向き（度、0 = 横線、反時計回り）。"""
    width: float
    angle: float
    size: int = 256
    spacing: float = 16.0  # 平行な線どうしの間隔（線に垂直な方向、px）
    contrast: float = 1.0  # 墨の濃さ（1.0 = 真っ黒、0.5 = 鉛筆のような灰色）
    jitter: bool = False   # True なら線の間隔を 12〜24px の整数でランダムにする（等間隔だと BM3D に有利すぎるため）

    @property
    def name(self) -> str:
        return f"w{self.width:g}_a{self.angle:g}_c{self.contrast:g}".replace(".", "p")

    def offsets(self) -> np.ndarray | None:
        """jitter=True のときの各線の位置（中央の線を 0 とした、線に垂直な方向の座標）。向きだけから決まる固定の乱数で作る。

        太さ・濃さが違っても同じ向きなら同じ並びになるので、太さ・濃さの効果をそのまま比べられる。"""
        if not self.jitter:
            return None
        rng = np.random.default_rng(hash_name(f"a{self.angle:g}"))
        half = self.size  # 画像の対角線の半分より十分広く並べる
        pos = [0.0]
        while pos[-1] < half:
            pos.append(pos[-1] + rng.integers(12, 25))
        neg = [0.0]
        while neg[-1] > -half:
            neg.append(neg[-1] - rng.integers(12, 25))
        return np.array(sorted(set(neg + pos)), dtype=np.float64)


def hash_name(name: str) -> int:
    """Python の hash() は実行ごとに変わるので、名前から再現できる整数を作る。"""
    import zlib
    return zlib.crc32(name.encode())


def _signed_offset(spec: LineSpec, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """点 (x, y) から最も近い中心線までの符号付き距離（線に垂直な方向）。

    線は画像中央の画素の中心 (size/2 + 0.5) を通るように置く。
    こうすると横線・縦線の 1px 線は、ちょうど 1 列の画素に収まる（ペンで描いた線に近い）。
    """
    th = math.radians(spec.angle)
    nx, ny = -math.sin(th), -math.cos(th)  # 線に垂直な単位ベクトル（画像の y は下向きなので符号に注意）
    c = spec.size / 2 + 0.5
    t = (x - c) * nx + (y - c) * ny
    offs = spec.offsets()
    if offs is not None:  # 間隔がばらばらのとき: 一番近い線までの距離
        k = np.clip(np.searchsorted(offs, t), 1, len(offs) - 1)
        lo, hi = t - offs[k - 1], t - offs[k]
        return np.where(np.abs(lo) < np.abs(hi), lo, hi)
    s = spec.spacing
    return (t + s / 2) % s - s / 2


def render(spec: LineSpec, ss: int = 8) -> tuple[torch.Tensor, np.ndarray]:
    """線画を描く。戻り値: (画像 (1,1,H,W)、値域 [0,1]、白=1) と、各画素の中心から中心線までの距離 (H,W)。"""
    n = spec.size
    # 画素 (i, j) の中を ss×ss の点に分け、線の内側（距離 ≤ 太さ/2）に入った点の割合を墨の濃さにする
    sub = (np.arange(ss) + 0.5) / ss
    jj = (np.arange(n)[:, None] + sub[None, :]).reshape(-1)  # x 方向の副画素の座標
    ii = jj.copy()
    X, Y = np.meshgrid(jj, ii)
    inside = np.abs(_signed_offset(spec, X, Y)) <= spec.width / 2
    cover = inside.reshape(n, ss, n, ss).mean(axis=(1, 3))
    img = 1.0 - spec.contrast * cover
    cx = np.arange(n) + 0.5
    Xc, Yc = np.meshgrid(cx, cx)
    dist = np.abs(_signed_offset(spec, Xc, Yc))
    return torch.from_numpy(img.astype(np.float32))[None, None], dist.astype(np.float32)


def e1_specs(widths=(1, 1.5, 2, 3, 4), n_angles: int = 8, contrasts=(1.0,), size: int = 256,
             jitter: bool = True) -> list[LineSpec]:
    """E1 の全条件（太さ × 向き × 墨の濃さ）。向きは 0° から 180° を n_angles 等分（8 なら 22.5° 刻み）。"""
    return [LineSpec(float(w), 180.0 * k / n_angles, size, 16.0, float(c), jitter)
            for c in contrasts for w in widths for k in range(n_angles)]
