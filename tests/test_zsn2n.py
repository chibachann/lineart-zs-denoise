"""ZS-N2N 実装の基本的な性質のテスト。"""
import torch

from zsdn.data import add_noise
from zsdn.zsn2n import ZSN2NNet, pair_downsampler


def test_param_count_matches_paper():
    """公式 Colab と同じ構成なら約 2.2 万パラメータ（論文の Table 4 は 22k）。"""
    assert sum(p.numel() for p in ZSN2NNet(3).parameters()) == 22275


def _line(kind: str, n: int = 8) -> torch.Tensor:
    """1 ピクセル幅の線を 1 本だけ描いた (1,1,n,n) の画像。"""
    img = torch.zeros(1, 1, n, n)
    for i in range(n):
        if kind == "h":
            img[0, 0, 3, i] = 1
        elif kind == "v":
            img[0, 0, i, 4] = 1
        elif kind == "d":
            img[0, 0, i, i] = 1
        elif kind == "a":
            img[0, 0, i, n - 1 - i] = 1
    return img


def test_axis_aligned_lines_appear_equally_in_both_halves():
    """横線・縦線は、間引いた 2 枚に同じ濃さで写る（仮説 H1 の対照）。"""
    for kind in ("h", "v"):
        d1, d2 = pair_downsampler(_line(kind))
        assert torch.equal(d1, d2)


def test_diagonal_lines_appear_in_only_one_half():
    """斜め 45° の 1px 線は、片方にだけ写り、もう片方は空になる（仮説 H1 の根拠）。"""
    for kind in ("d", "a"):
        d1, d2 = pair_downsampler(_line(kind))
        assert (d1.sum() == 0) != (d2.sum() == 0)
        assert max(d1.sum().item(), d2.sum().item()) == 4.0


def test_noise_is_reproducible():
    """同じシードなら同じノイズ（全手法で同じノイズ画像を使うための前提）。"""
    x = torch.rand(1, 3, 16, 16)
    assert torch.equal(add_noise(x, "gauss", 25, 0), add_noise(x, "gauss", 25, 0))
    assert not torch.equal(add_noise(x, "gauss", 25, 0), add_noise(x, "gauss", 25, 1))
