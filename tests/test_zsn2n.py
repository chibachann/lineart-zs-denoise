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


# ---- 部品ごとの効き方調査のための切り替え ----
import pytest

from zsdn.zsn2n import PAIR_KERNELS, ZSN2NConfig, denoise_zsn2n, zsn2n_loss


@pytest.mark.parametrize("kind", list(PAIR_KERNELS))
def test_every_pair_keeps_flat_images_flat(kind):
    """どの間引き方も、重みの合計が 1（一様な画像は同じ値のまま）。"""
    d1, d2 = pair_downsampler(torch.full((1, 1, 8, 8), 0.3), kind)
    assert torch.allclose(d1, torch.full_like(d1, 0.3)) and torch.allclose(d2, torch.full_like(d2, 0.3))


def test_horizontal_pair_splits_horizontal_line():
    """上下に分ける間引き方では、横線は片方にだけ写る（公式の斜め分けとは逆に、横線が弱点になる）。"""
    d1, d2 = pair_downsampler(_line("h"), "horiz")
    assert (d1.sum() == 0) != (d2.sum() == 0)


def test_default_options_keep_param_count():
    """既定の構造は公式と同じパラメータ数。幅・深さ・畳み込みの大きさを変えると増える。"""
    assert sum(p.numel() for p in ZSN2NNet(3, 48, 2, 3, "lrelu").parameters()) == 22275
    assert sum(p.numel() for p in ZSN2NNet(1, 48, 4).parameters()) > sum(p.numel() for p in ZSN2NNet(1).parameters())


def test_linear_net_is_linear():
    """活性化なしのネットは線形（入力を 2 倍すると、バイアスを除いた出力も 2 倍）。"""
    torch.manual_seed(0)
    net = ZSN2NNet(1, 8, act="none")
    x = torch.rand(1, 1, 8, 8)
    zero = net(torch.zeros_like(x))
    assert torch.allclose(net(2 * x) - zero, 2 * (net(x) - zero), atol=1e-5)


def test_loss_options_change_loss():
    """一貫性損失を外す・絶対誤差にする、で損失の値が変わる（設定が効いている）。"""
    torch.manual_seed(0)
    net, x = ZSN2NNet(1, 8), torch.rand(1, 1, 16, 16)
    base = zsn2n_loss(net, x).item()
    assert zsn2n_loss(net, x, ZSN2NConfig(cons_weight=0.0)).item() < base
    assert zsn2n_loss(net, x, ZSN2NConfig(loss="l1")).item() != base
    assert zsn2n_loss(net, x, ZSN2NConfig(residual=False)).item() != base


def test_flip_returns_image_in_original_orientation():
    """左右反転して処理しても、出力は元の向きに戻っている（学習なしなら入力そのまま）。"""
    x = torch.rand(1, 1, 16, 16)
    out, _ = denoise_zsn2n(x, ZSN2NConfig(iters=0, flip=True), 0, "cpu")
    ref, _ = denoise_zsn2n(x, ZSN2NConfig(iters=0), 0, "cpu")
    assert out.shape == x.shape
    # 学習していない小さなネットの出力はほぼ入力どおりなので、反転が戻っていれば ref とほぼ一致する
    assert (out - ref).abs().mean() < (torch.flip(out, dims=[-1]) - ref).abs().mean()
