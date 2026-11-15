"""合成線画ベンチマークと線の評価指標のテスト。"""
import pytest

from zsdn.lineart import LineSpec, e1_specs, render
from zsdn.linemetrics import line_metrics


@pytest.mark.parametrize("spec", e1_specs(contrasts=(1.0, 0.5)))
def test_clean_image_scores_perfect(spec):
    """正解の画像そのものを評価すると、線の指標はほぼ満点、太さの誤差はほぼ 0 になる。"""
    img, dist = render(spec)
    m = line_metrics(img, img, dist, spec.width, spec.contrast)
    assert m["recall"] >= 0.99 and m["precision"] >= 0.99 and m["cldice"] >= 0.99
    assert abs(m["width_err"]) < 0.02


def test_blank_output_scores_zero():
    """真っ白な出力（線が全部消えた）なら再現率も clDice も 0。"""
    img, dist = render(LineSpec(1, 45, jitter=True))
    m = line_metrics(img * 0 + 1, img, dist, 1.0)
    assert m["recall"] == 0 and m["cldice"] == 0


def test_ink_amount_matches_width():
    """線の間隔が 16px なら、墨の面積の割合は 太さ/16。"""
    for w in (1, 2, 4):
        img, _ = render(LineSpec(w, 30))
        assert abs((1 - img.mean().item()) - w / 16) < 0.005


def test_spacing_layout_depends_only_on_angle():
    """ランダムな線の並びは向きだけで決まり、太さ・濃さが違っても同じ。"""
    assert (LineSpec(1, 45, contrast=1.0, jitter=True).offsets() == LineSpec(3, 45, contrast=0.5, jitter=True).offsets()).all()


def test_blur_lowers_center_ink_but_keeps_ink_amount():
    """線をぼかすと、墨の量（width_est）は保たれるが、中心の濃さ（center_ink）は下がる。"""
    from scipy import ndimage
    spec = LineSpec(1, 45, jitter=True)
    img, dist = render(spec)
    blurred = 1 - ndimage.gaussian_filter(1 - img[0, 0].numpy(), 1.0)
    m = line_metrics(blurred, img, dist, 1.0)
    assert abs(m["width_err"]) < 0.1 and m["center_ink"] < 0.6
