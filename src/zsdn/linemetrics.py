"""線画の評価指標（E1）。PSNR だけでは「線が消えたか」が分からないので、線そのものを見る指標を用意する。

すべて「墨の濃さ」d = 1 − 画素値 で考える（紙 = 0、墨 = 1）。画像の端から margin px は評価に使わない。

- recall（再現率）: 正解の中心線の画素のうち、出力で 1px 以内に墨が残っている割合。線の消失・途切れを見る
- precision（適合率）: 出力で墨と判定した画素のうち、本当の線から（太さ/2 + 1）px 以内にある割合。ゴミ・にじみを見る
- f1: 上の 2 つの調和平均
- cldice: 中心線（細線化）どうしの一致度（Shit et al., CVPR 2021）。線のつながりを見る
- width_est / width_err: 線の周りの墨の量から推定した太さ（px）と、正解との差。線が薄く・細くなったかを見る
- center_ink: 正解の中心線の位置での墨の濃さを、正解の画像の同じ値で割ったもの（1 = 正解どおり）。線がぼけて薄まったかを見る
  （斜めの細い線は、正解でもアンチエイリアスで中心が 1 より薄いため、比で表す）
  （ぼけても墨の量は保たれるので、width_est だけでは分からない）
- width_half: 墨の濃さが半分以上の画素の面積から求めた「見た目の太さ」（px）。ぼけると太く、薄まると細く・0 になる
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

INK_THRESHOLD = 0.5  # 墨の濃さがこれ以上の画素を「墨」とみなす


def _darkness(img, contrast: float = 1.0) -> np.ndarray:
    """(1,1,H,W) のテンソルまたは (H,W) の配列を、墨の濃さ (H,W) にする。

    墨の色が灰色（contrast < 1）のときも、墨そのもの = 1 になるよう contrast で割る。"""
    a = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    a = a.reshape(a.shape[-2], a.shape[-1])
    return (1.0 - np.clip(a, 0.0, 1.0)) / contrast


def line_metrics(out, clean, dist: np.ndarray, width: float, contrast: float = 1.0, margin: int = 8) -> dict:
    """出力画像 out を、正解 clean と中心線までの距離 dist（lineart.render の戻り値）で評価する。"""
    d_out, d_gt = _darkness(out, contrast), _darkness(clean, contrast)
    h, w = d_out.shape
    roi = np.zeros((h, w), bool)
    roi[margin:h - margin, margin:w - margin] = True

    # 正解の線の領域（中心から太さ/2 以内。1px 未満の線でも 1 画素幅は確保）と、その中心線
    v_gt = dist <= max(width / 2, 0.5)
    s_gt = skeletonize(v_gt) & roi
    v_gt &= roi
    v_out = (d_out >= INK_THRESHOLD) & roi
    s_out = skeletonize(v_out)

    near_out = ndimage.binary_dilation(v_out, structure=np.ones((3, 3), bool))  # 1px の位置ずれは許す
    recall = (s_gt & near_out).sum() / max(s_gt.sum(), 1)
    precision = (v_out & (dist <= width / 2 + 1)).sum() / max(v_out.sum(), 1) if v_out.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # clDice: 出力の中心線が正解の線の中にあるか（tprec）× 正解の中心線が出力の線の中にあるか（tsens）
    tprec = (s_out & v_gt).sum() / max(s_out.sum(), 1) if s_out.any() else 0.0
    tsens = (s_gt & v_out).sum() / max(s_gt.sum(), 1)
    cldice = 2 * tprec * tsens / (tprec + tsens) if tprec + tsens > 0 else 0.0

    # 太さの推定: 線の周り（太さ/2 + 2 px 以内）の墨の量を、正解の墨の量と比べる。
    # 紙の部分に残ったノイズの底上げ分は、線から十分離れた画素の中央値を引いて除く
    band = (dist <= width / 2 + 2) & roi
    far = (dist >= width / 2 + 3) & roi
    bg = float(np.median(d_out[far])) if far.any() else 0.0
    ratio = float((d_out[band] - bg).sum() / max(d_gt[band].sum(), 1e-8))
    width_est = ratio * width
    center_ink = float(d_out[s_gt].mean() / max(d_gt[s_gt].mean(), 1e-8)) if s_gt.any() else 0.0
    # 見た目の太さ: 線の近く（太さ/2 + 3 px 以内）で墨と判定した面積 ÷ 正解の線の面積 × 太さ
    near = (dist <= width / 2 + 3) & roi
    gt_area = (d_gt[near] >= INK_THRESHOLD).sum()
    width_half = float((d_out[near] >= INK_THRESHOLD).sum() / max(gt_area, 1)) * width
    return {
        "recall": round(float(recall), 4), "precision": round(float(precision), 4), "f1": round(float(f1), 4),
        "cldice": round(float(cldice), 4), "ink_ratio": round(ratio, 4),
        "width_est": round(width_est, 3), "width_err": round(width_est - width, 3),
        "center_ink": round(center_ink, 4), "width_half": round(width_half, 3),
    }
