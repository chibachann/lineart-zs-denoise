"""部品ごとの効き方調査の集計: 部品を 1 つ変えた条件ごとに、線の指標を元の設定（base）と比べる。

使い方:
    uv run python -m zsdn.report_e2 results/e2_components/*/rows.csv \\
        --baselines results/e1_classic/*/rows.csv --mps results/e1_zsn2n/*/rows.csv
出力: 画面に Markdown、results/e2_summary.md、results/e2_rows.csv（全条件をまとめた表。図を作るのに使う）

比べ方:
  - 「斜め」= 0°・90° 以外の 6 方向、「軸」= 0°・90°
  - 元の設定との比較は、同じ線画・同じ σ の組を 1 対として、clDice の差に符号検定（両側）をかける
  - ベースライン（線の壊れ方調査で計算済みの identity / median3 / nlm / bm3d）は、同じ太さ（1〜3px）に絞って並べる
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

from zsdn.report_e1 import load, sign_test

# 表に並べる順番と、文書で使う日本語の名前
VARIANTS = {
    "base": "元の設定",
    "pair_horiz": "間引き: 上下に分ける",
    "pair_vert": "間引き: 左右に分ける",
    "pair_single": "間引き: 平均しない",
    "pair_flip": "間引き: 左右反転して処理",
    "loss_res_only": "損失: 一貫性損失なし",
    "loss_l1": "損失: 絶対誤差",
    "net_wide": "構造: 幅 2 倍",
    "net_deep": "構造: 層 2 倍",
    "net_k5": "構造: 5×5 畳み込み",
    "net_linear": "構造: 活性化なし",
    "direct": "残差学習なし",
    "zsn2n_mps": "元の設定（MPS、線の壊れ方調査）",
    "identity": "何もしない",
    "median3": "3×3 メディアン",
    "nlm": "NLM",
    "bm3d": "BM3D",
}
AXIS = (0.0, 90.0)


def order(m: str) -> int:
    """VARIANTS の並び順（知らない名前は最後）。"""
    return list(VARIANTS).index(m) if m in VARIANTS else len(VARIANTS)


def mean_or_blank(xs: list[float], fmt: str = "{:.3f}") -> str:
    """平均を文字列に。空なら空欄。"""
    return fmt.format(st.mean(xs)) if xs else ""


def summary_table(rows: list[dict]) -> str:
    """手法ごとの主な指標（全体平均と、細い線の斜め・軸の分け）。"""
    cols = [
        ("PSNR", lambda r: True, "psnr"),
        ("clDice 全体", lambda r: True, "cldice"),
        ("1px 軸 clDice", lambda r: r["width"] == 1 and r["angle"] in AXIS, "cldice"),
        ("1px 斜め clDice", lambda r: r["width"] == 1 and r["angle"] not in AXIS, "cldice"),
        ("1.5–2px 斜め clDice", lambda r: r["width"] in (1.5, 2) and r["angle"] not in AXIS, "cldice"),
        ("1.5–2px 斜め 中心の濃さ", lambda r: r["width"] in (1.5, 2) and r["angle"] not in AXIS, "center_ink"),
        ("3px 中心の濃さ", lambda r: r["width"] == 3, "center_ink"),
        ("適合率", lambda r: True, "precision"),
    ]
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out = "| 手法 | " + " | ".join(c[0] for c in cols) + " |\n|---" + "|---" * len(cols) + "|\n"
    for m in sorted(by, key=order):
        out += f"| {VARIANTS.get(m, m)} | " + " | ".join(
            mean_or_blank([r[k] for r in by[m] if f(r)], "{:.2f}" if k == "psnr" else "{:.3f}") for _, f, k in cols) + " |\n"
    return out


def angle_table(rows: list[dict], width: float, metric: str = "cldice") -> str:
    """太さを 1 つに固定して、手法 × 向きの表（σ・濃さで平均）。"""
    cells = defaultdict(list)
    for r in rows:
        if r["width"] == width:
            cells[(r["method"], r["angle"])].append(r[metric])
    ms = sorted({k[0] for k in cells}, key=order)
    angs = sorted({k[1] for k in cells})
    out = "| 手法 | " + " | ".join(f"{a:g}°" for a in angs) + " |\n|---" + "|---" * len(angs) + "|\n"
    for m in ms:
        out += f"| {VARIANTS.get(m, m)} | " + " | ".join(mean_or_blank(cells[(m, a)]) for a in angs) + " |\n"
    return out


def vs_base(rows: list[dict], metric: str = "cldice", filt=lambda r: True) -> str:
    """元の設定との対比較。同じ線画・同じ σ の組で metric の差（条件 − 元）をとり、符号検定。"""
    key = lambda r: (r["image"], r["level"])
    base = {key(r): r[metric] for r in rows if r["method"] == "base" and filt(r)}
    by = defaultdict(list)
    for r in rows:
        if r["method"] != "base" and filt(r) and key(r) in base:
            by[r["method"]].append(r[metric] - base[key(r)])
    out = "| 条件 | 対の数 | 差の平均 | 良くなった | 悪くなった | 符号検定 p |\n|---|---|---|---|---|---|\n"
    for m in sorted(by, key=order):
        pos, neg, p = sign_test(by[m])
        out += f"| {VARIANTS.get(m, m)} | {len(by[m])} | {st.mean(by[m]):+.3f} | {pos} | {neg} | {p:.2g} |\n"
    return out


def asym(rows: list[dict], width: float = 1.0) -> str:
    """45° と 135°（同じ傾きで左右が逆）の差。左右反転で入れ替わるかを見る。"""
    out = "| 手法 | 45° clDice | 135° clDice | 22.5° | 157.5° | 67.5° | 112.5° |\n|---|---|---|---|---|---|---|\n"
    cells = defaultdict(list)
    for r in rows:
        if r["width"] == width:
            cells[(r["method"], r["angle"])].append(r["cldice"])
    for m in sorted({k[0] for k in cells}, key=order):
        out += f"| {VARIANTS.get(m, m)} | " + " | ".join(
            mean_or_blank(cells[(m, a)]) for a in (45, 135, 22.5, 157.5, 67.5, 112.5)) + " |\n"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="部品ごとの効き方調査の集計")
    ap.add_argument("rows", nargs="+", type=Path)
    ap.add_argument("--baselines", nargs="*", type=Path, default=[], help="線の壊れ方調査の古典手法の rows.csv")
    ap.add_argument("--mps", nargs="*", type=Path, default=[], help="線の壊れ方調査の ZS-N2N（MPS）の rows.csv")
    args = ap.parse_args()

    rows = load(args.rows)
    widths = sorted({r["width"] for r in rows})
    extra = [r for r in load(args.baselines) if r["width"] in widths]
    for r in load(args.mps):
        if r["width"] in widths:
            r["method"] = "zsn2n_mps"
            extra.append(r)
    allr = rows + extra

    md = "# 部品ごとの効き方調査の集計\n\n"
    md += f"条件 {len({r['method'] for r in rows})} 個、{len(rows)} 行（比較用のベースライン {len(extra)} 行）。太さ {widths}\n\n"
    md += "## 主な指標（σ・濃さ・向きで平均）\n\n" + summary_table(allr) + "\n"
    md += "## 元の設定との対比較（clDice、全太さ）\n\n" + vs_base(rows) + "\n"
    md += "## 元の設定との対比較（clDice、1px の斜めだけ）\n\n" + vs_base(rows, filt=lambda r: r["width"] == 1 and r["angle"] not in AXIS) + "\n"
    md += "## 元の設定との対比較（中心の濃さ、1.5〜2px の斜めだけ）\n\n" + vs_base(
        rows, "center_ink", lambda r: r["width"] in (1.5, 2) and r["angle"] not in AXIS) + "\n"
    md += "## 元の設定との対比較（PSNR、全太さ）\n\n" + vs_base(rows, "psnr") + "\n"
    for w in widths:
        md += f"## {w:g}px: 手法 × 向き（clDice）\n\n" + angle_table(allr, w) + "\n"
    md += "## 1.5px: 手法 × 向き（中心の濃さ）\n\n" + angle_table(allr, 1.5, "center_ink") + "\n"
    md += "## 向きの左右差（1px、clDice）\n\n" + asym(allr) + "\n"
    print(md)
    root = args.rows[0].parents[2]
    (root / "e2_summary.md").write_text(md, encoding="utf-8")
    keys = ["method", "image", "width", "angle", "contrast", "level", "psnr", "recall", "precision", "cldice",
            "center_ink", "width_err", "width_half"]
    with (root / "e2_rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(allr)
    print(f"→ {root / 'e2_summary.md'}, {root / 'e2_rows.csv'}")


if __name__ == "__main__":
    main()
