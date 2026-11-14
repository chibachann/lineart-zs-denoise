"""結果の集計: rows.csv を条件ごとに平均し、論文の数値と並べて合否を判定する。

使い方:
    uv run python -m zsdn.report results/e0_gauss/<hash>/rows.csv [他の rows.csv ...]
出力: 画面に Markdown の表、各 rows.csv の隣に summary.md と summary.csv
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# ZS-N2N 論文（arXiv 2303.11253 v3）Table 1 の ZS-N2N の PSNR（dB）
PAPER = {
    ("kodak24", "gauss", 10.0): 33.69, ("kodak24", "gauss", 25.0): 29.07, ("kodak24", "gauss", 50.0): 24.81,
    ("mcmaster18", "gauss", 10.0): 34.21, ("mcmaster18", "gauss", 25.0): 28.80, ("mcmaster18", "gauss", 50.0): 24.02,
    ("kodak24", "poiss", 50.0): 29.45, ("kodak24", "poiss", 25.0): 27.52, ("kodak24", "poiss", 10.0): 24.92,
    ("mcmaster18", "poiss", 50.0): 30.36, ("mcmaster18", "poiss", 25.0): 28.41, ("mcmaster18", "poiss", 10.0): 25.75,
}
# 合格ライン（計画ページ G0）: Kodak24 σ=25 は ±0.3 dB、それ以外は ±0.5 dB
def tolerance(key: tuple) -> float:
    return 0.3 if key == ("kodak24", "gauss", 25.0) else 0.5


def summarize(rows_path: Path) -> list[dict]:
    """条件（データセット・ノイズ・強さ）ごとに、画像平均の PSNR をシード間で平均する。"""
    by = defaultdict(lambda: defaultdict(list))  # 条件 → シード → 画像ごとの PSNR
    secs, ssims = defaultdict(list), defaultdict(list)
    with rows_path.open() as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["noise"], float(r["level"]))
            by[key][(r["noise_seed"], r["init_seed"])].append(float(r["psnr"]))
            secs[key].append(float(r["seconds"]))
            ssims[key].append(float(r["ssim"]))
    out = []
    for key in sorted(by, key=lambda k: (k[0], k[1], -k[2] if k[1] == "poiss" else k[2])):
        seed_means = [st.mean(v) for v in by[key].values()]
        mean = st.mean(seed_means)
        paper = PAPER.get(key)
        n_img = len(next(iter(by[key].values())))
        out.append({
            "dataset": key[0], "noise": key[1], "level": key[2], "n_images": n_img, "n_seeds": len(seed_means),
            "psnr": round(mean, 2), "psnr_seed_std": round(st.pstdev(seed_means), 3) if len(seed_means) > 1 else "",
            "ssim": round(st.mean(ssims[key]), 4), "paper": paper if paper is not None else "",
            "diff": round(mean - paper, 2) if paper is not None else "",
            "pass": ("OK" if abs(mean - paper) <= tolerance(key) else "NG") if paper is not None else "",
            "sec_per_image": round(st.mean(secs[key]), 1),
        })
    return out


def to_markdown(rows: list[dict]) -> str:
    """集計結果を Markdown の表にする。"""
    head = "| データ | ノイズ | 強さ | 枚数 | シード数 | PSNR（自分） | 論文 | 差 | 判定 | SSIM | 秒/枚 |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {r['dataset']} | {r['noise']} | {r['level']:g} | {r['n_images']} | {r['n_seeds']} | {r['psnr']:.2f}"
        f"{' ±' + str(r['psnr_seed_std']) if r['psnr_seed_std'] != '' else ''} | {r['paper']} | "
        f"{('+' if isinstance(r['diff'], float) and r['diff'] > 0 else '') + str(r['diff'])} | {r['pass']} | {r['ssim']} | {r['sec_per_image']} |\n"
        for r in rows)
    return head + body


def main() -> None:
    for p in map(Path, sys.argv[1:]):
        rows = summarize(p)
        md = to_markdown(rows)
        print(f"## {p}\n\n{md}")
        (p.parent / "summary.md").write_text(md, encoding="utf-8")
        with (p.parent / "summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
