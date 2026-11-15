"""E1 の集計: 線がどの向き・どの太さで消えるかを、手法ごとに表にする。

使い方:
    uv run python -m zsdn.report_e1 results/e1_classic/*/rows.csv results/e1_zsn2n/*/rows.csv
出力: 画面に Markdown、最初の rows.csv の親の親（results/）に e1_summary.md

仮説 H1 の判定: ZS-N2N の 1px 線で「斜め（45°・135°）」と「軸に沿った線（0°・90°）」の clDice を比べる。
同じ太さ・濃さ・σ の組を 1 対とみなし、対ごとの差に符号検定（両側）をかける。
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

METRICS = ["psnr", "recall", "precision", "cldice", "width_err", "center_ink", "width_half"]
METHOD_ORDER = ["identity", "median3", "nlm", "bm3d", "zsn2n"]


def rescore(rows_path: Path) -> Path:
    """保存した出力画像（out/*.png）から、線の指標を今の linemetrics で計算し直す。

    指標を後から足したときに、重い計算（ZS-N2N）をやり直さずに済むようにするため。
    結果は rows.csv の隣の rows_rescored.csv に書く（rows.csv より新しければ作り直さない）。
    """
    import numpy as np
    import yaml
    from PIL import Image

    from zsdn.lineart import LineSpec, render
    from zsdn.linemetrics import line_metrics

    out = rows_path.with_name("rows_rescored.csv")
    if out.exists() and out.stat().st_mtime > rows_path.stat().st_mtime:
        return out
    bench = yaml.safe_load((rows_path.parent / "config.yaml").read_text(encoding="utf-8"))["bench"]
    with rows_path.open() as f:
        rows = list(csv.DictReader(f))
    gt_cache: dict = {}
    for r in rows:
        spec = LineSpec(float(r["width"]), float(r["angle"]), bench.get("size", 256), 16.0,
                        float(r["contrast"]), bench.get("jitter", True))
        if spec.name not in gt_cache:
            gt_cache[spec.name] = render(spec)
        clean, dist = gt_cache[spec.name]
        tag = f"{spec.name}_g{float(r['level']):g}_n{r['noise_seed']}_i{r['init_seed']}"
        img = np.asarray(Image.open(rows_path.parent / "out" / f"{r['method']}_{tag}.png"), dtype=np.float32) / 255
        r.update(line_metrics(img, clean, dist, spec.width, spec.contrast))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out


def load(paths: list[Path]) -> list[dict]:
    """rows.csv をまとめて読み（線の指標は出力画像から計算し直したもの）、数値の列を float にする。"""
    rows = []
    for p in paths:
        with rescore(p).open() as f:
            for r in csv.DictReader(f):
                for k in ["width", "angle", "contrast", "level", *METRICS, "f1", "ink_ratio", "width_est", "seconds"]:
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def table(rows: list[dict], row_key, col_key, metric: str, row_label: str, col_fmt=str) -> str:
    """rows を (row_key, col_key) でまとめた metric の平均の Markdown 表。"""
    cells = defaultdict(list)
    for r in rows:
        cells[(row_key(r), col_key(r))].append(r[metric])
    rk = sorted({k[0] for k in cells}, key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else m)
    ck = sorted({k[1] for k in cells})
    out = f"| {row_label} | " + " | ".join(col_fmt(c) for c in ck) + " |\n|---" + "|---" * len(ck) + "|\n"
    for a in rk:
        out += f"| {a} | " + " | ".join(
            f"{st.mean(cells[(a, c)]):.3f}" if (a, c) in cells else "" for c in ck) + " |\n"
    return out


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    """符号検定（両側）。戻り値: (正の数, 負の数, p 値)。差が 0 の対は除く。"""
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n, k = pos + neg, min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return pos, neg, p


def h1_test(rows: list[dict], method: str = "zsn2n", width: float = 1.0, metric: str = "cldice") -> str:
    """仮説 H1: 細い線では、斜め 45°/135° が 0°/90° より悪いか。濃さ × σ の組ごとに対にする。"""
    by = defaultdict(dict)
    for r in rows:
        if r["method"] == method and r["width"] == width and r["angle"] in (0, 45, 90, 135):
            by[(r["contrast"], r["level"])][r["angle"]] = r[metric]
    diffs = [(v[0] + v[90]) / 2 - (v[45] + v[135]) / 2 for v in by.values() if len(v) == 4]
    if not diffs:
        return f"- H1（{method}, {width:g}px）: データなし\n"
    pos, neg, p = sign_test(diffs)
    return (f"- H1（{method}, {width:g}px, {metric}）: 軸方向 − 斜め の差の平均 {st.mean(diffs):+.3f}、"
            f"{len(diffs)} 組中 軸方向が良い {pos} 組 / 斜めが良い {neg} 組、符号検定 p = {p:.3g}\n")


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    rows = load(paths)
    md = "# E1 の集計\n\n"
    md += f"行数 {len(rows)}、手法 {sorted({r['method'] for r in rows})}\n\n"
    for c in sorted({r["contrast"] for r in rows}, reverse=True):
        sub = [r for r in rows if r["contrast"] == c]
        md += f"## 墨の濃さ {c:g}\n\n"
        md += "### 手法 × σ（全太さ・全向きの平均）\n\n"
        for m in ["psnr", "cldice", "recall", "precision", "center_ink", "width_half", "width_err"]:
            md += f"**{m}**\n\n" + table(sub, lambda r: r["method"], lambda r: r["level"], m, "手法", lambda x: f"σ={x:g}") + "\n"
        md += "### 1px 線: 手法 × 向き（clDice、σ 平均）\n\n"
        md += table([r for r in sub if r["width"] == 1], lambda r: r["method"], lambda r: r["angle"], "cldice", "手法",
                    lambda x: f"{x:g}°") + "\n"
        md += "### ZS-N2N: 太さ × 向き（clDice、σ 平均）\n\n"
        md += table([r for r in sub if r["method"] == "zsn2n"], lambda r: f"{r['width']:g}px", lambda r: r["angle"],
                    "cldice", "太さ", lambda x: f"{x:g}°") + "\n"
        for m in ("center_ink", "width_half"):
            md += f"### ZS-N2N: 太さ × 向き（{m}、σ 平均）\n\n"
            md += table([r for r in sub if r["method"] == "zsn2n"], lambda r: f"{r['width']:g}px", lambda r: r["angle"],
                        m, "太さ", lambda x: f"{x:g}°") + "\n"
    md += "## 仮説 H1 の検定\n\n"
    for m in ("zsn2n", "bm3d", "nlm"):
        for w in (1.0, 1.5):
            md += h1_test(rows, m, w)
    print(md)
    out = paths[0].parents[2] / "e1_summary.md"
    out.write_text(md, encoding="utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
