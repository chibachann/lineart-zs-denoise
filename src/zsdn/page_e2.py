"""部品ごとの効き方調査の報告ページを組み立てる。

docs/components.template.html の {{HEATMAP}} {{TRADEOFF}} {{MONTAGE}} を、集計結果から作った図で置き換えて
docs/components.html に書き出す。図の色は DADS の公式トークン（CSS 変数）を使う。

使い方（先に report_e2 と figs_e2 を実行しておく）:
    uv run python -m zsdn.page_e2 results/e2_rows.csv docs/figs/e2_montage.png
"""
from __future__ import annotations

import base64
import csv
import statistics as st
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

from zsdn.report_e2 import AXIS, VARIANTS

ANGLES = [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5]
ZS_ROWS = ["base", "pair_horiz", "pair_vert", "pair_single", "pair_flip", "loss_res_only", "loss_l1",
           "net_wide", "net_deep", "net_k5", "net_linear", "direct"]
BASE_ROWS = ["identity", "median3", "nlm", "bm3d"]
# ヒートマップの段階（clDice の値 → 色の段）。DADS の Blue を薄い順に使う
BINS = [(0.1, "--heat-0"), (0.3, "--heat-1"), (0.5, "--heat-2"), (0.7, "--heat-3"), (0.9, "--heat-4"), (9, "--heat-5")]


def load(path: Path) -> list[dict]:
    """e2_rows.csv を読み、数値の列を float にする。"""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("width", "angle", "contrast", "level", "psnr", "cldice", "center_ink"):
            r[k] = float(r[k])
    return rows


def heatmap(rows: list[dict]) -> str:
    """1px の線: 条件 × 向きの clDice を色の段と数字で示すヒートマップ（SVG）。"""
    cells = defaultdict(list)
    for r in rows:
        if r["width"] == 1:
            cells[(r["method"], r["angle"])].append(r["cldice"])
    lw, cw, ch, gap, top = 200, 54, 28, 2, 34
    groups = [("ZS-N2N の条件", ZS_ROWS), ("比較手法", BASE_ROWS)]
    n_rows = sum(len(g[1]) for g in groups)
    h = top + n_rows * (ch + gap) + 30 * len(groups)
    w = lw + len(ANGLES) * (cw + gap)
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-labelledby="hm-t"><title id="hm-t">'
           '1px の線の clDice（条件 × 線の向き、σ=25、黒と灰色の線の平均）</title>']
    for j, a in enumerate(ANGLES):
        x = lw + j * (cw + gap) + cw / 2
        out.append(f'<text x="{x}" y="22" text-anchor="middle" class="ax">{a:g}°</text>')
    y = top
    for gname, ms in groups:
        out.append(f'<text x="0" y="{y + 18}" class="grp">{gname}</text>')
        y += 26
        for m in ms:
            label = VARIANTS.get(m, m)
            out.append(f'<text x="{lw - 10}" y="{y + ch / 2 + 5}" text-anchor="end" class="lab{" base" if m == "base" else ""}">'
                       f'{escape(label)}</text>')
            for j, a in enumerate(ANGLES):
                v = st.mean(cells[(m, a)])
                tok = next(t for lim, t in BINS if v < lim)
                dark = tok in ("--heat-3", "--heat-4", "--heat-5")
                x = lw + j * (cw + gap)
                out.append(f'<g><title>{escape(label)}・{a:g}°: clDice {v:.2f}</title>'
                           f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" fill="var({tok})"/>'
                           f'<text x="{x + cw / 2}" y="{y + ch / 2 + 5}" text-anchor="middle" '
                           f'class="v{" on" if dark else ""}">{v:.2f}</text></g>')
            if m == "base":
                out.append(f'<rect x="{lw - 3}" y="{y - 3}" width="{len(ANGLES) * (cw + gap) + 4}" height="{ch + 6}" '
                           'rx="4" fill="none" class="ring"/>')
            y += ch + gap
        y += 4
    out.append("</svg>")
    return "\n".join(out)


def tradeoff(rows: list[dict]) -> str:
    """横軸 PSNR、縦軸「1px の斜め線の clDice」の散布図（SVG）。右上ほど良い。"""
    by = defaultdict(lambda: {"psnr": [], "obl": []})
    for r in rows:
        by[r["method"]]["psnr"].append(r["psnr"])
        if r["width"] == 1 and r["angle"] not in AXIS:
            by[r["method"]]["obl"].append(r["cldice"])
    pts = {m: (st.mean(v["psnr"]), st.mean(v["obl"])) for m, v in by.items() if m in ZS_ROWS + BASE_ROWS}
    W, H, L, R, T, B = 720, 420, 56, 24, 20, 48
    x0, x1, y0, y1 = 18, 42, 0, 1
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    sy = lambda v: T + (1 - (v - y0) / (y1 - y0)) * (H - T - B)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="to-t"><title id="to-t">'
         '条件ごとの PSNR と、1px の斜め線の clDice（σ=25）</title>']
    for v in range(20, 43, 4):
        o.append(f'<line x1="{sx(v)}" x2="{sx(v)}" y1="{T}" y2="{H - B}" class="grid"/>'
                 f'<text x="{sx(v)}" y="{H - B + 18}" text-anchor="middle" class="ax">{v}</text>')
    for v in (0, 0.25, 0.5, 0.75, 1):
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v)}" y2="{sy(v)}" class="grid"/>'
                 f'<text x="{L - 8}" y="{sy(v) + 4}" text-anchor="end" class="ax">{v:g}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" class="axt">PSNR（dB、全太さの平均）→ 高いほど良い</text>')
    o.append(f'<text x="14" y="{(T + H - B) / 2}" text-anchor="middle" class="axt" '
             f'transform="rotate(-90 14 {(T + H - B) / 2})">1px 斜め線の clDice →</text>')
    # 名前を付ける点と、文字の位置（点からのずれ）。近くに固まる構造の変更などはまとめて 1 つの注記にする
    labels = {
        "base": ("元の設定", 10, 16), "net_linear": ("活性化なし", 10, -8), "direct": ("残差学習なし", 10, -8),
        "pair_horiz": ("上下・左右に分ける", 10, -10), "pair_single": ("平均しない", -10, -8),
        "loss_res_only": ("一貫性損失なし", -10, 18), "loss_l1": ("絶対誤差", 8, 20),
        "identity": ("何もしない", 10, 4), "median3": ("3×3 メディアン", 10, 18), "nlm": ("NLM", 10, 4), "bm3d": ("BM3D", -10, 18),
    }
    for m, (px, py) in sorted(pts.items(), key=lambda kv: kv[0] == "base"):
        cls = "pz" if m in ZS_ROWS else "pb"
        name = VARIANTS.get(m, m)
        o.append(f'<g><title>{escape(name)}: PSNR {px:.2f} dB、1px 斜め clDice {py:.2f}</title>'
                 f'<circle cx="{sx(px)}" cy="{sy(py)}" r="12" fill="transparent"/>'
                 f'<circle cx="{sx(px)}" cy="{sy(py)}" r="{6 if m == "base" else 5}" class="{cls}{" pbase" if m == "base" else ""}"/></g>')
        if m in labels:
            t, dx, dy = labels[m]
            o.append(f'<text x="{sx(px) + dx}" y="{sy(py) + dy}" text-anchor="{"end" if dx < 0 else "start"}" class="pl">{t}</text>')
    bx, by_ = pts["base"]
    o.append(f'<text x="{sx(bx) + 10}" y="{sy(0.24) - 14}" class="pl mut">構造の変更・左右反転（元の設定の近く）</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> None:
    rows = load(Path(sys.argv[1]))
    montage = base64.b64encode(Path(sys.argv[2]).read_bytes()).decode()
    tpl = Path("docs/components.template.html").read_text(encoding="utf-8")
    html = (tpl.replace("{{HEATMAP}}", heatmap(rows)).replace("{{TRADEOFF}}", tradeoff(rows))
            .replace("{{MONTAGE}}", f"data:image/png;base64,{montage}"))
    Path("docs/components.html").write_text(html, encoding="utf-8")
    print(f"→ docs/components.html ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
