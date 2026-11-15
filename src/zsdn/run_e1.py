"""E1 の実行: 合成線画ベンチマークで、手法ごとに線がどれだけ残るかを測る。

使い方:
    uv run python -m zsdn.run_e1 --config configs/e1_classic.yaml
    uv run python -m zsdn.run_e1 --config configs/e1_zsn2n.yaml --device mps

結果は results/<name>/<設定のハッシュ>/ に保存する（E0 と同じ形式）:
    rows.csv … 画像・ノイズ・シード・手法ごとに 1 行。再実行すると続きから計算する
    out/     … 出力画像（PNG、目で確かめる用）
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from zsdn.baselines import denoise_classic
from zsdn.data import add_noise
from zsdn.lineart import LineSpec, e1_specs, render
from zsdn.linemetrics import line_metrics
from zsdn.metrics import psnr
from zsdn.run import config_hash, git_commit, pick_device
from zsdn.zsn2n import ZSN2NConfig, denoise_zsn2n

FIELDS = ["exp", "image", "width", "angle", "contrast", "noise", "level", "noise_seed", "init_seed", "method",
          "psnr", "psnr_noisy", "recall", "precision", "f1", "cldice", "ink_ratio", "width_est", "width_err",
          "seconds", "device", "config_hash"]


def noisy_for(spec: LineSpec, clean: torch.Tensor, level: float, seed: int, clip: bool, cache_dir: Path) -> torch.Tensor:
    """ノイズ画像をキャッシュから読む（なければ作る）。全手法が同じノイズ画像を使うため。"""
    p = cache_dir / f"{spec.name}_gauss{level:g}{'' if clip else '_noclip'}_s{seed}.pt"
    if p.exists():
        return torch.load(p)
    noisy = add_noise(clean, "gauss", level, seed, clip)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(noisy, p)
    return noisy


def denoise(noisy: torch.Tensor, method: dict, level: float, init_seed: int, device: str) -> tuple[torch.Tensor, float]:
    """手法の設定（dict）に従ってノイズを除く。"""
    if method["name"] == "zsn2n":
        cfg = ZSN2NConfig(**{k: v for k, v in method.items() if k != "name"})
        return denoise_zsn2n(noisy, cfg, init_seed, device)
    return denoise_classic(noisy, method["name"], level)


def main() -> None:
    ap = argparse.ArgumentParser(description="E1: 合成線画ベンチマークの実行")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", help="設定ファイルの device を上書き")
    ap.add_argument("--max-images", type=int, help="先頭 N 枚だけ使う（動作確認用）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    if args.max_images:
        cfg["max_images"] = args.max_images
    device = pick_device(cfg.get("device", "cpu"))
    torch.set_num_threads(int(cfg.get("threads", 4)))
    h = config_hash(cfg)
    out_dir = Path(cfg.get("out_dir", "results")) / cfg["name"] / h
    (out_dir / "out").mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "git_commit": git_commit(), "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "device": device,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    b = cfg["bench"]
    specs = e1_specs(tuple(b["widths"]), b["n_angles"], tuple(b["contrasts"]), b.get("size", 256), b.get("jitter", True))
    specs = specs[: cfg.get("max_images")]
    cache_dir = Path(cfg.get("noise_cache", "data/noisy")) / "lineart"

    rows_path = out_dir / "rows.csv"
    done = set()
    if rows_path.exists():
        with rows_path.open() as f:
            done = {(r["image"], str(float(r["level"])), r["noise_seed"], r["init_seed"], r["method"])
                    for r in csv.DictReader(f)}

    # 学習しない手法はシードで結果が変わらないので、初期値のシードは最初の 1 つだけ使う
    jobs = []
    for spec in specs:
        for nz in cfg["noises"]:
            for ns in cfg.get("noise_seeds", [0]):
                for m in cfg["methods"]:
                    seeds = cfg.get("init_seeds", [0]) if m["name"] == "zsn2n" else cfg.get("init_seeds", [0])[:1]
                    for isd in seeds:
                        if (spec.name, str(float(nz["level"])), str(ns), str(isd), m["name"]) not in done:
                            jobs.append((spec, nz, ns, isd, m))
    print(f"[run_e1] {cfg['name']} ({h}) device={device} 残り {len(jobs)} 件 → {rows_path}", flush=True)

    new_file = not rows_path.exists()
    with rows_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        cache: dict[str, tuple] = {}
        for i, (spec, nz, ns, isd, m) in enumerate(jobs, 1):
            if spec.name not in cache:
                cache = {spec.name: render(spec)}  # 1 枚ずつ使うので、直前の 1 枚だけ覚えておく
            clean, dist = cache[spec.name]
            noisy = noisy_for(spec, clean, nz["level"], ns, nz.get("clip", False), cache_dir)
            out, sec = denoise(noisy, m, nz["level"], isd, device)
            lm = line_metrics(out, clean, dist, spec.width, spec.contrast)
            tag = f"{spec.name}_g{nz['level']:g}_n{ns}_i{isd}"
            Image.fromarray((out[0, 0].numpy() * 255).round().astype(np.uint8)).save(out_dir / "out" / f"{m['name']}_{tag}.png")
            row = {
                "exp": cfg["name"], "image": spec.name, "width": spec.width, "angle": spec.angle,
                "contrast": spec.contrast, "noise": "gauss", "level": nz["level"], "noise_seed": ns,
                "init_seed": isd, "method": m["name"], "psnr": round(psnr(clean, out), 4),
                "psnr_noisy": round(psnr(clean, noisy), 4), **lm, "seconds": round(sec, 2),
                "device": device if m["name"] == "zsn2n" else "cpu", "config_hash": h,
            }
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(jobs)}] {m['name']} {tag} PSNR {row['psnr']:.2f} recall {lm['recall']:.3f} "
                  f"clDice {lm['cldice']:.3f} 太さ {lm['width_est']:.2f}/{spec.width:g} {sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
