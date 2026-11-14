"""実験の実行: 設定ファイル（YAML）1つ → 結果フォルダ1つ。

使い方:
    uv run python -m zsdn.run --config configs/e0_gauss.yaml

結果は results/<name>/<設定のハッシュ>/ に保存する:
    config.yaml  … 実行した設定そのもの
    meta.json    … git のコミット ID、ライブラリのバージョン、実行環境
    rows.csv     … 画像・ノイズ・シードごとに 1 行。途中で止めても、再実行すると続きから計算する
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import yaml

from zsdn.data import cached_noisy, list_images, load_image
from zsdn.metrics import psnr, ssim
from zsdn.zsn2n import ZSN2NConfig, denoise_zsn2n

FIELDS = ["exp", "dataset", "image", "noise", "level", "noise_seed", "init_seed", "method",
          "psnr", "ssim", "psnr_noisy", "seconds", "device", "config_hash"]


def config_hash(cfg: dict) -> str:
    """設定内容から短いハッシュを作る（実行環境に関わる項目は除く）。"""
    core = {k: v for k, v in cfg.items() if k not in ("workers", "threads_per_worker", "device", "max_images")}
    return hashlib.sha1(json.dumps(core, sort_keys=True).encode()).hexdigest()[:10]


def git_commit() -> str:
    """現在の git のコミット ID（未コミットの変更があれば末尾に +dirty）。"""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def pick_device(name: str) -> str:
    """auto のときは MPS（Apple の GPU）→ CUDA → CPU の順で使えるものを選ぶ。"""
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_task(task: dict) -> dict:
    """1 枚・1 条件・1 シードを処理して、CSV の 1 行を返す（別プロセスで実行される）。"""
    torch.set_num_threads(task["threads"])
    clean = load_image(task["path"], task["crop"])
    noisy = cached_noisy(clean, task["noise"], task["level"], task["noise_seed"], Path(task["noise_cache"]))
    m = task["method"]
    if m["name"] == "zsn2n":
        cfg = ZSN2NConfig(**{k: v for k, v in m.items() if k != "name"})
        out, sec = denoise_zsn2n(noisy, cfg, task["init_seed"], task["device"])
    elif m["name"] == "identity":  # 比較用: 何もしない
        out, sec = noisy, 0.0
    else:
        raise ValueError(f"未知の手法: {m['name']}")
    return {
        "exp": task["exp"], "dataset": task["dataset"], "image": Path(task["path"]).stem,
        "noise": task["noise"], "level": task["level"], "noise_seed": task["noise_seed"],
        "init_seed": task["init_seed"], "method": m["name"],
        "psnr": round(psnr(clean, out), 4), "ssim": round(ssim(clean, out), 5),
        "psnr_noisy": round(psnr(clean, noisy), 4), "seconds": round(sec, 2),
        "device": task["device"], "config_hash": task["hash"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ゼロショットのノイズ除去の実験を実行する")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", help="設定ファイルの device を上書き（cpu / mps / cuda / auto）")
    ap.add_argument("--workers", type=int, help="並列プロセス数を上書き")
    ap.add_argument("--max-images", type=int, help="各データセットの先頭 N 枚だけ使う（動作確認用）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    if args.workers:
        cfg["workers"] = args.workers
    if args.max_images:
        cfg["max_images"] = args.max_images
    device = pick_device(cfg.get("device", "auto"))
    workers = 1 if device != "cpu" else int(cfg.get("workers", 1))  # GPU は 1 プロセスで使う
    h = config_hash(cfg)
    out_dir = Path(cfg.get("out_dir", "results")) / cfg["name"] / h
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps({
        "git_commit": git_commit(), "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "machine": platform.machine(), "device": device,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 既に計算済みの行は飛ばす（途中から再開できるように）
    rows_path = out_dir / "rows.csv"
    done = set()
    if rows_path.exists():
        with rows_path.open() as f:
            for r in csv.DictReader(f):
                done.add((r["dataset"], r["image"], r["noise"], str(float(r["level"])), r["noise_seed"], r["init_seed"]))

    tasks = []
    for ds in cfg["datasets"]:
        imgs = list_images(ds["dir"])[: cfg.get("max_images")]
        for p in imgs:
            for nz in cfg["noises"]:
                for ns in cfg.get("noise_seeds", [0]):
                    for isd in cfg.get("init_seeds", [0]):
                        key = (ds["name"], p.stem, nz["kind"], str(float(nz["level"])), str(ns), str(isd))
                        if key in done:
                            continue
                        cache = Path(cfg.get("noise_cache", "data/noisy")) / ds["name"] / f"{p.stem}_{nz['kind']}{nz['level']}_c{cfg.get('crop', 256)}_s{ns}.pt"
                        tasks.append({
                            "exp": cfg["name"], "dataset": ds["name"], "path": str(p), "crop": cfg.get("crop", 256),
                            "noise": nz["kind"], "level": nz["level"], "noise_seed": ns, "init_seed": isd,
                            "noise_cache": str(cache), "method": cfg["method"], "device": device,
                            "threads": int(cfg.get("threads_per_worker", 1)) if device == "cpu" else 1, "hash": h,
                        })

    print(f"[run] {cfg['name']} ({h}) device={device} workers={workers} 残り {len(tasks)} 件 → {rows_path}", flush=True)
    new_file = not rows_path.exists()
    with rows_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        if workers == 1:
            results = (run_task(t) for t in tasks)
        else:
            pool = ProcessPoolExecutor(max_workers=workers)
            results = (fut.result() for fut in as_completed([pool.submit(run_task, t) for t in tasks]))
        for i, row in enumerate(results, 1):
            w.writerow(row)
            f.flush()
            print(f"[{i}/{len(tasks)}] {row['dataset']} {row['image']} {row['noise']}{row['level']} "
                  f"PSNR {row['psnr']:.2f} dB（ノイズ画像 {row['psnr_noisy']:.2f}）{row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
