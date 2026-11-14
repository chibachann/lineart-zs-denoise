"""評価用の画像（Kodak24、McMaster18）をダウンロードする。

保存先: data/kodak24/kodim01.png … , data/mcmaster18/1.tif …
出典は data/SOURCES.txt に記録する。既にあるファイルは取り直さない。
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SOURCES = {
    "kodak24": {
        "files": [(f"kodim{i:02d}.png", f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png") for i in range(1, 25)],
        "note": "Kodak Lossless True Color Image Suite（r0k.us で配布）。研究用途で広く使われる評価画像",
    },
    "mcmaster18": {
        "files": [(f"{i}.tif", f"https://raw.githubusercontent.com/JingyunLiang/SwinIR/main/testsets/McMaster/{i}.tif") for i in range(1, 19)],
        "note": "McMaster 色画像データセット（SwinIR リポジトリ testsets/McMaster、リポジトリは Apache-2.0）",
    },
}


def main() -> None:
    lines = []
    for name, spec in SOURCES.items():
        folder = DATA / name
        folder.mkdir(parents=True, exist_ok=True)
        for fname, url in spec["files"]:
            dst = folder / fname
            if not dst.exists():
                req = urllib.request.Request(url, headers={"User-Agent": "zsdn-research-fetch"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    dst.write_bytes(r.read())
                print(f"取得: {dst.relative_to(ROOT)}")
        lines.append(f"{name}: {spec['note']}\n  URL の例: {spec['files'][0][1]}\n")
    (DATA / "SOURCES.txt").write_text("".join(lines), encoding="utf-8")
    print("完了。出典は data/SOURCES.txt に記録しました。")


if __name__ == "__main__":
    main()
