# lineart-zs-denoise

線画（細い線と平らな塗り）に対するゼロショットのノイズ除去を検証する研究用リポジトリです（テーマ A）。
計画は [docs/plan.html](docs/plan.html) を参照してください。

## 準備

```bash
uv sync                               # 依存のインストール（uv.lock で固定）
uv run python scripts/fetch_data.py   # Kodak24・McMaster18 を data/ に取得
```

## 実験の実行

1 回の実験 = 設定ファイル 1 つ（`configs/*.yaml`）→ 結果フォルダ 1 つ（`results/<name>/<設定のハッシュ>/`）。

```bash
uv run python -m zsdn.run --config configs/e0_gauss.yaml          # Mac では MPS（GPU）を自動で使う
uv run python -m zsdn.report results/e0_gauss/*/rows.csv          # 論文の数値と並べて集計
```

途中で止めても、同じコマンドで続きから計算します。

## Docker / Kubernetes（kind）

```bash
docker build -t zsdn:0.1.0 -f docker/Dockerfile .
make kind-up        # ローカルの Kubernetes クラスタを作り、イメージを読み込む
make kind-job CONFIG=configs/e0_gauss.yaml
```

詳しくは `Makefile` と `k8s/` を参照してください。コンテナ内の計算は CPU だけです（Mac の GPU はコンテナから使えません）。

## 構成

- `src/zsdn/zsn2n.py` … Zero-Shot Noise2Noise（公式 Colab と同じ設定）
- `src/zsdn/data.py` … 画像の読み込み、固定シードのノイズ作成と保存
- `src/zsdn/metrics.py` … PSNR・SSIM
- `src/zsdn/run.py` … 実験の実行（並列・途中再開あり）
- `src/zsdn/report.py` … 集計と論文値との比較
- `configs/` … 実験の設定
- `docker/`、`k8s/`、`Makefile` … コンテナと Kubernetes で動かすための設定
