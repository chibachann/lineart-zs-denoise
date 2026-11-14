# 実験用イメージ（CPU 版）。Mac（arm64）でもクラウド（amd64）でもビルドできる。
# データと結果はイメージに入れず、実行時に /work/data と /work/results にマウントする。
FROM python:3.12-slim

# uv（依存を uv.lock どおりに入れるため）
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/work/.venv/bin:$PATH"

WORKDIR /work

# 依存だけ先に入れる（ソースを変えても、このレイヤーは再利用される）
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# ソースと設定
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN uv sync --frozen --no-dev

# 結果の meta.json に記録するため、ビルド時のコミット ID を埋め込む
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["python", "-m", "zsdn.run"]
CMD ["--config", "configs/e0_gauss.yaml", "--device", "cpu"]
