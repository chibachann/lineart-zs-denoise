# よく使う操作のまとめ。make <ターゲット> で実行する。
REPO    := $(shell pwd)
COMMIT  := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
IMAGE   ?= zsdn:0.1.0
CONFIG  ?= configs/e0_gauss.yaml
EXTRA   ?=
# Job 名: 設定ファイル名 + 時刻（Kubernetes の名前に使える文字だけ）
JOBNAME := zsdn-$(shell basename $(CONFIG) .yaml | tr '_' '-')-$(shell date +%m%d%H%M%S)

.PHONY: data image kind-up kind-load kind-job kind-logs kind-down

data:            ## 評価画像を取得
	uv run python scripts/fetch_data.py

image:           ## 実験用 Docker イメージをビルド
	docker build --build-arg GIT_COMMIT=$(COMMIT) -t $(IMAGE) -f docker/Dockerfile .

kind-up:         ## kind クラスタを作り、ストレージを用意する
	mkdir -p data results
	sed 's#__REPO__#$(REPO)#g' k8s/kind-cluster.yaml | kind create cluster --config -
	kubectl apply -f k8s/storage.yaml

kind-load: image ## イメージを kind クラスタに読み込む
	kind load docker-image $(IMAGE) --name zsdn

kind-job:        ## 実験を Job として流す（例: make kind-job CONFIG=configs/e0_gauss.yaml EXTRA='--max-images 1'）
	sed -e 's#__NAME__#$(JOBNAME)#' -e 's#__IMAGE__#$(IMAGE)#' -e 's#__CONFIG__#$(CONFIG)#' \
	    -e 's#__EXTRA__#$(if $(EXTRA),$(foreach a,$(EXTRA),$(comma) "$(a)"),)#' k8s/job.yaml | kubectl apply -f -
	@echo "ログ: kubectl logs -f job/$(JOBNAME)"

kind-logs:       ## 最新の Job のログを表示
	kubectl logs -f $$(kubectl get jobs -l app=zsdn --sort-by=.metadata.creationTimestamp -o name | tail -1)

kind-down:       ## kind クラスタを削除
	kind delete cluster --name zsdn

comma := ,
