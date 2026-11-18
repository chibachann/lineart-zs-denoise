# 実験結果の退避記録

ディスク節約のため GitHub の Release（タグ results-archive）に上げてからローカルで削除したものを記録する。
退避は `scripts/archive_results.sh <フォルダ>` で行う（取り直して SHA-256 が一致した場合だけ削除し、この表に追記する）。
復元は `gh release download results-archive --pattern <ファイル名>` のあと、リポジトリの直下で `tar xzf <ファイル名>`。

## 監視の記録

- 部品ごとの効き方調査（2026-11-18〜19）: `scripts/disk_watch.sh` で 15 分ごとに監視。しきい値は results/ が 2GB 超、または空きが 20GB 未満。
  results/ は 44MB → 71MB、空きは 80GB のままで、しきい値に届かなかったため、退避・削除はしていない。

## 退避して削除したもの

| 日時 | 削除したフォルダ | ファイル数 | 大きさ | 保存先 | SHA-256 |
|---|---|---|---|---|---|
