#!/bin/zsh
# ディスク容量の監視: 一定間隔で空き容量と results/ の大きさを logs/disk.log に記録する。
# どちらかがしきい値を超えたら終了コード 10 で終わる（呼び出し側が、GitHub への退避と削除を行う合図）。
# 監視対象の実験プロセス（引数のパターンに合うもの）が終わったら、終了コード 0 で終わる。
#
# 使い方: scripts/disk_watch.sh <監視するプロセスのパターン> [結果の上限 MB] [空きの下限 GB] [間隔 秒]
pattern=${1:?プロセスのパターンを指定}
max_results_mb=${2:-2048}
min_free_gb=${3:-20}
interval=${4:-600}
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
while true; do
  free_gb=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  results_mb=$(du -sm results | awk '{print $1}')
  echo "$(date '+%F %T') free=${free_gb}GB results=${results_mb}MB" >> logs/disk.log
  if (( results_mb > max_results_mb || free_gb < min_free_gb )); then
    echo "$(date '+%F %T') しきい値超え（results>${max_results_mb}MB または 空き<${min_free_gb}GB）" >> logs/disk.log
    exit 10
  fi
  pgrep -f "$pattern" > /dev/null || exit 0
  sleep "$interval"
done
