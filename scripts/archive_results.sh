#!/bin/zsh
# 実験結果の退避: フォルダを tar.gz にまとめて GitHub の Release（タグ results-archive）に上げ、
# 上げたものを取り直して SHA-256 が一致した場合だけ、ローカルのフォルダを削除する。
# 削除した内容と保存先は docs/storage-log.md に追記する（このファイルはコミットして残す）。
#
# 使い方: scripts/archive_results.sh <退避するフォルダ（例: results/e1_classic）>
set -euo pipefail
src=${1:?退避するフォルダを指定}
[[ -d $src ]] || { echo "フォルダがありません: $src" >&2; exit 1; }
cd "$(dirname "$0")/.."
tag=results-archive
name="$(echo "$src" | tr '/' '_')_$(date +%Y%m%d-%H%M%S).tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

tar czf "$tmp/$name" "$src"
sum=$(shasum -a 256 "$tmp/$name" | awk '{print $1}')
size=$(du -sh "$src" | awk '{print $1}')
files=$(find "$src" -type f | wc -l | tr -d ' ')

# Release がなければ作る（実験結果の置き場。ソースコードのリリースではない）
gh release view "$tag" > /dev/null 2>&1 || gh release create "$tag" --title "実験結果の退避" \
  --notes "ローカルのディスク節約のため退避した実験結果（results/ 以下）。中身と元の場所は docs/storage-log.md を参照。"
gh release upload "$tag" "$tmp/$name"

# 取り直して一致を確かめる。一致しなければ削除しない
mkdir "$tmp/check"
gh release download "$tag" --pattern "$name" --dir "$tmp/check"
[[ $(shasum -a 256 "$tmp/check/$name" | awk '{print $1}') == "$sum" ]] || { echo "SHA-256 が一致しないので削除しません" >&2; exit 2; }

url=$(gh release view "$tag" --json assets --jq ".assets[] | select(.name==\"$name\") | .url")
rm -rf "$src"
log=docs/storage-log.md
[[ -f $log ]] || printf '# 実験結果の退避記録\n\nディスク節約のため GitHub の Release に上げてからローカルで削除したもの。復元は `gh release download results-archive --pattern <ファイル名>` のあと `tar xzf <ファイル名>`（リポジトリの直下で）。\n\n| 日時 | 削除したフォルダ | ファイル数 | 大きさ | 保存先 | SHA-256 |\n|---|---|---|---|---|---|\n' > "$log"
echo "| $(date '+%F %T') | \`$src\` | $files | $size | [$name]($url) | \`${sum:0:16}…\` |" >> "$log"
echo "退避して削除しました: $src → $url"
