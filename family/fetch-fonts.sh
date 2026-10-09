#!/bin/sh
# Downloads the OFL fonts used by index.html (kept out of git to save space)
mkdir -p "$(dirname "$0")/fonts" && cd "$(dirname "$0")/fonts" || exit 1
for f in gowunbatang/GowunBatang-Regular.ttf gowunbatang/GowunBatang-Bold.ttf nanumpenscript/NanumPenScript-Regular.ttf; do
  curl -sSfL -o "$(basename "$f")" "https://raw.githubusercontent.com/google/fonts/main/ofl/$f"
done
