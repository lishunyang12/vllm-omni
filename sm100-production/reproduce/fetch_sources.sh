#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
for item in baseline:63b4e65e9db1dfea45cc7d394902145c2f520821 candidate:150114b6f0079bcd479c6c87bd0b15174581d984; do
  name=${item%%:*} commit=${item#*:}
  target="$work/source/$name"
  mkdir -p "$target"
  if [[ -n $(ls -A "$target") ]]; then
    echo "Refusing to overwrite nonempty source checkout: $target" >&2
    exit 1
  fi
  curl -fL --retry 3 "https://codeload.github.com/lishunyang12/Sana/tar.gz/$commit" |
    tar -xz --strip-components=1 -C "$target"
done
