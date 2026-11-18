#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
for item in baseline:63b4e65e9db1dfea45cc7d394902145c2f520821 candidate:fe6be477faa270d75fd3e4dd26305b3631de9247; do
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
