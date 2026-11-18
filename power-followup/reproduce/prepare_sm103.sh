#!/bin/bash
set -Eeuo pipefail
work=${1:-/work}
pin=d9ef6321230cff13398cfc86aa107f97f093e925
mkdir -p "$work/source/sana-sm103"
curl -fL --retry 3 "https://codeload.github.com/lishunyang12/Sana/tar.gz/$pin" |
  tar xz --strip-components=1 -C "$work/source/sana-sm103"
printf '%s\n' "$pin" > "$work/source/sana-sm103/SOURCE_COMMIT"
