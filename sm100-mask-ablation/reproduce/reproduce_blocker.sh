#!/bin/bash
# Inside the pinned image with this evidence's reproduce/ mounted at /work/scripts.
# Uses synthetic inputs only; no H3 weights or private captures are needed.
set -Eeuo pipefail
export CUTE_DSL_DISABLE_FILE_CACHING=1
python3 /work/scripts/fetch_sources.py /work
bash /work/scripts/run_independent_process.sh
bash /work/scripts/run_extension_probe.sh
PYTHONPATH=/work/source/candidate/techniques/sparse_backends \
  python3 /work/scripts/compare_independent.py
