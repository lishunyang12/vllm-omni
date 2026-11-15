#!/usr/bin/env bash
# Run inside the pinned container after preparation in prepare.sh. Requires four GPUs.
set -euo pipefail
work=${1:-/work}
export HF_HOME="$work/hf-cache" UV_CACHE_DIR="$work/uv-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache"
export TORCH_HOME="$work/torch-cache" OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_DISABLE_PROGRESS_BARS=1
python3 "$work/run_matrix.py" "$work"
if [[ -f "$work/test_fp8_block_runtime.py" ]] && python3 -c 'import pytest' 2>/dev/null; then
  export PYTHONPATH="$work/source/candidate"
  if python3 -m pytest -q -s --tb=short "$work/test_fp8_block_runtime.py" > "$work/evidence/fp8-block-corrected.log" 2>&1; then
    echo 0 > "$work/evidence/fp8-block-corrected.exit"
  else
    echo 1 > "$work/evidence/fp8-block-corrected.exit"
  fi
fi
python3 "$work/score_outputs.py" "$work"

# On this Slurm run, keep the allocation until the outputs have been copied out.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo results_ready > "$work/status"
  for attempt in $(seq 1 120); do
    if [[ -f "$work/collected" ]]; then break; fi
    sleep 5
  done
fi
