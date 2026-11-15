#!/usr/bin/env bash
# Run inside the prepared, pinned container. Precision affects only the DiT.
set -euo pipefail
work=${1:?Usage: serve_online.sh WORK MODEL_KEY PRECISION [PORT]}
model_key=${2:?Choose h3_image, h3_video, qwen_image or wan22}
precision=${3:-mxfp8}
port=${4:-8000}
case "$precision" in bf16|mxfp8|fp8_per_block) ;; *) echo "Unsupported precision: $precision" >&2; exit 2;; esac
case "$model_key" in
 h3_image|h3_video) encoder_tp=4 ;;
 qwen_image|wan22) encoder_tp=1 ;;
 *) echo "Unsupported model: $model_key" >&2; exit 2 ;;
esac
export PYTHONPATH="$work/source/candidate${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$work/hf-cache" XDG_CACHE_HOME="$work/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$work/torchinductor-cache" TRITON_CACHE_DIR="$work/triton-cache"
export TORCH_HOME="$work/torch-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
model_path=$(python3 - "$work" "$model_key" <<'PYMODEL'
import json, sys
from pathlib import Path
work=Path(sys.argv[1])
p=work/'evidence/serving-models.json'
if not p.exists(): p=work/'evidence/models.json'
model=json.loads(p.read_text())[sys.argv[2]]
path=Path(model['path'])
if not path.is_dir(): raise SystemExit('Missing pinned model snapshot: '+str(path))
print(path)
PYMODEL
)
args=(serve "$model_path" --omni --served-model-name "$model_key"
      --host 0.0.0.0 --port "$port" --dtype bfloat16 --enforce-eager
      --num-gpus 4 --tensor-parallel-size 1 --ulysses-degree 4 --ring-degree 1
      --text-encoder-tp-size "$encoder_tp")
if [[ "$precision" != bf16 ]]; then
 args+=(--diffusion-quantization-config "{\"transformer\":{\"method\":\"$precision\"},\"text_encoder\":null}")
fi
if [[ "$model_key" == h3_video ]]; then args+=(--trust-remote-code); fi
if [[ "$model_key" == wan22 ]]; then args+=(--flow-shift 5.0); fi
printf 'Launching:'
printf ' %q' vllm "${args[@]}"
printf '\n'
exec vllm "${args[@]}"
