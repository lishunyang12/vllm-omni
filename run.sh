# Point Python at VoxCPM's ``src`` (parent of ``voxcpm/model`` and ``voxcpm/modules``) if not next to this repo.
export VLLM_OMNI_VOXCPM_CODE_PATH=/home/l00613087/voxcpm/VoxCPM/src
export ASCEND_RT_VISIBLE_DEVICES=1
export VOXCPM_MODEL=/home/l00613087/voxcpm/weights/VoxCPM1.5


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_NO_ASYNC="${SCRIPT_DIR}/vllm_omni/model_executor/stage_configs/voxcpm_no_async_chunk.yaml"
DEFAULT_ASYNC="${SCRIPT_DIR}/vllm_omni/model_executor/stage_configs/voxcpm.yaml"

# echo "[CASE 1] sync (voxcpm_no_async_chunk.yaml)"
# python examples/offline_inference/voxcpm/end2end.py \
#   --model "$VOXCPM_MODEL" \
#   --stage-configs-path "${VOXCPM_STAGE_CONFIG_SYNC:-$DEFAULT_NO_ASYNC}" \
#   --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."

echo "[CASE 2] streaming (voxcpm.yaml)"
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path "${VOXCPM_STAGE_CONFIG_STREAMING:-$DEFAULT_ASYNC}" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni." \
  --num-runs 3