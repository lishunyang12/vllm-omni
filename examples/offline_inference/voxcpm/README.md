# VoxCPM

This directory contains the minimal offline example for running native VoxCPM in vLLM Omni on the `pure_voxcpm` branch.

It covers:

- split-stage offline inference
- streaming with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- non-streaming with `vllm_omni/model_executor/stage_configs/voxcpm_no_async_chunk.yaml`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`

## Prerequisites

Install the VoxCPM codebase in one of these ways:

```bash
pip install voxcpm
```

or point vLLM Omni to the local VoxCPM source tree:

```bash
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

The example writes WAV files with `soundfile`:

```bash
pip install soundfile
```

## Model Path

Pass the native VoxCPM model directory directly. The original VoxCPM `config.json` can stay in native format. `vllm-omni` will render the HF-compatible config it needs at runtime.

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
```

## Quick Start

Text-only synthesis (non-streaming by default):

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Voice cloning:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio."
```

Streaming:

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Generated audio is saved to `output_audio/` by default for non-streaming, and to
`output_audio_streaming/` by default for streaming.

## Useful Arguments

- `--stage-configs-path`: override the split-stage config path explicitly
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length

## Omni async_chunk vs Qwen3-TTS (same transport, different Stage0 payloads)

Both pipelines use `async_chunk: true`, [`OmniChunkTransferAdapter`](../../../vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py), `SharedMemoryConnector`, and a `custom_process_next_stage_input_func` to build the Stage0→Stage1 payload (`code_predictor_codes`, `finished`, plus modality-specific fields).

| Aspect | Qwen3-TTS | VoxCPM |
|--------|-----------|--------|
| Stage0 `worker_type` | `ar` | `ar` |
| Stage0 scheduler | `OmniARScheduler` | `OmniARScheduler` |
| What each Stage0 step produces | One speech-token frame (`audio_codes` in pooler) | One latent chunk (`latent_audio_feat`) |
| “More chunks?” signal | Implicit via AR decode until EOS | Implicit via AR request completion on the last latent chunk |
| Stage1 scheduler | `OmniGenerationScheduler` | `OmniGenerationScheduler` |
| Stage1 payload | speech codes + chunk context | latent chunk + optional sample rate |
| Stage1 | Code2Wav | VAE decode (`trim_streaming_patch` trims overlap) |

**Stage0→Stage1 payload contract (VoxCPM streaming):** `latent2vae_async_chunk` sends `latent_audio_feat`, optional `sr`, `code_predictor_codes: [0]`, and `finished` when the Stage0 AR request has completed. This keeps VoxCPM aligned with the framework’s common async-chunk pattern used by `qwen3_tts`.

## Notes

- This branch keeps the split-stage `latent_generator -> vae` pipeline.
- Both streaming and non-streaming offline inference are kept.
- It does not include the single-stage `voxcpm_full.yaml` path.
- It does not include the OpenAI-compatible online speech serving adaptation.
- Voice cloning requires both `--ref-audio` and `--ref-text`.
