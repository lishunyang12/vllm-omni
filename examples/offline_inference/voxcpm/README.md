# VoxCPM

This directory contains the minimal offline example for running native VoxCPM in vLLM Omni on the `pure_voxcpm` branch.

It covers:

<<<<<<< HEAD
- split-stage inference with `vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`
- batch processing with voice cloning support
- OpenAI-compatible speech serving input adaptation for VoxCPM
=======
- split-stage offline inference
- streaming with `vllm_omni/model_executor/stage_configs/voxcpm.yaml`
- non-streaming with `vllm_omni/model_executor/stage_configs/voxcpm_no_async_chunk.yaml`
- text-only synthesis
- voice cloning with `ref_audio` + `ref_text`
- text-to-speech batch inputs from `--txt-prompts`
- voice cloning batch inputs from `--txt-prompts` or `--jsonl-prompts`
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

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

<<<<<<< HEAD
### Single Text Synthesis
=======
Text-only synthesis (non-streaming by default):
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

<<<<<<< HEAD
### Voice Cloning
=======
Voice cloning:
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --text "This sentence is synthesized with a cloned voice." \
<<<<<<< HEAD
  --prompt-audio /path/to/reference.wav \
  --prompt-text "Transcript of the reference audio."
```

### Batch Processing from Text File

Process multiple texts (one per line):
=======
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio."
```

Text batch (`--txt-prompts`, one text per line):
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
<<<<<<< HEAD
  --input example_texts.txt \
  --output-dir ./outs
```

### Batch Processing with Voice Cloning

Process multiple texts with a shared reference audio:
=======
  --txt-prompts /path/to/prompts.txt \
  --batch-size 4
```

Voice cloning batch with one shared reference (`--txt-prompts` + `--ref-audio` + `--ref-text`):
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
<<<<<<< HEAD
  --input example_texts.txt \
  --prompt-audio reference.wav \
  --prompt-text "reference transcript" \
  --output-dir ./outs
```

### Batch Processing from JSONL

Process multiple texts with individual reference audio files:
=======
  --txt-prompts /path/to/prompts.txt \
  --ref-audio /path/to/reference.wav \
  --ref-text "Transcript of the reference audio." \
  --batch-size 4
```

Voice cloning batch with per-item references (`--jsonl-prompts`):

```bash
cat >/tmp/voxcpm_clone_batch.jsonl <<'EOF'
{"text": "This is the first cloned sentence.", "ref_audio": "/path/to/ref_a.wav", "ref_text": "Transcript for reference A."}
{"text": "This is the second cloned sentence.", "ref_audio": "/path/to/ref_b.wav", "ref_text": "Transcript for reference B."}
EOF

python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
  --jsonl-prompts /tmp/voxcpm_clone_batch.jsonl \
  --batch-size 2
```

Streaming:
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d

```bash
python examples/offline_inference/voxcpm/end2end.py \
  --model "$VOXCPM_MODEL" \
<<<<<<< HEAD
  --jsonl-file example_batch.jsonl \
  --output-dir ./outs
```

JSONL file format (one JSON object per line):
```json
{"audio": "reference1.wav", "text": "This is the first example text."}
{"audio": "reference2.wav", "text": "This is the second example."}
{"audio": "reference3.wav", "text": "You can use absolute or relative paths."}
```

## Running Examples

You can run all examples at once using the provided script:

```bash
bash examples/offline_inference/voxcpm/run_examples.sh
```

Or run individual examples by copying the commands from the script.

## Useful Arguments

### Input Modes (mutually exclusive)

- `--text` / `-t`: single text to synthesize
- `--input` / `-i`: path to text file (one text per line)
- `--jsonl-file`: path to a JSONL file with audio/text pairs

### Voice Cloning

- `--prompt-audio` / `-pa`: reference audio file path (clone mode)
- `--prompt-text` / `-pt`: transcript of the reference audio

### Generation Parameters

- `--cfg-value`: guidance value passed to VoxCPM (default: 2.0)
- `--inference-timesteps`: number of diffusion timesteps (default: 10)
- `--min-len`: minimum token length (default: 2)
- `--max-new-tokens`: maximum token length (default: 4096)

### Output and Runtime

- `--output` / `-o`: output audio file path (single or clone mode)
- `--output-dir` / `-od`: output directory (batch mode only)
- `--stage-configs-path`: override the split-stage config path explicitly
- `--stage-init-timeout`: stage initialization timeout in seconds (default: 600)
- `--log-stats`: enable vLLM Omni stats logging

## Architecture

The script follows a clean, modular architecture similar to `qwen3_tts/end2end.py`:

- **`_build_synthesize_input(args)`**: Build inputs for single text synthesis (no voice cloning)
- **`_build_clone_input(args)`**: Build inputs for voice cloning with reference audio
- **`_build_batch_input(args)`**: Build inputs for batch processing from text file or JSONL file
- **`main(args)`**: Unified entry point that routes to appropriate command and calls `omni.generate()`

This design ensures:
- Clear separation of concerns
- Easy to extend with new modes
- Consistent with vLLM Omni patterns
- Minimal code duplication

## Notes

- This branch keeps the split-stage `latent_generator -> vae` pipeline and defaults to the async-chunk stage config.
- It does not include the single-stage `voxcpm_full.yaml` path.
- The OpenAI-compatible `/v1/audio/speech` path now accepts VoxCPM requests, but the model still relies on the split-stage native runtime underneath.
- Voice cloning requires both `--prompt-audio` and `--prompt-text` (or audio/text in JSONL).
- Batch processing with JSONL allows individual reference audio per text sample.
- The batch processing implementation follows VoxCPM's CLI patterns from `VoxCPM/src/voxcpm/cli.py`.
- Batch processing processes texts one by one (sequential), following VoxCPM's original behavior.
- All parameter names are consistent with VoxCPM's original CLI (excluding LoRA-related parameters).
=======
  --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm.yaml \
  --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

Generated audio is saved to `output_audio/` by default for non-streaming, and to
`output_audio_streaming/` by default for streaming.

The same batch flags also work with the streaming stage config. In streaming mode,
`--batch-size` controls how many requests are launched concurrently in each wave.

## Batch Input Formats

`--txt-prompts` expects one synthesis text per non-empty line:

```text
This is the first sentence.
This is the second sentence.
This is the third sentence.
```

`--jsonl-prompts` expects one JSON object per line. Each line must contain `text`.
For voice cloning rows, provide `ref_audio` and `ref_text` together:

```json
{"text": "Text-only row"}
{"text": "Clone row", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript."}
```

## Useful Arguments

- `--stage-configs-path`: override the split-stage config path explicitly
- `--txt-prompts`: load one synthesis text per line from a `.txt` file
- `--jsonl-prompts`: load prompts from `.jsonl`, including per-item voice cloning metadata
- `--batch-size`: requests submitted together per sync batch, or concurrent requests per streaming wave
- `--cfg-value`: guidance value passed to VoxCPM
- `--inference-timesteps`: number of diffusion timesteps
- `--min-len`: minimum token length
- `--max-new-tokens`: maximum token length

## Batching Notes

The script accepts prompt batches in both sync and streaming modes, but the
default VoxCPM stage configs still use `runtime.max_batch_size: 1`. That means:

- `--txt-prompts` and `--jsonl-prompts` already let you run many requests in one command.
- `--batch-size` controls prompt grouping in the script.
- To get true engine-level batching inside the stage runtime, also raise the batch-related limits in the stage config.

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
- Voice cloning always requires `ref_audio` and `ref_text` together, whether passed globally or inside JSONL rows.
>>>>>>> b91655143ab1a16b3105ed625c11ed7177464a4d
