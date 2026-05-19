# Lance

> Unified autoregressive + diffusion multimodal generation and understanding

## Summary

- Vendor: ByteDance
- Model: `bytedance-research/Lance` (`Lance_3B`, `Lance_3B_Video`)
- Task: text2img, text2video, image_edit, video_edit, img2text, video2text
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

Use this recipe to run Lance-3B via vLLM-Omni. Lance is a 3B unified
autoregressive + diffusion multimodal model on a Qwen2.5-VL backbone. It is
**BAGEL-lineage** (ByteDance Mixture-of-Transformers): the released `Lance_3B`
checkpoint reuses the BAGEL transformer core and specializes only the ViT
(Qwen2.5-VL vision), the VAE (Wan2.2) and the checkpoint layout. A single
self-contained diffusion stage covers all six modalities.

Lance has **no bespoke example script**: the Lance-specific system prompt /
Qwen chat template / vision-token framing is applied inside `LancePipeline`,
so the generation paths run through the **existing generic** example scripts
with a plain `--model bytedance-research/Lance` and the single-stage deploy
config `vllm_omni/deploy/lance.yaml`.

## References

- Offline examples (reused, generic):
  [`text_to_image`](../../examples/offline_inference/text_to_image/text_to_image.py),
  [`text_to_video`](../../examples/offline_inference/text_to_video/text_to_video.py),
  [`image_to_image`](../../examples/offline_inference/image_to_image/image_edit.py)
- Online serving: reuses BAGEL's OpenAI chat client
  [`examples/online_serving/bagel/openai_chat_client.py`](../../examples/online_serving/bagel/openai_chat_client.py)
- E2E tests:
  [`tests/e2e/online_serving/test_lance.py`](../../tests/e2e/online_serving/test_lance.py)
- Deploy config:
  [`vllm_omni/deploy/lance.yaml`](../../vllm_omni/deploy/lance.yaml)
- HuggingFace model page:
  [bytedance-research/Lance](https://huggingface.co/bytedance-research/Lance)

The HF repo bundles everything (`Lance_3B/`, `Lance_3B_Video/`,
`Qwen2.5-VL-ViT/`, `Wan2.2_VAE.pth`); no separate downloads are required.
`video_edit` / `video2text` require `--model bytedance-research/Lance/Lance_3B_Video`
so the 3-D `latent_pos_embed` table is loaded; the other paths can point at
the top-level `bytedance-research/Lance` repo and resolve the right
sub-checkpoint automatically.

## Hardware Support

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: CUDA ≥ 12.4
- vLLM-Omni version: 0.18.1.dev

Lance-3B BF16 footprint is ~7 GB LLM + Qwen2.5-VL ViT + Wan2.2 VAE,
comfortably within one A100 80GB (also tested on B300). The single-stage
deploy config targets one GPU.

#### Command

```bash
# Text-to-image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model bytedance-research/Lance \
    --stage-configs-path vllm_omni/deploy/lance.yaml \
    --prompt "a corgi astronaut on the moon, cinematic" \
    --num-inference-steps 30 --output ./out

# Text-to-video (video checkpoint)
python examples/offline_inference/text_to_video/text_to_video.py \
    --model bytedance-research/Lance/Lance_3B_Video \
    --stage-configs-path vllm_omni/deploy/lance.yaml \
    --prompt "a cat playing piano, cinematic" \
    --num-frames 25 --height 480 --width 768 \
    --num-inference-steps 30 --output ./out/lance.mp4

# Image edit
python examples/offline_inference/image_to_image/image_edit.py \
    --model bytedance-research/Lance \
    --stage-configs-path vllm_omni/deploy/lance.yaml \
    --image photo.jpg --prompt "make it a watercolor painting" \
    --output ./out/edited.png
```

#### Verification

```bash
pytest -s -v tests/e2e/online_serving/test_lance.py \
    -m "advanced_model" --run-level "advanced_model"
```

#### Notes

- Lance applies its upstream `inference_lance.sh` defaults internally:
  timestep-shift 3.5, text CFG 4.0, 768×768.
- No `--deploy-config` needed for the generic offline scripts:
  `--stage-configs-path vllm_omni/deploy/lance.yaml` is auto-detected as a
  deploy config (it has `stages:` and no legacy `stage_args`).
- `video_edit` (video→video), `img2text` (image→text) and `video2text`
  (video→text) need video input and/or text output, which the existing
  generic offline scripts do not provide. They are exercised through online
  serving + the e2e tests rather than a dedicated offline script.
- Greedy decoding emits an immediate EOS for many understanding prompts;
  the understanding paths enable sampling by default.

## Online Serving

Lance serves all six modalities via the OpenAI-compatible
`/v1/chat/completions` API and **reuses BAGEL's chat client** (identical
message format and `modalities` / `num_inference_steps` / `seed` / `height`
/ `width` knobs).

### Launch

```bash
vllm serve bytedance-research/Lance --omni \
    --deploy-config vllm_omni/deploy/lance.yaml --port 8091
# For video_edit / text2video, serve the video checkpoint instead:
#   vllm serve bytedance-research/Lance/Lance_3B_Video --omni \
#       --deploy-config vllm_omni/deploy/lance.yaml --port 8091
```

### Send Requests

```bash
# Text-to-image
python examples/online_serving/bagel/openai_chat_client.py \
    --prompt "A cute corgi astronaut on the moon, cinematic" \
    --modality text2img --output corgi.png

# Image edit
python examples/online_serving/bagel/openai_chat_client.py \
    --prompt "Convert this into a vibrant cartoon-style illustration" \
    --modality img2img --image-url path/to/photo.png --output edited.png

# Image understanding
python examples/online_serving/bagel/openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text --image-url photo.jpg
```
