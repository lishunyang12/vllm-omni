# PR 3079 HunyuanVideo-1.5 Trim Padding Benchmark

Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`

Runtime:

- GPU: `CUDA_VISIBLE_DEVICES=3`
- Attention backend: `DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN`
- Shape: `height=480`, `width=832`, `num_frames=33`, `num_inference_steps=50`
- Prompt: `A cat walks on the grass, realistic style.`
- Seed: `42`

## Results

| Variant | Total time | Forward time | Video |
| --- | ---: | ---: | --- |
| Baseline | 27.4157s | 27.0295s | `hv15_cudnn_baseline_50step_33f.mp4` |
| Trim encoder padding | 16.8355s | 16.4482s | `hv15_cudnn_trim_padding_50step_33f.mp4` |

Speedup:

- Total time: 38.6%
- Forward time: 39.1%

## FFmpeg Metrics

Optimized vs baseline:

- SSIM All: 0.694584
- PSNR average: 22.200918

Baseline self-run reference:

- SSIM All: 0.702010
- PSNR average: 22.064819

The optimized-vs-baseline metrics are within the observed CUDNN baseline
self-run variance for this setup.

## Reproduction

Baseline command before this patch:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=$PWD \
DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN \
/home/zjy/code/david/.venv/bin/python \
examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt 'A cat walks on the grass, realistic style.' \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --fps 24 \
  --seed 42 \
  --output bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_baseline_50step_33f.mp4 \
  --enable-diffusion-pipeline-profiler
```

Optimized command after this patch:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=$PWD \
DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN \
/home/zjy/code/david/.venv/bin/python \
examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt 'A cat walks on the grass, realistic style.' \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --fps 24 \
  --seed 42 \
  --output bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_trim_padding_50step_33f.mp4 \
  --enable-diffusion-pipeline-profiler
```

Metrics:

```bash
ffmpeg -hide_banner \
  -i bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_baseline_50step_33f.mp4 \
  -i bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_trim_padding_50step_33f.mp4 \
  -lavfi "[0:v][1:v]ssim" -f null -

ffmpeg -hide_banner \
  -i bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_baseline_50step_33f.mp4 \
  -i bench_outputs/pr3079_hv15_trim_padding_opt/hv15_cudnn_trim_padding_50step_33f.mp4 \
  -lavfi "[0:v][1:v]psnr" -f null -
```
