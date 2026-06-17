# JoyVL-Interaction

> Proactive vision-first streaming interaction (speak / silence / delegate)

## Summary

- Vendor: JD.com
- Model: `ydydy/JoyAI-VL-Interaction-Preview` (Qwen3-VL derivative, ~8B)
- Task: Real-time streaming video interaction — the model decides each tick
  whether to speak, stay silent, or delegate a hard task to a background brain
- Mode: Online serving; an interaction orchestrator in front of an OpenAI-compatible model server
- Maintainer: Community

## When to use this recipe

Use this for a "present" assistant over a continuous video stream (monitoring
and alerting, live commentary, real-time counting/translation, long-horizon
recall) rather than turn-based VQA. The model is a standard Qwen3-VL
autoregressive VLM, so it serves day-0 with `vllm serve`; the proactive behavior
comes from the control tokens `</silence>` / `</response>` / `<delegation>` and
the orchestration around them.

## References

- Paper: JoyAI-VL-Interaction (arXiv 2606.14777)
- Upstream: https://github.com/jd-opensource/JoyAI-VL-Interaction
- Example: [`examples/online_serving/joyvl_interaction/`](../../examples/online_serving/joyvl_interaction/README.md)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: from your current checkout

## GPU

### 1x GPU (>= 40GB)

#### Command

```bash
cd examples/online_serving/joyvl_interaction

# 1. model server — plain VLM, NOT --omni
bash start_model.sh                 # :8061

# 2. interaction orchestrator
bash start_interaction.sh           # :8070

# 3. demo
python gradio_demo.py --server http://127.0.0.1:8070
```

#### Verification

```bash
curl http://127.0.0.1:8070/health
python run_cli_demo.py sample.mp4 --query "Alert me if a fire breaks out"
```

#### Notes

- Serve the model **without `--omni`** — it is a standard Qwen3-VL VLM, not an
  omni/diffusion model; `--omni` routes it through the diffusion registry and fails.
- Serve with `skip_special_tokens=False` so the control tokens survive; the
  orchestrator sets this per request.
- Memory reuses the main model as its summarizer by default; point it at a
  dedicated Qwen3-VL-4B server for production, or disable with `NO_MEMORY=1`.
- Delegation is stubbed; speech (ASR/TTS) is external and pluggable.
