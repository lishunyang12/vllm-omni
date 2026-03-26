#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant Needle-in-a-Haystack benchmark.

Reproduces the evaluation from Blaizzy/mlx-vlm#858:
  - Multiple context lengths
  - Needle hidden at different depths
  - Measures: exact match, avg KV cache size, compression ratio

Approach: single forward pass per test, compare if the model's next-token
prediction contains the needle answer. No decode loop needed — if the model
"sees" the needle in the KV cache, the first tokens will reference it.

Usage:
    python tests/diffusion/quantization/bench_turboquant_needle.py
    python tests/diffusion/quantization/bench_turboquant_needle.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types

import torch

# ---------------------------------------------------------------------------
# Import TurboQuant
# ---------------------------------------------------------------------------
_tq = types.ModuleType("vllm_omni.quantization.turboquant")
sys.modules[_tq.__name__] = _tq
_spec = importlib.util.spec_from_file_location(
    _tq.__name__,
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "vllm_omni", "quantization", "turboquant.py",
    ),
)
_spec.loader.exec_module(_tq)
TurboQuantConfig = _tq.TurboQuantConfig
TurboQuantState = _tq.TurboQuantState


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

NEEDLE = "The secret project code name is AURORA-7749."
ANSWER_KEY = "AURORA-7749"

FILLER = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending "
    "reports, and projected revenue streams from various business units. "
    "The committee discussed infrastructure upgrades planned for the western "
    "regional offices and noted that maintenance schedules should be "
    "coordinated with the facilities management team. Several action items "
    "were assigned to team leads for follow-up before the next meeting.\n\n"
)


def build_haystack(tokenizer, target_tokens: int, needle_depth: float) -> str:
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = max(0, int(n_reps * needle_depth))
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Important Memo ---\n{NEEDLE}\n--- End Memo ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    question = "What is the secret project code name mentioned in the memo?"
    return (
        f"<|im_start|>user\n{haystack}\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\nThe secret project code name is"
    )


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_needle_test(
    model, tokenizer, target_tokens: int, needle_depth: float,
    kv_bits: int | None, device: torch.device,
) -> tuple[bool, int]:
    """Run one needle test. Returns (found, kv_cache_bytes)."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    prompt = build_haystack(tokenizer, target_tokens, needle_depth)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=target_tokens + 512,
    ).to(device)

    # Forward pass
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values

    if kv_bits is None:
        # Baseline: measure FP16 KV size, greedy decode 20 tokens
        kv_bytes = 0
        for i in range(num_layers):
            k, v = past_kv[i]
            kv_bytes += k.nelement() * k.element_size()
            kv_bytes += v.nelement() * v.element_size()

        # Greedy decode
        answer_tokens = []
        next_logits = outputs.logits[:, -1, :]
        for _ in range(20):
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            answer_tokens.append(next_id)
            out = model(input_ids=next_id, past_key_values=past_kv)
            next_logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

        answer = tokenizer.decode(
            torch.cat(answer_tokens, dim=-1)[0] if answer_tokens else torch.tensor([]),
            skip_special_tokens=True,
        )
        found = ANSWER_KEY in answer
        if not found:
            print(f"\n    [DEBUG full] generated: '{answer[:80]}'", end="")
        del past_kv
        return found, kv_bytes

    else:
        # TurboQuant: compress KV, decompress, replace in-place, decode
        config = TurboQuantConfig(bit_width=kv_bits, use_qjl=False)
        tq_bytes = 0

        # Compress each layer's KV, then decompress and write back
        # into the ORIGINAL cache to preserve all metadata
        for i in range(num_layers):
            k, v = past_kv[i]
            k_st = TurboQuantState(config, head_dim, i, device)
            v_st = TurboQuantState(config, head_dim, i + 10000, device)
            k_c = k_st.quantize(k)
            v_c = v_st.quantize(v)
            for t in k_c.values():
                if isinstance(t, torch.Tensor):
                    tq_bytes += t.nelement() * t.element_size()
            for t in v_c.values():
                if isinstance(t, torch.Tensor):
                    tq_bytes += t.nelement() * t.element_size()
            # Decompress and replace in the original cache in-place
            k_deq = k_st.dequantize(k_c)
            v_deq = v_st.dequantize(v_c)
            # Overwrite the cache tensors directly
            if hasattr(past_kv, 'key_cache'):
                past_kv.key_cache[i] = k_deq
                past_kv.value_cache[i] = v_deq
            elif hasattr(past_kv, '_data'):
                past_kv._data[i] = (k_deq, v_deq)

        # Greedy decode using the modified original cache
        answer_tokens = []
        next_logits = outputs.logits[:, -1, :]
        for _ in range(20):
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            answer_tokens.append(next_id)
            out = model(input_ids=next_id, past_key_values=past_kv)
            next_logits = out.logits[:, -1, :]
            past_kv = out.past_key_values

        answer = tokenizer.decode(
            torch.cat(answer_tokens, dim=-1)[0] if answer_tokens else torch.tensor([]),
            skip_special_tokens=True,
        )
        found = ANSWER_KEY in answer
        if not found:
            print(f"\n    [DEBUG TQ {kv_bits}-bit] generated: '{answer[:80]}'", end="")
        del past_kv
        return found, tq_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant Needle-in-a-Haystack")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    print(f"\nLoading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    cfg = model.config
    print(f"  layers={cfg.num_hidden_layers}, "
          f"kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}, "
          f"head_dim={cfg.hidden_size // cfg.num_attention_heads}")

    depths = [0.25, 0.5]
    total_tests = len(args.contexts) * len(depths)

    configs = [
        ("full", None),
        ("TurboQuant 2-bit", 2),
        ("TurboQuant 2.5", 2.5),
        ("TurboQuant 3-bit", 3),
        ("TurboQuant 3.5", 3.5),
        ("TurboQuant 4-bit", 4),
    ]

    print(f"\nModel: {args.model} (bf16)")
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")
    print(f"Context: {', '.join(f'{c // 1024}k' for c in args.contexts)}")
    print(f"Test: Needle-in-a-haystack style\n")

    results = {name: {"matches": 0, "bytes": []} for name, _ in configs}

    for ctx in args.contexts:
        for depth in depths:
            label = f"ctx={ctx}, depth={depth:.0%}"
            print(f"  {label}:", end="", flush=True)
            for name, bits in configs:
                torch.cuda.empty_cache()
                try:
                    found, cache_bytes = run_needle_test(
                        model, tokenizer, ctx, depth, bits, device,
                    )
                    results[name]["matches"] += int(found)
                    results[name]["bytes"].append(cache_bytes)
                    sym = "Y" if found else "N"
                except Exception as e:
                    sym = "E"
                    print(f"\n    {name} error: {e}", end="")
                print(f" [{name}: {sym}]", end="", flush=True)
            print()

    # Results table
    full_avg = sum(results["full"]["bytes"]) / max(len(results["full"]["bytes"]), 1)

    print(f"\n{'='*65}")
    print("Results\n")
    print(f"{'config':<22} {'exact match':>14} {'avg cache GB':>14} {'vs full':>14}")
    print("-" * 65)

    for name, _ in configs:
        r = results[name]
        m = r["matches"]
        if r["bytes"]:
            avg_b = sum(r["bytes"]) / len(r["bytes"])
            avg_gb = avg_b / 1024**3
            if name == "full":
                ratio = "1.0x"
            else:
                ratio = f"{full_avg / avg_b:.1f}x smaller"
        else:
            avg_gb = 0
            ratio = "N/A"
        print(f"{name:<22} {m}/{total_tests:>13} {avg_gb:>14.3f} {ratio:>14}")

    print()


if __name__ == "__main__":
    main()
