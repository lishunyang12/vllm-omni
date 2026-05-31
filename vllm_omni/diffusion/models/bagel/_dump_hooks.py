"""Layer-by-layer dump hooks for vllm-omni Lance/Bagel — env-var gated.

Set ``LANCE_DUMP_DIR`` before launching to enable.  Captures the same
checkpoint set as ``lance-upstream/lance_compare/dump_upstream.py`` so
``lance_compare/compare.py`` can diff the two runs.

Production code is unaffected when the env var is unset — the only
runtime cost is a single ``os.environ.get`` check at model init.
"""

from __future__ import annotations

import os
import pathlib

import torch

_INSTALLED = False
_DUMP_DIR: pathlib.Path | None = None


def _save(name: str, t):
    if _DUMP_DIR is None:
        return
    try:
        if torch.is_tensor(t):
            torch.save(t.detach().cpu(), _DUMP_DIR / name)
        else:
            torch.save(t, _DUMP_DIR / name)
        shape = tuple(t.shape) if torch.is_tensor(t) else "-"
        dtype = t.dtype if torch.is_tensor(t) else "-"
        print(f"[dump_omni] saved {name} shape={shape} dtype={dtype}", flush=True)
    except Exception as e:
        print(f"[dump_omni] save {name} failed: {e}", flush=True)


def install_if_env(bagel_module) -> None:
    """Install dump hooks on a freshly-built Bagel/LanceBagel model."""
    global _INSTALLED, _DUMP_DIR
    raw = os.environ.get("LANCE_DUMP_DIR")
    if not raw:
        return
    if _INSTALLED:
        return  # only install once per process
    _DUMP_DIR = pathlib.Path(raw)
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[dump_omni] writing to {_DUMP_DIR}", flush=True)

    step = {"i": 0}
    vt_call = {"last_step": -1, "sub_call": 1, "n": 0}
    prestack = {"vae2llm": False, "time_embed": False, "pos_embed": False, "iter0_active": False}
    state_ref = {"last_noise_shape": None}

    def _reset_all():
        step["i"] = 0
        vt_call["last_step"] = -1
        vt_call["sub_call"] = 1
        vt_call["n"] = 0
        prestack["vae2llm"] = False
        prestack["time_embed"] = False
        prestack["pos_embed"] = False
        prestack["iter0_active"] = False
        if "_layer_state" in state_ref:
            for k in list(state_ref["_layer_state"].keys()):
                state_ref["_layer_state"][k] = 0

    def _is_t_one(t):
        try:
            return abs(float(t.max().item()) - 1.0) < 1e-4
        except Exception:
            return False

    # ---- vae2llm: CK1 + xt-per-step + prestack ----
    if hasattr(bagel_module, "vae2llm"):

        def vae2llm_pre(_m, inputs):
            x_t = inputs[0]
            shape = tuple(x_t.shape)
            # Detect shape change → new request (resets state from dummy_run).
            if state_ref["last_noise_shape"] is not None and shape != state_ref["last_noise_shape"]:
                old = state_ref["last_noise_shape"]
                print(f"[dump_omni] noise shape changed {old} -> {shape}; resetting state", flush=True)
                _reset_all()
            state_ref["last_noise_shape"] = shape
            i = step["i"]
            if i == 0:
                _save("CK1_initial_noise.pt", x_t)
            if i >= 2:
                _save(f"xt_after_step{i - 2:02d}.pt", x_t)
            step["i"] += 1

        def vae2llm_post(_m, inputs, outputs):
            if prestack["iter0_active"] and not prestack["vae2llm"]:
                _save("upstream_step0_vae2llm_in.pt", inputs[0])
                _save("upstream_step0_vae2llm_out.pt", outputs)
                prestack["vae2llm"] = True
                prestack["iter0_active"] = False

        bagel_module.vae2llm.register_forward_pre_hook(vae2llm_pre)
        bagel_module.vae2llm.register_forward_hook(vae2llm_post)

    # ---- time_embedder ----
    if hasattr(bagel_module, "time_embedder"):

        def time_pre(_m, inputs):
            if prestack["time_embed"]:
                return
            if _is_t_one(inputs[0]):
                _save("upstream_step0_time_embed_in.pt", inputs[0])
                prestack["iter0_active"] = True

        def time_post(_m, inputs, outputs):
            if prestack["iter0_active"] and not prestack["time_embed"]:
                _save("upstream_step0_time_embed_out.pt", outputs)
                prestack["time_embed"] = True

        bagel_module.time_embedder.register_forward_pre_hook(time_pre)
        bagel_module.time_embedder.register_forward_hook(time_post)

    # ---- latent_pos_embed ----
    if hasattr(bagel_module, "latent_pos_embed"):

        def pos_pre(_m, inputs):
            if not prestack["pos_embed"]:
                _save("upstream_step0_pos_embed_in_ids.pt", inputs[0])

        def pos_post(_m, inputs, outputs):
            if not prestack["pos_embed"]:
                _save("upstream_step0_pos_embed_out.pt", outputs)
                prestack["pos_embed"] = True

        bagel_module.latent_pos_embed.register_forward_pre_hook(pos_pre)
        bagel_module.latent_pos_embed.register_forward_hook(pos_post)

    # ---- llm2vae: v_t per step ----
    if hasattr(bagel_module, "llm2vae"):

        def llm2vae_post(_m, _inputs, outputs):
            cur = step["i"] - 1
            if 0 <= cur < 30:
                if vt_call["last_step"] != cur:
                    _save(f"vt_step{cur:02d}_cond.pt", outputs)
                    vt_call["last_step"] = cur
                    vt_call["sub_call"] = 1
                else:
                    sub = vt_call["sub_call"]
                    tag = "cfg_text" if sub == 1 else "cfg_img"
                    _save(f"vt_step{cur:02d}_{tag}.pt", outputs)
                    vt_call["sub_call"] = sub + 1
            vt_call["n"] += 1

        bagel_module.llm2vae.register_forward_hook(llm2vae_post)

    # ---- ViT tower + connector ----
    vit_tower_n = {"n": 0}
    if hasattr(bagel_module, "vit_model"):

        def vit_pre(_m, args, kwargs):
            try:
                hs = kwargs.get("hidden_states") if "hidden_states" in kwargs else (args[0] if args else None)
                gt = kwargs.get("grid_thw") if "grid_thw" in kwargs else (args[1] if len(args) > 1 else None)
                n = vit_tower_n["n"]
                if torch.is_tensor(hs):
                    _save(f"upstream_vit_tower_in_call{n:02d}.pt", hs)
                if torch.is_tensor(gt):
                    _save(f"upstream_vit_tower_grid_thw_call{n:02d}.pt", gt)
            except Exception:
                pass

        def vit_post(_m, _inputs, outputs):
            try:
                if torch.is_tensor(outputs):
                    n = vit_tower_n["n"]
                    _save(f"upstream_vit_tower_out_call{n:02d}.pt", outputs)
                    vit_tower_n["n"] += 1
            except Exception:
                pass

        bagel_module.vit_model.register_forward_pre_hook(vit_pre, with_kwargs=True)
        bagel_module.vit_model.register_forward_hook(vit_post)

    if hasattr(bagel_module, "connector"):
        con_n = {"n": 0}

        def con_post(_m, _inputs, outputs):
            try:
                if torch.is_tensor(outputs):
                    n = con_n["n"]
                    _save(f"upstream_vit_connector_out_call{n:02d}.pt", outputs)
                    con_n["n"] += 1
            except Exception:
                pass

        bagel_module.connector.register_forward_hook(con_post)

    # ---- per-layer Qwen2 stack ----
    try:
        layers = bagel_module.language_model.model.layers
    except AttributeError:
        layers = None

    if layers is not None:
        nlayers = len(layers)
        layer_state = {i: 0 for i in range(nlayers)}
        state_ref["_layer_state"] = layer_state
        MAX_CAP = 6
        TAGS = ["a", "b", "c", "d", "e", "f"]

        def _make_layer_hook(idx):
            def hook(_m, inputs, kwargs, outputs):
                if not prestack.get("vae2llm"):
                    return
                s = layer_state[idx]
                if s >= MAX_CAP:
                    return
                h_in = inputs[0] if inputs else None
                if h_in is None:
                    for k in ("packed_query_sequence", "hidden_states", "x"):
                        if k in kwargs:
                            h_in = kwargs[k]
                            break
                if (
                    not torch.is_tensor(h_in)
                    or h_in.shape[0] < 1024
                    and h_in.dim() >= 2
                    and (h_in.shape[1] < 1024 if h_in.dim() > 1 else True)
                ):
                    return
                # Accept (S, H) or (B, S, H) — strip leading batch if present.
                if h_in.dim() == 3 and h_in.shape[0] == 1:
                    h_in_save = h_in[0]
                else:
                    h_in_save = h_in
                h_out = outputs[0] if isinstance(outputs, tuple) else outputs
                if torch.is_tensor(h_out) and h_out.dim() == 3 and h_out.shape[0] == 1:
                    h_out = h_out[0]
                tag = TAGS[s]
                _save(f"layer{idx:02d}_{tag}_in_step0.pt", h_in_save)
                _save(f"layer{idx:02d}_{tag}_out_step0.pt", h_out)
                if s == 0:
                    _save(f"layer{idx:02d}_in_step0.pt", h_in_save)
                    _save(f"layer{idx:02d}_out_step0.pt", h_out)
                layer_state[idx] = s + 1

            return hook

        for idx in range(nlayers):
            layers[idx].register_forward_hook(_make_layer_hook(idx), with_kwargs=True)

    # ---- CK7: final x_t before VAE decode ----
    # The Lance pipeline hands the final latent to wan_vae.decode; hook the
    # vae's decode method instead since it lives on a different module.
    try:
        from .. import lance as _lance_pkg  # noqa: F401
    except Exception:
        _lance_pkg = None

    # Walk the parent pipeline (if known) — fall back to monkey-patching the
    # WanVAE class globally.  Easier: patch WanVAE.decode at import time so
    # whichever instance gets used, we capture its input.
    try:
        from .wan_vae import LanceWanVAE  # type: ignore
    except Exception:
        try:
            from ..lance.wan_vae import LanceWanVAE  # type: ignore
        except Exception:
            LanceWanVAE = None  # type: ignore

    if LanceWanVAE is not None:
        _orig_decode = LanceWanVAE.decode

        def _patched_decode(self, latents, *a, **k):
            try:
                if torch.is_tensor(latents):
                    _save("CK7_final_xt_before_vae.pt", latents)
                elif isinstance(latents, list) and latents:
                    _save("CK7_final_xt_before_vae.pt", latents[0])
            except Exception:
                pass
            return _orig_decode(self, latents, *a, **k)

        LanceWanVAE.decode = _patched_decode

    _INSTALLED = True
    print(f"[dump_omni] hooks installed (nlayers={len(layers) if layers is not None else 0})", flush=True)
