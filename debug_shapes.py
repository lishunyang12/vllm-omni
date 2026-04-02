"""Debug script: check actual Q/K/V tensor shapes during HunyuanVideo inference.
Patches Attention.forward to log shapes on first call, then runs a short generation."""
import torch

# Monkey-patch Attention.forward to log actual tensor shapes
_shape_log = []
_logged_count = 0
_max_log = 10  # only log first 10 unique shapes


def _patch_attention():
    from vllm_omni.diffusion.attention.layer import Attention
    _orig_forward = Attention.forward

    def _logging_forward(self, query, key, value, attn_metadata=None):
        global _logged_count
        if _logged_count < _max_log:
            shape_key = (tuple(query.shape), tuple(key.shape))
            if shape_key not in [s[0:2] for s in _shape_log]:
                has_mask = (attn_metadata is not None and
                            attn_metadata.attn_mask is not None)
                mask_shape = (list(attn_metadata.attn_mask.shape)
                              if has_mask else None)
                mask_false = 0
                if has_mask:
                    mask_false = int((~attn_metadata.attn_mask).sum().item())

                entry = (
                    tuple(query.shape),
                    tuple(key.shape),
                    query.dtype,
                    mask_shape,
                    mask_false,
                )
                _shape_log.append(entry)
                _logged_count += 1
                B, S, H, D = query.shape
                print(f"[SHAPE] q={list(query.shape)} k={list(key.shape)} "
                      f"dtype={query.dtype} "
                      f"tokens={S} heads={H} headdim={D} "
                      f"mask={mask_shape} mask_false={mask_false}")
        return _orig_forward(self, query, key, value, attn_metadata)

    Attention.forward = _logging_forward
    print("[DEBUG] Attention.forward patched for shape logging")


_patch_attention()


def main():
    print("\n" + "=" * 60)
    print("Theoretical token counts for HunyuanVideo 1.5")
    print("=" * 60)
    for n_frames in [33, 61, 81, 121]:
        t = (n_frames - 1) // 4 + 1
        h = 480 // 8
        w = 832 // 8
        vae_tokens = t * h * w
        h_p = h // 2
        w_p = w // 2
        patch_tokens = t * h_p * w_p
        print(f"  {n_frames}f: VAE latent {t}x{h}x{w}={vae_tokens}, "
              f"after patch {t}x{h_p}x{w_p}={patch_tokens} tokens")

    print("\n" + "=" * 60)
    print("Running short generation to capture actual shapes...")
    print("=" * 60)

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.platforms import current_omni_platform

    model = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(42)

    omni = Omni(
        model=model,
        vae_use_tiling=True,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(),
    )

    print("\n[DEBUG] Starting generation (2 steps only)...")
    try:
        outputs = omni.generate(
            {"prompt": "A cat in a garden."},
            OmniDiffusionSamplingParams(
                height=480,
                width=832,
                num_frames=121,
                generator=generator,
                guidance_scale=6.0,
                num_inference_steps=2,
            ),
        )
        print("[DEBUG] Generation completed.")
    except Exception as e:
        print(f"[DEBUG] Generation error (may be expected): {e}")

    print("\n" + "=" * 60)
    print("Shape summary")
    print("=" * 60)
    for entry in _shape_log:
        q_shape, k_shape, dtype, mask_shape, mask_false = entry
        B, S, H, D = q_shape
        print(f"  q={list(q_shape)} k={list(k_shape)} "
              f"tokens={S} mask_false={mask_false}")


if __name__ == "__main__":
    main()
