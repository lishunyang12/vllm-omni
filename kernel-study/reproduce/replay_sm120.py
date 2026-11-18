"""Replay the identical captured H3 tensors using the actual SM120 backend."""
import torch
import replay

assert torch.cuda.get_device_capability() == (12, 0)
original_trace = replay.compile_trace


def trace(q, k, v, options, mask, variant, generated):
    # Both native and explicit warp select the same recipe on this GPU.
    return original_trace(q, k, v, options, mask, 'warp', generated)


replay.compile_trace = trace
replay.main()
