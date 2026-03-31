"""Reproduce online serving config path without starting a server."""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm_omni import Omni

model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
print(f"Initializing Omni with {model} (no config path, auto-detect)...")

try:
    omni = Omni(model=model)
    print(f"SUCCESS: {omni.engine.num_stages} stages initialized")
    for i, cfg in enumerate(omni.engine.stage_configs):
        devices = getattr(cfg, "runtime", {})
        if hasattr(devices, "devices"):
            devices = devices.devices
        else:
            devices = "?"
        print(f"  Stage {i}: devices={devices}")
    omni.shutdown()
except Exception as e:
    print(f"FAILED: {e}")
