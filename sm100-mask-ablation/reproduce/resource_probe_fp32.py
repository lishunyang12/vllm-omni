"""One untimed real-input forward: fixed FP32 pooling4/1, fresh CuTe artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
from sol_attn import interface, preprocess, sol_attn

POOL_HASHES = {
    "preprocess.py": "09039b41d8935b2b0df8e6cd17da73dd1d887f7144511f28e28755490ee2c297",
    "triton_ref/preprocess.py": "59a9422fd9db974bff2984eb60ecd93de55a3a14a6bb2fd6b93833cc4ed22953",
}


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = args.source_root.resolve() / "techniques/sparse_backends/sol_attn"
    assert Path(interface.__file__).resolve().parent == package
    assert os.environ.get("CUTE_DSL_DISABLE_FILE_CACHING") == "1"
    manifest = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
    }
    for name, expected in POOL_HASHES.items():
        assert manifest[name] == expected, f"Unexpected shared pooling source: {name}"
    for kernel in (preprocess._reduce_kc_kernel, preprocess._reduce_vc_kernel):
        kernel.configs = [triton.Config({}, num_warps=4, num_stages=1)]
        kernel.cache.clear()
    native_prepare = preprocess.prepare
    retained = []

    def capture_prepare(*values, **options):
        prepared = native_prepare(*values, **options)
        retained.append(prepared)
        return prepared

    preprocess.prepare = capture_prepare
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    for name, expected in payload["tensor_sha256"].items():
        assert digest(payload[name]) == expected, name
    with torch.inference_mode():
        qkv = [payload[name][:, :, :28].contiguous().cuda() for name in ("q", "k", "v")]
        assert all(value.shape[2] == 28 for value in qkv)
        mask = payload["exact_block_mask"]
        mask = None if mask is None else mask.cuda().contiguous()
        inputs = qkv + ([] if mask is None else [mask])
        before = [digest(value) for value in inputs]
        result = sol_attn(*qkv, **payload["options"], exact_block_mask=mask)
        torch.cuda.synchronize()
        assert len(retained) == 1
        assert bool(torch.isfinite(result).all())
        assert before == [digest(value) for value in inputs]
        prepared = {
            name: dict(sha256=digest(value), shape=list(value.shape), dtype=str(value.dtype))
            for name, value in zip(("kc", "vc", "threshold"), retained[0])
        }
        assert all(bool(torch.isfinite(value).all()) for value in retained[0])
    report = dict(
        input=str(args.input.resolve()),
        shape=list(qkv[0].shape),
        options=payload["options"],
        source_root=str(args.source_root.resolve()),
        source_sha256=manifest,
        input_sha256=before,
        prepared=prepared,
        output_sha256=digest(result),
        pooling_config={
            name: str(getattr(preprocess, name).best_config) for name in ("_reduce_kc_kernel", "_reduce_vc_kernel")
        },
        torch=torch.__version__,
        triton=triton.__version__,
        cuda=torch.version.cuda,
        capability=list(torch.cuda.get_device_capability()),
        gpu=torch.cuda.get_device_name(),
        environment={
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_STEP_ID", "CUTE_DSL_DUMP_DIR")
        },
        timing_samples=0,
        forward_calls=1,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
