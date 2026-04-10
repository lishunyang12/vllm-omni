# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone smoke test for the new deploy/pipeline config schema (PR #2383).

Loads the real production and CI deploy YAMLs from disk, exercises base_config
inheritance, the new platforms: deep-merge, and the pipeline registry merge.
Designed to run on a CPU-only Linux box without spinning up an engine.

Usage::

    python scripts/smoke_config_loading.py

Exits 0 on success, 1 on the first assertion failure. Each check prints a
``[OK]`` or ``[FAIL]`` line so reviewers can scan the output quickly.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make repo root importable when running directly from a checkout.
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    failures: list[str] = []

    def check(label: str, fn) -> None:
        try:
            fn()
            _ok(label)
        except AssertionError as e:
            _fail(f"{label}: {e}")
            failures.append(label)
        except Exception as e:
            _fail(f"{label}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failures.append(label)

    # Side-effect import: registers QWEN3_OMNI_PIPELINE
    import vllm_omni.model_executor.models.qwen3_omni.pipeline  # noqa: F401
    from vllm_omni.config.stage_config import (
        _PIPELINE_REGISTRY,
        _apply_platform_overrides,
        load_deploy_config,
        merge_pipeline_deploy,
    )

    prod_path = REPO_ROOT / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"
    ci_path = REPO_ROOT / "vllm_omni" / "deploy" / "ci" / "qwen3_omni_moe.yaml"

    print("=" * 72)
    print(f"Loading from: {REPO_ROOT}")
    print(f"  prod: {prod_path}")
    print(f"  ci:   {ci_path}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Production deploy YAML loads
    # ------------------------------------------------------------------
    def t_prod_loads():
        deploy = load_deploy_config(prod_path)
        assert len(deploy.stages) == 3, f"expected 3 stages, got {len(deploy.stages)}"
        assert deploy.async_chunk is True, "async_chunk should default to True in prod yaml"
        assert deploy.connectors is not None, "connectors block missing"
        assert "connector_of_shared_memory" in deploy.connectors, "shared-memory connector not registered"

    check("prod yaml loads with 3 stages + connectors + async_chunk", t_prod_loads)

    # ------------------------------------------------------------------
    # 2. Production stages have the expected CUDA defaults
    # ------------------------------------------------------------------
    def t_prod_cuda_defaults():
        deploy = load_deploy_config(prod_path)
        s0, s1, s2 = deploy.stages
        assert s0.gpu_memory_utilization == 0.9, f"stage 0 gpu_mem expected 0.9, got {s0.gpu_memory_utilization}"
        assert s0.devices == "0", f"stage 0 devices expected '0', got {s0.devices!r}"
        assert s1.gpu_memory_utilization == 0.6, f"stage 1 gpu_mem expected 0.6, got {s1.gpu_memory_utilization}"
        assert s2.max_num_seqs == 1, f"stage 2 max_num_seqs expected 1, got {s2.max_num_seqs}"
        assert s2.enforce_eager is True, f"stage 2 enforce_eager expected True, got {s2.enforce_eager}"

    check("prod cuda defaults: stage 0/1/2 fields match yaml", t_prod_cuda_defaults)

    # ------------------------------------------------------------------
    # 3. NPU platform override applies
    # ------------------------------------------------------------------
    def t_npu_overrides():
        deploy = load_deploy_config(prod_path)
        deploy = _apply_platform_overrides(deploy, platform="npu")
        s0 = deploy.stages[0]
        assert s0.tensor_parallel_size == 2, f"npu stage 0 tp expected 2, got {s0.tensor_parallel_size}"
        assert s0.devices == "0,1", f"npu stage 0 devices expected '0,1', got {s0.devices!r}"
        assert s0.gpu_memory_utilization == 0.6, "npu should override gpu_mem to 0.6"

    check("npu platform override layers tp=2 + devices='0,1' onto stage 0", t_npu_overrides)

    # ------------------------------------------------------------------
    # 4. XPU platform override applies
    # ------------------------------------------------------------------
    def t_xpu_overrides():
        deploy = load_deploy_config(prod_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")
        s0 = deploy.stages[0]
        assert s0.tensor_parallel_size == 4, f"xpu stage 0 tp expected 4, got {s0.tensor_parallel_size}"
        assert s0.devices == "0,1,2,3", f"xpu stage 0 devices expected '0,1,2,3', got {s0.devices!r}"
        assert s0.engine_extras.get("max_cudagraph_capture_size") == 0, (
            "xpu should set max_cudagraph_capture_size=0 via engine_extras"
        )

    check("xpu platform override layers tp=4 + devices='0,1,2,3' onto stage 0", t_xpu_overrides)

    # ------------------------------------------------------------------
    # 5. Unknown platform is a no-op
    # ------------------------------------------------------------------
    def t_unknown_noop():
        deploy = load_deploy_config(prod_path)
        original = deploy.stages[0].gpu_memory_utilization
        deploy = _apply_platform_overrides(deploy, platform="totally_made_up")
        assert deploy.stages[0].gpu_memory_utilization == original, "unknown platform should leave stage unchanged"

    check("unknown platform is a no-op", t_unknown_noop)

    # ------------------------------------------------------------------
    # 6. CI overlay inherits from prod via base_config:
    # ------------------------------------------------------------------
    def t_ci_inherits():
        deploy = load_deploy_config(ci_path)
        assert len(deploy.stages) == 3, f"ci should still have 3 stages, got {len(deploy.stages)}"
        s0 = deploy.stages[0]
        # CI overrides
        assert s0.max_num_seqs == 5, f"ci stage 0 max_num_seqs expected 5, got {s0.max_num_seqs}"
        assert s0.engine_extras.get("load_format") == "dummy", "ci should set load_format=dummy"
        # Inherited from prod base
        assert s0.gpu_memory_utilization == 0.9, "ci should inherit gpu_mem=0.9 from prod base"
        # Connectors inherited from prod
        assert deploy.connectors is not None, "ci should inherit connectors from base"
        assert "connector_of_shared_memory" in deploy.connectors

    check("ci overlay inherits prod connectors + applies ci stage overrides", t_ci_inherits)

    # ------------------------------------------------------------------
    # 7. CI sampling_params merge with base
    # ------------------------------------------------------------------
    def t_ci_sampling_merge():
        deploy = load_deploy_config(ci_path)
        s0_sp = deploy.stages[0].default_sampling_params
        # CI overrides max_tokens
        assert s0_sp["max_tokens"] == 150, f"ci stage 0 max_tokens expected 150, got {s0_sp.get('max_tokens')}"
        # Inherited from base
        assert s0_sp["temperature"] == 0.4, "ci should inherit temperature=0.4 from base"
        assert s0_sp["seed"] == 42, "ci should inherit seed=42 from base"

    check("ci sampling_params merges field-by-field with base", t_ci_sampling_merge)

    # ------------------------------------------------------------------
    # 8. CI + ROCm platform deep-merge
    #    base prod yaml has platforms.rocm.stages[0].enforce_eager=true
    #    ci yaml has platforms.rocm.stages[0].max_num_seqs=1
    #    after deep-merge BOTH should apply
    # ------------------------------------------------------------------
    def t_ci_rocm_deep_merge():
        deploy = load_deploy_config(ci_path)
        deploy = _apply_platform_overrides(deploy, platform="rocm")
        s0 = deploy.stages[0]
        assert s0.max_num_seqs == 1, f"deep-merge should apply ci's rocm max_num_seqs=1, got {s0.max_num_seqs}"
        assert s0.enforce_eager is True, (
            f"deep-merge should preserve base's rocm enforce_eager=True, got {s0.enforce_eager}"
        )

    check(
        "platforms: deep-merge layers ci rocm overrides onto base rocm overrides",
        t_ci_rocm_deep_merge,
    )

    # ------------------------------------------------------------------
    # 9. CI + XPU platform deep-merge
    # ------------------------------------------------------------------
    def t_ci_xpu_deep_merge():
        deploy = load_deploy_config(ci_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")
        s0 = deploy.stages[0]
        assert s0.tensor_parallel_size == 4, f"ci+xpu stage 0 should keep base xpu tp=4, got {s0.tensor_parallel_size}"
        assert s0.max_num_seqs == 1, f"ci+xpu stage 0 should apply ci xpu max_num_seqs=1, got {s0.max_num_seqs}"
        assert s0.devices == "0,1,2,3", f"ci+xpu devices expected '0,1,2,3', got {s0.devices!r}"

    check(
        "platforms: deep-merge layers ci xpu overrides onto base xpu overrides",
        t_ci_xpu_deep_merge,
    )

    # ------------------------------------------------------------------
    # 10. Pipeline registry has qwen3_omni_moe with the right shape
    # ------------------------------------------------------------------
    def t_pipeline_registry():
        assert "qwen3_omni_moe" in _PIPELINE_REGISTRY, "qwen3_omni_moe not in registry"
        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        assert len(pipeline.stages) == 3, f"expected 3 stages, got {len(pipeline.stages)}"
        errors = pipeline.validate()
        assert errors == [], f"pipeline has validation errors: {errors}"

    check("qwen3_omni_moe pipeline registered with 3 valid stages", t_pipeline_registry)

    # ------------------------------------------------------------------
    # 11. Full merge: pipeline + deploy → list[StageConfig]
    # ------------------------------------------------------------------
    def t_full_merge():
        deploy = load_deploy_config(prod_path)
        pipeline = _PIPELINE_REGISTRY["qwen3_omni_moe"]
        stages = merge_pipeline_deploy(pipeline, deploy)
        assert len(stages) == 3, f"merge produced {len(stages)} stages, expected 3"
        # Stage 0 (thinker) should pull deploy's gpu_mem AND pipeline's sampling_constraints
        s0 = stages[0]
        assert s0.yaml_engine_args.get("gpu_memory_utilization") == 0.9, (
            "merged stage 0 should have gpu_mem from deploy"
        )
        sp = s0.yaml_extras.get("default_sampling_params", {})
        assert sp.get("detokenize") is True, (
            "merged stage 0 should have detokenize=True from pipeline sampling_constraints"
        )
        # Deploy temperature still flows through
        assert sp.get("temperature") == 0.4, "merged stage 0 should have deploy temperature=0.4"

    check(
        "merge_pipeline_deploy combines deploy fields + pipeline sampling_constraints",
        t_full_merge,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 72)
    if failures:
        print(f"FAILED: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL OK — 11 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
