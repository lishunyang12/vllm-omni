# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone smoke test for ``_detect_explicit_cli_keys`` (PR #2383).

Exercises the parser-walking logic in
``vllm_omni/entrypoints/cli/serve.py::_detect_explicit_cli_keys`` directly,
without spinning up the full vLLM CLI. The unit tests in
``TestCLIExplicitPrecedence`` cover the same logic indirectly via
``_create_from_registry``; this script is a more targeted check on the argv
walker for cases where you want to verify the long-flag parsing in isolation.

Usage::

    python scripts/smoke_cli_explicit_keys.py

Exits 0 on success, 1 on the first assertion failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def main() -> int:
    failures: list[str] = []

    def check(label: str, expected, actual) -> None:
        if expected == actual:
            _ok(label)
        else:
            _fail(f"{label}\n        expected: {sorted(expected)}\n        actual:   {sorted(actual)}")
            failures.append(label)

    from vllm_omni.entrypoints.cli.serve import _detect_explicit_cli_keys

    print("=" * 72)
    print("Smoke testing _detect_explicit_cli_keys()")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Long flag with space-separated value
    # ------------------------------------------------------------------
    check(
        "long flag, space-separated value",
        {"max_num_seqs"},
        _detect_explicit_cli_keys(["--max-num-seqs", "8"]),
    )

    # ------------------------------------------------------------------
    # 2. Long flag with =-separated value
    # ------------------------------------------------------------------
    check(
        "long flag, =-separated value",
        {"gpu_memory_utilization"},
        _detect_explicit_cli_keys(["--gpu-memory-utilization=0.8"]),
    )

    # ------------------------------------------------------------------
    # 3. Multiple flags mixed
    # ------------------------------------------------------------------
    check(
        "multiple flags mixed (space + = + store_true)",
        {"max_num_seqs", "enforce_eager", "max_model_len"},
        _detect_explicit_cli_keys(["--max-num-seqs", "8", "--enforce-eager", "--max-model-len=4096"]),
    )

    # ------------------------------------------------------------------
    # 4. Empty argv
    # ------------------------------------------------------------------
    check(
        "empty argv → empty set",
        set(),
        _detect_explicit_cli_keys([]),
    )

    # ------------------------------------------------------------------
    # 5. Positional args (model name etc.) ignored
    # ------------------------------------------------------------------
    check(
        "positional args ignored, only --flags counted",
        {"omni"},
        _detect_explicit_cli_keys(["Qwen/Qwen3-Omni-30B-A3B-Instruct", "--omni"]),
    )

    # ------------------------------------------------------------------
    # 6. Hyphen → underscore conversion (this is the critical one for
    #    matching against argparse Namespace attribute names)
    # ------------------------------------------------------------------
    check(
        "hyphen → underscore conversion",
        {"stage_init_timeout"},
        _detect_explicit_cli_keys(["--stage-init-timeout", "1200"]),
    )

    # ------------------------------------------------------------------
    # 7. Multi-hyphen flag
    # ------------------------------------------------------------------
    check(
        "multi-hyphen flag converts to multi-underscore",
        {"max_num_batched_tokens"},
        _detect_explicit_cli_keys(["--max-num-batched-tokens=32768"]),
    )

    # ------------------------------------------------------------------
    # 8. Real-world serve invocation
    # ------------------------------------------------------------------
    check(
        "real-world serve invocation",
        {"omni", "port", "deploy_config", "max_num_seqs"},
        _detect_explicit_cli_keys(
            [
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "--omni",
                "--port",
                "8091",
                "--deploy-config",
                "vllm_omni/deploy/qwen3_omni_moe.yaml",
                "--max-num-seqs",
                "8",
            ]
        ),
    )

    # ------------------------------------------------------------------
    # 9. --stage-overrides JSON value (the value is a JSON blob, the
    #    KEY is what we record)
    # ------------------------------------------------------------------
    check(
        "--stage-overrides records the flag, not the JSON value",
        {"stage_overrides"},
        _detect_explicit_cli_keys(["--stage-overrides", '{"0": {"max_num_seqs": 2}}']),
    )

    # ------------------------------------------------------------------
    # 10. Standalone double-dash should not be recorded
    # ------------------------------------------------------------------
    check(
        "standalone -- (end-of-options marker) is not recorded",
        set(),
        _detect_explicit_cli_keys(["--"]),
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
    print("ALL OK — 10 checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
