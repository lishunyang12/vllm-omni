# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CLI Argument Classification
===========================

vLLM-Omni's CLI flags live in three buckets:

    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
    │ OrchestratorArgs │    │  OmniEngineArgs  │    │  (upstream vllm) │
    │                  │    │                  │    │    server/api    │
    │  stage_timeout   │    │  max_num_seqs    │    │  host, port      │
    │  worker_backend  │    │  gpu_mem_util    │    │  ssl_keyfile     │
    │  deploy_config   │    │  dtype, quant    │    │  api_key         │
    │     ...          │    │     ...          │    │     ...          │
    └──────────────────┘    └──────────────────┘    └──────────────────┘
            │                        │                        │
            ▼                        ▼                        ▼
       orchestrator              each stage               uvicorn /
       consumes                  engine                   FastAPI

Fields in ``SHARED_FIELDS`` (e.g. ``model``, ``log_stats``) flow to BOTH
orchestrator and engine by design.

Invariants enforced by ``tests/test_arg_routing.py``:

  1. ``OrchestratorArgs`` ∩ ``OmniEngineArgs`` ⊆ ``SHARED_FIELDS``
  2. Every CLI flag is classifiable into one of the three buckets
  3. User-typed flags that match none of the above are logged as dropped

Adding a new orchestrator-only flag → add a field to ``OrchestratorArgs``.
Everything else is automatic.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class OrchestratorArgs:
    """CLI flags consumed by the orchestrator.

    Contract: every field here is either
      (a) orchestrator-only (never needed by a stage engine), OR
      (b) orchestrator-read-then-redistributed (e.g. ``async_chunk`` is read
          from CLI, written to ``DeployConfig``, then propagated to every
          stage via ``merge_pipeline_deploy`` — not via direct kwargs
          forwarding).

    Fields that BOTH orchestrator and engine genuinely need (e.g. ``model``,
    ``log_stats``) should be listed in ``SHARED_FIELDS`` below; ``split_kwargs``
    will copy them to both buckets.
    """

    # === Lifecycle ===
    stage_init_timeout: int = 300
    init_timeout: int = 600

    # === Cross-stage Communication ===
    shm_threshold_bytes: int = 65536
    batch_timeout: int = 10

    # === Cluster / Backend ===
    worker_backend: str = "multi_process"
    ray_address: str | None = None

    # === Config Files ===
    stage_configs_path: str | None = None
    deploy_config: str | None = None
    stage_overrides: str | None = None  # raw JSON string; parsed downstream

    # === Mode Switches (orchestrator reads, DeployConfig redistributes) ===
    async_chunk: bool | None = None

    # === Observability ===
    log_stats: bool = False

    # === Headless Mode (also forwarded to engine — see SHARED_FIELDS) ===
    stage_id: int | None = None

    # === Pre-built Objects ===
    parallel_config: Any = None

    # === Multi-stage guards ===
    # --tokenizer is captured here so it does not propagate to every stage
    # uniformly (different stages often need different tokenizers, e.g.
    # qwen3_omni thinker vs talker). Users wanting a per-stage tokenizer
    # should set it in the deploy YAML.
    tokenizer: str | None = None


# Fields that live in BOTH OrchestratorArgs and OmniEngineArgs by design.
# Changes to this set are a review red flag — revisit the contract.
SHARED_FIELDS: frozenset[str] = frozenset(
    {
        "model",  # orch: detect model_type; engine: load weights
        "stage_id",  # orch: route (headless); engine: identity
        "log_stats",  # both want the flag
        "stage_configs_path",  # orch: load legacy YAML; engine: may reference for validation
    }
)


def orchestrator_field_names() -> frozenset[str]:
    """Return the names of every field on OrchestratorArgs."""
    return frozenset(f.name for f in fields(OrchestratorArgs))


def internal_blacklist_keys() -> frozenset[str]:
    """Return the set of CLI keys that must never be forwarded as per-stage
    engine overrides.

    This is the single source of truth for the old
    ``INTERNAL_STAGE_OVERRIDE_KEYS`` frozenset. Derived from OrchestratorArgs
    fields minus SHARED_FIELDS, so adding a new orchestrator-owned flag is a
    one-line change to the dataclass — this function updates automatically.
    """
    return orchestrator_field_names() - SHARED_FIELDS


def split_kwargs(
    kwargs: dict[str, Any],
    *,
    engine_cls: type | None = None,
    user_typed: set[str] | None = None,
    strict: bool = False,
) -> tuple[OrchestratorArgs, dict[str, Any]]:
    """Partition CLI kwargs into (orchestrator, engine) buckets.

    Args:
        kwargs: Raw dict, typically ``vars(args)``.
        engine_cls: Engine dataclass used to whitelist-filter the engine
            bucket. Defaults to ``OmniEngineArgs`` (imported lazily to avoid
            circular deps). Pass a custom class for testing.
        user_typed: Keys the user actually typed on the command line. Used
            to warn when a user-typed flag is unclassifiable.
        strict: If True, raise ``ValueError`` on ambiguous (double-classified
            but not in ``SHARED_FIELDS``) fields. Default False to keep the
            rollout non-breaking; flip to True in tests and CI.

    Returns:
        ``(orchestrator_args, engine_kwargs)``. ``engine_kwargs`` has already
        been whitelist-filtered against ``engine_cls`` — safe to pass directly
        to ``engine_cls(**engine_kwargs)``.
    """
    if engine_cls is None:
        # Lazy import to avoid circular dependency at module load time.
        from vllm_omni.engine.arg_utils import OmniEngineArgs

        engine_cls = OmniEngineArgs

    orch_fields = orchestrator_field_names()
    engine_fields = {f.name for f in fields(engine_cls)}

    orch_kwargs: dict[str, Any] = {}
    engine_candidate: dict[str, Any] = {}
    shared_values: dict[str, Any] = {}
    unclassified: dict[str, Any] = {}

    for key, value in kwargs.items():
        in_orch = key in orch_fields
        in_engine = key in engine_fields
        is_shared = key in SHARED_FIELDS

        if is_shared:
            shared_values[key] = value
        elif in_orch and in_engine:
            # Declared in both but not marked shared → ambiguous.
            msg = (
                f"Field {key!r} is defined on both OrchestratorArgs and "
                f"{engine_cls.__name__} but is not in SHARED_FIELDS. "
                f"This causes double-routing. Either remove the duplicate or "
                f"add {key!r} to SHARED_FIELDS if the sharing is intentional."
            )
            if strict:
                raise ValueError(msg)
            logger.error(msg)
            # Default: treat as orchestrator-only to preserve existing behavior.
            orch_kwargs[key] = value
        elif in_orch:
            orch_kwargs[key] = value
        elif in_engine:
            engine_candidate[key] = value
        else:
            unclassified[key] = value

    # Warn on user-typed but unclassifiable flags so we don't silently drop
    # something the user cared about (fixes the class of bug that spawned #873).
    if unclassified and user_typed:
        user_typed_unknown = sorted(k for k in unclassified if k in user_typed)
        if user_typed_unknown:
            logger.warning(
                "CLI flags not consumed by vllm-omni and dropped before "
                "per-stage engine construction: %s. If these are vllm "
                "frontend/uvicorn flags (host, port, ssl_*, api_key, …) this "
                "is expected; otherwise check your spelling.",
                user_typed_unknown,
            )

    # Engine bucket: shared + engine-only. We do NOT pass through unclassified
    # fields — that's exactly the server/uvicorn noise we want to shed.
    engine_kwargs = {**shared_values, **engine_candidate}

    # Construct the orchestrator dataclass. Shared fields that OrchestratorArgs
    # also declares get copied into its constructor.
    orch_init: dict[str, Any] = dict(orch_kwargs)
    for key, value in shared_values.items():
        if key in orch_fields:
            orch_init[key] = value
    orch_args = OrchestratorArgs(**orch_init)

    return orch_args, engine_kwargs


# ============================================================================
# Upstream server-dest derivation and argparse interop helpers.
# ============================================================================


def derive_server_dests_from_vllm_parser() -> frozenset[str]:
    """Derive the set of argparse dests that belong to vllm's frontend/server.

    Returns every dest registered by ``make_arg_parser`` that is NOT a field
    of ``OmniEngineArgs`` and NOT a field of ``OrchestratorArgs``. Useful for
    CI tests to assert all CLI flags are classifiable without maintaining
    a hardcoded server list.

    Returns empty frozenset if vllm's parser cannot be built (e.g. in a
    minimal test environment).
    """
    try:
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        logger.debug("Cannot import vllm parser — server-dest derivation skipped")
        return frozenset()

    try:
        parser = make_arg_parser(FlexibleArgumentParser())
        all_dests = {a.dest for a in parser._actions if a.dest and a.dest != "help"}
    except Exception as exc:
        logger.debug("Failed to build vllm parser: %s", exc)
        return frozenset()

    from vllm_omni.engine.arg_utils import OmniEngineArgs

    engine_fields = {f.name for f in fields(OmniEngineArgs)}
    orch_fields = orchestrator_field_names()

    return frozenset(all_dests - engine_fields - orch_fields - SHARED_FIELDS)


def orchestrator_args_from_argparse(args: Any) -> OrchestratorArgs:
    """Build an ``OrchestratorArgs`` from an ``argparse.Namespace``.

    Only copies attributes that exist on the namespace — missing fields fall
    back to the dataclass default. Useful when the full parser is already
    built and ``vars(args)`` would include noise.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(OrchestratorArgs):
        if hasattr(args, f.name):
            value = getattr(args, f.name)
            if value is not None or f.default is None:
                kwargs[f.name] = value
    return OrchestratorArgs(**kwargs)
