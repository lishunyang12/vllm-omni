# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer application of a skip-softmax calibration, matching TensorRT-LLM's
visual-gen sparse attention (``_layer_pattern_match_names`` / ``_is_disabled``).

Each attention layer is resolved by its FULL module name (e.g.
``transformer_2.blocks.5.attn1``), which serves two purposes at once:
  * ignore matching -- fnmatch the name (and a transformer-prefix-stripped variant,
    so relative ignore patterns like ``blocks.5.attn2`` match) against the ignore list;
  * expert routing -- the ``transformer.`` / ``transformer_2.`` prefix selects which
    per-expert calibration (a, b) applies (Wan A14B carries one per expert).

This is the single place the fragile name-matching lives, so it is unit-testable in
isolation (it was the source of the silent per-expert no-op bug).
"""

import fnmatch

_EXPERT_PREFIXES = ("transformer.", "transformer_2.")


def layer_match_names(module_name: str) -> tuple[str, ...]:
    """Full name plus transformer-prefix-stripped variants, for pattern matching.

    Mirrors TRT-LLM: relative ignore patterns (``blocks.5.attn2``) match a full name
    (``transformer.blocks.5.attn2``) via the stripped variant.
    """
    names = {module_name, module_name.replace("._orig_mod.", ".")}
    for name in tuple(names):
        for pfx in _EXPERT_PREFIXES:
            if name.startswith(pfx):
                names.add(name[len(pfx):])
    return tuple(names)


def is_ignored(module_name: str, ignore_patterns) -> bool:
    """True if this layer is in the calibration's ignore set (stays dense).

    Ancestor-aware: ModelOpt calibrates at the DiT attention-module granularity
    (``blocks.5.attn1`` = a ``WanAttention``), but our skippable unit is the inner
    ``Attention`` layer one level below (``blocks.5.attn1.attn``). So a pattern matches
    the layer itself OR any ancestor of it -- ignoring the module ignores its attention.
    Boundary-safe (``blocks.1`` does not match ``blocks.11``) because the ``.*`` suffix
    only matches at a dotted-segment boundary.
    """
    names = layer_match_names(module_name)
    for p in ignore_patterns or ():
        for n in names:
            if fnmatch.fnmatch(n, p) or fnmatch.fnmatch(n, p + ".*"):
                return True
    return False


def select_expert(module_name: str, expert_keys) -> str | None:
    """Which transformer/expert this layer belongs to, by full-name prefix.

    Longest key first so ``transformer_2`` wins over ``transformer``. Returns None for
    single-transformer models (the caller then uses the sole calibration).
    """
    for key in sorted(expert_keys or (), key=len, reverse=True):
        if module_name == key or module_name.startswith(key + "."):
            return key
    return None


def resolve_layer_calibration(module_name: str, calibration: dict) -> dict | None:
    """Resolve the effective {a, b, target_sparsity, formula} for one layer, or None
    if the layer is ignored (dense) or the model has no calibration.

    ``calibration`` is {"by_expert": {expert_key: {a,b,ignore,...}}} for multi-expert
    models, or a single {a,b,ignore,...} dict. Ignore matching uses each expert's own
    ignore list (falling back to a top-level one).
    """
    if not calibration:
        return None
    by_expert = calibration.get("by_expert")
    if by_expert:
        key = select_expert(module_name, by_expert.keys())
        entry = by_expert.get(key) if key else next(iter(by_expert.values()))
    else:
        entry = calibration
    if is_ignored(module_name, entry.get("ignore")):
        return None
    return {k: entry.get(k) for k in ("a", "b", "target_sparsity", "formula")}


def apply_to_pipeline(pipeline, calibration: dict) -> int:
    """Post-build walk: stamp each skippable trtllm attention layer with its per-layer
    calibration curve (a, b), resolved from its FULL module name (so per-expert routing
    and the ignore set are both handled by name -- the single mechanism TensorRT-LLM uses).

    Ignored / cross-attention layers are left untouched, so they stay dense (the impl is
    only skip-enabled once it holds a, b). Returns the number of layers stamped -- callers
    log it, and 0 when a calibration was expected is a loud signal the walk missed.

    Structural note: ``named_modules()`` yields canonical dotted paths
    (``transformer.blocks.5.attn1.attn`` vs ``transformer_2.blocks.5.attn1.attn``), which
    is the ONLY place the two Wan experts are distinguishable -- at build time both carry
    the same relative prefix, which is what silently disabled per-expert skip before.
    """
    if not calibration:
        return 0
    stamped = 0
    for name, module in pipeline.named_modules():
        impl = getattr(module, "attention", None)
        set_layer = getattr(impl, "set_layer_calibration", None)
        if set_layer is None:
            continue  # not a skip-capable (trtllm) attention layer
        per = resolve_layer_calibration(name, calibration)
        if per and per.get("a") is not None and per.get("b") is not None:
            set_layer(per["a"], per["b"])
            stamped += 1
    return stamped
