# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse a ModelOpt skip-softmax sparse_attention_config into the fields the backend
needs: the calibration curve (a, b), the ignore set, the default target_sparsity, and
the temporal gate. The formula is validated (fail-loud) rather than assumed."""

_SUPPORTED_FORMULA = "a*exp(b*target_sparsity)"


def parse_sparse_attention_config(sac: dict | None) -> dict | None:
    """ModelOpt sparse_attention_config -> {a, b, target_sparsity, disabled_until_timestep,
    ignore, formula}, or None if it carries no skip-softmax calibration.

    Handles both schemas (0.44 top-level; 0.45 config_groups.group_N) and both a,b keys
    ("coefficients" / "prefill"). Selects the skip_softmax group explicitly (a checkpoint
    may also carry a sparse_softmax N:M group). Raises on an unsupported formula so a
    future/hand-authored curve fails loud instead of being silently miscomputed.
    """
    if not sac:
        return None
    group: dict = {}
    for g in (sac.get("config_groups") or {}).values():
        if isinstance(g, dict) and (g.get("algorithm") == "skip_softmax" or g.get("threshold_scale_factor")):
            group = g
            break
    tsf = group.get("threshold_scale_factor") or sac.get("threshold_scale_factor") or {}
    ab = tsf.get("coefficients") or tsf.get("prefill") or {}
    if "a" not in ab or "b" not in ab:
        return None

    formula = (tsf.get("formula") or _SUPPORTED_FORMULA).replace(" ", "")
    if formula != _SUPPORTED_FORMULA:
        raise ValueError(
            f"unsupported skip-softmax formula {tsf.get('formula')!r}; this backend "
            f"implements '{_SUPPORTED_FORMULA}'."
        )

    def _phase_scalar(v):
        # DiT is single-phase; accept a flat scalar or a {"prefill": x} dict.
        if isinstance(v, dict):
            return float(v["prefill"]) if v.get("prefill") is not None else None
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "a": float(ab["a"]),
        "b": float(ab["b"]),
        "target_sparsity": _phase_scalar(group.get("target_sparsity")),
        "disabled_until_timestep": _phase_scalar(group.get("disabled_until_timestep")),
        "ignore": list(group.get("ignore") or []),
        "formula": _SUPPORTED_FORMULA,
    }
