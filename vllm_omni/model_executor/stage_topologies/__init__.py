# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stage topology definitions for vLLM-Omni pipeline models.

Topology YAML files in this directory define Tier-1 (internal) pipeline
structure: stages, their types, and data-flow connections.  Runtime
parameters (GPU memory, tensor-parallel size, etc.) are NOT stored here;
they come from CLI flags (Tier-2).
"""

from pathlib import Path

TOPOLOGIES_DIR = Path(__file__).parent


def get_topology_path(filename: str) -> Path:
    """Return the full path to a topology YAML file in this directory.

    Args:
        filename: Name of the YAML file (e.g., "qwen3_omni_moe.yaml").

    Returns:
        Absolute path to the topology file.
    """
    return TOPOLOGIES_DIR / filename
