# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.lora.raw_loader import (
    LTX25_DISTILLED_LORA_FILENAME,
    load_ltx_native_lora,
    resolve_ltx25_distilled_lora,
)
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX25_FULL_TWO_STAGE_RECIPE
from vllm_omni.diffusion.models.ltx2.ltx2_runtime import LTXRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SlotLayer:
    n_slices = 3
    output_slices = (2, 2, 2)

    def __init__(self) -> None:
        self.set_calls: list[int] = []
        self.reset_calls: list[int] = []
        self.toggle_calls: list[tuple[int, bool]] = []

    def set_lora(self, index: int, lora_a, lora_b) -> None:
        assert len(lora_a) == len(lora_b) == 3
        self.set_calls.append(index)

    def reset_lora(self, index: int) -> None:
        self.reset_calls.append(index)

    def set_lora_active(self, index: int, active: bool) -> None:
        self.toggle_calls.append((index, active))


def _logical_qkv_model(adapter_id: int):
    rank = 2
    loras = {}
    for name in ("to_q", "to_k", "to_v"):
        module_name = f"transformer_blocks.0.attn1.{name}"
        loras[module_name] = LoRALayerWeights(
            module_name=module_name,
            rank=rank,
            lora_alpha=rank,
            lora_a=torch.ones((rank, 4)),
            lora_b=torch.ones((2, rank)),
        )
    return type(
        "_LoRAModel",
        (),
        {
            "id": adapter_id,
            "loras": loras,
            "get_lora": lambda self, key: self.loras.get(key),
        },
    )()


def test_native_loader_maps_mixed_rank_official_keys_without_peft_scaling(tmp_path: Path) -> None:
    path = tmp_path / "lora.safetensors"
    save_file(
        {
            "diffusion_model.patchify_proj.lora_A.weight": torch.ones((1, 4)),
            "diffusion_model.patchify_proj.lora_B.weight": torch.ones((6, 1)),
            "diffusion_model.adaln_single.linear.lora_A.weight": torch.ones((1, 4)),
            "diffusion_model.adaln_single.linear.lora_B.weight": torch.ones((6, 1)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": torch.ones((2, 4)),
            "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": torch.ones((6, 2)),
        },
        str(path),
    )

    model, helper = load_ltx_native_lora(str(path), lora_model_id=7, dtype=torch.float32)

    assert helper.r == helper.lora_alpha == 2
    assert set(model.loras) == {"proj_in", "time_embed.linear", "transformer_blocks.0.attn1.to_q"}
    assert model.loras["proj_in"].lora_a.shape[0] == 1
    assert model.loras["transformer_blocks.0.attn1.to_q"].lora_a.shape[0] == 2
    assert all(weights.scaling == 1.0 for weights in model.loras.values())


def test_resolve_lora450_requires_official_local_artifact(tmp_path: Path) -> None:
    expected = tmp_path / LTX25_DISTILLED_LORA_FILENAME
    expected.parent.mkdir(parents=True)
    expected.touch()

    assert resolve_ltx25_distilled_lora(str(tmp_path)) == str(expected)

    expected.unlink()
    with pytest.raises(FileNotFoundError, match="Full LTX-2.5 two-stage requires"):
        resolve_ltx25_distilled_lora(str(tmp_path))


def test_resolve_lora450_threads_hub_revision(monkeypatch) -> None:
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return "/cache/lora.safetensors"

    monkeypatch.setattr("vllm_omni.diffusion.lora.raw_loader.hf_hub_download", fake_download)

    assert resolve_ltx25_distilled_lora("Lightricks/LTX-2.5", revision="pinned-revision") == "/cache/lora.safetensors"
    assert captured == {
        "repo_id": "Lightricks/LTX-2.5",
        "filename": LTX25_DISTILLED_LORA_FILENAME,
        "revision": "pinned-revision",
    }


def test_request_and_resident_adapters_use_independent_slots() -> None:
    pipeline = torch.nn.Module()
    pipeline.stacked_params_mapping = [
        (".to_qkv", ".to_q", "q"),
        (".to_qkv", ".to_k", "k"),
        (".to_qkv", ".to_v", "v"),
    ]
    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_lora_slots=2,
    )
    layer = _SlotLayer()
    manager._lora_modules = {"transformer.transformer_blocks.0.attn1.to_qkv": layer}
    manager._registered_adapters[11] = _logical_qkv_model(11)
    resident = _logical_qkv_model(22)
    manager._resident_adapters[1] = ("distilled_lora_450", 22, resident, 1.0, False)
    manager._resident_adapter_slots["distilled_lora_450"] = 1

    manager._activate_adapter(11, 0.5)
    manager._activate_adapter(22, 1.0, slot=1, resident=True)
    manager.set_resident_adapter_active("distilled_lora_450", False)
    manager.set_resident_adapter_active("distilled_lora_450", True)
    manager.deactivate_resident_adapters()

    assert layer.set_calls == [0, 1]
    assert layer.toggle_calls == [(1, False), (1, True), (1, False)]
    assert manager._active_adapter_id == 11


def test_resident_adapter_zero_layer_match_fails() -> None:
    manager = DiffusionLoRAManager(
        pipeline=torch.nn.Module(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_lora_slots=2,
    )
    resident = _logical_qkv_model(22)
    manager._resident_adapters[1] = ("distilled_lora_450", 22, resident, 1.0, False)

    with pytest.raises(RuntimeError, match="matched zero transformer layers"):
        manager._activate_adapter(22, 1.0, slot=1, resident=True)


def test_full_two_stage_phase_adapter_is_stage2_only() -> None:
    runtime = object.__new__(LTXRuntime)
    runtime.pipeline_recipe = LTX25_FULL_TWO_STAGE_RECIPE
    events: list[tuple[str | None, bool]] = []
    runtime.set_phase_adapter_controller(lambda name, active: events.append((name, active)))

    runtime._enter_phase(runtime.pipeline_recipe.phases[0])
    runtime._enter_phase(runtime.pipeline_recipe.phases[1])
    runtime._deactivate_phase_adapters()

    assert events == [
        (None, False),
        (None, False),
        ("distilled_lora_450", True),
        (None, False),
    ]


def test_full_two_stage_fails_before_stage1_without_lora450() -> None:
    runtime = object.__new__(LTXRuntime)
    runtime.pipeline_recipe = LTX25_FULL_TWO_STAGE_RECIPE
    runtime._phase_adapter_controller = None

    with pytest.raises(RuntimeError, match="resident adapters"):
        runtime._run_recipe(None, None, request_sigmas=None)
