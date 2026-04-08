import os
from collections import defaultdict
from collections.abc import Callable

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_ltx2_lora_to_diffusers,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from vllm.logger import init_logger

from vllm_omni.diffusion.utils.tf_utils import find_module_with_attr, get_transformer_from_pipeline

logger = init_logger(__name__)


lora_convert_mapping: dict[str, Callable] = {
    "LTX2Pipeline": _convert_non_diffusers_ltx2_lora_to_diffusers,
}


def _load_lora_state_dict(
    pretrained_model_name_or_path: str,
    weights_name: str | None = None,
    use_safetensors: bool = True,
    subfolder: str | None = None,
    cache_dir: str | None = None,
    force_download: bool = False,
    local_files_only: bool | None = None,
):
    # first we try to load it from local disk
    if use_safetensors and pretrained_model_name_or_path.endswith(".safetensors"):
        return load_file(pretrained_model_name_or_path)

    if os.path.isdir(pretrained_model_name_or_path):
        model_file = pretrained_model_name_or_path
        if subfolder:
            model_file = os.path.join(model_file, subfolder)
        if weights_name is None:
            raise ValueError("weights_name is required when loading from a directory")
        model_file = os.path.join(model_file, weights_name)
        return load_file(model_file)

    # finally, we try to load it from the internet
    try:
        model_file = hf_hub_download(
            pretrained_model_name_or_path,
            filename=weights_name,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
        )
        return load_file(model_file)
    except Exception as e:
        logger.error(f"Failed to download {weights_name} from {pretrained_model_name_or_path}: {e}")
        raise e


class LTX2LoRALoaderMinxin:
    transformer_name = "transformer"
    connectors_name = "connectors"

    lora_loaded = set()
    lora_loaded_deltas = {}

    def load_lora_weights(
        self,
        pretrained_model_name_or_path: str,
        adapter_name: str | None = None,
    ):
        if adapter_name in self.lora_loaded:
            return
        self.lora_loaded.add(adapter_name)

        state_dict = _load_lora_state_dict(pretrained_model_name_or_path)

        lora_state_dict = state_dict
        is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)
        has_connector = any(k.startswith("text_embedding_projection.") for k in state_dict)

        if is_non_diffusers_format:
            cls_name = self.__class__.__name__
            converter = lora_convert_mapping.get(cls_name, None)
            if converter is None:
                raise ValueError(f"Converter for Lora weights not found for {cls_name}")

            lora_state_dict = converter(state_dict)

        if has_connector:
            connector_state_dict = converter(state_dict, "text_embedding_projection")
            lora_state_dict.update(connector_state_dict)

        transformer_sd = {k: v for k, v in lora_state_dict.items() if k.startswith(self.transformer_name)}
        connectors_sd = {k: v for k, v in lora_state_dict.items() if k.startswith(self.connectors_name)}

        transformer = get_transformer_from_pipeline(self)
        lora_deltas = self.load_lora_into_module(
            transformer_sd,
            transformer,
            prefix=self.transformer_name,
        )

        connectors = find_module_with_attr(self, self.connectors_name).connectors
        if connectors_sd:
            lora_deltas.update(
                self.load_lora_into_module(
                    connectors_sd,
                    connectors,
                    prefix=self.connectors_name,
                )
            )

        self.lora_loaded_deltas[adapter_name] = lora_deltas

    def unload_lora_weights(self, adapter_name: str):
        if adapter_name not in self.lora_loaded:
            return
        lora_deltas = self.lora_loaded_deltas[adapter_name]

        transformer = get_transformer_from_pipeline(self)
        self.unload_module_lora(transformer, lora_deltas, prefix=self.transformer_name)

        connectors = find_module_with_attr(self, self.connectors_name).connectors
        self.unload_module_lora(connectors, lora_deltas, prefix=self.connectors_name)

        self.lora_loaded.remove(adapter_name)
        del self.lora_loaded_deltas[adapter_name]

    @classmethod
    def load_lora_into_module(
        cls,
        state_dict,
        module,
        prefix: str = "transformer",
    ):
        lora_deltas = {}
        lora_a_suffix = "lora_A.weight"
        lora_b_suffix = "lora_B.weight"

        # handle stacked QKV fusion
        stacked_sd = defaultdict(list)
        for key, param in state_dict.items():
            base_key = key[: -len(f".{lora_a_suffix}")]
            is_stacked_param = False
            if hasattr(module, "stacked_params_mapping"):
                # attn1.to_q.lora_A.weight
                #                           => attn1.to_q => attn1.to_q.delta
                # attn1.to_q.lora_B.weight
                for param_name, weight_name, _ in module.stacked_params_mapping:
                    if weight_name not in key:
                        continue
                    is_stacked_param = True
                    if key.endswith(lora_a_suffix):
                        a = param
                        b = state_dict[f"{base_key}.{lora_b_suffix}"]
                        delta = torch.matmul(b, a)
                        stacked_base_key = key.replace(weight_name, param_name)[: -len(f".{lora_a_suffix}")]
                        stacked_sd[stacked_base_key].append(delta)

                if is_stacked_param:
                    continue

                if key.endswith(lora_a_suffix):
                    a = param
                    b = state_dict[f"{base_key}.{lora_b_suffix}"]
                    delta = torch.matmul(b, a)
                    lora_deltas[base_key] = delta
            else:
                if key.endswith(lora_a_suffix):
                    a = param
                    b = state_dict[f"{base_key}.{lora_b_suffix}"]
                    delta = torch.matmul(b, a)
                    lora_deltas[base_key] = delta

        for stacked_base_key, delta_list in stacked_sd.items():
            stacked_delta = torch.concat(delta_list)
            lora_deltas[stacked_base_key] = stacked_delta

        lora_loaded_keys = set()
        lora_key_loaded_count = 0
        for name, params in module.named_parameters(prefix):
            if not name.endswith(".weight"):
                continue

            base_key = name[: -len(".weight")]
            lora_a_key = f"{base_key}.{lora_a_suffix}"
            lora_b_key = f"{base_key}.{lora_b_suffix}"

            if base_key not in lora_deltas:
                continue

            delta = lora_deltas[base_key].to(device=params.device, dtype=params.dtype)
            params.add_(delta)
            lora_key_loaded_count += 1
            del delta

            is_stacked_param = False
            if hasattr(module, "stacked_params_mapping"):
                for param_name, weight_name, _ in module.stacked_params_mapping:
                    if param_name not in base_key:
                        continue
                    is_stacked_param = True
                    lora_a_key = f"{base_key.replace(param_name, weight_name)}.{lora_a_suffix}"
                    lora_b_key = f"{base_key.replace(param_name, weight_name)}.{lora_b_suffix}"

                    if lora_a_key in state_dict:
                        lora_loaded_keys.add(lora_a_key)
                    else:
                        logger.warning(f"Failed to index lora key {lora_a_key}")
                    if lora_b_key in state_dict:
                        lora_loaded_keys.add(lora_b_key)
                    else:
                        logger.warning(f"Failed to index lora key {lora_b_key}")

                if not is_stacked_param:
                    lora_loaded_keys.add(lora_a_key)
                    lora_loaded_keys.add(lora_b_key)
            else:
                lora_loaded_keys.add(lora_a_key)
                lora_loaded_keys.add(lora_b_key)

        for k in state_dict:
            if k not in lora_loaded_keys:
                logger.warning(f"Missing loading lora key: {k}")

        logger.info(f"{lora_key_loaded_count} lora keys loaded into {module.__class__.__name__}.")

        return lora_deltas

    def unload_module_lora(
        self,
        module,
        lora_deltas,
        prefix: str = "transformer",
    ):
        lora_key_unload_count = 0
        for name, param in module.named_parameters(prefix):
            if not name.endswith(".weight"):
                continue

            base_key = name[: -len(".weight")]
            if base_key not in lora_deltas:
                continue

            delta = lora_deltas[base_key].to(device=param.device, dtype=param.dtype)
            param.sub_(delta)
            lora_key_unload_count += 1

        logger.info(f"Unload {lora_key_unload_count} lora keys from {module.__class__.__name__}.")
