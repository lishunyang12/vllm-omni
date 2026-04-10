import os
from collections import OrderedDict, defaultdict
from collections.abc import Callable

import torch
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_ltx2_lora_to_diffusers,
    _convert_non_diffusers_qwen_lora_to_diffusers,
    _convert_non_diffusers_wan_lora_to_diffusers,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from vllm.logger import init_logger

from vllm_omni.diffusion.utils.tf_utils import find_module_with_attr, get_transformer_from_pipeline

logger = init_logger(__name__)

LORA_DIFFUSERS_CONVERTER_REGISTRY = {}

lora_convert_mapping: dict[str, Callable] = {
    "LTX2Pipeline": _convert_non_diffusers_ltx2_lora_to_diffusers,
    "QwenImagePipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "QwenImageEditPipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "QwenImageEditPlusPipeline": _convert_non_diffusers_qwen_lora_to_diffusers,
    "Wan22Pipeline": _convert_non_diffusers_wan_lora_to_diffusers,
    "Wan22I2VPipeline": _convert_non_diffusers_wan_lora_to_diffusers,
}


def get_converter_by_pipeline(pipeline):
    return lora_convert_mapping.get(pipeline.__class__.__name__, None)


def _prepare_lora_deltas(lora_sd, module, lora_a_suffix="lora_A.weight", lora_b_suffix="lora_B.weight"):
    lora_deltas = {}

    # prepare stacked parameters mapping for later use
    # stacked_params_mapping                       param_to_weight_names
    # [(".to_qkv", ".to_q.", "q")
    # (".to_qkv", ".to_k.", "k")  ========> {".to_qkv": [".to_q", ".to_k", ".to_v"]}
    # (".to_qkv", ".to_v.", "v")]

    weight_to_param_name = {}
    if hasattr(module, "stacked_params_mapping"):
        weight_to_param_name = {weight_name: param_name for param_name, weight_name, _ in module.stacked_params_mapping}

    # stacked_sd is to store packed parameter deltas (format: [str, list[torch.Tensor]])
    # example: {".to_qkv": [delta_q, delta_k, delta_v]}
    stacked_sd = defaultdict(list)
    for key, param in lora_sd.items():
        base_key = key[: -len(f".{lora_a_suffix}")]

        is_stacked_param = False
        for weight_name, param_name in weight_to_param_name.items():
            if weight_name not in key:
                continue
            is_stacked_param = True
            # handle lora_a_key and lora_b_key together
            if key.endswith(lora_a_suffix):
                a = param
                b = lora_sd[f"{base_key}.{lora_b_suffix}"]
                delta = torch.matmul(b, a)
                stacked_base_key = key.replace(weight_name, param_name)[: -len(f".{lora_a_suffix}")]
                stacked_sd[stacked_base_key].append(delta)
            else:
                # lora_b_key was already handled, skip
                continue

        if is_stacked_param:
            # already handled, skip
            continue

        if key.endswith(lora_a_suffix):
            a = param
            b = lora_sd[f"{base_key}.{lora_b_suffix}"]
            delta = torch.matmul(b, a)
            lora_deltas[base_key] = delta
        else:
            # same as above
            continue

    for stacked_base_key, delta_list in stacked_sd.items():
        stacked_delta = torch.concat(delta_list)
        lora_deltas[stacked_base_key] = stacked_delta

    return lora_deltas


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


def _remap_state_dict_keys(sd, rules):
    """
    Remap keys in a state_dict based on a priority list of substring substitution rules.

    This utility function transforms parameter names by applying a sequence of
    (old_substring, new_substring) pairs.

    For each key in the input dictionary, the `rules` are evaluated in the given order.
    The **first** rule whose `old_substring` is found within the key triggers a
    replacement. Once a match occurs, the search stops and no further rules are
    considered for that key. Keys that do not match any rule are copied unchanged.

    Args:
        sd (dict[str, Any]): Original state_dict mapping parameter names to tensors.
        rules (List[Tuple[str, str]]): A list of (old_substring, new_substring) pairs.
            Rules are applied sequentially with first-match precedence.

    Returns:
        dict[str, Any]: A new state_dict with remapped keys. Values are unchanged.
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        matched = False
        for p1, p2 in rules:
            if p1 not in k:
                continue
            new_sd[k.replace(p1, p2)] = v
            matched = True
            break
        if not matched:
            new_sd[k] = v
    return new_sd


class LoraLoaderMixin:
    transformer_name = "transformer"
    lora_loaded = set()
    lora_loaded_deltas = {}

    @classmethod
    def load_lora_into_module(
        cls,
        state_dict,
        module,
        prefix: str = "transformer",
        lora_a_suffix: str = "lora_A.weight",
        lora_b_suffix: str = "lora_B.weight",
    ):
        lora_deltas = _prepare_lora_deltas(state_dict, module, lora_a_suffix, lora_b_suffix)
        lora_loaded_keys = set()
        lora_key_loaded_count = 0

        param_to_weight_names = defaultdict(list)
        if hasattr(module, "stacked_params_mapping"):
            for param_name, weight_name, _ in module.stacked_params_mapping:
                param_to_weight_names[param_name].append(weight_name)

        def update_loaded_keys(base_key):
            """
            This function updates lora_loaded_keys. It must be called after the parameter
            of base_key is merged into module.
            """
            lora_a_key = f"{base_key}.{lora_a_suffix}"
            lora_b_key = f"{base_key}.{lora_b_suffix}"

            is_stacked_param = False
            for param_name, weight_names in param_to_weight_names.items():
                if param_name not in base_key:
                    continue
                is_stacked_param = True
                for weight_name in weight_names:
                    lora_a_key = f"{base_key.replace(param_name, weight_name)}.{lora_a_suffix}"
                    lora_b_key = f"{base_key.replace(param_name, weight_name)}.{lora_b_suffix}"
                    for k in (lora_a_key, lora_b_key):
                        if k in state_dict:
                            lora_loaded_keys.add(k)
                        else:
                            logger.warning(f"Failed to index lora key {k}")
                break

            if not is_stacked_param:
                # sanity check is no need, as lora_deltas already checked
                lora_loaded_keys.add(lora_a_key)
                lora_loaded_keys.add(lora_b_key)

        for name, params in module.named_parameters(prefix):
            if not name.endswith(".weight"):
                continue

            base_key = name[: -len(".weight")]
            if base_key not in lora_deltas:
                continue

            delta = lora_deltas[base_key].to(device=params.device, dtype=params.dtype)
            params.add_(delta)
            lora_key_loaded_count += 1
            del delta

            update_loaded_keys(base_key)

        for k in state_dict:
            if k not in lora_loaded_keys:
                logger.warning(f"Missing loading lora key: {k}")

        logger.info(f"{lora_key_loaded_count} lora keys loaded into {module.__class__.__name__}.")

        return lora_deltas

    @classmethod
    def unload_module_lora(
        cls,
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


class QwenImageLoraLoaderMixin(LoraLoaderMixin):
    def load_lora_weights(
        self,
        pretrained_model_name_or_path: str,
        adapter_name: str | None = None,
    ):
        if adapter_name in self.lora_loaded:
            return
        self.lora_loaded.add(adapter_name)

        state_dict = _load_lora_state_dict(pretrained_model_name_or_path)

        has_alpha = any(k.endswith(".alpha") for k in state_dict)
        is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)
        if has_alpha or is_non_diffusers_format:
            converter = get_converter_by_pipeline(self)
            if converter is None:
                raise ValueError(f"Converter for Lora weights not found for {self.__class__.__name__}")

            state_dict = converter(state_dict)

        state_dict = _remap_state_dict_keys(state_dict, [(".to_out.0.", ".to_out.")])

        lora_deltas = self.load_lora_into_module(
            state_dict,
            self.transformer,
            prefix=self.transformer_name,
        )

        self.lora_loaded_deltas[adapter_name] = lora_deltas

    def unload_lora_weights(self, adapter_name: str):
        if adapter_name not in self.lora_loaded:
            return
        lora_deltas = self.lora_loaded_deltas[adapter_name]

        transformer = get_transformer_from_pipeline(self)
        self.unload_module_lora(transformer, lora_deltas, prefix=self.transformer_name)

        self.lora_loaded.remove(adapter_name)
        del self.lora_loaded_deltas[adapter_name]


class LTX2LoraLoaderMinxin(LoraLoaderMixin):
    connectors_name = "connectors"

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
            converter = get_converter_by_pipeline(self)
            if converter is None:
                raise ValueError(f"Converter for Lora weights not found for {self.__class__.__name__}")

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


class WanLoraLoaderMixin(LoraLoaderMixin):
    supported_ckpt_mapping = {
        "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors": "transformer",
        "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors": "transformer_2",
        "wan2.2_t2v_A14b_high_noise_lora_rank64_lightx2v_4step_1217.safetensors": "transformer",
        "wan2.2_t2v_A14b_low_noise_lora_rank64_lightx2v_4step_1217.safetensors": "transformer_2",
        "high_noise_model.safetensors": "transformer",
        "low_noise_model.safetensors": "transformer_2",
    }

    def load_lora_weights(
        self,
        pretrained_model_name_or_path: str,
        adapter_name: str | None = None,
    ):
        if adapter_name in self.lora_loaded:
            return
        self.lora_loaded.add(adapter_name)

        cls_name = self.__class__.__name__
        task = "i2v" if "I2V" in cls_name else "t2v"

        lora_paths = self._get_target_lora_paths(pretrained_model_name_or_path, self.has_transformer_2, task=task)
        for lora_path in lora_paths:
            state_dict = _load_lora_state_dict(lora_path)
            is_non_diffusers_format = any(k.startswith("diffusion_model.") for k in state_dict)
            if is_non_diffusers_format:
                converter = get_converter_by_pipeline(self)
                if converter is None:
                    raise ValueError(f"Converter for Lora weights not found for {self.__class__.__name__}")

                state_dict = converter(state_dict)

            state_dict = _remap_state_dict_keys(
                state_dict,
                [
                    (".ffn.net.0.", ".ffn.net_0."),
                    (".ffn.net.2.", ".ffn.net_2."),
                    (
                        ".to_out.0.",
                        ".to_out.",
                    ),
                ],
            )

            if self.has_transformer_2:
                # wan22 load path
                filename = os.path.basename(lora_path)
                target_module_name = self.supported_ckpt_mapping[filename]
                module = getattr(find_module_with_attr(self, target_module_name), target_module_name)
                self.load_lora_into_module(state_dict, module, prefix=self.transformer_name)
            else:
                # wan21 load path
                self.load_lora_into_module(state_dict, self.transformer, prefix=self.transformer_name)

    def unload_lora_weights(self, adapter_name: str):
        if adapter_name not in self.lora_loaded:
            return
        lora_deltas = self.lora_loaded_deltas[adapter_name]

        transformer = get_transformer_from_pipeline(self)
        self.unload_module_lora(transformer, lora_deltas, prefix=self.transformer_name)

        self.lora_loaded.remove(adapter_name)
        del self.lora_loaded_deltas[adapter_name]

    def _get_target_lora_paths(
        self,
        pretrained_model_name_or_path,
        is_wan22: bool = True,
        task: str = "t2v",
    ):
        lora_paths = []
        if os.path.isdir(pretrained_model_name_or_path):
            for filename in os.listdir(pretrained_model_name_or_path):
                if not filename.endswith(".safetensors"):
                    continue
                if filename not in self.supported_ckpt_mapping:
                    continue
                if task in filename.lower():
                    lora_paths.append(os.path.join(pretrained_model_name_or_path, filename))

            return lora_paths

        if is_wan22:
            raise ValueError("Wan22 distilled LoRA weights are not supported by directory")

        filename = os.path.basename(pretrained_model_name_or_path).lower()
        if task not in filename:
            raise ValueError(f"LoRA weights {pretrained_model_name_or_path} is not a {task} LoRA weights")

        lora_paths.append(pretrained_model_name_or_path)
        return lora_paths
