# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from typing import Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_FURA_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .config import FuRAConfig
from .layer import FuRALayer, Linear, _unwrap_base_layer


def _get_in_features(base_layer: nn.Module) -> int:
    return _unwrap_base_layer(base_layer).in_features


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[FuRALayer] | None:
    layer_cls: type[FuRALayer] | None = None
    # Handle wrappers like Gemma4ClippableLinear by checking for a .linear attribute
    actual_layer = target_base_layer
    if hasattr(target_base_layer, "linear") and isinstance(target_base_layer.linear, torch.nn.Linear):
        actual_layer = target_base_layer.linear

    if isinstance(actual_layer, torch.nn.Linear):
        layer_cls = Linear
    elif (quant_backend := resolve_quantization_backend(actual_layer)) is not None:
        layer_cls = {"linear": Linear}.get(quant_backend.layer_type)
    return layer_cls


def _resolve_input_factorization(
    model: nn.Module, fura_config: FuRAConfig, current_key: str, in_features: int
) -> Optional[tuple[int, int]]:
    """Resolve `config.input_factorization` to a concrete `(in_blocks, in_block_size)` pair for one module.

    `None` means "let the layer pick the near-square factorization".
    """
    spec = fura_config.input_factorization
    if spec is None or spec == "closest":
        return None
    if isinstance(spec, (tuple, list)):
        return tuple(spec)
    if isinstance(spec, dict):
        # Keys are matched as suffixes of the fully qualified module name, longest key first, so that
        # e.g. {"q_proj": ..., "layers.0.self_attn.q_proj": ...} resolves to the more specific entry.
        for key in sorted(spec, key=len, reverse=True):
            if current_key == key or current_key.endswith("." + key):
                return tuple(spec[key])
        return None

    # spec == "head": factorize the input dimension into (num_attention_heads, head_dim).
    model_config = getattr(model, "config", None)
    num_heads = getattr(model_config, "num_attention_heads", None)
    if num_heads is None:
        raise ValueError(
            "input_factorization='head' requires the base model to expose `config.num_attention_heads`, but it "
            "does not. Pass an explicit (in_blocks, in_block_size) tuple instead."
        )
    if in_features % num_heads != 0:
        raise ValueError(
            f"input_factorization='head' cannot be applied to module {current_key!r}: in_features={in_features} is "
            f"not divisible by num_attention_heads={num_heads}. Pass an explicit tuple or a dict instead."
        )
    return (num_heads, in_features // num_heads)


class FuRAModel(BaseTuner):
    """
    Creates a FuRA (Full-Rank Adaptation with Spectral Preconditioning) model from a pretrained model.

    Paper: https://arxiv.org/abs/2605.22869

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`FuRAConfig`]): The configuration of the FuRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
            Create empty adapter weights on meta device.
    """

    prefix: str = "fura_"
    tuner_layer_cls = FuRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_FURA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        fura_config: FuRAConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: Optional[str] = None,
        **optional_kwargs,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        kwargs = get_quantization_kwargs(self)

        base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        in_features = _get_in_features(base_layer)
        input_factorization = _resolve_input_factorization(self.model, fura_config, current_key, in_features)

        # If it is not a FuRALayer, create a new module, else update it with new adapter
        if not isinstance(target, FuRALayer):
            new_module = self._create_new_module(
                fura_config, adapter_name, target, input_factorization=input_factorization, **kwargs
            )
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                config=fura_config,
                input_factorization=input_factorization,
            )

    @classmethod
    def _get_adapter_state_dict(cls, model, config, adapter_name, state_dict, unwanted_adapter_names):
        to_return = super()._get_adapter_state_dict(model, config, adapter_name, state_dict, unwanted_adapter_names)
        if config.save_frozen_core:
            return to_return

        # The frozen core is a factorization of the pretrained weight and is recomputed by `update_layer` when the
        # adapter is injected, so it does not have to travel in the checkpoint.
        frozen_keys = set()
        for module_name, module in model.named_modules():
            if not isinstance(module, FuRALayer):
                continue
            for weight_name in module.frozen_peft_weight_names.get(adapter_name, ()):
                frozen_keys.add(f"{module_name}.{weight_name}.{adapter_name}")
        return {k: v for k, v in to_return.items() if k not in frozen_keys}

    @staticmethod
    def _create_new_module(
        fura_config: FuRAConfig,
        adapter_name: str,
        target: nn.Module,
        input_factorization: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> nn.Module:
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)
        if layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` (optionally quantized) "
                "is supported for FuRA."
            )

        if (layer_cls == Linear) and fura_config.fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            fura_config.fan_in_fan_out = False

        new_module = layer_cls(
            target, adapter_name, config=fura_config, input_factorization=input_factorization, **kwargs
        )
        return new_module
