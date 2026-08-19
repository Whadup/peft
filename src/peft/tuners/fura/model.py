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
from .layer import FuRALayer, Linear


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

        # If it is not a FuRALayer, create a new module, else update it with new adapter
        if not isinstance(target, FuRALayer):
            new_module = self._create_new_module(fura_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                config=fura_config,
            )

    @staticmethod
    def _create_new_module(fura_config: FuRAConfig, adapter_name: str, target: nn.Module, **kwargs) -> nn.Module:
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

        new_module = layer_cls(target, adapter_name, config=fura_config, **kwargs)
        return new_module

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """Freeze base model parameters and set proper requires_grad for FuRA adapter parameters."""
        super()._mark_only_adapters_as_trainable(model)
        for module in model.modules():
            if isinstance(module, FuRALayer):
                module._freeze_non_trainable_peft_weights()
