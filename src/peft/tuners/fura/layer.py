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

import math
import warnings
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import FuRAConfig

if is_bnb_available():
    import bitsandbytes as bnb
else:
    bnb = None


def _closest_factor_pair(d: int) -> tuple[int, int]:
    """Find factors (a, b) such that a * b = d and abs(a - b) is minimized."""
    root = int(d**0.5)
    best_a, best_b = 1, d
    best_diff = best_b - best_a
    for a in range(1, root + 1):
        if d % a == 0:
            b = d // a
            diff = abs(b - a)
            if diff < best_diff:
                best_a, best_b, best_diff = a, b, diff
    return best_a, best_b


def _resolve_blocktt_trainable_sides(
    left_size: int, right_size: int, train_position: str
) -> tuple[bool, bool]:
    if train_position not in {"small", "large", "both"}:
        raise ValueError("train_position must be one of: small, large, both")
    if train_position == "both":
        return True, True
    if train_position == "small":
        train_left = left_size <= right_size
        return train_left, not train_left
    train_left = left_size >= right_size
    return train_left, not train_left


def resolve_blocktt_s_merged_to(
    train_position: str,
    s_merged_to: Optional[str] = None,
    left_size: Optional[int] = None,
    right_size: Optional[int] = None,
) -> str:
    if s_merged_to is None:
        if train_position == "both":
            return "split"
        s_merged_to = "frozen"

    if s_merged_to in {"output", "input", "split", "keep_frozen", "keep_trainable"}:
        return s_merged_to

    if left_size is None or right_size is None:
        raise ValueError("left_size and right_size are required for frozen/trainable aliases")

    train_left, train_right = _resolve_blocktt_trainable_sides(
        left_size=left_size,
        right_size=right_size,
        train_position=train_position,
    )
    if train_left and train_right:
        raise ValueError(
            "BlockTT s_merged_to frozen/trainable is invalid when both cores are trainable. "
            "Use output, input, or split."
        )

    if s_merged_to == "trainable":
        return "output" if train_left else "input"
    return "input" if train_left else "output"


class FuRALayer(BaseTunerLayer):
    """
    FuRA (Full-Rank Adaptation with Spectral Preconditioning) layer.
    """

    adapter_layer_names: tuple[str, ...] = ("fura_l", "fura_r", "fura_s", "fura_bias")
    other_param_names: tuple[str, ...] = (
        "r",
        "m",
        "n",
        "a",
        "b",
        "rank",
        "decomp_mode",
        "train_position",
        "s_merged_to",
        "convert_mode",
        "init_mode",
        "fura_dropout",
        "is_quantized",
        "quant_layout",
        "_qfura_frozen_side",
        "_qfura_frozen_shape",
        "_qfura_frozen_dtype",
        "_qfura_compute_dtype",
        "_qfura_frozen_flat",
        "_qfura_frozen_blocks",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r: dict[str, Union[int, float, str]] = {}
        self.m: dict[str, int] = {}
        self.n: dict[str, int] = {}
        self.a: dict[str, int] = {}
        self.b: dict[str, int] = {}
        self.rank: dict[str, int] = {}
        self.decomp_mode: dict[str, str] = {}
        self.train_position: dict[str, str] = {}
        self.s_merged_to: dict[str, str] = {}
        self.convert_mode: dict[str, str] = {}
        self.init_mode: dict[str, str] = {}
        self.is_quantized: dict[str, bool] = {}
        self.quant_layout: dict[str, str] = {}

        # Parameter dicts
        self.fura_l = nn.ParameterDict({})
        self.fura_r = nn.ParameterDict({})
        self.fura_s = nn.ParameterDict({})
        self.fura_bias = nn.ParameterDict({})
        self.fura_dropout = nn.ModuleDict({})

        # QFuRA internal metadata per adapter
        self._qfura_frozen_side: dict[str, str] = {}
        self._qfura_frozen_shape: dict[str, tuple[int, ...]] = {}
        self._qfura_frozen_dtype: dict[str, torch.dtype] = {}
        self._qfura_compute_dtype: dict[str, torch.dtype] = {}
        self._qfura_frozen_flat = nn.ParameterDict({})
        self._qfura_frozen_blocks: dict[str, list[Any]] = {}

        self._disable_adapters = False
        self.merged_adapters: list[str] = []
        self._base_weight_before_merge: Optional[torch.Tensor] = None
        self._base_bias_before_merge: Optional[torch.Tensor] = None
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        # Handle wrappers like Gemma4ClippableLinear by checking for a .linear attribute
        actual_base = base_layer
        if hasattr(base_layer, "linear") and isinstance(base_layer.linear, nn.Linear):
            actual_base = base_layer.linear

        if isinstance(actual_base, nn.Linear):
            self.in_features, self.out_features = actual_base.in_features, actual_base.out_features
        elif hasattr(actual_base, "in_features") and hasattr(actual_base, "out_features"):
            self.in_features, self.out_features = actual_base.in_features, actual_base.out_features
        else:
            raise TypeError(f"Unsupported layer type '{type(base_layer)}' encountered for FuRALayer.")

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.fura_r, *self.fura_l}

    @staticmethod
    def _qr_decompose_blocks(blocks: torch.Tensor, use_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-block QR/LQ decomposition."""
        a = blocks.shape[-2]
        b = blocks.shape[-1]
        if a >= b:
            # LQ: blocks.T = Qp @ Rp -> blocks = Rp.T @ Qp.T = L @ Q
            Qp, Rp = torch.linalg.qr(blocks.transpose(-1, -2), mode="reduced")
            L = Rp.transpose(-1, -2)
            Q = Qp.transpose(-1, -2)
            l_used = L[:, :, :use_rank]
            r_used = Q[:, :use_rank, :]
        else:
            # QR: blocks = Q @ R
            Q, R = torch.linalg.qr(blocks, mode="reduced")
            l_used = Q[:, :, :use_rank]
            r_used = R[:, :use_rank, :]
        return l_used.contiguous(), r_used.contiguous()

    def update_layer(
        self,
        adapter_name: str,
        config: FuRAConfig,
        **kwargs,
    ) -> None:
        decomp_mode = config.decomp_mode
        train_position = config.train_position
        s_merged_to = config.s_merged_to
        convert_mode = config.convert_mode
        init_mode = config.init_mode
        r_arg = config.r
        fura_dropout = config.fura_dropout
        init_weights = config.init_weights
        is_quantized = config.is_quantized
        quant_layout = config.quant_layout

        self.r[adapter_name] = r_arg
        self.decomp_mode[adapter_name] = decomp_mode
        self.train_position[adapter_name] = train_position
        self.convert_mode[adapter_name] = convert_mode
        self.init_mode[adapter_name] = init_mode
        self.is_quantized[adapter_name] = is_quantized
        self.quant_layout[adapter_name] = quant_layout

        if fura_dropout > 0.0:
            self.fura_dropout[adapter_name] = nn.Dropout(p=fura_dropout)
        else:
            self.fura_dropout[adapter_name] = nn.Identity()

        # Compute output factorization
        if config.output_factorization is not None:
            out_blocks, out_block_size = config.output_factorization
            if out_blocks * out_block_size != self.out_features:
                raise ValueError(
                    f"output_factorization {config.output_factorization} does not match out_features={self.out_features}"
                )
        else:
            out_blocks, out_block_size = _closest_factor_pair(self.out_features)

        # Compute input factorization
        if isinstance(config.input_factorization, (tuple, list)):
            in_blocks, in_block_size = config.input_factorization
            if in_blocks * in_block_size != self.in_features:
                raise ValueError(
                    f"input_factorization {config.input_factorization} does not match in_features={self.in_features}"
                )
        else:
            in_blocks, in_block_size = _closest_factor_pair(self.in_features)

        # Determine m, a, n, b according to decomp_mode
        if decomp_mode == "square":
            m, a, n, b = out_blocks, out_block_size, in_blocks, in_block_size
        elif decomp_mode == "output_one_block":
            m, a, n, b = 1, self.out_features, in_blocks, in_block_size
        elif decomp_mode == "input_one_block":
            m, a, n, b = out_blocks, out_block_size, 1, self.in_features
        else:
            raise ValueError(f"Unknown decomp_mode: {decomp_mode}")

        self.m[adapter_name] = m
        self.a[adapter_name] = a
        self.n[adapter_name] = n
        self.b[adapter_name] = b

        # Resolve rank
        if isinstance(r_arg, str):
            if r_arg != "full":
                raise ValueError("rank as string must be 'full'")
            resolved_rank = min(b, a)
        elif isinstance(r_arg, float):
            target_params = r_arg * self.out_features * self.in_features
            rank_denominator = m * n * (a + b)
            approx_rank = target_params / rank_denominator
            low_rank = max(1, int(np.floor(approx_rank)))
            high_rank = max(1, int(np.ceil(approx_rank)))
            low_params = m * n * low_rank * (a + b)
            high_params = m * n * high_rank * (a + b)
            resolved_rank = low_rank if abs(low_params - target_params) <= abs(high_params - target_params) else high_rank
        elif isinstance(r_arg, int):
            resolved_rank = r_arg
        else:
            raise TypeError("r must be int, float, or 'full'")

        self.rank[adapter_name] = resolved_rank

        base_layer = self.get_base_layer()
        # Handle wrappers like Gemma4ClippableLinear by checking for a .linear attribute
        actual_base = base_layer
        if hasattr(base_layer, "linear") and isinstance(base_layer.linear, nn.Linear):
            actual_base = base_layer.linear

        weight = actual_base.weight
        if hasattr(weight, "data"):
            weight_tensor = weight.data
        else:
            weight_tensor = weight
        weight_device = weight_tensor.device
        weight_dtype = weight_tensor.dtype

        # Create parameter tensors
        # btt_l packed shape: (m, rank * n, a)
        # btt_r packed shape: (n, b, m * rank)
        fura_l = nn.Parameter(torch.zeros(m, resolved_rank * n, a, device=weight_device, dtype=weight_dtype))
        fura_r = nn.Parameter(torch.zeros(n, b, m * resolved_rank, device=weight_device, dtype=weight_dtype))
        self.fura_l[adapter_name] = fura_l
        self.fura_r[adapter_name] = fura_r

        # Optional bias
        if config.bias == "fura_only":
            self.fura_bias[adapter_name] = nn.Parameter(
                torch.zeros(self.out_features, device=weight_device, dtype=weight_dtype)
            )

        # Initialize parameters from base weight or random
        self.reset_fura_parameters(adapter_name, init_weights=init_weights, config=config)

        # Configure trainability (requires_grad)
        train_left, train_right = _resolve_blocktt_trainable_sides(
            left_size=self.fura_l[adapter_name].numel(),
            right_size=self.fura_r[adapter_name].numel(),
            train_position=train_position,
        )
        self.fura_l[adapter_name].requires_grad = train_left
        self.fura_r[adapter_name].requires_grad = train_right
        if adapter_name in self.fura_s:
            resolved_s = self.s_merged_to[adapter_name]
            self.fura_s[adapter_name].requires_grad = (resolved_s == "keep_trainable")

        frozen_names = []
        if not train_left:
            frozen_names.append("fura_l")
        if not train_right:
            frozen_names.append("fura_r")
        if adapter_name in self.fura_s and self.s_merged_to.get(adapter_name) != "keep_trainable":
            frozen_names.append("fura_s")
        self.frozen_peft_weight_names[adapter_name] = tuple(frozen_names)

        # If QFuRA NF4 quantization is requested, quantize the frozen core
        if is_quantized:
            self._quantize_frozen_core(adapter_name, layout=quant_layout)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    @torch.no_grad()
    def reset_fura_parameters(self, adapter_name: str, init_weights: Union[bool, str] = True, config: Optional[FuRAConfig] = None):
        m = self.m[adapter_name]
        n = self.n[adapter_name]
        a = self.a[adapter_name]
        b = self.b[adapter_name]
        rank = self.rank[adapter_name]
        train_position = self.train_position[adapter_name]
        s_merged_to = config.s_merged_to if config is not None else "keep_trainable"
        convert_mode = self.convert_mode[adapter_name]
        init_mode = self.init_mode[adapter_name]

        base_layer = self.get_base_layer()
        # Handle wrappers like Gemma4ClippableLinear by checking for a .linear attribute
        actual_base = base_layer
        if hasattr(base_layer, "linear") and isinstance(base_layer.linear, nn.Linear):
            actual_base = base_layer.linear

        weight = actual_base.weight
        if hasattr(weight, "data"):
            weight_tensor = weight.data
        else:
            weight_tensor = weight

        if config is not None and config.fan_in_fan_out:
            weight_tensor = weight_tensor.T

        param_dtype = weight_tensor.dtype
        device = weight_tensor.device

        # If init_weights is False, initialize randomly
        if init_weights is False or init_weights == "gaussian":
            target_sdv = (self.in_features + self.out_features) ** (-0.5)
            if init_mode == "default":
                nn.init.normal_(self.fura_r[adapter_name], std=target_sdv**0.5 / (rank**0.25))
                nn.init.normal_(self.fura_l[adapter_name], std=target_sdv**0.5 / (rank**0.25))
            else:  # mup
                std_r = np.sqrt(1 / b) * min(1, np.sqrt((m * rank) / b))
                std_l = np.sqrt(1 / (rank * n)) * min(1, np.sqrt(a / (rank * n)))
                nn.init.normal_(self.fura_r[adapter_name], std=std_r)
                nn.init.normal_(self.fura_l[adapter_name], std=std_l)

            if s_merged_to in {"keep_trainable", "keep_frozen"}:
                self.s_merged_to[adapter_name] = s_merged_to
                s_param = nn.Parameter(
                    torch.ones(m, n, rank, device=device, dtype=param_dtype),
                    requires_grad=(s_merged_to == "keep_trainable"),
                )
                self.fura_s[adapter_name] = s_param
            return

        # Decompose base weight via SVD or QR
        # Base weight shape (out_features, in_features) = (m*a, n*b)
        # Reshape to (m, a, n, b) -> permute to (m, n, a, b) -> (m*n, a, b)
        blocks = weight_tensor.reshape(m, a, n, b).permute(0, 2, 1, 3).reshape(m * n, a, b)
        decomp_dtype = torch.float32 if param_dtype in (torch.float16, torch.bfloat16) else param_dtype

        max_full_rank = min(a, b)
        use_rank = min(rank, max_full_rank)

        core_l = torch.zeros(m * n, a, rank, device=device, dtype=param_dtype)
        core_r = torch.zeros(m * n, rank, b, device=device, dtype=param_dtype)

        if convert_mode == "qr":
            l_used, r_used = self._qr_decompose_blocks(blocks.to(dtype=decomp_dtype), use_rank=use_rank)
            core_l[:, :, :use_rank] = l_used.to(dtype=param_dtype)
            core_r[:, :use_rank, :] = r_used.to(dtype=param_dtype)
            self.s_merged_to[adapter_name] = "none"
            if adapter_name in self.fura_s:
                del self.fura_s[adapter_name]
        else:
            # SVD decomposition
            U, S, Vh = torch.linalg.svd(blocks.to(dtype=decomp_dtype), full_matrices=False)

            merge_target = resolve_blocktt_s_merged_to(
                train_position=train_position,
                s_merged_to=s_merged_to,
                left_size=self.fura_l[adapter_name].numel(),
                right_size=self.fura_r[adapter_name].numel(),
            )
            self.s_merged_to[adapter_name] = merge_target

            u_used = U[:, :, :use_rank].to(dtype=param_dtype)
            vh_used = Vh[:, :use_rank, :].to(dtype=param_dtype)
            s_used = torch.clamp(S[:, :use_rank], min=0).to(dtype=param_dtype)

            if merge_target in {"keep_frozen", "keep_trainable"}:
                core_l[:, :, :use_rank] = u_used
                core_r[:, :use_rank, :] = vh_used
                s_keep = torch.zeros(m * n, rank, device=device, dtype=param_dtype)
                s_keep[:, :use_rank] = s_used
                self.fura_s[adapter_name] = nn.Parameter(
                    s_keep.reshape(m, n, rank),
                    requires_grad=(merge_target == "keep_trainable"),
                )
            elif merge_target == "split":
                sqrt_s = torch.sqrt(s_used)
                core_l[:, :, :use_rank] = u_used * sqrt_s.unsqueeze(1)
                core_r[:, :use_rank, :] = sqrt_s.unsqueeze(-1) * vh_used
                if adapter_name in self.fura_s:
                    del self.fura_s[adapter_name]
            elif merge_target == "output":
                core_l[:, :, :use_rank] = u_used * s_used.unsqueeze(1)
                core_r[:, :use_rank, :] = vh_used
                if adapter_name in self.fura_s:
                    del self.fura_s[adapter_name]
            else:  # input
                core_l[:, :, :use_rank] = u_used
                core_r[:, :use_rank, :] = s_used.unsqueeze(-1) * vh_used
                if adapter_name in self.fura_s:
                    del self.fura_s[adapter_name]

        # Reshape to packed format
        # core_l: (m, n, a, rank) -> permute(0, 1, 3, 2) -> (m, n, rank, a) -> (m, rank*n, a)
        # core_r: (m, n, rank, b) -> permute(1, 3, 0, 2) -> (n, b, m, rank) -> (n, b, m*rank)
        core_l = core_l.reshape(m, n, a, rank)
        core_r = core_r.reshape(m, n, rank, b)
        packed_l = core_l.permute(0, 1, 3, 2).reshape(m, rank * n, a)
        packed_r = core_r.permute(1, 3, 0, 2).reshape(n, b, m * rank)

        self.fura_l[adapter_name].data.copy_(packed_l)
        self.fura_r[adapter_name].data.copy_(packed_r)

    def _quantize_frozen_core(
        self,
        adapter_name: str,
        layout: str = "flat",
        compute_dtype: torch.dtype = torch.bfloat16,
        double_quant: bool = True,
        quant_type: str = "nf4",
    ) -> None:
        """Quantize the frozen BTT core into 4-bit (NF4) via bitsandbytes."""
        if not is_bnb_4bit_available():
            warnings.warn("bitsandbytes not available for 4-bit quantization; skipping QFuRA quantization.")
            return

        l_train = self.fura_l[adapter_name].requires_grad
        r_train = self.fura_r[adapter_name].requires_grad
        if l_train == r_train:
            raise ValueError("QFuRA quantization requires exactly one frozen core.")

        frozen_side = "fura_r" if l_train else "fura_l"
        frozen_param = getattr(self, frozen_side)[adapter_name]
        frozen_shape = tuple(frozen_param.shape)
        frozen_dtype = frozen_param.dtype

        self._qfura_frozen_side[adapter_name] = frozen_side
        self._qfura_frozen_shape[adapter_name] = frozen_shape
        self._qfura_frozen_dtype[adapter_name] = frozen_dtype
        self._qfura_compute_dtype[adapter_name] = compute_dtype

        if layout == "flat":
            flat = frozen_param.detach().reshape(-1, 1).contiguous()
            p4 = bnb.nn.Params4bit(
                flat,
                requires_grad=False,
                quant_type=quant_type,
                compress_statistics=double_quant,
                quant_storage=torch.uint8,
            ).to(device=frozen_param.device)
            self._qfura_frozen_flat[adapter_name] = p4
        else:
            block_list = []
            outer = frozen_shape[0]
            for i in range(outer):
                block = frozen_param[i].detach().contiguous()
                p4 = bnb.nn.Params4bit(
                    block,
                    requires_grad=False,
                    quant_type=quant_type,
                    compress_statistics=double_quant,
                    quant_storage=torch.uint8,
                ).to(device=frozen_param.device)
                self.register_parameter(f"_qfura_{adapter_name}_block_{i}", p4)
                block_list.append(p4)
            self._qfura_frozen_blocks[adapter_name] = block_list

        # Remove the unquantized frozen parameter
        del getattr(self, frozen_side)[adapter_name]

    def _dequantize_frozen_core(self, adapter_name: str) -> torch.Tensor:
        """Dequantize the frozen 4-bit core on-the-fly."""
        compute_dtype = self._qfura_compute_dtype[adapter_name]
        layout = self.quant_layout[adapter_name]
        shape = self._qfura_frozen_shape[adapter_name]

        if layout == "flat":
            p4 = self._qfura_frozen_flat[adapter_name]
            dequanted = bnb.functional.dequantize_4bit(p4.data, quant_state=p4.quant_state)
            return dequanted.reshape(shape).to(compute_dtype)
        else:
            blocks = []
            for p4 in self._qfura_frozen_blocks[adapter_name]:
                deq = bnb.functional.dequantize_4bit(p4.data, quant_state=p4.quant_state)
                blocks.append(deq)
            stacked = torch.stack(blocks, dim=0)
            return stacked.reshape(shape).to(compute_dtype)

    def _get_cores(self, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get (fura_l, fura_r) handling possible QFuRA dequantization."""
        if self.is_quantized.get(adapter_name, False) and adapter_name in self._qfura_frozen_side:
            frozen_dequant = self._dequantize_frozen_core(adapter_name)
            if self._qfura_frozen_side[adapter_name] == "fura_l":
                return frozen_dequant, self.fura_r[adapter_name]
            return self.fura_l[adapter_name], frozen_dequant
        return self.fura_l[adapter_name], self.fura_r[adapter_name]

    def _get_base_weight_before_merge(self) -> torch.Tensor:
        base_weight = self.get_base_layer().weight
        if self._base_weight_before_merge is None:
            self._base_weight_before_merge = base_weight.data.detach().clone().cpu()
        return self._base_weight_before_merge.to(device=base_weight.device, dtype=base_weight.dtype)

    def materialize_dense_weight(self, adapter_name: str) -> torch.Tensor:
        """Compute W_fura = L * S * R as a dense (out_features, in_features) matrix."""
        m = self.m[adapter_name]
        n = self.n[adapter_name]
        a = self.a[adapter_name]
        b = self.b[adapter_name]
        rank = self.rank[adapter_name]

        fura_l, fura_r = self._get_cores(adapter_name)

        # fura_r shape: (n, b, m * rank) -> reshape to (n, b, m, rank) -> permute(2, 0, 3, 1) -> (m, n, rank, b)
        # fura_l shape: (m, rank * n, a) -> reshape to (m, n, rank, a)
        r = fura_r.reshape(n, b, m, rank).permute(2, 0, 3, 1)
        l = fura_l.reshape(m, n, rank, a)

        if adapter_name in self.fura_s:
            l = l * self.fura_s[adapter_name].unsqueeze(-1)

        w_blocks = torch.einsum("mnra,mnrb->mnab", l, r)
        dense_weight = w_blocks.permute(0, 2, 1, 3).reshape(self.out_features, self.in_features)
        return dense_weight

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return delta weight: W_fura - W_base."""
        dense_weight = self.materialize_dense_weight(adapter_name)
        reference_weight = self._get_base_weight_before_merge()
        if getattr(self, "fan_in_fan_out", False):
            reference_weight = reference_weight.T

        delta = dense_weight - reference_weight.to(device=dense_weight.device, dtype=dense_weight.dtype)
        if getattr(self, "fan_in_fan_out", False):
            delta = delta.T
        return delta


class Linear(nn.Module, FuRALayer):
    """
    FuRA implemented in a dense Linear layer.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: FuRAConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        FuRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge active adapter weights into the base weights."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                delta_weight = self.get_delta_weight(active_adapter)

                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += delta_weight.to(orig_weights.device, dtype=orig_dtype)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in merged weights for adapter {active_adapter}"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += delta_weight.to(base_layer.weight.device, dtype=orig_dtype)

                if active_adapter in self.fura_bias and base_layer.bias is not None:
                    if self._base_bias_before_merge is None:
                        self._base_bias_before_merge = base_layer.bias.data.detach().clone().cpu()
                    base_layer.bias.data += self.fura_bias[active_adapter].data.to(base_layer.bias.device, dtype=orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        if self._base_weight_before_merge is not None:
            base_layer.weight.data.copy_(
                self._base_weight_before_merge.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
            )
            self._base_weight_before_merge = None

        if self._base_bias_before_merge is not None and base_layer.bias is not None:
            base_layer.bias.data.copy_(
                self._base_bias_before_merge.to(device=base_layer.bias.device, dtype=base_layer.bias.dtype)
            )
            self._base_bias_before_merge = None

        self.merged_adapters.clear()

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            active_adapters = [a for a in self.active_adapters if a in self._available_adapters]
            if not active_adapters:
                result = self.base_layer(x, *args, **kwargs)
            else:
                # Direct BlockTT forward for active adapters
                orig_shape = x.shape
                # Compute using active adapter
                adapter_name = active_adapters[0]
                m = self.m[adapter_name]
                n = self.n[adapter_name]
                a = self.a[adapter_name]
                b = self.b[adapter_name]
                rank = self.rank[adapter_name]

                fura_l, fura_r = self._get_cores(adapter_name)

                # Reshape x to (batch_n, n, b)
                x_reshaped = x.reshape(-1, n, b)
                batch_n = x_reshaped.shape[0]
                x_t = x_reshaped.transpose(0, 1).contiguous()  # (n, batch_n, b)

                compute_dtype = fura_r.dtype
                x_t = x_t.to(compute_dtype)

                # Step 1: (n, batch_n, b) @ (n, b, m * rank) -> (n, batch_n, m * rank)
                inner = torch.bmm(x_t, fura_r)
                inner = inner.reshape(n, batch_n, m, rank).permute(2, 1, 0, 3).contiguous()  # (m, batch_n, n, rank)

                inner = self.fura_dropout[adapter_name](inner)

                # Step 2: (m, batch_n, n * rank) @ (m, n * rank, a) -> (m, batch_n, a)
                l = fura_l
                if adapter_name in self.fura_s:
                    l = (
                        l.reshape(m, n, rank, a) * self.fura_s[adapter_name].unsqueeze(-1)
                    ).reshape(m, rank * n, a)

                out = torch.bmm(inner.reshape(m, batch_n, rank * n), l)
                out = out.permute(1, 0, 2).contiguous().reshape(*orig_shape[:-1], self.out_features)

                # Add base layer bias or adapter bias if present
                base_layer = self.get_base_layer()
                if base_layer.bias is not None:
                    out = out + base_layer.bias.to(out.dtype)
                if adapter_name in self.fura_bias:
                    out = out + self.fura_bias[adapter_name].to(out.dtype)

                result = out.to(previous_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fura." + rep
