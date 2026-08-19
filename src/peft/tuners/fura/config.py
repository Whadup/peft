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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class FuRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FuRAModel`].

    Paper: FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning
    (https://arxiv.org/abs/2605.22869)

    Args:
        r (`Union[int, float, str]`, *optional*, defaults to `"full"`):
            Per-block FuRA rank. Can be:
            - `"full"` (paper default): full rank per block, `min(a, b)`. This is what makes the decomposition lossless
              and the resulting update full-rank; the trainable budget is then determined entirely by the factorization
              (via `input_factorization` / `output_factorization`), not by this argument. With the default
              `decomp_mode="output_one_block"` and `train_position="small"` the trainable count is `|R| + |S| = n * r *
              b + n * r`, which for the automatic near-square factorization is approximately `d ** 1.5` for a `d x d`
              layer.
            - An integer (`int > 0`): fixed rank across blocks. Note that any value below `min(a, b)` gives up the
              full-rank property that distinguishes FuRA from low-rank adapters.
            - A float in `(0, 1)`: target size of *both* cores combined, as a fraction of the base weight's parameter
              count. This is not the trainable fraction: with the default `train_position="small"` only the smaller
              core (plus `S`) is trained, so the trainable fraction is considerably lower. The resulting rank is
              clamped to at least 1, so small values may overshoot the requested budget.
        decomp_mode (`str`, *optional*, defaults to `"output_one_block"`):
            Decomposition mode for Block Tensor-Train (BTT):
            - `"output_one_block"`: m=1, a=out_features, n=in_blocks, b=in_block_size (FuRA paper default).
            - `"input_one_block"`: m=out_blocks, a=out_block_size, n=1, b=in_features.
            - `"square"`: m=out_blocks, a=out_block_size, n=in_blocks, b=in_block_size.
        train_position (`str`, *optional*, defaults to `"small"`):
            Which side of the BTT core to train:
            - `"small"`: Train smaller core, freeze larger core initialized from SVD (FuRA paper default).
            - `"large"`: Train larger core, freeze smaller core.
            - `"both"`: Train both cores.
        s_merged_to (`str`, *optional*, defaults to `"keep_trainable"`):
            How singular values S are treated:
            - `"keep_trainable"`: Kept as an independent trainable parameter tensor `fura_s` (FuRA default).
            - `"keep_frozen"`: Kept as an independent frozen parameter tensor `fura_s`.
            - `"output"`: Merged into output/left core.
            - `"input"`: Merged into input/right core.
            - `"split"`: Square root split between left and right cores.
            - `"trainable"`: Merged into the trainable core.
            - `"frozen"`: Merged into the frozen core.
        convert_mode (`str`, *optional*, defaults to `"svd"`):
            Block factor decomposition algorithm: `"svd"` or `"qr"`.
        init_mode (`str`, *optional*, defaults to `"default"`):
            Scaling init mode: `"default"` or `"mup"`.
        output_factorization (`Optional[tuple[int, int]]`, *optional*, defaults to `None`):
            Custom (out_blocks, out_block_size) factorization for output dimension.
        input_factorization (`Optional[Union[tuple[int, int], str, dict]]`, *optional*, defaults to `None`):
            Custom (in_blocks, in_block_size) factorization for input dimension, or `"head"` / `"closest"`.
        fura_dropout (`float`, *optional*, defaults to `0.0`):
            Dropout probability for FuRA intermediate representation.
        fan_in_fan_out (`bool`, *optional*, defaults to `False`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out), e.g. GPT-2 Conv1D.
        bias (`str`, *optional*, defaults to `"none"`):
            Bias type for FuRA. Can be `'none'`, `'all'` or `'fura_only'`.
        target_modules (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            List of module names or regex expression of the module names to replace with FuRA.
        exclude_modules (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            List of module names or regex expression of the module names to exclude from FuRA.
        init_weights (`Union[bool, str]`, *optional*, defaults to `True`):
            Whether to initialize adapter weights from base layer SVD (`True`) or random (`False`).
        quant_layout (`str`, *optional*, defaults to `"flat"`):
            QFuRA 4-bit quantization layout: `"flat"` or `"per_core_block"`.
        is_quantized (`bool`, *optional*, defaults to `False`):
            Whether to 4-bit quantize the frozen BTT core (QFuRA).
        modules_to_save (`Optional[list[str]]`, *optional*, defaults to `None`):
            List of modules apart from FuRA layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Optional[Union[list[int], int]]`, *optional*, defaults to `None`):
            The layer indexes to transform.
        layers_pattern (`Optional[Union[list[str], str]]`, *optional*, defaults to `None`):
            The layer pattern name.
    """

    r: Union[int, float, str] = field(
        default="full",
        metadata={
            "help": "FuRA rank: 'full' for full-rank SVD, int > 0 for fixed rank, or float in (0, 1) for budget ratio."
        },
    )
    decomp_mode: str = field(
        default="output_one_block",
        metadata={"help": "Decomposition mode: 'output_one_block', 'input_one_block', or 'square'."},
    )
    train_position: str = field(
        default="small",
        metadata={"help": "Which BTT core to train: 'small', 'large', or 'both'."},
    )
    s_merged_to: str = field(
        default="keep_trainable",
        metadata={
            "help": (
                "Where singular values are merged: 'keep_trainable', 'keep_frozen', "
                "'output', 'input', 'split', 'trainable', 'frozen'."
            )
        },
    )
    convert_mode: str = field(
        default="svd",
        metadata={"help": "Factorization method: 'svd' or 'qr'."},
    )
    init_mode: str = field(
        default="default",
        metadata={"help": "Scaling mode: 'default' or 'mup'."},
    )
    output_factorization: Optional[tuple[int, int]] = field(
        default=None,
        metadata={"help": "Optional (out_blocks, out_block_size) custom tuple for out_features factorization."},
    )
    input_factorization: Optional[Union[tuple[int, int], str, dict]] = field(
        default=None,
        metadata={"help": "Optional (in_blocks, in_block_size) tuple, 'head', 'closest', or dict mapping."},
    )
    fura_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for intermediate representation."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set to True if layer stores weight like (fan_in, fan_out)."},
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for FuRA. Can be 'none', 'all' or 'fura_only'."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of module names to replace with FuRA."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of module names to exclude from FuRA."},
    )
    init_weights: Union[bool, str] = field(
        default=True,
        metadata={"help": "Whether to initialize from base layer SVD (True) or random (False)."},
    )
    quant_layout: str = field(
        default="flat",
        metadata={"help": "QFuRA 4-bit layout: 'flat' or 'per_core_block'."},
    )
    is_quantized: bool = field(
        default=False,
        metadata={"help": "Whether to 4-bit quantize the frozen core (QFuRA)."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from FuRA layers to be set as trainable and saved."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "The layer indexes to transform."},
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The layer pattern name."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.FURA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        valid_decomp_modes = {
            "output_one_block",
            "input_one_block",
            "square",
            "input",
            "output",
            "input_block",
            "output_block",
        }
        if self.decomp_mode not in valid_decomp_modes:
            raise ValueError(f"decomp_mode must be one of {valid_decomp_modes}, got {self.decomp_mode!r}")
        # Normalize decomp_mode aliases
        if self.decomp_mode in {"input", "input_block"}:
            self.decomp_mode = "input_one_block"
        elif self.decomp_mode in {"output", "output_block"}:
            self.decomp_mode = "output_one_block"

        valid_train_positions = {"small", "large", "both"}
        if self.train_position not in valid_train_positions:
            raise ValueError(f"train_position must be one of {valid_train_positions}, got {self.train_position!r}")

        valid_s_merged_to = {
            "keep_trainable",
            "keep_frozen",
            "output",
            "input",
            "split",
            "trainable",
            "frozen",
        }
        if self.s_merged_to not in valid_s_merged_to:
            raise ValueError(f"s_merged_to must be one of {valid_s_merged_to}, got {self.s_merged_to!r}")

        valid_convert_modes = {"svd", "qr"}
        if self.convert_mode.lower() not in valid_convert_modes:
            raise ValueError(f"convert_mode must be one of {valid_convert_modes}, got {self.convert_mode!r}")
        self.convert_mode = self.convert_mode.lower()

        valid_quant_layouts = {"flat", "per_core_block"}
        if self.quant_layout not in valid_quant_layouts:
            raise ValueError(f"quant_layout must be one of {valid_quant_layouts}, got {self.quant_layout!r}")

        if self.bias not in {"none", "all", "fura_only"}:
            raise ValueError(f"bias must be one of 'none', 'all', or 'fura_only', got {self.bias!r}")

        # Check rank
        if isinstance(self.r, str):
            if self.r != "full":
                raise ValueError(f"r as string must be 'full', got {self.r!r}")
        elif isinstance(self.r, float):
            if not (0 < self.r < 1):
                raise ValueError(f"r as float must be between 0 and 1, got {self.r}")
        elif isinstance(self.r, int):
            if self.r <= 0:
                raise ValueError(f"r as integer must be > 0, got {self.r}")
        else:
            raise TypeError(f"r must be an int, a float in (0, 1), or 'full', got {type(self.r).__name__}")
