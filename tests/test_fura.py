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

import copy
import tempfile

import pytest
import torch
from torch import nn

from peft import (
    FuRAConfig,
    PeftModel,
    get_peft_model,
)
from peft.tuners.fura.layer import FuRALayer


class MLP(nn.Module):
    def __init__(self, in_features=32, hidden_features=64, out_features=16, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden_features, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


class TestFuRA:
    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.in_features = 32
        self.hidden_features = 64
        self.out_features = 16
        self.batch_size = 4
        self.x = torch.randn(self.batch_size, self.in_features)

    def test_initialization_output_match(self):
        """Test that at initialization, FuRA matches base model output exactly."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        y_base = model(self.x)

        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
        )
        peft_model = get_peft_model(model, config)
        y_peft = peft_model(self.x)

        max_diff = (y_peft - y_base).abs().max().item()
        assert max_diff < 1e-05, f"Step 0 output mismatch: {max_diff}"

    def test_gradient_flow_small_core(self):
        """Test that with train_position='small', large core L is frozen and small core R + S are trainable."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
        )
        peft_model = get_peft_model(model, config)

        # Verify requires_grad
        for name, module in peft_model.named_modules():
            if isinstance(module, FuRALayer):
                assert not module.fura_l["default"].requires_grad
                assert module.fura_r["default"].requires_grad
                assert module.fura_s["default"].requires_grad
                assert not module.get_base_layer().weight.requires_grad

        out = peft_model(self.x)
        loss = out.sum()
        loss.backward()

        for name, module in peft_model.named_modules():
            if isinstance(module, FuRALayer):
                assert module.fura_l["default"].grad is None
                assert module.fura_r["default"].grad is not None
                assert module.fura_s["default"].grad is not None

    def test_train_positions(self):
        """Test train_position='large' and 'both'."""
        for train_pos in ["large", "both"]:
            model = MLP(self.in_features, self.hidden_features, self.out_features)
            config = FuRAConfig(
                target_modules=["lin0", "lin1"],
                r="full",
                decomp_mode="output_one_block",
                train_position=train_pos,
                s_merged_to="keep_trainable",
            )
            peft_model = get_peft_model(model, config)
            for _, module in peft_model.named_modules():
                if isinstance(module, FuRALayer):
                    if train_pos == "large":
                        assert module.fura_l["default"].requires_grad
                        assert not module.fura_r["default"].requires_grad
                    elif train_pos == "both":
                        assert module.fura_l["default"].requires_grad
                        assert module.fura_r["default"].requires_grad

    def test_decomposition_modes(self):
        """Test output_one_block, input_one_block, and square decomposition modes."""
        modes = ["output_one_block", "input_one_block", "square"]
        for mode in modes:
            model = MLP(self.in_features, self.hidden_features, self.out_features)
            y_base = model(self.x)
            config = FuRAConfig(
                target_modules=["lin0", "lin1"],
                r="full",
                decomp_mode=mode,
                train_position="small",
                s_merged_to="keep_trainable",
            )
            peft_model = get_peft_model(model, config)
            y_peft = peft_model(self.x)
            diff = (y_peft - y_base).abs().max().item()
            assert diff < 0.0001, f"Mode {mode} step 0 mismatch: {diff}"

    def test_s_merged_to_modes(self):
        """Test various s_merged_to modes."""
        modes = ["keep_trainable", "keep_frozen", "output", "input", "split"]
        for s_mode in modes:
            model = MLP(self.in_features, self.hidden_features, self.out_features)
            y_base = model(self.x)
            config = FuRAConfig(
                target_modules=["lin0", "lin1"],
                r="full",
                decomp_mode="output_one_block",
                train_position="small",
                s_merged_to=s_mode,
            )
            peft_model = get_peft_model(model, config)
            y_peft = peft_model(self.x)
            diff = (y_peft - y_base).abs().max().item()
            assert diff < 0.0001, f"s_merged_to={s_mode} step 0 mismatch: {diff}"

    def test_convert_mode_qr(self):
        """Test convert_mode='qr'."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        y_base = model(self.x)
        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            convert_mode="qr",
            train_position="small",
            s_merged_to="output",
        )
        peft_model = get_peft_model(model, config)
        y_peft = peft_model(self.x)
        diff = (y_peft - y_base).abs().max().item()
        assert diff < 0.0001, f"QR convert_mode step 0 mismatch: {diff}"

    def test_ranks(self):
        """Test integer rank, float rank, and 'full' rank."""
        for r_val in ["full", 8, 0.25]:
            model = MLP(self.in_features, self.hidden_features, self.out_features)
            config = FuRAConfig(
                target_modules=["lin0", "lin1"],
                r=r_val,
                decomp_mode="output_one_block",
                train_position="small",
                s_merged_to="keep_trainable",
            )
            peft_model = get_peft_model(model, config)
            y_peft = peft_model(self.x)
            assert y_peft.shape == (self.batch_size, self.out_features)

    def test_merge_and_unmerge(self):
        """Test merging weights into base model and unmerging."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        y_base = model(self.x)

        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
        )
        peft_model = get_peft_model(model, config)

        # Train 1 step
        optimizer = torch.optim.SGD(peft_model.parameters(), lr=0.05)
        loss = peft_model(self.x).sum()
        loss.backward()
        optimizer.step()

        y_trained = peft_model(self.x)

        # Merge
        peft_model.merge_adapter()
        y_merged = peft_model(self.x)
        diff_merged = (y_merged - y_trained).abs().max().item()
        assert diff_merged < 1e-05, f"Merged output mismatch: {diff_merged}"

        # Unmerge
        peft_model.unmerge_adapter()
        y_unmerged = peft_model(self.x)
        diff_unmerged = (y_unmerged - y_trained).abs().max().item()
        assert diff_unmerged < 1e-05, f"Unmerged output mismatch: {diff_unmerged}"

        # Merge and unload
        merged_model = peft_model.merge_and_unload()
        y_unloaded = merged_model(self.x)
        diff_unloaded = (y_unloaded - y_trained).abs().max().item()
        assert diff_unloaded < 1e-05, f"Unloaded output mismatch: {diff_unloaded}"

    def test_disable_adapter(self):
        """Test disabling adapter returns original base model output."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        y_base = model(self.x)

        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
        )
        peft_model = get_peft_model(model, config)

        # Train 1 step so adapter diverges from base
        optimizer = torch.optim.SGD(peft_model.parameters(), lr=0.1)
        loss = peft_model(self.x).sum()
        loss.backward()
        optimizer.step()

        with peft_model.disable_adapter():
            y_disabled = peft_model(self.x)
            diff = (y_disabled - y_base).abs().max().item()
            assert diff < 1e-05, f"Disabled adapter output mismatch: {diff}"

    def test_save_and_load(self):
        """Test saving and loading adapter checkpoint."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        base_copy = copy.deepcopy(model)

        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            decomp_mode="output_one_block",
            train_position="small",
            s_merged_to="keep_trainable",
        )
        peft_model = get_peft_model(model, config)

        optimizer = torch.optim.SGD(peft_model.parameters(), lr=0.05)
        loss = peft_model(self.x).sum()
        loss.backward()
        optimizer.step()

        y_trained = peft_model(self.x)

        with tempfile.TemporaryDirectory() as tmpdir:
            peft_model.save_pretrained(tmpdir)
            loaded_model = PeftModel.from_pretrained(base_copy, tmpdir)
            loaded_model.eval()
            y_loaded = loaded_model(self.x)
            diff = (y_loaded - y_trained).abs().max().item()
            assert diff < 1e-05, f"Save/load output mismatch: {diff}"

    def test_multi_adapters(self):
        """Test multiple adapters on the same model."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        config1 = FuRAConfig(target_modules=["lin0", "lin1"], r="full")
        config2 = FuRAConfig(target_modules=["lin0", "lin1"], r=8)

        peft_model = get_peft_model(model, config1, adapter_name="adapter1")
        peft_model.add_adapter("adapter2", config2)

        peft_model.set_adapter("adapter1")
        assert peft_model.active_adapter == "adapter1"
        y1 = peft_model(self.x)

        peft_model.set_adapter("adapter2")
        assert peft_model.active_adapter == "adapter2"
        y2 = peft_model(self.x)

        assert y1.shape == (self.batch_size, self.out_features)
        assert y2.shape == (self.batch_size, self.out_features)

        # Delete adapter2
        peft_model.delete_adapter("adapter2")
        assert "adapter2" not in peft_model.peft_config

    def test_bias_options(self):
        """Test bias='none', 'fura_only', 'all'."""
        for bias_option in ["none", "fura_only", "all"]:
            model = MLP(self.in_features, self.hidden_features, self.out_features, bias=True)
            config = FuRAConfig(
                target_modules=["lin0", "lin1"],
                r="full",
                bias=bias_option,
            )
            peft_model = get_peft_model(model, config)
            y = peft_model(self.x)
            assert y.shape == (self.batch_size, self.out_features)

            if bias_option == "all":
                for name, module in peft_model.named_modules():
                    if isinstance(module, FuRALayer):
                        assert module.get_base_layer().bias.requires_grad
            elif bias_option == "fura_only":
                for name, module in peft_model.named_modules():
                    if isinstance(module, FuRALayer):
                        assert module.fura_bias["default"].requires_grad

    def test_fura_dropout(self):
        """Test fura_dropout."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        config = FuRAConfig(
            target_modules=["lin0", "lin1"],
            r="full",
            fura_dropout=0.2,
        )
        peft_model = get_peft_model(model, config)
        peft_model.train()
        y_train = peft_model(self.x)
        peft_model.eval()
        y_eval = peft_model(self.x)
        assert y_train.shape == (self.batch_size, self.out_features)
        assert y_eval.shape == (self.batch_size, self.out_features)


class TestFuRAConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"init_mode": "deafult"}, "init_mode must be one of"),
            ({"input_factorization": "typo"}, "input_factorization as string must be"),
            ({"input_factorization": (1, 2, 3)}, "input_factorization must be None"),
            ({"input_factorization": {"lin0": 8}}, "input_factorization dict values must be"),
            ({"output_factorization": (1, 2, 3)}, "output_factorization must be None"),
            ({"save_frozen_core": False, "is_quantized": True}, "not supported together with is_quantized"),
            ({"save_frozen_core": False, "init_weights": False}, "requires init_weights=True"),
        ],
    )
    def test_invalid_config_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            FuRAConfig(target_modules=["lin0"], **kwargs)


class TestFuRAInputFactorization:
    @pytest.mark.parametrize(
        "input_factorization, expected",
        [
            (None, (4, 8)),
            ("closest", (4, 8)),
            ((2, 16), (2, 16)),
            ({"lin0": (8, 4)}, (8, 4)),
            ({"unrelated": (8, 4)}, (4, 8)),
        ],
    )
    def test_input_factorization_forms(self, input_factorization, expected):
        model = MLP(in_features=32, hidden_features=64, out_features=16)
        config = FuRAConfig(target_modules=["lin0"], input_factorization=input_factorization)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0
        assert (layer.n["default"], layer.b["default"]) == expected

    def test_input_factorization_mismatch_raises(self):
        model = MLP(in_features=32, hidden_features=64, out_features=16)
        config = FuRAConfig(target_modules=["lin0"], input_factorization=(3, 5))
        with pytest.raises(ValueError, match="does not match in_features"):
            get_peft_model(model, config)

    def test_input_factorization_head_without_model_config_raises(self):
        model = MLP(in_features=32, hidden_features=64, out_features=16)
        config = FuRAConfig(target_modules=["lin0"], input_factorization="head")
        with pytest.raises(ValueError, match="num_attention_heads"):
            get_peft_model(model, config)


class TestFuRAFrozenWeights:
    def test_frozen_names_are_per_layer(self):
        # `frozen_peft_weight_names` is a mutable class attribute, so each layer must own its own dict.
        model = MLP(in_features=32, hidden_features=64, out_features=4, bias=False)
        config = FuRAConfig(target_modules=["lin0", "lin1"], train_position="small")
        peft_model = get_peft_model(model, config)
        lin0 = peft_model.base_model.model.lin0
        lin1 = peft_model.base_model.model.lin1

        assert FuRALayer.frozen_peft_weight_names == {}
        assert lin0.frozen_peft_weight_names is not lin1.frozen_peft_weight_names
        for layer in (lin0, lin1):
            small = "fura_r" if layer.fura_r["default"].numel() <= layer.fura_l["default"].numel() else "fura_l"
            large = "fura_l" if small == "fura_r" else "fura_r"
            assert getattr(layer, small)["default"].requires_grad
            assert not getattr(layer, large)["default"].requires_grad
            assert layer.frozen_peft_weight_names["default"] == (large,)


class TestFuRASingleActiveAdapter:
    def _two_adapter_model(self):
        model = MLP(in_features=32, hidden_features=64, out_features=16)
        peft_model = get_peft_model(model, FuRAConfig(target_modules=["lin0"]))
        peft_model.add_adapter("other", FuRAConfig(target_modules=["lin0"]))
        peft_model.base_model.set_adapter(["default", "other"])
        return peft_model

    def test_forward_with_two_active_adapters_raises(self):
        peft_model = self._two_adapter_model()
        with pytest.raises(ValueError, match="only one adapter can be active"):
            peft_model(torch.randn(4, 32))

    def test_merging_two_adapters_raises(self):
        peft_model = self._two_adapter_model()
        with pytest.raises(ValueError, match="single merged adapter"):
            peft_model.base_model.merge_adapter()


class TestFuRABiasMerge:
    def test_fura_only_bias_merges_into_bias_free_base_layer(self):
        torch.manual_seed(0)
        model = MLP(in_features=32, hidden_features=64, out_features=16, bias=False)
        config = FuRAConfig(target_modules=["lin0"], bias="fura_only")
        peft_model = get_peft_model(model, config)
        with torch.no_grad():
            peft_model.base_model.model.lin0.fura_bias["default"].fill_(0.5)

        x = torch.randn(4, 32)
        expected = peft_model(x)
        merged = peft_model.merge_and_unload()
        assert torch.allclose(merged(x), expected, atol=1e-5, rtol=1e-5)

    def test_unmerge_removes_the_created_bias(self):
        model = MLP(in_features=32, hidden_features=64, out_features=16, bias=False)
        config = FuRAConfig(target_modules=["lin0"], bias="fura_only")
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0

        assert layer.get_base_layer().bias is None
        layer.merge()
        assert layer.get_base_layer().bias is not None
        layer.unmerge()
        assert layer.get_base_layer().bias is None


class TestFuRASaveFrozenCore:
    @pytest.mark.parametrize("save_frozen_core", [True, False])
    def test_roundtrip(self, save_frozen_core, tmp_path):
        torch.manual_seed(0)
        model = MLP(in_features=32, hidden_features=64, out_features=16)
        base_copy = copy.deepcopy(model)
        config = FuRAConfig(target_modules=["lin0", "lin1"], save_frozen_core=save_frozen_core)
        peft_model = get_peft_model(model, config)
        # train a little so that the trainable core actually differs from its initialization
        for layer_name in ("lin0", "lin1"):
            layer = getattr(peft_model.base_model.model, layer_name)
            trainable = "fura_r" if layer.fura_r["default"].requires_grad else "fura_l"
            with torch.no_grad():
                getattr(layer, trainable)["default"].add_(0.01)

        x = torch.randn(4, 32)
        expected = peft_model(x)
        peft_model.save_pretrained(tmp_path)
        reloaded = PeftModel.from_pretrained(base_copy, tmp_path)
        assert torch.allclose(reloaded(x), expected, atol=1e-5, rtol=1e-5)

    def test_frozen_core_is_omitted_from_the_checkpoint(self):
        from peft.utils import get_peft_model_state_dict

        model = MLP(in_features=32, hidden_features=64, out_features=16)
        config = FuRAConfig(target_modules=["lin0"], save_frozen_core=False)
        peft_model = get_peft_model(model, config)
        layer = peft_model.base_model.model.lin0
        frozen = layer.frozen_peft_weight_names["default"]

        state_dict = get_peft_model_state_dict(peft_model)
        assert frozen
        for name in frozen:
            assert not any(f".{name}." in key for key in state_dict)
