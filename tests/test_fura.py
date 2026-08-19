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
import unittest

import pytest
import torch
import torch.nn as nn

from peft import (
    FuRAConfig,
    PeftModel,
    get_peft_model,
)
from peft.tuners.fura.layer import FuRALayer, Linear


class MLP(nn.Module):
    def __init__(self, in_features=32, hidden_features=64, out_features=16, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(in_features, hidden_features, bias=bias)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        return self.lin1(self.relu(self.lin0(x)))


class TestFuRA(unittest.TestCase):
    def setUp(self):
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
        self.assertLess(max_diff, 1e-5, f"Step 0 output mismatch: {max_diff}")

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
                self.assertFalse(module.fura_l["default"].requires_grad)
                self.assertTrue(module.fura_r["default"].requires_grad)
                self.assertTrue(module.fura_s["default"].requires_grad)
                self.assertFalse(module.get_base_layer().weight.requires_grad)

        out = peft_model(self.x)
        loss = out.sum()
        loss.backward()

        for name, module in peft_model.named_modules():
            if isinstance(module, FuRALayer):
                self.assertIsNone(module.fura_l["default"].grad)
                self.assertIsNotNone(module.fura_r["default"].grad)
                self.assertIsNotNone(module.fura_s["default"].grad)

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
                        self.assertTrue(module.fura_l["default"].requires_grad)
                        self.assertFalse(module.fura_r["default"].requires_grad)
                    elif train_pos == "both":
                        self.assertTrue(module.fura_l["default"].requires_grad)
                        self.assertTrue(module.fura_r["default"].requires_grad)

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
            self.assertLess(diff, 1e-4, f"Mode {mode} step 0 mismatch: {diff}")

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
            self.assertLess(diff, 1e-4, f"s_merged_to={s_mode} step 0 mismatch: {diff}")

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
        self.assertLess(diff, 1e-4, f"QR convert_mode step 0 mismatch: {diff}")

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
            self.assertEqual(y_peft.shape, (self.batch_size, self.out_features))

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
        self.assertLess(diff_merged, 1e-5, f"Merged output mismatch: {diff_merged}")

        # Unmerge
        peft_model.unmerge_adapter()
        y_unmerged = peft_model(self.x)
        diff_unmerged = (y_unmerged - y_trained).abs().max().item()
        self.assertLess(diff_unmerged, 1e-5, f"Unmerged output mismatch: {diff_unmerged}")

        # Merge and unload
        merged_model = peft_model.merge_and_unload()
        y_unloaded = merged_model(self.x)
        diff_unloaded = (y_unloaded - y_trained).abs().max().item()
        self.assertLess(diff_unloaded, 1e-5, f"Unloaded output mismatch: {diff_unloaded}")

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
            self.assertLess(diff, 1e-5, f"Disabled adapter output mismatch: {diff}")

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
            self.assertLess(diff, 1e-5, f"Save/load output mismatch: {diff}")

    def test_multi_adapters(self):
        """Test multiple adapters on the same model."""
        model = MLP(self.in_features, self.hidden_features, self.out_features)
        config1 = FuRAConfig(target_modules=["lin0", "lin1"], r="full")
        config2 = FuRAConfig(target_modules=["lin0", "lin1"], r=8)

        peft_model = get_peft_model(model, config1, adapter_name="adapter1")
        peft_model.add_adapter("adapter2", config2)

        peft_model.set_adapter("adapter1")
        self.assertEqual(peft_model.active_adapter, "adapter1")
        y1 = peft_model(self.x)

        peft_model.set_adapter("adapter2")
        self.assertEqual(peft_model.active_adapter, "adapter2")
        y2 = peft_model(self.x)

        self.assertEqual(y1.shape, (self.batch_size, self.out_features))
        self.assertEqual(y2.shape, (self.batch_size, self.out_features))

        # Delete adapter2
        peft_model.delete_adapter("adapter2")
        self.assertNotIn("adapter2", peft_model.peft_config)

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
            self.assertEqual(y.shape, (self.batch_size, self.out_features))

            if bias_option == "all":
                for name, module in peft_model.named_modules():
                    if isinstance(module, FuRALayer):
                        self.assertTrue(module.get_base_layer().bias.requires_grad)
            elif bias_option == "fura_only":
                for name, module in peft_model.named_modules():
                    if isinstance(module, FuRALayer):
                        self.assertTrue(module.fura_bias["default"].requires_grad)

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
        self.assertEqual(y_train.shape, (self.batch_size, self.out_features))
        self.assertEqual(y_eval.shape, (self.batch_size, self.out_features))


if __name__ == "__main__":
    unittest.main()
