<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning

FuRA factorizes each target weight into a Block Tensor-Train (BTT) pair of cores, obtained from a per-block SVD of the
pretrained weight. One core is kept frozen and the other is trained, so the adapted layer expresses a *full-rank*
update while training only a small fraction of the parameters. Because the layer materializes the whole weight
(`W = L @ S @ R`) instead of adding a low-rank delta, FuRA supports one active adapter per layer at a time.

Paper: [FuRA: Full-Rank Parameter-Efficient Fine-Tuning with Spectral Preconditioning](https://arxiv.org/abs/2605.22869).

## Choosing the parameter budget

With the default `r="full"`, the rank is `min(a, b)` per block and the decomposition is lossless. The trainable budget
is then set by the block factorization rather than by `r`: with `decomp_mode="output_one_block"` and
`train_position="small"` the trainable count is `|R| + |S| = n * r * b + n * r`, roughly `d ** 1.5` for a `d x d` layer
under the automatic near-square factorization. Use `input_factorization` / `output_factorization` to control it
explicitly. Setting `r` to an integer below `min(a, b)` gives up the full-rank property.

`train_position` and `s_merged_to` select which of the design corners from the paper is used: training the input core
with the singular values kept as a separate trainable tensor is the default.

FuRA currently has the following constraint:

- Only `nn.Linear` layers are supported.

## Quickstart

```python
from transformers import AutoModelForSequenceClassification

from peft import FuRAConfig, TaskType, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

peft_config = FuRAConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## FuRAConfig

[[autodoc]] tuners.fura.config.FuRAConfig

## FuRAModel

[[autodoc]] tuners.fura.model.FuRAModel

## FuRALayer

[[autodoc]] tuners.fura.layer.FuRALayer
