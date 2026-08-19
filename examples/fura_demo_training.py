"""Minimal FuRA / QFuRA supervised fine-tuning demo.

Run with:
    python examples/fura_demo_training.py            # FuRA
    python examples/fura_demo_training.py --qfura    # QFuRA (4-bit frozen core, needs bitsandbytes + CUDA)
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from peft import FuRAConfig, get_peft_model


MODEL_ID = "facebook/opt-350m"
DATASET_ID = "timdettmers/openassistant-guanaco"


def train_fura(use_qfura=False):
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)

    print(f"Applying {'QFuRA' if use_qfura else 'FuRA'}...")
    # With r="full" the decomposition is lossless and the trainable budget follows from the block factorization,
    # so the adapter starts out equivalent to the base model.
    config = FuRAConfig(
        r="full",
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        is_quantized=use_qfura,
        quant_layout="flat",
        train_position="small",
        s_merged_to="keep_trainable",
        decomp_mode="output_one_block",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID, split="train[:1000]")

    sft_config = SFTConfig(
        output_dir="./fura_demo_results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        dataset_text_field="text",
        max_length=512,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qfura", action="store_true", help="Use QFuRA (quantized frozen core) instead of FuRA")
    args = parser.parse_args()

    train_fura(use_qfura=args.qfura)
