import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

from peft import FuRAConfig, get_peft_model


def train_fura(use_qfura=False):
    model_id = "google/gemma-4-E2B"
    print(f"Loading base model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

    # FuRA / QFuRA Configuration
    print(f"Applying {'QFuRA' if use_qfura else 'FuRA'}...")
    config = FuRAConfig(
        r="full",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        is_quantized=use_qfura,  # True for QFuRA, False for FuRA
        quant_layout="flat",
        train_position="small",
        s_merged_to="keep_trainable",
        decomp_mode="output_one_block",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Load a larger subset of a dataset for better stability
    print("Loading dataset...")
    dataset = load_dataset(
        "Patil/uncensored-chat",
        split="train[:1000]",
    )

    import ast

    def format_example(example):
        messages = ast.literal_eval(example["accepted"])

        text = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                text += f"<start_of_turn>user\n{content}<end_of_turn>\n"
            elif role == "assistant":
                text += f"<start_of_turn>model\n{content}<end_of_turn>\n"

        return {"text": text}

    dataset = dataset.map(
        format_example,
        remove_columns=dataset.column_names,
    )

    # SFT configuration
    sft_config = SFTConfig(
        output_dir="./fura_demo_results",
        # Training
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        # Sequence handling
        dataset_text_field="text",
        max_length=16384,
        # Precision
        bf16=True,
        # Logging / saving
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        # Useful for PEFT training
        remove_unused_columns=False,
    )

    # ------------------------------------------------------------------
    # SFT Trainer
    # ------------------------------------------------------------------
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
    parser.add_argument("--qfura", action="store_true", help="Use QFuRA (quantized) instead of FuRA")
    args = parser.parse_args()

    train_fura(use_qfura=args.qfura)
