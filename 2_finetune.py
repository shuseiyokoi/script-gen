#!/usr/bin/env python3
"""
Step 2: Fine-tune a model on the generated dataset.

Uses HuggingFace TRL SFTTrainer with QLoRA (4-bit quantization via bitsandbytes)
so the fine-tune runs on a single consumer GPU.

Default base model: google/gemma-2-2b-it  (small, fast, good quality)
Override: BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python 2_finetune.py

Input:  dataset/finetune.jsonl
Output: output/model/  (LoRA adapter weights)
"""

import os
from pathlib import Path

# datasets (pyarrow) must be imported before torch — importing torch first causes
# a native crash (segfault) on Windows due to a conflicting OpenMP runtime.
from datasets import load_dataset
from dotenv import load_dotenv
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-2b-it")
DATASET_PATH = os.getenv("DATASET_PATH", "dataset/finetune.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output/model")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
HF_TOKEN = os.getenv("HF_TOKEN", "")


def load_quantization_config() -> BitsAndBytesConfig | None:
    if not torch.cuda.is_available():
        print("No CUDA GPU detected — training in fp32/bf16 on CPU (slow).")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main() -> None:
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run 1_generate_dataset.py first."
        )

    print(f"Base model : {BASE_MODEL}")
    print(f"Dataset    : {DATASET_PATH}")
    print(f"Output     : {OUTPUT_DIR}")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ──────────────────────────────────────────────────────────────────
    bnb_config = load_quantization_config()
    model_kwargs = dict(
        token=HF_TOKEN or None,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **model_kwargs)
    model.config.use_cache = False

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"Loaded {len(dataset)} training examples")

    def fold_system_into_user(example: dict) -> dict:
        # Gemma's chat template rejects a "system" role — fold it into the user turn.
        messages = example["messages"]
        system = next((m["content"] for m in messages if m["role"] == "system"), None)
        rest = [m for m in messages if m["role"] != "system"]
        if system and rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": f"{system}\n\n{rest[0]['content']}"}
        return {"messages": rest}

    dataset = dataset.map(fold_system_into_user)

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    # ── LoRA config ────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )

    # ── Training args ──────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_available(),
        fp16=False,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        logging_steps=10,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        dataset_text_field=None,
        load_best_model_at_end=True,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\nStarting training …")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR} …")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
