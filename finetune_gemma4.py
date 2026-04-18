#!/usr/bin/env python3
"""
Fine-tune Gemma 4 for Esports Betting on RTX 5090.

Run this on your Windows PC with the RTX 5090.

Setup (run once):
    pip install unsloth datasets trl peft transformers torch

Usage:
    python finetune_gemma4.py --data alpaca_format.jsonl
    python finetune_gemma4.py --data alpaca_format.jsonl --epochs 5 --lr 1e-4

After training:
    1. The model saves to ./gemma4-esports-lora/
    2. Run: python finetune_gemma4.py --export (converts to GGUF for Ollama)
    3. Run: ollama create gemma4-esports -f Modelfile.esports
"""
import argparse
import json
import os
import sys


def train(data_path: str, epochs: int = 3, lr: float = 2e-4, batch_size: int = 4):
    """Fine-tune Gemma 4 with LoRA on esports betting data."""
    print("="*60)
    print("GEMMA 4 ESPORTS FINE-TUNING")
    print("="*60)

    # Check GPU
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.0f}GB VRAM)")
    else:
        print("WARNING: No GPU found! Training will be very slow.")
        sys.exit(1)

    # Load model
    print("\nLoading Gemma 4 8B (4-bit quantized)...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2-9b-it-bnb-4bit",  # Gemma 4 base
        max_seq_length=2048,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )
    print(f"Model loaded! Parameters: {model.num_parameters():,}")

    # Add LoRA adapters (trainable layers on top of frozen base)
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                  # LoRA rank — higher = more capacity, more VRAM
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves 60% VRAM
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Load training data
    print(f"\nLoading training data from {data_path}...")
    from datasets import Dataset

    raw_data = []
    with open(data_path) as f:
        for line in f:
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(raw_data):,} training examples")

    # Format for chat template
    SYSTEM_PROMPT = (
        "You are an esports latency arbitrage trader. "
        "Analyze the game state and decide: BUY or SKIP. "
        "Reply ONLY with JSON: {\"action\":\"buy_a/buy_b/skip\","
        "\"confidence\":0.0-1.0,\"reason\":\"brief\"}"
    )

    def format_example(example):
        """Convert to chat format for training."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    # Training
    print(f"\nStarting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    print("This should take ~15-30 minutes on your RTX 5090...")

    from trl import SFTTrainer
    from transformers import TrainingArguments

    output_dir = "gemma4-esports-lora"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=True,  # pack short examples together for efficiency
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
        ),
    )

    # Train
    print("\n" + "="*60)
    print("TRAINING STARTED")
    print("="*60)
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}/")

    # Quick test
    print("\n" + "="*60)
    print("TESTING FINE-TUNED MODEL")
    print("="*60)
    FastLanguageModel.for_inference(model)

    test_prompts = [
        "GAME: CS2 | Team A vs Team B\nMap: de_dust2 | Round: 8-3 | Series: 0-0\nEconomy: Team A=$24,000 | Team B=$4,200\nMarket: bid=62c ask=66c spread=6%\nShould we BUY Team A?",
        "GAME: DOTA2 | OG vs Secret\nKills: OG 25 - 10 Secret | 30min\nGOLD LEAD: OG +15,000 gold\nMarket: bid=72c ask=76c spread=5%\nShould we BUY OG?",
        "GAME: CS2 | NaVi vs FaZe\nMap: de_mirage | Round: 6-6 | Series: 0-0\nEconomy: NaVi=$8,000 | FaZe=$8,500\nMarket: bid=49c ask=53c spread=8%\nShould we BUY NaVi?",
    ]

    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs, max_new_tokens=100,
            temperature=0.1, do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"Response: {response[:120]}")

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Export to GGUF: python finetune_gemma4.py --export")
    print(f"  2. Create Ollama model: ollama create gemma4-esports -f Modelfile.esports")
    print(f"  3. Update your Mac bot: change OLLAMA_MODEL=gemma4-esports in .env")


def export_to_ollama():
    """Export the fine-tuned LoRA model to GGUF format for Ollama."""
    print("Exporting to GGUF for Ollama...")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="gemma4-esports-lora",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Save as GGUF (Q4_K_M quantization — best speed/quality balance)
    model.save_pretrained_gguf(
        "gemma4-esports-gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Create Modelfile for Ollama
    modelfile_content = """FROM ./gemma4-esports-gguf/unsloth.Q4_K_M.gguf

SYSTEM "You are an esports latency arbitrage trader fine-tuned on real CS2/Dota2/LoL/Valorant match data. Analyze game state and reply with JSON: {\\"action\\":\\"buy_a/buy_b/skip\\",\\"confidence\\":0.0-1.0,\\"reason\\":\\"brief\\"}. RULES: Gold > kills in Dota2. Gun rounds > eco in CS2. Spread >12% = skip. Price outside 30-75c = skip."

PARAMETER temperature 0
PARAMETER num_predict 120
"""

    with open("Modelfile.esports", "w") as f:
        f.write(modelfile_content)

    print(f"\nExported! Now run:")
    print(f"  ollama create gemma4-esports -f Modelfile.esports")
    print(f"\nThen on your Mac, update .env:")
    print(f"  OLLAMA_MODEL=gemma4-esports")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for esports betting")
    parser.add_argument("--data", default="alpaca_format.jsonl", help="Training data JSONL file")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--export", action="store_true", help="Export trained model to GGUF/Ollama")
    args = parser.parse_args()

    if args.export:
        export_to_ollama()
    else:
        train(args.data, args.epochs, args.lr, args.batch)


if __name__ == "__main__":
    main()
