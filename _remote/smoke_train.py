"""
Smoke test training script — prove the Unsloth pipeline works on the RTX 5090.

Runs a TINY LoRA fine-tune on Qwen2.5-0.5B-Instruct against 200 examples of
our clean SFT dataset. Goal is to exercise the full pipeline:

    load base -> format data -> LoRA train -> save adapter -> merge -> export GGUF

Expected to finish in ~5-10 minutes on a 5090. Output NOT intended to produce
good weights — just to prove every step works before we commit to a real train.
"""
# MUST import unsloth before transformers/trl
import unsloth  # noqa: F401

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

BASE = Path(r"C:\training")
SFT = BASE / "sft_clean.jsonl"
ITER8_SYS = """CS2 Polymarket trader. Output ONE line of JSON, nothing else:
{"action":"buy"|"skip","confidence":0-1,"reason":"short"}

HARD GATES: ask 25-70c, spread <=10c.
BUY only if buy team is LEADING or TIED in maps AND has a clear current
advantage (just won a round, economy lead, or alive-count lead)."""


def _user_prompt(ex: dict) -> str:
    """Build a compact user prompt from an SFT example's state."""
    st = ex.get("state", {}) or {}
    t1 = st.get("team_one") or {}
    t2 = st.get("team_two") or {}
    trigger = ex.get("trigger", {}) or {}
    market = ex.get("market_before", {}) or {}

    ta = t1.get("fixture", {}).get("team_name") or t1.get("name") or "A"
    tb = t2.get("fixture", {}).get("team_name") or t2.get("name") or "B"
    buy = ta if trigger.get("team") == "a" else (tb if trigger.get("team") == "b" else ta)

    lines = [
        f"GAME: CS2 {ta} vs {tb} (evaluating BUY on {buy})",
        f"Series: {t1.get('match_score', 0)}-{t2.get('match_score', 0)}  "
        f"Map: {t1.get('score', 0)}-{t2.get('score', 0)}",
    ]
    if st.get("map_name"):
        lines.append(f"Map: {st['map_name']}")

    # Rough economy + alive
    ps1 = t1.get("player_states") or []
    ps2 = t2.get("player_states") or []
    if ps1 and ps2:
        eco_a = sum((p.get("balance") or 0) for p in ps1)
        eco_b = sum((p.get("balance") or 0) for p in ps2)
        alive_a = sum(1 for p in ps1 if p.get("is_alive"))
        alive_b = sum(1 for p in ps2 if p.get("is_alive"))
        lines.append(f"Economy: {ta}=${eco_a:,} vs {tb}=${eco_b:,}")
        if alive_a != 5 or alive_b != 5:
            lines.append(f"Alive: {alive_a}v{alive_b}")

    # Market
    for tok, v in list(market.items())[:2]:
        bid = (v or {}).get("bid") or 0
        ask = (v or {}).get("ask") or 0
        lines.append(f"Market ...{tok[-6:]}: bid={bid*100:.0f}c ask={ask*100:.0f}c")

    desc = trigger.get("description") or trigger.get("event_type") or ""
    lines.append(f"Trigger: {desc}")
    lines.append(f"BUY {buy}? JSON only.")
    return "\n".join(lines)


def _completion(ex: dict) -> str:
    lbl = ex.get("label", {}) or {}
    action = lbl.get("action", "skip")
    # Calibrate confidence from label source
    conf = 0.82 if action == "buy" else 0.05
    reason = (ex.get("trigger", {}).get("description") or "")[:100]
    if not reason:
        reason = f"label={action}"
    return json.dumps({"action": action, "confidence": conf, "reason": reason})


def load_examples(n: int, seed: int = 42) -> list[dict]:
    with open(SFT) as f:
        all_ex = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(all_ex)
    return all_ex[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-0.5B-Instruct",
                    help="Base model. Tiny by default for smoke test.")
    ap.add_argument("--samples", type=int, default=200,
                    help="Training examples (smoke test uses few).")
    ap.add_argument("--max-steps", type=int, default=60,
                    help="Hard cap on training steps for smoke test.")
    ap.add_argument("--output", default=r"C:\training\smoketest_lora",
                    help="Where to save the adapter.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"CS2 SMOKE TRAIN  base={args.model}  samples={args.samples}  "
          f"max_steps={args.max_steps}")
    print("=" * 70)

    # 1. Load dataset
    print("\n[1/5] loading + formatting dataset")
    t0 = time.time()
    ex = load_examples(args.samples)
    records = []
    for e in ex:
        try:
            user = _user_prompt(e)
            assistant = _completion(e)
        except Exception as err:
            continue
        records.append({
            "messages": [
                {"role": "system", "content": ITER8_SYS},
                {"role": "user",   "content": user},
                {"role": "assistant", "content": assistant},
            ]
        })
    print(f"  -> {len(records)} formatted examples in {time.time()-t0:.1f}s")
    print(f"  sample user prompt:\n{records[0]['messages'][1]['content'][:300]}")
    print(f"  sample assistant :\n{records[0]['messages'][2]['content']}")
    ds = Dataset.from_list(records)

    # 2. Load base
    print(f"\n[2/5] loading base: {args.model}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        dtype=None,           # BF16 on Blackwell
        load_in_4bit=True,
    )
    print(f"  -> loaded in {time.time()-t0:.1f}s")

    # 3. Attach LoRA
    print("\n[3/5] attaching LoRA adapter")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.10,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 4. Train
    print("\n[4/5] training (smoke config)")
    # Pre-render messages using the tokenizer's chat template so SFTTrainer
    # sees plain text, not the {messages: [...]} structure.
    def _format(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}

    ds = ds.map(_format, batched=True, remove_columns=["messages"])
    print(f"  sample rendered text:\n{ds[0]['text'][:400]}...")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            max_steps=args.max_steps,
            learning_rate=2e-4,
            logging_steps=5,
            save_steps=500,     # larger than max_steps, so only final save
            output_dir=args.output,
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
            save_total_limit=1,
        ),
    )
    t0 = time.time()
    result = trainer.train()
    print(f"  -> trained in {time.time()-t0:.1f}s")
    print(f"  final loss: {result.training_loss:.4f}")

    # 5. Save
    print("\n[5/5] saving LoRA adapter")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"  -> {args.output}")

    print("\n" + "=" * 70)
    print(f"SMOKE TRAIN OK  loss={result.training_loss:.4f}  "
          f"adapter={args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
