"""
Real training with BUY-class oversampling to fight the 9:1 skip:buy imbalance.

The previous run's model memorized "always skip" because that's correct 92% of
the time. This run duplicates each BUY example N times (N=8 by default) so the
model sees buys and skips in roughly equal proportions.

Everything else (base model, hyperparams, split) stays identical for apples-to-
apples comparison.
"""
import unsloth  # noqa: F401 — must come first

import argparse
import json
import random
import time
from pathlib import Path
from collections import Counter

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
    conf = 0.82 if action == "buy" else 0.05
    reason = (ex.get("trigger", {}).get("description") or "")[:100] or f"label={action}"
    return json.dumps({"action": action, "confidence": conf, "reason": reason})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
    ap.add_argument("--buy-oversample", type=int, default=8,
                    help="duplicate each BUY example N times to balance classes")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.10)  # lower than 0.15 now
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--output", default=r"C:\training\cs2_qwen14b_weighted_v1")
    args = ap.parse_args()

    print("=" * 70)
    print(f"WEIGHTED TRAIN  base={args.model}")
    print(f"  buy_oversample={args.buy_oversample}x")
    print(f"  epochs={args.epochs}  rank={args.lora_rank}  dropout={args.lora_dropout}")
    print(f"  effective_batch={args.batch*args.grad_accum}")
    print("=" * 70)

    # 1. Load + split by match_id (same split as unweighted run for apples-to-apples)
    print("\n[1/6] loading + oversampling dataset")
    with open(SFT) as f:
        all_ex = [json.loads(line) for line in f]
    rng = random.Random(42)
    rng.shuffle(all_ex)

    match_ids = sorted({e["match_id"] for e in all_ex})
    split = int(len(match_ids) * 0.80)
    train_match_ids = set(match_ids[:split])

    train_ex = [e for e in all_ex if e["match_id"] in train_match_ids]
    test_ex = [e for e in all_ex if e["match_id"] not in train_match_ids]

    # Count original distribution
    orig = Counter(e["label"]["action"] for e in train_ex)
    print(f"  original train split: {dict(orig)}")

    # Oversample buys
    balanced = []
    for e in train_ex:
        if e["label"]["action"] == "buy":
            balanced.extend([e] * args.buy_oversample)
        else:
            balanced.append(e)
    rng.shuffle(balanced)

    bal = Counter(e["label"]["action"] for e in balanced)
    print(f"  after {args.buy_oversample}x buy oversample: {dict(bal)}")
    print(f"  skip:buy ratio now: {bal['skip']/max(1,bal['buy']):.2f}:1  (was {orig['skip']/max(1,orig['buy']):.2f}:1)")

    # Format
    records = []
    for e in balanced:
        try:
            records.append({"messages": [
                {"role": "system", "content": ITER8_SYS},
                {"role": "user",   "content": _user_prompt(e)},
                {"role": "assistant", "content": _completion(e)},
            ]})
        except Exception:
            continue
    ds = Dataset.from_list(records)
    print(f"  total train examples: {len(ds)}  (held-out test: {len(test_ex)})")

    # 2. Load base
    print(f"\n[2/6] loading base: {args.model}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  -> loaded in {time.time()-t0:.1f}s")

    # 3. LoRA
    print("\n[3/6] attaching LoRA")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 4. Tokenize via chat template
    print("\n[4/6] rendering chat template")
    def _format(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}
    ds = ds.map(_format, batched=True, remove_columns=["messages"])
    print(f"  sample render:\n{ds[0]['text'][:300]}")

    # 5. Train
    print(f"\n[5/6] training ({args.epochs} epoch)")
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            save_steps=1000000,
            output_dir=args.output,
            bf16=True,
            optim="adamw_8bit",
            report_to="none",
            save_total_limit=1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
        ),
    )
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0
    print(f"  -> trained in {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  final train_loss: {result.training_loss:.4f}")

    # 6. Save
    print("\n[6/6] saving adapter + merged 16-bit")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    merged_out = args.output + "_merged"
    model.save_pretrained_merged(merged_out, tokenizer, save_method="merged_16bit")
    print(f"  adapter: {args.output}")
    print(f"  merged : {merged_out}")

    print("\n" + "=" * 70)
    print(f"TRAIN OK  loss={result.training_loss:.4f}  elapsed={elapsed/60:.1f}min")
    print("=" * 70)


if __name__ == "__main__":
    main()
