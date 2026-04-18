"""
Real training run — Qwen2.5-14B-Instruct + full clean SFT dataset.

HONEST EXPECTATIONS:
  Dataset is small (53 matches, 1.8 days temporal) — model WILL overfit.
  Purpose of this run: exercise the full 14B pipeline end-to-end, get a
  trained adapter we can score with evaluator.py. If it beats Haiku on the
  held-out test set, great. If it loses, that's valuable data too.

CONFIG:
  base:         unsloth/Qwen2.5-14B-Instruct-bnb-4bit (QLoRA, ~8 GB VRAM)
  samples:      all 4981 clean examples
  epochs:       1 (one pass — more = certain overfit on our tiny set)
  lora_rank:    32
  lora_dropout: 0.15 (high — fights overfit)
  batch:        effective 16 (bs=2, grad_accum=8)

TIME ESTIMATE: ~40-60 min on RTX 5090 for 1 epoch
"""
import unsloth  # noqa: F401 — must come first

import argparse
import json
import random
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


def load_examples(n: int | None, seed: int = 42) -> list[dict]:
    with open(SFT) as f:
        all_ex = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(all_ex)
    return all_ex if n is None else all_ex[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
                    help="Base model. 4-bit unsloth variant for QLoRA on 5090.")
    ap.add_argument("--samples", type=int, default=0,
                    help="0 = all examples")
    ap.add_argument("--epochs", type=int, default=1,
                    help="1 by default to fight overfitting on small corpus")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.15,
                    help="High (0.15) because small corpus = high overfit risk")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--output", default=r"C:\training\cs2_qwen14b_v1")
    args = ap.parse_args()

    print("=" * 70)
    print(f"REAL TRAIN  base={args.model}")
    print(f"  epochs={args.epochs}  rank={args.lora_rank}  dropout={args.lora_dropout}  lr={args.lr}")
    print(f"  effective_batch={args.batch*args.grad_accum}  max_seq={args.max_seq_len}")
    print("=" * 70)

    print("\n[1/6] loading dataset")
    t0 = time.time()
    n = args.samples if args.samples > 0 else None
    ex = load_examples(n)
    records = []
    for e in ex:
        try:
            records.append({"messages": [
                {"role": "system", "content": ITER8_SYS},
                {"role": "user",   "content": _user_prompt(e)},
                {"role": "assistant", "content": _completion(e)},
            ]})
        except Exception:
            continue
    print(f"  -> {len(records)} formatted examples in {time.time()-t0:.1f}s")

    # 80/20 train/test split on match_id so eval set is truly held-out
    # (we'll run evaluator.py on the test set after training)
    match_ids = sorted({e["match_id"] for e in ex})
    split = int(len(match_ids) * 0.80)
    train_match_ids = set(match_ids[:split])
    train_records = []
    test_count = 0
    for e, r in zip(ex, records):
        if e["match_id"] in train_match_ids:
            train_records.append(r)
        else:
            test_count += 1
    ds = Dataset.from_list(train_records)
    print(f"  train: {len(ds)}  held-out test (for evaluator.py later): {test_count}")

    print(f"\n[2/6] loading base: {args.model}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"  -> loaded in {time.time()-t0:.1f}s")

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

    print("\n[4/6] rendering chat template")
    def _format(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False,
                                                add_generation_prompt=False)
                 for m in batch["messages"]]
        return {"text": texts}
    ds = ds.map(_format, batched=True, remove_columns=["messages"])
    print(f"  sample render:\n{ds[0]['text'][:300]}")

    print(f"\n[5/6] training ({args.epochs} epoch{'s' if args.epochs!=1 else ''})")
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
            save_steps=1000000,       # effectively only save at end
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

    print("\n[6/6] saving adapter + merged 16-bit model")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    merged_out = args.output + "_merged"
    model.save_pretrained_merged(merged_out, tokenizer, save_method="merged_16bit")
    print(f"  adapter: {args.output}")
    print(f"  merged : {merged_out}")

    print("\n" + "=" * 70)
    print(f"TRAIN OK  loss={result.training_loss:.4f}  elapsed={elapsed/60:.1f}min")
    print(f"  evaluate with: python evaluator.py --model ollama://<tag> --samples 200")
    print("=" * 70)


if __name__ == "__main__":
    main()
