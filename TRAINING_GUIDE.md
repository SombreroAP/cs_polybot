# Training Guide — CS2 arbitrage model fine-tune on RTX 5090

A pragmatic playbook for when + how to fine-tune a model on the SFT dataset we've been building.

## TL;DR

**Don't train yet.** Collect 3 more weeks of live VPS data, then run the evaluator first. Only fine-tune when the audit flips all verdicts to ✅ AND the evaluator shows today's production model has a clear weakness worth targeting.

Until then, the best move is **prompt iteration + few-shot example injection**. We already shootout-verified that a clever prompt on Haiku beats every fine-tuned local model we tried.

---

## 1. Where the data is now (2026-04-18)

Run anytime to refresh this:
```bash
python data/processor.py audit
```

Current state (after SFT rebuild with dedupe):

| Check | Status | Value |
|---|---|---|
| Volume | ⚠️ | 124 unique matches, 6,654 examples |
| Class balance | ✅ | 10.7:1 skip:buy (healthy) |
| Features rich | ✅ | 95% player_states + HP + economy + map |
| Match concentration | ✅ | No single match > 5% of data |
| Duplicates | ✅ | 0% rapid-fire pairs (dedupe in processor) |
| Temporal spread | ⚠️ | Only **5 days** — high meta-specific overfit risk |

**Gating verdicts before pulling the training trigger:**
1. `TRAIN_READY_VOLUME` flips ✅ at **150+ unique matches**
2. `TEMPORAL` flips ✅ at **30+ days of data**

VPS records ~5 CS2 matches per day when live events run. Expect both to flip by **mid-May 2026** on current pace.

---

## 2. What "perfect training data" looks like

Target state before we hit `train_cs2_lora.py`:

```
✅  300-500 unique matches         (diversity)
✅  30,000+ clean SFT examples      (volume, post-dedupe)
✅  4-6 weeks of calendar spread    (temporal spread)
✅  5-10% BUY, 90-95% SKIP balance  (matches expected live distribution)
✅  95%+ feature completeness       (player_states, HP, economy, map)
✅  15%+ strong buys (PnL ≥ 20%)    (label quality: clear edges, not noise)
✅  0% rapid-fire duplicates        (we already enforce this)
⚠️  No target leakage               (needs manual verification — see §6)
⚠️  Hold-out set: ~20% unseen matches for eval
```

### Features we want in every training example

- **Map state**: map_score_a/b, round_score_a/b, side_a, map_name, round_phase, round_time_remaining, is_bomb_planted
- **Per-player**: health, balance (money), kills_in_round, deaths_in_round, is_alive, has_bomb, has_defuse_kit, primary_weapon
- **Team-level derivations**: total_economy, alive_count, avg_hp, kill_lead, awp_holders
- **Market**: bid/ask/spread for both tokens, liquidity, 30s/60s bid momentum
- **Trigger**: event_type, team, description
- **Hindsight label**: action + realized PnL in ±90s window + what the winning side did

Most of these come from rich bo3.gg snapshots — the pipeline already captures them when the provider sends them. A few derivations (team-level aggregates, bid momentum windows) are added by `data/processor.py` when it normalizes recordings.

---

## 3. The evaluator (our compass before/after training)

```bash
# Single model on 200 held-out examples
python evaluator.py --model haiku-4-5 --samples 200

# Head-to-head comparison
python evaluator.py --compare haiku-4-5 ollama://qwen3:30b-a3b --samples 200

# Full run (big, slow, use only when you need to commit to a model)
python evaluator.py --model haiku-4-5 --samples all
```

Outputs:

- Per-example jsonl: `data/evals/<model>_<timestamp>.jsonl` (one row per decision)
- Cached responses in `data/evals/cache.sqlite` — reruns are free
- Console summary: accuracy, precision/recall on the BUY class, simulated PnL, confusion matrix, latency p50/p95, cost

### Metrics and what they mean

| Metric | Meaning | What good looks like |
|---|---|---|
| `accuracy` | % predictions matching hindsight label | >92% (baseline: 92% by always predicting SKIP given 92% of labels are skip) |
| `precision_buy` | of BUYs the model made, % that were right | **>80%** — the North Star |
| `recall_buy` | of true BUY opportunities, % the model caught | 30-60% (low is OK — we prefer fewer higher-quality calls) |
| `sim_pnl_pct_on_buys` | sum of hindsight PnL% on every BUY prediction | **positive and ≥ 5× the skip baseline** |
| `avg_pnl_per_buy_pct` | mean hindsight PnL per BUY call | **>15%** (clears fees + slippage) |
| `total_cost_usd` | API cost on the evaluation run | for ranking cost-vs-quality |

### Gate to deploy a new model

A new candidate must beat the current production on the held-out test set by:
- Precision_buy: **at least +3 percentage points**
- Sim PnL: **at least +20%**
- Latency p95: **within 2× of current** (if slower, fine-grained cost/benefit)

---

## 4. Which models to train (ranked)

All candidates assume QLoRA fine-tune on your RTX 5090 (32 GB Blackwell). Ordered by expected value for our use case:

### 🥇 Qwen2.5-14B-Instruct (primary pick)

- **Size**: 14B params, ~9 GB at Q4, ~18 GB with LoRA adapters in FP16
- **Why**: same family as our baseline qwen3 + deepseek-r1:14b. Instruct-tuned = follows the iter8 prompt well. Strong at structured JSON.
- **LoRA config**: r=32, alpha=64, dropout=0.10, target q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
- **Training time**: ~2 hours for 2 epochs on 30k examples
- **Ollama-friendly**: easy GGUF export, deploy as `cs2-qwen14b` tag

### 🥈 deepseek-r1-distill-qwen-14b (reasoning variant)

- **Size**: 14B, but ships with chain-of-thought reasoning already baked in
- **Why**: our local shootout leader was `deepseek-r1:14b` — a distilled qwen2.5 with reasoning. Fine-tuning this keeps the reasoning and adds task specificity.
- **Risk**: reasoning models resist SFT — you may end up fighting the built-in thinking templates. Needs custom chat template.
- **Experiment first**: try ONLY after Qwen2.5-14B-Instruct is working.

### 🥉 Llama-3.1-8B-Instruct (lightweight ensemble partner)

- **Size**: 8B, ~6 GB at Q4
- **Why**: faster inference (<200ms on 5090), ideal as the second voice in a 2-model ensemble (vote with Haiku/Qwen14B). Our shootout showed llama3.1:8b at +$1.06 — cheap floor.
- **Training time**: ~45 min on 30k examples
- **Good for**: low-latency pre-filter before a more expensive Haiku/Qwen14B decision.

### ❌ What NOT to train on

- **Qwen3-30B-A3B** — MoE models resist LoRA fine-tuning; adapter rarely sticks to the gated experts you care about.
- **Llama-3.3-70B** — won't fit in 32 GB with reasonable training config.
- **Anything > 32B dense** — needs multi-GPU.
- **Base (non-instruct) models** — we'd have to teach JSON output from scratch; 10× the data required.

---

## 5. Training toolchain (on RTX 5090)

Install on the gaming PC ONE TIME:

```bash
# Fresh conda/venv recommended
python -m venv ~/unsloth-env
source ~/unsloth-env/bin/activate       # or Scripts\activate on Windows
pip install --upgrade pip
pip install "unsloth[cu128] @ git+https://github.com/unslothai/unsloth"
pip install datasets trl transformers peft bitsandbytes accelerate
# Optional for GGUF export → Ollama:
pip install sentencepiece protobuf
```

Why Unsloth (over HuggingFace PEFT or Axolotl):
- 2× faster on consumer GPUs
- BF16 native on Blackwell (no FP16 dance)
- Drop-in QLoRA that "just works"
- Direct GGUF export

---

## 6. The training script (skeleton — don't run until audit = all ✅)

```python
# train_cs2_lora.py — run on the gaming PC
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-14B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,           # QLoRA — 4-bit base, full-precision adapter
)

# 2. Attach LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32, lora_alpha=64, lora_dropout=0.10,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",
)

# 3. Load dataset (generate with `python data/make_unsloth_dataset.py`)
ds = load_dataset("json", data_files="data/training/unsloth.jsonl", split="train")
ds = ds.train_test_split(test_size=0.10, seed=42)

# 4. Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,    # effective batch = 16
        num_train_epochs=2,                # STOP at 2
        learning_rate=1e-4,
        warmup_ratio=0.05,
        logging_steps=25,
        eval_strategy="steps", eval_steps=100,
        save_steps=500,
        bf16=True,
        optim="adamw_8bit",
        output_dir="./cs2-qwen-lora",
    ),
)
trainer.train()
model.save_pretrained("./cs2-qwen-lora-final")

# 5. Export to GGUF + Ollama
import subprocess
subprocess.run([
    "python", "-m", "unsloth.save_to_gguf",
    "--model", "./cs2-qwen-lora-final",
    "--output", "./cs2-qwen.gguf",
    "--quantization", "Q4_K_M",
])
```

### Target leakage audit (manual, one-time, before first real run)

Before any serious train:
1. Open 3 random examples from `data/training/sft.jsonl`.
2. Verify: does the `state` block show only the round the trigger fired IN, and not the OUTCOME of that round? (The score should be the score BEFORE the trigger event, not after.)
3. Run `python evaluator.py --model ollama://qwen3:30b-a3b --samples 50` on a RAW qwen (no training). If it already gets 100% accuracy, the training data is leaking the answer — stop immediately.

---

## 7. Pipeline diagram

```
                  ┌────────────────────┐
  [live bots]────▶│ recorder (VPS)     │────▶ data/recordings/*.jsonl
                  └──────┬─────────────┘
                         │ data/processor.py ingest
                         ▼
                  ┌────────────────────┐
                  │ data/recordings/   │ (unified corpus)
                  └──────┬─────────────┘
                         │ processor.py process-all
                         ▼
                  ┌────────────────────┐
                  │ data/processed/    │ (normalized event streams)
                  └──────┬─────────────┘
                         │ processor.py build-sft (with dedupe)
                         ▼
                  ┌────────────────────┐
                  │ data/training/     │ sft.jsonl + rl.jsonl
                  └──────┬─────────────┘
              ┌──────────┴─────────┐
              ▼                    ▼
     ┌─────────────┐        ┌──────────────────┐
     │ evaluator.py│◀───────│ candidate model  │   (Haiku / Sonnet / Qwen / ...)
     │ (read-only) │        └──────────────────┘
     └──────┬──────┘
            │ beats production?
            ▼
     ┌──────────────────────────────────────┐
     │ IF YES: promote to .env; redeploy    │
     │ IF NO:  stay with incumbent          │
     └──────────────────────────────────────┘
```

Training (only when audit all-green) adds one more branch:

```
  data/training/sft.jsonl ─► train_cs2_lora.py ─► GGUF ─► ollama create ─► evaluator.py
```

---

## 8. Timeline (realistic)

| Week | Action | Gate |
|---|---|---|
| **Now** | Evaluator live, data audit green on quality metrics | 124 matches, 5 days |
| **Week 1** | Collect live data. Twice a week: `processor.py ingest && processor.py audit` | ~35 more matches |
| **Week 2** | Run `evaluator.py --compare haiku-4-5 sonnet-4-5 ollama://qwen3:30b-a3b --samples 500` as baseline | Establish scoreboard |
| **Week 3** | Audit flips TRAIN_READY_VOLUME to ✅ (150+ matches); train FIRST small-batch model | Qwen2.5-14B on ~20k clean examples |
| **Week 4** | Evaluate trained model. Promote only if it beats Haiku | Deploy decision |
| Ongoing | Monthly: re-audit, re-evaluate, retrain if corpus grew 30%+ | |

---

## 9. Small-batch test (optional, now — pressure-test the pipeline)

If you want to prove the training pipeline works end-to-end WITHOUT waiting:

```bash
# On the gaming PC
pip install unsloth datasets trl
python -c "
from unsloth import FastLanguageModel
m, t = FastLanguageModel.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', load_in_4bit=True)
print('loaded', m.config.name_or_path)
"
```

If that line works, the toolchain is good. Then run a 100-step training on a TINY subset (5 min on a 5090) — just to shake out dataset format + export errors. No expectation of usable weights.

I'll wire this up as `train_smoketest.py` when you want it. For now, the priority is letting data accumulate.

---

## 10. Summary

**What I'd tell you if you asked me for one-sentence advice:**

> Keep the VPS recording, run `python data/processor.py audit` weekly, and don't touch `train_cs2_lora.py` until that audit comes back green on volume and temporal. Use `evaluator.py` relentlessly in the meantime — it tells you what your current prompts and models are actually doing.

The evaluator is the tool that makes everything else rigorous. Training without it is gambling; training with it is engineering.
