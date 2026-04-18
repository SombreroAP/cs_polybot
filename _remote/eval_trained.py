"""
Evaluate the trained Qwen2.5-14B on the held-out test split.

Runs LOCALLY on the gaming PC (where the merged model lives) via transformers
directly — no Ollama / GGUF needed. Loads the merged fp16 model, runs each
held-out example, parses responses, compares to label.

Output: JSON report to C:\\training\\eval_trained.json

Usage:
    venv_smoke\\Scripts\\python.exe eval_trained.py
"""
import unsloth  # noqa: F401 — must come first

import json
import re
import time
from pathlib import Path

from unsloth import FastLanguageModel

BASE = Path(r"C:\training")
MERGED_MODEL = BASE / "cs2_qwen14b_v1_merged"
ADAPTER_MODEL = BASE / "cs2_qwen14b_v1"           # fallback — adapter only
SFT = BASE / "sft_clean.jsonl"
OUT = BASE / "eval_trained.json"

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


def parse_response(text: str) -> dict:
    if not text:
        return {"action": "parse_error", "confidence": 0.0, "reason": "empty"}
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"action": "parse_error", "confidence": 0.0, "reason": text[:60]}
    try:
        d = json.loads(text[start:end+1])
    except Exception:
        return {"action": "parse_error", "confidence": 0.0, "reason": text[start:end+1][:60]}
    return {
        "action": (d.get("action") or "skip").lower(),
        "confidence": float(d.get("confidence") or 0.0),
        "reason": (d.get("reason") or "")[:200],
    }


def main():
    # Deterministic held-out split by match_id (same logic as real_train.py)
    with open(SFT) as f:
        all_ex = [json.loads(line) for line in f]
    match_ids = sorted({e["match_id"] for e in all_ex})
    split = int(len(match_ids) * 0.80)
    test_match_ids = set(match_ids[split:])
    held_out = [e for e in all_ex if e["match_id"] in test_match_ids]
    print(f"[eval] held-out examples: {len(held_out)} (from {len(test_match_ids)} matches)")

    model_path = str(MERGED_MODEL if MERGED_MODEL.exists() else ADAPTER_MODEL)
    print(f"[eval] loading {model_path}")
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print(f"[eval] model loaded in {time.time()-t0:.1f}s")

    # Scoring
    tp = fp = tn = fn = parse_err = 0
    buy_pnls = []       # hindsight PnL on BUY predictions (sum)
    avoided = []        # hindsight LOSS avoided on SKIP predictions (label=skip)
    missed = []         # hindsight GAIN missed on SKIP predictions (label=buy)
    per_example = []

    print(f"[eval] running {len(held_out)} examples through trained model")
    for i, ex in enumerate(held_out):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": ITER8_SYS},
             {"role": "user",   "content": _user_prompt(ex)}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        import torch
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=80,
                                 temperature=0.01, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        latency_ms = (time.time() - t0) * 1000
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        pred = parse_response(response)
        label = ex["label"]["action"]
        pnl = ex.get("hindsight_pnl_pct", 0.0)
        if pred["action"] == "parse_error":
            parse_err += 1
        if pred["action"] == "buy":
            if label == "buy":
                tp += 1; buy_pnls.append(pnl)
            else:
                fp += 1; buy_pnls.append(pnl)  # hindsight: this would have lost
        else:
            if label == "skip":
                tn += 1; avoided.append(-pnl)
            else:
                fn += 1; missed.append(pnl)

        per_example.append({
            "i": i, "match_id": ex["match_id"], "label": label,
            "pred": pred, "hindsight_pnl_pct": pnl, "latency_ms": round(latency_ms),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(held_out)}] acc={(tp+tn)/(i+1):.2%}  "
                  f"buy_preds={tp+fp}  parse_err={parse_err}")

    total = len(held_out)
    acc = (tp + tn) / total if total else 0
    prec_buy = tp / (tp + fp) if (tp + fp) else 0
    recall_buy = tp / (tp + fn) if (tp + fn) else 0

    summary = {
        "model": model_path,
        "total": total,
        "parse_errors": parse_err,
        "accuracy": round(acc, 3),
        "precision_buy": round(prec_buy, 3),
        "recall_buy": round(recall_buy, 3),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "sim_pnl_pct_on_buys": round(sum(buy_pnls), 1),
        "avg_pnl_per_buy_pct": round(sum(buy_pnls) / max(1, tp + fp), 2),
        "avoided_loss_pct": round(sum(avoided), 1),
        "missed_gain_pct": round(sum(missed), 1),
    }

    print("\n" + "=" * 70)
    print(f"EVAL TRAINED  {model_path}")
    print("=" * 70)
    for k, v in summary.items():
        if k != "confusion":
            print(f"  {k:<22} {v}")
    print(f"  confusion:")
    print(f"                pred=BUY   pred=SKIP")
    print(f"    label=BUY    {tp:>5}      {fn:>5}")
    print(f"    label=SKIP   {fp:>5}      {tn:>5}")
    print("=" * 70)

    with open(OUT, "w") as f:
        json.dump({"summary": summary, "per_example": per_example}, f, indent=2)
    print(f"\n[eval] detailed report written to {OUT}")


if __name__ == "__main__":
    main()
