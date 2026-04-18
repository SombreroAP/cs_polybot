#!/usr/bin/env python3
"""
Evaluator — score ANY model candidate against the SFT dataset.

Usage:
    # Baseline: Haiku + iter8 on 100 random examples
    python evaluator.py --model haiku-4-5 --samples 100

    # Compare two models head-to-head on the same held-out set
    python evaluator.py --compare haiku-4-5 sonnet-4-5 --samples 200

    # Use local Ollama instead of Claude
    python evaluator.py --model ollama://qwen3:30b-a3b --samples 50

    # Full run (slow, costly) — all examples
    python evaluator.py --model haiku-4-5 --samples all

Output:
    - Per-example results streamed to data/evals/<model>_<ts>.jsonl
    - Summary table: accuracy, PnL, cost, latency, confusion matrix

Caching:
    Every (model, prompt_hash) response is cached in data/evals/cache.sqlite
    so reruns are free.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent
SFT_PATH = REPO / "data" / "training" / "sft.jsonl"
EVALS_DIR = REPO / "data" / "evals"
CACHE_DB = EVALS_DIR / "cache.sqlite"

# Load .env so ANTHROPIC_API_KEY / OLLAMA_URL work
try:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env", override=False)
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Cache (SQLite) — so a re-run is free after the first pass
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_cache() -> sqlite3.Connection:
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            key TEXT PRIMARY KEY,
            model TEXT,
            prompt_hash TEXT,
            response_json TEXT,
            latency_ms REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            created_at REAL
        )
    """)
    conn.commit()
    return conn


def _cache_key(model: str, prompt: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{model}|{h}"


def _cache_get(conn, model: str, prompt: str) -> Optional[dict]:
    row = conn.execute(
        "SELECT response_json, latency_ms, input_tokens, output_tokens, cost_usd "
        "FROM responses WHERE key = ?",
        (_cache_key(model, prompt),)
    ).fetchone()
    if not row:
        return None
    resp, lat, ti, to, cost = row
    return {
        "response": json.loads(resp) if resp else None,
        "latency_ms": lat, "input_tokens": ti, "output_tokens": to,
        "cost_usd": cost, "cached": True,
    }


def _cache_put(conn, model: str, prompt: str, result: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
        (_cache_key(model, prompt),
         model,
         hashlib.sha256(prompt.encode()).hexdigest()[:16],
         json.dumps(result.get("response")),
         result.get("latency_ms"), result.get("input_tokens"),
         result.get("output_tokens"), result.get("cost_usd"),
         time.time())
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt rebuilder — reconstruct the exact prompt our live bot would send
# given an SFT example's state. Must match edge_analyst.iter8's _build_prompt.
# ─────────────────────────────────────────────────────────────────────────────

def build_iter8_prompt(ex: dict) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) matching iter8's format.

    Input: an SFT example from data/training/sft.jsonl.
    Output: prompts that should look ~identical to what the live bot would
    send when faced with this situation.
    """
    # Read the frozen iter8 system prompt from disk (single source of truth)
    iter8 = (REPO / "_logs" / "edge_analyst.iter8_frozen.py").read_text()
    system = iter8.split('SYSTEM_PROMPT = """')[1].split('"""')[0].strip()

    # Reconstruct user prompt from the state snapshot
    state = ex.get("state", {}) or {}
    t1 = state.get("team_one") or {}
    t2 = state.get("team_two") or {}
    trigger = ex.get("trigger", {}) or {}
    market = ex.get("market_before", {}) or {}

    ta = t1.get("name") or t1.get("fixture", {}).get("team_name", "Team A")
    tb = t2.get("name") or t2.get("fixture", {}).get("team_name", "Team B")
    # Which team is the 'buy' side? We label buy=a if the trigger fired for
    # team a, else team b. Fall back to team a.
    buy = ta if trigger.get("team") == "a" else (tb if trigger.get("team") == "b" else ta)
    other = tb if buy == ta else ta

    lines = [
        f"GAME: CS2 | {ta} vs {tb}  (evaluating BUY on {buy})",
        f"Series: {t1.get('match_score', 0)}-{t2.get('match_score', 0)}"
        f"  Map: {t1.get('score', 0)}-{t2.get('score', 0)}",
    ]
    map_name = state.get("map_name") or ""
    if map_name:
        lines.append(f"Map: {map_name}")

    # Economy + alive from player_states if rich
    ps1 = t1.get("player_states") or []
    ps2 = t2.get("player_states") or []
    if ps1 and ps2:
        eco_a = sum(p.get("money", 0) or 0 for p in ps1)
        eco_b = sum(p.get("money", 0) or 0 for p in ps2)
        alive_a = sum(1 for p in ps1 if (p.get("hp", 0) or 0) > 0)
        alive_b = sum(1 for p in ps2 if (p.get("hp", 0) or 0) > 0)
        lines.append(f"Economy: {ta}=${eco_a:,} vs {tb}=${eco_b:,}")
        if alive_a != 5 or alive_b != 5:
            lines.append(f"Alive: {alive_a}v{alive_b}")

    # Market
    for tok, v in list(market.items())[:2]:
        bid = v.get("bid") or 0
        ask = v.get("ask") or 0
        lines.append(f"PM token ...{tok[-6:]}: bid={bid*100:.0f}¢ ask={ask*100:.0f}¢")

    # Trigger (single event, iter8-style at the END)
    desc = trigger.get("description") or trigger.get("event_type") or ""
    lines.append("")
    lines.append(f"Trigger: {desc}")
    lines.append("")
    lines.append(f"BUY {buy}? JSON only: "
                 '{"action":"buy or skip","confidence":0.0-1.0,"reason":"one sentence"}')
    return system, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Model adapters — unified `call(prompt) -> dict` interface
# ─────────────────────────────────────────────────────────────────────────────

class ModelAdapter:
    name: str
    def call(self, system: str, user: str) -> dict: ...


# Per-token pricing table (USD / 1M tokens). Update when Anthropic changes rates.
CLAUDE_PRICING = {
    "claude-haiku-4-5-20251001":  {"in": 0.80,  "out": 4.00},
    "claude-sonnet-4-5-20251001": {"in": 3.00,  "out": 15.00},
    "claude-opus-4-7-20260115":   {"in": 15.00, "out": 75.00},
}

# Short aliases
CLAUDE_ALIASES = {
    "haiku-4-5":  "claude-haiku-4-5-20251001",
    "sonnet-4-5": "claude-sonnet-4-5-20251001",
    "opus-4-7":   "claude-opus-4-7-20260115",
    "haiku":      "claude-haiku-4-5-20251001",
    "sonnet":     "claude-sonnet-4-5-20251001",
    "opus":       "claude-opus-4-7-20260115",
}


class ClaudeAdapter(ModelAdapter):
    def __init__(self, model: str):
        import anthropic
        self.name = CLAUDE_ALIASES.get(model, model)
        api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY not set. "
                             "Check `.env` or shell. (Hint: `unset ANTHROPIC_API_KEY` "
                             "then re-source .env if Claude for Desktop set an empty one.)")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.pricing = CLAUDE_PRICING.get(self.name, {"in": 1.0, "out": 5.0})

    def call(self, system: str, user: str) -> dict:
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.name,
            max_tokens=200,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.time() - t0) * 1000
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        usage = resp.usage
        cost = (usage.input_tokens  * self.pricing["in"]  / 1e6 +
                usage.output_tokens * self.pricing["out"] / 1e6)
        return {
            "response": text, "latency_ms": latency_ms,
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cost_usd": cost, "cached": False,
        }


class OllamaAdapter(ModelAdapter):
    def __init__(self, model: str):
        import requests
        self.requests = requests
        self.name = model
        self.url = os.environ.get("OLLAMA_URL") or "http://192.168.76.196:11434"
        if not self.url:
            raise SystemExit("OLLAMA_URL not set")

    def call(self, system: str, user: str) -> dict:
        t0 = time.time()
        r = self.requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 200},
                "keep_alive": "30m",
            },
            timeout=120,
        )
        r.raise_for_status()
        d = r.json()
        latency_ms = (time.time() - t0) * 1000
        text = d.get("message", {}).get("content", "")
        return {
            "response": text, "latency_ms": latency_ms,
            "input_tokens": d.get("prompt_eval_count", 0),
            "output_tokens": d.get("eval_count", 0),
            "cost_usd": 0.0, "cached": False,
        }


def make_adapter(model_spec: str) -> ModelAdapter:
    if model_spec.startswith("ollama://"):
        return OllamaAdapter(model_spec[len("ollama://"):])
    if model_spec.startswith("claude-") or model_spec in CLAUDE_ALIASES:
        return ClaudeAdapter(model_spec)
    # Default: Claude
    return ClaudeAdapter(model_spec)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing + scoring
# ─────────────────────────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """Extract {action, confidence, reason} from a model response, best-effort."""
    if not text:
        return {"action": "parse_error", "confidence": 0.0, "reason": "empty"}
    # Strip any <think>...</think>
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # Strip markdown code fences
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if "{" in p:
                text = p
                break
    # Find the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
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


@dataclass
class EvalResult:
    total: int = 0
    parse_errors: int = 0
    tp: int = 0  # predicted buy, label buy
    fp: int = 0  # predicted buy, label skip
    tn: int = 0  # predicted skip, label skip
    fn: int = 0  # predicted skip, label buy
    sim_pnl_pct: float = 0.0     # sum of hindsight PnL on BUY predictions
    avoided_loss_pct: float = 0.0 # sum of hindsight LOSS on SKIP where label was SKIP
    missed_gain_pct: float = 0.0  # sum of hindsight GAIN on SKIP where label was BUY (i.e. missed)
    latencies_ms: list[float] = None
    costs_usd: list[float] = None

    def summary(self) -> dict:
        if self.total == 0:
            return {"error": "no examples"}
        acc = (self.tp + self.tn) / self.total
        precision = self.tp / max(1, self.tp + self.fp)
        recall    = self.tp / max(1, self.tp + self.fn)
        return {
            "total":           self.total,
            "parse_errors":    self.parse_errors,
            "accuracy":        round(acc, 3),
            "precision_buy":   round(precision, 3),
            "recall_buy":      round(recall, 3),
            "confusion": {
                "true_buy":  self.tp, "false_buy":  self.fp,
                "true_skip": self.tn, "false_skip": self.fn,
            },
            "sim_pnl_pct_on_buys":  round(self.sim_pnl_pct, 1),
            "avg_pnl_per_buy_pct":  round(self.sim_pnl_pct / max(1, self.tp + self.fp), 2),
            "avoided_loss_pct":     round(self.avoided_loss_pct, 1),
            "missed_gain_pct":      round(self.missed_gain_pct, 1),
            "latency_p50_ms":       round(statistics.median(self.latencies_ms or [0])),
            "latency_p95_ms":       round(_p95(self.latencies_ms or [0])),
            "total_cost_usd":       round(sum(self.costs_usd or [0]), 4),
            "avg_cost_per_call":    round(sum(self.costs_usd or [0]) / max(1, self.total), 5),
        }


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = int(0.95 * (len(s) - 1))
    return s[i]


def score_example(pred: dict, ex: dict, res: EvalResult, pnl_pct: float) -> None:
    true_label = ex["label"]["action"]
    if pred["action"] == "parse_error":
        res.parse_errors += 1
    if pred["action"] == "buy":
        if true_label == "buy":
            res.tp += 1
            res.sim_pnl_pct += pnl_pct  # already +ve
        else:
            res.fp += 1
            res.sim_pnl_pct += pnl_pct  # -ve (would have lost)
    else:  # skip (or parse_error treated as skip)
        if true_label == "skip":
            res.tn += 1
            res.avoided_loss_pct += -pnl_pct  # pnl_pct is negative; we avoided it
        else:
            res.fn += 1
            res.missed_gain_pct += pnl_pct  # positive; we missed this gain


# ─────────────────────────────────────────────────────────────────────────────
# Main run
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(model_spec: str, n_samples: int, seed: int = 42,
             held_out_only: bool = True, verbose: bool = True) -> dict:
    """Score `model_spec` against n examples from the SFT dataset.

    If held_out_only=True (default), uses examples whose match_id is in the
    held-out test split (last 20% of matches, deterministic).
    """
    if not SFT_PATH.exists():
        print(f"[evaluator] {SFT_PATH} missing — run `python data/processor.py build-sft` first")
        return {}
    EVALS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all examples, group by match_id for the split
    all_examples = []
    with open(SFT_PATH) as f:
        for line in f:
            all_examples.append(json.loads(line))

    # Deterministic held-out split: sort match IDs, take last 20%
    all_matches = sorted({e["match_id"] for e in all_examples})
    split_idx = int(len(all_matches) * 0.80)
    test_matches = set(all_matches[split_idx:])
    if verbose:
        print(f"[evaluator] held-out test set: {len(test_matches)} matches "
              f"(out of {len(all_matches)} total)")

    pool = [e for e in all_examples if e["match_id"] in test_matches] \
           if held_out_only else all_examples

    rng = random.Random(seed)
    if n_samples < len(pool):
        pool = rng.sample(pool, n_samples)

    adapter = make_adapter(model_spec)
    conn = _ensure_cache()
    res = EvalResult()
    res.latencies_ms = []
    res.costs_usd = []

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = adapter.name.replace("/", "-").replace(":", "_")
    out_path = EVALS_DIR / f"{safe_name}_{timestamp}.jsonl"

    if verbose:
        print(f"[evaluator] model={adapter.name}  samples={len(pool)}  held_out={held_out_only}")
        print(f"[evaluator] writing per-example results to {out_path}")

    with open(out_path, "w") as out:
        for i, ex in enumerate(pool):
            system, user = build_iter8_prompt(ex)
            prompt_full = system + "\n\n" + user

            cached = _cache_get(conn, adapter.name, prompt_full)
            if cached:
                result = cached
            else:
                try:
                    result = adapter.call(system, user)
                except Exception as e:
                    result = {"response": f"__ERROR__ {e}",
                              "latency_ms": 0, "input_tokens": 0,
                              "output_tokens": 0, "cost_usd": 0, "cached": False}
                if not result["response"].startswith("__ERROR__"):
                    _cache_put(conn, adapter.name, prompt_full, result)

            pred = parse_response(result.get("response", ""))
            pnl_pct = ex.get("hindsight_pnl_pct", 0.0)
            res.total += 1
            res.latencies_ms.append(result.get("latency_ms", 0))
            res.costs_usd.append(result.get("cost_usd", 0))
            score_example(pred, ex, res, pnl_pct)

            out.write(json.dumps({
                "i": i,
                "match_id": ex["match_id"],
                "decision_ts": ex["decision_ts"],
                "label": ex["label"]["action"],
                "label_pnl_pct": pnl_pct,
                "pred": pred,
                "latency_ms": result.get("latency_ms"),
                "cost_usd": result.get("cost_usd"),
                "cached": result.get("cached", False),
            }) + "\n")

            if verbose and (i + 1) % 50 == 0:
                print(f"[evaluator] {i+1}/{len(pool)} — "
                      f"acc={((res.tp+res.tn)/res.total):.2%} "
                      f"parse_err={res.parse_errors} "
                      f"cost=${sum(res.costs_usd):.3f}")

    summary = res.summary()
    summary["model"] = adapter.name
    summary["samples"] = len(pool)
    summary["output_file"] = str(out_path)

    # Print
    if verbose:
        print("\n" + "=" * 70)
        print(f"EVAL  {adapter.name}  ({len(pool)} examples)")
        print("=" * 70)
        for k, v in summary.items():
            if k not in ("confusion", "model", "samples", "output_file"):
                print(f"  {k:<24} {v}")
        print("  confusion matrix:")
        cm = summary["confusion"]
        print(f"                pred=BUY   pred=SKIP")
        print(f"    label=BUY   {cm['true_buy']:>7}      {cm['false_skip']:>7}")
        print(f"    label=SKIP  {cm['false_buy']:>7}      {cm['true_skip']:>7}")
        print("=" * 70)

    return summary


def run_compare(models: list[str], n_samples: int) -> list[dict]:
    results = []
    for m in models:
        r = run_eval(m, n_samples)
        results.append(r)
    print("\n" + "=" * 80)
    print(f"COMPARISON  ({n_samples} examples each)")
    print("=" * 80)
    header = f"  {'MODEL':<40} {'ACC':>6} {'PREC_BUY':>10} {'RECALL_BUY':>11} {'PnL%':>8} {'COST$':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r['model']:<40} {r['accuracy']:>6} "
              f"{r['precision_buy']:>10} {r['recall_buy']:>11} "
              f"{r['sim_pnl_pct_on_buys']:>8} {r['total_cost_usd']:>8}")
    print("=" * 80)
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate model candidates on the SFT dataset.")
    ap.add_argument("--model", default=None, help="single-model mode. e.g. haiku-4-5, ollama://qwen3:30b-a3b")
    ap.add_argument("--compare", nargs="+", default=None, help="compare mode: list of models")
    ap.add_argument("--samples", default="100",
                    help="number of examples (or 'all')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true",
                    help="use all examples, not just held-out test split")
    args = ap.parse_args()

    if args.compare is None and args.model is None:
        ap.error("provide --model OR --compare")

    if args.samples == "all":
        n = 10_000_000
    else:
        n = int(args.samples)

    if args.compare:
        run_compare(args.compare, n)
    else:
        run_eval(args.model, n, seed=args.seed, held_out_only=not args.all)


if __name__ == "__main__":
    main()
