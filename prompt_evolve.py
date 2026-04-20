#!/usr/bin/env python3
"""
Prompt evolver — auto-generates candidate prompt variants and scores them
against the held-out SFT test set using evaluator.py.

Strategy:
  1. Start with the current production prompt (iter8, pulled from edge_analyst.iter8_frozen.py).
  2. Generate N variants by applying a library of prompt transforms:
       - more selective (tighten thresholds)
       - add / remove specific rules
       - shorter / longer format
       - different tone (assertive vs exploratory)
  3. Score each with evaluator.py on the same held-out set.
  4. Rank by sim_pnl + precision_buy; promote winners.
  5. Optionally: feed promoted variants back as parents for the next round.

Usage:
    python prompt_evolve.py --samples 100                   # single round
    python prompt_evolve.py --samples 100 --rounds 3        # 3 rounds of evolution
    python prompt_evolve.py --samples 100 --commit          # save winner to _logs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent

# Re-use evaluator machinery
sys.path.insert(0, str(HERE))
import evaluator  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Prompt variants — functions that take the current system prompt and mutate it
# ─────────────────────────────────────────────────────────────────────────────

# iter8 baseline — pulled from the frozen file
BASELINE_PATH = HERE / "_logs" / "edge_analyst.iter8_frozen.py"


def load_baseline() -> str:
    if not BASELINE_PATH.exists():
        return "CS2 Polymarket trader."  # fallback
    text = BASELINE_PATH.read_text()
    return text.split('SYSTEM_PROMPT = """')[1].split('"""')[0].strip()


@dataclass
class Variant:
    name: str
    system_prompt: str
    parent: str = "iter8"
    changes: list[str] = field(default_factory=list)


def v_baseline(base: str) -> Variant:
    return Variant(name="v0_iter8_baseline", system_prompt=base)


def v_stricter_tail(base: str) -> Variant:
    """Tighten the tail-risk guard to 65¢ from 70¢."""
    new = base.replace("ask > 0.70 is a", "ask > 0.65 is a")
    new = new.replace("For ask > 0.70:", "For ask > 0.65:")
    return Variant(name="v1_stricter_tail",
                   system_prompt=new,
                   changes=["tail-risk threshold 70 -> 65¢"])


def v_require_2_signals_on_map_lead(base: str) -> Variant:
    """Require 2 signals even when buy side is leading in maps (was 1)."""
    new = base.replace(
        "BUY SIDE LEADING in maps (1-0, or 2-1 closeout): STRONG edge → BUY when ≥1 round signal agrees.",
        "BUY SIDE LEADING in maps (1-0, or 2-1 closeout): STRONG edge → BUY when ≥2 round signals agree (previous: ≥1 caused more false-positives).",
    )
    return Variant(name="v2_require_2_signals_map_lead",
                   system_prompt=new,
                   changes=["map-leading requirement 1 -> 2 signals"])


def v_drop_momentum_rules(base: str) -> Variant:
    """Remove the MOMENTUM SANITY section entirely."""
    if "MOMENTUM SANITY:" not in base:
        return v_baseline(base)
    idx = base.index("MOMENTUM SANITY:")
    new = base[:idx].rstrip()
    return Variant(name="v3_no_momentum_rules",
                   system_prompt=new,
                   changes=["dropped MOMENTUM SANITY section"])


def v_add_kambi_sanity(base: str) -> Variant:
    """Add a hard no-go on Kambi mentions."""
    addition = "\n\nKAMBI: You have NO knowledge of Kambi odds. Never cite them. If tempted, SKIP."
    return Variant(name="v4_kambi_silence",
                   system_prompt=base + addition,
                   changes=["added explicit Kambi silence rule"])


def v_emphasize_skip_default(base: str) -> Variant:
    """Prepend a hard 'default to skip' line."""
    addition = ("Default action is SKIP. Only BUY when the setup is obvious AND one of the "
                "explicit BUY cases below applies. When in doubt, SKIP.\n\n")
    return Variant(name="v5_skip_by_default",
                   system_prompt=addition + base,
                   changes=["prepended skip-by-default directive"])


def v_shorter_reason(base: str) -> Variant:
    """Demand shorter reasons (models waste tokens otherwise)."""
    new = base.replace("short, <180 chars",
                       "very short, <80 chars — one factual sentence")
    return Variant(name="v6_shorter_reason",
                   system_prompt=new,
                   changes=["reason <180 chars -> <80 chars"])


def v_no_thinking_tags(base: str) -> Variant:
    addition = "\n\nDO NOT use <think> tags or chain-of-thought. Output JSON directly."
    return Variant(name="v7_no_think",
                   system_prompt=base + addition,
                   changes=["explicit no-think directive"])


VARIANT_FNS: list[Callable[[str], Variant]] = [
    v_baseline,
    v_stricter_tail,
    v_require_2_signals_on_map_lead,
    v_drop_momentum_rules,
    v_add_kambi_sanity,
    v_emphasize_skip_default,
    v_shorter_reason,
    v_no_thinking_tags,
]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — plug each variant's system prompt into evaluator at runtime
# ─────────────────────────────────────────────────────────────────────────────

class VariantEvaluator(evaluator.ClaudeAdapter):
    """ClaudeAdapter that uses a custom system prompt."""

    def __init__(self, model: str, system_override: str):
        super().__init__(model)
        self._system_override = system_override

    def call(self, system: str, user: str) -> dict:
        return super().call(self._system_override, user)


def score_variant(variant: Variant, n_samples: int, model: str = "haiku-4-5") -> dict:
    """Run the evaluator against n held-out samples with the variant's prompt."""
    print(f"\n[evolve] scoring {variant.name} ({n_samples} examples)")

    adapter = VariantEvaluator(model, variant.system_prompt)
    conn = evaluator._ensure_cache()

    # Load examples (same held-out logic as evaluator.run_eval)
    all_ex = []
    with open(evaluator.SFT_PATH) as f:
        for line in f:
            all_ex.append(json.loads(line))
    match_ids = sorted({e["match_id"] for e in all_ex})
    split = int(len(match_ids) * 0.80)
    test_matches = set(match_ids[split:])
    pool = [e for e in all_ex if e["match_id"] in test_matches]
    import random
    rng = random.Random(42)
    if n_samples < len(pool):
        pool = rng.sample(pool, n_samples)

    res = evaluator.EvalResult()
    res.latencies_ms = []
    res.costs_usd = []

    for ex in pool:
        system, user = evaluator.build_iter8_prompt(ex)
        # Force our system override via the adapter; evaluator builds user prompt
        result = adapter.call(system, user)
        pred = evaluator.parse_response(result.get("response", ""))
        pnl_pct = ex.get("hindsight_pnl_pct", 0.0)
        res.total += 1
        res.latencies_ms.append(result.get("latency_ms", 0))
        res.costs_usd.append(result.get("cost_usd", 0))
        evaluator.score_example(pred, ex, res, pnl_pct)

    summary = res.summary()
    summary["variant"] = variant.name
    summary["changes"] = variant.changes
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Ranking + promotion
# ─────────────────────────────────────────────────────────────────────────────

def rank_variants(results: list[dict]) -> list[dict]:
    """Composite ranking: simulate pnl first, then precision_buy, then recall."""
    def key(r):
        # Primary: sim_pnl (higher = better; losing negative)
        # Secondary: precision_buy (higher = better)
        # Tertiary: -cost (lower = better)
        return (-r.get("sim_pnl_pct_on_buys", 0),
                -r.get("precision_buy", 0),
                r.get("total_cost_usd", 999))
    return sorted(results, key=key)


def print_leaderboard(results: list[dict]) -> None:
    ranked = rank_variants(results)
    print("\n" + "=" * 90)
    print("PROMPT EVOLUTION LEADERBOARD")
    print("=" * 90)
    print(f"{'rank':>4} {'variant':<35} {'acc':>5} {'prec':>5} {'rec':>5} "
          f"{'pnl_pct':>8} {'cost$':>7}")
    print("-" * 90)
    for i, r in enumerate(ranked, 1):
        print(f"{i:>4} {r.get('variant','?')[:35]:<35} "
              f"{r.get('accuracy',0):>5.2f} "
              f"{r.get('precision_buy',0):>5.2f} "
              f"{r.get('recall_buy',0):>5.2f} "
              f"{r.get('sim_pnl_pct_on_buys',0):>8.1f} "
              f"{r.get('total_cost_usd',0):>7.4f}")
    print("=" * 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=80,
                    help="examples per variant (keeps Claude cost reasonable)")
    ap.add_argument("--model", default="haiku-4-5")
    ap.add_argument("--commit", action="store_true",
                    help="save the winner to _logs/edge_analyst.iter_evolved.py")
    args = ap.parse_args()

    base = load_baseline()
    print(f"[evolve] baseline loaded — {len(base)} chars")
    print(f"[evolve] testing {len(VARIANT_FNS)} variants × {args.samples} samples "
          f"against {args.model}")
    print(f"[evolve] estimated cost: ~${len(VARIANT_FNS) * args.samples * 0.002:.2f}")

    results = []
    for fn in VARIANT_FNS:
        v = fn(base)
        summary = score_variant(v, args.samples, model=args.model)
        results.append(summary)
        print(f"   -> {v.name}: acc={summary.get('accuracy',0):.2%} "
              f"prec_buy={summary.get('precision_buy',0):.2f} "
              f"pnl={summary.get('sim_pnl_pct_on_buys',0):.1f}")

    print_leaderboard(results)

    # Save round report
    out = HERE / "data" / "evals" / f"prompt_evolve_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[evolve] round report saved to {out}")

    if args.commit:
        winner = rank_variants(results)[0]
        if winner["variant"] == "v0_iter8_baseline":
            print("[evolve] baseline still winning — no change to commit")
        else:
            # Find the variant object
            for fn in VARIANT_FNS:
                v = fn(base)
                if v.name == winner["variant"]:
                    dst = HERE / "_logs" / f"edge_analyst.{v.name}.py"
                    template = (HERE / "_logs" / "edge_analyst.iter8_frozen.py").read_text()
                    new_content = template.replace(base, v.system_prompt)
                    dst.write_text(new_content)
                    print(f"[evolve] wrote winner prompt to {dst}")
                    break


if __name__ == "__main__":
    main()
