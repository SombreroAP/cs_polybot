#!/usr/bin/env python3
"""
Labeler threshold sweep — find the (tp_pct, window_s) combo that produces a
healthy SFT corpus from the current processed recordings.

Sweeps a grid of filter params, reports:
  - example count
  - skip:buy ratio
  - % of BUY labels that are "strong" (hindsight_pnl >= 2× tp_pct threshold)
  - % of BUY labels that are "borderline" (within 20% of tp threshold)

Picks the combo that maximises useful-examples = balanced + strong-signal.

Usage:
    python sweep_labeler.py            # run sweep, print table
    python sweep_labeler.py --commit   # after inspecting, rebuild final SFT
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data"))
import processor  # noqa: E402

# Grid of (tp_pct, window_s, min_gap_s) to test. sl_pct always matches tp_pct.
# min_gap_s is the major lever — 10s dedupe was killing 58% of events, test lower.
TP_GRID = [0.02, 0.04, 0.06]
WINDOW_GRID = [120, 300]
GAP_GRID = [2, 3, 5, 10]


def analyze(sft_path: str) -> dict:
    """Compute balance + strong-signal stats on an SFT file."""
    buys = skips = 0
    strong_buys = borderline_buys = 0
    avg_buy_pnl = 0.0
    total_buy_pnl = 0.0
    with open(sft_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            action = d.get("label", {}).get("action")
            pnl = abs(d.get("hindsight_pnl_pct", 0.0))
            if action == "buy":
                buys += 1
                total_buy_pnl += pnl
                # tp_pct is the threshold used at build time — we use pnl to judge strength
            elif action == "skip":
                skips += 1
    if buys > 0:
        avg_buy_pnl = total_buy_pnl / buys
    return {
        "buys": buys, "skips": skips,
        "ratio_skip_buy": round(skips / max(1, buys), 2),
        "avg_buy_hindsight_pct": round(avg_buy_pnl, 2),
    }


def score(stats: dict) -> float:
    """Single-number quality score for a labeler combo.

    Higher is better. Rewards:
      - reasonable example count (saturates above 15k)
      - healthy skip:buy ratio (peaks at 10:1, penalize <3:1 or >25:1)
      - strong average hindsight PnL (real edge, not noise)
    """
    n = stats["buys"] + stats["skips"]
    if n < 1000:
        return 0  # useless for training
    ratio = stats["ratio_skip_buy"]
    avg = stats["avg_buy_hindsight_pct"]

    # Volume score: 1.0 at 15k+, 0.3 at 1k
    vol_score = min(1.0, n / 15000) * 0.7 + 0.3
    # Balance score: best at 8-12:1, linearly decays outside
    if 5 <= ratio <= 15:
        bal_score = 1.0
    elif 3 <= ratio <= 20:
        bal_score = 0.7
    else:
        bal_score = 0.3
    # Strength: reward >4% avg hindsight (real edge), penalize <2%
    if avg >= 4:
        str_score = 1.0
    elif avg >= 2:
        str_score = 0.6
    else:
        str_score = 0.2
    return round(vol_score * bal_score * str_score, 3)


def sweep() -> list[dict]:
    results = []
    tmp_out = HERE / "data" / "training" / "_sweep_tmp.jsonl"
    tmp_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'tp_pct':>7} {'window':>7} {'gap':>5} {'examples':>9} "
          f"{'buys':>5} {'skips':>6} {'s:b':>5} {'pnl%':>6} {'score':>6}")
    print("-" * 78)

    for tp in TP_GRID:
        for w in WINDOW_GRID:
            for g in GAP_GRID:
                t0 = time.time()
                build_res = processor.build_sft(
                    out_path=str(tmp_out),
                    tp_pct=tp, sl_pct=tp, window_s=w, min_gap_s=g,
                    quiet=True,
                )
                if build_res.get("examples", 0) == 0:
                    stats = {"buys": 0, "skips": 0, "ratio_skip_buy": 0, "avg_buy_hindsight_pct": 0}
                else:
                    stats = analyze(str(tmp_out))
                row = {
                    "tp_pct": tp, "window_s": w, "min_gap_s": g,
                    "examples": build_res.get("examples", 0),
                    **stats,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                row["score"] = score(row)
                results.append(row)
                print(f"{tp:>7.3f} {w:>7} {g:>5} {row['examples']:>9} "
                      f"{row['buys']:>5} {row['skips']:>6} "
                      f"{row['ratio_skip_buy']:>5.1f} "
                      f"{row['avg_buy_hindsight_pct']:>6.1f} "
                      f"{row['score']:>6.3f}")

    # Cleanup
    if tmp_out.exists():
        tmp_out.unlink()

    # Rank
    ranked = sorted(results, key=lambda r: -r["score"])
    print()
    print("=" * 78)
    print("TOP 5 COMBOS")
    print("=" * 78)
    print(f"{'rank':>4} {'tp_pct':>7} {'window':>7} {'gap':>5} {'examples':>9} "
          f"{'s:b':>5} {'pnl%':>6} {'score':>6}")
    for i, r in enumerate(ranked[:5], start=1):
        print(f"{i:>4} {r['tp_pct']:>7.3f} {r['window_s']:>7} {r['min_gap_s']:>5} "
              f"{r['examples']:>9} {r['ratio_skip_buy']:>5.1f} "
              f"{r['avg_buy_hindsight_pct']:>6.1f} {r['score']:>6.3f}")
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true",
                    help="after sweep, rebuild final SFT with winner")
    args = ap.parse_args()

    print("Starting labeler threshold sweep...")
    print(f"  grid: tp_pct={TP_GRID} × window_s={WINDOW_GRID}")
    print(f"  total combos: {len(TP_GRID) * len(WINDOW_GRID)}")
    print()
    ranked = sweep()

    if args.commit and ranked:
        winner = ranked[0]
        print(f"\n[commit] rebuilding final SFT with winner: "
              f"tp={winner['tp_pct']}, window={winner['window_s']}s, gap={winner['min_gap_s']}s")
        res = processor.build_sft(tp_pct=winner["tp_pct"],
                                  sl_pct=winner["tp_pct"],
                                  window_s=winner["window_s"],
                                  min_gap_s=winner["min_gap_s"])
        print(f"[commit] wrote {res['examples']} examples to {res['out']}")


if __name__ == "__main__":
    main()
