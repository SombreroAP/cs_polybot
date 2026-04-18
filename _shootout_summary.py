#!/usr/bin/env python3
"""Summarise a model shootout — parses each _logs/shootout/*.log and prints
a side-by-side table. Scores are fully comparable because every run used
the same data, code path, and balance.
"""
import glob
import os
import re
from collections import Counter, defaultdict

LOGS_DIR = "_logs/shootout"


def parse_log(path: str) -> dict:
    r = {
        "label": os.path.splitext(os.path.basename(path))[0],
        "trades": 0, "wins": 0, "losses": 0,
        "final_balance": 0.0, "pnl": 0.0,
        "qwen_calls": 0, "buys": 0, "skips": 0,
        "exit_reasons": Counter(),
        "exit_pnl_by_reason": defaultdict(float),
        "entry_prices": [],
        "confidences": [],
        "latencies_ms": [],
        "drops": 0,
        "matches_done": 0,
    }
    if not os.path.exists(path):
        return r
    with open(path, errors="ignore") as fh:
        for line in fh:
            # Final summary line
            m = re.search(r"BACKTEST COMPLETE — (\d+) matches \| PnL \$([+-]?[\d.]+) \| (\d+) trades \((\d+)W/(\d+)L\) \| (\d+) qwen calls", line)
            if m:
                r["matches_done"] = int(m.group(1))
                r["pnl"] = float(m.group(2))
                r["trades"] = int(m.group(3))
                r["wins"] = int(m.group(4))
                r["losses"] = int(m.group(5))
                r["qwen_calls"] = int(m.group(6))
            # Final balance
            m = re.search(r"Balance:\s+\$([\d.]+)", line)
            if m:
                r["final_balance"] = float(m.group(1))
            m = re.search(r"Qwen:\s+\d+ calls \((\d+) buys, (\d+) skips\)", line)
            if m:
                r["buys"] = int(m.group(1))
                r["skips"] = int(m.group(2))
            # Entry prices
            m = re.search(r"BUY\s+\S+\s+@\s+([\d.]+)\s+bet=\$[\d.]+\s+tp=\d+% sl=\d+%\s+conf=([\d.]+)", line)
            if m:
                r["entry_prices"].append(float(m.group(1)))
                r["confidences"].append(float(m.group(2)))
            # Exit reasons
            m = re.search(r"EXIT\s+\S+\s+@\s+[\d.]+\s+\((\w+)\)\s+pnl=\$([+-]?[\d.]+)", line)
            if m:
                reason = m.group(1)
                pnl = float(m.group(2))
                r["exit_reasons"][reason] += 1
                r["exit_pnl_by_reason"][reason] += pnl
            # LLM latency
            m = re.search(r"\[LOCAL-LLM\].*?(\d+)ms\s+\(wait=", line)
            if m:
                r["latencies_ms"].append(int(m.group(1)))
            # Dropped calls (back-pressure)
            if "[LOCAL-LLM] DROP" in line:
                r["drops"] += 1
    return r


def fmt(results: list) -> None:
    if not results:
        print("No shootout logs found in", LOGS_DIR)
        return
    # Header
    labels = [r["label"] for r in results]
    colw = max(12, max(len(l) for l in labels) + 2)
    print(f"\n{'Metric':<26s}" + "".join(f"{l:>{colw}s}" for l in labels))
    print("-" * (26 + colw * len(results)))

    def row(name, vals, fmt_each=lambda v: str(v)):
        print(f"{name:<26s}" + "".join(f"{fmt_each(v):>{colw}s}" for v in vals))

    row("Matches completed", [r["matches_done"] for r in results])
    row("Total qwen calls", [r["qwen_calls"] for r in results])
    row("Buys / Skips",
        [f"{r['buys']}/{r['skips']}" for r in results])
    row("Buy rate %",
        [100 * r["buys"] / max(1, r["qwen_calls"]) for r in results],
        lambda v: f"{v:.1f}%")
    row("Trades (W/L)",
        [f"{r['trades']} ({r['wins']}/{r['losses']})" for r in results])
    row("Win rate %",
        [100 * r["wins"] / max(1, r["trades"]) for r in results],
        lambda v: f"{v:.1f}%")
    row("Final balance",
        [r["final_balance"] for r in results],
        lambda v: f"${v:.2f}")
    row("Net PnL",
        [r["pnl"] for r in results],
        lambda v: f"${v:+.2f}")
    row("Avg entry price",
        [sum(r["entry_prices"]) / max(1, len(r["entry_prices"])) for r in results],
        lambda v: f"{v:.3f}")
    row("Avg confidence",
        [sum(r["confidences"]) / max(1, len(r["confidences"])) for r in results],
        lambda v: f"{v:.2f}")
    row("Avg LLM latency",
        [sum(r["latencies_ms"]) / max(1, len(r["latencies_ms"])) for r in results],
        lambda v: f"{v:.0f}ms")
    row("Dropped calls",
        [r["drops"] for r in results])

    # Exit reasons per model
    all_reasons = sorted({k for r in results for k in r["exit_reasons"]})
    if all_reasons:
        print("\nExit reasons (count | $pnl):")
        for reason in all_reasons:
            row(f"  {reason}",
                [(r["exit_reasons"].get(reason, 0),
                  r["exit_pnl_by_reason"].get(reason, 0.0)) for r in results],
                lambda v: f"{v[0]}|${v[1]:+.0f}")

    # Verdict
    print()
    ranked = sorted(results, key=lambda r: r["pnl"], reverse=True)
    print("Ranking by PnL:")
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['label']:<35s} PnL ${r['pnl']:+.2f}  ({r['wins']}W/{r['losses']}L)")


def main():
    paths = sorted(glob.glob(os.path.join(LOGS_DIR, "*.log")))
    results = [parse_log(p) for p in paths]
    fmt(results)


if __name__ == "__main__":
    main()
