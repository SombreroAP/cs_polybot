#!/usr/bin/env python3
"""
Model replay harness — feed historical esports scenarios through any Ollama model,
compare decisions to ground-truth 60s price moves, score on accuracy + simulated PnL.

Usage:
  python3 replay_harness.py --model qwen3:30b-a3b --n 200
  python3 replay_harness.py --model qwen3:30b-a3b-think --think --n 200
  python3 replay_harness.py --model llama3:70b-instruct-q4_k_m --n 200

Outputs per-model: accuracy, skip-recall on losers, sim pnl, avg latency, cost.
"""
import argparse, json, random, time, re, sys, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA = os.getenv("OLLAMA_URL", "http://192.168.76.196:11435")
DATA = "/Users/andrew/Documents/Claude/Projects/Trading/esports/data/training/training_data.jsonl"

SYSTEM = """You are an esports betting analyst. Given a market context, decide whether to BUY the token or SKIP.

BUY when: ask in 20¢-75¢, favorable game state (economy/kills/momentum), no red flags.
SKIP when: ask >75¢ (asymmetric risk), stale data, weak signal, contradictory indicators.

Output ONLY a single JSON object: {"action":"buy","reason":"<15 words"} OR {"action":"skip","reason":"<15 words"}. Nothing else."""


def load_samples(n, seed=42, break_even_move_pct=6.0):
    """Load N random labeled samples (prompt + 60s outcome).

    Realistic costs: 2% fee + ~4% spread = 6% break-even move per trade.
    Ground truth = BUY only if 60s move beats break-even, SKIP otherwise.
    Include ALL samples so the mix reflects real event distribution (noise + signal).
    """
    random.seed(seed)
    samples = []
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            move = d["metadata"].get("price_move_60s", 0)
            samples.append({
                "prompt": d["prompt"],
                "move_60s": move,
                "bid": d["metadata"].get("bid_at_event", 0),
                "spread": d["metadata"].get("spread_pct", 0),
                # Honest ground truth: BUY only if move clears the 6% round-trip cost
                "ground_truth": "buy" if move > break_even_move_pct else "skip",
                "net_move_pct": move - break_even_move_pct,  # profit after cost if bought
            })
    random.shuffle(samples)
    return samples[:n]


def ask_ollama(prompt, model, think=False, timeout=60):
    """Return (action, latency_ms, raw_response)."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "num_predict": 2000 if think else 200},
        "keep_alive": -1,
    }
    if think:
        payload["think"] = True
    else:
        payload["think"] = False
        payload["format"] = "json"

    req = urllib.request.Request(
        f"{OLLAMA}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return "err", (time.time() - t0) * 1000, str(e)[:100]

    content = d.get("message", {}).get("content", "") or d.get("response", "")
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    latency = (time.time() - t0) * 1000

    # Extract action from JSON
    for m in re.finditer(r'\{[^{}]*?"action"[^{}]*?\}', content, re.DOTALL):
        try:
            j = json.loads(m.group(0))
            a = str(j.get("action", "")).lower().strip()
            if "buy" in a: return "buy", latency, content[:120]
            if "skip" in a: return "skip", latency, content[:120]
        except Exception:
            pass
    return "err", latency, content[:120]


def run_test(model, think, samples, parallel=1, verbose=False):
    """Return dict of metrics."""
    results = []
    t_start = time.time()

    if parallel == 1:
        for i, s in enumerate(samples):
            a, lat, raw = ask_ollama(s["prompt"], model, think=think)
            results.append({"action": a, "latency_ms": lat, "gt": s["ground_truth"], "move": s["move_60s"]})
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}] model={model} elapsed={time.time()-t_start:.0f}s",
                      file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(ask_ollama, s["prompt"], model, think): s for s in samples}
            for i, fut in enumerate(as_completed(futs)):
                s = futs[fut]
                a, lat, raw = fut.result()
                results.append({"action": a, "latency_ms": lat, "gt": s["ground_truth"], "move": s["move_60s"]})
                if verbose and (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{len(samples)}] parallel", file=sys.stderr)

    total_elapsed = time.time() - t_start
    n = len(results)
    errs = sum(1 for r in results if r["action"] == "err")
    valid = [r for r in results if r["action"] != "err"]

    # Classification metrics
    tp = sum(1 for r in valid if r["action"] == "buy" and r["gt"] == "buy")   # correct BUY
    tn = sum(1 for r in valid if r["action"] == "skip" and r["gt"] == "skip") # correct SKIP
    fp = sum(1 for r in valid if r["action"] == "buy" and r["gt"] == "skip")  # bought a loser
    fn = sum(1 for r in valid if r["action"] == "skip" and r["gt"] == "buy")  # missed a winner
    n_val = len(valid)

    # Simulated pnl: $30 bet per BUY, 60s hold.
    # Cost: 2% fee + ~4% spread = 6% round-trip break-even.
    # Net pnl per trade = bet × (move% - 6%) / 100
    bet = 30.0
    cost_pct = 6.0   # break-even move needed
    sim_pnl_model = sum(bet * (r["move"] - cost_pct) / 100 for r in valid if r["action"] == "buy")
    sim_pnl_baseline = sum(bet * (r["move"] - cost_pct) / 100 for r in valid)
    # "Oracle" upper bound: buy only when ground truth says buy
    sim_pnl_oracle = sum(bet * (r["move"] - cost_pct) / 100 for r in valid if r["gt"] == "buy")

    avg_lat = sum(r["latency_ms"] for r in valid) / max(1, n_val)
    buy_rate = sum(1 for r in valid if r["action"] == "buy") / max(1, n_val) * 100

    return {
        "model": model, "think": think, "n": n, "errs": errs,
        "accuracy": (tp + tn) / max(1, n_val),
        "skip_recall": tn / max(1, tn + fp),      # of actual losers, how many did model skip?
        "buy_precision": tp / max(1, tp + fp),    # of model BUYs, how many were actual winners?
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "buy_rate_pct": buy_rate,
        "sim_pnl_model": sim_pnl_model,
        "sim_pnl_buy_all": sim_pnl_baseline,
        "sim_pnl_oracle": sim_pnl_oracle,
        "sim_pnl_delta_vs_buyall": sim_pnl_model - sim_pnl_baseline,
        "capture_rate_vs_oracle": sim_pnl_model / sim_pnl_oracle if sim_pnl_oracle > 0 else 0,
        "avg_latency_ms": avg_lat,
        "total_wallclock_s": total_elapsed,
    }


def print_report(r):
    print(f"\n═══ {r['model']} ({'think' if r['think'] else 'no-think'}) ═══")
    print(f"  Samples: {r['n']}  Errors: {r['errs']}  Buy rate: {r['buy_rate_pct']:.0f}%")
    print(f"  Accuracy:       {r['accuracy']*100:.1f}%")
    print(f"  Skip recall:    {r['skip_recall']*100:.1f}%  (of losers, did we skip?)")
    print(f"  Buy precision:  {r['buy_precision']*100:.1f}%  (of buys, were they winners?)")
    print(f"  Confusion: TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")
    print(f"  Sim PnL (follow model): ${r['sim_pnl_model']:+.2f}")
    print(f"  Sim PnL (BUY-all):      ${r['sim_pnl_buy_all']:+.2f}")
    print(f"  Sim PnL (oracle):       ${r['sim_pnl_oracle']:+.2f}  ← theoretical max")
    print(f"  Delta vs BUY-all:       ${r['sim_pnl_delta_vs_buyall']:+.2f}")
    print(f"  Capture rate vs oracle: {r['capture_rate_vs_oracle']*100:.1f}%")
    print(f"  Avg latency: {r['avg_latency_ms']:.0f}ms  Total: {r['total_wallclock_s']:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--think", action="store_true")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"Loading {args.n} labeled samples (seed={args.seed})...", file=sys.stderr)
    samples = load_samples(args.n, args.seed)
    print(f"Loaded {len(samples)}. Testing {args.model} (think={args.think}) parallel={args.parallel}", file=sys.stderr)
    r = run_test(args.model, args.think, samples, parallel=args.parallel, verbose=True)
    print_report(r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
