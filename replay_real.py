#!/usr/bin/env python3
"""
Model benchmark on REAL traded data (85 samples from trades.db).
Each sample has: real fill price, real exit, real pnl (fees + SL/TP logic included).

If model says BUY  → sim pnl = actual trade pnl (we'd have placed this trade)
If model says SKIP → sim pnl = $0 (we'd have skipped)

This avoids the selection-bias of the 10,701 training set.

Usage:
  python3 replay_real.py --model qwen3:30b-a3b
  python3 replay_real.py --model qwen3:30b-a3b-think --think
  python3 replay_real.py --model qwen3:30b-a3b-think --think --n 85 --bootstrap 500
"""
import argparse, json, sqlite3, random, time, re, sys, os, urllib.request, urllib.error

OLLAMA = os.getenv("OLLAMA_URL", "http://192.168.76.196:11435")
DB = "/Users/andrew/Documents/Claude/Projects/Trading/esports/data/trades.db"

# Anthropic pricing per 1M tokens (input, output) — update as needed
CLAUDE_PRICING = {
    "claude-haiku-4-5":   (0.80, 4.00),
    "claude-sonnet-4-5":  (3.00, 15.00),
    "claude-opus-4-5":    (15.00, 75.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
    "claude-haiku-4-5-20251001":  (0.80, 4.00),
}


def ask_claude(prompt, model, system, timeout=60):
    """Return (action, latency_ms, raw_response, tokens_in, tokens_out, cost_usd)."""
    try:
        import anthropic  # noqa
    except ImportError:
        return "err", 0, "anthropic SDK not installed", 0, 0, 0
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    t0 = time.time()
    try:
        resp = client.messages.create(
            model=model, max_tokens=200, temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return "err", (time.time() - t0) * 1000, str(e)[:120], 0, 0, 0
    latency = (time.time() - t0) * 1000
    text = resp.content[0].text if resp.content else ""
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    rate_in, rate_out = CLAUDE_PRICING.get(model, (3.0, 15.0))
    cost = (in_tok * rate_in + out_tok * rate_out) / 1_000_000

    # Extract JSON action
    for m in re.finditer(r'\{[^{}]*?"action"[^{}]*?\}', text, re.DOTALL):
        try:
            j = json.loads(m.group(0))
            a = str(j.get("action", "")).lower().strip()
            if "buy" in a:  return "buy", latency, text[:80], in_tok, out_tok, cost
            if "skip" in a: return "skip", latency, text[:80], in_tok, out_tok, cost
        except Exception: pass
    c80 = text[:80].lower()
    if "skip" in c80: return "skip", latency, text[:80], in_tok, out_tok, cost
    if "buy" in c80:  return "buy", latency, text[:80], in_tok, out_tok, cost
    return "err", latency, text[:80], in_tok, out_tok, cost

SYSTEM = """You analyze esports betting markets to decide BUY or SKIP. Goal: profitable trades, not just any trade.

BUY when ALL of:
- ask is 20¢-75¢ (sweet zone — outside this, asymmetric risk)
- buy_team has at least 2 of: economy advantage (>$3k), kill lead, recent round win, positive 10s/30s bid momentum, favorable map-side
- no clear red flags (opposing team has all the advantages)

SKIP when:
- ask outside 20¢-75¢
- Mixed/weak signals (no clear edge)
- Buy_team has economy disadvantage AND is trailing
- Momentum strongly negative (>-2¢ on 10s and 30s)

Be selective but not paranoid — aim for ~30-40% BUY rate on quality setups.

Output ONLY a single JSON object: {"action":"buy","reason":"<15 words"} OR {"action":"skip","reason":"<15 words"}. Nothing else."""


def load_real_trades():
    """Pull 85 resolved trades with full context + real pnl."""
    con = sqlite3.connect(DB)
    rows = con.execute("""
        SELECT team, fill_price, pnl, exit_reason, analysis, amount, strategy
        FROM positions
        WHERE resolved=1 AND length(analysis)>200
          AND exit_reason IN ('take_profit','stop_loss','time_exit')
    """).fetchall()
    con.close()
    samples = []
    for team, fp, pnl, exit_r, ana_s, amt, strat in rows:
        try:
            ana = json.loads(ana_s)
        except Exception:
            continue
        g = ana.get("game_state") or {}
        m = ana.get("market_state") or {}
        events = ana.get("event_window", [])
        if not g or not m:
            continue
        # Build the same prompt format the live bot uses
        prompt = build_prompt(g, m, events, team)
        samples.append({
            "prompt": prompt,
            "team": team, "fill_price": fp, "amount": amt,
            "real_pnl": pnl,                        # ground truth: actual realized pnl
            "exit_reason": exit_r,
            "is_winner": pnl > 0,
        })
    return samples


def build_prompt(game, market, events, buy_team):
    """Match the edge_analyst.py prompt format closely."""
    ta = game.get("team_a", "A"); tb = game.get("team_b", "B")
    sa, sb = game.get("score_a", 0), game.get("score_b", 0)
    ra, rb = game.get("round_a", 0), game.get("round_b", 0)
    eco_a = game.get("economy_a", 0); eco_b = game.get("economy_b", 0)
    kills_a = game.get("kills_a", 0); kills_b = game.get("kills_b", 0)
    side = game.get("side", "")
    bid = market.get("bid", 0); ask = market.get("ask", 0)
    spread = market.get("spread", 0); price = market.get("price", 0)
    vol = market.get("volume", 0); liq = market.get("liquidity", 0)
    mtype = market.get("market_type", "series")
    stale = market.get("staleness_seconds", 0)
    m10 = market.get("momentum_10", {}).get("bid_change", 0) if isinstance(market.get("momentum_10"), dict) else 0
    m30 = market.get("momentum_30", {}).get("bid_change", 0) if isinstance(market.get("momentum_30"), dict) else 0
    m60 = market.get("momentum_60", {}).get("bid_change", 0) if isinstance(market.get("momentum_60"), dict) else 0
    kambi = market.get("kambi", {})

    lines = [
        f"GAME: CS2 | {ta} vs {tb}  (evaluating BUY on {buy_team})",
        f"Series: {sa}-{sb}  Map rounds: {ra}-{rb}",
    ]
    if side:
        lines.append(f"Side: {buy_team} on {side.upper()}")
    if eco_a or eco_b:
        diff = eco_a - eco_b
        who = ta if diff > 0 else tb if diff < 0 else "tied"
        lines.append(f"Economy: {ta}=${eco_a:,} vs {tb}=${eco_b:,} ({who} +${abs(diff):,})")
    if kills_a or kills_b:
        lines.append(f"Round kills: {ta} {kills_a} - {kills_b} {tb}")
    lines.append(f"PM {mtype}: bid={bid*100:.0f}¢ ask={ask*100:.0f}¢ mid={price*100:.0f}¢ spread={spread*100:.0f}% vol=${vol:,.0f} liq=${liq:,.0f}")
    if m10 or m30 or m60:
        lines.append(f"Bid momentum: 10s={m10*100:+.1f}¢  30s={m30*100:+.1f}¢  60s={m60*100:+.1f}¢")
    if stale > 5:
        lines.append(f"Orderbook stale: {stale:.0f}s old")
    if kambi:
        ka = kambi.get("odds_a", 0); kb = kambi.get("odds_b", 0)
        if ka > 0 and kb > 0:
            lines.append(f"Kambi: {ta}={1/ka*100:.0f}¢ vs {tb}={1/kb*100:.0f}¢")
    if events:
        lines.append(f"Recent events ({len(events)}):")
        for e in events[-5:]:
            lines.append(f"- {e.get('desc', e.get('description',''))[:60]}")
    lines.append(f"\nShould we BUY {buy_team}?")
    return "\n".join(lines)


def ask_ollama(prompt, model, think=False, timeout=120):
    payload = {
        "model": model, "stream": False, "keep_alive": "5m",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0, "num_predict": 2500 if think else 250},
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
    except Exception as e:
        return "err", (time.time() - t0) * 1000, str(e)[:100]

    content = d.get("message", {}).get("content", "") or d.get("response", "")
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    latency = (time.time() - t0) * 1000

    # Extract JSON action
    for m in re.finditer(r'\{[^{}]*?"action"[^{}]*?\}', content, re.DOTALL):
        try:
            j = json.loads(m.group(0))
            a = str(j.get("action", "")).lower().strip()
            if "buy" in a: return "buy", latency, content[:80]
            if "skip" in a: return "skip", latency, content[:80]
        except Exception:
            continue
    # Fallback: look for bare "buy" / "skip" in first 50 chars
    c50 = content[:80].lower()
    if "skip" in c50: return "skip", latency, content[:80]
    if "buy" in c50:  return "buy", latency, content[:80]
    return "err", latency, content[:80]


def bootstrap_ci(values, n_iter=500, alpha=0.05):
    """Return (low, high) 95% CI of mean via bootstrap."""
    if not values: return (0.0, 0.0)
    means = []
    for _ in range(n_iter):
        sample = [random.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(alpha/2 * n_iter)]
    high = means[int((1-alpha/2) * n_iter)]
    return (low, high)


def run_benchmark(model, think, samples, bootstrap=500, verbose=True, use_claude=False):
    random.seed(42)
    results = []
    total_cost = 0.0
    total_in = total_out = 0
    t_start = time.time()
    for i, s in enumerate(samples):
        if use_claude:
            a, lat, _, ti, to, cost = ask_claude(s["prompt"], model, SYSTEM)
            total_cost += cost; total_in += ti; total_out += to
        else:
            a, lat, _ = ask_ollama(s["prompt"], model, think=think)
        results.append({"action": a, "latency_ms": lat, **s})
        if verbose and (i + 1) % 10 == 0:
            extra = f" cost=${total_cost:.3f}" if use_claude else ""
            print(f"  [{i+1}/{len(samples)}] elapsed={time.time()-t_start:.0f}s{extra}", file=sys.stderr)

    valid = [r for r in results if r["action"] != "err"]
    errs = len(results) - len(valid)

    # Simulate PnL: follow model's advice
    pnl_follow  = sum(r["real_pnl"] for r in valid if r["action"] == "buy")
    pnl_buy_all = sum(r["real_pnl"] for r in valid)
    pnl_oracle  = sum(r["real_pnl"] for r in valid if r["real_pnl"] > 0)

    # Per-trade stats for bootstrap CI
    per_trade_pnl = [r["real_pnl"] if r["action"] == "buy" else 0.0 for r in valid]
    ci_low, ci_high = bootstrap_ci(per_trade_pnl, n_iter=bootstrap)

    # Classification on real outcome (winner vs loser)
    tp = sum(1 for r in valid if r["action"] == "buy"  and r["real_pnl"] > 0)  # caught winner
    tn = sum(1 for r in valid if r["action"] == "skip" and r["real_pnl"] < 0)  # avoided loser
    fp = sum(1 for r in valid if r["action"] == "buy"  and r["real_pnl"] < 0)  # bought loser
    fn = sum(1 for r in valid if r["action"] == "skip" and r["real_pnl"] > 0)  # missed winner
    accuracy = (tp + tn) / max(1, len(valid))
    skip_recall = tn / max(1, tn + fp) if (tn + fp) > 0 else 0
    buy_precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0

    avg_lat = sum(r["latency_ms"] for r in valid) / max(1, len(valid))
    buy_rate = sum(1 for r in valid if r["action"] == "buy") / max(1, len(valid)) * 100

    return {
        "model": model, "think": think, "n": len(results), "errs": errs,
        "buy_rate_pct": buy_rate,
        "accuracy": accuracy, "skip_recall": skip_recall, "buy_precision": buy_precision,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pnl_follow_model": pnl_follow,
        "pnl_buy_all": pnl_buy_all,
        "pnl_oracle": pnl_oracle,
        "pnl_per_trade_ci_low": ci_low * len(valid),
        "pnl_per_trade_ci_high": ci_high * len(valid),
        "avg_latency_ms": avg_lat,
        "wallclock_s": time.time() - t_start,
        "total_cost_usd": total_cost,
        "total_in_tokens": total_in,
        "total_out_tokens": total_out,
    }


def print_report(r):
    print(f"\n═══ {r['model']} ({'THINK' if r['think'] else 'no-think'}) ═══")
    print(f"  Samples: {r['n']}  Errors: {r['errs']}  Buy rate: {r['buy_rate_pct']:.0f}%")
    print(f"  Accuracy:      {r['accuracy']*100:.1f}%")
    print(f"  Skip recall:   {r['skip_recall']*100:.1f}%  (loser avoidance)")
    print(f"  Buy precision: {r['buy_precision']*100:.1f}%  (of buys, winners)")
    print(f"  Confusion: TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")
    print(f"  ─── PnL (REAL $ from trades.db) ───")
    print(f"  Follow model: ${r['pnl_follow_model']:+.2f}   [95% CI: ${r['pnl_per_trade_ci_low']:+.2f} .. ${r['pnl_per_trade_ci_high']:+.2f}]")
    print(f"  Buy-all:      ${r['pnl_buy_all']:+.2f}")
    print(f"  Oracle (max): ${r['pnl_oracle']:+.2f}")
    delta = r["pnl_follow_model"] - r["pnl_buy_all"]
    print(f"  Delta vs Buy-all: ${delta:+.2f}")
    cap = r["pnl_follow_model"] / r["pnl_oracle"] * 100 if r["pnl_oracle"] > 0 else 0
    print(f"  Oracle capture:   {cap:.1f}%")
    print(f"  Avg latency: {r['avg_latency_ms']:.0f}ms   Total: {r['wallclock_s']:.0f}s")
    if r.get("total_cost_usd", 0) > 0:
        print(f"  COST: ${r['total_cost_usd']:.4f} for {r['n']} calls = ${r['total_cost_usd']/r['n']*1000:.2f}/1000 calls")
        print(f"  Tokens: {r['total_in_tokens']:,} in / {r['total_out_tokens']:,} out")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--think", action="store_true")
    p.add_argument("--n", type=int, default=0, help="0 = all")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out", default="")
    p.add_argument("--claude", action="store_true", help="use Claude API (Anthropic) instead of Ollama")
    args = p.parse_args()

    samples = load_real_trades()
    if args.n and args.n < len(samples):
        random.seed(42); random.shuffle(samples)
        samples = samples[:args.n]
    print(f"Loaded {len(samples)} REAL trades (win rate baseline: "
          f"{sum(1 for s in samples if s['is_winner'])/len(samples)*100:.0f}%)", file=sys.stderr)

    r = run_benchmark(args.model, args.think, samples, bootstrap=args.bootstrap, use_claude=args.claude)
    print_report(r)
    if args.out:
        with open(args.out, "w") as f: json.dump(r, f, indent=2)
        print(f"\nSaved {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
