"""
Market-making simulator on book-stream data.

Strategy: at each book update, quote a bid at best_bid+δ and ask at best_ask−δ
(improving the inside by δ¢). Hold the quote until either:
  (a) it fills — detected when the NEXT book update shows the opposite side
      crossing our price, OR
  (b) we refresh — at the next book update we cancel + re-quote at the new inside.

Fill detection rules between book[i] and book[i+1]:
  - bid filled  → next_best_ask <= our_bid_price (someone offered at/below us)
  - ask filled  → next_best_bid >= our_ask_price (someone bid at/above us)
  - both        → counts as 1 round-trip, average of two prices

Inventory limits: stop quoting the side that would push |inventory| over cap.

PnL accounting (USD-denominated):
  - Each bid fill: cash -= price * size; inventory += size
  - Each ask fill: cash += price * size; inventory -= size
  - At end of token (match ends): liquidate inventory at best_bid (long) or
    best_ask (short) of the last available book point.

Reports per-token + overall: fills, win-rate per round-trip, average per-fill PnL,
total cash PnL, max inventory exposure, time-in-market.
"""
import json, os, glob, math
import numpy as np
from collections import defaultdict


# Configuration
QUOTE_DELTA = 0.01      # how much we improve the inside by (1¢)
QUOTE_SIZE = 1.0         # 1 share per quote (so PnL is in $ per share)
MAX_INVENTORY = 5.0      # cap absolute inventory at this size
SLIPPAGE_BUFFER = 0.003  # per fill, to cover gas + tx friction
TRADING_FEE = 0.0        # Polymarket V3 outcome markets
LATENCY_PENALTY = 0.0    # set >0 to penalize aged quotes (toxic flow proxy)


def load_book_streams(processed_dir="data/processed"):
    streams = defaultdict(list)
    for path in glob.glob(os.path.join(processed_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") != "book":
                    continue
                tok = r.get("tokenId")
                bid = r.get("bid")
                ask = r.get("ask")
                if not tok or bid is None or ask is None:
                    continue
                streams[tok].append((r["ts"], float(bid), float(ask)))
    for tok in streams:
        streams[tok].sort(key=lambda x: x[0])
    return streams


def simulate_token(pts,
                   quote_delta=QUOTE_DELTA,
                   quote_size=QUOTE_SIZE,
                   max_inv=MAX_INVENTORY,
                   slippage=SLIPPAGE_BUFFER,
                   min_spread=0.0):
    """Run MM on one token's book stream. Return per-token stats dict."""
    if len(pts) < 2:
        return None

    cash = 0.0
    inv = 0.0
    fills = []           # list of (ts, side, price, mid_after_30s, mid_after_60s)
    max_abs_inv = 0.0

    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0

    # Precompute mids for adverse-selection metrics
    mids = [(t, (b + a) / 2) for t, b, a in pts]

    def mid_at(t_target):
        # binary search
        lo, hi = 0, len(mids) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if mids[mid][0] <= t_target:
                lo = mid
            else:
                hi = mid - 1
        return mids[lo][1]

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i + 1]
        spread = ask - bid

        # Fill detection from quotes placed at PREVIOUS update
        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            price = bid_q_price
            cash -= price * quote_size
            inv += quote_size
            fills.append((t_next, "bid_fill", price,
                          mid_at(t_next + 30), mid_at(t_next + 60)))
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size
            inv -= quote_size
            fills.append((t_next, "ask_fill", price,
                          mid_at(t_next + 30), mid_at(t_next + 60)))

        max_abs_inv = max(max_abs_inv, abs(inv))

        # Spread filter — only quote when current spread ≥ min_spread
        if spread < min_spread:
            have_bid_q = False
            have_ask_q = False
            continue

        new_bid_q = bid_next + quote_delta
        new_ask_q = ask_next - quote_delta
        if new_bid_q >= new_ask_q:
            have_bid_q = False
            have_ask_q = False
        else:
            have_bid_q = inv + quote_size <= max_inv
            have_ask_q = inv - quote_size >= -max_inv
            bid_q_price = new_bid_q
            ask_q_price = new_ask_q

    t_last, bid_last, ask_last = pts[-1]
    if inv > 0:
        cash += inv * bid_last
        inv = 0
    elif inv < 0:
        cash -= (-inv) * ask_last
        inv = 0
    cash -= len(fills) * slippage * quote_size

    # Compute adverse selection on the fills
    adverse_30s = []
    for (ts, side, price, mid30, mid60) in fills:
        # For bid_fill we expect post-fill mid to DROP if we got picked off
        # For ask_fill we expect post-fill mid to RISE if we got picked off
        # signed adverse = mid moves AGAINST us
        if side == "bid_fill":
            adverse = price - mid30  # positive = bad (mid dropped below our buy price)
        else:
            adverse = mid30 - price  # positive = bad (mid rose above our sell price)
        adverse_30s.append(adverse)

    return {
        "n_fills": len(fills),
        "pnl_usd": cash,
        "max_abs_inv": max_abs_inv,
        "duration_s": pts[-1][0] - pts[0][0],
        "n_book_pts": len(pts),
        "mean_adverse_30s_cents": np.mean(adverse_30s) * 100 if adverse_30s else 0,
    }


def run_sweep(streams, **kwargs):
    """Run simulate_token across all tokens; return aggregate stats."""
    results = []
    for tok, pts in streams.items():
        r = simulate_token(pts, **kwargs)
        if r is not None and r["n_fills"] > 0:
            results.append(r)
    total_pnl = sum(r["pnl_usd"] for r in results)
    total_fills = sum(r["n_fills"] for r in results)
    n_tokens_active = len(results)
    pnls = sorted([r["pnl_usd"] for r in results])
    win_tokens = sum(1 for p in pnls if p > 0)
    mean_adverse = np.mean([r["mean_adverse_30s_cents"] for r in results if r["n_fills"] > 0])
    return {
        "total_pnl": total_pnl,
        "total_fills": total_fills,
        "n_tokens_active": n_tokens_active,
        "avg_pnl_per_fill_cents": total_pnl / max(1, total_fills) * 100,
        "win_token_pct": win_tokens / max(1, n_tokens_active) * 100,
        "median_token_pnl": pnls[len(pnls) // 2] if pnls else 0,
        "mean_adverse_30s_cents": mean_adverse,
    }


def main():
    print("Loading book streams…")
    streams = load_book_streams()
    print(f"  {len(streams)} tokens, {sum(len(v) for v in streams.values())} book points\n")

    print("=" * 100)
    print("MARKET-MAKING STRATEGY SWEEP")
    print("=" * 100)
    print(f"{'strategy':<48} {'fills':>6} {'tokens':>7} "
          f"{'total_$':>9} {'¢/fill':>8} {'win_tok%':>9} {'adverse30s¢':>12}")
    print("-" * 100)

    configs = [
        # (label, quote_delta, min_spread)
        ("improve_inside_1c — no spread filter",    +0.01, 0.00),
        ("improve_inside_1c — spread>=3¢",          +0.01, 0.03),
        ("improve_inside_1c — spread>=5¢",          +0.01, 0.05),
        ("improve_inside_1c — spread>=10¢",         +0.01, 0.10),
        ("inside_touch_0c — no spread filter",      +0.00, 0.00),
        ("inside_touch_0c — spread>=5¢",            +0.00, 0.05),
        ("inside_touch_0c — spread>=10¢",           +0.00, 0.10),
        ("inside_touch_0c — spread>=20¢",           +0.00, 0.20),
        ("behind_inside_1c — spread>=10¢",          -0.01, 0.10),
        ("behind_inside_1c — spread>=20¢",          -0.01, 0.20),
        ("behind_inside_2c — spread>=10¢",          -0.02, 0.10),
        ("behind_inside_3c — spread>=20¢",          -0.03, 0.20),
        ("behind_inside_5c — spread>=20¢",          -0.05, 0.20),
    ]
    for label, qd, ms in configs:
        r = run_sweep(streams, quote_delta=qd, min_spread=ms)
        sign = "+" if r["total_pnl"] >= 0 else ""
        print(f"{label:<48} {r['total_fills']:>6} {r['n_tokens_active']:>7} "
              f"{sign}{r['total_pnl']:>+8.1f} {r['avg_pnl_per_fill_cents']:>+7.2f}¢ "
              f"{r['win_token_pct']:>7.0f}% {r['mean_adverse_30s_cents']:>+10.2f}¢")


if __name__ == "__main__":
    main()
