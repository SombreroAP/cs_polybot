"""
Realistic exit-policy backtest.

For each predicted-buy from the GBM, replay the actual market book updates
in the next 1200s and apply a real exit policy:
  - TAKE PROFIT at +4% from entry ask
  - STOP LOSS  at −4% from entry ask
  - EXPIRY exit at the bid available at t = entry_ts + 1200s

Compares against the hindsight-max we'd previously assumed. Tells us what %
of the +7.25% headline we'd actually realize in live trading.
"""
import json, os, glob, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from walkforward import build_rows_with_ts, to_arr

random.seed(42); np.random.seed(42)


WINDOW_S = 1200


def load_book_streams(processed_dir="data/processed"):
    """Build a per-tokenId list of (ts, bid, ask) sorted by ts."""
    streams = {}
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
                if not tok:
                    continue
                streams.setdefault(tok, []).append(
                    (r["ts"], r.get("bid"), r.get("ask"))
                )
    for tok, lst in streams.items():
        lst.sort(key=lambda x: x[0])
    return streams


def realized_pnl_from_replay(token_id, entry_ts, entry_ask, stream,
                              tp_pct=0.04, sl_pct=0.04, trailing_stop_pct=None):
    """Walk the book stream forward from entry_ts and apply exit policy.

    Parameters
    ----------
    tp_pct : float
        Fixed take-profit threshold in % of entry_ask.
    sl_pct : float
        Fixed stop-loss threshold (positive number).
    trailing_stop_pct : float or None
        If set, use a trailing stop: lock in highest bid - trail%.
        Overrides fixed SL.

    Returns realized return as fraction or None."""
    if entry_ask is None or entry_ask <= 0:
        return None
    import bisect
    keys = [t for t, _, _ in stream]
    i = bisect.bisect_right(keys, entry_ts)
    window_end = entry_ts + WINDOW_S
    tp_price = entry_ask * (1 + tp_pct)
    sl_price = entry_ask * (1 - sl_pct)
    last_bid = None
    max_bid_seen = None
    while i < len(stream):
        t, b, a = stream[i]
        if t > window_end:
            break
        if b is not None:
            last_bid = b
            if max_bid_seen is None or b > max_bid_seen:
                max_bid_seen = b
            if b >= tp_price:
                return (b - entry_ask) / entry_ask
            if trailing_stop_pct is not None and max_bid_seen is not None:
                trail_price = max_bid_seen * (1 - trailing_stop_pct)
                if b <= trail_price and max_bid_seen > entry_ask:
                    return (b - entry_ask) / entry_ask
            if b <= sl_price:
                return (b - entry_ask) / entry_ask
        i += 1
    if last_bid is None:
        return None
    return (last_bid - entry_ask) / entry_ask


def main():
    sft_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/sft_window1200.jsonl"
    print(f"Loading {sft_path}…")
    rows = build_rows_with_ts(sft_path)
    print(f"Total rows: {len(rows)}  matches: {len({r['match_id'] for r in rows})}")

    print("Loading per-token book streams from data/processed/…")
    streams = load_book_streams()
    print(f"Loaded {len(streams)} token streams with "
          f"{sum(len(v) for v in streams.values())} total book updates")

    # Re-run rolling walk-forward predictions
    import lightgbm as lgb
    earliest = {}
    for r in rows:
        earliest[r["match_id"]] = min(earliest.get(r["match_id"], float("inf")), r["decision_ts"])
    ordered = [m for m, _ in sorted(earliest.items(), key=lambda kv: kv[1])]
    rows_by_match = {}
    for r in rows:
        rows_by_match.setdefault(r["match_id"], []).append(r)

    # For each window, train, predict, and collect predictions + raw row for replay
    all_preds = []
    print(f"Rolling walk-forward {len(ordered)} matches…")
    for start in range(0, len(ordered) - 95, 15):
        train_match = ordered[start: start + 80]
        test_match = ordered[start + 80: start + 95]
        tr = [r for m in train_match for r in rows_by_match[m]]
        te = [r for m in test_match for r in rows_by_match[m]]
        if not tr or not te:
            continue
        X_tr, y_tr, _ = to_arr(tr)
        X_te, y_te, pnl_te = to_arr(te)
        pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        m = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5,
            scale_pos_weight=pos_w, random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        for r, p in zip(te, proba):
            all_preds.append({**r, "pred_proba": float(p)})

    print(f"Walk-forward produced {len(all_preds)} predictions.")

    # Need original token_id for each row — we have to dig into the raw SFT
    # Build a lookup from (match_id, decision_ts) -> token_id and entry_ask
    print("Building token_id / entry_ask lookup from raw SFT…")
    token_lookup = {}
    with open(sft_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            mid = str(e.get("match_id", ""))
            ts = float(e.get("decision_ts", 0))
            mb = e.get("market_before") or {}
            # The buy is on whichever token corresponds to the favored side at decision time
            # Conservatively use the FIRST token (matches how we built basic features)
            if mb:
                tok, vals = next(iter(mb.items()))
                token_lookup[(mid, round(ts, 3))] = (tok, vals.get("ask"))

    # Evaluate multiple exit policies on the top filter
    def evaluate(name, filt, tp, sl, trail=None):
        candidates = [p for p in all_preds if filt(p)]
        realized = []
        for p in candidates:
            key = (p["match_id"], round(p["decision_ts"], 3))
            tok_info = token_lookup.get(key)
            if not tok_info:
                continue
            tok, entry_ask = tok_info
            if not entry_ask:
                continue
            stream = streams.get(tok)
            if not stream:
                continue
            rp = realized_pnl_from_replay(tok, p["decision_ts"], entry_ask, stream,
                                          tp_pct=tp, sl_pct=sl, trailing_stop_pct=trail)
            if rp is None:
                continue
            realized.append(rp - 0.003)
        if not realized:
            print(f"{name:>50} {0:>5}      -        -")
            return
        win = sum(1 for r in realized if r > 0) / len(realized)
        avg = sum(realized) / len(realized)
        total = sum(realized)
        print(f"{name:>50} {len(realized):>5} {win*100:>5.1f}% "
              f"{avg*100:>+7.2f}%   sum={total*100:>+8.1f}%")

    print(f"\nKey insight from spread analysis: bets only viable when spread ≤ ~3¢.")
    print(f"Re-evaluating top filters with spread restriction:\n")

    # Combine the model filter with a spread filter
    def combined(model_filter, max_spread):
        return lambda p: model_filter(p) and (p["best_ask"] - p["best_bid"]) <= max_spread

    print(f"{'policy':>60} {'bets':>5} {'win':>6} {'avg_real':>9} {'sum_pnl':>13}")
    print("-" * 105)

    for max_sp in [0.02, 0.03, 0.04]:
        print(f"\n  ── spread ≤ {max_sp*100:.0f}¢ ──")
        evaluate(f"GBM τ≥0.50 + tight",
                 combined(lambda p: p["pred_proba"] >= 0.50, max_sp), 0.04, 0.04)
        evaluate(f"GBM τ≥0.70 + tight",
                 combined(lambda p: p["pred_proba"] >= 0.70, max_sp), 0.04, 0.04)
        evaluate(f"GBM τ≥0.85 + tight",
                 combined(lambda p: p["pred_proba"] >= 0.85, max_sp), 0.04, 0.04)
        evaluate(f"EV-rule edge≥0.05 + tight",
                 combined(lambda p: p["pred_proba"] > p["best_ask"] + 0.05, max_sp), 0.04, 0.04)
        evaluate(f"EV-rule edge≥0.10 + tight",
                 combined(lambda p: p["pred_proba"] > p["best_ask"] + 0.10, max_sp), 0.04, 0.04)
        evaluate(f"EV-rule edge≥0.15 + tight",
                 combined(lambda p: p["pred_proba"] > p["best_ask"] + 0.15, max_sp), 0.04, 0.04)

    # Also try multiple TP/SL on the BEST candidate to nail down the exit policy
    print(f"\n\n── best filter + TP/SL sweep ──")
    best = combined(lambda p: p["pred_proba"] >= 0.85, 0.02)
    for tp, sl in [(0.02, 0.02), (0.02, 0.03), (0.03, 0.02), (0.03, 0.03),
                   (0.04, 0.02), (0.04, 0.04), (0.05, 0.025), (0.06, 0.03)]:
        evaluate(f"GBM τ≥0.85 sp≤2¢ TP=+{tp*100:.0f}% SL=-{sl*100:.0f}%", best, tp, sl)
    for trail in [0.015, 0.02, 0.025, 0.03]:
        evaluate(f"GBM τ≥0.85 sp≤2¢ trail {trail*100:.1f}%", best, 0.10, 0.10, trail)


if __name__ == "__main__":
    main()
