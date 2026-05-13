"""
Final pipeline — runs once the fresh sft.jsonl is built.

Steps:
  1. Load fresh sft.jsonl
  2. Build rows with context features
  3. Rolling walk-forward (multiple time windows) with LightGBM
  4. EV-rule filter: bet only when model_proba > market_ask + edge
  5. Combined filter: EV rule + avoid low-mid markets + avoid 1-round-close games
  6. Position sizing by edge magnitude (Kelly-fractional)
  7. Full report with bootstrap CIs
"""
import random
import numpy as np
from walkforward import build_rows_with_ts, to_arr
from backtest import realized_pnl, equity_curve, drawdown_max, bootstrap_ci

random.seed(42); np.random.seed(42)


def rolling_with_context(rows, train_w=80, test_w=15, step=15):
    """Like rolling_predictions but also keeps the row's full context for filtering."""
    earliest = {}
    for r in rows:
        earliest[r["match_id"]] = min(earliest.get(r["match_id"], float("inf")), r["decision_ts"])
    ordered = [m for m, _ in sorted(earliest.items(), key=lambda kv: kv[1])]
    rows_by_match = {}
    for r in rows:
        rows_by_match.setdefault(r["match_id"], []).append(r)

    import lightgbm as lgb
    aggregated = []
    print(f"Rolling walk-forward: {len(ordered)} matches, window={train_w} train + {test_w} test, step={step}")
    for start in range(0, len(ordered) - train_w - test_w + 1, step):
        train_match = ordered[start: start + train_w]
        test_match = ordered[start + train_w: start + train_w + test_w]
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
            aggregated.append({
                **r,
                "pred_proba": float(p),
            })
        print(f"  window {start}-{start+train_w}/{start+train_w+test_w}: "
              f"train={len(tr)} test={len(te)} buys_in_test={int(y_te.sum())}")
    return aggregated


def metric_block(bets, label):
    if not bets:
        print(f"  {label}: 0 bets")
        return
    pnls = [b["hindsight_pnl"] / 100 - 0.003 for b in bets]  # apply slippage buffer
    win = sum(1 for p in pnls if p > 0) / len(pnls)
    mean, lo, hi = bootstrap_ci(pnls)
    curve = equity_curve([(b["decision_ts"], b["label"], p) for b, p in zip(bets, pnls)])
    print(f"  {label}: n={len(bets):>4}  win_rate={win*100:5.1f}%  "
          f"avg_pnl={mean*100:+6.2f}%  CI=[{lo*100:+6.2f}, {hi*100:+6.2f}]  "
          f"final=${curve[-1][1]:,.0f}  max_DD={drawdown_max(curve)*100:.1f}%")


def main():
    import sys
    sft_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/sft.jsonl"
    print(f"\nLoading {sft_path}…")
    rows = build_rows_with_ts(sft_path)
    print(f"Total rows: {len(rows)}  unique matches: {len({r['match_id'] for r in rows})}")
    if not rows:
        print("No rows — abort"); return
    print(f"Date range: {min(r['decision_ts'] for r in rows):.0f} → {max(r['decision_ts'] for r in rows):.0f}")

    agg = rolling_with_context(rows, train_w=80, test_w=15, step=15)
    print(f"\nAggregated {len(agg)} predictions.")

    print("\n" + "="*88)
    print("EXPERIMENT MATRIX — all numbers from honest rolling walk-forward")
    print("="*88)

    # 1. Baseline: threshold on model proba
    for tau in [0.3, 0.5, 0.7, 0.85]:
        bets = [b for b in agg if b["pred_proba"] >= tau]
        metric_block(bets, f"GBM τ≥{tau:.2f}")

    print()
    # 2. EV rule: model_proba > market_ask + edge
    for edge in [0.0, 0.05, 0.10, 0.15, 0.20]:
        bets = [b for b in agg if b["pred_proba"] > b["best_ask"] + edge]
        metric_block(bets, f"EV-rule edge≥{edge:.2f}")

    print()
    # 3. Combined: EV rule + segment filters
    print("Combined filter: EV-rule + mid in [0.15, 0.85] + |score_diff|!=1")
    for edge in [0.0, 0.05, 0.10, 0.15]:
        bets = [b for b in agg if (
            b["pred_proba"] > b["best_ask"] + edge
            and 0.15 <= b["mid"] <= 0.85
            and abs(b["score_diff"]) != 1
        )]
        metric_block(bets, f"combined edge≥{edge:.2f}")

    print()
    # 4. Sized-position simulation with combined filter
    print("Kelly-fractional sizing (combined filter, edge≥0.05):")
    bankroll = 10000.0
    bets = sorted([b for b in agg if (
        b["pred_proba"] > b["best_ask"] + 0.05
        and 0.15 <= b["mid"] <= 0.85
        and abs(b["score_diff"]) != 1
    )], key=lambda b: b["decision_ts"])
    curve = []
    for b in bets:
        # Fractional Kelly: f = (p × (1-ask)/ask − (1-p)) / ((1-ask)/ask)
        p = b["pred_proba"]; ask = max(0.05, b["best_ask"])
        b_odds = (1 - ask) / ask
        kelly = max(0, (p * b_odds - (1 - p)) / b_odds)
        stake = bankroll * min(0.02, kelly * 0.25)  # 1/4 Kelly, capped at 2%
        pnl_pct = b["hindsight_pnl"] / 100 - 0.003
        bankroll += stake * pnl_pct
        curve.append((b["decision_ts"], bankroll))
    if curve:
        peak = max(b for _, b in curve)
        dd = (peak - min(b for _, b in curve if curve.index((_, b)) >= curve.index((curve[0])))) / peak if peak > 0 else 0
        print(f"  bets={len(bets)}  final=${curve[-1][1]:,.0f}  "
              f"return={(curve[-1][1]/10000-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
