"""
Rolling-window walk-forward: train on first N matches, predict the next M,
slide forward. Aggregate predictions across all windows so we get N×M total
test rows instead of just one test split's worth.

This gives us many more bets to evaluate at high thresholds — critical for
narrowing the bootstrap CIs.
"""
import random
import numpy as np
from walkforward import build_rows_with_ts, to_arr
from backtest import report as bt_report

random.seed(42); np.random.seed(42)


def rolling_predictions(rows, train_window_matches=80, test_window_matches=15, step=15):
    """
    Slide a (train_window_matches, test_window_matches) window forward by `step`
    matches each iteration. For each window:
      - train on rows from train matches
      - score test rows from test matches
      - append (decision_ts, label, predicted_proba, hindsight_pnl) to results
    Returns the concatenated list.
    """
    # Order matches by earliest decision_ts
    earliest = {}
    for r in rows:
        mid = r["match_id"]
        earliest[mid] = min(earliest.get(mid, float("inf")), r["decision_ts"])
    ordered_matches = [m for m, _ in sorted(earliest.items(), key=lambda kv: kv[1])]

    rows_by_match = {}
    for r in rows:
        rows_by_match.setdefault(r["match_id"], []).append(r)

    import lightgbm as lgb

    aggregated = []
    n_matches = len(ordered_matches)
    print(f"Rolling walk-forward: {n_matches} total matches")

    for start in range(0, n_matches - train_window_matches - test_window_matches + 1, step):
        train_matches = ordered_matches[start: start + train_window_matches]
        test_matches = ordered_matches[start + train_window_matches:
                                       start + train_window_matches + test_window_matches]
        train_rows = [r for m in train_matches for r in rows_by_match[m]]
        test_rows = [r for m in test_matches for r in rows_by_match[m]]
        if not train_rows or not test_rows:
            continue
        X_tr, y_tr, _ = to_arr(train_rows)
        X_te, y_te, pnl_te = to_arr(test_rows)
        pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=pos_w, random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        for r, p in zip(test_rows, proba):
            aggregated.append({
                "decision_ts": r["decision_ts"],
                "label": r["label"],
                "pred_proba": float(p),
                "pnl": r["hindsight_pnl"],
            })
        print(f"  window matches[{start}:{start+train_window_matches}]->test"
              f"[{start+train_window_matches}:{start+train_window_matches+test_window_matches}] "
              f"train_rows={len(train_rows)} test_rows={len(test_rows)} "
              f"buy_in_test={int(y_te.sum())}")
    print(f"\nAggregated {len(aggregated)} predictions across rolling windows.")
    return aggregated


def main():
    print("Building rows with context features (excluding price-history extras)…")
    rows = build_rows_with_ts("data/training/sft.jsonl")
    print(f"Total: {len(rows)} rows from {len({r['match_id'] for r in rows})} matches")

    agg = rolling_predictions(rows, train_window_matches=80, test_window_matches=15, step=15)
    bt_report(agg, label="LightGBM rolling walk-forward (aggregated)")


if __name__ == "__main__":
    main()
