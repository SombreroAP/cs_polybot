"""
Try stronger regularization + smaller model to see if the time-generalization
failure is just classic overfitting. Hypothesis: 24 features × 11K rows is over-
fitting; lighter model should generalize better in walk-forward.
"""
import random
import numpy as np
from walkforward import build_rows_with_ts, to_arr
from rolling_walkforward import rolling_predictions
from backtest import report as bt_report

random.seed(42); np.random.seed(42)


def main():
    rows = build_rows_with_ts("data/training/sft.jsonl")

    # Same setup but train each window with stronger regularization
    earliest = {}
    for r in rows:
        earliest[r["match_id"]] = min(earliest.get(r["match_id"], float("inf")), r["decision_ts"])
    ordered = [m for m, _ in sorted(earliest.items(), key=lambda kv: kv[1])]
    rows_by_match = {}
    for r in rows:
        rows_by_match.setdefault(r["match_id"], []).append(r)

    import lightgbm as lgb
    agg = []
    print("Heavy-regularization rolling walk-forward (num_leaves=15, n_est=150, reg=2.0)")
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
            n_estimators=150, learning_rate=0.03, num_leaves=15,
            min_child_samples=50, reg_alpha=2.0, reg_lambda=2.0,
            scale_pos_weight=pos_w, random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        for r, p in zip(te, proba):
            agg.append({
                "decision_ts": r["decision_ts"], "label": r["label"],
                "pred_proba": float(p), "pnl": r["hindsight_pnl"],
            })
    print(f"Aggregated {len(agg)} predictions.")
    bt_report(agg, label="Heavy-regularized LightGBM rolling walk-forward")


if __name__ == "__main__":
    main()
