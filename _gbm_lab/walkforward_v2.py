"""
Walk-forward v2 — runs the time-split, applies isotonic calibration on val,
and emits a full backtest report on the test set with bootstrap CIs.
"""
import random, time
import numpy as np
from walkforward import build_rows_with_ts, chronological_match_split, to_arr
from backtest import report as bt_report

random.seed(42); np.random.seed(42)


def main():
    print("Building rows with timestamps + context features…")
    rows = build_rows_with_ts("data/training/sft.jsonl")
    tr, va, te = chronological_match_split(rows)
    X_tr, y_tr, _ = to_arr(tr)
    X_va, y_va, pnl_va = to_arr(va)
    X_te, y_te, pnl_te = to_arr(te)
    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    print("\nTraining LightGBM…")
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)

    proba_va = lgbm.predict_proba(X_va)[:, 1]
    proba_te = lgbm.predict_proba(X_te)[:, 1]

    # Isotonic calibration fit on val
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_va, y_va)
    proba_va_cal = iso.transform(proba_va)
    proba_te_cal = iso.transform(proba_te)

    # Backtest rows
    def rows_for(rows_list, proba):
        return [{
            "decision_ts": r["decision_ts"],
            "label": r["label"],
            "pred_proba": float(p),
            "pnl": r["hindsight_pnl"],
        } for r, p in zip(rows_list, proba)]

    print("\n--- UNCALIBRATED test set ---")
    bt_report(rows_for(te, proba_te), label="LightGBM TEST (raw)")

    print("\n--- ISOTONIC-CALIBRATED test set ---")
    bt_report(rows_for(te, proba_te_cal), label="LightGBM TEST (calibrated)")

    # Sanity: how well-calibrated is the val set?
    print("\n--- Calibration sanity on VAL (deciles of calibrated proba) ---")
    order = np.argsort(proba_va_cal)
    nb = 10
    chunks = np.array_split(order, nb)
    print(f"{'bucket':>8} {'avg_proba':>10} {'actual_buy_rate':>16} {'n':>5}")
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        avg_p = proba_va_cal[idx].mean()
        actual = y_va[idx].mean()
        print(f"  {i+1:>5} {avg_p:>10.3f} {actual:>15.3f} {len(idx):>5}")


if __name__ == "__main__":
    main()
