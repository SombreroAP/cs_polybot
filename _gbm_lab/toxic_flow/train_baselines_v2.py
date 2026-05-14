"""Train baselines on v2 dataset (richer features)."""
import os, sys, time, pickle
import numpy as np
import pandas as pd

FEATURES_V2 = [
    "spread",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "vol_30s", "vol_60s", "vol_300s",
    "spread_change_30s",
    "tod_sin", "tod_cos",
    "dow_sin", "dow_cos",
    "comp_drift_30s", "abs_comp_drift_30s", "comp_mid",
    "comp_sum", "comp_sum_dev",
]
TARGET = "target_abs_drift_60s"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/training/toxic_flow_v2.parquet"
    print(f"Loading {path}…")
    df = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    n = len(df); n_tr = int(n * 0.7); n_va = int(n * 0.15)
    train, val, test = df.iloc[:n_tr], df.iloc[n_tr:n_tr+n_va], df.iloc[n_tr+n_va:]
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    X_tr = train[FEATURES_V2].values; y_tr = train[TARGET].values
    X_va = val[FEATURES_V2].values; y_va = val[TARGET].values
    X_te = test[FEATURES_V2].values; y_te = test[TARGET].values

    def metrics(name, y_true, y_pred):
        err = y_pred - y_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        ss_res = float(np.sum(err**2)); ss_tot = float(np.sum((y_true - y_true.mean())**2))
        r2 = 1 - ss_res / max(ss_tot, 1e-9)
        print(f"  {name:>30}: MAE={mae*100:.3f}c  RMSE={rmse*100:.3f}c  R²={r2:+.4f}")
        return {"mae": mae, "rmse": rmse, "r2": r2}

    os.makedirs("_gbm_lab/toxic_flow/artifacts", exist_ok=True)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    print("\n=== LightGBM v2 ===")
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                          min_child_samples=50, reg_alpha=0.1, reg_lambda=0.5,
                          random_state=42, verbose=-1)
    t0 = time.time(); m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                            callbacks=[lgb.early_stopping(20)]); print(f"  fit {time.time()-t0:.1f}s")
    metrics("LightGBM_v2", y_te, m.predict(X_te))
    pickle.dump({"model": m, "features": FEATURES_V2},
                open("_gbm_lab/toxic_flow/artifacts/lightgbm_v2.pkl", "wb"))

    print("\n=== XGBoost v2 ===")
    m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6,
                         min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                         random_state=42, tree_method="hist", early_stopping_rounds=20)
    t0 = time.time(); m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"  fit {time.time()-t0:.1f}s")
    metrics("XGBoost_v2", y_te, m.predict(X_te))
    pickle.dump({"model": m, "features": FEATURES_V2},
                open("_gbm_lab/toxic_flow/artifacts/xgboost_v2.pkl", "wb"))

    print("\n=== CatBoost v2 ===")
    m = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
                          loss_function="RMSE", random_seed=42, verbose=False,
                          early_stopping_rounds=30)
    t0 = time.time(); m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    print(f"  fit {time.time()-t0:.1f}s")
    metrics("CatBoost_v2", y_te, m.predict(X_te))
    pickle.dump({"model": m, "features": FEATURES_V2},
                open("_gbm_lab/toxic_flow/artifacts/catboost_v2.pkl", "wb"))

    # Reload xgboost v1 (v1 features) for comparison
    print("\n=== Comparison (v1 XGBoost) on v2 test split ===")
    # v1 features subset
    F1 = ["spread", "mid", "past_drift_5s", "past_drift_10s", "past_drift_30s",
          "past_drift_60s", "past_drift_300s",
          "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
          "abs_past_drift_60s", "abs_past_drift_300s",
          "velocity_1s", "accel_5_10", "tod_sin", "tod_cos"]
    v1 = pickle.load(open("_gbm_lab/toxic_flow/artifacts/xgboost.pkl", "rb"))
    if all(f in df.columns for f in F1):
        Xt_v1 = test[F1].values
        metrics("XGBoost_v1 (on v2 split)", y_te, v1["model"].predict(Xt_v1))

    # Feature importance
    import lightgbm as lgb
    lgbm = pickle.load(open("_gbm_lab/toxic_flow/artifacts/lightgbm_v2.pkl", "rb"))
    imp = sorted(zip(FEATURES_V2, lgbm["model"].feature_importances_), key=lambda kv: -kv[1])
    print("\nLightGBM_v2 feature importance (top 15):")
    for f, i in imp[:15]:
        print(f"  {f}: {i}")


if __name__ == "__main__":
    main()
