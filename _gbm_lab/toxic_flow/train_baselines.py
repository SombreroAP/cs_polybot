"""
Train CPU baselines (LightGBM/XGBoost/CatBoost/LinReg) on toxic-flow regression.

Time-ordered 70/15/15 train/val/test split. Reports val + test MAE/RMSE/R²
and saves model files to _gbm_lab/toxic_flow/artifacts/.
"""
import os, sys, time, pickle
import numpy as np
import pandas as pd

FEATURES = [
    "spread", "mid",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "tod_sin", "tod_cos",
]
TARGET = "target_abs_drift_60s"


def load_and_split(path):
    print(f"Loading {path}…")
    df = pd.read_parquet(path)
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_tr = int(n * 0.70)
    n_va = int(n * 0.15)
    train = df.iloc[:n_tr]
    val = df.iloc[n_tr:n_tr+n_va]
    test = df.iloc[n_tr+n_va:]
    print(f"  rows: train={len(train):,} val={len(val):,} test={len(test):,}")
    print(f"  date range train: ts {train['ts'].min():.0f}→{train['ts'].max():.0f}")
    print(f"  date range test:  ts {test['ts'].min():.0f}→{test['ts'].max():.0f}")
    return train, val, test


def metrics(name, y_true, y_pred):
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    # R² (variance explained)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    print(f"  {name:>30}: MAE={mae*100:.3f}c  RMSE={rmse*100:.3f}c  R²={r2:+.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    df_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/toxic_flow.parquet"
    train, val, test = load_and_split(df_path)
    X_tr = train[FEATURES].values
    y_tr = train[TARGET].values
    X_va = val[FEATURES].values
    y_va = val[TARGET].values
    X_te = test[FEATURES].values
    y_te = test[TARGET].values

    os.makedirs("_gbm_lab/toxic_flow/artifacts", exist_ok=True)
    results = {}

    # 1. Naive baselines
    print("\n=== Naive baselines ===")
    y_pred_mean = np.full_like(y_te, y_tr.mean())
    results["mean"] = metrics("mean(train)", y_te, y_pred_mean)
    # Heuristic: use past_drift_30s as the prediction
    y_pred_heur = np.abs(X_te[:, FEATURES.index("past_drift_30s")])
    results["heuristic_past_30s"] = metrics("|past_drift_30s|", y_te, y_pred_heur)

    # 2. Linear regression
    print("\n=== Linear regression ===")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    t0 = time.time(); lr.fit(X_tr, y_tr); print(f"  fit in {time.time()-t0:.1f}s")
    results["linreg"] = metrics("LinearRegression", y_te, lr.predict(X_te))
    pickle.dump({"model": lr, "features": FEATURES},
                open("_gbm_lab/toxic_flow/artifacts/linreg.pkl", "wb"))

    # 3. LightGBM
    print("\n=== LightGBM ===")
    import lightgbm as lgb
    lgbm = lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, verbose=-1,
    )
    t0 = time.time()
    lgbm.fit(X_tr, y_tr,
             eval_set=[(X_va, y_va)],
             callbacks=[lgb.early_stopping(20)])
    print(f"  fit in {time.time()-t0:.1f}s  best_iter={lgbm.best_iteration_}")
    results["lightgbm"] = metrics("LightGBM", y_te, lgbm.predict(X_te))
    pickle.dump({"model": lgbm, "features": FEATURES},
                open("_gbm_lab/toxic_flow/artifacts/lightgbm.pkl", "wb"))

    # 4. XGBoost
    print("\n=== XGBoost ===")
    import xgboost as xgb
    xgbm = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, tree_method="hist", early_stopping_rounds=20,
    )
    t0 = time.time()
    xgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"  fit in {time.time()-t0:.1f}s  best_iter={xgbm.best_iteration}")
    results["xgboost"] = metrics("XGBoost", y_te, xgbm.predict(X_te))
    pickle.dump({"model": xgbm, "features": FEATURES},
                open("_gbm_lab/toxic_flow/artifacts/xgboost.pkl", "wb"))

    # 5. CatBoost
    print("\n=== CatBoost ===")
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="RMSE", random_seed=42, verbose=False,
        early_stopping_rounds=20,
    )
    t0 = time.time()
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    print(f"  fit in {time.time()-t0:.1f}s")
    results["catboost"] = metrics("CatBoost", y_te, cb.predict(X_te))
    pickle.dump({"model": cb, "features": FEATURES},
                open("_gbm_lab/toxic_flow/artifacts/catboost.pkl", "wb"))

    # Summary
    print("\n" + "="*70)
    print("TEST-SET SUMMARY (sorted by MAE ascending — lower is better)")
    print("="*70)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["mae"])
    for name, m in ranked:
        print(f"  {name:>20}: MAE={m['mae']*100:.3f}c  R²={m['r2']:+.4f}")

    # Feature importance from best tree model
    print("\nLightGBM feature importance (top 10):")
    imp = sorted(zip(FEATURES, lgbm.feature_importances_), key=lambda kv: -kv[1])
    for f, i in imp[:10]:
        print(f"  {f}: {i}")


if __name__ == "__main__":
    main()
