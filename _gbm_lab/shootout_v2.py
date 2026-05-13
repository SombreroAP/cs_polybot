"""
GBM shootout v2 — same four models, but with the richer feature set
(match context features added).
"""
import json, random, time
import numpy as np
from features import extract_with_context, CTX_FEATURES

random.seed(42); np.random.seed(42)


def _first_market_prices(market: dict):
    if not market:
        return 0.5, 0.5
    first = next(iter(market.values()), {})
    bid = first.get("bid", 0.5) or 0.5
    ask = first.get("ask", 0.5) or 0.5
    try:
        return float(bid), float(ask)
    except Exception:
        return 0.5, 0.5


def build_rows(sft_path):
    rows = []
    for row, ctx in extract_with_context(sft_path):
        lbl = (row.get("label") or {})
        action = lbl.get("action", "skip")
        state = (row.get("state") or {})
        trig = (row.get("trigger") or {})
        t1 = state.get("team_one") or {}
        t2 = state.get("team_two") or {}
        bid, ask = _first_market_prices(row.get("market_before") or {})
        event_type = trig.get("event_type", "UNKNOWN")

        out = {
            "match_id": str(row.get("match_id", "")),
            "label": action,
            "hindsight_pnl": row.get("hindsight_pnl_pct", 0),
            "round_number": state.get("round_number", 0) or 0,
            "score_a": t1.get("score") or 0, "score_b": t2.get("score") or 0,
            "score_diff": (t1.get("score") or 0) - (t2.get("score") or 0),
            "score_total": (t1.get("score") or 0) + (t2.get("score") or 0),
            "match_score_a": t1.get("match_score") or 0,
            "match_score_b": t2.get("match_score") or 0,
            "match_score_diff": (t1.get("match_score") or 0) - (t2.get("match_score") or 0),
            "money_a": t1.get("equipment_value") or 0,
            "money_b": t2.get("equipment_value") or 0,
            "money_diff": (t1.get("equipment_value") or 0) - (t2.get("equipment_value") or 0),
            "alive_a": t1.get("players_alive") or 5,
            "alive_b": t2.get("players_alive") or 5,
            "alive_diff": (t1.get("players_alive") or 5) - (t2.get("players_alive") or 5),
            "best_bid": bid, "best_ask": ask,
            "spread": ask - bid, "mid": (bid + ask) / 2,
            "event_round_end": 1 if event_type in ("GAME_EVENT_MATCH_END_ROUND", "round_end") else 0,
            "event_kill": 1 if event_type in ("GAME_EVENT_PLAYER_KILL", "kill_streak") else 0,
            "event_objective": 1 if "objective" in event_type.lower() else 0,
            "event_freeze": 1 if "FREEZE" in event_type or "freeze" in event_type else 0,
            "round_phase_buy": 1 if state.get("round_phase") == "BUY_TIME" else 0,
            "round_phase_live": 1 if state.get("round_phase") == "IN_PROGRESS" else 0,
        }
        out.update(ctx)
        rows.append(out)
    return rows


BASIC = [
    "round_number", "score_a", "score_b", "score_diff", "score_total",
    "match_score_a", "match_score_b", "match_score_diff",
    "money_a", "money_b", "money_diff",
    "alive_a", "alive_b", "alive_diff",
    "best_bid", "best_ask", "spread", "mid",
    "event_round_end", "event_kill", "event_objective", "event_freeze",
    "round_phase_buy", "round_phase_live",
]
FEATURES = BASIC + CTX_FEATURES


def split(rows, frac=0.8, seed=42):
    ids = sorted({r["match_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    tr = set(ids[: int(len(ids) * frac)])
    return [r for r in rows if r["match_id"] in tr], [r for r in rows if r["match_id"] not in tr]


def to_arr(rows):
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "buy" else 0 for r in rows], dtype=int)
    pnl = np.array([r["hindsight_pnl"] for r in rows], dtype=float)
    return X, y, pnl


def evaluate(name, proba, y, pnl):
    print(f"\n=== {name} ===")
    print(f"{'τ':>5} {'buys':>5} {'prec':>6} {'rec':>6} {'avg_pnl':>9} {'sim_pnl':>10}")
    best = None
    for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        pred = (proba >= tau).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n = tp + fp
        if n == 0:
            print(f"{tau:>5} {n:>5} {'-':>6} {'-':>6} {'-':>9} {'-':>10}")
            continue
        prec = tp / n
        rec = tp / max(1, tp + fn)
        avg = pnl[pred == 1].mean()
        sim = pnl[pred == 1].sum()
        print(f"{tau:>5} {n:>5} {prec:>.3f} {rec:>.3f} {avg:>+8.2f}% {sim:>+9.1f}%")
        if n >= 20 and (best is None or avg > best[4]):
            best = (tau, n, prec, rec, avg, sim)
    if best:
        tau, n, prec, rec, avg, sim = best
        print(f"  BEST (n>=20): τ={tau:.2f} buys={n} prec={prec:.3f} rec={rec:.3f} avg_pnl={avg:+.2f}% sim_pnl={sim:+.1f}%")
    return best


def main():
    print(f"Features: {len(FEATURES)} ({len(BASIC)} basic + {len(CTX_FEATURES)} context)")
    rows = build_rows("data/training/sft.jsonl")
    tr, te = split(rows)
    print(f"Train: {len(tr)} rows from {len({r['match_id'] for r in tr})} matches")
    print(f"Test:  {len(te)} rows from {len({r['match_id'] for r in te})} matches")
    X_tr, y_tr, _ = to_arr(tr)
    X_te, y_te, pnl_te = to_arr(te)

    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    t0 = time.time()
    sk = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    sk.fit(X_tr, y_tr)
    print(f"\nsklearn GBM trained in {time.time()-t0:.1f}s")
    evaluate("sklearn GradientBoostingClassifier", sk.predict_proba(X_te)[:, 1], y_te, pnl_te)

    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    print(f"\nLightGBM trained in {time.time()-t0:.1f}s")
    evaluate("LightGBM", lgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)

    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_w, random_state=42, tree_method="hist", eval_metric="logloss",
    )
    xgbm.fit(X_tr, y_tr)
    print(f"\nXGBoost trained in {time.time()-t0:.1f}s")
    evaluate("XGBoost", xgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)

    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=42, verbose=False,
    )
    cb.fit(X_tr, y_tr)
    print(f"\nCatBoost trained in {time.time()-t0:.1f}s")
    evaluate("CatBoost", cb.predict_proba(X_te)[:, 1], y_te, pnl_te)

    # Save best model + features for downstream productionisation
    best_model = xgbm  # tentative; will revise after seeing numbers
    import pickle, os
    os.makedirs("_gbm_lab/artifacts", exist_ok=True)
    with open("_gbm_lab/artifacts/xgb_v1.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": FEATURES}, f)
    print("\nSaved _gbm_lab/artifacts/xgb_v1.pkl")

    # Feature importance from XGB
    imp = sorted(zip(FEATURES, xgbm.feature_importances_), key=lambda kv: -kv[1])
    print("\nXGB feature importance (top 15):")
    for f, i in imp[:15]:
        print(f"  {f}: {i:.3f}")


if __name__ == "__main__":
    main()
