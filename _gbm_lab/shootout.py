"""
GBM shootout — same features as Tier 1C, four classifiers head-to-head.
Goal: confirm LightGBM/XGBoost/CatBoost beat sklearn's GradientBoostingClassifier
on the same feature set. If yes, we know which model to use going forward.

Reproducible: seed=42 throughout, same match-id split as Tier 1C.
"""
import json, random, time
import numpy as np

random.seed(42); np.random.seed(42)


def extract_rows(sft_path: str):
    rows = []
    with open(sft_path) as f:
        for line in f:
            try:
                e = json.loads(line)
                lbl = (e.get("label") or {})
                action = lbl.get("action", "skip")
                state = (e.get("state") or {})
                market = (e.get("market_before") or {})
                trig = (e.get("trigger") or {})
                t1 = state.get("team_one") or {}
                t2 = state.get("team_two") or {}
                best_bid = best_ask = 0.5
                if market:
                    first = next(iter(market.values()), {})
                    best_bid = float(first.get("bid", 0.5) or 0.5)
                    best_ask = float(first.get("ask", 0.5) or 0.5)
                event_type = trig.get("event_type", "UNKNOWN")
                rows.append({
                    "match_id": str(e.get("match_id", "")),
                    "label": action,
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
                    "best_bid": best_bid, "best_ask": best_ask,
                    "spread": best_ask - best_bid, "mid": (best_bid + best_ask) / 2,
                    "event_round_end": 1 if event_type in ("GAME_EVENT_MATCH_END_ROUND", "round_end") else 0,
                    "event_kill": 1 if event_type in ("GAME_EVENT_PLAYER_KILL", "kill_streak") else 0,
                    "event_objective": 1 if "objective" in event_type.lower() else 0,
                    "event_freeze": 1 if "FREEZE" in event_type or "freeze" in event_type else 0,
                    "round_phase_buy": 1 if state.get("round_phase") == "BUY_TIME" else 0,
                    "round_phase_live": 1 if state.get("round_phase") == "IN_PROGRESS" else 0,
                    "hindsight_pnl": e.get("hindsight_pnl_pct", 0),
                })
            except Exception:
                pass
    return rows


FEATURES = [
    "round_number", "score_a", "score_b", "score_diff", "score_total",
    "match_score_a", "match_score_b", "match_score_diff",
    "money_a", "money_b", "money_diff",
    "alive_a", "alive_b", "alive_diff",
    "best_bid", "best_ask", "spread", "mid",
    "event_round_end", "event_kill", "event_objective", "event_freeze",
    "round_phase_buy", "round_phase_live",
]


def match_split(rows, train_frac=0.80, seed=42):
    match_ids = sorted({r["match_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(match_ids)
    split = int(len(match_ids) * train_frac)
    train_match = set(match_ids[:split])
    train_rows = [r for r in rows if r["match_id"] in train_match]
    test_rows = [r for r in rows if r["match_id"] not in train_match]
    return train_rows, test_rows


def to_arrays(rows):
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([1 if r["label"] == "buy" else 0 for r in rows], dtype=int)
    pnl = np.array([r["hindsight_pnl"] for r in rows], dtype=float)
    return X, y, pnl


def evaluate(name, proba, y_test, pnl_test, taus=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    print(f"\n=== {name} ===")
    print(f"{'τ':>5} {'buys':>5} {'prec':>6} {'rec':>6} {'avg_pnl':>9} {'sim_pnl':>10}")
    best = None
    for tau in taus:
        pred = (proba >= tau).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        n_buys = tp + fp
        if n_buys == 0:
            print(f"{tau:>5} {n_buys:>5} {'-':>6} {'-':>6} {'-':>9} {'-':>10}")
            continue
        prec = tp / n_buys
        rec = tp / max(1, tp + fn)
        avg_pnl = pnl_test[pred == 1].mean()
        sim_pnl = pnl_test[pred == 1].sum()
        print(f"{tau:>5} {n_buys:>5} {prec:>.3f} {rec:>.3f} {avg_pnl:>+8.2f}% {sim_pnl:>+9.1f}%")
        if n_buys >= 20:
            entry = (tau, n_buys, prec, rec, avg_pnl, sim_pnl)
            if best is None or entry[4] > best[4]:
                best = entry
    if best is not None:
        tau, n, prec, rec, avg, sim = best
        print(f"  BEST (n_buys>=20, by avg_pnl): τ={tau:.2f} buys={n} prec={prec:.3f} rec={rec:.3f} avg_pnl={avg:+.2f}% sim_pnl={sim:+.1f}%")
    return best


def main():
    rows = extract_rows("data/training/sft.jsonl")
    train_rows, test_rows = match_split(rows)
    print(f"Train: {len(train_rows)}  Test: {len(test_rows)}")
    X_tr, y_tr, _ = to_arrays(train_rows)
    X_te, y_te, pnl_te = to_arrays(test_rows)

    # Baseline: sklearn (Tier 1C reference)
    from sklearn.ensemble import GradientBoostingClassifier
    t0 = time.time()
    sk = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    sk.fit(X_tr, y_tr)
    print(f"sklearn GBM trained in {time.time()-t0:.1f}s")
    evaluate("sklearn GradientBoostingClassifier (baseline)", sk.predict_proba(X_te)[:, 1], y_te, pnl_te)

    # LightGBM
    import lightgbm as lgb
    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=-1, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_weight, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    print(f"\nLightGBM trained in {time.time()-t0:.1f}s (scale_pos_weight={pos_weight:.1f})")
    evaluate("LightGBM", lgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)

    # XGBoost
    import xgboost as xgb
    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_weight, random_state=42,
        tree_method="hist", eval_metric="logloss",
    )
    xgbm.fit(X_tr, y_tr)
    print(f"\nXGBoost trained in {time.time()-t0:.1f}s")
    evaluate("XGBoost", xgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)

    # CatBoost
    from catboost import CatBoostClassifier
    t0 = time.time()
    cb = CatBoostClassifier(
        iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function="Logloss", auto_class_weights="Balanced",
        random_seed=42, verbose=False,
    )
    cb.fit(X_tr, y_tr)
    print(f"\nCatBoost trained in {time.time()-t0:.1f}s")
    evaluate("CatBoost", cb.predict_proba(X_te)[:, 1], y_te, pnl_te)


if __name__ == "__main__":
    main()
