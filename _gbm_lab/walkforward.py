"""
Walk-forward validation — the real overfitting check.

Strategy: sort matches by their earliest decision_ts, train on the first 70%
chronologically, evaluate on the next 15% (validation, for picking τ), then
test on the final 15% (held-out, never seen). If the +EV from shootout_v2
survives the time split, it's real. If not, we were lucky on a random split.
"""
import json, random, time
import numpy as np
from features import extract_with_context, CTX_FEATURES
from shootout_v2 import build_rows, BASIC, FEATURES, to_arr

random.seed(42); np.random.seed(42)


def time_split(rows, train_frac=0.70, val_frac=0.15):
    """Order matches by their earliest decision_ts, then split chronologically."""
    by_match = {}
    for r in rows:
        # decision_ts isn't on the row dict; we need to add it during build_rows.
        # For now we'll fall back to using min(decision_ts) per match read from sft.
        pass


def build_rows_with_ts(sft_path: str = "data/training/sft.jsonl"):
    """Same as shootout_v2.build_rows but preserves decision_ts."""
    rows = []
    for row, ctx in extract_with_context(sft_path):
        lbl = (row.get("label") or {})
        action = lbl.get("action", "skip")
        state = (row.get("state") or {})
        trig = (row.get("trigger") or {})
        t1 = state.get("team_one") or {}
        t2 = state.get("team_two") or {}
        market = row.get("market_before") or {}
        first = next(iter(market.values()), {}) if market else {}
        try:
            bid = float(first.get("bid", 0.5) or 0.5)
            ask = float(first.get("ask", 0.5) or 0.5)
        except Exception:
            bid, ask = 0.5, 0.5
        event_type = trig.get("event_type", "UNKNOWN")
        out = {
            "match_id": str(row.get("match_id", "")),
            "decision_ts": float(row.get("decision_ts", 0) or 0),
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


def chronological_match_split(rows, train_frac=0.70, val_frac=0.15):
    """Order matches by earliest decision_ts; then 70/15/15 chronologically."""
    earliest = {}
    for r in rows:
        mid = r["match_id"]
        earliest[mid] = min(earliest.get(mid, float("inf")), r["decision_ts"])
    ordered = sorted(earliest.items(), key=lambda kv: kv[1])
    n = len(ordered)
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    train_match = {m for m, _ in ordered[:n_tr]}
    val_match = {m for m, _ in ordered[n_tr:n_tr + n_va]}
    test_match = {m for m, _ in ordered[n_tr + n_va:]}
    # Date ranges for reporting
    from datetime import datetime, timezone
    def dt(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"Match-id split (chronological, by earliest decision_ts):")
    print(f"  TRAIN: {len(train_match)} matches  ({dt(ordered[0][1])} → {dt(ordered[n_tr-1][1])})")
    print(f"  VAL:   {len(val_match)} matches  ({dt(ordered[n_tr][1])} → {dt(ordered[n_tr+n_va-1][1])})")
    print(f"  TEST:  {len(test_match)} matches  ({dt(ordered[n_tr+n_va][1])} → {dt(ordered[-1][1])})")
    tr = [r for r in rows if r["match_id"] in train_match]
    va = [r for r in rows if r["match_id"] in val_match]
    te = [r for r in rows if r["match_id"] in test_match]
    return tr, va, te


def metrics(name, proba, y, pnl):
    print(f"\n=== {name} ===")
    print(f"{'τ':>5} {'buys':>5} {'prec':>6} {'rec':>6} {'avg_pnl':>9} {'sim_pnl':>10}")
    out = {}
    for tau in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
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
        out[tau] = (n, prec, rec, avg, sim)
    return out


def main():
    print("Building rows with timestamps…")
    rows = build_rows_with_ts("data/training/sft.jsonl")
    print(f"Total: {len(rows)} rows from {len({r['match_id'] for r in rows})} matches")

    tr, va, te = chronological_match_split(rows)
    print(f"Rows -> train={len(tr)}  val={len(va)}  test={len(te)}")

    X_tr, y_tr, _ = to_arr(tr)
    X_va, y_va, pnl_va = to_arr(va)
    X_te, y_te, pnl_te = to_arr(te)

    pos_w = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    import lightgbm as lgb
    import xgboost as xgb

    print("\n--- LightGBM walk-forward ---")
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=pos_w, random_state=42, verbose=-1,
    )
    lgbm.fit(X_tr, y_tr)
    print(f"trained {time.time()-t0:.1f}s")
    val_m = metrics("LightGBM on VAL (for τ picking)", lgbm.predict_proba(X_va)[:, 1], y_va, pnl_va)
    test_m = metrics("LightGBM on TEST (held-out future)", lgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)

    print("\n--- XGBoost walk-forward ---")
    t0 = time.time()
    xgbm = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=6,
        min_child_weight=2, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_w, random_state=42, tree_method="hist", eval_metric="logloss",
    )
    xgbm.fit(X_tr, y_tr)
    print(f"trained {time.time()-t0:.1f}s")
    val_x = metrics("XGBoost on VAL", xgbm.predict_proba(X_va)[:, 1], y_va, pnl_va)
    test_x = metrics("XGBoost on TEST", xgbm.predict_proba(X_te)[:, 1], y_te, pnl_te)


if __name__ == "__main__":
    main()
