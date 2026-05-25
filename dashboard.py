"""
Real-time monitoring dashboard for the Esports Latency Bot.
Flask + SSE for live updates, adapted from BTC bot dashboard pattern.

State is written to a temp JSON file by the bot thread and read by Flask
workers — no shared queue means no cross-thread locking issues.
"""
import json
import os
import time
import threading
import logging
from flask import Flask, Response, render_template, jsonify

import config

logger = logging.getLogger(__name__)

# Use absolute template path so it works regardless of cwd
_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_dir, "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

_STATE_FILE = "/tmp/esports-bot-state.json"
_latest_state: str = ""  # pre-serialized JSON string
_state_lock = threading.Lock()
_state_version = 0


_last_file_write = 0

def push_state(state):
    """Write bot state — called from bot thread on every update."""
    global _latest_state, _state_version, _last_file_write
    d = state.to_dict()
    try:
        _json_cache = json.dumps(d)
    except (TypeError, ValueError):
        return
    with _state_lock:
        _latest_state = _json_cache
        _state_version += 1
    # Write to file only every 5s (slow IO), memory is always fresh
    now = time.time()
    if now - _last_file_write > 5:
        _last_file_write = now
        try:
            tmp = _STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write(_json_cache)
            os.replace(tmp, _STATE_FILE)
        except Exception:
            pass


def _read_state():
    """Read latest state as JSON string — safe to call from any thread."""
    with _state_lock:
        return _latest_state, _state_version


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/logs")
def api_logs():
    """Return last 100 lines from the bot log."""
    import os, subprocess
    # Try multiple log file locations
    log_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nohup.out"),
        "/tmp/edge-bot.log",
        "nohup.out",
    ]
    for log_path in log_paths:
        try:
            if os.path.exists(log_path):
                result = subprocess.run(["tail", "-100", log_path], capture_output=True, text=True, timeout=2)
                lines = result.stdout.strip().split("\n") if result.stdout else []
                if lines and len(lines) > 0 and lines[0]:
                    return jsonify({"lines": lines[-100:]})
        except Exception:
            continue
    return jsonify({"lines": ["No log file found"]})


@app.route("/api/state")
def api_state():
    """Polling endpoint — returns current state as pre-serialized JSON."""
    state_json, _ = _read_state()
    if not state_json:
        return jsonify({"status": "no_update"})
    return app.response_class(
        response=state_json,
        status=200,
        mimetype='application/json'
    )


@app.route("/api/stream")
def api_stream():
    """Legacy SSE endpoint — redirects to polling. Kept to avoid 404s from cached pages."""
    state_json, _ = _read_state()
    if not state_json:
        return jsonify({"status": "no_update"})
    return app.response_class(response=state_json, status=200, mimetype='application/json')


@app.route("/api/mm")
def api_mm():
    """Market-making stats — aggregate + top tokens. Empty if runner not ready."""
    try:
        import os
        from mm_live import get_runner
        runner = get_runner()
        stats = runner.aggregate_stats()
        # Surface the live-trading flag so the dashboard can show LIVE vs SHADOW
        stats["live_trading"] = os.environ.get("MM_LIVE_TRADING", "false").lower() == "true"
        try:
            from mm_live_trader import get_trader
            stats["live_trader"] = get_trader().status()
        except Exception as e:
            stats["live_trader"] = {"error": str(e)[:100]}
        return jsonify({
            "stats": stats,
            "top_tokens": runner.top_tokens(8),
        })
    except Exception as e:
        return jsonify({"stats": {}, "top_tokens": [], "error": str(e)[:200]})


@app.route("/api/mm_feed")
def api_mm_feed():
    """Recent MM decisions + fills feed (30 of each)."""
    try:
        import sqlite3
        from mm_live import get_runner
        runner = get_runner()
        with sqlite3.connect(str(runner.db_path), timeout=5.0) as c:
            c.execute("PRAGMA busy_timeout=5000")
            fills = c.execute(
                "SELECT ts, token_id, match_id, side, price, "
                "sim_inv_after, sim_cash_after FROM mm_fills "
                "ORDER BY ts DESC LIMIT 30"
            ).fetchall()
            decisions = c.execute(
                "SELECT ts, token_id, action, bid_price, ask_price, "
                "book_bid, book_ask, recent_drift_cents, reason "
                "FROM mm_decisions ORDER BY ts DESC LIMIT 30"
            ).fetchall()
        return jsonify({
            "fills": [
                {"ts": r[0], "token_id": r[1] or "", "match_id": r[2] or "",
                 "side": r[3], "price": r[4], "sim_inv_after": r[5],
                 "sim_cash_after": r[6]}
                for r in fills
            ],
            "decisions": [
                {"ts": r[0], "token_id": r[1] or "", "action": r[2],
                 "bid_price": r[3], "ask_price": r[4],
                 "book_bid": r[5], "book_ask": r[6],
                 "recent_drift_cents": r[7], "reason": r[8]}
                for r in decisions
            ],
        })
    except Exception as e:
        return jsonify({"fills": [], "decisions": [], "error": str(e)[:200]})


@app.route("/api/mm_daily")
def api_mm_daily():
    """Daily sim-PnL series for the PnL-curve dashboard.

    Per UTC day: fill count + realized sim cash (ask_fill=+price, bid_fill=-price)
    plus a running cumulative. Also returns headline today / 7d / lifetime numbers
    and a per-game split.
    """
    try:
        import sqlite3, time
        from mm_live import get_runner
        runner = get_runner()
        with sqlite3.connect(str(runner.db_path), timeout=5.0) as c:
            c.execute("PRAGMA busy_timeout=5000")
            rows = c.execute(
                "SELECT date(ts,'unixepoch') d, COUNT(*) n, "
                "COALESCE(SUM(CASE WHEN side='ask_fill' THEN price ELSE -price END),0) cash "
                "FROM mm_fills GROUP BY d ORDER BY d"
            ).fetchall()
            now = time.time()
            lifetime = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(CASE WHEN side='ask_fill' THEN price "
                "ELSE -price END),0) FROM mm_fills"
            ).fetchone()
            def window(secs):
                r = c.execute(
                    "SELECT COUNT(*), COALESCE(SUM(CASE WHEN side='ask_fill' THEN price "
                    "ELSE -price END),0) FROM mm_fills WHERE ts > ?", (now - secs,)
                ).fetchone()
                return {"fills": r[0] or 0, "cash": round(r[1] or 0.0, 4)}
        days, cum = [], 0.0
        for d, n, cash in rows:
            cum += cash
            days.append({"date": d, "fills": n, "cash": round(cash, 4),
                         "cum": round(cum, 4)})
        n_days = max(1, len(days))
        return jsonify({
            "days": days,
            "today": window(86400),
            "d7": window(7 * 86400),
            "lifetime": {"fills": lifetime[0] or 0, "cash": round(lifetime[1] or 0.0, 4)},
            "avg_per_day": round((lifetime[1] or 0.0) / n_days, 4),
            "target_per_day": 100.0,
        })
    except Exception as e:
        return jsonify({"days": [], "error": str(e)[:200]})


@app.route("/pnl")
def pnl_page():
    """Focused daily sim-PnL curve dashboard."""
    return render_template("pnl.html")


# ── Market-metadata cache (token_id -> readable match info) ─────────────────
# The MM runner only knows token_ids; team names live in Polymarket market
# data. We keep a throttled cache (refresh <=1/20s) so the live dashboard can
# show readable match names without hammering the Gamma API on every refresh.
_meta_cache: dict = {}
_meta_last = 0.0
_meta_lock = threading.Lock()


def _get_market_meta():
    global _meta_cache, _meta_last
    now = time.time()
    with _meta_lock:
        if now - _meta_last < 20 and _meta_cache:
            return _meta_cache
    try:
        from market import MarketFinder
        mf = MarketFinder()
        mkts = mf.fetch_esports_markets()  # 30s internal cache too
        meta = {}
        for m in mkts:
            info = {
                "game": m.game, "team_a": m.team_a or "", "team_b": m.team_b or "",
                "question": m.question or "", "liquidity": m.liquidity or 0,
                "market_id": m.market_id,
                "price_a": getattr(m, "price_a", None),
                "price_b": getattr(m, "price_b", None),
            }
            if m.token_id_a:
                meta[m.token_id_a] = dict(info, side="a")
            if m.token_id_b:
                meta[m.token_id_b] = dict(info, side="b")
        with _meta_lock:
            _meta_cache = meta
            _meta_last = now
        return meta
    except Exception:
        return _meta_cache


def _tok_label(meta, token_id):
    m = meta.get(token_id)
    if not m:
        return {"label": (token_id[:10] + "…") if token_id else "?", "game": "?"}
    ta, tb = m.get("team_a", ""), m.get("team_b", "")
    side = m.get("side")
    matchup = f"{ta} vs {tb}" if ta and tb else (m.get("question", "")[:40])
    # which side this token represents
    this = ta if side == "a" else tb if side == "b" else ""
    return {"label": matchup, "side_team": this, "game": m.get("game", "?"),
            "question": m.get("question", "")[:60]}


@app.route("/api/mm_positions")
def api_mm_positions():
    """Current open inventory per token, enriched with match names + mark-to-market."""
    try:
        from mm_live import get_runner
        runner = get_runner()
        meta = _get_market_meta()
        out = []
        with runner._lock:
            strat_items = list(runner._strategies.items())
            last_book = dict(runner._last_book)
            last_quote = {k: dict(v) for k, v in runner._last_quote.items()}
            tok_game = dict(getattr(runner, "_token_game", {}))
        for tok, s in strat_items:
            inv = getattr(s, "inventory", 0.0) or 0.0
            cash = getattr(s, "cash", 0.0) or 0.0
            book = last_book.get(tok, {})
            bid, ask = book.get("bid"), book.get("ask")
            mid = (bid + ask) / 2 if (bid and ask) else None
            # mark-to-market: liquidate inventory at current mid
            mtm = cash + (inv * mid if (mid and inv) else 0.0)
            lbl = _tok_label(meta, tok)
            q = last_quote.get(tok, {})
            if abs(inv) < 1e-9 and not q.get("bid_price") and not q.get("ask_price"):
                continue  # flat + not quoting → skip
            out.append({
                "token_id": tok, "game": tok_game.get(tok, lbl.get("game", "?")),
                "match": lbl["label"], "side_team": lbl.get("side_team", ""),
                "inventory": round(inv, 2), "cash": round(cash, 4),
                "mtm": round(mtm, 4), "n_fills": getattr(s, "n_fills", 0),
                "book_bid": bid, "book_ask": ask,
                "our_bid": q.get("bid_price"), "our_ask": q.get("ask_price"),
            })
        # open positions first (inventory != 0), then active quotes
        out.sort(key=lambda r: (abs(r["inventory"]) == 0, -abs(r["mtm"])))
        return jsonify({"positions": out, "count": len(out)})
    except Exception as e:
        return jsonify({"positions": [], "error": str(e)[:200]})


@app.route("/api/mm_markets")
def api_mm_markets():
    """Active esports markets we could/do quote — the 'matches' view.

    Shows every active market from Polymarket with: game, matchup, liquidity,
    current book, and whether the MM is engaged (has a strategy / active quote).
    """
    try:
        from mm_live import get_runner
        runner = get_runner()
        meta = _get_market_meta()
        with runner._lock:
            strat_toks = set(runner._strategies.keys())
            quoting = {k for k, v in runner._last_quote.items()
                       if v.get("bid_price") is not None or v.get("ask_price") is not None}
            tok_game = dict(getattr(runner, "_token_game", {}))
        # collapse to one row per market_id (side a)
        seen = set(); rows = []
        for tok, m in meta.items():
            mid_ = m.get("market_id")
            if mid_ in seen or m.get("side") != "a":
                continue
            seen.add(mid_)
            tok_b = None
            for t2, m2 in meta.items():
                if m2.get("market_id") == mid_ and m2.get("side") == "b":
                    tok_b = t2; break
            engaged = (tok in strat_toks or tok in quoting
                       or (tok_b and (tok_b in strat_toks or tok_b in quoting)))
            rows.append({
                "game": m.get("game", "?"),
                "match": f"{m.get('team_a','')} vs {m.get('team_b','')}".strip(" vs"),
                "question": m.get("question", "")[:60],
                "liquidity": round(m.get("liquidity", 0) or 0),
                "price_a": m.get("price_a"), "price_b": m.get("price_b"),
                "quoting": bool(tok in quoting or (tok_b and tok_b in quoting)),
                "engaged": bool(engaged),
            })
        rows.sort(key=lambda r: (not r["quoting"], not r["engaged"], -r["liquidity"]))
        n_q = sum(1 for r in rows if r["quoting"])
        return jsonify({"markets": rows, "count": len(rows), "quoting": n_q})
    except Exception as e:
        return jsonify({"markets": [], "error": str(e)[:200]})


@app.route("/live")
def live_page():
    """Live command-center dashboard: positions, markets, decisions feed."""
    return render_template("live.html")


def create_app(bot):
    """Create Flask app wired to the bot."""
    bot.on_state_change(push_state)
    return app
