#!/usr/bin/env python3
"""
Replay Backtest — runs all recorded CS2 matches through the NEW rich-data
pipeline (per-kill / HP / alive-count / buy-type events) and records what qwen
would have done.

Standalone — does NOT touch the live bot, live DB, or live dashboard.
Serves its own dashboard on port 8083.

Usage:
    python3 replay_backtest.py                       # process all Test Data/*.jsonl, 90-min budget
    python3 replay_backtest.py --duration 5400       # explicit time budget in seconds
    python3 replay_backtest.py --calls-per-match 40  # qwen calls cap per match
    python3 replay_backtest.py --port 8083           # dashboard port

State is entirely in-memory. A JSON snapshot is written to
replay_backtest_results/<timestamp>.json every 30s for later inspection.
"""
import argparse
import asyncio
import glob
import gzip
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional


def _open_recording(path: str):
    """Open a .jsonl or .jsonl.gz file for text reading."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _resolve_mw_tokens(meta_raw: dict) -> tuple[str, str, str, str]:
    """Return (team1_name, team2_name, team1_token, team2_token) for Match Winner.
    Supports THREE META formats seen across our corpus:
      1) token_map: {token_id: {market_type, outcome_index}}  (Test Data)
      2) markets:   [{type, token_ids, outcomes, ...}]        (old_recordings)
      3) token_id_a / token_id_b flat keys                    (live MatchRecorder)
    """
    team1 = meta_raw.get("team1") or ""
    team2 = meta_raw.get("team2") or ""

    # Format 3: flat token_id_a / token_id_b (emitted by match_recorder.py).
    # Checked first because the live recorder is now the dominant producer.
    t1_flat = meta_raw.get("token_id_a") or ""
    t2_flat = meta_raw.get("token_id_b") or ""
    if t1_flat and t2_flat:
        return team1, team2, str(t1_flat), str(t2_flat)

    # Format 1: token_map
    tm = meta_raw.get("token_map") or {}
    if tm:
        t1_tok = t2_tok = ""
        for tok, info in tm.items():
            if info.get("market_type") == "Match Winner":
                if info.get("outcome_index") == 0:
                    t1_tok = tok
                elif info.get("outcome_index") == 1:
                    t2_tok = tok
        if t1_tok and t2_tok:
            return team1, team2, t1_tok, t2_tok

    # Format 2: markets array (old_recordings)
    for mk in meta_raw.get("markets") or []:
        if mk.get("type") == "Match Winner":
            toks = mk.get("token_ids") or []
            outs = mk.get("outcomes") or []
            if isinstance(toks, str):
                try: toks = json.loads(toks.replace("'", '"'))
                except Exception: toks = []
            if isinstance(outs, str):
                try: outs = json.loads(outs.replace("'", '"'))
                except Exception: outs = []
            if len(toks) >= 2:
                t1 = outs[0] if len(outs) >= 1 else team1
                t2 = outs[1] if len(outs) >= 2 else team2
                return t1, t2, str(toks[0]), str(toks[1])

    return team1, team2, "", ""

from flask import Flask, jsonify

# Quiet the noisy feed + market loggers (our own logs carry the signal)
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("backtest")
logger.setLevel(logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from feeds.cs2_bo3 import CS2Bo3Feed  # noqa: E402
from feeds.base import GameEvent, EventType  # noqa: E402
from edge_analyst import EdgeAnalyst  # noqa: E402


# Events that trigger a qwen call during replay
TRIGGER_EVENTS = {
    EventType.ROUND_END,
    EventType.MAP_WIN,
    EventType.OPEN_KILL,
    EventType.CLUTCH_WIN,
    EventType.KILL_STREAK,
    EventType.MATCH_END,
    EventType.ECONOMY_SHIFT,
}


@dataclass
class SimPosition:
    match_id: str
    team: str
    team_letter: str          # "a" or "b"
    token_id: str
    fill_price: float
    shares: float
    bet: float
    tp_pct: float
    sl_pct: float
    spread_at_entry: float    # used for emergency-threshold calc (mirrors prod)
    opened_at_snap: int       # snapshot index when opened
    opened_at_ts: float       # wall-clock when opened (for logs)
    opened_at_rec_ts: float   # RECORDING timestamp when opened — drives TP/SL/time-exit
    reason: str = ""
    exit_reason: str = ""
    exit_price: float = 0.0
    exit_snap: int = 0
    pnl: float = 0.0
    resolved: bool = False


@dataclass
class MatchResult:
    file: str
    match_id: str
    team_a: str = ""
    team_b: str = ""
    series_result: str = ""  # e.g. "2-1" with winning team
    winner: str = ""         # team name
    snapshots: int = 0
    events_emitted: dict = field(default_factory=dict)
    qwen_calls: int = 0
    qwen_buys: int = 0
    qwen_skips: int = 0
    trades: list = field(default_factory=list)
    pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    duration_s: float = 0.0
    status: str = "queued"   # queued|running|done|error
    error: str = ""


# ─── Shared live state (for dashboard) ─────────────────────────────────────────
STATE = {
    "started_at": 0.0,
    "duration_budget_s": 0,
    "elapsed_s": 0,
    "remaining_s": 0,
    "matches_total": 0,
    "matches_done": 0,
    "current_match": "",
    "current_match_snap": 0,
    "current_match_total_snaps": 0,
    "balance": 5000.0,        # backtest-only: larger starting capital so flat $10 bets
    "starting_balance": 5000.0,  # can run across all 72 matches without exhausting funds
    "pnl": 0.0,
    "open_positions": [],
    "matches": [],             # list of MatchResult dicts
    "recent_decisions": [],    # last 40 qwen decisions
    "recent_events": [],       # last 40 emitted events (meaningful ones)
    "qwen_calls": 0,
    "qwen_buys": 0,
    "qwen_skips": 0,
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "status": "starting",
    "model": "",              # ollama model currently being used (shootout label)
    "tag": "",                # run tag (for shootout index)
    "files_source": "",       # file list that was loaded
}
STATE_LOCK = threading.Lock()


def _update(k, v):
    with STATE_LOCK:
        STATE[k] = v

def _append_bounded(key, value, limit):
    with STATE_LOCK:
        lst = STATE.get(key, [])
        lst.append(value)
        STATE[key] = lst[-limit:]


# ─── Flask dashboard on separate port ──────────────────────────────────────────
app = Flask(__name__)


DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<title>Replay Backtest — Rich Data</title>
<style>
  body { font-family: -apple-system, Arial, sans-serif; margin: 0; padding: 20px;
         background: #0d1117; color: #c9d1d9; }
  h1 { color: #58a6ff; margin-top: 0; }
  h2 { color: #7ee787; margin-top: 30px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
  .card { background: #161b22; padding: 14px; border-radius: 8px; border: 1px solid #30363d; }
  .label { font-size: 11px; color: #8b949e; text-transform: uppercase; }
  .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .value.pos { color: #7ee787; }
  .value.neg { color: #ff7b72; }
  table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid #21262d; }
  th { color: #8b949e; font-weight: normal; text-transform: uppercase; font-size: 10px; }
  tr:hover { background: #161b22; }
  .buy { color: #7ee787; }
  .skip { color: #8b949e; }
  .win { color: #7ee787; }
  .loss { color: #ff7b72; }
  .progress { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 8px; }
  .progress-bar { height: 100%; background: linear-gradient(90deg, #58a6ff, #7ee787); }
  .small { font-size: 11px; color: #8b949e; }
  .footer { margin-top: 40px; font-size: 11px; color: #484f58; }
</style>
</head>
<body>
<h1>⚡ Replay Backtest — Rich-Data Pipeline</h1>
<div class="small">Runs recorded CS2 matches through the new kill/HP/alive-count/buy-type feed + qwen.
Separate from the live bot (port 8082) and the live DB. In-memory state only.
<br/><a href="/shootout" style="color:#58a6ff">→ shootout index (all models)</a></div>

<div id="root"></div>

<script>
async function refresh() {
  const r = await fetch('/api/state'); const d = await r.json();
  const pnlCls = d.pnl >= 0 ? 'pos' : 'neg';
  const status = d.status || 'running';
  const pct = d.duration_budget_s > 0 ? Math.min(100, 100 * d.elapsed_s / d.duration_budget_s) : 0;
  const matchPct = d.current_match_total_snaps > 0 ? Math.min(100, 100 * d.current_match_snap / d.current_match_total_snaps) : 0;

  let html = '';
  if (d.model || d.tag) {
    html += `<div class="card" style="margin-bottom:14px;border-left:4px solid #f1e05a">
      <div class="label">MODEL UNDER TEST</div>
      <div class="value" style="color:#f1e05a">${d.model || '—'}</div>
      <div class="small">tag: ${d.tag || '—'} · files: ${d.files_source || 'auto-scan'}</div>
    </div>`;
  }
  html += `
  <div class="grid">
    <div class="card">
      <div class="label">Balance</div>
      <div class="value">$${d.balance.toFixed(2)}</div>
      <div class="small">start $${d.starting_balance.toFixed(2)}</div>
    </div>
    <div class="card">
      <div class="label">P&L</div>
      <div class="value ${pnlCls}">${d.pnl >= 0 ? '+' : ''}$${d.pnl.toFixed(2)}</div>
      <div class="small">${d.total_trades} trades · ${d.wins}W / ${d.losses}L</div>
    </div>
    <div class="card">
      <div class="label">Qwen Calls</div>
      <div class="value">${d.qwen_calls}</div>
      <div class="small">${d.qwen_buys} buy · ${d.qwen_skips} skip</div>
    </div>
    <div class="card">
      <div class="label">Status</div>
      <div class="value">${status.toUpperCase()}</div>
      <div class="small">${d.matches_done}/${d.matches_total} matches</div>
    </div>
  </div>

  <div class="card">
    <div class="small">Time budget: ${(d.elapsed_s/60).toFixed(1)}m / ${(d.duration_budget_s/60).toFixed(0)}m (remaining ${(d.remaining_s/60).toFixed(1)}m)</div>
    <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
    <div class="small" style="margin-top:12px">Current match: <b>${d.current_match || '—'}</b></div>
    <div class="progress"><div class="progress-bar" style="width:${matchPct}%"></div></div>
    <div class="small">snapshot ${d.current_match_snap}/${d.current_match_total_snaps}</div>
  </div>
  `;

  // Open positions
  if (d.open_positions.length > 0) {
    html += '<h2>Open Positions</h2><table><tr><th>Match</th><th>Team</th><th>Fill</th><th>Cur</th><th>PnL%</th><th>Held (snaps)</th></tr>';
    d.open_positions.forEach(p => {
      const pnlPct = ((p.current_price || p.fill_price) - p.fill_price) / p.fill_price * 100;
      const cls = pnlPct >= 0 ? 'win' : 'loss';
      html += `<tr><td>${p.match_label}</td><td>${p.team}</td><td>${p.fill_price.toFixed(3)}</td><td>${(p.current_price||p.fill_price).toFixed(3)}</td><td class="${cls}">${pnlPct.toFixed(1)}%</td><td>${p.age_snaps}</td></tr>`;
    });
    html += '</table>';
  }

  // Per-match results
  html += '<h2>Matches</h2><table><tr><th>File</th><th>Teams</th><th>Result</th><th>Snaps</th><th>Qwen</th><th>Trades</th><th>P&L</th><th>Status</th></tr>';
  d.matches.forEach(m => {
    const cls = m.pnl >= 0 ? 'win' : 'loss';
    html += `<tr>
      <td class="small">${m.file}</td>
      <td>${m.team_a} vs ${m.team_b}</td>
      <td>${m.series_result || '—'} ${m.winner ? '→ '+m.winner : ''}</td>
      <td>${m.snapshots}</td>
      <td>${m.qwen_calls} (${m.qwen_buys}B/${m.qwen_skips}S)</td>
      <td>${m.trades.length} (${m.wins}W/${m.losses}L)</td>
      <td class="${cls}">${m.pnl >= 0 ? '+' : ''}$${m.pnl.toFixed(2)}</td>
      <td>${m.status}</td>
    </tr>`;
  });
  html += '</table>';

  // Recent qwen decisions
  html += '<h2>Recent Qwen Decisions (last 40)</h2><table><tr><th>Time</th><th>Match</th><th>Trig</th><th>Action</th><th>Conf</th><th>Bet</th><th>TP/SL</th><th>Reason</th></tr>';
  d.recent_decisions.slice().reverse().forEach(x => {
    const cls = x.action === 'buy' ? 'buy' : 'skip';
    html += `<tr><td class="small">${x.t}</td><td>${x.match}</td><td class="small">${x.trigger}</td><td class="${cls}">${x.action.toUpperCase()}</td><td>${(x.confidence||0).toFixed(2)}</td><td>$${(x.bet_size||0).toFixed(0)}</td><td class="small">${(x.tp_pct*100||0).toFixed(0)}/${(x.sl_pct*100||0).toFixed(0)}%</td><td class="small">${(x.reason||'').slice(0,90)}</td></tr>`;
  });
  html += '</table>';

  // Recent events
  html += '<h2>Recent Events (last 40)</h2><table><tr><th>Time</th><th>Match</th><th>Event</th><th>Team</th><th>Description</th></tr>';
  d.recent_events.slice().reverse().forEach(x => {
    html += `<tr><td class="small">${x.t}</td><td>${x.match}</td><td>${x.type}</td><td>${x.team}</td><td class="small">${x.description}</td></tr>`;
  });
  html += '</table>';

  html += `<div class="footer">Auto-refresh every 2s. Results JSON saved to replay_backtest_results/.</div>`;

  document.getElementById('root').innerHTML = html;
}
refresh(); setInterval(refresh, 2000);
</script>
</body></html>
"""


@app.route("/")
def dashboard():
    return DASHBOARD_HTML


@app.route("/shootout")
def shootout_index():
    """Serve the shootout index page (written by _model_shootout.sh).
    Falls back to a friendly message if the file doesn't exist yet."""
    idx_path = "_logs/shootout/index.html"
    if os.path.exists(idx_path):
        try:
            with open(idx_path) as f:
                return f.read()
        except Exception as e:
            return f"<pre>could not read {idx_path}: {e}</pre>"
    return ("<html><body style='font-family:Arial;background:#0d1117;color:#c9d1d9;padding:30px'>"
            "<h2>No shootout in progress</h2>"
            "<p>Run <code>bash _model_shootout.sh</code> to start one. This page will auto-populate.</p>"
            "</body></html>")


@app.route("/api/state")
def api_state():
    with STATE_LOCK:
        # Shallow copy — values are primitives or already-snapshot dicts
        out = dict(STATE)
    return jsonify(out)


def run_flask(port: int):
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


# ─── The actual backtest loop ──────────────────────────────────────────────────
async def run_backtest(files: list[str], duration_s: int, calls_per_match: int):
    STATE["started_at"] = time.time()
    STATE["duration_budget_s"] = duration_s
    STATE["matches_total"] = len(files)
    STATE["matches"] = [asdict(MatchResult(file=os.path.basename(f), match_id="")) for f in files]
    _update("status", "running")

    analyst = EdgeAnalyst()

    results: list[MatchResult] = []

    for idx, path in enumerate(files):
        elapsed = time.time() - STATE["started_at"]
        _update("elapsed_s", int(elapsed))
        _update("remaining_s", max(0, int(duration_s - elapsed)) if duration_s > 0 else 0)
        if duration_s > 0 and elapsed >= duration_s:
            logger.info(f"Time budget exhausted ({elapsed:.0f}s) — stopping at match {idx}/{len(files)}")
            break

        res = await replay_one(path, analyst, calls_per_match, duration_s, idx)
        results.append(res)
        with STATE_LOCK:
            STATE["matches"][idx] = asdict(res)
            STATE["matches_done"] = idx + 1

        # Snapshot results to disk
        try:
            out_dir = "replay_backtest_results"
            os.makedirs(out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            with open(f"{out_dir}/progress_{stamp}.json", "w") as f:
                json.dump({k: STATE[k] for k in STATE if k not in ("open_positions",)}, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"snapshot write failed: {e}")

    # Final summary
    final_pnl = sum(m.pnl for m in results)
    total_trades = sum(len(m.trades) for m in results)
    wins = sum(m.wins for m in results)
    losses = sum(m.losses for m in results)
    qwen_calls = sum(m.qwen_calls for m in results)
    logger.info("━" * 70)
    logger.info(f"BACKTEST COMPLETE — {len(results)} matches | PnL ${final_pnl:+.2f} "
                f"| {total_trades} trades ({wins}W/{losses}L) | {qwen_calls} qwen calls")
    logger.info("━" * 70)
    _update("status", "done")

    # Final snapshot
    try:
        out_dir = "replay_backtest_results"
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        with open(f"{out_dir}/FINAL_{stamp}.json", "w") as f:
            json.dump({k: STATE[k] for k in STATE}, f, indent=2, default=str)
    except Exception:
        pass


async def replay_one(path: str, analyst: EdgeAnalyst, calls_cap: int,
                     duration_s: int, idx: int) -> MatchResult:
    res = MatchResult(file=os.path.basename(path), match_id="")
    start_ts = time.time()
    res.status = "running"

    # Per-match feed instance (so we don't cross-contaminate player tracking)
    feed = CS2Bo3Feed()
    events_emitted: list[GameEvent] = []
    def collector(ev: GameEvent):
        events_emitted.append(ev)
    feed.on_event(collector)

    # First pass: just find _META (usually the first line) — stream to avoid loading
    # huge (>400 MB) files into memory all at once.
    match_id: str = ""
    team_a_token: str = ""
    team_b_token: str = ""
    try:
        with _open_recording(path) as fh:
            for raw in fh:
                if '"_META"' not in raw:  # fast filter
                    continue
                try:
                    d = json.loads(raw)
                except Exception:
                    continue
                if d.get("message_type") == "_META":
                    r = d.get("raw", {}) or {}
                    match_id = str(r.get("match_id", ""))
                    res.match_id = match_id
                    t1n, t2n, t1t, t2t = _resolve_mw_tokens(r)
                    res.team_a = t1n or r.get("team1", "")
                    res.team_b = t2n or r.get("team2", "")
                    team_a_token = t1t
                    team_b_token = t2t
                    break
    except Exception as e:
        res.status = "error"; res.error = str(e); return res

    if not match_id:
        res.status = "error"; res.error = "no _META"; return res

    await feed.subscribe_match(match_id)

    # The feed's state.team_a/team_b ordering does NOT always match META's
    # team1/team2 ordering — the bo3.gg snapshot decides on-the-fly. We need to
    # know, for each *feed-side*, which META token to use. We'll resolve this
    # after the first few snapshots populate feed team names, by fuzzy-matching
    # feed name to META team1/team2.
    meta_team1 = res.team_a
    meta_team2 = res.team_b

    def _fuzzy_same(a: str, b: str) -> bool:
        a = (a or "").lower().strip()
        b = (b or "").lower().strip()
        if not a or not b: return False
        if a == b: return True
        if a in b or b in a: return True
        # common abbreviation pattern: first 3 chars
        return a[:3] == b[:3] and len(a) >= 3

    # Will be set lazily once feed state has team names
    feed_token_for_a: str = ""   # token to buy when prob_a >= 0.5 (favor state.team_a)
    feed_token_for_b: str = ""
    mapping_resolved = False

    match_label = f"{res.team_a}vs{res.team_b}"[:40]
    _update("current_match", match_label)
    _update("current_match_snap", 0)
    _update("current_match_total_snaps", 0)  # we don't pre-count now — file might be huge
    logger.info(f"[{idx+1}] REPLAY {match_label} — streaming "
                f"({os.path.getsize(path)/1e6:.0f} MB)")

    # Live state of orderbooks
    books: dict[str, dict] = {}
    # Positions (simulated)
    positions: list[SimPosition] = []
    balance = STATE["balance"]

    qwen_calls = 0
    buys = 0; skips = 0
    snap_idx = 0
    last_qwen_snap = -999  # snapshot-count throttle (was wall-clock, broken in fast replay)
    match_start = time.time()
    current_rec_ts = 0.0   # latest RECORDING timestamp seen — drives TP/SL/time-exit logic
    PER_MATCH_CAP_S = 240  # don't let one match eat the whole 90-min budget

    # Second pass: streaming read (supports .gz too)
    try:
        fh = _open_recording(path)
    except Exception as e:
        res.status = "error"; res.error = str(e); return res

    for raw in fh:
        # Time budget check (global + per-match) — skipped when duration_s <= 0 (unlimited).
        if duration_s > 0:
            elapsed_total = time.time() - STATE["started_at"]
            if elapsed_total >= duration_s:
                logger.info(f"  time budget hit mid-match — stopping replay")
                break
            if time.time() - match_start > PER_MATCH_CAP_S:
                logger.info(f"  per-match cap ({PER_MATCH_CAP_S}s) hit — moving to next match")
                break

        try:
            d = json.loads(raw)
        except Exception:
            continue
        mt = d.get("message_type", "")
        r = d.get("raw", {}) or {}

        # Track recording time from any message carrying ts
        msg_ts = d.get("ts") or 0
        try:
            msg_ts = float(msg_ts)
            if msg_ts > current_rec_ts:
                current_rec_ts = msg_ts
        except Exception:
            pass

        # Orderbook updates (multiple formats across recording eras)
        if mt in ("_PM_book", "_PM_best_bid_ask", "_PRICE_SNAPSHOT", "_PM_book_fresh"):
            tok = r.get("tokenId") or r.get("token_id") or r.get("asset_id")
            if not tok: continue
            bid = None; ask = None
            # bestBid/bestAsk flat format
            if r.get("bestBid") is not None: bid = r.get("bestBid")
            elif r.get("bid") is not None: bid = r.get("bid")
            if r.get("bestAsk") is not None: ask = r.get("bestAsk")
            elif r.get("ask") is not None: ask = r.get("ask")
            # _PM_book_fresh: full depth — take top of book
            if bid is None and isinstance(r.get("bids"), list) and r["bids"]:
                try:
                    bid = max(float(b.get("price", 0)) for b in r["bids"])
                except Exception: pass
            if ask is None and isinstance(r.get("asks"), list) and r["asks"]:
                try:
                    ask = min(float(a.get("price", 0)) for a in r["asks"] if float(a.get("price", 0)) > 0)
                except Exception: pass
            if bid is None and ask is None:
                continue
            books.setdefault(tok, {})
            if bid is not None: books[tok]["bid"] = float(bid)
            if ask is not None: books[tok]["ask"] = float(ask)
            # After orderbook update, check position exits using RECORDING time
            exited = _check_exits(positions, books, snap_idx, current_rec_ts)
            for exit_info in exited:
                balance += exit_info["proceeds"]
                logger.info(f"  EXIT {exit_info['team']:18s} @ {exit_info['exit_price']:.3f} "
                            f"({exit_info['reason']})  pnl=${exit_info['pnl']:+.2f}")
            continue

        if mt != "SNAPSHOT_MATCH_UPDATE":
            continue

        snap_idx += 1
        _update("current_match_snap", snap_idx)
        # Process snapshot
        before = len(events_emitted)
        try:
            feed._process_snapshot(match_id, r)
        except Exception as e:
            logger.debug(f"process_snapshot err: {e}")
            continue
        new_events = events_emitted[before:]

        # Record noteworthy events
        for ev in new_events:
            if ev.event_type in {EventType.OPEN_KILL, EventType.KILL_STREAK,
                                 EventType.CLUTCH_WIN, EventType.MAP_WIN,
                                 EventType.ROUND_END, EventType.MATCH_END,
                                 EventType.HP_DAMAGE, EventType.ECONOMY_SHIFT,
                                 EventType.OBJECTIVE_TAKEN}:
                _append_bounded("recent_events", {
                    "t": time.strftime("%H:%M:%S"), "match": match_label,
                    "type": ev.event_type.value, "team": ev.team,
                    "description": ev.description[:80],
                }, 40)

        # Decide: should we ask qwen?
        if qwen_calls >= calls_cap:
            continue
        trigger_found = next((ev for ev in new_events if ev.event_type in TRIGGER_EVENTS), None)
        if not trigger_found:
            continue
        # Throttle by SIMULATED snapshots (1 Hz in recording), not wall clock —
        # otherwise fast replay runs through everything before qwen ever catches up.
        # Require at least 5 snapshots (~5s simulated) between consecutive calls.
        if snap_idx - last_qwen_snap < 5:
            continue
        last_qwen_snap = snap_idx

        state = feed.get_match_state(match_id)
        if not state:
            continue
        ex = state.extra or {}

        # Resolve feed-side → META token mapping once feed has team names
        if not mapping_resolved and state.team_a and state.team_b:
            if _fuzzy_same(state.team_a, meta_team1):
                feed_token_for_a = team_a_token
                feed_token_for_b = team_b_token
            elif _fuzzy_same(state.team_a, meta_team2):
                # feed's team_a is actually META team2 — SWAP tokens
                feed_token_for_a = team_b_token
                feed_token_for_b = team_a_token
            else:
                # ambiguous — default to order
                feed_token_for_a = team_a_token
                feed_token_for_b = team_b_token
                logger.warning(f"  team mapping ambiguous: feed='{state.team_a}' meta=('{meta_team1}','{meta_team2}')")
            mapping_resolved = True
            logger.info(f"  team mapping: feed.team_a='{state.team_a}' -> token_{('A' if feed_token_for_a == team_a_token else 'B')}, "
                        f"feed.team_b='{state.team_b}' -> token_{('A' if feed_token_for_b == team_a_token else 'B')}")

        if not mapping_resolved:
            continue  # can't trust token mapping yet

        # Choose buy team: bet on the team that JUST HAD the positive event.
        # The feed's win_probability_a model has been shown to be systematically
        # wrong on mid-match CS2 series prediction — we buy momentum instead.
        event_team = trigger_found.team or ""
        if _fuzzy_same(event_team, state.team_a):
            buy_letter = "a"; buy_token = feed_token_for_a; buy_name = state.team_a
        elif _fuzzy_same(event_team, state.team_b):
            buy_letter = "b"; buy_token = feed_token_for_b; buy_name = state.team_b
        else:
            # Event team couldn't be matched (MATCH_END, ECONOMY_SHIFT may lack
            # team) — fall back to feed model pick
            prob_a = state.win_probability_a
            if prob_a >= 0.5:
                buy_letter = "a"; buy_token = feed_token_for_a; buy_name = state.team_a
            else:
                buy_letter = "b"; buy_token = feed_token_for_b; buy_name = state.team_b

        book = books.get(buy_token, {})
        bid = float(book.get("bid", 0) or 0)
        ask = float(book.get("ask", 0) or 0)
        mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
        spread = max(0.0, ask - bid) if (ask > bid) else 0.0
        # Skip obviously untradable
        if ask <= 0 or bid <= 0 or ask >= 0.95 or spread > 0.3:
            continue

        game_state = {
            "game": "cs2",
            "team_a": state.team_a, "team_b": state.team_b,
            "score_a": state.score_a, "score_b": state.score_b,
            "round_a": state.round_score_a, "round_b": state.round_score_b,
            "buy_team": buy_name,
            "economy_a": ex.get("team_a_money", 0),
            "economy_b": ex.get("team_b_money", 0),
            "side": ex.get("team_a_current_side", ""),
            "round_kills_a": ex.get("round_kills_a", 0),
            "round_kills_b": ex.get("round_kills_b", 0),
            "model_price": (state.win_probability_a if buy_letter == "a" else 1 - state.win_probability_a),
            "alive_a": ex.get("alive_a", 5), "alive_b": ex.get("alive_b", 5),
            "avg_hp_a": ex.get("avg_hp_a", 100), "avg_hp_b": ex.get("avg_hp_b", 100),
            "damage_a": ex.get("damage_a", 0), "damage_b": ex.get("damage_b", 0),
            "has_awp_a": bool(ex.get("has_awp_a", False)),
            "has_awp_b": bool(ex.get("has_awp_b", False)),
            "has_defuse_kit_a": bool(ex.get("has_defuse_kit_a", False)),
            "bomb_carrier": ex.get("bomb_carrier", ""),
            "buy_type_a": ex.get("buy_type_a", ""),
            "buy_type_b": ex.get("buy_type_b", ""),
            "map_name": ex.get("map_name", ""),
        }
        market_state = {
            "price": mid, "bid": bid, "ask": ask, "spread": spread,
            "volume": 10000, "liquidity": 5000, "market_type": "series",
            "question": f"{state.team_a} vs {state.team_b}",
            "momentum_10": {}, "momentum_30": {}, "momentum_60": {},
            "staleness_seconds": 0,
            "balance": balance, "open_positions": len([p for p in positions if not p.resolved]),
        }
        window = [
            {"description": e.description[:80], "type": e.event_type.value, "team": e.team}
            for e in new_events[:5]
        ]

        qwen_calls += 1
        try:
            decision = await analyst.should_buy(window, game_state, market_state)
        except Exception as e:
            logger.debug(f"qwen err: {e}")
            continue
        if decision is None:
            continue

        # Record decision
        _append_bounded("recent_decisions", {
            "t": time.strftime("%H:%M:%S"), "match": match_label,
            "trigger": trigger_found.event_type.value,
            "action": decision.action, "confidence": decision.confidence,
            "bet_size": decision.bet_size, "tp_pct": decision.tp_pct,
            "sl_pct": decision.sl_pct, "reason": decision.reason,
        }, 40)

        if decision.action == "buy":
            buys += 1
            # Restored iter3b-style sizing: trust qwen's values, clamped to safety bounds.
            # iter6 showed Python sizing gave more variation but didn't help PnL — the
            # uniform iter3b numbers were actually well-tuned for this market. Keep them.
            raw_bet = float(decision.bet_size or 0)
            if raw_bet <= 0: raw_bet = 15.0
            max_bet = max(5.0, 0.20 * balance)
            bet = max(5.0, min(raw_bet, max_bet))
            if balance - bet < 20.0:
                bet = max(0.0, balance - 20.0)
            tp = decision.tp_pct if 0.02 <= decision.tp_pct <= 0.30 else 0.10
            sl = decision.sl_pct if 0.03 <= decision.sl_pct <= 0.40 else 0.05
            # confidence still plumbed for logging even without sizing use
            conf = max(0.0, min(1.0, float(decision.confidence or 0.0)))
            shares = bet / ask if ask > 0 else 0
            if shares > 0 and bet >= 5.0:
                pos = SimPosition(
                    match_id=match_id, team=buy_name, team_letter=buy_letter,
                    token_id=buy_token, fill_price=ask, shares=shares, bet=bet,
                    tp_pct=tp, sl_pct=sl, spread_at_entry=spread,
                    opened_at_snap=snap_idx, opened_at_ts=time.time(),
                    opened_at_rec_ts=current_rec_ts,
                    reason=decision.reason[:140],
                )
                positions.append(pos)
                balance -= bet
                logger.info(f"  BUY  {buy_name:18s} @ {ask:.3f}  bet=${bet:.2f}  "
                            f"tp={tp*100:.0f}% sl={sl*100:.0f}%  conf={conf:.2f}  "
                            f"rd {state.score_a}-{state.score_b} "
                            f"alive {ex.get('alive_a',5)}v{ex.get('alive_b',5)}")
        else:
            skips += 1

        # Update shared state
        with STATE_LOCK:
            STATE["qwen_calls"] += 1
            if decision.action == "buy": STATE["qwen_buys"] += 1
            else: STATE["qwen_skips"] += 1
            STATE["open_positions"] = [
                {"match_label": match_label, "team": p.team,
                 "fill_price": p.fill_price,
                 "current_price": books.get(p.token_id, {}).get("bid", p.fill_price),
                 "age_snaps": snap_idx - p.opened_at_snap}
                for p in positions if not p.resolved
            ]

    try: fh.close()
    except Exception: pass

    # ─── End of match: close any still-open positions ─────────────────────
    # Determine real winner from final match state. Winner's token is what
    # matters (not the NAME string) — compare p.token_id against the winning
    # token so case/format mismatches don't break resolution.
    state = feed.get_match_state(match_id)
    maps_needed = (state.total_maps // 2) + 1 if state else 2
    winning_token: str = ""
    if state:
        if state.score_a >= maps_needed:
            res.winner = state.team_a
            res.series_result = f"{state.score_a}-{state.score_b}"
            winning_token = feed_token_for_a  # which META token corresponds to feed team_a
        elif state.score_b >= maps_needed:
            res.winner = state.team_b
            res.series_result = f"{state.score_a}-{state.score_b}"
            winning_token = feed_token_for_b
        else:
            res.series_result = f"{state.score_a}-{state.score_b} (partial)"

    for p in positions:
        if p.resolved: continue
        book = books.get(p.token_id, {})
        final_bid = book.get("bid", p.fill_price * 0.95)
        if winning_token:
            if p.token_id == winning_token:
                # Correct pick — market resolves to $1.00
                p.exit_price = 1.0
                p.exit_reason = "resolved_win"
            else:
                p.exit_price = 0.0
                p.exit_reason = "resolved_loss"
        else:
            p.exit_price = final_bid
            p.exit_reason = "replay_ended"
        p.exit_snap = snap_idx
        p.pnl = (p.exit_price - p.fill_price) * p.shares
        p.resolved = True
        balance += p.shares * p.exit_price
        logger.info(f"  CLOSE {p.team:18s} @ {p.exit_price:.3f} ({p.exit_reason})  "
                    f"pnl=${p.pnl:+.2f}  shares={p.shares:.1f}")

    # Compile match stats
    res.snapshots = snap_idx
    res.events_emitted = {
        t.value: sum(1 for e in events_emitted if e.event_type == t)
        for t in set(e.event_type for e in events_emitted)
    }
    res.qwen_calls = qwen_calls
    res.qwen_buys = buys
    res.qwen_skips = skips
    res.trades = [{
        "team": p.team, "fill": round(p.fill_price, 3), "exit": round(p.exit_price, 3),
        "shares": round(p.shares, 2), "bet": round(p.bet, 2),
        "pnl": round(p.pnl, 2), "exit_reason": p.exit_reason,
        "opened_snap": p.opened_at_snap, "closed_snap": p.exit_snap,
        "reason": p.reason,
    } for p in positions]
    res.pnl = sum(p.pnl for p in positions)
    res.wins = sum(1 for p in positions if p.pnl > 0)
    res.losses = sum(1 for p in positions if p.pnl < 0)
    res.duration_s = time.time() - start_ts
    res.status = "done"

    # Update shared running totals
    with STATE_LOCK:
        STATE["balance"] = balance
        STATE["pnl"] = balance - STATE["starting_balance"]
        STATE["total_trades"] += len(positions)
        STATE["wins"] += res.wins
        STATE["losses"] += res.losses
        STATE["open_positions"] = []

    logger.info(f"  [DONE] {match_label}  snaps={snap_idx}  qwen={qwen_calls}  "
                f"trades={len(positions)}  pnl=${res.pnl:+.2f}  bal=${balance:.2f}")
    return res


def _check_exits(positions: list, books: dict, snap_idx: int, now_rec_ts: float) -> list:
    """Mirror executor.check_exit_conditions exactly, using RECORDING time.

    Returns list of dicts describing closed positions so the caller can
    update its running balance.
    """
    exited = []
    for p in positions:
        if p.resolved: continue
        book = books.get(p.token_id, {})
        current_bid = float(book.get("bid", 0) or 0)
        if current_bid <= 0: continue

        # age in RECORDED seconds
        age = (now_rec_ts - p.opened_at_rec_ts) if (now_rec_ts and p.opened_at_rec_ts) else 0.0
        pnl_pct = (current_bid - p.fill_price) / p.fill_price if p.fill_price > 0 else 0

        # spread-adjusted emergency threshold (matches production)
        spread_baseline_pnl = -(p.spread_at_entry / p.fill_price) if p.fill_price > 0 else 0
        emergency_threshold = spread_baseline_pnl - 0.30

        # take profit always active
        reason = None
        if pnl_pct >= p.tp_pct:
            reason = "take_profit"
        else:
            # grace period scales with spread
            grace = 30 if p.spread_at_entry >= 0.15 else 15
            if age < grace:
                if pnl_pct <= emergency_threshold:
                    reason = "stop_loss_emergency"
            else:
                if pnl_pct <= -p.sl_pct:
                    reason = "stop_loss"
                elif pnl_pct <= emergency_threshold:
                    reason = "stop_loss_emergency"
                elif age > 900:
                    reason = "time_exit"

        if reason:
            p.resolved = True
            p.exit_price = current_bid
            p.exit_reason = reason
            p.exit_snap = snap_idx
            p.pnl = (current_bid - p.fill_price) * p.shares
            proceeds = p.shares * current_bid
            exited.append({
                "team": p.team, "exit_price": current_bid,
                "reason": reason, "pnl": p.pnl, "proceeds": proceeds,
            })
    return exited


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=0,
                   help="total time budget in seconds. 0 (default) = UNLIMITED — run all files to completion. "
                        "The qwen backpressure semaphore prevents request backlog regardless of duration.")
    p.add_argument("--calls-per-match", type=int, default=50,
                   help="max qwen calls per match")
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--data-dir", default="Test Data", help="folder with *.jsonl recordings")
    p.add_argument("--files-list", default="",
                   help="optional: path to a text file with one recording path per line "
                        "(overrides --data-dir scan). Use _audit/good_files.txt to run only vetted files.")
    p.add_argument("--starting-balance", type=float, default=1000.0,
                   help="starting simulated balance (default $1000, same as live bot)")
    p.add_argument("--tag", default="", help="free-form tag for output filename")
    p.add_argument("--model", default="",
                   help="override OLLAMA_MODEL for this run (e.g. qwen2.5-coder:32b). "
                        "Enables head-to-head model shootouts without editing config.")
    args = p.parse_args()

    # Optional model override (applied before any EdgeAnalyst is constructed)
    if args.model:
        import config as _cfg
        _cfg.OLLAMA_MODEL = args.model
        print(f"[BACKTEST] Model override: OLLAMA_MODEL = {args.model}")

    # Update starting balance + run metadata in shared state (dashboard reads these)
    STATE["balance"] = args.starting_balance
    STATE["starting_balance"] = args.starting_balance
    STATE["model"] = args.model or ""
    STATE["tag"] = args.tag or ""
    STATE["files_source"] = args.files_list or ""

    # If --files-list is given, load it and skip the scan
    if args.files_list:
        try:
            with open(args.files_list) as fh:
                files = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
            files = [f for f in files if os.path.exists(f)]
            if not files:
                print(f"[BACKTEST] --files-list {args.files_list} has no existing paths")
                return
            total_mb = sum(os.path.getsize(f) for f in files) / 1e6
            print(f"[BACKTEST] Loaded {len(files)} files from {args.files_list} ({total_mb:.0f} MB)")
            print(f"[BACKTEST] Budget: {args.duration/60:.0f} min | calls/match: {args.calls_per_match}")
            print(f"[BACKTEST] Dashboard: http://localhost:{args.port}/")
            print()
            flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
            flask_thread.start()
            time.sleep(1)
            asyncio.run(run_backtest(files, args.duration, args.calls_per_match))
            print("\nBacktest complete.")
            print(f"  Balance: ${STATE['balance']:.2f}")
            print(f"  P&L:     ${STATE['pnl']:+.2f}")
            print(f"  Trades:  {STATE['total_trades']} ({STATE['wins']}W / {STATE['losses']}L)")
            print(f"  Qwen:    {STATE['qwen_calls']} calls ({STATE['qwen_buys']} buys, {STATE['qwen_skips']} skips)")
            return
        except Exception as e:
            print(f"[BACKTEST] error loading {args.files_list}: {e}")
            return

    # Recursively find all .jsonl and .jsonl.gz files under data_dir.
    # Skip enrichment-only stubs, global PM feeds, collector logs, mac metadata.
    all_files = []
    for root, _dirs, filenames in os.walk(args.data_dir):
        for fn in filenames:
            if fn.startswith(".") or fn.startswith("_"):
                continue
            if ".enrich." in fn:            # enrichment stubs — no game data
                continue
            if not (fn.endswith(".jsonl") or fn.endswith(".jsonl.gz")):
                continue
            path = os.path.join(root, fn)
            # Filter obviously empty files (≥20 KB for plain, ≥50 KB for gz)
            sz = os.path.getsize(path)
            if fn.endswith(".gz"):
                if sz < 50_000: continue
            else:
                if sz < 20_000: continue
            all_files.append(path)

    if not all_files:
        print(f"No usable .jsonl files in {args.data_dir}")
        return

    # De-dup by match_id (leading digits in filename) — keep the LARGEST file per match_id
    # so we pick the most complete recording when multiple sessions exist.
    # Prefer plain .jsonl over .gz when sizes are comparable (faster to read).
    import re
    best_per_match: dict = {}
    unmatched = []
    for path in all_files:
        base = os.path.basename(path)
        m = re.match(r"^(\d+)_", base)
        if m:
            mid = m.group(1)
            prev = best_per_match.get(mid)
            cur_sz = os.path.getsize(path)
            if prev is None:
                best_per_match[mid] = path
            else:
                prev_sz = os.path.getsize(prev)
                # uncompressed rough-up for fairness: gz is typically 5-10x compressed
                cur_eff = cur_sz * (7 if path.endswith(".gz") else 1)
                prev_eff = prev_sz * (7 if prev.endswith(".gz") else 1)
                if cur_eff > prev_eff:
                    best_per_match[mid] = path
        else:
            unmatched.append(path)
    files = sorted(list(best_per_match.values()) + unmatched)

    print(f"[BACKTEST] Scanned {len(all_files)} files → {len(files)} unique matches after dedup:")
    total_mb = 0.0
    for f in files:
        size_mb = os.path.getsize(f) / 1e6
        total_mb += size_mb
        rel = os.path.relpath(f, args.data_dir)
        tag = "gz" if f.endswith(".gz") else "  "
        print(f"   [{tag}] {rel}  ({size_mb:.1f} MB)")
    print(f"[BACKTEST] Total: {total_mb/1024:.2f} GB across {len(files)} matches "
          f"(plain {sum(1 for f in files if not f.endswith('.gz'))} / gz {sum(1 for f in files if f.endswith('.gz'))})")
    print(f"[BACKTEST] Budget: {args.duration/60:.0f} min | calls/match: {args.calls_per_match}")
    print(f"[BACKTEST] Dashboard: http://localhost:{args.port}/")
    print()

    # Start dashboard in a thread
    flask_thread = threading.Thread(target=run_flask, args=(args.port,), daemon=True)
    flask_thread.start()
    time.sleep(1)

    # Run the async backtest
    asyncio.run(run_backtest(files, args.duration, args.calls_per_match))

    print("\nBacktest complete. Final state:")
    print(f"  Balance: ${STATE['balance']:.2f}")
    print(f"  P&L:     ${STATE['pnl']:+.2f}")
    print(f"  Trades:  {STATE['total_trades']} ({STATE['wins']}W / {STATE['losses']}L)")
    print(f"  Qwen:    {STATE['qwen_calls']} calls ({STATE['qwen_buys']} buys, {STATE['qwen_skips']} skips)")


if __name__ == "__main__":
    main()
