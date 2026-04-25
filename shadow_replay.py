#!/usr/bin/env python3
"""
Shadow replay — for every Haiku decision the production bot logged, replay the
exact same prompt through a local 5090 model and compare the two decisions.

Why: the cleanest possible "could a local model replace Haiku?" measurement.
We already paid Haiku once; replaying its inputs locally is free, and gives us
a rolling agreement rate + cost-replaced signal as data accumulates.

Inputs (read-only):
    The VPS sqlite (`data/trades.db` on the production host) has table
    `claude_decisions` with the exact event_window/game_state/market_state that
    Haiku saw, plus Haiku's recorded action/confidence/cost. We `scp` the db
    to a tmp file each run.

Output:
    data/shadow_replay/<model>_<ts>.jsonl — per-decision local result
    data/shadow_replay/summary.jsonl     — append-only daily roll-ups
    Optional Telegram message via --telegram

Usage:
    # Replay last 24 h of decisions through ministral-3:14b on the 5090
    python shadow_replay.py --model ollama://ministral-3:14b --since-hours 24 --telegram

    # Quick smoke test (5 rows)
    python shadow_replay.py --model ollama://mistral-small3.2:24b --limit 5

    # Compare two models on the same window
    python shadow_replay.py --model ollama://ministral-3:14b --since-hours 168
    python shadow_replay.py --model ollama://mistral-small3.2:24b --since-hours 168
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# Reuse the production prompt builder + system prompt — guarantees the local
# model sees the exact same input Haiku saw.
import edge_analyst as _ea  # noqa: E402

# `_build_prompt` is an instance method but doesn't touch `self` — bind it to a
# stub object so we can call it without instantiating EdgeAnalyst (which would
# pull in anthropic + aiohttp + db setup).
class _Stub: pass
_STUB = _Stub()


def build_user_prompt(events, game, market) -> str:
    return _ea.EdgeAnalyst._build_prompt(_STUB, events, game, market)


def call_local_model(url: str, model: str, system: str, user: str,
                     timeout: float = 30.0) -> dict:
    t0 = time.time()
    try:
        r = requests.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 200},
                "keep_alive": "1h",
            },
            timeout=timeout,
        )
        r.raise_for_status()
        text = (r.json().get("message") or {}).get("content", "") or ""
    except Exception as e:
        return {"action": "error", "confidence": 0.0,
                "reason": f"http_error: {type(e).__name__}",
                "latency_ms": (time.time() - t0) * 1000}
    latency = (time.time() - t0) * 1000
    try:
        d = json.loads(text)
    except Exception:
        i, j = text.find("{"), text.rfind("}")
        try:
            d = json.loads(text[i:j + 1]) if 0 <= i < j else {}
        except Exception:
            d = {}
    if not isinstance(d, dict):
        d = {}
    return {
        "action": (d.get("action") or "parse_err").lower(),
        "confidence": float(d.get("confidence") or 0.0),
        "reason": (d.get("reason") or "")[:200],
        "latency_ms": latency,
    }


def fetch_decisions(vps_host: str, since_ts: float) -> list[dict]:
    """Pull a coherent snapshot of the VPS sqlite. sqlite is in WAL mode so the
    `.db` file alone is stale — recent writes live in `.db-wal` until checkpoint.
    Easiest reliable path: ssh-checkpoint in-place, then scp the merged `.db`."""
    tmp = Path("/tmp/cs2bot_trades_replay.db")
    # Ask sqlite on the VPS to merge WAL into the main file (read-only safe).
    subprocess.run(
        ["ssh", vps_host,
         "cd ~/esports && sqlite3 data/trades.db 'PRAGMA wal_checkpoint(PASSIVE);' >/dev/null"],
        check=False, timeout=30,
    )
    subprocess.run(
        ["scp", "-q", f"{vps_host}:~/esports/data/trades.db", str(tmp)],
        check=True, timeout=120,
    )
    # Also pull the WAL+shm in case the checkpoint didn't fully flush; sqlite
    # will read them transparently if they're alongside the .db.
    for ext in ("-wal", "-shm"):
        subprocess.run(
            ["scp", "-q",
             f"{vps_host}:~/esports/data/trades.db{ext}",
             str(tmp) + ext],
            check=False, timeout=120,
        )
    conn = sqlite3.connect(str(tmp))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM claude_decisions WHERE timestamp >= ? ORDER BY timestamp",
        (since_ts,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            ev = json.loads(r["event_window"]) if r["event_window"] else []
            gs = json.loads(r["game_state"]) if r["game_state"] else {}
            ms = json.loads(r["market_state"]) if r["market_state"] else {}
        except Exception:
            continue
        out.append({
            "timestamp": r["timestamp"], "match_id": r["match_id"],
            "team": r["team"], "game": r["game"],
            "haiku_action": r["action"], "haiku_confidence": r["confidence"],
            "haiku_reason": r["reason"] or "", "haiku_cost": r["cost"] or 0,
            "events": ev, "game_state": gs, "market_state": ms,
        })
    conn.close()
    return out


def telegram(msg: str) -> bool:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except Exception:
        pass
    tok = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    uid = os.environ.get("TELEGRAM_USER_ID", "")
    if not tok or not uid:
        print("[shadow_replay] TELEGRAM_BOT_TOKEN / _USER_ID missing — skipping send")
        return False
    try:
        requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            data={"chat_id": uid, "text": msg}, timeout=10,
        ).raise_for_status()
        return True
    except Exception as e:
        print(f"[shadow_replay] telegram send failed: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama://ministral-3:14b")
    ap.add_argument("--ollama-url", default="http://192.168.76.196:11434")
    ap.add_argument("--vps", default="bot@85.137.174.57")
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on rows to replay (0 = no cap)")
    ap.add_argument("--telegram", action="store_true",
                    help="send summary to Telegram on exit")
    args = ap.parse_args()

    if not args.model.startswith("ollama://"):
        sys.exit("only ollama:// models supported (yet)")
    model = args.model[len("ollama://"):]

    since_ts = time.time() - args.since_hours * 3600
    print(f"[shadow_replay] pulling claude_decisions since "
          f"{time.ctime(since_ts)} from {args.vps}")
    rows = fetch_decisions(args.vps, since_ts)
    if args.limit and len(rows) > args.limit:
        rows = rows[:args.limit]
    if not rows:
        msg = f"Shadow replay {model}: no Haiku decisions in last {args.since_hours}h"
        print(msg)
        if args.telegram:
            telegram(msg)
        return

    print(f"[shadow_replay] {len(rows)} decisions, model={model}")
    out_dir = REPO / "data" / "shadow_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_name = model.replace(":", "-").replace("/", "-")
    out_file = out_dir / f"{safe_name}_{ts}.jsonl"

    agree = parse_err = 0
    haiku_buy_local_skip = haiku_skip_local_buy = 0
    latencies: list[float] = []
    haiku_total_cost = 0.0
    divergences: list[dict] = []

    with open(out_file, "w") as fh:
        for i, row in enumerate(rows):
            user = build_user_prompt(
                row["events"], row["game_state"], row["market_state"]
            )
            local = call_local_model(args.ollama_url, model, _ea.SYSTEM_PROMPT, user)
            haiku = row["haiku_action"]
            same = (local["action"] == haiku)
            if local["action"] in ("error", "parse_err"):
                parse_err += 1
            elif same:
                agree += 1
            else:
                if haiku == "buy" and local["action"] == "skip":
                    haiku_buy_local_skip += 1
                elif haiku == "skip" and local["action"] == "buy":
                    haiku_skip_local_buy += 1
                if len(divergences) < 5:
                    divergences.append({
                        "match_id": row["match_id"], "team": row["team"],
                        "haiku": {"action": haiku,
                                  "conf": row["haiku_confidence"],
                                  "reason": row["haiku_reason"][:80]},
                        "local": {"action": local["action"],
                                  "conf": local["confidence"],
                                  "reason": local["reason"][:80]},
                    })
            latencies.append(local["latency_ms"])
            haiku_total_cost += float(row["haiku_cost"] or 0)

            fh.write(json.dumps({
                "ts": row["timestamp"], "match_id": row["match_id"],
                "team": row["team"], "haiku_action": haiku,
                "haiku_conf": row["haiku_confidence"],
                "local_action": local["action"],
                "local_conf": local["confidence"],
                "local_latency_ms": local["latency_ms"],
                "agree": same,
            }) + "\n")
            if (i + 1) % 50 == 0:
                ag = agree / (i + 1) if (i + 1) else 0
                print(f"  [{i+1}/{len(rows)}] agreement={ag:.1%} "
                      f"haiku-buy-local-skip={haiku_buy_local_skip} "
                      f"haiku-skip-local-buy={haiku_skip_local_buy}")

    n = len(rows)
    valid = max(1, n - parse_err)
    summary = {
        "ts": time.time(), "model": model, "n_decisions": n,
        "agreement_pct": round(100 * agree / valid, 1),
        "haiku_buy_local_skip": haiku_buy_local_skip,
        "haiku_skip_local_buy": haiku_skip_local_buy,
        "parse_errors": parse_err,
        "local_latency_p50_ms": int(statistics.median(latencies)) if latencies else 0,
        "local_latency_p95_ms": int(sorted(latencies)[int(len(latencies)*0.95)]) if len(latencies) > 1 else 0,
        "haiku_cost_replayed": round(haiku_total_cost, 4),
        "out_file": str(out_file),
        "divergences_sample": divergences,
    }
    print("\n" + "=" * 70)
    print(f"SHADOW REPLAY  {model}  ({n} decisions)")
    print("=" * 70)
    for k, v in summary.items():
        if k != "divergences_sample":
            print(f"  {k:<24} {v}")
    print("=" * 70)
    with open(out_dir / "summary.jsonl", "a") as fh:
        fh.write(json.dumps(summary) + "\n")

    if args.telegram:
        msg = (
            f"Shadow replay: {model}\n"
            f"window: last {args.since_hours:.0f}h\n"
            f"\n"
            f"replayed:        {n}\n"
            f"agreement:       {summary['agreement_pct']}%\n"
            f"haiku BUY → local SKIP: {haiku_buy_local_skip}\n"
            f"haiku SKIP → local BUY: {haiku_skip_local_buy}\n"
            f"parse errors:    {parse_err}\n"
            f"local p50:       {summary['local_latency_p50_ms']}ms\n"
            f"haiku cost replayed: ${summary['haiku_cost_replayed']:.4f}"
        )
        telegram(msg)


if __name__ == "__main__":
    main()
