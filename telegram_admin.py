#!/usr/bin/env python3
"""
Telegram admin interface — runs on the VPS as a systemd service.

Lets the authorized user (TELEGRAM_USER_ID from .env) control + query the bot
over Telegram. No python-telegram-bot dependency — uses raw long-polling.

Slash commands:
  /status       Current bot state: mode, balance, uptime, positions, today PnL
  /positions    Open positions with live PnL
  /pnl [today|week|month|all]
  /balance      Current wallet balance + starting balance
  /pause        Stop opening new trades (existing positions continue)
  /resume       Allow new trades again
  /shadow       Switch to shadow mode (log decisions, no orders)
  /live         Switch to live trading mode (real orders) — requires confirmation
  /hardstop     Trigger HARD_STOP.lock — bot refuses to start until manually cleared
  /config       Print current risk config (.env risk block)
  /gpu          Gaming PC GPU status (for training runs)
  /data         Dataset state: matches, SFT examples, audit verdicts
  /tail N       Last N lines of cs2bot log
  /health       Recorder + auto-pipeline + eval-watchdog timer health
  /help         List of all commands

Free-text (any non-slash message):
  Relayed to Claude Haiku with CURRENT bot state injected as context.
  e.g. "why did you enter that FURIA trade?" → Claude reads shadow_trades,
       finds the FURIA row, explains from llm_reason + market_state_snapshot.

Auth: only accepts messages from TELEGRAM_USER_ID. Others are ignored silently.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Env loading
# ─────────────────────────────────────────────────────────────────────────────

def _env() -> dict:
    out = {}
    env_path = HERE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k] = v
                os.environ.setdefault(k, v)
    return out


E = _env()
TG_TOKEN = E.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_USER  = E.get("TELEGRAM_USER_ID", "").strip()

if not TG_TOKEN or not TG_USER:
    raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_USER_ID must be set in .env")

API = f"https://api.telegram.org/bot{TG_TOKEN}"


def tg_get(method: str, params: dict = None) -> dict:
    url = f"{API}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=35) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"ok": False, "error": str(e)}


def tg_send(chat_id: str, text: str, parse_mode: Optional[str] = None) -> dict:
    """Send a text message (chunked if needed; Telegram limit = 4096 chars)."""
    chunks = []
    for i in range(0, len(text), 4000):
        chunks.append(text[i:i + 4000])
    last = {}
    for ch in chunks:
        data = {"chat_id": chat_id, "text": ch}
        if parse_mode:
            data["parse_mode"] = parse_mode
        try:
            req = urllib.request.Request(f"{API}/sendMessage",
                                          data=urllib.parse.urlencode(data).encode())
            with urllib.request.urlopen(req, timeout=15) as r:
                last = json.loads(r.read())
        except Exception as e:
            last = {"ok": False, "error": str(e)}
    return last


# ─────────────────────────────────────────────────────────────────────────────
# State probes — cheap reads from local VPS filesystem + services
# ─────────────────────────────────────────────────────────────────────────────

def _sh(cmd: str, timeout: int = 10) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=HERE)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR: {e}"


def state_recorder() -> dict:
    active = _sh("systemctl is-active cs2bot.service")
    uptime = _sh("systemctl show cs2bot.service -p ActiveEnterTimestamp --value")
    files = _sh("ls /home/bot/esports/data/recordings/*.jsonl 2>/dev/null | wc -l")
    return {
        "service": active,
        "since": uptime,
        "recording_files": files,
    }


def state_balance() -> dict:
    # Prefer the risk_state.json if the bot has been running
    p = HERE / "data" / "risk_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"current_balance": float(E.get("STARTING_BALANCE", "1000.0")),
            "starting_balance": float(E.get("STARTING_BALANCE", "1000.0")),
            "today_realized_pnl": 0.0,
            "open_positions_count": 0}


def state_positions() -> list[dict]:
    """Load active positions — real if LIVE, shadow if SHADOW_TRADING."""
    db_path = HERE / "data" / "trades.db"
    if not db_path.exists():
        return []
    import sqlite3
    try:
        conn = sqlite3.connect(str(db_path))
        # Prefer live positions table if it has open rows
        rows = []
        try:
            cur = conn.execute("""
                SELECT match_id, team, fill_price, bet_size_usd, opened_ts,
                       COALESCE(take_profit_target, 0), COALESCE(stop_loss_target, 0)
                FROM positions WHERE status='open'
            """)
            for r in cur.fetchall():
                rows.append({"match_id": r[0], "team": r[1], "fill": r[2],
                             "bet": r[3], "opened_ts": r[4],
                             "tp": r[5], "sl": r[6], "source": "live"})
        except sqlite3.OperationalError:
            pass  # no positions table yet

        # If SHADOW_TRADING, also show recent shadow trades (last 24h)
        if E.get("SHADOW_TRADING", "true").lower() in ("1", "true", "yes"):
            try:
                cur = conn.execute("""
                    SELECT match_id, team, fill_price, bet_usd, ts, tp_pct, sl_pct
                    FROM shadow_trades
                    WHERE ts >= ?
                    ORDER BY ts DESC LIMIT 15
                """, (time.time() - 86400,))
                for r in cur.fetchall():
                    rows.append({"match_id": r[0], "team": r[1], "fill": r[2],
                                 "bet": r[3], "opened_ts": r[4],
                                 "tp": r[5], "sl": r[6], "source": "shadow"})
            except sqlite3.OperationalError:
                pass  # no shadow_trades table yet

        conn.close()
        return rows
    except Exception:
        return []


def state_trades_today() -> dict:
    db_path = HERE / "data" / "trades.db"
    if not db_path.exists():
        return {"trades_today": 0, "pnl_today": 0.0, "w": 0, "l": 0}
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        # Today UTC starts at midnight
        today_start = int(time.mktime(time.strptime(
            time.strftime("%Y-%m-%d", time.gmtime()) + " 00:00:00", "%Y-%m-%d %H:%M:%S")))
        cur = conn.execute("""
            SELECT COUNT(*),
                   COALESCE(SUM(pnl_usd), 0),
                   SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END)
            FROM trades WHERE exit_ts >= ?
        """, (today_start,))
        n, pnl, w, l = cur.fetchone()
        conn.close()
        return {"trades_today": n or 0, "pnl_today": pnl or 0.0,
                "w": w or 0, "l": l or 0}
    except Exception:
        return {"trades_today": 0, "pnl_today": 0.0, "w": 0, "l": 0}


def state_mode() -> str:
    """Determine current trading mode from .env + lock file."""
    if (HERE / "HARD_STOP.lock").exists():
        return "🛑 HARD_STOP"
    if E.get("SHADOW_TRADING", "true").lower() in ("1", "true", "yes"):
        return "🔵 SHADOW (no real orders)"
    if E.get("RECORD_ONLY", "false").lower() in ("1", "true", "yes"):
        return "🟡 RECORD_ONLY (no trading)"
    pause_flag = HERE / ".bot_paused"
    if pause_flag.exists():
        return "⏸️  PAUSED (existing positions still monitored)"
    tw = float(E.get("TRAINING_WHEELS_MAX_PCT", "0") or 0)
    if tw > 0:
        return f"🟢 LIVE (training wheels: {tw*100:.1f}% per trade)"
    return "🟢 LIVE (full sizing)"


def state_audit() -> str:
    """Read the latest audit snapshot if available (from daily_report run)."""
    out = _sh("python3 data/processor.py audit 2>&1 | tail -15", timeout=90)
    return out


def state_gpu() -> str:
    """Probe gaming PC GPU (optional — may not be reachable)."""
    out = _sh("ssh -o ConnectTimeout=5 -o BatchMode=yes "
              "Andre@192.168.76.196 nvidia-smi.exe --query-gpu="
              "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu "
              "--format=csv,noheader,nounits", timeout=10)
    if "ERR" in out or not out or "," not in out:
        return "🔴 gaming PC unreachable"
    try:
        util, vram_used, vram_total, power, temp = [p.strip() for p in out.split(",")]
        return (f"🟢 RTX 5090: util {util}%  vram {int(vram_used)/1024:.1f}/"
                f"{int(vram_total)/1024:.0f} GB  {float(power):.0f}W  {temp}°C")
    except Exception:
        return f"? parse: {out[:80]}"


def state_disk() -> str:
    out = _sh("df -BG / | tail -1")
    try:
        parts = out.split()
        free = int(parts[3].rstrip("G"))
        used_pct = int(parts[4].rstrip("%"))
        dot = "🟢" if free >= 20 else "🟡" if free >= 10 else "🔴"
        return f"{dot} {free} GB free ({used_pct}% used)"
    except Exception:
        return f"? {out[:60]}"


def state_timers() -> str:
    """Which systemd timers are active."""
    out = _sh("systemctl list-timers cs2bot-* --no-pager | head -20")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Handlers for each slash command
# ─────────────────────────────────────────────────────────────────────────────

def _duration_since(ts_str: str) -> str:
    """Parse 'Mon 2026-04-17 20:00:57 UTC' and return 'Xd Yh' uptime."""
    try:
        import re
        m = re.match(r"\S+\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", ts_str)
        if not m:
            return ts_str
        start = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        elapsed = time.time() - start
        d = int(elapsed // 86400)
        h = int((elapsed % 86400) // 3600)
        return f"{d}d {h}h"
    except Exception:
        return ts_str


def cmd_status(_args: str) -> str:
    rec = state_recorder()
    bal = state_balance()
    pos = state_positions()
    td = state_trades_today()
    mode = state_mode()
    lines = [
        f"🤖 Bot Status",
        f"",
        f"Mode:     {mode}",
        f"Service:  {rec['service']} · up {_duration_since(rec['since'])}",
        f"Recordings: {rec['recording_files']} files",
        f"",
        f"Balance:  ${bal['current_balance']:.2f}  "
        f"(start ${bal['starting_balance']:.2f})",
        f"Today:    {td['trades_today']} trades, "
        f"{td['w']}W/{td['l']}L, PnL ${td['pnl_today']:+.2f}",
        f"Positions: {len(pos)} open",
    ]
    if pos:
        lines.append("")
        lines.append("Open:")
        for p in pos[:5]:
            lines.append(f"  • {p['team']} @ {p['fill']:.3f}  "
                         f"bet ${p['bet']:.0f}  tp{p['tp']*100:.0f}% sl{p['sl']*100:.0f}%")
    lines.append("")
    lines.append(f"Disk: {state_disk()}")
    lines.append(f"GPU:  {state_gpu()}")
    return "\n".join(lines)


def cmd_positions(_args: str) -> str:
    pos = state_positions()
    if not pos:
        return "No open positions."
    lines = [f"{len(pos)} open:"]
    for p in pos:
        age_s = time.time() - p["opened_ts"]
        age = f"{int(age_s/60)}m" if age_s < 3600 else f"{int(age_s/3600)}h"
        lines.append(f"  {p['team']:<14} @ {p['fill']:.3f}  "
                     f"bet ${p['bet']:.0f}  {age}  tp {p['tp']*100:.0f}% / sl {p['sl']*100:.0f}%")
    return "\n".join(lines)


def cmd_pnl(args: str) -> str:
    window = (args or "today").strip().lower()
    db_path = HERE / "data" / "trades.db"
    if not db_path.exists():
        return "No trade history yet."
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    now = int(time.time())
    windows = {"today": 86400, "week": 86400 * 7, "month": 86400 * 30, "all": 10**10}
    w_secs = windows.get(window, 86400)
    cur = conn.execute("""
        SELECT COUNT(*), COALESCE(SUM(pnl_usd), 0),
               SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl_usd <= 0 THEN 1 ELSE 0 END),
               COALESCE(AVG(CASE WHEN pnl_usd > 0 THEN pnl_usd END), 0),
               COALESCE(AVG(CASE WHEN pnl_usd <= 0 THEN pnl_usd END), 0)
        FROM trades WHERE exit_ts >= ?
    """, (now - w_secs,))
    n, pnl, w, l, avg_win, avg_loss = cur.fetchone()
    conn.close()
    if not n:
        return f"No trades in last {window}."
    win_rate = w / n if n else 0
    return (f"PnL — last {window}\n\n"
            f"Trades:     {n}\n"
            f"Wins:       {w} · Losses: {l}  ({win_rate*100:.0f}% WR)\n"
            f"Net PnL:    ${pnl:+.2f}\n"
            f"Avg win:    ${avg_win:+.2f}\n"
            f"Avg loss:   ${avg_loss:+.2f}")


def cmd_balance(_args: str) -> str:
    bal = state_balance()
    dd = (bal["starting_balance"] - bal["current_balance"]) / bal["starting_balance"]
    return (f"Balance:  ${bal['current_balance']:.2f}\n"
            f"Start:    ${bal['starting_balance']:.2f}\n"
            f"Drawdown: {dd*100:+.1f}%\n"
            f"Today:    ${bal['today_realized_pnl']:+.2f}\n"
            f"Open pos: {bal['open_positions_count']}")


def cmd_pause(_args: str) -> str:
    flag = HERE / ".bot_paused"
    flag.write_text(f"paused at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return "⏸️  Paused. New trades blocked. Existing positions keep their TP/SL monitors."


def cmd_resume(_args: str) -> str:
    flag = HERE / ".bot_paused"
    if flag.exists():
        flag.unlink()
        return "▶️  Resumed. New trades allowed."
    return "Bot was not paused."


def cmd_shadow(_args: str) -> str:
    _set_env_flag("SHADOW_TRADING", "true")
    return ("🔵 Shadow mode ON. Bot will decide + log trades but send ZERO orders to "
            "Polymarket. Flip back with /live.")


def cmd_live(args: str) -> str:
    if (args or "").strip() != "CONFIRM":
        return ("⚠️ Going LIVE moves real money. If you're sure, send:\n\n"
                "/live CONFIRM\n\n"
                "This disables SHADOW_TRADING. Training-wheels cap "
                f"({E.get('TRAINING_WHEELS_MAX_PCT', '?')}%) still applies.")
    _set_env_flag("SHADOW_TRADING", "false")
    return ("🟢 LIVE mode ON. Bot will place real orders starting on next signal.\n"
            f"Training wheels: {E.get('TRAINING_WHEELS_MAX_PCT', '?')}% per trade\n"
            "Kill-switch: /hardstop  or  /shadow")


def cmd_hardstop(_args: str) -> str:
    lock = HERE / "HARD_STOP.lock"
    lock.write_text(f"manual hardstop via Telegram at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    # Also restart the bot to make the change take effect
    _sh("sudo -n systemctl restart cs2bot.service")
    return ("🛑 HARD_STOP.lock written. Bot refuses to trade until the file is deleted.\n"
            "To re-enable: ssh to VPS and delete HARD_STOP.lock, then /resume.")


def cmd_config(_args: str) -> str:
    keys = ["PER_TRADE_BASE_PCT", "PER_TRADE_STRONG_PCT", "PER_TRADE_MAX_PCT",
            "CONFIDENCE_FLOOR", "TAKE_PROFIT_PCT", "STOP_LOSS_PCT",
            "MAX_CONCURRENT_POSITIONS", "DAILY_LOSS_STOP_PCT", "HARD_STOP_PCT",
            "COOLDOWN_AFTER_N_LOSSES", "MIN_LIQUIDITY_USD",
            "TRAINING_WHEELS_MAX_PCT", "SHADOW_TRADING", "EDGE_CLAUDE_MODEL"]
    lines = ["⚙️  Risk Config"]
    for k in keys:
        v = E.get(k, "—")
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def cmd_gpu(_args: str) -> str:
    return f"GPU: {state_gpu()}"


def cmd_data(_args: str) -> str:
    out = state_audit()
    # Return only the summary + verdict section
    return f"Dataset audit:\n\n{out[:2000]}"


def cmd_tail(args: str) -> str:
    try:
        n = int((args or "50").strip())
    except ValueError:
        n = 50
    n = max(5, min(200, n))
    out = _sh(f"tail -n {n} _logs/cs2bot.stderr.log 2>/dev/null || echo '(no log)'")
    return f"Last {n} lines:\n\n{out[:3500]}"


def cmd_health(_args: str) -> str:
    lines = ["🧰 Systemd timer health"]
    lines.append(state_timers())
    return "\n".join(lines[:20])


def cmd_help(_args: str) -> str:
    return (
        "Commands:\n"
        "  /status        bot state snapshot\n"
        "  /positions     open positions\n"
        "  /pnl [window]  today|week|month|all\n"
        "  /balance       wallet + drawdown\n"
        "  /pause  /resume\n"
        "  /shadow        go to shadow mode\n"
        "  /live CONFIRM  go live with real orders\n"
        "  /hardstop      permanent kill + lock\n"
        "  /config        current risk config\n"
        "  /gpu           gaming PC state\n"
        "  /data          dataset audit\n"
        "  /tail N        last N log lines\n"
        "  /health        systemd timer health\n"
        "  /help\n\n"
        "Or type anything in plain English — I'll think about it against the "
        "current bot state.")


COMMANDS = {
    "/status":    cmd_status,
    "/positions": cmd_positions,
    "/pnl":       cmd_pnl,
    "/balance":   cmd_balance,
    "/pause":     cmd_pause,
    "/resume":    cmd_resume,
    "/shadow":    cmd_shadow,
    "/live":      cmd_live,
    "/hardstop":  cmd_hardstop,
    "/config":    cmd_config,
    "/gpu":       cmd_gpu,
    "/data":      cmd_data,
    "/tail":      cmd_tail,
    "/health":    cmd_health,
    "/help":      cmd_help,
    "/start":     cmd_help,
}


def _set_env_flag(key: str, value: str) -> None:
    """Update a single key in .env — preserves other lines."""
    env_path = HERE / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n")
    os.environ[key] = value
    E[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Free-text chat — relay to Claude Haiku with bot state as context
# ─────────────────────────────────────────────────────────────────────────────

def claude_chat(user_message: str) -> str:
    """Route a free-text question to Claude with the bot's current state."""
    api_key = E.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return "ANTHROPIC_API_KEY not set — can't do free-text chat."

    # Build bot-state context (compact)
    try:
        rec = state_recorder()
        bal = state_balance()
        pos = state_positions()
        td = state_trades_today()
        mode = state_mode()
        ctx = (
            f"BOT STATE:\n"
            f"  mode: {mode}\n"
            f"  service: {rec['service']}\n"
            f"  balance: ${bal['current_balance']:.2f}  (start ${bal['starting_balance']:.2f})\n"
            f"  today: {td['trades_today']} trades {td['w']}W/{td['l']}L PnL ${td['pnl_today']:+.2f}\n"
            f"  open positions: {len(pos)}\n"
            f"  recordings: {rec['recording_files']} files\n"
        )
        if pos:
            ctx += "  positions:\n"
            for p in pos[:5]:
                ctx += (f"    - {p['team']} @ {p['fill']:.3f}  bet ${p['bet']:.0f}\n")
    except Exception as e:
        ctx = f"(state probe failed: {e})"

    system = (
        "You're the admin interface of a CS2 Polymarket trading bot. "
        "The user is the bot owner. Answer concisely based on the bot state below. "
        "If they ask about a decision or trade, read the relevant fields from the "
        "state context. If you don't have the info, say so directly. Keep "
        "responses under 800 chars for Telegram delivery."
    )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=E.get("EDGE_CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": f"{ctx}\n\nUSER: {user_message}"}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as e:
        return f"Claude error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Long-polling loop
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print(f"[admin] starting; authorized user = {TG_USER}")
    offset = 0
    # Send a "hello" on startup so you know the admin is up
    tg_send(TG_USER, "🤖 Telegram admin is online. /help for commands.")

    while True:
        try:
            r = tg_get("getUpdates", {"offset": offset, "timeout": 30})
            if not r.get("ok"):
                print(f"[admin] getUpdates failed: {r}")
                time.sleep(5)
                continue
            for upd in r.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                chat_id = str(msg["chat"]["id"])
                user_id = str(msg.get("from", {}).get("id", ""))
                text = (msg.get("text") or "").strip()
                if user_id != TG_USER:
                    # Ignore non-authorized users silently
                    continue
                if not text:
                    continue

                print(f"[admin] <- {text[:80]}")

                # Dispatch
                if text.startswith("/"):
                    parts = text.split(maxsplit=1)
                    cmd = parts[0].lower()
                    # Strip @botname suffix if present
                    cmd = cmd.split("@")[0]
                    args = parts[1] if len(parts) > 1 else ""
                    handler = COMMANDS.get(cmd)
                    if handler:
                        reply = handler(args)
                    else:
                        reply = f"Unknown command {cmd}. /help"
                else:
                    reply = claude_chat(text)

                tg_send(chat_id, reply)
                print(f"[admin] -> {reply[:80]}")
        except KeyboardInterrupt:
            print("[admin] stopping")
            return
        except Exception as e:
            print(f"[admin] loop error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()
