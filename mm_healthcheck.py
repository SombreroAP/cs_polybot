#!/usr/bin/env python3
"""MM shadow-run health check. Prints a compact one-block status report.
Called by the overnight monitor every ~5 min. Exit 0 = healthy, 2 = degraded.
"""
import sqlite3
import subprocess
import time
import os
import json

DB = "/home/bot/esports/data/mm.db"
STDERR_LOG = "/home/bot/esports/_logs/cs2bot.stderr.log"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception as e:
        return f"ERR:{e}"


def main():
    now = time.time()
    issues = []

    # 1. process alive + uptime
    active = sh("systemctl is-active cs2bot.service")
    uptime_raw = sh("systemctl show cs2bot.service -p ActiveEnterTimestampMonotonic --value")
    try:
        mono = int(uptime_raw) / 1e6
        with open("/proc/uptime") as f:
            sys_up = float(f.read().split()[0])
        svc_up_min = (sys_up - mono) / 60
    except Exception:
        svc_up_min = -1
    if active != "active":
        issues.append(f"SERVICE {active}")

    # 2. DB activity
    d5 = d30 = f5 = f60 = f_life = 0
    cash24 = cash_life = 0.0
    actions = {}
    try:
        c = sqlite3.connect(DB, timeout=8)
        c.execute("PRAGMA busy_timeout=8000")
        d5 = c.execute("SELECT COUNT(*) FROM mm_decisions WHERE ts>?", (now-300,)).fetchone()[0]
        d30 = c.execute("SELECT COUNT(*) FROM mm_decisions WHERE ts>?", (now-1800,)).fetchone()[0]
        f5 = c.execute("SELECT COUNT(*) FROM mm_fills WHERE ts>?", (now-300,)).fetchone()[0]
        f60 = c.execute("SELECT COUNT(*) FROM mm_fills WHERE ts>?", (now-3600,)).fetchone()[0]
        f_life = c.execute("SELECT COUNT(*) FROM mm_fills").fetchone()[0]
        # sim cash pnl: ask_fill = +price, bid_fill = -price
        cash24 = c.execute("SELECT COALESCE(SUM(CASE WHEN side='ask_fill' THEN price ELSE -price END),0) "
                           "FROM mm_fills WHERE ts>?", (now-86400,)).fetchone()[0]
        cash_life = c.execute("SELECT COALESCE(SUM(CASE WHEN side='ask_fill' THEN price ELSE -price END),0) "
                              "FROM mm_fills").fetchone()[0]
        for a, n in c.execute("SELECT action,COUNT(*) FROM mm_decisions WHERE ts>? GROUP BY action", (now-1800,)):
            actions[a] = n
        c.close()
    except Exception as e:
        issues.append(f"DB:{e}")

    # 3. is it actually trading? (decisions flowing)
    if svc_up_min > 6 and d5 == 0:
        issues.append("NO DECISIONS in 5min")

    # 4. recent errors in log
    errs = sh(f"tail -400 {STDERR_LOG} 2>/dev/null | grep -cE 'Traceback|CRITICAL|\\[WATCHDOG\\]'")
    try:
        nerr = int(errs)
    except Exception:
        nerr = 0
    if nerr > 0:
        issues.append(f"{nerr} errors/tracebacks in recent log")

    # 5. MM mode + queue via API
    qd = qdrop = -1
    live_mode = "?"
    api = sh("curl -s -m 6 http://localhost:8082/api/mm")
    try:
        j = json.loads(api)
        st = j.get("stats", {})
        qd = st.get("queue_depth", -1)
        qdrop = st.get("queue_dropped_lifetime", -1)
        live_mode = "LIVE" if st.get("live_trading") else "SHADOW"
    except Exception:
        issues.append("API unreachable")
    if isinstance(qd, int) and qd > 10000:
        issues.append(f"queue backing up ({qd})")

    status = "HEALTHY" if not issues else "DEGRADED"
    print(f"=== MM HEALTH [{status}] {time.strftime('%H:%M:%S UTC', time.gmtime())} ===")
    print(f"  service: {active}  up={svc_up_min:.0f}min  mode={live_mode}")
    print(f"  decisions: 5min={d5}  30min={d30}   actions(30m)={actions}")
    print(f"  fills: 5min={f5}  60min={f60}  lifetime={f_life}")
    print(f"  sim PnL: 24h=${cash24:+.3f}  lifetime=${cash_life:+.3f}")
    print(f"  queue: depth={qd}  dropped_lifetime={qdrop}")
    if issues:
        print(f"  ISSUES: {'; '.join(issues)}")
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
