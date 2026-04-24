#!/usr/bin/env python3
"""
Daily / weekly report → Telegram.

Runs on the VPS via a systemd timer. Gathers:
  - Recorder uptime + file count + matches/day
  - Data audit (training-readiness verdicts)
  - Recent commits on main (what the assistant has been doing)
  - Evaluator runs in the last 24h
  - Gates progress + ETA

Modes:
  python daily_report.py            # daily — sent every morning
  python daily_report.py --weekly   # weekly deep audit — sent Sundays

Reads TELEGRAM_BOT_TOKEN + TELEGRAM_USER_ID from .env.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(HERE / ".env", override=False)
except ImportError:
    # Minimal fallback loader
    env_file = HERE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_USER = os.environ.get("TELEGRAM_USER_ID", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Data gathering
# ─────────────────────────────────────────────────────────────────────────────

def _sh(cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=HERE)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR: {e}"


def recorder_health() -> dict:
    """Service status + uptime + recent recording activity."""
    active = _sh("systemctl is-active cs2bot.service", 5).strip()
    started = _sh("systemctl show cs2bot.service -p ActiveEnterTimestamp --value",
                  5).strip()
    rec_dir = HERE / "data" / "recordings"
    file_count = 0
    total_mb = 0.0
    last_24h_count = 0
    now = time.time()
    if rec_dir.exists():
        for p in rec_dir.iterdir():
            if p.suffix != ".jsonl":
                continue
            file_count += 1
            try:
                s = p.stat()
                total_mb += s.st_size / 1e6
                if now - s.st_mtime < 86400:
                    last_24h_count += 1
            except Exception:
                pass
    uptime_s = 0
    if started and started != "n/a":
        try:
            # Best effort: parse "Mon 2026-04-17 20:00:57 UTC"
            import email.utils
            # systemd's format is not RFC 2822; use loose parse
            from datetime import datetime as dt
            fmt_iso = re.match(r"\S+\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", started)
            if fmt_iso:
                start_dt = dt.strptime(fmt_iso.group(1), "%Y-%m-%d %H:%M:%S")
                uptime_s = now - start_dt.timestamp()
        except Exception:
            pass
    # Disk metrics — surface capacity so we catch "disk full" 20 days from now
    disk_free_gb = 0.0
    disk_used_pct = 0
    try:
        disk_raw = _sh("df -BG / | tail -1", 5)
        # "/dev/... 97G 27G 67G 29% /"
        parts = disk_raw.split()
        if len(parts) >= 5:
            disk_free_gb = float(parts[3].rstrip("G"))
            disk_used_pct = int(parts[4].rstrip("%"))
    except Exception:
        pass

    return {
        "active": active,
        "uptime_hours": round(uptime_s / 3600, 1),
        "uptime_days": round(uptime_s / 86400, 2),
        "total_files": file_count,
        "total_mb": round(total_mb, 1),
        "new_last_24h": last_24h_count,
        "disk_free_gb": disk_free_gb,
        "disk_used_pct": disk_used_pct,
    }


def run_audit() -> dict:
    """Run processor.py audit + parse verdicts."""
    out = _sh("python3 data/processor.py build-sft 2>&1 && "
              "python3 data/processor.py audit 2>&1", 180)
    # Parse the verdict block
    verdicts = []
    for line in out.splitlines():
        m = re.match(r"\s*([✅⚠️❌])\s+(\w+)\s+(.+)", line)
        if m:
            verdicts.append({"status": m.group(1),
                             "check": m.group(2),
                             "note": m.group(3).strip()})
    # Extract key numbers
    def _grab(pat):
        m = re.search(pat, out)
        return m.group(1) if m else None
    return {
        "total_examples": _grab(r"total_examples\s+(\d+)"),
        "unique_matches": _grab(r"unique_matches\s+(\d+)"),
        "buy_labels": _grab(r"buy_labels\s+(\d+)"),
        "skip_labels": _grab(r"skip_labels\s+(\d+)"),
        "temporal_days": _grab(r"temporal_range_days\s+([\d.]+)"),
        "verdicts": verdicts,
        "raw_tail": out.splitlines()[-25:],
    }


def gates_eta(audit: dict, recorder: dict) -> dict:
    """Days to each training gate based on current rate."""
    try:
        matches = int(audit.get("unique_matches") or 0)
    except Exception:
        matches = 0
    days_recording = recorder.get("uptime_days", 0) or 0.01
    # Conservative: assume only live VPS matches count (~5-10/day), not historical
    # Our measured rate is ~5 training-ready matches/day after the Test Data backfill
    rate = max(0.5, recorder.get("new_last_24h", 0) or 5)   # fallback to 5/day
    # But filter to TRAINING-ready matches per day. File rate != training rate.
    # For simplicity: training rate ≈ new_files_24h × 0.3 (30% pass audit)
    training_rate_per_day = rate * 0.3
    gate_volume = 150
    gate_temporal = 30

    days_to_volume = max(0, (gate_volume - matches) / max(1, training_rate_per_day))
    days_to_temporal = max(0, gate_temporal - days_recording)
    days_to_both = max(days_to_volume, days_to_temporal)
    return {
        "current_matches": matches,
        "current_days": round(days_recording, 1),
        "training_rate_per_day_est": round(training_rate_per_day, 1),
        "days_to_volume": round(days_to_volume, 1),
        "days_to_temporal": round(days_to_temporal, 1),
        "days_to_both": round(days_to_both, 1),
    }


def recent_commits(max_n: int = 6) -> list[dict]:
    """Last N git commits (what the assistant has done lately)."""
    out = _sh(
        'git -C "' + str(HERE) + '" log --since="1 week ago" '
        '--pretty=format:"%h|%ad|%s" --date=format:"%m-%d %H:%M" | head -' + str(max_n),
        10,
    )
    commits = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            commits.append({"hash": parts[0], "date": parts[1], "subject": parts[2]})
    return commits


def recent_evals() -> list[dict]:
    """Eval runs in the last 48h (if any)."""
    d = HERE / "data" / "evals"
    if not d.exists():
        return []
    out = []
    now = time.time()
    for p in sorted(d.glob("*.jsonl"), key=lambda x: -x.stat().st_mtime)[:6]:
        if now - p.stat().st_mtime > 48 * 3600:
            break
        try:
            lines = p.read_text().strip().splitlines()
            total = len(lines)
            correct = 0
            for line in lines:
                try:
                    d_ = json.loads(line)
                    if d_.get("pred", {}).get("action") == d_.get("label"):
                        correct += 1
                except Exception:
                    continue
            out.append({
                "model": p.stem.split("_")[0][:25],
                "n": total,
                "acc": round(correct / max(1, total), 3),
                "hours_ago": round((now - p.stat().st_mtime) / 3600, 1),
            })
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Message assembly
# ─────────────────────────────────────────────────────────────────────────────

def format_daily(rec: dict, audit: dict, gates: dict,
                 commits: list, evals: list) -> str:
    lines = []
    day = datetime.now(timezone.utc).strftime("%a %Y-%m-%d")
    lines.append(f"📊 CS2 Bot daily — {day}")
    lines.append("")

    # Recorder
    dot = "🟢" if rec["active"] == "active" else "🔴"
    lines.append(f"{dot} Recorder: {rec['active']} · "
                 f"up {rec['uptime_days']}d · "
                 f"{rec['total_files']} files · "
                 f"+{rec['new_last_24h']} in last 24h")

    # Disk capacity — alert if low
    disk_dot = "🟢"
    if rec.get("disk_free_gb", 0) < 10:
        disk_dot = "🔴"
    elif rec.get("disk_free_gb", 0) < 20:
        disk_dot = "🟡"
    # Runway: avg 1 GB/day growth, project days remaining
    runway_days = rec.get("disk_free_gb", 0) / 1.0  # conservative: 1 GB/day
    lines.append(f"{disk_dot} Disk: {rec.get('disk_free_gb', 0):.0f} GB free "
                 f"({rec.get('disk_used_pct', 0)}% used) · "
                 f"~{runway_days:.0f}-day runway at 1 GB/day")

    # Training gate progress
    lines.append("")
    lines.append(f"🎯 Training gate: "
                 f"{gates['current_matches']}/150 matches "
                 f"({gates['current_days']}/30 days)")
    lines.append(f"   ETA to both ✅: ~{gates['days_to_both']:.0f} days "
                 f"(@ {gates['training_rate_per_day_est']:.1f} matches/day)")

    # Audit verdicts (top 4 only, compact)
    if audit.get("verdicts"):
        lines.append("")
        lines.append("🧪 Audit:")
        for v in audit["verdicts"][:6]:
            lines.append(f"   {v['status']} {v['check']}: {v['note'][:55]}")

    # Recent commits
    if commits:
        lines.append("")
        lines.append("🔨 Recent changes:")
        for c in commits[:4]:
            subj = c["subject"][:60]
            lines.append(f"   • {c['date']}  {subj}")

    # Recent evals
    if evals:
        lines.append("")
        lines.append("📈 Recent eval runs:")
        for e in evals[:3]:
            lines.append(f"   {e['model']} · n={e['n']} · acc={e['acc']:.1%} "
                         f"({e['hours_ago']}h ago)")

    lines.append("")
    lines.append(f"Dataset: {audit.get('total_examples', '?')} ex · "
                 f"{audit.get('buy_labels', '?')} buy / "
                 f"{audit.get('skip_labels', '?')} skip")
    return "\n".join(lines)


def format_weekly(rec: dict, audit: dict, gates: dict,
                  commits: list, evals: list) -> str:
    # Same as daily but more commits shown + audit tail
    msg = format_daily(rec, audit, gates, commits, evals)
    msg += "\n\n🧾 Weekly audit raw tail:\n" + "\n".join(
        f"   {l[:68]}" for l in audit.get("raw_tail", [])[-12:])
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# Telegram delivery
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_USER:
        print("!! TELEGRAM_BOT_TOKEN or TELEGRAM_USER_ID missing — cannot send")
        return False
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_USER,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    req = urllib.request.Request(url, data=data)
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        if resp.get("ok"):
            return True
        print(f"!! telegram api returned: {resp.get('description')}")
        return False
    except Exception as e:
        print(f"!! telegram send failed: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--stdout", action="store_true",
                    help="print only, don't send to Telegram")
    args = ap.parse_args()

    print(f"[report] gathering @ {datetime.now(timezone.utc).isoformat()}")
    rec = recorder_health()
    print(f"[report] recorder: {rec}")
    audit = run_audit()
    print(f"[report] audit: matches={audit.get('unique_matches')} "
          f"examples={audit.get('total_examples')}")
    gates = gates_eta(audit, rec)
    commits = recent_commits(max_n=8 if args.weekly else 4)
    evals = recent_evals()

    msg = format_weekly(rec, audit, gates, commits, evals) if args.weekly \
          else format_daily(rec, audit, gates, commits, evals)

    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)

    if not args.stdout:
        if send_telegram(msg):
            print(f"\n[report] ✅ sent to Telegram")
            return 0
        print("\n[report] ❌ Telegram send failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
