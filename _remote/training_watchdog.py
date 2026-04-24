#!/usr/bin/env python3
"""
Training watchdog — runs on the Mac, polls the Qwen training state every 5 min.

Responsibilities:
  1. Detect if training is alive (python process on gaming PC) vs dead.
  2. Parse log for progress, loss, errors.
  3. If TRAIN OK -> run eval + Telegram user.
  4. If CRASHED -> Telegram user with diagnosis + next-step recommendation.
  5. Persist a health log we can inspect after the fact.

Designed to keep running even if I (the assistant) am not in session.
"""
import json, os, re, subprocess, time, urllib.request, urllib.parse
from pathlib import Path

REPO = Path("/Users/andrew/Documents/Claude/Projects/Trading/esports")
VPS = "Andre@192.168.76.196"
LOG_LOCAL = Path("/tmp/qwen36_train.log")
HEALTH_LOG = Path("/tmp/training_watchdog.log")
INTERVAL_S = 300     # 5 minutes

def _env() -> dict:
    out = {}
    for line in (REPO / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out

E = _env()
TG_TOKEN = E["TELEGRAM_BOT_TOKEN"]
TG_CHAT  = E["TELEGRAM_USER_ID"]

def telegram(text: str):
    try:
        data = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", data=data)
        urllib.request.urlopen(req, timeout=15).read()
        log("[tg] sent")
    except Exception as e:
        log(f"[tg] FAILED: {e}")

def log(s: str):
    msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}"
    print(msg)
    with open(HEALTH_LOG, "a") as f:
        f.write(msg + "\n")

def ssh_capture(cmd: str, timeout: int = 15) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", VPS, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"__ERR__ {e}"

def check_state() -> dict:
    """Return a snapshot of current training state."""
    state = {"ts": time.time()}

    # 1. log file freshness + content
    if LOG_LOCAL.exists():
        text = LOG_LOCAL.read_text(errors="replace")
        state["log_size"] = len(text)
        state["log_age_s"] = time.time() - LOG_LOCAL.stat().st_mtime
        state["log_tail"] = text[-3000:]
    else:
        state["log_size"] = 0

    # 2. python process on gaming PC?
    r = ssh_capture('tasklist /FI "IMAGENAME eq python.exe" /NH')
    state["python_running"] = "python.exe" in r and "No tasks" not in r

    # 3. GPU state
    r = ssh_capture(
        "nvidia-smi.exe --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits")
    if not r.startswith("__ERR__"):
        try:
            parts = [p.strip() for p in r.split(",")]
            state["gpu_util"] = int(parts[0])
            state["vram_used_mib"] = int(parts[1])
            state["vram_total_mib"] = int(parts[2])
            state["power_w"] = float(parts[3])
            state["temp_c"] = int(parts[4])
        except Exception:
            pass

    # 4. parse log for progress + completion
    tail = state.get("log_tail", "")
    # Losses
    losses = re.findall(r"'loss': '([-\d.eE+]+)'[^}]*'epoch': '([\d.]+)'", tail)
    if losses:
        state["last_loss"] = float(losses[-1][0])
        state["last_epoch"] = float(losses[-1][1])
    # Step progress
    step_matches = re.findall(r"(\d+)/(\d+)\s*\[\s*([\d:]+)\s*<\s*([\d:]+|\?)[^,]*,\s*([\d.]+)s?/it", tail)
    if step_matches:
        r = step_matches[-1]
        state["step"] = int(r[0])
        state["total_steps"] = int(r[1])
        state["step_pct"] = round(100 * state["step"] / state["total_steps"], 1)
        state["elapsed"] = r[2]
        state["eta"] = r[3]
        state["sec_per_step"] = float(r[4]) if r[4] else 0
    # Completion / failure
    state["completed"] = "TRAIN OK" in tail
    state["failed"] = any(s in tail for s in
                          ["Traceback", "CUDA out of memory", "access violation",
                           "Faulting application", "RuntimeError"])
    return state

def format_telegram(state: dict, reason: str) -> str:
    lines = [f"Qwen 3.6 training — {reason}", ""]
    if "last_loss" in state:
        lines.append(f"Last loss:  {state['last_loss']:.4f}  (epoch {state.get('last_epoch','?')})")
    if "step" in state:
        lines.append(f"Step:       {state['step']}/{state['total_steps']}  ({state['step_pct']}%)")
        lines.append(f"Elapsed:    {state.get('elapsed','?')}  ETA {state.get('eta','?')}")
    if "gpu_util" in state:
        lines.append(f"GPU:        {state['gpu_util']}% util  "
                     f"{state['vram_used_mib']}/{state['vram_total_mib']} MiB  "
                     f"{state.get('power_w',0):.0f}W {state.get('temp_c',0)}C")
    lines.append(f"Python:     {'alive' if state.get('python_running') else 'NOT RUNNING'}")
    return "\n".join(lines)

def main():
    log("watchdog starting — interval 5 min")
    notified_completion = False
    notified_failure = False
    prev_step = -1
    stagnant_checks = 0

    while True:
        state = check_state()

        # Summarize
        summary = f"py={'yes' if state.get('python_running') else 'NO'}"
        if "step" in state:
            summary += f" step {state['step']}/{state['total_steps']}"
        if "last_loss" in state:
            summary += f" loss {state['last_loss']:.3f}"
        if "gpu_util" in state:
            summary += f" gpu {state['gpu_util']}% vram {state.get('vram_used_mib',0)/1024:.1f}G"
        if state.get("completed"): summary += " COMPLETED"
        if state.get("failed"):    summary += " FAILED"
        log(summary)

        # Detect completion
        if state.get("completed") and not notified_completion:
            telegram(format_telegram(state, "TRAINING COMPLETE — running eval next"))
            notified_completion = True
            # Kick off eval automatically
            log("completion detected — firing eval subprocess")
            subprocess.Popen(
                ["ssh", VPS,
                 "cd /d C:\\training && C:\\training\\venv_smoke\\Scripts\\python.exe -u eval_trained.py"],
                stdout=open("/tmp/qwen36_eval.log", "w"),
                stderr=subprocess.STDOUT,
            )

        # Detect failure
        elif state.get("failed") and not notified_failure:
            telegram(format_telegram(state, "CRASHED — see /tmp/qwen36_train.log"))
            notified_failure = True

        # Detect silent death — python process dead but log wasn't completed/failed
        elif not state.get("python_running") and not state.get("completed"):
            log_age = state.get("log_age_s", 0)
            if log_age > 120:  # 2 min stale
                telegram(format_telegram(state,
                         f"Python died silently (log stale {log_age:.0f}s). Check gaming PC."))
                notified_failure = True

        # Detect stalled progress
        cur_step = state.get("step", 0)
        if cur_step == prev_step and state.get("python_running"):
            stagnant_checks += 1
            if stagnant_checks == 3:   # 15 min of no step progress
                telegram("Qwen 3.6 training — stalled: 15 min no step progress. Still loading weights? GPU idle?")
        else:
            stagnant_checks = 0
            prev_step = cur_step

        time.sleep(INTERVAL_S)

if __name__ == "__main__":
    main()
