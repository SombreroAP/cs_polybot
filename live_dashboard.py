#!/usr/bin/env python3
"""
Live dashboard — one screen showing:
  - Gaming PC GPU + training run state
  - VPS recorder status + latest matches
  - Local dataset audit + evaluator results

Opens in Chrome at http://localhost:8090/. Auto-refreshes in the browser.
The server scrapes fresh data every 3s via SSH / local file reads.

Usage:
    python live_dashboard.py
    # or --no-open to skip the Chrome launch
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = Path(__file__).resolve().parent

GAMING_PC = "Andre@192.168.10.13"
VPS = "bot@85.137.174.57"
OLLAMA_URL = "http://192.168.10.13:11434"

# In-memory snapshot written by the poller thread, read by HTTP handler
SNAPSHOT: dict = {
    "ts": 0,
    "gpu": {}, "train": {}, "vps": {}, "dataset": {}, "evals": [],
    "errors": [],
}
SNAPSHOT_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Pollers — gather fresh data every N seconds
# ─────────────────────────────────────────────────────────────────────────────

def ssh(host: str, cmd: str, timeout: int = 8) -> str:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=4", "-o", "BatchMode=yes", host, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"__ERR__ {e}"


def poll_gpu() -> dict:
    out = ssh(GAMING_PC,
              "nvidia-smi.exe --query-gpu=utilization.gpu,memory.used,memory.total,"
              "power.draw,temperature.gpu --format=csv,noheader,nounits")
    if out.startswith("__ERR__"):
        return {"error": out[9:], "online": False}
    parts = [p.strip() for p in out.split(",")]
    if len(parts) < 5:
        return {"error": f"parse: {out[:80]}", "online": False}
    try:
        return {
            "online": True,
            "util_pct": int(parts[0]),
            "vram_used_mib": int(parts[1]),
            "vram_total_mib": int(parts[2]),
            "power_w": float(parts[3]),
            "temp_c": int(parts[4]),
        }
    except Exception as e:
        return {"error": str(e), "online": False}


def poll_training() -> dict:
    """Parse training log. Prefer the Mac-side live log (fresher + no SSH cost)."""
    log_file = "/tmp/real_train_live.log"
    tail = ""
    if Path(log_file).exists():
        try:
            with open(log_file) as f:
                tail = "".join(f.readlines()[-60:])
        except Exception as e:
            tail = f"__ERR__ {e}"
    if not tail:
        # Fallback: check remote log file
        log_file = "C:\\training\\real_train.log"
        tail = ssh(GAMING_PC, f'powershell -Command "Get-Content \'{log_file}\' -Tail 50"', timeout=5)
    if tail.startswith("__ERR__"):
        return {"error": tail[9:], "log_file": log_file}
    log = log_file

    # Extract latest loss row — match trl's dict-like log
    loss_rows = re.findall(r"\{'loss': '([-\d.eE+]+)'[^}]*'epoch': '([\d.]+)'[^}]*\}", tail)
    # Step progress — "5/309 [00:17<15:35, 3.08s/it]"  (last one wins)
    step_rows = re.findall(r"(\d+)/(\d+)\s*\[\s*([\d:]+)\s*<\s*([\d:]+|\?)[^,]*,\s*([\d.]+)s?/it",
                           tail)

    latest_loss = latest_epoch = None
    if loss_rows:
        latest_loss = float(loss_rows[-1][0])
        latest_epoch = float(loss_rows[-1][1])

    # Latest step progress
    step_info = {}
    if step_rows:
        r = step_rows[-1]
        step_info = {
            "current": int(r[0]),
            "total": int(r[1]),
            "elapsed": r[2],
            "eta": r[3],
            "sec_per_step": float(r[4]) if r[4] else 0,
        }
        step_info["pct"] = round(100 * step_info["current"] / max(1, step_info["total"]), 1)

    # Detect completion
    completed = "TRAIN OK" in tail or "train_runtime" in tail
    failed = "Traceback" in tail or "CUDA out of memory" in tail

    # Detect phase
    phase = "idle"
    if "Loading weights" in tail and not step_rows:
        phase = "loading base model"
    elif "Tokenizing" in tail and not step_rows:
        phase = "tokenizing dataset"
    elif step_rows and not completed:
        phase = "training"
    elif "Preparing safetensor" in tail:
        phase = "merging + saving"
    elif completed:
        phase = "done"
    elif failed:
        phase = "failed"

    return {
        "log_file": log,
        "latest_loss": latest_loss,
        "latest_epoch": latest_epoch,
        "loss_history": [float(r[0]) for r in loss_rows[-20:]],
        "epoch_history": [float(r[1]) for r in loss_rows[-20:]],
        "step": step_info,
        "phase": phase,
        "completed": completed,
        "failed": failed,
        "tail": [l for l in tail.split("\n")[-15:] if l.strip()],
    }


def poll_vps() -> dict:
    status = ssh(VPS, "systemctl is-active cs2bot.service; "
                      "systemctl show cs2bot.service -p ActiveEnterTimestamp --value; "
                      "ls /home/bot/esports/data/recordings/ | wc -l; "
                      "du -sh /home/bot/esports/data/recordings/")
    if status.startswith("__ERR__"):
        return {"error": status[9:], "online": False}
    lines = [l for l in status.split("\n") if l.strip()]
    out = {"online": True, "raw": lines}
    if len(lines) >= 4:
        out["active"] = lines[0].strip()
        out["started_at"] = lines[1].strip()
        try:
            out["recording_count"] = int(lines[2].strip())
        except Exception:
            pass
        out["recordings_size"] = lines[3].split()[0] if lines[3].split() else "?"
    return out


def poll_dataset() -> dict:
    """Local file counts + sizes."""
    def _size_mb(p: Path) -> float:
        if not p.exists():
            return 0.0
        if p.is_file():
            return p.stat().st_size / 1e6
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

    sft = HERE / "data" / "training" / "sft.jsonl"
    sft_lines = 0
    if sft.exists():
        with open(sft) as f:
            for _ in f:
                sft_lines += 1
    return {
        "recordings_files": len(list((HERE / "data" / "recordings").glob("*.jsonl"))),
        "recordings_size_mb": round(_size_mb(HERE / "data" / "recordings"), 1),
        "processed_files": len(list((HERE / "data" / "processed").glob("*.jsonl"))),
        "sft_examples": sft_lines,
        "sft_size_mb": round(_size_mb(sft), 1),
    }


def poll_evals() -> list[dict]:
    """Latest evaluator results (if any)."""
    evals_dir = HERE / "data" / "evals"
    if not evals_dir.exists():
        return []
    out = []
    for p in sorted(evals_dir.glob("*.jsonl"), key=lambda x: -x.stat().st_mtime)[:5]:
        label = p.stem
        try:
            lines = p.read_text().strip().split("\n")
            n = len(lines)
            correct = 0
            for line in lines:
                try:
                    d = json.loads(line)
                    if d.get("pred", {}).get("action") == d.get("label"):
                        correct += 1
                except Exception:
                    continue
            out.append({
                "label": label,
                "examples": n,
                "accuracy": round(correct / max(1, n), 3),
                "ts": int(p.stat().st_mtime),
            })
        except Exception:
            continue
    return out


def poll_everything() -> None:
    while True:
        try:
            snap = {
                "ts": int(time.time()),
                "gpu": poll_gpu(),
                "train": poll_training(),
                "vps": poll_vps(),
                "dataset": poll_dataset(),
                "evals": poll_evals(),
            }
            with SNAPSHOT_LOCK:
                SNAPSHOT.update(snap)
        except Exception as e:
            with SNAPSHOT_LOCK:
                SNAPSHOT["errors"] = [str(e)] + SNAPSHOT.get("errors", [])[:4]
        time.sleep(3)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server — serves the HTML + /api/state
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>CS2 Bot — Live Dashboard</title>
<style>
  * { box-sizing: border-box; }
  body { font: 13px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         margin: 0; background: #0d1117; color: #c9d1d9; }
  h1 { font-size: 18px; margin: 0; padding: 14px 20px; color: #58a6ff;
       background: #010409; border-bottom: 1px solid #30363d; display: flex;
       justify-content: space-between; align-items: center; }
  h1 .sub { font-size: 11px; color: #8b949e; font-weight: 400; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 14px; min-height: 120px; }
  .card h2 { font-size: 11px; color: #8b949e; text-transform: uppercase;
             margin: 0 0 8px 0; letter-spacing: 0.5px; }
  .big { font-size: 28px; font-weight: 600; color: #58a6ff; line-height: 1.2; }
  .big.ok  { color: #3fb950; }
  .big.warn { color: #d29922; }
  .big.bad  { color: #f85149; }
  .muted { color: #8b949e; font-size: 11px; }
  .row { display: flex; justify-content: space-between; padding: 2px 0; }
  .row strong { color: #c9d1d9; font-weight: 500; }
  pre.log { background: #010409; color: #8b949e; font-size: 11px;
            padding: 10px; border-radius: 6px; overflow: auto; max-height: 200px;
            margin: 8px 0 0 0; white-space: pre-wrap; word-break: break-all; }
  .bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-top: 6px; }
  .bar > div { height: 100%; background: linear-gradient(90deg, #3fb950, #58a6ff); transition: width 0.3s; }
  .bar.warn > div { background: linear-gradient(90deg, #d29922, #f85149); }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; vertical-align: middle; }
  .dot.ok { background: #3fb950; box-shadow: 0 0 8px #3fb950; }
  .dot.warn { background: #d29922; }
  .dot.bad { background: #f85149; }
  .dot.off { background: #484f58; }
  .wide { grid-column: span 2; }
  .mega { grid-column: span 4; }
  .loss-chart { height: 80px; width: 100%; margin-top: 8px; }
  table { width: 100%; font-size: 11px; border-collapse: collapse; margin-top: 4px; }
  table td, table th { padding: 3px 6px; text-align: left; border-bottom: 1px solid #21262d; }
  table th { color: #8b949e; text-transform: uppercase; font-size: 10px; font-weight: 500; }
  .pulse { animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1 } 50% { opacity: 0.5 } }
  footer { padding: 10px 20px; color: #484f58; font-size: 10px; border-top: 1px solid #30363d; }
</style>
</head><body>
<h1>⚡ CS2 Polymarket Bot — Live
  <span class="sub" id="age">—</span>
</h1>

<div class="grid">
  <div class="card"><h2>🎮 Gaming PC GPU</h2>
    <div id="gpu-big" class="big">—</div>
    <div class="muted" id="gpu-sub">—</div>
    <div class="bar"><div id="gpu-bar" style="width:0%"></div></div>
  </div>

  <div class="card"><h2>💾 VRAM</h2>
    <div id="vram-big" class="big">—</div>
    <div class="muted" id="vram-sub">—</div>
    <div class="bar"><div id="vram-bar" style="width:0%"></div></div>
  </div>

  <div class="card"><h2>🌡 Power / Temp</h2>
    <div id="pwr-big" class="big">—</div>
    <div class="muted" id="pwr-sub">—</div>
  </div>

  <div class="card"><h2>☁️ VPS Recorder</h2>
    <div id="vps-big" class="big">—</div>
    <div class="muted" id="vps-sub">—</div>
  </div>

  <div class="card wide"><h2>🔥 Training — loss curve</h2>
    <div id="loss-big" class="big">—</div>
    <div class="muted" id="loss-sub">—</div>
    <canvas id="loss-chart" class="loss-chart"></canvas>
  </div>

  <div class="card wide"><h2>⏱ Training progress</h2>
    <div id="step-big" class="big">—</div>
    <div class="muted" id="step-sub">—</div>
    <div class="bar"><div id="step-bar" style="width:0%"></div></div>
    <div class="row"><strong>Phase</strong>          <span id="phase">—</span></div>
    <div class="row"><strong>Step</strong>           <span id="step-counter">—</span></div>
    <div class="row"><strong>Elapsed / ETA</strong>  <span id="eta">—</span></div>
    <div class="row"><strong>Sec / step</strong>     <span id="sps">—</span></div>
  </div>

  <div class="card wide"><h2>📊 Dataset</h2>
    <div class="row"><strong>Recordings</strong>          <span id="ds-rec">—</span></div>
    <div class="row"><strong>Processed</strong>           <span id="ds-proc">—</span></div>
    <div class="row"><strong>SFT examples</strong>        <span id="ds-sft">—</span></div>
    <div class="row"><strong>SFT file size</strong>       <span id="ds-sft-size">—</span></div>
    <div class="row"><strong>Recordings size (local)</strong> <span id="ds-size">—</span></div>
  </div>

  <div class="card mega"><h2>🪵 Training log (tail)</h2>
    <pre class="log" id="train-tail">waiting for training to start…</pre>
  </div>

  <div class="card mega"><h2>📈 Recent evaluator runs</h2>
    <table><thead><tr><th>ran</th><th>model</th><th>examples</th><th>accuracy</th></tr></thead>
      <tbody id="evals-body"><tr><td colspan=4 class="muted">no eval runs yet</td></tr></tbody>
    </table>
  </div>
</div>

<footer>polls every 3s · refreshes in browser every 3s · dashboard at http://localhost:8090/</footer>

<script>
function fmtAgo(ts) {
  if (!ts) return '—';
  const s = Math.round(Date.now()/1000 - ts);
  if (s < 10) return 'live';
  if (s < 60) return s + 's ago';
  return Math.round(s/60) + 'm ago';
}
function _set(id, txt, cls) {
  const el = document.getElementById(id); if (!el) return;
  el.textContent = txt;
  if (cls !== undefined) el.className = 'big ' + cls;
}
function drawLossChart(losses, epochs) {
  const c = document.getElementById('loss-chart');
  const ctx = c.getContext('2d');
  c.width = c.clientWidth * devicePixelRatio;
  c.height = c.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const W = c.clientWidth, H = c.clientHeight;
  ctx.clearRect(0,0,W,H);
  if (!losses || losses.length < 2) {
    ctx.fillStyle = '#484f58'; ctx.font = '10px monospace';
    ctx.fillText('no data yet', 10, H/2);
    return;
  }
  const lo = Math.min(...losses) * 0.95, hi = Math.max(...losses) * 1.05;
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2;
  ctx.beginPath();
  losses.forEach((l, i) => {
    const x = (i / (losses.length-1)) * (W - 20) + 10;
    const y = H - 10 - ((l - lo)/(hi - lo)) * (H - 20);
    if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
  // Dots
  ctx.fillStyle = '#3fb950';
  losses.forEach((l,i) => {
    const x = (i / (losses.length-1)) * (W - 20) + 10;
    const y = H - 10 - ((l - lo)/(hi - lo)) * (H - 20);
    ctx.beginPath(); ctx.arc(x,y,2,0,2*Math.PI); ctx.fill();
  });
  // Range label
  ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
  ctx.fillText(hi.toFixed(2), W-40, 12);
  ctx.fillText(lo.toFixed(2), W-40, H-4);
}
async function refresh() {
  try {
    const r = await fetch('/api/state'); const d = await r.json();
    document.getElementById('age').textContent = 'last poll ' + fmtAgo(d.ts);

    // GPU
    const g = d.gpu || {};
    if (g.online) {
      _set('gpu-big', g.util_pct + '%', g.util_pct > 50 ? 'ok' : '');
      document.getElementById('gpu-sub').textContent = 'RTX 5090 active';
      document.getElementById('gpu-bar').style.width = g.util_pct + '%';
      _set('vram-big', (g.vram_used_mib/1024).toFixed(1) + ' GB',
           g.vram_used_mib / g.vram_total_mib > 0.9 ? 'warn' : '');
      document.getElementById('vram-sub').textContent = 'of ' + (g.vram_total_mib/1024).toFixed(0) + ' GB';
      document.getElementById('vram-bar').style.width = (g.vram_used_mib / g.vram_total_mib * 100) + '%';
      _set('pwr-big', Math.round(g.power_w) + 'W');
      document.getElementById('pwr-sub').textContent = g.temp_c + '°C';
    } else {
      _set('gpu-big', 'offline', 'bad');
      document.getElementById('gpu-sub').textContent = g.error || 'unknown';
    }

    // VPS
    const v = d.vps || {};
    if (v.online && v.active === 'active') {
      _set('vps-big', 'LIVE', 'ok');
      document.getElementById('vps-sub').textContent =
        (v.recording_count||0) + ' files · ' + (v.recordings_size||'?');
    } else if (v.online) {
      _set('vps-big', v.active || 'unknown', 'warn');
    } else {
      _set('vps-big', 'offline', 'bad');
    }

    // Training
    const t = d.train || {};
    if (t.failed) {
      _set('loss-big', 'FAILED', 'bad');
      document.getElementById('loss-sub').textContent = 'training crashed — see log below';
    } else if (t.completed) {
      _set('loss-big', (t.latest_loss||0).toFixed(4), 'ok');
      document.getElementById('loss-sub').textContent =
        'TRAIN DONE · final loss ' + (t.latest_loss||0).toFixed(4);
    } else if (t.latest_loss !== null && t.latest_loss !== undefined) {
      _set('loss-big', t.latest_loss.toFixed(4), '');
      document.getElementById('loss-sub').textContent =
        'epoch ' + (t.latest_epoch||0).toFixed(2) + ' · training…';
      document.getElementById('loss-big').classList.add('pulse');
    } else {
      _set('loss-big', '—');
      document.getElementById('loss-sub').textContent = 'waiting for first loss step';
    }
    drawLossChart(t.loss_history || [], t.epoch_history || []);

    // Step progress
    const step = t.step || {};
    document.getElementById('phase').textContent = t.phase || '—';
    if (step.total) {
      _set('step-big', step.pct + '%', step.pct >= 100 ? 'ok' : '');
      document.getElementById('step-sub').textContent =
        'step ' + step.current + ' of ' + step.total;
      document.getElementById('step-bar').style.width = step.pct + '%';
      document.getElementById('step-counter').textContent = step.current + ' / ' + step.total;
      document.getElementById('eta').textContent = step.elapsed + ' · ETA ' + step.eta;
      document.getElementById('sps').textContent = step.sec_per_step + 's';
    } else {
      _set('step-big', '—');
      document.getElementById('step-sub').textContent = t.phase || 'not training';
      document.getElementById('step-bar').style.width = '0%';
    }
    const tail = (t.tail || []).filter(s => s && s.trim()).join('\n');
    document.getElementById('train-tail').textContent = tail || 'waiting for training to start…';

    // Dataset
    const s = d.dataset || {};
    document.getElementById('ds-rec').textContent = (s.recordings_files||0) + ' files';
    document.getElementById('ds-proc').textContent = (s.processed_files||0) + ' files';
    document.getElementById('ds-sft').textContent = (s.sft_examples||0).toLocaleString();
    document.getElementById('ds-sft-size').textContent = (s.sft_size_mb||0).toFixed(1) + ' MB';
    document.getElementById('ds-size').textContent = (s.recordings_size_mb||0).toFixed(0) + ' MB';

    // Evals
    const tb = document.getElementById('evals-body');
    if (d.evals && d.evals.length > 0) {
      tb.innerHTML = d.evals.map(e => {
        const age = fmtAgo(e.ts);
        const acc = (e.accuracy*100).toFixed(1) + '%';
        const cls = e.accuracy >= 0.9 ? 'ok' : (e.accuracy >= 0.5 ? 'warn' : 'bad');
        return `<tr><td>${age}</td><td>${e.label}</td><td>${e.examples}</td>
                <td class="big ${cls}" style="font-size:13px">${acc}</td></tr>`;
      }).join('');
    }
  } catch (e) {
    console.error('refresh failed', e);
  }
}
refresh(); setInterval(refresh, 3000);
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/api/state":
            with SNAPSHOT_LOCK:
                body = json.dumps(SNAPSHOT).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404); self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    # Start the poller thread
    threading.Thread(target=poll_everything, daemon=True).start()

    url = f"http://localhost:{args.port}/"
    print(f"[dashboard] serving at {url}")
    print(f"[dashboard] opens in Chrome (use --no-open to skip)")

    if not args.no_open:
        try:
            # macOS: open -a "Google Chrome" URL
            subprocess.Popen(["open", "-a", "Google Chrome", url])
        except Exception:
            webbrowser.open(url)

    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopping")
        srv.server_close()


if __name__ == "__main__":
    main()
