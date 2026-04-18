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


def create_app(bot):
    """Create Flask app wired to the bot."""
    bot.on_state_change(push_state)
    return app
