#!/usr/bin/env python3
"""
Tiny always-on HTTP server for the shootout index page.
Runs on port 8085 independently of replay_backtest.py so the link
stays alive during warmup/model-swap gaps.

Serves:
  GET /                  → _logs/shootout/index.html
  GET /live              → redirect to http://<host>:8084/  (active backtest)
  GET /any-file.log      → plaintext log file from _logs/shootout/
"""
import http.server
import os
import socketserver
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_logs", "shootout")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8085


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence default access log

    def do_GET(self):
        path = self.path.lstrip("/").split("?", 1)[0]
        if path in ("", "index.html"):
            idx = os.path.join(ROOT, "index.html")
            if os.path.exists(idx):
                with open(idx, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = (b"<html><body style='font-family:Arial;background:#0d1117;color:#c9d1d9;padding:30px'>"
                    b"<h2>Shootout not started yet</h2>"
                    b"<p>Launch <code>bash _model_shootout.sh</code>.</p>"
                    b"</body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "live":
            self.send_response(302)
            host = self.headers.get("Host", "localhost:8085").split(":")[0]
            self.send_header("Location", f"http://{host}:8084/")
            self.end_headers()
            return
        if path.endswith(".log") and "/" not in path and ".." not in path:
            p = os.path.join(ROOT, path)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.allow_reuse_address = True
        print(f"[shootout-index] serving {ROOT} at http://localhost:{PORT}/")
        httpd.serve_forever()
