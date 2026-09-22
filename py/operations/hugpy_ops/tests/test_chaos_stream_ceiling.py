"""chaos client: /chat/stream must honour a HARD wall-clock ceiling even when
the server streams 'awaiting-load' progress forever (the per-socket-read
timeout never fires while data keeps flowing)."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hugpy_ops.chaos.client import CentralClient


@pytest.fixture
def endless_load_server():
    cancels: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if self.path.startswith("/llm/chat/cancel/"):
                cancels.append(self.path.rsplit("/", 1)[-1])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"cancelled": true}')
                return
            if self.path == "/chat/stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for _ in range(2000):
                        ev = {"type": "status", "stage": "awaiting-load",
                              "worker_name": "faux", "progress": 0.0}
                        self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                        time.sleep(0.1)
                except Exception:
                    pass
                return
            self.send_response(404)
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", cancels
    finally:
        srv.shutdown()


def test_chat_stream_honours_wall_ceiling_and_cancels(endless_load_server):
    base, cancels = endless_load_server
    client = CentralClient(base)
    t0 = time.time()
    term = client.chat_stream("faux-model", "hi", "chaos-ceiling-test",
                              max_new_tokens=8, ceiling_s=1.5, read_timeout_s=1.0)
    elapsed = time.time() - t0
    assert term is not None
    assert elapsed < 6.0
    assert term["outcome"] == "load-timeout"
    assert term["served_worker"] == "faux"
    assert any(s[0] == "awaiting-load" for s in term["stages"])
    assert term["error"] and "ceiling" in term["error"]
    assert "chaos-ceiling-test" in cancels
