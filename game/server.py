"""Run the game in a background thread (for run_e2e.py and tests)."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from flask import Flask
from werkzeug.serving import WSGIRequestHandler, make_server


class _QuietHandler(WSGIRequestHandler):
    def log_request(self, *args, **kwargs) -> None:  # keep e2e/test output readable
        pass


@contextmanager
def serve_in_thread(app: Flask, host: str = "127.0.0.1", port: int = 0):
    """Yield the base URL of ``app`` served on ``port`` (0 = any free port)."""
    server = make_server(host, port, app, threaded=True, request_handler=_QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
