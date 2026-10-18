"""Serve the fake RAG over HTTP: POST /ask {"question": ...}, GET /health.

python examples/fake_rag/server.py --port 8765
FAKE_RAG_MODE=regressed python examples/fake_rag/server.py --port 8766
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rag import ask


class Handler(BaseHTTPRequestHandler):
    mode = "good"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"status": "ok", "mode": self.mode})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/ask":
            self._send(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            question = body["question"]
        except (ValueError, KeyError, TypeError):
            self._send(422, {"error": 'expected {"question": "..."}'})
            return
        self._send(200, ask(question, self.mode))

    def log_message(self, format: str, *args: object) -> None:
        pass  # quiet


def serve(port: int, mode: str) -> ThreadingHTTPServer:
    handler = type("ModeHandler", (Handler,), {"mode": mode})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", default=os.environ.get("FAKE_RAG_MODE", "good"),
                    choices=["good", "regressed"])  # fmt: skip
    args = ap.parse_args()
    server = serve(args.port, args.mode)
    print(f"fake RAG ({args.mode}) on http://127.0.0.1:{args.port}/ask", flush=True)
    server.serve_forever()
