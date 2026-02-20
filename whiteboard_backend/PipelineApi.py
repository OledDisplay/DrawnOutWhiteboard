# hot_api_server.py
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

import shared_models

# Import your pipeline module so /get_images can call it.
# Rename "pipeline_script" to the filename (without .py) that contains get_images().
import pipeline_script  # <-- CHANGE THIS


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
            return
        self._send(404, {"ok": False, "err": "not_found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            self._send(400, {"ok": False, "err": "bad_json"})
            return

        if self.path == "/get_images":
            # Expected input:
            # { "prompts": {"Eukaryotic cell":"Biology", ...}, "gpu_index":0 }
            prompts = body.get("prompts") or {}
            gpu_index = int(body.get("gpu_index", 0) or 0)
            try:
                out = pipeline_script.get_images(prompts, gpu_index=gpu_index)
                self._send(200, {"ok": True, "result": out})
            except Exception as e:
                self._send(500, {"ok": False, "err": repr(e)})
            return

        self._send(404, {"ok": False, "err": "not_found"})


def main():
    # Load everything ONCE and keep it hot
    shared_models.init_hot_models(
        qwen_model_id="Qwen/Qwen3-VL-2B-Instruct",
        gpu_index=0,
        cpu_threads=4,
        warmup=True,
    )

    host = "127.0.0.1"
    port = 8787
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[hot_api] listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
