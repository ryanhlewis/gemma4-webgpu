from __future__ import annotations

import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
PORT = int(os.environ.get("PORT", "7860"))

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/wasm", ".wasm")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def translate_path(self, path):
        resolved = Path(super().translate_path(path))
        if resolved.is_dir():
            return str(DIST / "index.html")
        if not resolved.exists() and not path.startswith("/assets/"):
            return str(DIST / "index.html")
        return str(resolved)

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)


if __name__ == "__main__":
    if not (DIST / "index.html").exists():
        raise SystemExit("dist/index.html is missing. Run `npm run build` before deploying.")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving {DIST} on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
