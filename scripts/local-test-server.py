from __future__ import annotations

import argparse
import mimetypes
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
MODEL_CACHE = ROOT / "model-cache"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".gguf")


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(path)
        rel = urllib.parse.unquote(parsed.path.lstrip("/"))
        if rel.startswith("model-cache/"):
            return str(ROOT / rel)
        if rel.startswith("assets/"):
            return str(DIST / rel)
        target = DIST / rel
        if target.exists() and target.is_file():
            return str(target)
        return str(DIST / "index.html")

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the built app and local model-cache for browser testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8041)
    args = parser.parse_args()

    if not (DIST / "index.html").exists():
        raise SystemExit("dist/index.html is missing. Run `npm run build` first.")
    if not MODEL_CACHE.exists():
        raise SystemExit("model-cache/ is missing. Download model files before local model testing.")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {DIST} and {MODEL_CACHE} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
