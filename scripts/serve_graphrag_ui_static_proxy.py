#!/usr/bin/env python3
"""Serve the built frontend SPA and proxy API calls to the local backend.

This is a small fallback server for local GraphRAG UI testing when Vite is not
available or the dev server cannot start. Build the frontend first, then run:

  python3 scripts/serve_graphrag_ui_static_proxy.py --port 55846

The server intentionally keeps behavior simple: non-API routes fall back to
index.html, while /api/* and /v1/* are proxied to the configured backend.
"""

from __future__ import annotations

import argparse
import http.client
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DIST_DIR = ROOT_DIR / "frontend" / "dist-graphrag"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the static/proxy server. Default: %(default)s",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=55846,
        help="Port for the static/proxy server. Default: %(default)s",
    )
    parser.add_argument(
        "--dist-dir",
        type=Path,
        default=DEFAULT_DIST_DIR,
        help="Built frontend dist directory. Default: %(default)s",
    )
    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8799",
        help="Backend URL for /api/* and /v1/* proxying. Default: %(default)s",
    )
    return parser.parse_args()


def parse_backend_target(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", ""}:
        raise ValueError("Only http backend URLs are supported by this lightweight proxy.")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    return host, port


def make_handler(
    *,
    dist_dir: Path,
    backend_host: str,
    backend_port: int,
) -> type[BaseHTTPRequestHandler]:
    resolved_dist = dist_dir.resolve()

    class StaticProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            sys.stderr.write(f"{self.address_string()} - {fmt % args}\n")

        def do_HEAD(self) -> None:
            self._handle_static_or_proxy(send_body=False)

        def do_GET(self) -> None:
            self._handle_static_or_proxy(send_body=True)

        def do_POST(self) -> None:
            self._proxy_to_backend()

        def do_PUT(self) -> None:
            self._proxy_to_backend()

        def do_PATCH(self) -> None:
            self._proxy_to_backend()

        def do_DELETE(self) -> None:
            self._proxy_to_backend()

        def do_OPTIONS(self) -> None:
            self._proxy_to_backend()

        def _handle_static_or_proxy(self, *, send_body: bool) -> None:
            if self.path.startswith(("/api/", "/v1/")):
                self._proxy_to_backend(send_body=send_body)
                return

            parsed = urlsplit(self.path)
            relative_path = parsed.path.lstrip("/")
            target = (resolved_dist / relative_path).resolve() if relative_path else resolved_dist / "index.html"
            try:
                target.relative_to(resolved_dist)
            except ValueError:
                target = resolved_dist / "index.html"
            if not target.is_file():
                target = resolved_dist / "index.html"

            try:
                payload = target.read_bytes()
            except OSError as exc:
                self.send_error(404, f"Unable to read frontend asset: {exc}")
                return

            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                self.wfile.write(payload)

        def _proxy_to_backend(self, *, send_body: bool = True) -> None:
            body_length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(body_length) if body_length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection", "accept-encoding"}
            }
            connection = http.client.HTTPConnection(backend_host, backend_port, timeout=120)
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if key.lower() in {"connection", "transfer-encoding", "content-encoding"}:
                        continue
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if send_body:
                    self.wfile.write(payload)
            except OSError as exc:
                self.send_error(502, f"Backend proxy failed: {exc}")
            finally:
                connection.close()

    return StaticProxyHandler


def main() -> int:
    args = parse_args()
    dist_dir = args.dist_dir.resolve()
    index_html = dist_dir / "index.html"
    if not index_html.is_file():
        print(
            f"Frontend build not found at {index_html}. Run `npm --prefix frontend run build` first.",
            file=sys.stderr,
        )
        return 2

    try:
        backend_host, backend_port = parse_backend_target(args.backend_url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    handler = make_handler(
        dist_dir=dist_dir,
        backend_host=backend_host,
        backend_port=backend_port,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Serving {dist_dir} at http://{args.host}:{args.port} "
        f"with /api and /v1 proxied to http://{backend_host}:{backend_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
