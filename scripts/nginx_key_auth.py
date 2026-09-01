#!/usr/bin/env python3
"""Bearer-key auth service for the nginx auth_request subrequest pattern.

nginx config:
    location = /_qwen_auth {
        internal;
        proxy_pass http://127.0.0.1:8787/auth;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-Authorization $http_authorization;
    }

Keys are read from a file (one per line, 'name:key' or bare key), root-only:
    /etc/nginx/api_keys
Reloaded on every request so rotation needs no restart/redeploy.

Run:  python3 nginx_key_auth.py --keys /etc/nginx/api_keys --port 8787
Systemd unit: /etc/systemd/system/nginx-key-auth.service
"""
from __future__ import annotations

import argparse
import hmac
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def load_keys(path: Path) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    keys = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # allow "name:key" comments-style entries; use the part after last ':'
        keys.append(line.split(":", 1)[1] if (":" in line and not line.startswith("sk-")) else line)
    return keys


class AuthHandler(BaseHTTPRequestHandler):
    keys_file: Path = Path("/etc/nginx/api_keys")

    def log_message(self, fmt, *args):  # quiet; nginx logs the outcome anyway
        pass

    def do_GET(self):
        if self.path != "/auth":
            self.send_response(404)
            self.end_headers()
            return
        header = self.headers.get("X-Original-Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        keys = load_keys(self.keys_file)
        ok = any(k and hmac.compare_digest(k, token) for k in keys)
        self.send_response(204 if ok else 401)
        self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="/etc/nginx/api_keys")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    AuthHandler.keys_file = Path(args.keys)
    if not load_keys(AuthHandler.keys_file):
        print(f"warning: no keys found in {args.keys}", file=sys.stderr)
    srv = ThreadingHTTPServer((args.host, args.port), AuthHandler)
    print(f"listening on {args.host}:{args.port}, keys={args.keys}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
