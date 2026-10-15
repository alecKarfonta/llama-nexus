"""Optional management-API token authentication.

[2026-10-15] | INFO | AUTH.MANAGEMENT_TOKEN | The management API historically
had zero authentication. For localhost single-user installs that is fine, but
LAN-exposed installs need a gate. When MANAGEMENT_AUTH_TOKEN is set, all
non-public routes require `Authorization: Bearer <token>` (or
`X-Management-Token`). Off by default — zero behavior change for local use.
"""

import os
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from enhanced_logger import enhanced_logger as logger

# Paths that stay open regardless (health probes, docs) so orchestrators and
# the frontend can detect the backend before authentication is configured.
PUBLIC_PATHS = {
    "/health",
    "/api/v1/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def get_management_token() -> Optional[str]:
    """Return the configured management token, or None when auth is off."""
    return os.getenv("MANAGEMENT_AUTH_TOKEN", "").strip() or None


class ManagementAuthMiddleware(BaseHTTPMiddleware):
    """Require a bearer token for management routes when configured.

    [2026-10-15] | INFO | AUTH.MIDDLEWARE | Active only when
    MANAGEMENT_AUTH_TOKEN is set (e.g. by install.sh for LAN mode).
    """

    async def dispatch(self, request: Request, call_next):
        token = get_management_token()
        if token is None:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # [2026-10-15] | DEBUG | AUTH.MIDDLEWARE | Accept Bearer header or the
        # X-Management-Token fallback (handy for WebSocket clients).
        auth_header = request.headers.get("authorization", "")
        bearer = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else ""
        alt = request.headers.get("x-management-token", "").strip()
        provided = bearer or alt

        # WebSocket upgrade requests can't add custom headers from browsers;
        # allow token via query param for the /ws routes only.
        if not provided and path.startswith("/ws"):
            provided = request.query_params.get("token", "").strip()

        if provided != token:
            logger.warning(f"Management auth rejected {request.method} {path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Management API token required"},
            )

        return await call_next(request)
