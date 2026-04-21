"""
Middleware Module - Logging + Rate Limiting (Clean & Safe)
"""
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# In-memory store (consider Redis for production scaling)
rate_limit_storage = defaultdict(list)


# -------------------------------
# 🔍 Logging Middleware
# -------------------------------
class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request with execution time"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Time: {duration:.4f}s"
        )

        return response




# -------------------------------
# 🔐 Auth Middleware
# -------------------------------
class AuthMiddleware(BaseHTTPMiddleware):
    """
    Simple API Key Authentication Middleware
    """

    def __init__(self, app, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key or "dev-secret-key"  # fallback for dev

        # Public endpoints (no auth required)
        self.public_paths = [
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/dashboard"
        ]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # ✅ Allow public routes
        if any(path.startswith(p) for p in self.public_paths):
            return await call_next(request)

        # ✅ Allow static + websocket
        if path.startswith("/static") or path.startswith("/ws"):
            return await call_next(request)

        # 🔐 Check API key
        client_key = request.headers.get("x-api-key")

        if not client_key or client_key != self.api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "success": False,
                    "error": "Unauthorized",
                    "message": "Invalid or missing API key"
                }
            )

        return await call_next(request)
# -------------------------------
# 🚦 Rate Limiting Middleware
# -------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced Rate limiting with local bypass and transparency"""

    def __init__(self, app, requests_per_minute: int = 500):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.whitelist = ["127.0.0.1", "localhost"]

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        # ✅ Bypass for WebSocket, static files, localhost
        if (
            request.url.path.startswith("/ws")
            or request.url.path.startswith("/static")
            or client_ip in self.whitelist
        ):
            return await call_next(request)

        client_id = f"ip:{client_ip}"
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Cleanup old timestamps
        rate_limit_storage[client_id] = [
            ts for ts in rate_limit_storage[client_id]
            if ts > minute_ago
        ]

        current_usage = len(rate_limit_storage[client_id])

        # 🚫 Limit exceeded
        if current_usage >= self.requests_per_minute:
            logger.warning(f"Rate limit hit for {client_id}: {current_usage}/min")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "success": False,
                    "error": "Rate limit exceeded",
                    "limit": self.requests_per_minute,
                    "retry_after": "60 seconds"
                }
            )

        # ✅ Record request
        rate_limit_storage[client_id].append(now)

        response = await call_next(request)

        # 📊 Add headers for frontend visibility
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - current_usage - 1
        )

        return response


# -------------------------------
# ⚙️ Setup Function
# -------------------------------
def setup_middleware(app):
    """Attach middleware to FastAPI app"""

    # Order matters
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware, api_key="dev-secret-key")  # 🔐 ADD THIS
    app.add_middleware(RateLimitMiddleware, requests_per_minute=500)

    logger.info("✅ Middleware loaded: Logging + Auth + RateLimit")

    return app