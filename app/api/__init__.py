"""
__init__.py
Part of the app/api module.
Exports API routes, WebSocket handlers, and middleware.
"""

from .routes import router as api_router
from .websocket import router as ws_router, manager, websocket_endpoint
from .middleware import (
    setup_middleware,
    RateLimitMiddleware,
    LoggingMiddleware,
    AuthMiddleware
)

__all__ = [
    'api_router',
    'ws_router',
    'manager',
    'websocket_endpoint',
    'setup_middleware',
    'RateLimitMiddleware',
    'LoggingMiddleware',
    'AuthMiddleware'
]