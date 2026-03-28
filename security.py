"""Rate limiting middleware and security utilities"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from time import time
from collections import defaultdict
from typing import Dict, Tuple


class RateLimiter:
    """Simple in-memory rate limiter (200 requests per minute per IP)"""
    
    def __init__(self, max_requests: int = 200, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if client is within rate limit"""
        now = time()
        
        # Clean old requests outside the window
        self.requests[client_ip] = [
            timestamp for timestamp in self.requests[client_ip]
            if now - timestamp < self.window_seconds
        ]
        
        # Check if over limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        
        # Add new request
        self.requests[client_ip].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=200, window_seconds=60)


async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce rate limiting"""
    
    # Skip health check endpoints
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get client IP (handle reverse proxy)
    client_ip = request.client.host
    if forwarded_for := request.headers.get("x-forwarded-for"):
        client_ip = forwarded_for.split(",")[0].strip()
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Max 200 requests per minute."}
        )
    
    return await call_next(request)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling reverse proxies"""
    if forwarded_for := request.headers.get("x-forwarded-for"):
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
