"""Security middleware"""

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time

from backend.security.ddos.protector import DDoSProtector
from backend.config.settings import settings

# Global DDoS protector
ddos_protector = DDoSProtector(
    requests_per_minute=settings.security.RATE_LIMIT_PER_MINUTE,
    burst_threshold=100,
    block_duration=300
)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for DDoS protection"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip security for health check
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check DDoS protection
        if settings.security.DDOS_PROTECTION_ENABLED:
            allowed, reason = ddos_protector.check_request(client_ip)
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {reason}"
                )
        
        # Process request
        response = await call_next(request)
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request logging middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
