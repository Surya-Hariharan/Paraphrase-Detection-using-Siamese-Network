"""DDoS protection system"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ClientProfile:
    ip: str
    request_count: int = 0
    first_seen: float = 0
    last_seen: float = 0
    blocked_until: float = 0
    threat_level: ThreatLevel = ThreatLevel.LOW


class DDoSProtector:
    """Simplified DDoS protection"""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_threshold: int = 100,
        block_duration: int = 300
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_threshold = burst_threshold
        self.block_duration = block_duration
        
        self.clients: Dict[str, ClientProfile] = {}
        self.request_times: Dict[str, list] = defaultdict(list)
    
    def check_request(self, ip: str) -> tuple[bool, Optional[str]]:
        """
        Check if request should be allowed
        
        Returns:
            (allowed, reason)
        """
        now = time.time()
        
        # Get or create client profile
        if ip not in self.clients:
            self.clients[ip] = ClientProfile(
                ip=ip,
                first_seen=now,
                last_seen=now
            )
        
        client = self.clients[ip]
        
        # Check if blocked
        if client.blocked_until > now:
            return False, f"Blocked until {int(client.blocked_until - now)}s"
        
        # Clean old requests (older than 1 minute)
        cutoff = now - 60
        self.request_times[ip] = [t for t in self.request_times[ip] if t > cutoff]
        
        # Add current request
        self.request_times[ip].append(now)
        client.request_count += 1
        client.last_seen = now
        
        # Check rate limit
        recent_requests = len(self.request_times[ip])
        
        if recent_requests > self.burst_threshold:
            # Critical threat - long block
            client.threat_level = ThreatLevel.CRITICAL
            client.blocked_until = now + (self.block_duration * 2)
            return False, "Burst limit exceeded - blocked"
        
        if recent_requests > self.requests_per_minute:
            # High threat - normal block
            client.threat_level = ThreatLevel.HIGH
            client.blocked_until = now + self.block_duration
            return False, "Rate limit exceeded"
        
        if recent_requests > self.requests_per_minute * 0.8:
            # Medium threat - warning
            client.threat_level = ThreatLevel.MEDIUM
        
        return True, None
    
    def cleanup(self, max_age: int = 600):
        """Remove old client data"""
        now = time.time()
        cutoff = now - max_age
        
        # Remove old clients
        old_clients = [
            ip for ip, client in self.clients.items()
            if client.last_seen < cutoff
        ]
        
        for ip in old_clients:
            del self.clients[ip]
            if ip in self.request_times:
                del self.request_times[ip]
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        now = time.time()
        
        active_clients = sum(
            1 for client in self.clients.values()
            if client.last_seen > now - 60
        )
        
        blocked_clients = sum(
            1 for client in self.clients.values()
            if client.blocked_until > now
        )
        
        threat_counts = {
            "low": sum(1 for c in self.clients.values() if c.threat_level == ThreatLevel.LOW),
            "medium": sum(1 for c in self.clients.values() if c.threat_level == ThreatLevel.MEDIUM),
            "high": sum(1 for c in self.clients.values() if c.threat_level == ThreatLevel.HIGH),
            "critical": sum(1 for c in self.clients.values() if c.threat_level == ThreatLevel.CRITICAL),
        }
        
        return {
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "blocked_clients": blocked_clients,
            "threat_levels": threat_counts
        }
