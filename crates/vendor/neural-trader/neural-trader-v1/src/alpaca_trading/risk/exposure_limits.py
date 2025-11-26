"""Dynamic exposure limits with real-time tracking.

Manages symbol-level, sector, and portfolio-wide exposure limits.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)


class LimitType(Enum):
    """Types of exposure limits."""
    POSITION_SIZE = "position_size"
    SECTOR_EXPOSURE = "sector_exposure"
    CONCENTRATION = "concentration"
    NOTIONAL_VALUE = "notional_value"
    DELTA_EXPOSURE = "delta_exposure"
    BETA_WEIGHTED = "beta_weighted"


@dataclass
class ExposureLimit:
    """Definition of an exposure limit."""
    limit_type: LimitType
    identifier: str  # Symbol, sector, or "portfolio"
    limit_value: Decimal
    current_value: Decimal = Decimal("0")
    utilization_pct: float = 0.0
    is_breached: bool = False
    is_warning: bool = False  # 80% of limit
    last_update_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class LimitViolation:
    """Record of a limit violation."""
    limit_type: LimitType
    identifier: str
    current_value: Decimal
    limit_value: Decimal
    excess_amount: Decimal
    severity: str  # "warning", "breach", "critical"
    timestamp_ms: int
    action_required: str


class ExposureLimits:
    """Dynamic exposure limit management system."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        # Active limits by type and identifier
        self.limits: Dict[tuple[LimitType, str], ExposureLimit] = {}
        
        # Current exposures
        self.position_exposures: Dict[str, Decimal] = {}
        self.sector_exposures: Dict[str, Decimal] = {}
        self.portfolio_metrics = {
            "total_notional": Decimal("0"),
            "total_delta": Decimal("0"),
            "total_beta_weighted": Decimal("0")
        }
        
        # Violation tracking
        self.violations: List[LimitViolation] = []
        self.violation_callbacks = []
        
        # Default limits
        self._load_default_limits()
        
    async def initialize(self):
        """Initialize Redis connection and load saved limits."""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        
        # Load saved limits from Redis
        await self._load_limits_from_redis()
        
        logger.info("Exposure limits initialized")
        
    async def close(self):
        """Clean up resources."""
        if self.redis:
            await self.redis.close()
            
    def _load_default_limits(self):
        """Load default exposure limits."""
        # Portfolio-wide limits
        self.add_limit(LimitType.NOTIONAL_VALUE, "portfolio", Decimal("1000000"))
        self.add_limit(LimitType.DELTA_EXPOSURE, "portfolio", Decimal("50000"))
        self.add_limit(LimitType.BETA_WEIGHTED, "portfolio", Decimal("100000"))
        
        # Default sector limits (30% of portfolio)
        for sector in ["Technology", "Financials", "Healthcare", "Energy", "Consumer"]:
            self.add_limit(LimitType.SECTOR_EXPOSURE, sector, Decimal("300000"))
            
        # Concentration limit (10% max per position)
        self.add_limit(LimitType.CONCENTRATION, "max_position_pct", Decimal("10"))
        
    async def _load_limits_from_redis(self):
        """Load saved limits from Redis."""
        # Load position limits
        position_limits = await self.redis.hgetall("limits:positions")
        for symbol, limit_str in position_limits.items():
            self.add_limit(LimitType.POSITION_SIZE, symbol, Decimal(limit_str))
            
        # Load sector limits
        sector_limits = await self.redis.hgetall("limits:sectors")
        for sector, limit_str in sector_limits.items():
            self.add_limit(LimitType.SECTOR_EXPOSURE, sector, Decimal(limit_str))
            
        # Load portfolio limits
        portfolio_limits = await self.redis.hgetall("limits:portfolio")
        for limit_type_str, limit_str in portfolio_limits.items():
            try:
                limit_type = LimitType(limit_type_str)
                self.add_limit(limit_type, "portfolio", Decimal(limit_str))
            except ValueError:
                logger.warning(f"Unknown limit type: {limit_type_str}")
                
    def add_limit(self, limit_type: LimitType, identifier: str, limit_value: Decimal):
        """Add or update an exposure limit."""
        key = (limit_type, identifier)
        
        self.limits[key] = ExposureLimit(
            limit_type=limit_type,
            identifier=identifier,
            limit_value=limit_value
        )
        
    async def update_exposure(self, symbol: str, exposure: Decimal, 
                            sector: Optional[str] = None,
                            delta: Optional[Decimal] = None,
                            beta: Optional[float] = None):
        """Update exposure for a symbol with < 2ms latency."""
        start_time = time.perf_counter()
        
        # Update position exposure
        self.position_exposures[symbol] = exposure
        
        # Update sector exposure if provided
        if sector:
            current_sector_exposure = sum(
                exp for sym, exp in self.position_exposures.items()
                if self._get_sector(sym) == sector
            )
            self.sector_exposures[sector] = current_sector_exposure
            
        # Update portfolio metrics
        self.portfolio_metrics["total_notional"] = sum(self.position_exposures.values())
        
        if delta is not None:
            # Track delta exposure
            await self._update_delta_exposure(symbol, delta)
            
        if beta is not None:
            # Update beta-weighted exposure
            beta_weighted = exposure * Decimal(str(beta))
            await self._update_beta_weighted_exposure(symbol, beta_weighted)
            
        # Check all limits
        violations = await self._check_all_limits()
        
        # Process violations
        if violations:
            await self._process_violations(violations)
            
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 2:
            logger.warning(f"Exposure update took {elapsed_ms:.2f}ms")
            
    async def _update_delta_exposure(self, symbol: str, delta: Decimal):
        """Update delta exposure tracking."""
        key = f"delta:{symbol}"
        await self.redis.hset("exposures:delta", symbol, str(delta))
        
        # Calculate total portfolio delta
        all_deltas = await self.redis.hgetall("exposures:delta")
        total_delta = sum(Decimal(d) for d in all_deltas.values())
        self.portfolio_metrics["total_delta"] = total_delta
        
    async def _update_beta_weighted_exposure(self, symbol: str, beta_weighted: Decimal):
        """Update beta-weighted exposure tracking."""
        await self.redis.hset("exposures:beta_weighted", symbol, str(beta_weighted))
        
        # Calculate total beta-weighted exposure
        all_beta = await self.redis.hgetall("exposures:beta_weighted")
        total_beta = sum(Decimal(b) for b in all_beta.values())
        self.portfolio_metrics["total_beta_weighted"] = total_beta
        
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (would use real mapping in production)."""
        # Simplified sector mapping
        tech_symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
        financial_symbols = ["JPM", "BAC", "GS", "MS", "WFC"]
        
        if symbol in tech_symbols:
            return "Technology"
        elif symbol in financial_symbols:
            return "Financials"
        else:
            return "Other"
            
    async def _check_all_limits(self) -> List[LimitViolation]:
        """Check all limits and return violations."""
        violations = []
        current_time = int(time.time() * 1000)
        
        # Check position limits
        for symbol, exposure in self.position_exposures.items():
            key = (LimitType.POSITION_SIZE, symbol)
            if key in self.limits:
                limit = self.limits[key]
                limit.current_value = exposure
                limit.utilization_pct = float(exposure / limit.limit_value * 100)
                
                if exposure > limit.limit_value:
                    limit.is_breached = True
                    violations.append(self._create_violation(
                        limit, exposure, "breach", current_time
                    ))
                elif limit.utilization_pct > 80:
                    limit.is_warning = True
                    violations.append(self._create_violation(
                        limit, exposure, "warning", current_time
                    ))
                    
        # Check sector limits
        for sector, exposure in self.sector_exposures.items():
            key = (LimitType.SECTOR_EXPOSURE, sector)
            if key in self.limits:
                limit = self.limits[key]
                limit.current_value = exposure
                limit.utilization_pct = float(exposure / limit.limit_value * 100)
                
                if exposure > limit.limit_value:
                    limit.is_breached = True
                    violations.append(self._create_violation(
                        limit, exposure, "breach", current_time
                    ))
                    
        # Check portfolio limits
        for metric_name, value in self.portfolio_metrics.items():
            if metric_name == "total_notional":
                limit_type = LimitType.NOTIONAL_VALUE
            elif metric_name == "total_delta":
                limit_type = LimitType.DELTA_EXPOSURE
            elif metric_name == "total_beta_weighted":
                limit_type = LimitType.BETA_WEIGHTED
            else:
                continue
                
            key = (limit_type, "portfolio")
            if key in self.limits:
                limit = self.limits[key]
                limit.current_value = value
                limit.utilization_pct = float(value / limit.limit_value * 100)
                
                if value > limit.limit_value:
                    limit.is_breached = True
                    violations.append(self._create_violation(
                        limit, value, "critical", current_time
                    ))
                    
        # Check concentration limits
        total_exposure = self.portfolio_metrics["total_notional"]
        if total_exposure > 0:
            for symbol, exposure in self.position_exposures.items():
                concentration_pct = exposure / total_exposure * 100
                
                key = (LimitType.CONCENTRATION, "max_position_pct")
                if key in self.limits:
                    limit = self.limits[key]
                    if concentration_pct > limit.limit_value:
                        violations.append(LimitViolation(
                            limit_type=LimitType.CONCENTRATION,
                            identifier=symbol,
                            current_value=concentration_pct,
                            limit_value=limit.limit_value,
                            excess_amount=concentration_pct - limit.limit_value,
                            severity="breach",
                            timestamp_ms=current_time,
                            action_required=f"Reduce {symbol} position by {float(concentration_pct - limit.limit_value):.1f}%"
                        ))
                        
        return violations
        
    def _create_violation(self, limit: ExposureLimit, current_value: Decimal, 
                         severity: str, timestamp: int) -> LimitViolation:
        """Create a limit violation record."""
        excess = current_value - limit.limit_value
        
        if limit.limit_type == LimitType.POSITION_SIZE:
            action = f"Reduce {limit.identifier} position by ${float(excess):,.0f}"
        elif limit.limit_type == LimitType.SECTOR_EXPOSURE:
            action = f"Reduce {limit.identifier} sector exposure by ${float(excess):,.0f}"
        else:
            action = f"Reduce {limit.limit_type.value} by ${float(excess):,.0f}"
            
        return LimitViolation(
            limit_type=limit.limit_type,
            identifier=limit.identifier,
            current_value=current_value,
            limit_value=limit.limit_value,
            excess_amount=excess,
            severity=severity,
            timestamp_ms=timestamp,
            action_required=action
        )
        
    async def _process_violations(self, violations: List[LimitViolation]):
        """Process limit violations."""
        # Add to violation history
        self.violations.extend(violations)
        
        # Keep only recent violations (last 1000)
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]
            
        # Store critical violations in Redis
        for violation in violations:
            if violation.severity in ["breach", "critical"]:
                key = f"violations:{violation.limit_type.value}:{violation.identifier}"
                await self.redis.hset(key, mapping={
                    "current_value": str(violation.current_value),
                    "limit_value": str(violation.limit_value),
                    "excess_amount": str(violation.excess_amount),
                    "severity": violation.severity,
                    "timestamp_ms": str(violation.timestamp_ms),
                    "action_required": violation.action_required
                })
                await self.redis.expire(key, 86400)  # 24 hours
                
        # Notify callbacks
        for callback in self.violation_callbacks:
            await callback(violations)
            
    def add_violation_callback(self, callback):
        """Add a callback for limit violations."""
        self.violation_callbacks.append(callback)
        
    async def adjust_limit(self, limit_type: LimitType, identifier: str, 
                          new_limit: Decimal, reason: str):
        """Dynamically adjust an exposure limit."""
        key = (limit_type, identifier)
        
        # Store old limit for audit
        old_limit = self.limits.get(key)
        if old_limit:
            await self.redis.hset("limits:audit", 
                f"{limit_type.value}:{identifier}:{int(time.time())}",
                str({
                    "old_limit": str(old_limit.limit_value),
                    "new_limit": str(new_limit),
                    "reason": reason,
                    "timestamp": int(time.time() * 1000)
                })
            )
            
        # Update limit
        self.add_limit(limit_type, identifier, new_limit)
        
        # Store in Redis
        if limit_type == LimitType.POSITION_SIZE:
            await self.redis.hset("limits:positions", identifier, str(new_limit))
        elif limit_type == LimitType.SECTOR_EXPOSURE:
            await self.redis.hset("limits:sectors", identifier, str(new_limit))
        else:
            await self.redis.hset("limits:portfolio", limit_type.value, str(new_limit))
            
        logger.info(f"Adjusted {limit_type.value} limit for {identifier}: "
                   f"{old_limit.limit_value if old_limit else 'N/A'} -> {new_limit} ({reason})")
                   
    async def get_limit_status(self) -> Dict[str, Any]:
        """Get current status of all limits."""
        status = {
            "timestamp": int(time.time() * 1000),
            "limits": [],
            "violations": [],
            "summary": {
                "total_limits": len(self.limits),
                "breached": sum(1 for l in self.limits.values() if l.is_breached),
                "warnings": sum(1 for l in self.limits.values() if l.is_warning),
                "ok": sum(1 for l in self.limits.values() if not l.is_breached and not l.is_warning)
            }
        }
        
        # Add limit details
        for limit in self.limits.values():
            status["limits"].append({
                "type": limit.limit_type.value,
                "identifier": limit.identifier,
                "limit": float(limit.limit_value),
                "current": float(limit.current_value),
                "utilization_pct": limit.utilization_pct,
                "status": "breached" if limit.is_breached else "warning" if limit.is_warning else "ok"
            })
            
        # Add recent violations
        recent_violations = self.violations[-10:] if self.violations else []
        for violation in recent_violations:
            status["violations"].append({
                "type": violation.limit_type.value,
                "identifier": violation.identifier,
                "excess": float(violation.excess_amount),
                "severity": violation.severity,
                "action": violation.action_required,
                "timestamp": violation.timestamp_ms
            })
            
        return status