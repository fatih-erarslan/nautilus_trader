"""Real-time position monitoring with multi-asset aggregation.

Tracks positions, margin requirements, and buying power in real-time.
"""

import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from decimal import Decimal
from collections import defaultdict
import redis.asyncio as redis
from alpaca.trading.models import Position, Account
from alpaca.trading.client import TradingClient
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSummary:
    """Summary of position with calculated metrics."""
    symbol: str
    quantity: int
    side: str
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    margin_requirement: Decimal
    sector: Optional[str] = None
    asset_class: str = "equity"
    last_update_ms: int = 0


@dataclass
class MarginRequirements:
    """Margin requirements for positions."""
    initial_margin: Decimal
    maintenance_margin: Decimal
    buying_power_used: Decimal
    available_buying_power: Decimal
    margin_call_threshold: Decimal
    excess_liquidity: Decimal


class PositionMonitor:
    """Live position tracking with margin monitoring."""
    
    def __init__(self, trading_client: TradingClient, redis_url: str = "redis://localhost:6379"):
        self.client = trading_client
        self.redis: Optional[redis.Redis] = None
        self.redis_url = redis_url
        self.positions: Dict[str, PositionSummary] = {}
        self.account: Optional[Account] = None
        self.sector_map: Dict[str, str] = {}  # Symbol to sector mapping
        self.margin_requirements: Dict[str, Decimal] = {}  # Symbol to margin req
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def initialize(self):
        """Initialize Redis and load initial data."""
        self.redis = await redis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        
        # Load initial positions
        await self.refresh_positions()
        
        # Load sector mappings (in production, from data provider)
        await self._load_sector_mappings()
        
        logger.info("Position monitor initialized")
        
    async def close(self):
        """Clean up resources."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        if self.redis:
            await self.redis.close()
            
    async def start_monitoring(self, update_interval: float = 1.0):
        """Start continuous position monitoring."""
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(update_interval)
        )
        
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._running:
            try:
                await self.refresh_positions()
                await self.update_margin_requirements()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(5)
                
    async def refresh_positions(self):
        """Refresh all positions from Alpaca."""
        start_time = time.perf_counter()
        
        try:
            # Get current positions
            alpaca_positions = self.client.get_all_positions()
            
            # Get account info for margin data
            self.account = self.client.get_account()
            
            # Update position summaries
            current_symbols = set()
            
            for pos in alpaca_positions:
                summary = await self._create_position_summary(pos)
                self.positions[pos.symbol] = summary
                current_symbols.add(pos.symbol)
                
                # Store in Redis
                await self._store_position_redis(summary)
                
            # Remove closed positions
            for symbol in list(self.positions.keys()):
                if symbol not in current_symbols:
                    del self.positions[symbol]
                    await self.redis.delete(f"position:{symbol}")
                    
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > 2:
                logger.warning(f"Position refresh took {elapsed_ms:.2f}ms")
                
        except Exception as e:
            logger.error(f"Error refreshing positions: {e}")
            
    async def _create_position_summary(self, position: Position) -> PositionSummary:
        """Create position summary with calculations."""
        # Calculate market value
        market_value = abs(position.qty) * position.current_price
        
        # Calculate cost basis
        cost_basis = abs(position.qty) * position.avg_entry_price
        
        # Calculate P&L
        if position.side == "long":
            unrealized_pnl = market_value - cost_basis
        else:
            unrealized_pnl = cost_basis - market_value
            
        # Calculate P&L percentage
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else Decimal("0")
        
        # Get margin requirement
        margin_req = await self._calculate_margin_requirement(
            position.symbol, position.qty, position.current_price
        )
        
        # Get sector
        sector = self.sector_map.get(position.symbol, "Unknown")
        
        # Determine asset class
        asset_class = self._determine_asset_class(position.symbol)
        
        return PositionSummary(
            symbol=position.symbol,
            quantity=int(position.qty),
            side=position.side,
            market_value=market_value,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            margin_requirement=margin_req,
            sector=sector,
            asset_class=asset_class,
            last_update_ms=int(time.time() * 1000)
        )
        
    async def _calculate_margin_requirement(self, symbol: str, quantity: Decimal, price: Decimal) -> Decimal:
        """Calculate margin requirement for a position."""
        # Get cached requirement if available
        cached_req = self.margin_requirements.get(symbol)
        if cached_req:
            return abs(quantity) * price * cached_req
            
        # Default margin requirements by asset type
        if self._is_option(symbol):
            # Options have complex margin requirements
            # Simplified: 20% of underlying value
            margin_pct = Decimal("0.20")
        elif symbol in ["SPY", "QQQ", "IWM"]:  # Major ETFs
            margin_pct = Decimal("0.25")
        else:
            # Standard equity margin requirement
            margin_pct = Decimal("0.30")
            
        self.margin_requirements[symbol] = margin_pct
        
        # Store in Redis
        await self.redis.hset("margin:requirements", symbol, str(margin_pct))
        
        return abs(quantity) * price * margin_pct
        
    def _determine_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        if self._is_option(symbol):
            return "option"
        elif symbol.endswith("USD"):
            return "crypto"
        elif len(symbol) <= 5:
            return "equity"
        else:
            return "other"
            
    def _is_option(self, symbol: str) -> bool:
        """Check if symbol is an option."""
        return len(symbol) > 10 and any(c in symbol for c in ['C', 'P'])
        
    async def _load_sector_mappings(self):
        """Load sector mappings from cache or API."""
        # Try to load from Redis
        sectors = await self.redis.hgetall("symbol:sectors")
        if sectors:
            self.sector_map = sectors
            return
            
        # In production, would fetch from data provider
        # For now, use some common mappings
        default_sectors = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "AMZN": "Consumer Discretionary",
            "JPM": "Financials",
            "BAC": "Financials",
            "XOM": "Energy",
            "CVX": "Energy",
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "SPY": "ETF",
            "QQQ": "ETF"
        }
        
        # Store in Redis
        if default_sectors:
            await self.redis.hset("symbol:sectors", mapping=default_sectors)
            await self.redis.expire("symbol:sectors", 86400)  # 24 hours
            
        self.sector_map = default_sectors
        
    async def _store_position_redis(self, summary: PositionSummary):
        """Store position summary in Redis."""
        key = f"position:{summary.symbol}"
        
        await self.redis.hset(key, mapping={
            "quantity": str(summary.quantity),
            "side": summary.side,
            "market_value": str(summary.market_value),
            "cost_basis": str(summary.cost_basis),
            "unrealized_pnl": str(summary.unrealized_pnl),
            "unrealized_pnl_pct": str(summary.unrealized_pnl_pct),
            "margin_requirement": str(summary.margin_requirement),
            "sector": summary.sector or "",
            "asset_class": summary.asset_class,
            "last_update_ms": str(summary.last_update_ms)
        })
        
        await self.redis.expire(key, 3600)  # 1 hour
        
    async def update_margin_requirements(self):
        """Update account margin requirements."""
        if not self.account:
            return
            
        # Calculate total margin used
        total_initial_margin = Decimal("0")
        total_maintenance_margin = Decimal("0")
        
        for summary in self.positions.values():
            # Initial margin (for new positions)
            total_initial_margin += summary.margin_requirement
            
            # Maintenance margin (typically 25% for equities)
            total_maintenance_margin += summary.market_value * Decimal("0.25")
            
        # Calculate margin metrics
        buying_power = Decimal(self.account.buying_power)
        equity = Decimal(self.account.equity)
        
        margin_reqs = MarginRequirements(
            initial_margin=total_initial_margin,
            maintenance_margin=total_maintenance_margin,
            buying_power_used=equity - buying_power,
            available_buying_power=buying_power,
            margin_call_threshold=total_maintenance_margin * Decimal("1.2"),
            excess_liquidity=equity - total_maintenance_margin
        )
        
        # Store in Redis
        await self.redis.hset("margin:account", mapping={
            "initial_margin": str(margin_reqs.initial_margin),
            "maintenance_margin": str(margin_reqs.maintenance_margin),
            "buying_power_used": str(margin_reqs.buying_power_used),
            "available_buying_power": str(margin_reqs.available_buying_power),
            "margin_call_threshold": str(margin_reqs.margin_call_threshold),
            "excess_liquidity": str(margin_reqs.excess_liquidity),
            "timestamp": str(int(time.time() * 1000))
        })
        
    async def get_aggregated_view(self) -> Dict[str, Any]:
        """Get aggregated position view by sector and asset class."""
        start_time = time.perf_counter()
        
        # Aggregate by sector
        sector_exposure = defaultdict(lambda: {
            "market_value": Decimal("0"),
            "unrealized_pnl": Decimal("0"),
            "positions": 0
        })
        
        # Aggregate by asset class
        asset_class_exposure = defaultdict(lambda: {
            "market_value": Decimal("0"),
            "unrealized_pnl": Decimal("0"),
            "positions": 0
        })
        
        # Total metrics
        total_market_value = Decimal("0")
        total_unrealized_pnl = Decimal("0")
        
        for summary in self.positions.values():
            # Sector aggregation
            sector = summary.sector or "Unknown"
            sector_exposure[sector]["market_value"] += summary.market_value
            sector_exposure[sector]["unrealized_pnl"] += summary.unrealized_pnl
            sector_exposure[sector]["positions"] += 1
            
            # Asset class aggregation
            asset_class_exposure[summary.asset_class]["market_value"] += summary.market_value
            asset_class_exposure[summary.asset_class]["unrealized_pnl"] += summary.unrealized_pnl
            asset_class_exposure[summary.asset_class]["positions"] += 1
            
            # Totals
            total_market_value += summary.market_value
            total_unrealized_pnl += summary.unrealized_pnl
            
        # Calculate concentration metrics
        concentration_by_sector = {
            sector: float(data["market_value"] / total_market_value * 100) if total_market_value > 0 else 0
            for sector, data in sector_exposure.items()
        }
        
        aggregated = {
            "total_positions": len(self.positions),
            "total_market_value": float(total_market_value),
            "total_unrealized_pnl": float(total_unrealized_pnl),
            "sector_exposure": {
                sector: {
                    "market_value": float(data["market_value"]),
                    "unrealized_pnl": float(data["unrealized_pnl"]),
                    "positions": data["positions"],
                    "concentration_pct": concentration_by_sector[sector]
                }
                for sector, data in sector_exposure.items()
            },
            "asset_class_exposure": {
                asset_class: {
                    "market_value": float(data["market_value"]),
                    "unrealized_pnl": float(data["unrealized_pnl"]),
                    "positions": data["positions"]
                }
                for asset_class, data in asset_class_exposure.items()
            },
            "calculation_time_ms": (time.perf_counter() - start_time) * 1000
        }
        
        # Store in Redis
        await self.redis.set(
            "position:aggregated",
            str(aggregated),
            ex=60  # 1 minute cache
        )
        
        return aggregated
        
    async def check_concentration_limits(self) -> List[Dict[str, Any]]:
        """Check for concentration limit violations."""
        violations = []
        
        # Get aggregated view
        aggregated = await self.get_aggregated_view()
        total_value = aggregated["total_market_value"]
        
        if total_value == 0:
            return violations
            
        # Check single position concentration
        for symbol, summary in self.positions.items():
            position_pct = float(summary.market_value / total_value * 100)
            
            # Single position limit: 10%
            if position_pct > 10:
                violations.append({
                    "type": "position_concentration",
                    "symbol": symbol,
                    "current_pct": position_pct,
                    "limit_pct": 10,
                    "severity": "high" if position_pct > 15 else "medium"
                })
                
        # Check sector concentration
        for sector, data in aggregated["sector_exposure"].items():
            sector_pct = data["concentration_pct"]
            
            # Sector limit: 30%
            if sector_pct > 30:
                violations.append({
                    "type": "sector_concentration",
                    "sector": sector,
                    "current_pct": sector_pct,
                    "limit_pct": 30,
                    "severity": "high" if sector_pct > 40 else "medium"
                })
                
        return violations