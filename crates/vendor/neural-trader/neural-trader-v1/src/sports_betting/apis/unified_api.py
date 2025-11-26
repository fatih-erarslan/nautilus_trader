"""
Unified Sports Betting API Layer

Provides a single interface for multiple sports betting providers with:
- Provider abstraction and consistent API
- Intelligent failover and redundancy
- Data normalization across different providers  
- Load balancing and performance optimization
- Unified data models and response formats
- Advanced caching and request optimization

Author: Agent 1 - Sports Betting API Integration  
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib
import statistics

from .the_odds_api import TheOddsAPI, Sport as TheOddsSport, Event as TheOddsEvent, ArbitrageOpportunity
from .betfair_api import BetfairAPI, MarketBook, BetDetails, Position

logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Supported betting data providers."""
    THE_ODDS_API = "the_odds_api"
    BETFAIR = "betfair"
    PINNACLE = "pinnacle"  # Future implementation
    SMARKETS = "smarkets"  # Future implementation

class DataType(Enum):
    """Types of betting data."""
    ODDS = "odds"
    EVENTS = "events"
    MARKETS = "markets"
    POSITIONS = "positions"
    ORDERS = "orders"
    BALANCE = "balance"

class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    MAINTENANCE = "maintenance"

@dataclass
class UnifiedEvent:
    """Normalized event data structure."""
    id: str
    provider: str
    sport: str
    league: str
    home_team: str
    away_team: str
    commence_time: datetime
    status: str = "upcoming"
    score: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class UnifiedOdds:
    """Normalized odds data structure."""
    event_id: str
    provider: str
    bookmaker: str
    market_type: str
    selection: str
    odds: float
    size: Optional[float] = None
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class UnifiedPosition:
    """Normalized position data structure."""
    id: str
    provider: str
    market_id: str
    selection_id: str
    side: str  # "back" or "lay"
    stake: float
    price: float
    matched: float
    remaining: float
    pnl: float
    commission: float
    status: str

@dataclass
class ProviderConfig:
    """Configuration for a betting provider."""
    provider_type: ProviderType
    enabled: bool
    priority: int
    weight: float
    config: Dict
    rate_limit: float
    timeout: float
    retry_count: int

@dataclass
class ProviderHealth:
    """Health status of a provider."""
    provider: ProviderType
    status: ProviderStatus
    last_check: datetime
    response_time: float
    success_rate: float
    error_count: int
    total_requests: int
    last_error: Optional[str] = None

class LoadBalancer:
    """Intelligent load balancer for multiple providers."""
    
    def __init__(self):
        self.provider_weights: Dict[ProviderType, float] = {}
        self.provider_performance: Dict[ProviderType, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_used: Dict[ProviderType, datetime] = {}
        
    def update_performance(self, provider: ProviderType, response_time: float, success: bool):
        """Update provider performance metrics."""
        self.provider_performance[provider].append({
            'response_time': response_time,
            'success': success,
            'timestamp': datetime.now()
        })
        self.last_used[provider] = datetime.now()
    
    def get_best_provider(self, providers: List[ProviderType], 
                         exclude: List[ProviderType] = None) -> Optional[ProviderType]:
        """Select best provider based on performance and availability."""
        exclude = exclude or []
        available_providers = [p for p in providers if p not in exclude]
        
        if not available_providers:
            return None
        
        # Score providers based on multiple factors
        provider_scores = {}
        
        for provider in available_providers:
            performance_data = list(self.provider_performance[provider])
            
            if not performance_data:
                # No performance data - give neutral score
                provider_scores[provider] = 0.5
                continue
            
            # Calculate success rate
            recent_data = [p for p in performance_data if (datetime.now() - p['timestamp']).seconds < 300]
            if recent_data:
                success_rate = sum(1 for p in recent_data if p['success']) / len(recent_data)
                avg_response_time = statistics.mean(p['response_time'] for p in recent_data)
            else:
                success_rate = sum(1 for p in performance_data if p['success']) / len(performance_data)
                avg_response_time = statistics.mean(p['response_time'] for p in performance_data)
            
            # Score based on success rate and response time
            time_score = max(0, 1 - (avg_response_time / 5.0))  # Penalize slow responses
            overall_score = (success_rate * 0.7) + (time_score * 0.3)
            
            # Apply provider weight
            weight = self.provider_weights.get(provider, 1.0)
            provider_scores[provider] = overall_score * weight
        
        # Return provider with highest score
        return max(provider_scores.items(), key=lambda x: x[1])[0]

class DataNormalizer:
    """Normalize data across different providers."""
    
    @staticmethod
    def normalize_event(event_data: Any, provider: ProviderType) -> UnifiedEvent:
        """Normalize event data from any provider."""
        if provider == ProviderType.THE_ODDS_API:
            return DataNormalizer._normalize_theoddsapi_event(event_data)
        elif provider == ProviderType.BETFAIR:
            return DataNormalizer._normalize_betfair_event(event_data)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def _normalize_theoddsapi_event(event: TheOddsEvent) -> UnifiedEvent:
        """Normalize TheOddsAPI event."""
        return UnifiedEvent(
            id=event.id,
            provider=ProviderType.THE_ODDS_API.value,
            sport=event.sport,
            league="unknown",  # TheOddsAPI doesn't provide league info directly
            home_team=event.home_team,
            away_team=event.away_team,
            commence_time=event.commence_time,
            status="completed" if event.completed else "upcoming",
            score=event.score,
            metadata={}
        )
    
    @staticmethod
    def _normalize_betfair_event(event_data: Dict) -> UnifiedEvent:
        """Normalize Betfair event data."""
        return UnifiedEvent(
            id=event_data.get("event", {}).get("id", ""),
            provider=ProviderType.BETFAIR.value,
            sport=event_data.get("eventType", {}).get("name", ""),
            league=event_data.get("competition", {}).get("name", ""),
            home_team=event_data.get("event", {}).get("name", "").split(" v ")[0] if " v " in event_data.get("event", {}).get("name", "") else "",
            away_team=event_data.get("event", {}).get("name", "").split(" v ")[1] if " v " in event_data.get("event", {}).get("name", "") else "",
            commence_time=datetime.fromisoformat(event_data.get("event", {}).get("openDate", "").replace('Z', '+00:00')) if event_data.get("event", {}).get("openDate") else datetime.now(),
            status="upcoming",
            metadata=event_data
        )
    
    @staticmethod
    def normalize_odds(odds_data: Any, provider: ProviderType, event_id: str) -> List[UnifiedOdds]:
        """Normalize odds data from any provider."""
        if provider == ProviderType.THE_ODDS_API:
            return DataNormalizer._normalize_theoddsapi_odds(odds_data, event_id)
        elif provider == ProviderType.BETFAIR:
            return DataNormalizer._normalize_betfair_odds(odds_data, event_id)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod 
    def _normalize_theoddsapi_odds(event: TheOddsEvent, event_id: str) -> List[UnifiedOdds]:
        """Normalize TheOddsAPI odds."""
        unified_odds = []
        for odds in event.odds:
            unified_odds.append(UnifiedOdds(
                event_id=event_id,
                provider=ProviderType.THE_ODDS_API.value,
                bookmaker=odds.bookmaker,
                market_type=odds.market,
                selection=odds.selection,
                odds=odds.odds,
                last_update=odds.last_update
            ))
        return unified_odds
    
    @staticmethod
    def _normalize_betfair_odds(market_book: MarketBook, event_id: str) -> List[UnifiedOdds]:
        """Normalize Betfair odds."""
        unified_odds = []
        for runner in market_book.runners:
            # Add back prices
            for price_size in runner.back_prices:
                unified_odds.append(UnifiedOdds(
                    event_id=event_id,
                    provider=ProviderType.BETFAIR.value,
                    bookmaker="betfair",
                    market_type="back",
                    selection=str(runner.selection_id),
                    odds=price_size.price,
                    size=price_size.size,
                    last_update=datetime.now()
                ))
            
            # Add lay prices
            for price_size in runner.lay_prices:
                unified_odds.append(UnifiedOdds(
                    event_id=event_id,
                    provider=ProviderType.BETFAIR.value,
                    bookmaker="betfair",
                    market_type="lay",
                    selection=str(runner.selection_id),
                    odds=price_size.price,
                    size=price_size.size,
                    last_update=datetime.now()
                ))
        
        return unified_odds

class UnifiedSportsAPI:
    """
    Unified sports betting API providing abstraction over multiple providers.
    
    Features:
    - Single interface for multiple betting providers
    - Intelligent failover and load balancing
    - Data normalization and consistent responses
    - Advanced caching and performance optimization
    - Health monitoring and auto-recovery
    - Arbitrage detection across providers
    """
    
    def __init__(self, provider_configs: List[ProviderConfig]):
        self.provider_configs = {config.provider_type: config for config in provider_configs}
        self.providers: Dict[ProviderType, Any] = {}
        self.provider_health: Dict[ProviderType, ProviderHealth] = {}
        
        # Load balancing and failover
        self.load_balancer = LoadBalancer()
        self.data_normalizer = DataNormalizer()
        
        # Caching
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.default_cache_duration = timedelta(minutes=5)
        
        # Performance tracking
        self.request_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0, 'success': 0, 'total_time': 0.0, 'errors': []
        })
        
        logger.info("UnifiedSportsAPI initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_providers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize_providers(self):
        """Initialize all configured providers."""
        for provider_type, config in self.provider_configs.items():
            if not config.enabled:
                continue
            
            try:
                if provider_type == ProviderType.THE_ODDS_API:
                    provider = TheOddsAPI(
                        api_key=config.config['api_key'],
                        enable_websocket=config.config.get('enable_websocket', True)
                    )
                    self.providers[provider_type] = await provider.__aenter__()
                    
                elif provider_type == ProviderType.BETFAIR:
                    provider = BetfairAPI(
                        app_key=config.config['app_key'],
                        username=config.config['username'],
                        password=config.config['password'],
                        cert_file=config.config.get('cert_file')
                    )
                    self.providers[provider_type] = await provider.__aenter__()
                
                # Initialize health monitoring
                self.provider_health[provider_type] = ProviderHealth(
                    provider=provider_type,
                    status=ProviderStatus.HEALTHY,
                    last_check=datetime.now(),
                    response_time=0.0,
                    success_rate=1.0,
                    error_count=0,
                    total_requests=0
                )
                
                # Set load balancer weights
                self.load_balancer.provider_weights[provider_type] = config.weight
                
                logger.info(f"Initialized provider: {provider_type.value}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type.value}: {e}")
                self.provider_health[provider_type] = ProviderHealth(
                    provider=provider_type,
                    status=ProviderStatus.DOWN,
                    last_check=datetime.now(),
                    response_time=0.0,
                    success_rate=0.0,
                    error_count=1,
                    total_requests=1,
                    last_error=str(e)
                )
    
    async def cleanup(self):
        """Clean up all providers."""
        for provider_type, provider in self.providers.items():
            try:
                if hasattr(provider, '__aexit__'):
                    await provider.__aexit__(None, None, None)
                elif hasattr(provider, 'cleanup'):
                    await provider.cleanup()
                logger.info(f"Cleaned up provider: {provider_type.value}")
            except Exception as e:
                logger.error(f"Error cleaning up {provider_type.value}: {e}")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for request."""
        key_data = f"{method}:{json.dumps(kwargs, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_ttl:
            return False
        return datetime.now() < self.cache_ttl[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any, ttl: Optional[timedelta] = None):
        """Cache result with TTL."""
        self.cache[cache_key] = result
        self.cache_ttl[cache_key] = datetime.now() + (ttl or self.default_cache_duration)
    
    async def _execute_with_failover(self, method_name: str, data_type: DataType,
                                    provider_method: Callable, *args, **kwargs) -> Any:
        """Execute method with intelligent failover."""
        cache_key = self._get_cache_key(method_name, *args, **kwargs)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for {method_name}")
            return self.cache[cache_key]
        
        # Get available providers in order of preference
        available_providers = [
            p for p, health in self.provider_health.items()
            if health.status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]
            and p in self.providers
        ]
        
        errors = []
        
        for attempt in range(3):  # Max 3 attempts across providers
            provider_type = self.load_balancer.get_best_provider(
                available_providers, 
                exclude=[p for p, errs in errors]
            )
            
            if not provider_type:
                break
            
            provider = self.providers[provider_type]
            start_time = time.time()
            
            try:
                # Execute provider method
                result = await provider_method(provider, *args, **kwargs)
                
                # Record success
                response_time = time.time() - start_time
                self.load_balancer.update_performance(provider_type, response_time, True)
                self._update_health_metrics(provider_type, response_time, True)
                
                # Cache successful result
                self._cache_result(cache_key, result)
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                self.load_balancer.update_performance(provider_type, response_time, False)
                self._update_health_metrics(provider_type, response_time, False, str(e))
                
                errors.append((provider_type, str(e)))
                logger.warning(f"Provider {provider_type.value} failed: {e}")
        
        # All providers failed
        error_msg = f"All providers failed for {method_name}: {errors}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    def _update_health_metrics(self, provider_type: ProviderType, 
                              response_time: float, success: bool, error: str = None):
        """Update provider health metrics."""
        health = self.provider_health[provider_type]
        health.last_check = datetime.now()
        health.response_time = response_time
        health.total_requests += 1
        
        if success:
            health.success_rate = (health.success_rate * (health.total_requests - 1) + 1) / health.total_requests
        else:
            health.error_count += 1
            health.last_error = error
            health.success_rate = (health.success_rate * (health.total_requests - 1)) / health.total_requests
        
        # Update status based on metrics
        if health.success_rate > 0.95:
            health.status = ProviderStatus.HEALTHY
        elif health.success_rate > 0.8:
            health.status = ProviderStatus.DEGRADED
        else:
            health.status = ProviderStatus.DOWN
    
    async def get_events(self, sport: str, days_ahead: int = 7) -> List[UnifiedEvent]:
        """Get upcoming events across all providers."""
        async def provider_method(provider, sport, days_ahead):
            if isinstance(provider, TheOddsAPI):
                events = await provider.get_events(sport, days_ahead)
                return [self.data_normalizer.normalize_event(event, ProviderType.THE_ODDS_API) for event in events]
            elif isinstance(provider, BetfairAPI):
                # Betfair implementation would go here
                return []
            else:
                return []
        
        return await self._execute_with_failover(
            "get_events", DataType.EVENTS, provider_method, sport, days_ahead
        )
    
    async def get_odds(self, sport: str, markets: List[str] = None, 
                      regions: List[str] = None) -> List[UnifiedOdds]:
        """Get odds across all providers with normalization."""
        async def provider_method(provider, sport, markets, regions):
            if isinstance(provider, TheOddsAPI):
                events = await provider.get_odds(sport, markets, regions)
                unified_odds = []
                for event in events:
                    odds_list = self.data_normalizer.normalize_odds(event, ProviderType.THE_ODDS_API, event.id)
                    unified_odds.extend(odds_list)
                return unified_odds
            elif isinstance(provider, BetfairAPI):
                # Betfair implementation would go here
                return []
            else:
                return []
        
        return await self._execute_with_failover(
            "get_odds", DataType.ODDS, provider_method, sport, markets, regions
        )
    
    async def find_arbitrage_opportunities(self, sport: str, min_profit_margin: float = 0.01) -> List[Dict]:
        """Find arbitrage opportunities across all providers."""
        # Get odds from all providers
        all_odds = []
        for provider_type, provider in self.providers.items():
            try:
                if isinstance(provider, TheOddsAPI):
                    events = await provider.get_odds(sport)
                    provider_arbitrage = provider.find_arbitrage_opportunities(events, min_profit_margin)
                    for arb in provider_arbitrage:
                        arb_dict = asdict(arb)
                        arb_dict['provider'] = provider_type.value
                        all_odds.append(arb_dict)
            except Exception as e:
                logger.error(f"Error getting arbitrage from {provider_type.value}: {e}")
        
        # Sort by profit margin
        return sorted(all_odds, key=lambda x: x['profit_margin'], reverse=True)
    
    async def place_bet(self, provider: ProviderType, market_id: str, 
                       selection_id: str, side: str, price: float, size: float) -> Dict:
        """Place bet through specified provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not available")
        
        provider_instance = self.providers[provider]
        
        if isinstance(provider_instance, BetfairAPI):
            # Implement Betfair bet placement
            from .betfair_api import PlaceInstruction, OrderType, Side as BetfairSide
            
            instruction = PlaceInstruction(
                order_type=OrderType.LIMIT,
                selection_id=int(selection_id),
                handicap=0.0,
                side=BetfairSide.BACK if side.lower() == 'back' else BetfairSide.LAY,
                limit_order={"price": price, "size": size}
            )
            
            result = await provider_instance.place_orders(market_id, [instruction])
            return result
        else:
            raise ValueError(f"Betting not supported for provider {provider.value}")
    
    async def get_positions(self, provider: ProviderType) -> List[UnifiedPosition]:
        """Get current positions from specified provider."""
        if provider not in self.providers:
            return []
        
        provider_instance = self.providers[provider]
        
        if isinstance(provider_instance, BetfairAPI):
            orders = await provider_instance.list_current_orders()
            positions = []
            
            for order in orders:
                position = UnifiedPosition(
                    id=order.bet_id,
                    provider=provider.value,
                    market_id="",  # Would need to track this
                    selection_id=str(order.bet_id),  # Simplified
                    side="back" if order.side.value == "B" else "lay",
                    stake=order.size,
                    price=order.price,
                    matched=order.size_matched,
                    remaining=order.size_remaining,
                    pnl=0.0,  # Would calculate based on current prices
                    commission=0.0,  # Would get from Betfair
                    status=order.status.value
                )
                positions.append(position)
            
            return positions
        
        return []
    
    def get_provider_health(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        health_data = {}
        
        for provider_type, health in self.provider_health.items():
            health_data[provider_type.value] = {
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "response_time": health.response_time,
                "success_rate": health.success_rate,
                "error_count": health.error_count,
                "total_requests": health.total_requests,
                "last_error": health.last_error
            }
        
        return health_data
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = {
            "total_entries": len(self.cache),
            "hit_rate": 0.0,  # Would track cache hits vs misses
            "memory_usage": sum(len(str(v)) for v in self.cache.values())
        }
        
        provider_stats = {}
        for provider_type, provider in self.providers.items():
            if hasattr(provider, 'get_performance_metrics'):
                provider_stats[provider_type.value] = provider.get_performance_metrics()
        
        return {
            "cache_stats": cache_stats,
            "provider_health": self.get_provider_health(),
            "provider_performance": provider_stats,
            "load_balancer_weights": {
                p.value: w for p, w in self.load_balancer.provider_weights.items()
            }
        }

# Example usage and configuration
async def main():
    """Example usage of UnifiedSportsAPI."""
    
    # Configure providers
    provider_configs = [
        ProviderConfig(
            provider_type=ProviderType.THE_ODDS_API,
            enabled=True,
            priority=1,
            weight=1.0,
            config={
                'api_key': 'YOUR_THEODDS_API_KEY',
                'enable_websocket': True
            },
            rate_limit=10.0,
            timeout=30.0,
            retry_count=3
        ),
        ProviderConfig(
            provider_type=ProviderType.BETFAIR,
            enabled=True,
            priority=2,
            weight=1.5,  # Higher weight for exchange data
            config={
                'app_key': 'YOUR_BETFAIR_APP_KEY',
                'username': 'YOUR_USERNAME',
                'password': 'YOUR_PASSWORD',
                'cert_file': 'path/to/betfair.crt'
            },
            rate_limit=5.0,
            timeout=30.0,
            retry_count=2
        )
    ]
    
    # Initialize unified API
    async with UnifiedSportsAPI(provider_configs) as unified_api:
        try:
            # Get events
            events = await unified_api.get_events("americanfootball_nfl", days_ahead=7)
            print(f"Found {len(events)} NFL events")
            
            # Get odds
            odds = await unified_api.get_odds("basketball_nba")
            print(f"Found {len(odds)} NBA odds entries")
            
            # Find arbitrage opportunities
            arbitrage = await unified_api.find_arbitrage_opportunities("soccer_epl")
            print(f"Found {len(arbitrage)} arbitrage opportunities")
            
            if arbitrage:
                best_arb = arbitrage[0]
                print(f"Best arbitrage: {best_arb['profit_margin']:.2%} profit")
            
            # Check provider health
            health = unified_api.get_provider_health()
            for provider, status in health.items():
                print(f"{provider}: {status['status']} ({status['success_rate']:.1%} success)")
            
            # Get performance metrics
            metrics = unified_api.get_performance_metrics()
            print(f"Cache entries: {metrics['cache_stats']['total_entries']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())