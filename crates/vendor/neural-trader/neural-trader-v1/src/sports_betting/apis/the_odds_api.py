"""
TheOddsAPI Integration for Real-Time Sports Betting Data

Comprehensive integration with TheOddsAPI providing:
- Real-time odds fetching for major sports
- WebSocket streaming for live updates
- Advanced rate limiting and error handling
- Historical odds tracking
- Market depth analysis
- Arbitrage opportunity detection

Author: Agent 1 - Sports Betting API Integration
"""

import asyncio
import json
import logging
import time
import aiohttp
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib
import hmac
import base64

logger = logging.getLogger(__name__)

class Sport(Enum):
    """Supported sports for betting data."""
    AMERICANFOOTBALL_NFL = "americanfootball_nfl"
    BASKETBALL_NBA = "basketball_nba"
    BASKETBALL_NCAA = "basketball_ncaab"
    BASEBALL_MLB = "baseball_mlb"
    ICEHOCKEY_NHL = "icehockey_nhl"
    SOCCER_EPL = "soccer_epl"
    SOCCER_UEFA_CHAMPS_LEAGUE = "soccer_uefa_champs_league"
    SOCCER_FIFA_WORLD_CUP = "soccer_fifa_world_cup"
    TENNIS_ATP = "tennis_atp"
    TENNIS_WTA = "tennis_wta"
    BOXING = "boxing_boxing"
    MMA = "mixed_martial_arts_ufc"

class Market(Enum):
    """Supported betting markets."""
    H2H = "h2h"  # Head to head / moneyline
    SPREADS = "spreads"  # Point spreads
    TOTALS = "totals"  # Over/under totals
    OUTRIGHTS = "outrights"  # Tournament winners

class Region(Enum):
    """Supported regions for odds."""
    US = "us"
    UK = "uk" 
    AU = "au"
    EU = "eu"

@dataclass
class Odds:
    """Individual odds data structure."""
    bookmaker: str
    market: str
    selection: str
    odds: float
    point: Optional[float] = None
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()

@dataclass
class Event:
    """Sports event data structure."""
    id: str
    sport: str
    commence_time: datetime
    home_team: str
    away_team: str
    odds: List[Odds]
    completed: bool = False
    score: Optional[Dict] = None
    
@dataclass
class ArbitrageOpportunity:
    """Arbitrage betting opportunity."""
    event_id: str
    sport: str
    home_team: str
    away_team: str
    bookmaker_1: str
    bookmaker_2: str
    odds_1: float
    odds_2: float
    profit_margin: float
    required_stake_1: float
    required_stake_2: float
    total_stake: float
    guaranteed_profit: float
    confidence: float

class RateLimiter:
    """Advanced rate limiting with burst capacity and adaptive throttling."""
    
    def __init__(self, requests_per_second: float = 10, burst_capacity: int = 50):
        self.requests_per_second = requests_per_second
        self.burst_capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_update = time.time()
        self.request_times = deque(maxlen=100)
        
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        current_time = time.time()
        
        # Refill tokens based on time passed
        time_passed = current_time - self.last_update
        self.tokens = min(
            self.burst_capacity,
            self.tokens + time_passed * self.requests_per_second
        )
        self.last_update = current_time
        
        # Check if we have tokens available
        if self.tokens >= 1:
            self.tokens -= 1
            self.request_times.append(current_time)
            return True
        
        # Calculate wait time
        wait_time = (1 - self.tokens) / self.requests_per_second
        await asyncio.sleep(wait_time)
        
        self.tokens = 0
        self.request_times.append(time.time())
        return True
    
    def get_current_rate(self) -> float:
        """Get current request rate per second."""
        if len(self.request_times) < 2:
            return 0.0
        
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t <= 60]
        return len(recent_requests) / 60.0

class CircuitBreaker:
    """Circuit breaker pattern for API resilience."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class TheOddsAPI:
    """
    Comprehensive TheOddsAPI integration with advanced features.
    
    Features:
    - Real-time odds fetching with caching
    - WebSocket streaming for live updates
    - Advanced rate limiting and circuit breaker
    - Historical odds tracking and analysis
    - Arbitrage opportunity detection
    - Market depth analysis
    - Multi-sport support
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    def __init__(self, api_key: str, enable_websocket: bool = True):
        self.api_key = api_key
        self.enable_websocket = enable_websocket
        
        # Rate limiting and reliability
        self.rate_limiter = RateLimiter(requests_per_second=5)  # Conservative rate
        self.circuit_breaker = CircuitBreaker()
        
        # Data storage
        self.events_cache: Dict[str, Event] = {}
        self.odds_history: Dict[str, List[Odds]] = defaultdict(list)
        self.bookmaker_reliability: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # WebSocket management
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        logger.info("TheOddsAPI client initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make rate-limited API request with error handling."""
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open - API temporarily unavailable")
        
        await self.rate_limiter.acquire()
        
        url = f"{self.BASE_URL}/{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key
        
        try:
            self.request_count += 1
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    self.circuit_breaker.record_success()
                    return data
                elif response.status == 429:
                    # Rate limit exceeded - adaptive backoff
                    await asyncio.sleep(60)
                    return await self._make_request(endpoint, params)
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
                    
        except Exception as e:
            self.error_count += 1
            self.circuit_breaker.record_failure()
            logger.error(f"API request failed: {e}")
            raise
    
    async def get_sports(self) -> List[Dict]:
        """Get list of available sports."""
        try:
            return await self._make_request("sports")
        except Exception as e:
            logger.error(f"Failed to get sports: {e}")
            return []
    
    async def get_events(self, sport: Union[Sport, str], days_ahead: int = 7) -> List[Event]:
        """
        Get upcoming events for a sport.
        
        Args:
            sport: Sport identifier or Sport enum
            days_ahead: How many days ahead to fetch events
            
        Returns:
            List of Event objects
        """
        sport_key = sport.value if isinstance(sport, Sport) else sport
        
        try:
            params = {
                'sport': sport_key,
                'daysFrom': days_ahead
            }
            
            data = await self._make_request("sports/{}/events".format(sport_key), params)
            
            events = []
            for event_data in data:
                event = Event(
                    id=event_data['id'],
                    sport=event_data['sport_key'],
                    commence_time=datetime.fromisoformat(event_data['commence_time'].replace('Z', '+00:00')),
                    home_team=event_data['home_team'],
                    away_team=event_data['away_team'],
                    odds=[]
                )
                events.append(event)
                self.events_cache[event.id] = event
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get events for {sport_key}: {e}")
            return []
    
    async def get_odds(self, sport: Union[Sport, str], markets: List[Union[Market, str]] = None,
                      regions: List[Union[Region, str]] = None, bookmakers: List[str] = None) -> List[Event]:
        """
        Get odds for events in a sport.
        
        Args:
            sport: Sport identifier
            markets: List of markets to fetch (h2h, spreads, totals)
            regions: List of regions for odds
            bookmakers: Specific bookmakers to include
            
        Returns:
            List of Event objects with odds data
        """
        sport_key = sport.value if isinstance(sport, Sport) else sport
        
        if markets is None:
            markets = [Market.H2H, Market.SPREADS, Market.TOTALS]
        if regions is None:
            regions = [Region.US]
        
        # Convert enums to strings
        market_keys = [m.value if isinstance(m, Market) else m for m in markets]
        region_keys = [r.value if isinstance(r, Region) else r for r in regions]
        
        try:
            params = {
                'sport': sport_key,
                'markets': ','.join(market_keys),
                'regions': ','.join(region_keys),
                'oddsFormat': 'decimal',
                'dateFormat': 'iso'
            }
            
            if bookmakers:
                params['bookmakers'] = ','.join(bookmakers)
            
            data = await self._make_request(f"sports/{sport_key}/odds", params)
            
            events = []
            for event_data in data:
                odds_list = []
                
                for bookmaker_data in event_data.get('bookmakers', []):
                    bookmaker = bookmaker_data['key']
                    
                    for market_data in bookmaker_data.get('markets', []):
                        market = market_data['key']
                        
                        for outcome in market_data.get('outcomes', []):
                            odds = Odds(
                                bookmaker=bookmaker,
                                market=market,
                                selection=outcome['name'],
                                odds=outcome['price'],
                                point=outcome.get('point')
                            )
                            odds_list.append(odds)
                            
                            # Store in history for trend analysis
                            odds_key = f"{event_data['id']}_{bookmaker}_{market}_{outcome['name']}"
                            self.odds_history[odds_key].append(odds)
                
                event = Event(
                    id=event_data['id'],
                    sport=event_data['sport_key'],
                    commence_time=datetime.fromisoformat(event_data['commence_time'].replace('Z', '+00:00')),
                    home_team=event_data['home_team'],
                    away_team=event_data['away_team'],
                    odds=odds_list
                )
                
                events.append(event)
                self.events_cache[event.id] = event
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to get odds for {sport_key}: {e}")
            return []
    
    async def get_historical_odds(self, event_id: str, date: datetime) -> List[Odds]:
        """Get historical odds for a specific event and date."""
        try:
            params = {
                'date': date.isoformat()
            }
            
            data = await self._make_request(f"historical/sports/odds", params)
            
            odds_list = []
            for event_data in data:
                if event_data['id'] == event_id:
                    for bookmaker_data in event_data.get('bookmakers', []):
                        bookmaker = bookmaker_data['key']
                        
                        for market_data in bookmaker_data.get('markets', []):
                            market = market_data['key']
                            
                            for outcome in market_data.get('outcomes', []):
                                odds = Odds(
                                    bookmaker=bookmaker,
                                    market=market,
                                    selection=outcome['name'],
                                    odds=outcome['price'],
                                    point=outcome.get('point'),
                                    last_update=date
                                )
                                odds_list.append(odds)
            
            return odds_list
            
        except Exception as e:
            logger.error(f"Failed to get historical odds: {e}")
            return []
    
    def find_arbitrage_opportunities(self, events: List[Event], min_profit_margin: float = 0.01) -> List[ArbitrageOpportunity]:
        """
        Find arbitrage opportunities across bookmakers.
        
        Args:
            events: List of events with odds
            min_profit_margin: Minimum profit margin to consider (1% default)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        for event in events:
            # Group odds by market
            market_odds = defaultdict(lambda: defaultdict(list))
            
            for odds in event.odds:
                market_odds[odds.market][odds.selection].append(odds)
            
            # Check each market for arbitrage
            for market, selections in market_odds.items():
                if market == Market.H2H.value and len(selections) >= 2:
                    # For head-to-head markets, check all bookmaker combinations
                    selection_names = list(selections.keys())
                    
                    if len(selection_names) >= 2:
                        sel1, sel2 = selection_names[0], selection_names[1]
                        
                        for odds1 in selections[sel1]:
                            for odds2 in selections[sel2]:
                                if odds1.bookmaker != odds2.bookmaker:
                                    # Calculate arbitrage
                                    arb = self._calculate_arbitrage(
                                        event, odds1, odds2, min_profit_margin
                                    )
                                    if arb:
                                        opportunities.append(arb)
        
        return sorted(opportunities, key=lambda x: x.profit_margin, reverse=True)
    
    def _calculate_arbitrage(self, event: Event, odds1: Odds, odds2: Odds, 
                           min_profit_margin: float) -> Optional[ArbitrageOpportunity]:
        """Calculate arbitrage opportunity between two odds."""
        try:
            # Calculate implied probabilities
            prob1 = 1.0 / odds1.odds
            prob2 = 1.0 / odds2.odds
            
            # Check if arbitrage exists
            total_prob = prob1 + prob2
            if total_prob >= 1.0:
                return None
            
            profit_margin = (1.0 - total_prob)
            if profit_margin < min_profit_margin:
                return None
            
            # Calculate optimal stakes for $1000 total investment
            total_stake = 1000.0
            stake1 = total_stake * prob1 / total_prob
            stake2 = total_stake * prob2 / total_prob
            
            # Calculate guaranteed profit
            payout1 = stake1 * odds1.odds
            payout2 = stake2 * odds2.odds
            guaranteed_profit = min(payout1, payout2) - total_stake
            
            # Calculate confidence based on bookmaker reliability
            confidence = (
                self.bookmaker_reliability[odds1.bookmaker] * 
                self.bookmaker_reliability[odds2.bookmaker]
            )
            
            return ArbitrageOpportunity(
                event_id=event.id,
                sport=event.sport,
                home_team=event.home_team,
                away_team=event.away_team,
                bookmaker_1=odds1.bookmaker,
                bookmaker_2=odds2.bookmaker,
                odds_1=odds1.odds,
                odds_2=odds2.odds,
                profit_margin=profit_margin,
                required_stake_1=stake1,
                required_stake_2=stake2,
                total_stake=total_stake,
                guaranteed_profit=guaranteed_profit,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating arbitrage: {e}")
            return None
    
    async def start_websocket_streaming(self, sports: List[Union[Sport, str]], 
                                       callback: Callable[[Dict], None]):
        """
        Start WebSocket streaming for real-time odds updates.
        
        Args:
            sports: List of sports to stream
            callback: Function to call with updates
        """
        if not self.enable_websocket:
            logger.warning("WebSocket streaming is disabled")
            return
        
        # Note: TheOddsAPI doesn't have native WebSocket support
        # This simulates real-time updates by polling at intervals
        async def poll_for_updates():
            while True:
                try:
                    for sport in sports:
                        events = await self.get_odds(sport)
                        for event in events:
                            # Check for odds changes
                            if event.id in self.events_cache:
                                old_event = self.events_cache[event.id]
                                changes = self._detect_odds_changes(old_event, event)
                                if changes:
                                    await callback({
                                        'type': 'odds_update',
                                        'event_id': event.id,
                                        'sport': event.sport,
                                        'changes': changes,
                                        'timestamp': datetime.now().isoformat()
                                    })
                    
                    # Poll every 30 seconds
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    logger.error(f"WebSocket polling error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        # Start polling task
        asyncio.create_task(poll_for_updates())
    
    def _detect_odds_changes(self, old_event: Event, new_event: Event) -> List[Dict]:
        """Detect changes in odds between two versions of an event."""
        changes = []
        
        # Create lookup for old odds
        old_odds_lookup = {}
        for odds in old_event.odds:
            key = f"{odds.bookmaker}_{odds.market}_{odds.selection}"
            old_odds_lookup[key] = odds
        
        # Check for changes in new odds
        for new_odds in new_event.odds:
            key = f"{new_odds.bookmaker}_{new_odds.market}_{new_odds.selection}"
            
            if key in old_odds_lookup:
                old_odds = old_odds_lookup[key]
                if abs(old_odds.odds - new_odds.odds) > 0.01:  # Significant change
                    changes.append({
                        'bookmaker': new_odds.bookmaker,
                        'market': new_odds.market,
                        'selection': new_odds.selection,
                        'old_odds': old_odds.odds,
                        'new_odds': new_odds.odds,
                        'change': new_odds.odds - old_odds.odds,
                        'change_percent': ((new_odds.odds - old_odds.odds) / old_odds.odds) * 100
                    })
            else:
                # New odds appeared
                changes.append({
                    'bookmaker': new_odds.bookmaker,
                    'market': new_odds.market,
                    'selection': new_odds.selection,
                    'new_odds': new_odds.odds,
                    'type': 'new_odds'
                })
        
        return changes
    
    def get_market_depth_analysis(self, event_id: str) -> Dict[str, Any]:
        """
        Analyze market depth and liquidity for an event.
        
        Returns comprehensive market analysis including:
        - Bookmaker coverage
        - Odds spread analysis
        - Market efficiency metrics
        - Arbitrage potential
        """
        if event_id not in self.events_cache:
            return {"error": "Event not found"}
        
        event = self.events_cache[event_id]
        analysis = {
            "event_id": event_id,
            "sport": event.sport,
            "home_team": event.home_team,
            "away_team": event.away_team,
            "commence_time": event.commence_time.isoformat(),
            "analysis_time": datetime.now().isoformat(),
            "markets": {}
        }
        
        # Group odds by market
        market_odds = defaultdict(lambda: defaultdict(list))
        for odds in event.odds:
            market_odds[odds.market][odds.selection].append(odds)
        
        # Analyze each market
        for market, selections in market_odds.items():
            market_analysis = {
                "bookmaker_count": len(set(odds.bookmaker for odds_list in selections.values() for odds in odds_list)),
                "selections": {}
            }
            
            for selection, odds_list in selections.items():
                if odds_list:
                    odds_values = [odds.odds for odds in odds_list]
                    market_analysis["selections"][selection] = {
                        "bookmaker_count": len(odds_list),
                        "best_odds": max(odds_values),
                        "worst_odds": min(odds_values),
                        "average_odds": np.mean(odds_values),
                        "std_deviation": np.std(odds_values),
                        "spread": max(odds_values) - min(odds_values),
                        "spread_percent": ((max(odds_values) - min(odds_values)) / np.mean(odds_values)) * 100,
                        "bookmakers": [{"name": odds.bookmaker, "odds": odds.odds} for odds in odds_list]
                    }
            
            # Calculate market efficiency
            if len(selections) >= 2:
                total_best_implied_prob = sum(
                    1.0 / max(odds.odds for odds in odds_list)
                    for odds_list in selections.values()
                    if odds_list
                )
                market_analysis["efficiency"] = {
                    "total_implied_probability": total_best_implied_prob,
                    "overround": max(0, total_best_implied_prob - 1.0),
                    "efficiency_score": 1.0 / total_best_implied_prob if total_best_implied_prob > 0 else 0
                }
            
            analysis["markets"][market] = market_analysis
        
        # Find best arbitrage opportunity for this event
        arb_opportunities = self.find_arbitrage_opportunities([event])
        if arb_opportunities:
            best_arb = arb_opportunities[0]
            analysis["best_arbitrage"] = asdict(best_arb)
        
        return analysis
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get API client performance metrics."""
        uptime = time.time() - self.start_time
        success_rate = ((self.request_count - self.error_count) / self.request_count * 100) if self.request_count > 0 else 0
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "successful_requests": self.request_count - self.error_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "current_request_rate": self.rate_limiter.get_current_rate(),
            "circuit_breaker_state": self.circuit_breaker.state,
            "cached_events": len(self.events_cache),
            "odds_history_size": sum(len(odds_list) for odds_list in self.odds_history.values()),
            "average_bookmaker_reliability": np.mean(list(self.bookmaker_reliability.values())) if self.bookmaker_reliability else 1.0
        }
    
    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()
        
        # Close WebSocket connections
        for ws in self.websocket_connections.values():
            await ws.close()
        
        logger.info("TheOddsAPI client cleaned up")

# Example usage and testing
async def main():
    """Example usage of TheOddsAPI client."""
    api_key = "YOUR_API_KEY_HERE"
    
    async with TheOddsAPI(api_key) as odds_api:
        try:
            # Get available sports
            sports = await odds_api.get_sports()
            print(f"Available sports: {len(sports)}")
            
            # Get upcoming NFL events
            nfl_events = await odds_api.get_events(Sport.AMERICANFOOTBALL_NFL)
            print(f"Upcoming NFL events: {len(nfl_events)}")
            
            # Get odds for NBA
            nba_odds = await odds_api.get_odds(
                Sport.BASKETBALL_NBA,
                markets=[Market.H2H, Market.SPREADS],
                regions=[Region.US]
            )
            print(f"NBA events with odds: {len(nba_odds)}")
            
            # Find arbitrage opportunities
            if nba_odds:
                arbitrage_opps = odds_api.find_arbitrage_opportunities(nba_odds)
                print(f"Arbitrage opportunities found: {len(arbitrage_opps)}")
                
                for opp in arbitrage_opps[:3]:  # Show top 3
                    print(f"Arbitrage: {opp.home_team} vs {opp.away_team}")
                    print(f"  Profit margin: {opp.profit_margin:.2%}")
                    print(f"  Guaranteed profit: ${opp.guaranteed_profit:.2f}")
            
            # Get market depth analysis
            if nba_odds:
                event = nba_odds[0]
                depth_analysis = odds_api.get_market_depth_analysis(event.id)
                print(f"Market depth analysis for {event.home_team} vs {event.away_team}")
                print(f"  Markets analyzed: {len(depth_analysis.get('markets', {}))}")
            
            # Show performance metrics
            metrics = odds_api.get_performance_metrics()
            print(f"API Performance:")
            print(f"  Success rate: {metrics['success_rate']:.1f}%")
            print(f"  Total requests: {metrics['total_requests']}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())