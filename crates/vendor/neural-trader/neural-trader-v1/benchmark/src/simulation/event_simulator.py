"""
Event simulation for market events, news, and announcements.
"""
import asyncio
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import numpy as np


class EventType(Enum):
    """Types of market events."""
    NEWS = "news"
    EARNINGS = "earnings"
    ECONOMIC_DATA = "economic_data"
    TRADING_HALT = "trading_halt"
    CIRCUIT_BREAKER = "circuit_breaker"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    FLASH_CRASH = "flash_crash"
    REGULATORY = "regulatory"


class NewsImpact(Enum):
    """News sentiment impact levels."""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class MarketEvent:
    """Represents a market event."""
    event_id: str
    event_type: EventType
    timestamp: float
    symbol: Optional[str]  # None for market-wide events
    data: Dict[str, Any]
    impact_magnitude: float  # 0.0 to 1.0
    duration: Optional[float] = None  # For events with duration
    
    def is_active(self, current_time: float) -> bool:
        """Check if event is currently active."""
        if self.duration is None:
            return False
        return current_time >= self.timestamp and current_time < self.timestamp + self.duration


@dataclass
class NewsEvent(MarketEvent):
    """News event with sentiment analysis."""
    headline: str
    sentiment: NewsImpact
    source: str
    
    @classmethod
    def create(cls, symbol: str, headline: str, sentiment: NewsImpact, 
               source: str = "Reuters") -> "NewsEvent":
        """Create a news event."""
        impact_map = {
            NewsImpact.VERY_NEGATIVE: 0.8,
            NewsImpact.NEGATIVE: 0.4,
            NewsImpact.NEUTRAL: 0.1,
            NewsImpact.POSITIVE: 0.4,
            NewsImpact.VERY_POSITIVE: 0.8
        }
        
        return cls(
            event_id=f"NEWS_{int(time.time() * 1000000)}",
            event_type=EventType.NEWS,
            timestamp=time.time(),
            symbol=symbol,
            data={
                "headline": headline,
                "sentiment": sentiment.value,
                "source": source
            },
            impact_magnitude=impact_map[sentiment],
            headline=headline,
            sentiment=sentiment,
            source=source
        )


class EventSimulator:
    """Simulates market events during trading."""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.event_history: List[MarketEvent] = []
        self.event_callbacks: Dict[EventType, List[Callable]] = {}
        self.scheduled_events: List[MarketEvent] = []
        
        # Event generation parameters
        self.news_frequency = 0.01  # Probability per tick
        self.earnings_frequency = 0.0001
        self.halt_frequency = 0.0005
        
        # Pre-configured news templates
        self.news_templates = {
            NewsImpact.VERY_POSITIVE: [
                "{symbol} beats earnings estimates by 20%",
                "{symbol} announces major partnership deal",
                "{symbol} receives FDA approval for new drug"
            ],
            NewsImpact.POSITIVE: [
                "{symbol} reports strong quarterly growth",
                "{symbol} raises full-year guidance",
                "Analysts upgrade {symbol} to buy"
            ],
            NewsImpact.NEUTRAL: [
                "{symbol} announces board changes",
                "{symbol} to present at investor conference",
                "{symbol} trading volume above average"
            ],
            NewsImpact.NEGATIVE: [
                "{symbol} misses revenue expectations",
                "{symbol} lowers quarterly guidance",
                "Analysts downgrade {symbol} outlook"
            ],
            NewsImpact.VERY_NEGATIVE: [
                "{symbol} warns of significant losses",
                "SEC investigates {symbol} accounting practices",
                "{symbol} announces major layoffs"
            ]
        }
    
    def register_callback(self, event_type: EventType, callback: Callable):
        """Register callback for specific event type."""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def schedule_event(self, event: MarketEvent):
        """Schedule an event to occur."""
        self.scheduled_events.append(event)
        self.scheduled_events.sort(key=lambda e: e.timestamp)
    
    def generate_random_news(self, symbol: str) -> Optional[NewsEvent]:
        """Generate random news event."""
        if random.random() > self.news_frequency:
            return None
        
        # Random sentiment with bias towards neutral
        sentiment_weights = [0.1, 0.2, 0.4, 0.2, 0.1]  # Very neg to very pos
        sentiment = random.choices(list(NewsImpact), weights=sentiment_weights)[0]
        
        # Select random template
        templates = self.news_templates[sentiment]
        headline = random.choice(templates).format(symbol=symbol)
        
        return NewsEvent.create(symbol, headline, sentiment)
    
    def generate_earnings_event(self, symbol: str) -> Optional[MarketEvent]:
        """Generate earnings announcement."""
        if random.random() > self.earnings_frequency:
            return None
        
        # Earnings surprise
        surprise = np.random.normal(0, 0.1)  # ±10% surprise on average
        
        event = MarketEvent(
            event_id=f"EARNINGS_{symbol}_{int(time.time() * 1000000)}",
            event_type=EventType.EARNINGS,
            timestamp=time.time(),
            symbol=symbol,
            data={
                "eps_actual": 1.0 + surprise,
                "eps_estimate": 1.0,
                "revenue_actual": 1000000 * (1 + surprise),
                "revenue_estimate": 1000000
            },
            impact_magnitude=min(abs(surprise) * 5, 1.0)
        )
        
        return event
    
    def generate_trading_halt(self, symbol: str, reason: str = "Pending news") -> MarketEvent:
        """Generate trading halt event."""
        return MarketEvent(
            event_id=f"HALT_{symbol}_{int(time.time() * 1000000)}",
            event_type=EventType.TRADING_HALT,
            timestamp=time.time(),
            symbol=symbol,
            data={"reason": reason},
            impact_magnitude=0.5,
            duration=300  # 5 minute halt
        )
    
    def generate_flash_crash_event(self, symbols: Optional[List[str]] = None) -> MarketEvent:
        """Generate flash crash event affecting multiple symbols."""
        affected_symbols = symbols or self.symbols
        
        return MarketEvent(
            event_id=f"FLASH_CRASH_{int(time.time() * 1000000)}",
            event_type=EventType.FLASH_CRASH,
            timestamp=time.time(),
            symbol=None,  # Market-wide event
            data={
                "affected_symbols": affected_symbols,
                "magnitude": 0.1,  # 10% crash
                "recovery_time": 300  # 5 minutes
            },
            impact_magnitude=1.0,
            duration=300
        )
    
    def generate_economic_data(self, indicator: str, actual: float, 
                              expected: float) -> MarketEvent:
        """Generate economic data release."""
        surprise = (actual - expected) / expected
        
        return MarketEvent(
            event_id=f"ECON_{indicator}_{int(time.time() * 1000000)}",
            event_type=EventType.ECONOMIC_DATA,
            timestamp=time.time(),
            symbol=None,  # Market-wide
            data={
                "indicator": indicator,
                "actual": actual,
                "expected": expected,
                "surprise": surprise
            },
            impact_magnitude=min(abs(surprise) * 10, 1.0)
        )
    
    async def simulate_events(self, duration: float, tick_rate: int = 1000):
        """Run event simulation for specified duration."""
        start_time = time.time()
        tick_interval = 1.0 / tick_rate
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Process scheduled events
            self._process_scheduled_events(current_time)
            
            # Generate random events
            for symbol in self.symbols:
                # Random news
                news_event = self.generate_random_news(symbol)
                if news_event:
                    self._trigger_event(news_event)
                
                # Random earnings
                earnings_event = self.generate_earnings_event(symbol)
                if earnings_event:
                    self._trigger_event(earnings_event)
                
                # Random halts
                if random.random() < self.halt_frequency:
                    halt_event = self.generate_trading_halt(symbol)
                    self._trigger_event(halt_event)
            
            await asyncio.sleep(tick_interval)
    
    def _process_scheduled_events(self, current_time: float):
        """Process events scheduled for current time."""
        while self.scheduled_events and self.scheduled_events[0].timestamp <= current_time:
            event = self.scheduled_events.pop(0)
            self._trigger_event(event)
    
    def _trigger_event(self, event: MarketEvent):
        """Trigger an event and notify callbacks."""
        self.event_history.append(event)
        
        # Call registered callbacks
        if event.event_type in self.event_callbacks:
            for callback in self.event_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
    
    def get_active_events(self, current_time: float) -> List[MarketEvent]:
        """Get currently active events."""
        return [
            event for event in self.event_history
            if event.is_active(current_time)
        ]
    
    def get_symbol_events(self, symbol: str, 
                         time_window: Optional[float] = None) -> List[MarketEvent]:
        """Get events for a specific symbol."""
        current_time = time.time()
        
        symbol_events = [
            event for event in self.event_history
            if event.symbol == symbol or event.symbol is None  # Include market-wide
        ]
        
        if time_window:
            symbol_events = [
                event for event in symbol_events
                if current_time - event.timestamp <= time_window
            ]
        
        return symbol_events
    
    def calculate_event_impact(self, symbol: str, current_price: float) -> float:
        """Calculate cumulative impact of recent events on price."""
        # Get recent events (last 5 minutes)
        recent_events = self.get_symbol_events(symbol, time_window=300)
        
        if not recent_events:
            return current_price
        
        # Calculate cumulative impact
        cumulative_impact = 0.0
        
        for event in recent_events:
            # Decay impact over time
            age = time.time() - event.timestamp
            decay_factor = np.exp(-age / 300)  # 5 minute half-life
            
            # Apply impact based on event type and sentiment
            if event.event_type == EventType.NEWS:
                sentiment_value = event.data.get("sentiment", 0)
                impact = sentiment_value * 0.01 * event.impact_magnitude * decay_factor
                cumulative_impact += impact
            
            elif event.event_type == EventType.EARNINGS:
                surprise = event.data.get("surprise", 0)
                impact = surprise * event.impact_magnitude * decay_factor
                cumulative_impact += impact
            
            elif event.event_type == EventType.FLASH_CRASH:
                if event.is_active(time.time()):
                    impact = -event.data.get("magnitude", 0.1)
                    cumulative_impact += impact
        
        # Apply impact to price
        return current_price * (1 + cumulative_impact)
    
    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary statistics of events."""
        total_events = len(self.event_history)
        
        events_by_type = {}
        for event_type in EventType:
            count = sum(1 for e in self.event_history if e.event_type == event_type)
            events_by_type[event_type.value] = count
        
        # News sentiment distribution
        news_events = [e for e in self.event_history if e.event_type == EventType.NEWS]
        sentiment_dist = {}
        for impact in NewsImpact:
            count = sum(1 for e in news_events if e.data.get("sentiment") == impact.value)
            sentiment_dist[impact.name] = count
        
        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "news_sentiment_distribution": sentiment_dist,
            "average_impact_magnitude": np.mean([e.impact_magnitude for e in self.event_history])
            if self.event_history else 0.0
        }