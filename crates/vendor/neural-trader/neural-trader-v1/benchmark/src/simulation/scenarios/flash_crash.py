"""
Flash crash scenario configuration.
"""
from typing import Dict, List
import numpy as np
from ..market_simulator import SimulationConfig, MarketSimulator
from ..price_generator import MarketRegime
from ..event_simulator import EventSimulator, NewsImpact, NewsEvent, MarketEvent, EventType


class FlashCrashScenario:
    """Configure simulation for flash crash conditions."""
    
    def __init__(self, symbols: List[str], duration: float = 1800):  # 30 minutes
        self.symbols = symbols
        self.duration = duration
        self.crash_time = duration * 0.3  # Crash at 30% through simulation
        self.crash_magnitude = 0.15  # 15% crash
        self.recovery_time = 300  # 5 minute recovery
        
    def create_config(self) -> SimulationConfig:
        """Create flash crash simulation configuration."""
        # Normal initial prices
        initial_prices = {
            symbol: 100.0 + i * 5.0
            for i, symbol in enumerate(self.symbols)
        }
        
        # Mix of participants that can trigger cascade
        participant_counts = {
            "market_maker": 2,  # Few market makers (liquidity crisis)
            "random_trader": 5,
            "momentum_trader": 15  # Many momentum traders (cascade effect)
        }
        
        # Very high correlation during crash
        n = len(self.symbols)
        correlation_matrix = np.full((n, n), 0.9)  # Extreme correlation
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return SimulationConfig(
            symbols=self.symbols,
            duration=self.duration,
            tick_rate=5000,  # Very high frequency
            initial_prices=initial_prices,
            participant_counts=participant_counts,
            correlation_matrix=correlation_matrix
        )
    
    def configure_simulator(self, simulator: MarketSimulator):
        """Configure simulator for flash crash conditions."""
        # Normal market initially
        for symbol, generator in simulator.price_generators.items():
            generator.set_market_regime(MarketRegime.NORMAL)
            
            # Normal volatility initially
            generator.config.volatility = 0.15
            generator.config.drift = 0.05
            
            # Set up for crash
            generator.config.jump_frequency = 0.0  # No random jumps
            
            # Will trigger flash crash later
        
        # Very sensitive circuit breakers
        simulator.enable_circuit_breakers(threshold=0.02)  # 2% moves
    
    def configure_events(self, event_simulator: EventSimulator):
        """Configure events for flash crash."""
        # Pre-crash: Normal market with slight tension
        for i in range(5):
            symbol = np.random.choice(self.symbols)
            
            news_event = NewsEvent.create(
                symbol=symbol,
                headline=f"{symbol} trading near day's high",
                sentiment=NewsImpact.NEUTRAL
            )
            news_event.timestamp = i * 60  # Every minute before crash
            event_simulator.schedule_event(news_event)
        
        # Trigger event (e.g., fat finger trade, algorithm error)
        trigger_event = MarketEvent(
            event_id="FLASH_CRASH_TRIGGER",
            event_type=EventType.FLASH_CRASH,
            timestamp=self.crash_time,
            symbol=None,  # Market-wide
            data={
                "trigger": "Algorithmic trading error detected",
                "initial_symbol": self.symbols[0],
                "cascade_start": True
            },
            impact_magnitude=1.0
        )
        event_simulator.schedule_event(trigger_event)
        
        # Cascade of halt events during crash
        for i, symbol in enumerate(self.symbols):
            halt_event = event_simulator.generate_trading_halt(
                symbol=symbol,
                reason="Circuit breaker - rapid price decline"
            )
            # Stagger halts as crash propagates
            halt_event.timestamp = self.crash_time + i * 2
            halt_event.duration = 60  # 1 minute halts
            event_simulator.schedule_event(halt_event)
        
        # Panic news during crash
        panic_headlines = [
            "BREAKING: Markets in freefall, cause unknown",
            "Trading halted across multiple exchanges",
            "Liquidity crisis as market makers withdraw",
            "Algorithmic trading systems malfunctioning",
            "Regulators investigating market anomaly"
        ]
        
        for i, headline in enumerate(panic_headlines):
            news_event = NewsEvent.create(
                symbol=None,  # Market-wide
                headline=headline,
                sentiment=NewsImpact.VERY_NEGATIVE
            )
            news_event.timestamp = self.crash_time + i * 10
            event_simulator.schedule_event(news_event)
        
        # Recovery phase news
        recovery_headlines = [
            "Markets stabilizing after sharp decline",
            "Regulators: No fundamental issues found",
            "Market makers returning to provide liquidity",
            "Technical glitch identified and resolved",
            "Markets recovering from morning flash crash"
        ]
        
        for i, headline in enumerate(recovery_headlines):
            news_event = NewsEvent.create(
                symbol=None,
                headline=headline,
                sentiment=NewsImpact.POSITIVE
            )
            news_event.timestamp = self.crash_time + self.recovery_time + i * 30
            event_simulator.schedule_event(news_event)
        
        # Post-crash investigation
        investigation_event = MarketEvent(
            event_id="SEC_INVESTIGATION",
            event_type=EventType.REGULATORY,
            timestamp=self.crash_time + 600,  # 10 minutes after
            symbol=None,
            data={
                "announcement": "SEC announces full investigation of market events",
                "impact": "Trading restrictions may apply"
            },
            impact_magnitude=0.3
        )
        event_simulator.schedule_event(investigation_event)
    
    def trigger_flash_crash(self, simulator: MarketSimulator, event_simulator: EventSimulator):
        """Manually trigger the flash crash at specified time."""
        # This would be called during simulation at crash_time
        for symbol, generator in simulator.price_generators.items():
            generator.trigger_flash_crash(
                crash_magnitude=self.crash_magnitude,
                recovery_time=self.recovery_time
            )
    
    def get_expected_metrics(self) -> Dict[str, float]:
        """Get expected metrics for validation."""
        return {
            "max_drawdown": 0.15,     # 15% crash
            "crash_duration": 300,     # 5 minutes
            "recovery_time": 300,      # 5 minutes to recover
            "volatility_spike": 5.0,   # 5x normal volatility during crash
            "circuit_breaker_triggers": len(self.symbols),  # One per symbol
            "liquidity_drop": 0.80,    # 80% drop in liquidity
            "correlation_spike": 0.95,  # Near-perfect correlation
            "bid_ask_spread_max": 0.01  # 1% max spread
        }
    
    @staticmethod
    def validate_results(results: Dict[str, float], expected: Dict[str, float]) -> bool:
        """Validate if results match flash crash characteristics."""
        # Check for significant drawdown
        if results.get("max_drawdown", 0) < expected["max_drawdown"] * 0.8:
            return False
        
        # Check for volatility spike
        if results.get("volatility_spike", 0) < 3.0:
            return False
        
        # Check circuit breakers triggered
        if results.get("circuit_breaker_triggers", 0) < 1:
            return False
        
        # Check for liquidity drop (wide spreads)
        if results.get("bid_ask_spread_max", 0) < 0.005:
            return False
        
        # Check for recovery (prices should recover somewhat)
        final_drawdown = results.get("final_drawdown", 1.0)
        if final_drawdown > expected["max_drawdown"] * 0.5:
            return False  # Should recover at least 50%
        
        return True