"""
Bear market scenario configuration.
"""
from typing import Dict, List
import numpy as np
from ..market_simulator import SimulationConfig, MarketSimulator
from ..price_generator import MarketRegime, PriceGeneratorConfig
from ..event_simulator import EventSimulator, NewsImpact, NewsEvent, EventType


class BearMarketScenario:
    """Configure simulation for bear market conditions."""
    
    def __init__(self, symbols: List[str], duration: float = 3600):
        self.symbols = symbols
        self.duration = duration
        self.base_volatility = 0.35  # Higher volatility in bear markets
        self.base_drift = -0.25  # Strong negative drift (-25% annual)
        
    def create_config(self) -> SimulationConfig:
        """Create bear market simulation configuration."""
        # Initial prices
        initial_prices = {
            symbol: 100.0 * (1 + 0.1 * i)
            for i, symbol in enumerate(self.symbols)
        }
        
        # Fewer momentum traders, more market makers needed
        participant_counts = {
            "market_maker": 5,  # More market makers for liquidity
            "random_trader": 15,  # More panic selling
            "momentum_trader": 3  # Fewer trend followers
        }
        
        # High correlation in bear market (everything falls together)
        n = len(self.symbols)
        correlation_matrix = np.full((n, n), 0.8)  # Very high correlation
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return SimulationConfig(
            symbols=self.symbols,
            duration=self.duration,
            tick_rate=1000,
            initial_prices=initial_prices,
            participant_counts=participant_counts,
            correlation_matrix=correlation_matrix
        )
    
    def configure_simulator(self, simulator: MarketSimulator):
        """Configure simulator for bear market conditions."""
        # Set all price generators to trending down
        for symbol, generator in simulator.price_generators.items():
            generator.set_market_regime(MarketRegime.TRENDING_DOWN)
            
            # Update configuration for bear market
            generator.config.volatility = self.base_volatility
            generator.config.drift = self.base_drift
            
            # Higher jump frequency (more negative surprises)
            generator.config.jump_frequency = 0.01
            generator.config.jump_size_mean = -0.02  # Negative bias
            generator.config.jump_size_std = 0.01
            
            # Enable volatility clustering (common in bear markets)
            generator.enable_volatility_clustering(
                persistence=0.9,
                mean_reversion=0.1
            )
        
        # Tighter circuit breakers (more likely to trigger)
        simulator.enable_circuit_breakers(threshold=0.05)
    
    def configure_events(self, event_simulator: EventSimulator):
        """Configure events for bear market."""
        # More negative news
        event_simulator.news_frequency = 0.03  # Very frequent news
        event_simulator.halt_frequency = 0.001  # More trading halts
        
        # Schedule cascade of negative events
        current_time = 0
        
        # Initial shock event
        initial_crash = event_simulator.generate_economic_data(
            indicator="GDP",
            actual=-2.5,
            expected=1.5
        )
        initial_crash.timestamp = current_time + 60  # 1 minute in
        event_simulator.schedule_event(initial_crash)
        
        # Follow-up negative news
        negative_headlines = [
            "Federal Reserve hints at prolonged recession",
            "Major bank warns of credit crisis",
            "Unemployment reaches decade high",
            "Corporate earnings forecast slashed industry-wide",
            "Global trade tensions escalate significantly"
        ]
        
        for i, headline in enumerate(negative_headlines):
            # Market-wide negative news
            news_event = NewsEvent.create(
                symbol=None,  # Market-wide
                headline=headline,
                sentiment=NewsImpact.VERY_NEGATIVE
            )
            news_event.timestamp = current_time + (i + 1) * 300  # Every 5 minutes
            event_simulator.schedule_event(news_event)
        
        # Company-specific bad news
        for i in range(15):
            symbol = np.random.choice(self.symbols)
            
            company_headlines = [
                f"{symbol} announces major layoffs",
                f"{symbol} revenue misses by 20%",
                f"SEC investigates {symbol} accounting",
                f"{symbol} loses major customer",
                f"Credit downgrade for {symbol}"
            ]
            
            news_event = NewsEvent.create(
                symbol=symbol,
                headline=np.random.choice(company_headlines),
                sentiment=np.random.choice([NewsImpact.NEGATIVE, NewsImpact.VERY_NEGATIVE])
            )
            
            news_event.timestamp = current_time + np.random.uniform(0, self.duration)
            event_simulator.schedule_event(news_event)
        
        # Schedule periodic trading halts
        for i in range(3):
            symbol = np.random.choice(self.symbols)
            halt_event = event_simulator.generate_trading_halt(
                symbol=symbol,
                reason="Circuit breaker triggered"
            )
            halt_event.timestamp = current_time + (i + 1) * self.duration / 4
            event_simulator.schedule_event(halt_event)
    
    def get_expected_metrics(self) -> Dict[str, float]:
        """Get expected metrics for validation."""
        return {
            "average_return": -0.20,  # -20% return expected
            "max_drawdown": 0.30,     # Large drawdowns
            "volatility": 0.35,       # High volatility
            "sharpe_ratio": -0.8,     # Negative risk-adjusted returns
            "win_rate": 0.35,         # Most trades lose money
            "circuit_breaker_triggers": 3  # Multiple halts expected
        }
    
    @staticmethod
    def validate_results(results: Dict[str, float], expected: Dict[str, float]) -> bool:
        """Validate if results match bear market characteristics."""
        # Check if returns are negative
        if results.get("average_return", 0) > 0:
            return False
        
        # Check if drawdowns are significant
        if results.get("max_drawdown", 0) < expected["max_drawdown"] * 0.5:
            return False
        
        # Check if volatility is elevated
        if results.get("volatility", 0) < expected["volatility"] * 0.7:
            return False
        
        # Check if circuit breakers triggered
        if results.get("circuit_breaker_triggers", 0) < 1:
            return False
        
        return True