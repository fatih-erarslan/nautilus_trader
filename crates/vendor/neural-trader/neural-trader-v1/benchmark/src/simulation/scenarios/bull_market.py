"""
Bull market scenario configuration.
"""
from typing import Dict, List, Optional
import numpy as np
from ..market_simulator import SimulationConfig, MarketSimulator
from ..price_generator import MarketRegime, PriceGeneratorConfig
from ..event_simulator import EventSimulator, NewsImpact, NewsEvent


class BullMarketScenario:
    """Configure simulation for bull market conditions."""
    
    def __init__(self, symbols: List[str], duration: float = 3600):
        self.symbols = symbols
        self.duration = duration
        self.base_volatility = 0.15  # Lower volatility in bull markets
        self.base_drift = 0.20  # Strong positive drift (20% annual)
        
    def create_config(self) -> SimulationConfig:
        """Create bull market simulation configuration."""
        # Initial prices
        initial_prices = {
            symbol: 100.0 * (1 + 0.1 * i)  # Slight variation
            for i, symbol in enumerate(self.symbols)
        }
        
        # More market makers and momentum traders in bull market
        participant_counts = {
            "market_maker": 3,
            "random_trader": 8,
            "momentum_trader": 10  # More momentum traders
        }
        
        # Positive correlation in bull market
        n = len(self.symbols)
        correlation_matrix = np.full((n, n), 0.6)  # High correlation
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
        """Configure simulator for bull market conditions."""
        # Set all price generators to trending up
        for symbol, generator in simulator.price_generators.items():
            generator.set_market_regime(MarketRegime.TRENDING_UP)
            
            # Update configuration for bull market
            generator.config.volatility = self.base_volatility
            generator.config.drift = self.base_drift
            
            # Lower jump frequency (fewer negative surprises)
            generator.config.jump_frequency = 0.001
            generator.config.jump_size_mean = 0.01  # Positive bias
            generator.config.jump_size_std = 0.005
        
        # Set wider circuit breakers (less likely to trigger)
        simulator.enable_circuit_breakers(threshold=0.10)
    
    def configure_events(self, event_simulator: EventSimulator):
        """Configure events for bull market."""
        # More positive news in bull market
        event_simulator.news_frequency = 0.02  # More frequent news
        
        # Schedule positive events
        current_time = 0
        for i in range(10):  # 10 positive news events
            symbol = np.random.choice(self.symbols)
            
            headlines = [
                f"{symbol} announces record quarterly profits",
                f"{symbol} beats analyst expectations significantly",
                f"Major institution increases {symbol} price target",
                f"{symbol} wins major government contract",
                f"{symbol} market share grows to all-time high"
            ]
            
            news_event = NewsEvent.create(
                symbol=symbol,
                headline=np.random.choice(headlines),
                sentiment=NewsImpact.VERY_POSITIVE
            )
            
            # Schedule throughout the simulation
            news_event.timestamp = current_time + (i * self.duration / 10)
            event_simulator.schedule_event(news_event)
        
        # Few negative events (market corrections)
        for i in range(2):
            symbol = np.random.choice(self.symbols)
            
            news_event = NewsEvent.create(
                symbol=symbol,
                headline=f"{symbol} faces minor supply chain issues",
                sentiment=NewsImpact.NEGATIVE
            )
            
            news_event.timestamp = current_time + np.random.uniform(0, self.duration)
            event_simulator.schedule_event(news_event)
    
    def get_expected_metrics(self) -> Dict[str, float]:
        """Get expected metrics for validation."""
        return {
            "average_return": 0.15,  # 15% return expected
            "max_drawdown": 0.05,    # Small drawdowns
            "volatility": 0.15,      # Lower volatility
            "sharpe_ratio": 1.5,     # Good risk-adjusted returns
            "win_rate": 0.65         # Most trades profitable
        }
    
    @staticmethod
    def validate_results(results: Dict[str, float], expected: Dict[str, float]) -> bool:
        """Validate if results match bull market characteristics."""
        # Check if returns are positive and substantial
        if results.get("average_return", 0) < expected["average_return"] * 0.5:
            return False
        
        # Check if drawdowns are limited
        if results.get("max_drawdown", 1.0) > expected["max_drawdown"] * 2:
            return False
        
        # Check if volatility is reasonable
        if results.get("volatility", 1.0) > expected["volatility"] * 1.5:
            return False
        
        return True