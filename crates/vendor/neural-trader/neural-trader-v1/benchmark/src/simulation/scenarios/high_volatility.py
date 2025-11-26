"""
High volatility market scenario configuration.
"""
from typing import Dict, List
import numpy as np
from ..market_simulator import SimulationConfig, MarketSimulator
from ..price_generator import MarketRegime, PriceGeneratorConfig
from ..event_simulator import EventSimulator, NewsImpact, NewsEvent, MarketEvent, EventType


class HighVolatilityScenario:
    """Configure simulation for high volatility conditions."""
    
    def __init__(self, symbols: List[str], duration: float = 3600):
        self.symbols = symbols
        self.duration = duration
        self.base_volatility = 0.50  # Extreme volatility (50% annual)
        self.base_drift = 0.0  # No clear direction
        
    def create_config(self) -> SimulationConfig:
        """Create high volatility simulation configuration."""
        # Initial prices with more variation
        initial_prices = {
            symbol: 100.0 * (1 + 0.2 * np.random.randn())
            for symbol in self.symbols
        }
        
        # Mixed participant types
        participant_counts = {
            "market_maker": 4,  # Need liquidity providers
            "random_trader": 20,  # More noise traders
            "momentum_trader": 8  # Trend chasers add to volatility
        }
        
        # Lower correlation (decorrelated chaos)
        n = len(self.symbols)
        correlation_matrix = np.random.uniform(0.1, 0.4, size=(n, n))
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return SimulationConfig(
            symbols=self.symbols,
            duration=self.duration,
            tick_rate=2000,  # Higher tick rate for volatility
            initial_prices=initial_prices,
            participant_counts=participant_counts,
            correlation_matrix=correlation_matrix
        )
    
    def configure_simulator(self, simulator: MarketSimulator):
        """Configure simulator for high volatility conditions."""
        for symbol, generator in simulator.price_generators.items():
            generator.set_market_regime(MarketRegime.HIGH_VOLATILITY)
            
            # Extreme volatility configuration
            generator.config.volatility = self.base_volatility
            generator.config.drift = self.base_drift
            
            # Frequent jumps in both directions
            generator.config.jump_frequency = 0.02
            generator.config.jump_size_mean = 0.0
            generator.config.jump_size_std = 0.02
            
            # Strong volatility clustering
            generator.enable_volatility_clustering(
                persistence=0.95,  # Very persistent
                mean_reversion=0.05
            )
            
            # Intraday patterns amplified
            generator.enable_intraday_patterns(
                market_open=9.5,
                market_close=16.0
            )
        
        # Circuit breakers trigger frequently
        simulator.enable_circuit_breakers(threshold=0.03)  # 3% moves
    
    def configure_events(self, event_simulator: EventSimulator):
        """Configure events for high volatility market."""
        # Very frequent news with mixed sentiment
        event_simulator.news_frequency = 0.05
        event_simulator.earnings_frequency = 0.001
        event_simulator.halt_frequency = 0.002
        
        # Rapid sentiment changes
        sentiments = list(NewsImpact)
        headlines_by_sentiment = {
            NewsImpact.VERY_POSITIVE: [
                "Breaking: {symbol} announces breakthrough technology",
                "{symbol} receives massive government contract",
                "Surprise: {symbol} profits triple expectations"
            ],
            NewsImpact.POSITIVE: [
                "{symbol} beats earnings estimates",
                "Analysts upgrade {symbol} rating",
                "{symbol} expands into new markets"
            ],
            NewsImpact.NEUTRAL: [
                "{symbol} trading volatile on high volume",
                "Options activity unusual for {symbol}",
                "{symbol} management addresses concerns"
            ],
            NewsImpact.NEGATIVE: [
                "{symbol} warns on earnings",
                "Regulatory concerns for {symbol}",
                "{symbol} loses key executive"
            ],
            NewsImpact.VERY_NEGATIVE: [
                "Breaking: {symbol} under federal investigation",
                "{symbol} product recall announced",
                "Major lawsuit filed against {symbol}"
            ]
        }
        
        # Schedule rapid-fire contradictory news
        for i in range(50):  # Many events
            symbol = np.random.choice(self.symbols)
            sentiment = np.random.choice(sentiments)
            headlines = headlines_by_sentiment[sentiment]
            
            news_event = NewsEvent.create(
                symbol=symbol,
                headline=np.random.choice(headlines).format(symbol=symbol),
                sentiment=sentiment
            )
            
            # Random timing throughout simulation
            news_event.timestamp = np.random.uniform(0, self.duration)
            event_simulator.schedule_event(news_event)
        
        # Schedule multiple mini flash crashes
        for i in range(5):
            # Pick random subset of symbols
            affected = np.random.choice(self.symbols, size=np.random.randint(1, len(self.symbols)), replace=False)
            
            flash_event = MarketEvent(
                event_id=f"MINI_FLASH_{i}",
                event_type=EventType.FLASH_CRASH,
                timestamp=np.random.uniform(0, self.duration),
                symbol=None,
                data={
                    "affected_symbols": list(affected),
                    "magnitude": np.random.uniform(0.03, 0.08),
                    "recovery_time": np.random.uniform(60, 300)
                },
                impact_magnitude=0.8,
                duration=300
            )
            event_simulator.schedule_event(flash_event)
        
        # Contradictory economic data
        indicators = ["CPI", "Unemployment", "GDP", "Retail Sales"]
        for indicator in indicators:
            # Large surprise in random direction
            surprise_direction = np.random.choice([-1, 1])
            actual = 2.0 + surprise_direction * np.random.uniform(1, 3)
            expected = 2.0
            
            econ_event = event_simulator.generate_economic_data(
                indicator=indicator,
                actual=actual,
                expected=expected
            )
            econ_event.timestamp = np.random.uniform(0, self.duration)
            event_simulator.schedule_event(econ_event)
    
    def get_expected_metrics(self) -> Dict[str, float]:
        """Get expected metrics for validation."""
        return {
            "average_return": 0.0,    # No clear direction
            "max_drawdown": 0.25,     # Large swings
            "volatility": 0.50,       # Extreme volatility
            "sharpe_ratio": 0.0,      # Poor risk-adjusted returns
            "win_rate": 0.50,         # Random outcomes
            "circuit_breaker_triggers": 10,  # Many halts
            "average_spread": 0.002,  # Wide spreads
            "price_reversals": 20     # Many direction changes
        }
    
    @staticmethod
    def validate_results(results: Dict[str, float], expected: Dict[str, float]) -> bool:
        """Validate if results match high volatility characteristics."""
        # Check if volatility is extreme
        if results.get("volatility", 0) < expected["volatility"] * 0.7:
            return False
        
        # Check for many circuit breaker triggers
        if results.get("circuit_breaker_triggers", 0) < 5:
            return False
        
        # Check for wide spreads
        if results.get("average_spread", 0) < expected["average_spread"] * 0.5:
            return False
        
        # Returns should be near zero (no clear trend)
        if abs(results.get("average_return", 1.0)) > 0.1:
            return False
        
        return True