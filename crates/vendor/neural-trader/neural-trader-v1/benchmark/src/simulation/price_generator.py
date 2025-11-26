"""
Realistic price generation for market simulation.
Supports multiple models: Brownian motion, jump diffusion, GARCH, etc.
"""
import asyncio
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, AsyncIterator
from scipy import stats


class MarketRegime(Enum):
    """Market regime states."""
    NORMAL = "NORMAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGE_BOUND = "RANGE_BOUND"


@dataclass
class PriceGeneratorConfig:
    """Configuration for price generation."""
    initial_price: float
    volatility: float  # Annual volatility
    drift: float      # Annual drift
    tick_size: float = 0.01
    update_frequency: int = 1000  # Updates per second
    jump_frequency: float = 0.0   # Probability of jump per update
    jump_size_mean: float = 0.0   # Mean jump size as fraction
    jump_size_std: float = 0.01   # Jump size standard deviation


@dataclass
class PriceUpdate:
    """Single price update event."""
    symbol: str
    price: float
    volume: int
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    halted: bool = False
    regime: MarketRegime = MarketRegime.NORMAL


class PriceModel:
    """Base class for price models."""
    
    def __init__(self, config: PriceGeneratorConfig):
        self.config = config
        self._rng = np.random.default_rng()
    
    def next_price(self, current_price: float, dt: float) -> float:
        """Generate next price given current price and time step."""
        raise NotImplementedError
    
    @staticmethod
    def create(model_type: str, config: PriceGeneratorConfig) -> "PriceModel":
        """Factory method to create price models."""
        if model_type == "brownian":
            return BrownianMotionModel(config)
        elif model_type == "jump_diffusion":
            return JumpDiffusionModel(config)
        elif model_type == "garch":
            return GARCHModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BrownianMotionModel(PriceModel):
    """Geometric Brownian Motion price model."""
    
    def next_price(self, current_price: float, dt: float) -> float:
        """Generate next price using GBM."""
        # Convert annual parameters to time step
        mu = self.config.drift * dt
        sigma = self.config.volatility * np.sqrt(dt)
        
        # Generate random shock
        shock = self._rng.normal(0, 1)
        
        # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift_term = (mu - 0.5 * sigma * sigma)
        diffusion_term = sigma * shock
        
        next_price = current_price * np.exp(drift_term + diffusion_term)
        
        # Round to tick size
        return round(next_price / self.config.tick_size) * self.config.tick_size


class JumpDiffusionModel(PriceModel):
    """Jump diffusion model (Merton model)."""
    
    def next_price(self, current_price: float, dt: float) -> float:
        """Generate next price with possible jumps."""
        # First apply standard Brownian motion
        brownian = BrownianMotionModel(self.config)
        next_price = brownian.next_price(current_price, dt)
        
        # Check for jump
        if self._rng.random() < self.config.jump_frequency * dt:
            # Generate jump size
            jump_size = self._rng.normal(
                self.config.jump_size_mean,
                self.config.jump_size_std
            )
            next_price *= (1 + jump_size)
        
        return round(next_price / self.config.tick_size) * self.config.tick_size


class GARCHModel(PriceModel):
    """GARCH(1,1) model for volatility clustering."""
    
    def __init__(self, config: PriceGeneratorConfig):
        super().__init__(config)
        self.alpha = 0.1  # Innovation weight
        self.beta = 0.85  # Volatility persistence
        self.omega = 0.05  # Long-term variance weight
        self.current_variance = config.volatility ** 2
        self.last_return = 0
    
    def next_price(self, current_price: float, dt: float) -> float:
        """Generate next price with GARCH volatility."""
        # Update conditional variance
        self.current_variance = (
            self.omega * self.config.volatility ** 2 +
            self.alpha * self.last_return ** 2 +
            self.beta * self.current_variance
        )
        
        # Current volatility
        current_vol = np.sqrt(self.current_variance)
        
        # Generate return
        mu = self.config.drift * dt
        sigma = current_vol * np.sqrt(dt)
        shock = self._rng.normal(0, 1)
        
        log_return = mu - 0.5 * sigma * sigma + sigma * shock
        self.last_return = log_return
        
        next_price = current_price * np.exp(log_return)
        return round(next_price / self.config.tick_size) * self.config.tick_size


class PriceGenerator:
    """Main price generator with advanced features."""
    
    def __init__(self, symbol: str, config: PriceGeneratorConfig,
                 model_type: str = "brownian"):
        self.symbol = symbol
        self.config = config
        self.model = PriceModel.create(model_type, config)
        self.current_price = config.initial_price
        self.current_time = time.time()
        
        # Market state
        self.market_regime = MarketRegime.NORMAL
        self.is_halted = False
        self.circuit_breaker_threshold = None
        
        # Intraday patterns
        self.intraday_enabled = False
        self.market_open = 9.5
        self.market_close = 16.0
        self.current_time_of_day = 10.0
        
        # Order flow impact
        self.order_flow_impact = 0.0
        self.impact_coefficient = 0.0
        self.impact_decay = 0.9
        
        # Volatility clustering
        self.volatility_clustering = False
        self.vol_persistence = 0.9
        self.vol_mean_reversion = 0.1
        self.current_vol_multiplier = 1.0
        
        # Flash crash state
        self.flash_crash_active = False
        self.crash_start_time = None
        self.crash_magnitude = 0.0
        self.crash_recovery_time = 0.0
        
        # Historical replay
        self.historical_mode = False
        self.historical_data = None
        self.replay_index = 0
    
    def generate_update(self) -> PriceUpdate:
        """Generate a single price update."""
        # Time step based on update frequency
        dt = 1.0 / self.config.update_frequency
        
        # Check if halted
        if self.is_halted:
            return self._create_halted_update()
        
        # Apply market regime adjustments
        effective_config = self._adjust_for_regime()
        
        # Apply intraday patterns
        if self.intraday_enabled:
            effective_config = self._apply_intraday_patterns(effective_config)
        
        # Apply volatility clustering
        if self.volatility_clustering:
            effective_config.volatility *= self.current_vol_multiplier
        
        # Generate base price
        self.model.config = effective_config
        new_price = self.model.next_price(self.current_price, dt)
        
        # Update volatility multiplier after generating price
        if self.volatility_clustering:
            if hasattr(self, '_last_price') and self._last_price > 0:
                recent_return = abs((new_price - self._last_price) / self._last_price)
                # Update vol multiplier with persistence and mean reversion
                self.current_vol_multiplier = (
                    self.vol_persistence * self.current_vol_multiplier +
                    self.vol_mean_reversion * (1.0 + 10 * recent_return)
                )
            self._last_price = new_price
        
        # Apply order flow impact
        if self.order_flow_impact != 0:
            new_price *= (1 + self.order_flow_impact)
            self.order_flow_impact *= self.impact_decay
        
        # Apply flash crash if active
        if self.flash_crash_active:
            new_price = self._apply_flash_crash(new_price)
        
        # Check circuit breaker
        if self.circuit_breaker_threshold:
            price_change = abs(new_price - self.current_price) / self.current_price
            if price_change > self.circuit_breaker_threshold:
                self.is_halted = True
                return self._create_halted_update()
        
        # Update state
        self.current_price = new_price
        self.current_time = time.time()
        
        # Generate volume
        volume = self._generate_volume()
        
        # Create update
        spread = self.config.tick_size * np.random.randint(1, 5)
        return PriceUpdate(
            symbol=self.symbol,
            price=new_price,
            volume=volume,
            timestamp=self.current_time,
            bid=new_price - spread/2,
            ask=new_price + spread/2,
            regime=self.market_regime
        )
    
    def _adjust_for_regime(self) -> PriceGeneratorConfig:
        """Adjust parameters based on market regime."""
        config = PriceGeneratorConfig(
            initial_price=self.config.initial_price,
            volatility=self.config.volatility,
            drift=self.config.drift,
            tick_size=self.config.tick_size,
            update_frequency=self.config.update_frequency,
            jump_frequency=self.config.jump_frequency,
            jump_size_mean=self.config.jump_size_mean,
            jump_size_std=self.config.jump_size_std
        )
        
        if self.market_regime == MarketRegime.HIGH_VOLATILITY:
            config.volatility *= 2.5
            config.jump_frequency *= 2
        elif self.market_regime == MarketRegime.TRENDING_UP:
            config.drift *= 3
        elif self.market_regime == MarketRegime.TRENDING_DOWN:
            config.drift *= -3
        elif self.market_regime == MarketRegime.RANGE_BOUND:
            config.volatility *= 0.5
            config.drift = 0
        
        return config
    
    def _apply_intraday_patterns(self, config: PriceGeneratorConfig) -> PriceGeneratorConfig:
        """Apply U-shaped intraday volatility pattern."""
        # Time from market open (0 to 1)
        time_fraction = (self.current_time_of_day - self.market_open) / (self.market_close - self.market_open)
        time_fraction = max(0, min(1, time_fraction))
        
        # U-shaped multiplier (high at open/close)
        u_shape = 1 + 0.5 * (np.cos(2 * np.pi * time_fraction) + 1)
        
        config.volatility *= u_shape
        return config
    
    def _generate_volume(self) -> int:
        """Generate realistic volume."""
        base_volume = 1000000
        
        # Intraday volume pattern
        if self.intraday_enabled:
            time_fraction = (self.current_time_of_day - self.market_open) / (self.market_close - self.market_open)
            time_fraction = max(0, min(1, time_fraction))
            
            # U-shaped volume
            volume_multiplier = 1 + 2 * (np.cos(2 * np.pi * time_fraction) + 1)
            base_volume *= volume_multiplier
        
        # Regime adjustments
        if self.market_regime == MarketRegime.HIGH_VOLATILITY:
            base_volume *= 2
        
        # Add randomness
        volume = int(base_volume * (1 + 0.3 * np.random.randn()))
        return max(100, volume)  # Minimum volume
    
    def _create_halted_update(self) -> PriceUpdate:
        """Create update for halted trading."""
        return PriceUpdate(
            symbol=self.symbol,
            price=self.current_price,
            volume=0,
            timestamp=time.time(),
            halted=True,
            regime=self.market_regime
        )
    
    def _apply_flash_crash(self, price: float) -> float:
        """Apply flash crash dynamics."""
        elapsed = time.time() - self.crash_start_time
        
        if elapsed > self.crash_recovery_time:
            self.flash_crash_active = False
            return price
        
        # Crash and recovery pattern
        crash_fraction = elapsed / self.crash_recovery_time
        
        if crash_fraction < 0.2:  # Initial crash phase
            # Immediate drop to target level
            crash_factor = 1 - self.crash_magnitude
        elif crash_fraction < 0.3:  # Bottom phase
            # Stay at bottom briefly
            crash_factor = 1 - self.crash_magnitude
        else:  # Recovery phase
            recovery_fraction = (crash_fraction - 0.3) / 0.7
            crash_factor = (1 - self.crash_magnitude) + self.crash_magnitude * recovery_fraction
        
        # Apply crash factor to base price, not the new price
        return self.config.initial_price * crash_factor
    
    def set_market_regime(self, regime: MarketRegime):
        """Set the current market regime."""
        self.market_regime = regime
    
    def enable_intraday_patterns(self, market_open: float, market_close: float):
        """Enable intraday patterns."""
        self.intraday_enabled = True
        self.market_open = market_open
        self.market_close = market_close
    
    def set_time_of_day(self, time_of_day: float):
        """Set current time of day for intraday patterns."""
        self.current_time_of_day = time_of_day
    
    def set_circuit_breaker(self, threshold: float):
        """Set circuit breaker threshold."""
        self.circuit_breaker_threshold = threshold
    
    def trigger_flash_crash(self, crash_magnitude: float, recovery_time: float):
        """Trigger a flash crash event."""
        self.flash_crash_active = True
        self.crash_start_time = time.time()
        self.crash_magnitude = crash_magnitude
        self.crash_recovery_time = recovery_time
    
    def set_order_flow_impact(self, impact_coefficient: float, decay_rate: float):
        """Configure order flow impact on prices."""
        self.impact_coefficient = impact_coefficient
        self.impact_decay = decay_rate
    
    def generate_update_with_order_flow(self, net_order_flow: float) -> PriceUpdate:
        """Generate update with order flow impact."""
        # Calculate price impact
        self.order_flow_impact = self.impact_coefficient * net_order_flow / 1000000
        return self.generate_update()
    
    def enable_volatility_clustering(self, persistence: float, mean_reversion: float):
        """Enable GARCH-like volatility clustering."""
        self.volatility_clustering = True
        self.vol_persistence = persistence
        self.vol_mean_reversion = mean_reversion
    
    def reset(self, initial_price: Optional[float] = None):
        """Reset generator to initial state."""
        if initial_price:
            self.current_price = initial_price
            self.config.initial_price = initial_price
        else:
            self.current_price = self.config.initial_price
        
        self.is_halted = False
        self.order_flow_impact = 0.0
        self.current_vol_multiplier = 1.0
    
    async def stream_prices(self, duration: float) -> AsyncIterator[PriceUpdate]:
        """Stream prices asynchronously for given duration."""
        start_time = time.time()
        interval = 1.0 / self.config.update_frequency
        
        while time.time() - start_time < duration:
            yield self.generate_update()
            await asyncio.sleep(interval)
    
    @classmethod
    def create_correlated(cls, symbols: List[str], correlation_matrix: np.ndarray,
                         base_config: PriceGeneratorConfig) -> Dict[str, "PriceGenerator"]:
        """Create multiple correlated price generators."""
        n = len(symbols)
        generators = {}
        
        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(correlation_matrix)
        
        # Create generators with correlated random streams
        for i, symbol in enumerate(symbols):
            generator = cls(symbol, base_config)
            generator._correlation_weights = L[i]
            generator._correlated_generators = generators
            generators[symbol] = generator
        
        return generators
    
    @classmethod
    def generate_correlated_updates(cls, generators: Dict[str, "PriceGenerator"]) -> List[PriceUpdate]:
        """Generate correlated price updates."""
        # Generate independent shocks
        n = len(generators)
        independent_shocks = np.random.randn(n)
        
        # Apply correlation
        updates = []
        for i, (symbol, generator) in enumerate(generators.items()):
            if hasattr(generator, '_correlation_weights'):
                # Apply correlated shock
                correlated_shock = np.dot(generator._correlation_weights, independent_shocks)
                
                # Create a custom RNG that returns our correlated shock
                class CorrelatedRNG:
                    def __init__(self, shock):
                        self.shock = shock
                        self.used = False
                    
                    def normal(self, loc, scale):
                        if not self.used:
                            self.used = True
                            return self.shock
                        return np.random.normal(loc, scale)
                    
                    def random(self):
                        return np.random.random()
                
                # Temporarily override the random generator
                original_rng = generator.model._rng
                generator.model._rng = CorrelatedRNG(correlated_shock)
                
                update = generator.generate_update()
                generator.model._rng = original_rng
                updates.append(update)
            else:
                updates.append(generator.generate_update())
        
        return updates
    
    @classmethod
    def from_historical(cls, symbol: str, prices: List[float], 
                       volumes: List[int], timestamps: List[float]) -> "PriceGenerator":
        """Create generator that replays historical data."""
        config = PriceGeneratorConfig(
            initial_price=prices[0],
            volatility=np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252),
            drift=0.0
        )
        
        generator = cls(symbol, config)
        generator.historical_mode = True
        generator.historical_data = {
            'prices': prices,
            'volumes': volumes,
            'timestamps': timestamps
        }
        return generator
    
    def replay_next(self) -> PriceUpdate:
        """Replay next historical data point."""
        if not self.historical_mode or self.replay_index >= len(self.historical_data['prices']):
            raise ValueError("No more historical data")
        
        update = PriceUpdate(
            symbol=self.symbol,
            price=self.historical_data['prices'][self.replay_index],
            volume=self.historical_data['volumes'][self.replay_index],
            timestamp=self.historical_data['timestamps'][self.replay_index]
        )
        
        self.replay_index += 1
        return update