"""Strategy performance benchmark implementation for AI News Trading platform."""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StrategyPerformanceResult:
    """Result container for strategy performance measurements."""
    
    strategy_name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    beta: float
    alpha: float
    calmar_ratio: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "strategy_name": self.strategy_name,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_duration": self.avg_trade_duration,
            "volatility": self.volatility,
            "beta": self.beta,
            "alpha": self.alpha,
            "calmar_ratio": self.calmar_ratio,
        }


@dataclass
class Trade:
    """Individual trade record."""
    
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    return_pct: float


class StrategyBenchmark:
    """Benchmarks trading strategy performance."""
    
    def __init__(self, config):
        """Initialize strategy benchmark."""
        self.config = config
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
    def benchmark_strategy(
        self,
        strategy_name: str,
        price_data: Optional[np.ndarray] = None,
        duration_days: int = 365,
        initial_capital: float = 100000.0
    ) -> StrategyPerformanceResult:
        """Benchmark a trading strategy."""
        
        # Generate or use provided price data
        if price_data is None:
            price_data = self._generate_synthetic_price_data(duration_days)
        
        # Generate trades based on strategy
        trades = self._generate_strategy_trades(strategy_name, price_data, initial_capital)
        
        # Calculate performance metrics
        returns = self._calculate_returns(trades, price_data, initial_capital)
        
        return self._calculate_performance_metrics(
            strategy_name, trades, returns, duration_days
        )
    
    def compare_strategies(
        self,
        strategy_names: List[str],
        duration_days: int = 365,
        initial_capital: float = 100000.0
    ) -> Dict[str, StrategyPerformanceResult]:
        """Compare multiple strategies on the same data."""
        # Generate common price data for fair comparison
        price_data = self._generate_synthetic_price_data(duration_days)
        
        results = {}
        for strategy_name in strategy_names:
            results[strategy_name] = self.benchmark_strategy(
                strategy_name, price_data, duration_days, initial_capital
            )
        
        return results
    
    def _generate_synthetic_price_data(self, duration_days: int) -> np.ndarray:
        """Generate synthetic price data for benchmarking."""
        np.random.seed(42)  # For reproducible results
        
        # Parameters for geometric Brownian motion
        mu = 0.0002  # Daily drift (about 5% annual)
        sigma = 0.02  # Daily volatility (about 32% annual)
        initial_price = 100.0
        
        # Generate price path
        dt = 1.0  # Daily steps
        steps = duration_days
        
        # Random walks
        dW = np.random.normal(0, math.sqrt(dt), steps)
        
        # Geometric Brownian motion
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW
        log_prices = np.cumsum(log_returns)
        prices = initial_price * np.exp(log_prices)
        
        # Add some regime changes for more realistic data
        regime_changes = np.random.choice([0, 1], size=steps, p=[0.95, 0.05])
        volatility_multiplier = np.where(regime_changes, 2.0, 1.0)
        
        # Apply volatility regime changes
        for i in range(1, len(prices)):
            if regime_changes[i]:
                shock = np.random.normal(0, 0.05) * volatility_multiplier[i]
                prices[i] *= (1 + shock)
        
        return prices
    
    def _generate_strategy_trades(
        self,
        strategy_name: str,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate trades based on strategy logic."""
        if strategy_name == "momentum":
            return self._momentum_strategy_trades(price_data, initial_capital)
        elif strategy_name == "mean_reversion":
            return self._mean_reversion_strategy_trades(price_data, initial_capital)
        elif strategy_name == "arbitrage":
            return self._arbitrage_strategy_trades(price_data, initial_capital)
        elif strategy_name == "buy_and_hold":
            return self._buy_and_hold_strategy_trades(price_data, initial_capital)
        elif strategy_name == "mirror":
            return self._mirror_strategy_trades(price_data, initial_capital)
        elif strategy_name == "swing":
            return self._swing_strategy_trades(price_data, initial_capital)
        elif strategy_name == "swing_optimized":
            return self._swing_optimized_strategy_trades(price_data, initial_capital)
        elif strategy_name == "mean_reversion_optimized":
            return self._mean_reversion_optimized_strategy_trades(price_data, initial_capital)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _momentum_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate momentum strategy trades."""
        trades = []
        lookback = 20
        threshold = 0.02
        position = None
        position_size = 0.1  # 10% of capital per trade
        
        for i in range(lookback, len(price_data) - 1):
            if i < lookback:
                continue
                
            # Calculate momentum
            momentum = (price_data[i] - price_data[i - lookback]) / price_data[i - lookback]
            
            current_price = price_data[i]
            next_price = price_data[i + 1]
            
            # Entry signals
            if position is None:
                if momentum > threshold:
                    # Long position
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'long',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity
                    }
                elif momentum < -threshold:
                    # Short position
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'short',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity
                    }
            
            # Exit signals (reverse momentum or stop loss)
            elif position is not None:
                should_exit = False
                
                if position['side'] == 'long':
                    # Exit long if momentum reverses or stop loss
                    if momentum < 0 or (current_price / position['entry_price']) < 0.95:
                        should_exit = True
                else:  # short position
                    # Exit short if momentum reverses or stop loss
                    if momentum > 0 or (current_price / position['entry_price']) > 1.05:
                        should_exit = True
                
                if should_exit:
                    # Calculate PnL
                    if position['side'] == 'long':
                        pnl = position['quantity'] * (current_price - position['entry_price'])
                        return_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:  # short
                        pnl = position['quantity'] * (position['entry_price'] - current_price)
                        return_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        quantity=position['quantity'],
                        side=position['side'],
                        pnl=pnl,
                        return_pct=return_pct
                    )
                    trades.append(trade)
                    position = None
        
        return trades
    
    def _mean_reversion_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate enhanced mean reversion strategy trades with advanced algorithms."""
        trades = []
        
        # Multi-timeframe parameters
        short_window, medium_window, long_window = 20, 50, 100
        min_window = max(short_window, medium_window, long_window)
        
        # Advanced thresholds
        base_z_threshold = 2.0
        rsi_overbought, rsi_oversold = 70, 30
        bb_multiplier = 2.0
        volume_threshold = 1.2  # 20% above average volume
        
        position = None
        position_size = 0.05  # 5% of capital per trade
        
        # Calculate RSI helper function
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50
            deltas = np.diff(prices[-period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        # Pre-calculate volume moving average for efficiency
        volume_data = np.random.uniform(0.8, 1.2, len(price_data))  # Simulated volume
        
        for i in range(min_window, len(price_data) - 1):
            current_price = price_data[i]
            
            # Multi-timeframe z-score analysis
            short_prices = price_data[i - short_window:i]
            medium_prices = price_data[i - medium_window:i]
            long_prices = price_data[i - long_window:i]
            
            # Calculate z-scores for different timeframes
            short_ma, short_std = np.mean(short_prices), np.std(short_prices)
            medium_ma, medium_std = np.mean(medium_prices), np.std(medium_prices)
            long_ma, long_std = np.mean(long_prices), np.std(long_prices)
            
            short_z = (current_price - short_ma) / short_std if short_std > 0 else 0
            medium_z = (current_price - medium_ma) / medium_std if medium_std > 0 else 0
            long_z = (current_price - long_ma) / long_std if long_std > 0 else 0
            
            # Bollinger Bands calculation
            bb_upper = medium_ma + (bb_multiplier * medium_std)
            bb_lower = medium_ma - (bb_multiplier * medium_std)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # RSI calculation
            rsi = calculate_rsi(price_data[max(0, i-20):i+1])
            
            # Volume confirmation
            recent_volume = np.mean(volume_data[max(0, i-10):i+1])
            avg_volume = np.mean(volume_data[max(0, i-50):i+1])
            volume_confirmation = recent_volume > (avg_volume * volume_threshold)
            
            # Market regime detection (volatility-based)
            recent_volatility = np.std(price_data[max(0, i-20):i+1])
            long_volatility = np.std(price_data[max(0, i-100):i+1])
            volatility_regime = recent_volatility / long_volatility if long_volatility > 0 else 1
            
            # Adaptive threshold based on regime
            adaptive_threshold = base_z_threshold * (1 + min(volatility_regime * 0.3, 0.5))
            
            # Enhanced entry signals with multi-confirmation
            if position is None:
                # Long signal: Multi-factor confirmation
                long_signal = (
                    medium_z < -adaptive_threshold and  # Primary z-score signal
                    short_z < -1.5 and  # Short-term confirmation
                    bb_position < 0.1 and  # Below lower Bollinger Band
                    rsi < rsi_oversold and  # RSI oversold
                    volume_confirmation  # Volume spike
                )
                
                # Short signal: Multi-factor confirmation
                short_signal = (
                    medium_z > adaptive_threshold and  # Primary z-score signal
                    short_z > 1.5 and  # Short-term confirmation
                    bb_position > 0.9 and  # Above upper Bollinger Band
                    rsi > rsi_overbought and  # RSI overbought
                    volume_confirmation  # Volume spike
                )
                
                if long_signal:
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'long',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_z_score': medium_z,
                        'entry_rsi': rsi,
                        'target_price': medium_ma,  # Target mean reversion
                        'stop_loss': current_price * 0.95,  # 5% stop loss
                        'partial_exits': []
                    }
                elif short_signal:
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'short',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_z_score': medium_z,
                        'entry_rsi': rsi,
                        'target_price': medium_ma,  # Target mean reversion
                        'stop_loss': current_price * 1.05,  # 5% stop loss
                        'partial_exits': []
                    }
            
            # Enhanced exit logic with multiple conditions
            elif position is not None:
                should_exit = False
                exit_reason = ""
                
                # Time-based decay (positions older than 20 periods)
                if i - position['entry_time'] > 20:
                    should_exit = True
                    exit_reason = "time_decay"
                
                # Stop loss
                if position['side'] == 'long' and current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif position['side'] == 'short' and current_price >= position['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss"
                
                # Profit target (mean reversion achieved)
                if position['side'] == 'long':
                    if current_price >= position['target_price'] and medium_z > -0.5:
                        should_exit = True
                        exit_reason = "profit_target"
                elif position['side'] == 'short':
                    if current_price <= position['target_price'] and medium_z < 0.5:
                        should_exit = True
                        exit_reason = "profit_target"
                
                # Momentum reversal signal (trend change)
                if position['side'] == 'long' and short_z > 1.0 and rsi > 60:
                    should_exit = True
                    exit_reason = "momentum_reversal"
                elif position['side'] == 'short' and short_z < -1.0 and rsi < 40:
                    should_exit = True
                    exit_reason = "momentum_reversal"
                
                # Regime change (volatility spike)
                if volatility_regime > 2.0:
                    should_exit = True
                    exit_reason = "regime_change"
                
                if should_exit:
                    # Calculate PnL
                    if position['side'] == 'long':
                        pnl = position['quantity'] * (current_price - position['entry_price'])
                        return_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = position['quantity'] * (position['entry_price'] - current_price)
                        return_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        quantity=position['quantity'],
                        side=position['side'],
                        pnl=pnl,
                        return_pct=return_pct
                    )
                    trades.append(trade)
                    position = None
        
        return trades
    
    def _mean_reversion_optimized_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate optimized mean reversion strategy trades with enhanced signal generation."""
        trades = []
        position = None
        
        # Optimized parameters based on backtesting
        lookback_window = 40
        z_threshold = 1.8
        position_size = 0.10  # 10% per trade
        stop_loss_pct = 0.04  # 4% stop loss
        take_profit_pct = 0.05  # 5% take profit
        max_hold_days = 10
        
        for i in range(lookback_window, len(price_data) - 1):
            current_price = price_data[i]
            
            # Calculate multiple mean reversion signals
            lookback_prices = price_data[i-lookback_window:i]
            
            # Signal 1: Z-Score with exponential weighting (stronger signal)
            half_life = 8
            weights = np.exp(-np.arange(len(lookback_prices))[::-1] / half_life)
            weights /= weights.sum()
            
            ewm_mean = np.average(lookback_prices, weights=weights)
            ewm_var = np.average((lookback_prices - ewm_mean)**2, weights=weights)
            ewm_std = np.sqrt(ewm_var)
            
            z_score_signal = 0.0
            if ewm_std > 0:
                z_score = (current_price - ewm_mean) / ewm_std
                if abs(z_score) > z_threshold:
                    z_score_signal = -np.clip(z_score / 3.0, -1.0, 1.0)
            
            # Signal 2: Bollinger Band reversal
            bb_mean = np.mean(lookback_prices)
            bb_std = np.std(lookback_prices)
            bollinger_signal = 0.0
            
            if bb_std > 0:
                upper_band = bb_mean + (2.0 * bb_std)
                lower_band = bb_mean - (2.0 * bb_std)
                
                if current_price > upper_band:
                    bollinger_signal = -0.8  # Strong short signal
                elif current_price < lower_band:
                    bollinger_signal = 0.8   # Strong long signal
                elif current_price > bb_mean + (1.5 * bb_std):
                    bollinger_signal = -0.4  # Moderate short signal
                elif current_price < bb_mean - (1.5 * bb_std):
                    bollinger_signal = 0.4   # Moderate long signal
            
            # Signal 3: RSI extremes
            rsi_signal = 0.0
            if len(lookback_prices) >= 15:
                returns = np.diff(lookback_prices)
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                
                if len(gains) > 0 and len(losses) > 0:
                    avg_gain = np.mean(gains[-14:])
                    avg_loss = np.mean(losses[-14:])
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        if rsi > 80:  # Very overbought
                            rsi_signal = -0.7
                        elif rsi > 70:  # Overbought
                            rsi_signal = -0.4
                        elif rsi < 20:  # Very oversold
                            rsi_signal = 0.7
                        elif rsi < 30:  # Oversold
                            rsi_signal = 0.4
            
            # Signal 4: Price velocity reversal
            velocity_signal = 0.0
            if len(lookback_prices) >= 5:
                recent_change = (current_price - lookback_prices[-5]) / lookback_prices[-5]
                if recent_change > 0.05:  # Sharp rise
                    velocity_signal = -0.5
                elif recent_change < -0.05:  # Sharp fall
                    velocity_signal = 0.5
            
            # Combine signals with optimized weights
            signals = [z_score_signal, bollinger_signal, rsi_signal, velocity_signal]
            weights = [0.40, 0.35, 0.15, 0.10]  # Z-score and Bollinger get most weight
            combined_signal = sum(s * w for s, w in zip(signals, weights))
            
            # Calculate signal confidence based on agreement
            non_zero_signals = [s for s in signals if abs(s) > 0.1]
            if len(non_zero_signals) >= 2:
                signal_agreement = 1.0 - min(np.std(non_zero_signals) / 0.6, 1.0)
                signal_strength = min(abs(combined_signal) * 1.5, 1.0)
                confidence = (signal_agreement + signal_strength) / 2
            else:
                confidence = 0.2
            
            # Calculate market volatility for position sizing
            recent_returns = np.diff(lookback_prices[-10:]) / lookback_prices[-10:-1]
            volatility = np.std(recent_returns) if len(recent_returns) > 1 else 0.02
            vol_adjustment = min(0.02 / max(volatility, 0.01), 2.0)  # Reduce size in high vol
            
            # Entry logic with reasonable thresholds
            if position is None and confidence > 0.5 and abs(combined_signal) > 0.3:
                
                # Dynamic position sizing based on signal strength and volatility
                signal_strength_mult = min(abs(combined_signal) * 1.5, 1.2)
                confidence_mult = confidence
                adjusted_position_size = position_size * signal_strength_mult * confidence_mult * vol_adjustment
                adjusted_position_size = max(0.02, min(adjusted_position_size, 0.15))  # 2-15% range
                
                # Calculate stop loss and take profit
                if combined_signal > 0:  # Long position
                    stop_loss = current_price * (1 - stop_loss_pct)
                    take_profit = current_price * (1 + take_profit_pct * abs(combined_signal))
                else:  # Short position
                    stop_loss = current_price * (1 + stop_loss_pct)
                    take_profit = current_price * (1 - take_profit_pct * abs(combined_signal))
                
                quantity = (initial_capital * adjusted_position_size) / current_price
                
                position = {
                    'side': 'long' if combined_signal > 0 else 'short',
                    'entry_time': i,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'signal_strength': combined_signal,
                    'confidence': confidence,
                    'max_hold_days': max_hold_days,
                    'best_price': current_price
                }
            
            # Position management
            elif position is not None:
                days_held = i - position['entry_time']
                
                # Update best price for trailing
                if position['side'] == 'long':
                    if current_price > position['best_price']:
                        position['best_price'] = current_price
                        # Dynamic trailing stop after 2% profit
                        if (current_price - position['entry_price']) / position['entry_price'] > 0.02:
                            position['stop_loss'] = position['best_price'] * 0.98
                else:  # short
                    if current_price < position['best_price']:
                        position['best_price'] = current_price
                        # Dynamic trailing stop after 2% profit
                        if (position['entry_price'] - current_price) / position['entry_price'] > 0.02:
                            position['stop_loss'] = position['best_price'] * 1.02
                
                should_exit = False
                
                # Stop loss check
                if ((position['side'] == 'long' and current_price <= position['stop_loss']) or
                    (position['side'] == 'short' and current_price >= position['stop_loss'])):
                    should_exit = True
                
                # Take profit check
                elif ((position['side'] == 'long' and current_price >= position['take_profit']) or
                      (position['side'] == 'short' and current_price <= position['take_profit'])):
                    should_exit = True
                
                # Maximum holding period
                elif days_held >= position['max_hold_days']:
                    should_exit = True
                
                # Mean reversion check - exit if we've captured significant reversal
                elif days_held >= 2:
                    if position['side'] == 'long':
                        profit_pct = (current_price - position['entry_price']) / position['entry_price']
                        if profit_pct > 0.025:  # 2.5% profit
                            should_exit = True
                    else:  # short
                        profit_pct = (position['entry_price'] - current_price) / position['entry_price']
                        if profit_pct > 0.025:  # 2.5% profit
                            should_exit = True
                
                if should_exit:
                    # Calculate trade result
                    if position['side'] == 'long':
                        pnl = position['quantity'] * (current_price - position['entry_price'])
                        return_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:  # short
                        pnl = position['quantity'] * (position['entry_price'] - current_price)
                        return_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        quantity=position['quantity'],
                        side=position['side'],
                        pnl=pnl,
                        return_pct=return_pct
                    )
                    trades.append(trade)
                    position = None
        
        return trades
    
    def _arbitrage_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate arbitrage strategy trades (simulated)."""
        trades = []
        position_size = 0.02  # 2% of capital per trade
        
        # Simulate arbitrage opportunities (price inefficiencies)
        for i in range(len(price_data) - 2):
            # Simulate finding arbitrage opportunity (1% chance per day)
            if np.random.random() < 0.01:
                entry_price = price_data[i]
                # Arbitrage typically has small but consistent profits
                profit_margin = np.random.uniform(0.001, 0.005)  # 0.1% to 0.5%
                exit_price = entry_price * (1 + profit_margin)
                
                quantity = (initial_capital * position_size) / entry_price
                pnl = quantity * profit_margin * entry_price
                
                trade = Trade(
                    entry_time=i,
                    exit_time=i + 1,  # Very short duration
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    side='long',
                    pnl=pnl,
                    return_pct=profit_margin
                )
                trades.append(trade)
        
        return trades
    
    def _buy_and_hold_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate buy and hold strategy trade."""
        entry_price = price_data[0]
        exit_price = price_data[-1]
        quantity = initial_capital / entry_price
        pnl = quantity * (exit_price - entry_price)
        return_pct = (exit_price - entry_price) / entry_price
        
        trade = Trade(
            entry_time=0,
            exit_time=len(price_data) - 1,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side='long',
            pnl=pnl,
            return_pct=return_pct
        )
        
        return [trade]

    def _mirror_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate mirror strategy trades based on institutional-like signals."""
        trades = []
        position = None
        position_size = 0.08  # 8% of capital per trade (institutional style)
        
        # Mirror trading parameters
        signal_window = 15  # Look for institutional-like patterns
        confidence_threshold = 0.65
        tracking_delay = 2  # Days delay in getting institutional signals
        
        for i in range(signal_window + tracking_delay, len(price_data) - 1):
            current_price = price_data[i]
            
            # Simulate institutional signal detection
            # Look for large volume moves and institutional patterns
            recent_volatility = np.std(price_data[i-signal_window:i])
            price_momentum = (price_data[i] - price_data[i-signal_window]) / price_data[i-signal_window]
            
            # Simulate institutional confidence based on price patterns
            institutional_confidence = self._calculate_institutional_confidence(
                price_data[i-signal_window:i], price_momentum, recent_volatility
            )
            
            # Entry signals based on simulated institutional activity
            if position is None and institutional_confidence > confidence_threshold:
                if price_momentum > 0.02:  # Institutional buying signal
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'long',
                        'entry_time': i + tracking_delay,  # Delay in following
                        'entry_price': current_price * 1.005,  # Slight slippage
                        'quantity': quantity,
                        'confidence': institutional_confidence
                    }
                elif price_momentum < -0.02:  # Institutional selling signal
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'short',
                        'entry_time': i + tracking_delay,
                        'entry_price': current_price * 0.995,  # Slight slippage
                        'quantity': quantity,
                        'confidence': institutional_confidence
                    }
            
            # Exit signals based on institutional behavior
            elif position is not None:
                should_exit = False
                days_held = i - position['entry_time']
                position_return = (current_price - position['entry_price']) / position['entry_price']
                
                # Mirror typical institutional holding periods and exits
                if position['side'] == 'long':
                    # Exit on institutional profit taking or position reversal
                    if (position_return > 0.15 and days_held > 30) or position_return < -0.08:
                        should_exit = True
                else:  # short position
                    if (position_return > 0.12 and days_held > 20) or position_return < -0.10:
                        should_exit = True
                
                # Exit after maximum holding period (institutional style)
                if days_held > 90:  # 90 days max hold
                    should_exit = True
                
                if should_exit:
                    # Calculate PnL with institutional-style execution
                    execution_price = current_price * (0.998 if position['side'] == 'long' else 1.002)
                    
                    if position['side'] == 'long':
                        pnl = position['quantity'] * (execution_price - position['entry_price'])
                        return_pct = (execution_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = position['quantity'] * (position['entry_price'] - execution_price)
                        return_pct = (position['entry_price'] - execution_price) / position['entry_price']
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=execution_price,
                        quantity=position['quantity'],
                        side=position['side'],
                        pnl=pnl,
                        return_pct=return_pct
                    )
                    trades.append(trade)
                    position = None
        
        return trades

    def _swing_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate swing strategy trades."""
        trades = []
        position = None
        position_size = 0.06  # 6% of capital per trade
        
        # Swing trading parameters
        short_window = 10
        long_window = 30
        rsi_window = 14
        
        for i in range(long_window, len(price_data) - 1):
            current_price = price_data[i]
            
            # Calculate moving averages
            short_ma = np.mean(price_data[i-short_window:i])
            long_ma = np.mean(price_data[i-long_window:i])
            
            # Calculate RSI
            rsi = self._calculate_rsi(price_data[i-rsi_window:i+1])
            
            # Calculate price momentum
            momentum = (current_price - price_data[i-5]) / price_data[i-5]
            
            # Entry signals
            if position is None:
                # Bullish swing setup
                if (short_ma > long_ma and rsi < 70 and momentum > 0.01 and 
                    current_price > short_ma * 1.02):
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'long',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_rsi': rsi
                    }
                # Bearish swing setup
                elif (short_ma < long_ma and rsi > 30 and momentum < -0.01 and 
                      current_price < short_ma * 0.98):
                    quantity = (initial_capital * position_size) / current_price
                    position = {
                        'side': 'short',
                        'entry_time': i,
                        'entry_price': current_price,
                        'quantity': quantity,
                        'entry_rsi': rsi
                    }
            
            # Exit signals
            elif position is not None:
                should_exit = False
                position_return = (current_price - position['entry_price']) / position['entry_price']
                days_held = i - position['entry_time']
                
                if position['side'] == 'long':
                    # Exit long position
                    if (rsi > 75 or short_ma < long_ma or 
                        position_return > 0.08 or position_return < -0.04 or
                        days_held > 15):
                        should_exit = True
                else:
                    # Exit short position
                    if (rsi < 25 or short_ma > long_ma or 
                        position_return > 0.06 or position_return < -0.05 or
                        days_held > 12):
                        should_exit = True
                
                if should_exit:
                    if position['side'] == 'long':
                        pnl = position['quantity'] * (current_price - position['entry_price'])
                        return_pct = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = position['quantity'] * (position['entry_price'] - current_price)
                        return_pct = (position['entry_price'] - current_price) / position['entry_price']
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        quantity=position['quantity'],
                        side=position['side'],
                        pnl=pnl,
                        return_pct=return_pct
                    )
                    trades.append(trade)
                    position = None
        
        return trades

    def _swing_optimized_strategy_trades(
        self,
        price_data: np.ndarray,
        initial_capital: float
    ) -> List[Trade]:
        """Generate optimized swing strategy trades with advanced pattern recognition."""
        trades = []
        position = None
        portfolio_heat = 0.0
        max_portfolio_heat = 0.06  # 6% max portfolio risk
        consecutive_losses = 0
        
        # Optimized parameters
        atr_window = 14
        pattern_lookback = 20
        max_position_pct = 0.12  # Max 12% per position
        max_risk_per_trade = 0.015  # 1.5% risk per trade
        
        # Calculate ATR for the entire series
        atr_values = []
        for j in range(atr_window, len(price_data)):
            high_low_diff = np.std(price_data[j-atr_window:j]) * 2.0  # Approximation
            atr_values.append(high_low_diff)
        
        for i in range(max(30, pattern_lookback), len(price_data) - 1):
            if i - atr_window >= len(atr_values):
                continue
                
            current_price = price_data[i]
            atr = atr_values[i - atr_window] if i - atr_window < len(atr_values) else current_price * 0.02
            
            # Calculate indicators
            ma_50 = np.mean(price_data[max(0, i-50):i])
            ma_20 = np.mean(price_data[max(0, i-20):i])
            rsi = self._calculate_rsi(price_data[max(0, i-14):i+1])
            
            # Calculate momentum and volatility
            momentum = (current_price - price_data[i-10]) / price_data[i-10] if i >= 10 else 0
            recent_volatility = np.std(price_data[max(0, i-20):i]) / np.mean(price_data[max(0, i-20):i])
            
            # Simulate volume (random but consistent)
            np.random.seed(i)
            volume_ratio = np.random.uniform(0.8, 2.0)
            
            # Market regime identification
            trend_strength = (current_price - ma_50) / ma_50
            if recent_volatility > 0.035:  # Slightly higher threshold
                market_regime = 'volatile'
                regime_multiplier = 0.7  # Less penalty for volatility
            elif trend_strength > 0.04:
                market_regime = 'bull'
                regime_multiplier = 1.15  # Slightly less aggressive
            elif trend_strength < -0.04:
                market_regime = 'bear'
                regime_multiplier = 0.6  # Less penalty
            else:
                market_regime = 'sideways'
                regime_multiplier = 0.85
            
            # Entry logic with pattern recognition
            if position is None and portfolio_heat < max_portfolio_heat:
                patterns_detected = []
                
                # Pattern 1: Bullish Flag (relaxed conditions)
                if i >= 20:
                    trend_prices = price_data[i-20:i-10]
                    flag_prices = price_data[i-10:i]
                    trend_return = (trend_prices[-1] - trend_prices[0]) / trend_prices[0]
                    flag_volatility = np.std(flag_prices) / np.mean(flag_prices)
                    
                    if (trend_return > 0.03 and flag_volatility < 0.025 and 
                        min(flag_prices) > trend_prices[-1] * 0.96):
                        patterns_detected.append(('bullish_flag', 0.80))
                
                # Pattern 2: MA Pullback (more opportunities)
                distance_to_ma = abs(current_price - ma_50) / ma_50
                if (distance_to_ma < 0.03 and rsi < 45 and trend_strength > -0.02 and 
                    current_price > ma_50 * 0.985):
                    patterns_detected.append(('ma_pullback', 0.70))
                
                # Pattern 3: Consolidation Breakout (adjusted)
                if i >= 15:
                    consolidation_prices = price_data[i-15:i-1]
                    consolidation_range = (max(consolidation_prices) - min(consolidation_prices)) / np.mean(consolidation_prices)
                    breakout = current_price > max(consolidation_prices) * 1.005
                    
                    if consolidation_range < 0.06 and breakout and volume_ratio > 1.3:
                        patterns_detected.append(('breakout', 0.75))
                
                # Pattern 4: Momentum Surge (new pattern)
                if momentum > 0.02 and rsi > 45 and rsi < 65 and volume_ratio > 1.2:
                    patterns_detected.append(('momentum_surge', 0.72))
                
                # Take the highest confidence pattern
                if patterns_detected:
                    best_pattern = max(patterns_detected, key=lambda x: x[1])
                    pattern_type, confidence = best_pattern
                    
                    # Adjust confidence for market regime (less harsh)
                    if market_regime == 'bear':
                        confidence *= 0.8
                    elif market_regime == 'volatile':
                        confidence *= 0.85
                    
                    if confidence >= 0.55:  # Lowered from 0.65 for more opportunities
                        # Calculate dynamic position size
                        stop_loss = current_price - (1.5 * atr)
                        risk_per_share = current_price - stop_loss
                        
                        # Kelly Criterion sizing
                        win_rate = 0.60  # More realistic
                        avg_win = 2.2  # risk/reward ratio
                        kelly_fraction = (win_rate * avg_win - (1 - win_rate)) / avg_win
                        kelly_fraction = max(0.1, min(kelly_fraction, 0.25))  # Ensure minimum position
                        
                        # Calculate position size with all adjustments
                        max_risk_amount = initial_capital * max_risk_per_trade
                        base_shares = max_risk_amount / risk_per_share
                        
                        # Apply adjustments (more balanced)
                        volatility_factor = min(1.0, 0.025 / max(recent_volatility, 0.01))
                        adjusted_shares = base_shares * kelly_fraction * confidence * volatility_factor * regime_multiplier
                        
                        # Check position limits
                        position_value = adjusted_shares * current_price
                        if position_value > initial_capital * max_position_pct:
                            adjusted_shares = (initial_capital * max_position_pct) / current_price
                        
                        # Final position
                        position_risk_pct = (adjusted_shares * risk_per_share) / initial_capital
                        
                        if portfolio_heat + position_risk_pct <= max_portfolio_heat:
                            position = {
                                'side': 'long',
                                'entry_time': i,
                                'entry_price': current_price,
                                'quantity': adjusted_shares,
                                'stop_loss': stop_loss,
                                'pattern': pattern_type,
                                'entry_regime': market_regime,
                                'target1': current_price + (2 * atr),
                                'target2': current_price + (3 * atr),
                                'highest_price': current_price
                            }
                            portfolio_heat += position_risk_pct
            
            # Exit logic
            elif position is not None:
                position_return = (current_price - position['entry_price']) / position['entry_price']
                days_held = i - position['entry_time']
                
                # Update highest price for trailing stop
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                
                should_exit = False
                exit_reason = ''
                
                # Stop loss
                if current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = 'stop_loss'
                    consecutive_losses += 1
                # Target hit
                elif current_price >= position['target2']:
                    should_exit = True
                    exit_reason = 'target2'
                    consecutive_losses = 0
                elif current_price >= position['target1'] and days_held > 3:
                    should_exit = True
                    exit_reason = 'target1'
                    consecutive_losses = 0
                # Trailing stop (activated after 4% profit)
                elif position_return > 0.04:
                    trailing_stop = position['highest_price'] - (1.5 * atr)
                    if current_price <= trailing_stop:
                        should_exit = True
                        exit_reason = 'trailing_stop'
                        consecutive_losses = 0
                # Time stop
                elif days_held > 8:
                    should_exit = True
                    exit_reason = 'time_stop'
                # Adverse regime change
                elif position['entry_regime'] == 'bull' and market_regime in ['bear', 'volatile']:
                    should_exit = True
                    exit_reason = 'regime_change'
                
                if should_exit:
                    pnl = position['quantity'] * (current_price - position['entry_price'])
                    
                    # Update portfolio heat
                    position_risk_pct = (position['quantity'] * (position['entry_price'] - position['stop_loss'])) / initial_capital
                    portfolio_heat = max(0, portfolio_heat - position_risk_pct)
                    
                    trade = Trade(
                        entry_time=position['entry_time'],
                        exit_time=i,
                        entry_price=position['entry_price'],
                        exit_price=current_price,
                        quantity=position['quantity'],
                        side='long',
                        pnl=pnl,
                        return_pct=position_return
                    )
                    trades.append(trade)
                    position = None
        
        return trades

    def _calculate_institutional_confidence(self, price_window, momentum, volatility):
        """Calculate simulated institutional confidence score."""
        # Simulate institutional decision-making factors
        trend_strength = abs(momentum) * 10  # Scale momentum
        volatility_factor = max(0, 1 - volatility * 20)  # Penalize high volatility
        
        # Simulate volume and liquidity factors
        volume_factor = np.random.uniform(0.7, 1.0)  # Mock volume analysis
        
        # Combine factors
        confidence = (trend_strength * 0.4 + volatility_factor * 0.3 + volume_factor * 0.3)
        return max(0, min(confidence, 1.0))

    def _calculate_rsi(self, prices):
        """Calculate Relative Strength Index."""
        if len(prices) < 2:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001  # Avoid division by zero
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_returns(
        self,
        trades: List[Trade],
        price_data: np.ndarray,
        initial_capital: float
    ) -> np.ndarray:
        """Calculate daily returns from trades."""
        daily_returns = np.zeros(len(price_data))
        
        for trade in trades:
            # Distribute trade return over the holding period
            duration = max(1, trade.exit_time - trade.entry_time)
            daily_return = trade.return_pct / duration
            
            for day in range(int(trade.entry_time), int(trade.exit_time) + 1):
                if day < len(daily_returns):
                    daily_returns[day] += daily_return
        
        return daily_returns
    
    def _calculate_performance_metrics(
        self,
        strategy_name: str,
        trades: List[Trade],
        returns: np.ndarray,
        duration_days: int
    ) -> StrategyPerformanceResult:
        """Calculate comprehensive performance metrics."""
        if not trades:
            # Return zero metrics if no trades
            return StrategyPerformanceResult(
                strategy_name=strategy_name,
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                volatility=0.0,
                beta=0.0,
                alpha=0.0,
                calmar_ratio=0.0
            )
        
        # Basic metrics
        total_pnl = sum(trade.pnl for trade in trades)
        total_return = sum(trade.return_pct for trade in trades)
        
        # Annualized return
        years = duration_days / 365.0
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Win rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Volatility (annualized)
        daily_volatility = np.std(returns)
        volatility = daily_volatility * math.sqrt(252)  # Annualized
        
        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * math.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Average trade duration
        avg_trade_duration = np.mean([t.exit_time - t.entry_time for t in trades])
        
        # Beta and Alpha (against market - using random market returns)
        market_returns = np.random.normal(0.0005, 0.02, len(returns))  # Mock market
        
        if len(returns) > 1 and np.var(market_returns) > 0:
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
            alpha = annualized_return - (self.risk_free_rate + beta * (np.mean(market_returns) * 252 - self.risk_free_rate))
        else:
            beta = 0.0
            alpha = 0.0
        
        return StrategyPerformanceResult(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            volatility=volatility,
            beta=beta,
            alpha=alpha,
            calmar_ratio=calmar_ratio
        )


class StrategyProfiler:
    """Profiles strategy performance across different market conditions."""
    
    def __init__(self, config):
        """Initialize strategy profiler."""
        self.config = config
        self.benchmark = StrategyBenchmark(config)
    
    def profile_strategy_across_conditions(
        self,
        strategy_name: str,
        market_conditions: List[str] = None
    ) -> Dict[str, StrategyPerformanceResult]:
        """Profile strategy performance across different market conditions."""
        if market_conditions is None:
            market_conditions = ['bull', 'bear', 'sideways', 'volatile']
        
        results = {}
        
        for condition in market_conditions:
            # Generate condition-specific price data
            price_data = self._generate_market_condition_data(condition)
            
            result = self.benchmark.benchmark_strategy(
                strategy_name, price_data, duration_days=252
            )
            results[f"{strategy_name}_{condition}"] = result
        
        return results
    
    def _generate_market_condition_data(self, condition: str) -> np.ndarray:
        """Generate price data for specific market conditions."""
        np.random.seed(hash(condition) % 2**32)  # Deterministic but varied
        
        if condition == 'bull':
            mu = 0.001  # Strong positive drift
            sigma = 0.015  # Moderate volatility
        elif condition == 'bear':
            mu = -0.0008  # Negative drift
            sigma = 0.025  # Higher volatility
        elif condition == 'sideways':
            mu = 0.0001  # Very small drift
            sigma = 0.01  # Low volatility
        elif condition == 'volatile':
            mu = 0.0002  # Small drift
            sigma = 0.04  # High volatility
        else:
            mu = 0.0002
            sigma = 0.02
        
        # Generate price path
        steps = 252  # One year
        initial_price = 100.0
        
        dW = np.random.normal(0, 1, steps)
        log_returns = (mu - 0.5 * sigma**2) + sigma * dW
        log_prices = np.cumsum(log_returns)
        prices = initial_price * np.exp(log_prices)
        
        return prices