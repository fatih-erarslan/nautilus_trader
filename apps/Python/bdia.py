#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 22:28:04 2025

@author: ashina
"""

"""
Crypto Trading Agent Framework with GPU Acceleration
===================================================

This framework implements a classical agentic reasoning approach for crypto trading
with GPU acceleration for parallel processing of multiple markets and strategies.

Key components:
1. Belief System - Market data representation and processing
2. Desire System - Goal setting and performance metrics
3. Intention System - Strategy selection and execution
4. Action System - Trade execution and order management
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Callable, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta
from enum import Enum, auto
import warnings

# Optional imports for hardware acceleration
try:
    from hardware_manager import HardwareManager
    from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType
    HARDWARE_ACCEL_AVAILABLE = True
except ImportError:
    HARDWARE_ACCEL_AVAILABLE = False
    warnings.warn("Hardware acceleration modules not available. Using fallback implementation.")
    
    # Define dummy classes if the imports fail
    class HardwareManager:
        @classmethod
        def get_manager(cls, **kwargs):
            return cls(**kwargs)
            
        def __init__(self, **kwargs):
            self.quantum_available = False
            self.gpu_available = False
            
        def initialize_hardware(self):
            return False
            
        def _get_quantum_device(self, qubits):
            return {"device": "lightning.kokkos", "wires": qubits, "shots": None}
            
        def get_optimal_device(self, **kwargs):
            return {"type": "gpu", "available": True}
    
    class HardwareAccelerator:
        def __init__(self, **kwargs):
            self.gpu_available = False
            
        def get_accelerator_type(self):
            return "cpu"
            
        def get_torch_device(self):
            return None
    
    class AcceleratorType(Enum):
        CPU = auto()
        CUDA = auto()
        ROCM = auto()
        MPS = auto()

# GPU acceleration imports
try:
    import cupy as cp
    import cudf
    HAS_GPU = True
except ImportError:
    cp = np
    HAS_GPU = False
    logging.warning("GPU libraries not found. Running in CPU mode.")
    
# ======== DATA STRUCTURES ========

@dataclass
class MarketState:
    """Represents the current state of a market"""
    symbol: str
    price: float
    timestamp: float
    volume_24h: float
    high_24h: float
    low_24h: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    order_book: Optional[Dict] = None
    historical_data: Optional[pd.DataFrame] = None


@dataclass
class TradingParameters:
    """Parameters for trading strategies"""
    max_position_size: float
    risk_per_trade: float
    stop_loss_pct: float
    take_profit_pct: float
    timeframe: str
    indicators: Dict


@dataclass
class AgentBelief:
    """The agent's current beliefs about the market"""
    market_states: Dict[str, MarketState]
    portfolio: Dict[str, float]
    cash_balance: float
    indicators: Dict[str, Dict]
    correlations: Optional[np.ndarray] = None
    market_regime: str = "unknown"
    risk_level: str = "moderate"
    
    
@dataclass
class AgentDesire:
    """Goals and objectives of the agent"""
    target_return_pct: float
    max_drawdown_pct: float
    risk_reward_ratio: float
    diversification_target: int
    preferred_timeframe: str
    

@dataclass
class AgentIntention:
    """Current active intentions of the agent"""
    watchlist: List[str]
    active_strategies: Dict[str, str]
    pending_orders: Dict
    risk_management_rules: Dict
    priority_markets: List[str]
    

@dataclass
class TradeAction:
    """Represents a specific trade action to be taken"""
    symbol: str
    action: str  # "BUY", "SELL"
    quantity: float
    price: Optional[float] = None  # Market or limit price
    order_type: str = "MARKET"  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT"
    time_in_force: str = "GTC"  # "GTC", "IOC", "FOK"
    timestamp: float = time.time()
    strategy_id: str = ""
    confidence: float = 0.0


# ======== BELIEF SYSTEM ========

class BeliefSystem:
    """
    Manages the agent's beliefs about the market state and portfolio.
    Uses GPU acceleration when available for data processing.
    """
    
    def __init__(self, symbols: List[str], use_gpu: bool = True):
        self.symbols = symbols
        self.use_gpu = use_gpu and HAS_GPU
        # Initialize market_states with None or a default MarketState structure
        self.market_states: Dict[str, Optional[MarketState]] = {symbol: None for symbol in symbols}
        self.indicators = {symbol: {} for symbol in symbols}
        self.historical_data: Dict[str, Union[pd.DataFrame, 'cudf.DataFrame']] = {} # Stores historical data for each symbol
        self.portfolio: Dict[str, float] = {}
        self.cash_balance: float = 0.0
        self.risk_level: str = "moderate" # Default risk level
        self.last_update: float = 0.0
        self.overall_market_regime: str = "unknown" # Added for storing overall regime
        self.symbol_specific_regimes: Dict[str, str] = {} # Added for storing per-symbol regimes
        self.correlations_matrix: Optional[np.ndarray] = None # Added for storing correlations
        
    def update_market_data(self, market_data: Dict):
        """Update market states with new data (primarily ticker/price data)"""
        self.last_update = time.time()
        
        for symbol, data in market_data.items():
            if symbol in self.symbols:
                # Create or update MarketState. Historical data is linked in get_current_beliefs.
                current_historical_data = self.historical_data.get(symbol)
                self.market_states[symbol] = MarketState(
                    symbol=symbol,
                    price=data['price'],
                    timestamp=data['timestamp'],
                    volume_24h=data['volume_24h'],
                    high_24h=data['high_24h'],
                    low_24h=data['low_24h'],
                    bid=data.get('bid'),
                    ask=data.get('ask'),
                    order_book=data.get('order_book'),
                    historical_data=current_historical_data # Link existing historical data
                )
    
    def update_historical_data(self, symbol: str, data: pd.DataFrame):
        """Update historical price data for technical analysis"""
        if self.use_gpu:
            try:
                # Convert to GPU DataFrame if using GPU
                gpu_df = cudf.DataFrame.from_pandas(data)
                self.historical_data[symbol] = gpu_df
            except Exception as e:
                logging.error(f"GPU conversion failed for {symbol}: {e}. Falling back to CPU DataFrame.")
                self.historical_data[symbol] = data
        else:
            self.historical_data[symbol] = data
        
        # Update the historical_data in the corresponding MarketState object if it exists
        if symbol in self.market_states and self.market_states[symbol] is not None:
            if self.use_gpu and HAS_GPU and isinstance(self.historical_data[symbol], cudf.DataFrame):
                 self.market_states[symbol].historical_data = self.historical_data[symbol] # type: ignore
            else:
                 self.market_states[symbol].historical_data = self.historical_data[symbol] # type: ignore
    
    def calculate_indicators(self, parallel: bool = True):
        """Calculate technical indicators for all symbols"""
        if parallel:
            with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
                futures = {executor.submit(self._calculate_symbol_indicators, symbol): 
                          symbol for symbol in self.symbols}
        else:
            for symbol in self.symbols:
                self._calculate_symbol_indicators(symbol)
    
    def _calculate_symbol_indicators(self, symbol: str):
        """Calculate indicators for a single symbol"""
        if symbol not in self.historical_data:
            return
            
        data = self.historical_data[symbol]
        
        if self.use_gpu:
            # GPU calculations
            try:
                # Moving averages
                self.indicators[symbol]['sma_20'] = self._gpu_sma(data['close'], 20)
                self.indicators[symbol]['sma_50'] = self._gpu_sma(data['close'], 50)
                self.indicators[symbol]['sma_200'] = self._gpu_sma(data['close'], 200)
                
                # Exponential moving average
                self.indicators[symbol]['ema_12'] = self._gpu_ema(data['close'], 12)
                self.indicators[symbol]['ema_26'] = self._gpu_ema(data['close'], 26)
                
                # MACD
                self.indicators[symbol]['macd'] = (
                    self.indicators[symbol]['ema_12'] - self.indicators[symbol]['ema_26']
                )
                self.indicators[symbol]['macd_signal'] = self._gpu_ema(
                    self.indicators[symbol]['macd'], 9
                )
                
                # RSI
                self.indicators[symbol]['rsi'] = self._gpu_rsi(data['close'], 14)
                
                # Bollinger Bands
                self.indicators[symbol]['bb_middle'] = self.indicators[symbol]['sma_20']
                std = self._gpu_std(data['close'], 20)
                self.indicators[symbol]['bb_upper'] = self.indicators[symbol]['bb_middle'] + (std * 2)
                self.indicators[symbol]['bb_lower'] = self.indicators[symbol]['bb_middle'] - (std * 2)
                
                # Convert back to CPU for compatibility with rest of system
                for key, value in self.indicators[symbol].items():
                    if hasattr(value, 'get'):  # Check if it's a GPU array
                        self.indicators[symbol][key] = value.get()
                        
            except Exception as e:
                logging.error(f"GPU indicator calculation failed: {e}")
                self._cpu_calculate_indicators(symbol, data)
        else:
            # CPU calculations
            self._cpu_calculate_indicators(symbol, data)
    
    def _cpu_calculate_indicators(self, symbol: str, data: pd.DataFrame):
        """Calculate indicators using CPU"""
        # Moving averages
        self.indicators[symbol]['sma_20'] = data['close'].rolling(20).mean().values
        self.indicators[symbol]['sma_50'] = data['close'].rolling(50).mean().values
        self.indicators[symbol]['sma_200'] = data['close'].rolling(200).mean().values
        
        # Exponential moving average
        self.indicators[symbol]['ema_12'] = data['close'].ewm(span=12, adjust=False).mean().values
        self.indicators[symbol]['ema_26'] = data['close'].ewm(span=26, adjust=False).mean().values
        
        # MACD
        self.indicators[symbol]['macd'] = (
            self.indicators[symbol]['ema_12'] - self.indicators[symbol]['ema_26']
        )
        self.indicators[symbol]['macd_signal'] = pd.Series(
            self.indicators[symbol]['macd']
        ).ewm(span=9, adjust=False).mean().values
        
        # RSI
        delta = data['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        self.indicators[symbol]['rsi'] = 100 - (100 / (1 + rs)).values
        
        # Bollinger Bands
        self.indicators[symbol]['bb_middle'] = self.indicators[symbol]['sma_20']
        std = data['close'].rolling(20).std().values
        self.indicators[symbol]['bb_upper'] = self.indicators[symbol]['bb_middle'] + (std * 2)
        self.indicators[symbol]['bb_lower'] = self.indicators[symbol]['bb_middle'] - (std * 2)
    
    def _gpu_sma(self, series, window):
        """Calculate Simple Moving Average using GPU"""
        if isinstance(series, pd.Series):
            series = cp.array(series.values)
        elif not isinstance(series, cp.ndarray):
            series = cp.array(series)
            
        output = cp.zeros_like(series)
        output[:] = cp.nan
        
        for i in range(window-1, len(series)):
            output[i] = cp.mean(series[i-window+1:i+1])
            
        return output
    
    def _gpu_ema(self, series, span):
        """Calculate Exponential Moving Average using GPU"""
        if isinstance(series, pd.Series):
            series = cp.array(series.values)
        elif not isinstance(series, cp.ndarray):
            series = cp.array(series)
            
        alpha = 2 / (span + 1)
        output = cp.zeros_like(series)
        output[:] = cp.nan
        
        # Initialize with SMA
        output[span-1] = cp.mean(series[:span])
        
        # Calculate EMA
        for i in range(span, len(series)):
            output[i] = alpha * series[i] + (1 - alpha) * output[i-1]
            
        return output
    
    def _gpu_std(self, series, window):
        """Calculate Standard Deviation using GPU"""
        if isinstance(series, pd.Series):
            series = cp.array(series.values)
        elif not isinstance(series, cp.ndarray):
            series = cp.array(series)
            
        output = cp.zeros_like(series)
        output[:] = cp.nan
        
        for i in range(window-1, len(series)):
            output[i] = cp.std(series[i-window+1:i+1])
            
        return output
    
    def _gpu_rsi(self, series, window):
        """Calculate Relative Strength Index using GPU"""
        if isinstance(series, pd.Series):
            series = cp.array(series.values)
        elif not isinstance(series, cp.ndarray):
            series = cp.array(series)
            
        delta = cp.zeros_like(series)
        delta[1:] = series[1:] - series[:-1]
        
        gain = cp.zeros_like(delta)
        loss = cp.zeros_like(delta)
        
        gain = cp.where(delta > 0, delta, 0)
        loss = cp.where(delta < 0, -delta, 0)
        
        output = cp.zeros_like(series)
        output[:] = cp.nan
        
        for i in range(window, len(series)):
            avg_gain = cp.mean(gain[i-window+1:i+1])
            avg_loss = cp.mean(loss[i-window+1:i+1])
            
            if avg_loss == 0:
                output[i] = 100
            else:
                rs = avg_gain / avg_loss
                output[i] = 100 - (100 / (1 + rs))
                
        return output
    
    def calculate_correlations(self):
        """Calculate correlation matrix between assets"""
        if not self.historical_data:
            return None
            
        # Extract closing prices for each symbol
        symbols = list(self.historical_data.keys())
        if not symbols:
            return None
            
        # Prepare price data
        price_data = {}
        for symbol in symbols:
            if self.use_gpu:
                try:
                    price_data[symbol] = self.historical_data[symbol]['close'].to_array()
                except:
                    # Fallback if GPU DataFrame doesn't support to_array
                    price_data[symbol] = self.historical_data[symbol]['close'].values
            else:
                price_data[symbol] = self.historical_data[symbol]['close'].values
        
        # Create correlation matrix
        n = len(symbols)
        corr_matrix = np.zeros((n, n))
        
        if self.use_gpu:
            try:
                # Move all data to GPU
                gpu_prices = {}
                for symbol in symbols:
                    gpu_prices[symbol] = cp.array(price_data[symbol])
                
                # Calculate correlations on GPU
                for i in range(n):
                    for j in range(i, n):
                        if i == j:
                            corr_matrix[i, j] = 1.0
                        else:
                            # Calculate correlation coefficient
                            x = gpu_prices[symbols[i]]
                            y = gpu_prices[symbols[j]]
                            
                            # Remove NaN values
                            mask = cp.logical_and(~cp.isnan(x), ~cp.isnan(y))
                            x = x[mask]
                            y = y[mask]
                            
                            if len(x) > 1:  # Need at least 2 points for correlation
                                corr = cp.corrcoef(x, y)[0, 1]
                                corr_matrix[i, j] = corr
                                corr_matrix[j, i] = corr
                            else:
                                corr_matrix[i, j] = 0
                                corr_matrix[j, i] = 0
                                
                # Move result back to CPU
                corr_matrix = cp.asnumpy(corr_matrix) if hasattr(cp, 'asnumpy') else corr_matrix
                
            except Exception as e:
                logging.error(f"GPU correlation calculation failed: {e}")
                corr_matrix = self._calculate_cpu_correlations(symbols, price_data)
        else:
            corr_matrix = self._calculate_cpu_correlations(symbols, price_data)
            
        return corr_matrix
    
    def _calculate_cpu_correlations(self, symbols, price_data):
        """Calculate correlation matrix using CPU"""
        n = len(symbols)
        corr_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    # Calculate correlation coefficient
                    x = price_data[symbols[i]]
                    y = price_data[symbols[j]]
                    
                    # Create mask for valid data points
                    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 1:  # Need at least 2 points for correlation
                        corr = np.corrcoef(x_clean, y_clean)[0, 1]
                        corr_matrix[i, j] = corr
                        corr_matrix[j, i] = corr
                    else:
                        corr_matrix[i, j] = 0
                        corr_matrix[j, i] = 0
                        
        return corr_matrix
    
    def detect_market_regime(self):
        """Detect current market regime (trending, ranging, volatile)"""
        regimes = {}
        
        for symbol in self.symbols:
            if symbol not in self.historical_data:
                continue
                
            try:
                # Get relevant indicators
                if self.use_gpu:
                    close = self.historical_data[symbol]['close'].to_array()
                    high = self.historical_data[symbol]['high'].to_array()
                    low = self.historical_data[symbol]['low'].to_array()
                else:
                    close = self.historical_data[symbol]['close'].values
                    high = self.historical_data[symbol]['high'].values
                    low = self.historical_data[symbol]['low'].values
                
                # 1. Volatility - Using ATR concept
                ranges = high[-20:] - low[-20:]
                atr = np.mean(ranges)
                atr_pct = atr / close[-1] * 100
                
                # 2. Trend strength - Using moving averages
                sma_20 = self.indicators[symbol]['sma_20'][-1]
                sma_50 = self.indicators[symbol]['sma_50'][-1]
                
                # 3. RSI for overbought/oversold
                rsi = self.indicators[symbol]['rsi'][-1]
                
                # Determine regime
                if atr_pct > 5:  # High volatility
                    regimes[symbol] = "volatile"
                elif abs(sma_20 - sma_50) / sma_50 * 100 > 2:  # Trending
                    if sma_20 > sma_50:
                        regimes[symbol] = "uptrend"
                    else:
                        regimes[symbol] = "downtrend"
                else:  # Ranging
                    regimes[symbol] = "ranging"
                    
                # Override based on RSI extremes
                if rsi > 70:
                    regimes[symbol] = "overbought"
                elif rsi < 30:
                    regimes[symbol] = "oversold"
                    
            except Exception as e:
                logging.warning(f"Failed to detect market regime for {symbol}: {e}")
                regimes[symbol] = "unknown"
        
        # Determine overall market regime (simple majority)
        all_regimes = list(regimes.values())
        if all_regimes:
            # Count occurrences of each regime
            regime_counts = {}
            for regime in all_regimes:
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
            
            # Find most common regime
            overall_regime = max(regime_counts, key=regime_counts.get)
        else:
            overall_regime = "unknown"
            
        return overall_regime, regimes
    
    def get_current_beliefs(self) -> AgentBelief:
        """Get current beliefs about the market"""
        # Calculate/update market regime and correlations
        # These might be computationally intensive, consider caching or periodic updates
        self.overall_market_regime, self.symbol_specific_regimes = self.detect_market_regime()
        self.correlations_matrix = self.calculate_correlations()

        # Ensure MarketState objects within self.market_states have their historical_data field populated
        # This is now handled by update_historical_data and update_market_data linking them.
        # If a MarketState was created before historical_data was available, this ensures it's linked.
        current_market_states_for_belief = {}
        for sym, ms in self.market_states.items():
            if ms is not None:
                # Ensure historical_data attribute of MarketState instance is up-to-date
                hist_data_for_sym = self.historical_data.get(sym)
                if ms.historical_data is not hist_data_for_sym : # Check if it needs update
                    # Use dataclasses.replace to create a new instance if immutable, or directly set if mutable
                    # For simplicity, assuming direct update is fine as MarketState is a dataclass
                    ms.historical_data = hist_data_for_sym
                current_market_states_for_belief[sym] = ms
            else:
                current_market_states_for_belief[sym] = None
        
        return AgentBelief(
            market_states=current_market_states_for_belief, # Use the potentially updated market_states
            portfolio=self.portfolio,
            cash_balance=self.cash_balance,
            indicators=self.indicators, # Indicators are calculated based on self.historical_data
            correlations=self.correlations_matrix,
            market_regime=self.overall_market_regime,
            risk_level=self.risk_level
            )

# ======== DESIRE SYSTEM ========

class DesireSystem:
    """
    Manages the agent's desires (goals) and objectives.
    Adjusts goals based on market conditions and performance.
    """
    
    def __init__(self, 
                target_return_pct: float = 20.0,
                max_drawdown_pct: float = 10.0,
                risk_reward_ratio: float = 2.0,
                diversification_target: int = 5,
                preferred_timeframe: str = "1h"):
        
        self.target_return_pct = target_return_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.risk_reward_ratio = risk_reward_ratio
        self.diversification_target = diversification_target
        self.preferred_timeframe = preferred_timeframe
        
        # Performance tracking
        self.current_return_pct = 0.0
        self.current_drawdown_pct = 0.0
        self.win_rate = 0.0
        self.performance_history = []
        
    def get_current_desires(self) -> AgentDesire:
        """Get the current desires of the agent"""
        return AgentDesire(
            target_return_pct=self.target_return_pct,
            max_drawdown_pct=self.max_drawdown_pct,
            risk_reward_ratio=self.risk_reward_ratio,
            diversification_target=self.diversification_target,
            preferred_timeframe=self.preferred_timeframe
        )
    
    def adjust_desires(self, beliefs: AgentBelief, performance_metrics: Dict):
        """Adjust desires based on beliefs and performance"""
        # 1. Adjust based on market regime
        if beliefs.market_regime == "volatile":
            # Reduce risk in volatile markets
            self.max_drawdown_pct = max(5.0, self.max_drawdown_pct * 0.8)
            self.risk_reward_ratio = min(3.0, self.risk_reward_ratio * 1.2)
        elif beliefs.market_regime in ["uptrend", "downtrend"]:
            # Increase targets in trending markets
            self.target_return_pct = min(40.0, self.target_return_pct * 1.1)
        elif beliefs.market_regime == "ranging":
            # Adjust for range-bound markets
            self.target_return_pct = max(10.0, self.target_return_pct * 0.9)
            
        # 2. Adjust based on performance
        if performance_metrics.get('win_rate', 0) < 0.4:
            # Poor win rate, increase risk-reward ratio
            self.risk_reward_ratio = min(4.0, self.risk_reward_ratio * 1.2)
            
        if performance_metrics.get('current_drawdown_pct', 0) > self.max_drawdown_pct * 0.8:
            # Getting close to max drawdown, reduce risk
            self.target_return_pct = max(5.0, self.target_return_pct * 0.9)
            
        # 3. Adjust diversification based on correlations
        if beliefs.correlations is not None:
            # If high average correlation, increase diversification target
            avg_corr = np.nanmean(beliefs.correlations[np.triu_indices_from(beliefs.correlations, k=1)])
            if avg_corr > 0.7:  # High correlation
                self.diversification_target = min(10, self.diversification_target + 1)
            elif avg_corr < 0.3:  # Low correlation
                self.diversification_target = max(3, self.diversification_target - 1)
                
        # Log adjusted desires
        logging.info(f"Adjusted desires: target_return={self.target_return_pct}%, "
                     f"max_drawdown={self.max_drawdown_pct}%, "
                     f"risk_reward={self.risk_reward_ratio}, "
                     f"diversification={self.diversification_target}")


# ======== INTENTION SYSTEM ========

class IntentionSystem:
    """
    Manages the agent's intentions (planned actions) based on beliefs and desires.
    Prioritizes markets and strategies for execution.
    """
    
    def __init__(self, available_strategies: Dict[str, Callable]):
        self.available_strategies = available_strategies
        self.watchlist = []
        self.active_strategies = {}
        self.pending_orders = {}
        self.risk_management_rules = {
            "max_position_size_pct": 5.0,  # Max position size as % of portfolio
            "max_open_positions": 10,
            "max_risk_per_trade_pct": 1.0,  # Max risk per trade as % of portfolio
            "correlation_threshold": 0.7   # Max correlation for diversification
        }
        self.priority_markets = []
        
    def form_intentions(self, beliefs: AgentBelief, desires: AgentDesire) -> AgentIntention:
        """Form intentions based on current beliefs and desires"""
        # 1. Prioritize markets based on opportunities
        priority_markets = self._prioritize_markets(beliefs)
        
        # 2. Select strategies based on market regime and desires
        active_strategies = self._select_strategies(beliefs, desires)
        
        # 3. Define watchlist (markets to monitor closely)
        watchlist = priority_markets[:min(len(priority_markets), 10)]
        
        # 4. Adjust risk management rules based on market conditions
        risk_rules = self._adjust_risk_rules(beliefs, desires)
        
        # Create and return agent intention
        return AgentIntention(
            watchlist=watchlist,
            active_strategies=active_strategies,
            pending_orders=self.pending_orders,
            risk_management_rules=risk_rules,
            priority_markets=priority_markets
        )
    
    def _prioritize_markets(self, beliefs: AgentBelief) -> List[str]:
        """Prioritize markets based on opportunities"""
        market_scores = {}
        
        for symbol in beliefs.market_states:
            if beliefs.market_states[symbol] is None:
                continue
                
            # Skip if indicators not available
            if symbol not in beliefs.indicators or not beliefs.indicators[symbol]:
                continue
                
            score = 0
            
            # 1. Score based on volatility
            market_state = beliefs.market_states[symbol]
            if market_state.high_24h > 0:
                volatility = (market_state.high_24h - market_state.low_24h) / market_state.price
                score += volatility * 10  # Higher volatility, higher score
            
            # 2. Score based on indicators
            indicators = beliefs.indicators[symbol]
            
            # MACD crossover signals
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'][-2:]
                macd_signal = indicators['macd_signal'][-2:]
                
                if len(macd) >= 2 and len(macd_signal) >= 2:
                    # MACD crossing above signal line (bullish)
                    if macd[-2] < macd_signal[-2] and macd[-1] > macd_signal[-1]:
                        score += 5
                    # MACD crossing below signal line (bearish)
                    elif macd[-2] > macd_signal[-2] and macd[-1] < macd_signal[-1]:
                        score += 5
            
            # RSI oversold/overbought
            if 'rsi' in indicators:
                rsi = indicators['rsi'][-1]
                if rsi < 30:  # Oversold
                    score += 3
                elif rsi > 70:  # Overbought
                    score += 3
            
            # Bollinger Band squeeze (potential for breakout)
            if 'bb_upper' in indicators and 'bb_lower' in indicators:
                bb_upper = indicators['bb_upper'][-10:]
                bb_lower = indicators['bb_lower'][-10:]
                
                if len(bb_upper) >= 10 and len(bb_lower) >= 10:
                    band_width_now = bb_upper[-1] - bb_lower[-1]
                    band_width_before = bb_upper[-10] - bb_lower[-10]
                    
                    if band_width_now < band_width_before * 0.8:  # Squeeze
                        score += 4
            
            # 3. Score based on volume
            avg_volume = np.nanmean([
                state.volume_24h for state in beliefs.market_states.values() 
                if state is not None
            ])
            
            if market_state.volume_24h > avg_volume * 1.5:
                score += 3  # Higher than average volume
            
            market_scores[symbol] = score
        
        # Sort markets by score, descending
        sorted_markets = sorted(market_scores.keys(), key=lambda x: market_scores[x], reverse=True)
        return sorted_markets
    
    def _select_strategies(self, beliefs: AgentBelief, desires: AgentDesire) -> Dict[str, str]:
        """Select strategies based on market regime and desires"""
        active_strategies = {}
        
        # Default strategy based on market regime
        default_strategy = {
            "uptrend": "trend_following",
            "downtrend": "trend_following",
            "ranging": "mean_reversion",
            "volatile": "volatility_breakout",
            "overbought": "mean_reversion",
            "oversold": "mean_reversion",
            "unknown": "adaptive"
        }.get(beliefs.market_regime, "adaptive")
        
        # Assign strategies to priority markets
        for symbol in self.priority_markets[:desires.diversification_target]:
            # Check if symbol has a market-specific regime
            symbol_regime = "unknown"
            
            # Check if we need a specialized strategy for this market
            if symbol in beliefs.indicators and beliefs.indicators[symbol]:
                indicators = beliefs.indicators[symbol]
                
                # RSI-based decision
                if 'rsi' in indicators:
                    rsi = indicators['rsi'][-1]
                    if rsi < 30:
                        active_strategies[symbol] = "oversold_reversal"
                        continue
                    elif rsi > 70:
                        active_strategies[symbol] = "overbought_reversal"
                        continue
                
                # Trend strength based decision
                if 'sma_20' in indicators and 'sma_50' in indicators:
                    sma_20 = indicators['sma_20'][-1]
                    sma_50 = indicators['sma_50'][-1]
                    price = beliefs.market_states[symbol].price
                    
                    # Strong uptrend
                    if price > sma_20 > sma_50 and (sma_20 - sma_50) / sma_50 > 0.03:
                        active_strategies[symbol] = "trend_following"
                        continue
                    # Strong downtrend
                    elif price < sma_20 < sma_50 and (sma_50 - sma_20) / sma_50 > 0.03:
                        active_strategies[symbol] = "trend_following"
                        continue
                    # Ranging market
                    elif abs(sma_20 - sma_50) / sma_50 < 0.01:
                        active_strategies[symbol] = "mean_reversion"
                        continue
            
            # Default to market regime strategy if no specialized strategy chosen
            active_strategies[symbol] = default_strategy
            
        return active_strategies
    
    def _adjust_risk_rules(self, beliefs: AgentBelief, desires: AgentDesire) -> Dict:
        """Adjust risk management rules based on market conditions"""
        rules = self.risk_management_rules.copy()
        
        # Adjust position size based on market regime
        if beliefs.market_regime == "volatile":
            rules["max_position_size_pct"] = min(3.0, rules["max_position_size_pct"] * 0.7)
            rules["max_risk_per_trade_pct"] = min(0.5, rules["max_risk_per_trade_pct"] * 0.7)
        elif beliefs.market_regime in ["uptrend", "downtrend"]:
            # Slightly increase position size in trending markets
            rules["max_position_size_pct"] = min(7.0, rules["max_position_size_pct"] * 1.1)
        
        # Adjust max open positions based on diversification target
        rules["max_open_positions"] = max(desires.diversification_target, 
                                          min(15, desires.diversification_target * 1.5))
        
        # Adjust correlation threshold based on market condition
        if beliefs.market_regime == "volatile":
            rules["correlation_threshold"] = 0.5  # Lower correlation threshold in volatile markets
        
        return rules
    
    def validate_intention(self, intention: AgentIntention, beliefs: AgentBelief) -> AgentIntention:
        """Validate and correct intentions if necessary"""
        validated = intention
        
        # 1. Check if we have too many active strategies
        max_strategies = beliefs.risk_level == "aggressive" and 10 or 5
        if len(validated.active_strategies) > max_strategies:
            # Keep only the highest priority ones
            top_symbols = validated.priority_markets[:max_strategies]
            validated.active_strategies = {
                symbol: strategy for symbol, strategy in validated.active_strategies.items()
                if symbol in top_symbols
            }
        
        # 2. Check if pending orders comply with risk rules
        validated.pending_orders = self._validate_pending_orders(
            validated.pending_orders, 
            validated.risk_management_rules,
            beliefs
        )
        
        return validated
    
    def _validate_pending_orders(self, 
                                pending_orders: Dict, 
                                risk_rules: Dict,
                                beliefs: AgentBelief) -> Dict:
        """Validate pending orders against risk rules"""
        # Filter orders that don't comply with risk rules
        valid_orders = {}
        
        # Calculate total portfolio value
        portfolio_value = beliefs.cash_balance
        for symbol, amount in beliefs.portfolio.items():
            if symbol in beliefs.market_states and beliefs.market_states[symbol]:
                portfolio_value += amount * beliefs.market_states[symbol].price
        
        # Check each order
        for order_id, order in pending_orders.items():
            # Skip if already validated
            if order.get('validated', False):
                valid_orders[order_id] = order
                continue
                
            symbol = order['symbol']
            action = order['action']
            quantity = order['quantity']
            price = order.get('price') or beliefs.market_states[symbol].price
            
            # Check position size
            position_value = quantity * price
            position_size_pct = (position_value / portfolio_value) * 100
            
            if position_size_pct > risk_rules['max_position_size_pct']:
                # Adjust quantity to match max position size
                adjusted_quantity = (risk_rules['max_position_size_pct'] / 100) * portfolio_value / price
                order['quantity'] = adjusted_quantity
                order['adjusted'] = True
                
            # Check for excessive risk per trade
            if 'stop_loss' in order:
                risk_amount = abs(price - order['stop_loss']) * quantity
                risk_pct = (risk_amount / portfolio_value) * 100
                
                if risk_pct > risk_rules['max_risk_per_trade_pct']:
                    # Adjust quantity to match max risk
                    price_diff = abs(price - order['stop_loss'])
                    if price_diff > 0:
                        max_risk_amount = (risk_rules['max_risk_per_trade_pct'] / 100) * portfolio_value
                        adjusted_quantity = max_risk_amount / price_diff
                        order['quantity'] = min(order['quantity'], adjusted_quantity)
                        order['adjusted'] = True
            
            # Mark as validated
            order['validated'] = True
            valid_orders[order_id] = order
                
        return valid_orders


# ======== ACTION SYSTEM ========

class ActionSystem:
    """
    Executes trading actions based on intentions.
    Manages order execution and tracking.
    """
    
    def __init__(self, exchange_connector):
        self.exchange = exchange_connector
        self.executed_trades = []
        self.active_orders = {}
        self.execution_stats = {
            "successful_orders": 0,
            "failed_orders": 0,
            "slippage_total": 0,
            "avg_execution_time": 0
        }
        
    def execute_actions(self, 
                       intentions: AgentIntention, 
                       beliefs: AgentBelief, 
                       strategies: Dict[str, Callable]) -> List[TradeAction]:
        """Execute trading actions based on intentions"""
        executed_actions = []
        
        # 1. Process pending orders from intentions
        for order_id, order in intentions.pending_orders.items():
            if order.get('executed', False):
                continue
                
            try:
                result = self._execute_order(order)
                order['executed'] = True
                order['execution_result'] = result
                executed_actions.append(TradeAction(
                    symbol=order['symbol'],
                    action=order['action'],
                    quantity=order['quantity'],
                    price=result.get('executed_price'),
                    order_type=order['order_type'],
                    time_in_force=order.get('time_in_force', 'GTC'),
                    timestamp=time.time(),
                    strategy_id=order.get('strategy_id', '')
                ))
                self.execution_stats["successful_orders"] += 1
            except Exception as e:
                logging.error(f"Order execution failed: {e}")
                order['executed'] = False
                order['error'] = str(e)
                self.execution_stats["failed_orders"] += 1
        
        # 2. Generate new orders from active strategies
        for symbol, strategy_name in intentions.active_strategies.items():
            if strategy_name not in strategies:
                logging.warning(f"Strategy {strategy_name} not found")
                continue
                
            # Skip if market data not available
            if (symbol not in beliefs.market_states or 
                beliefs.market_states[symbol] is None or
                symbol not in beliefs.indicators):
                continue
                
            # Execute strategy
            strategy_func = strategies[strategy_name]
            try:
                actions = strategy_func(
                    symbol=symbol,
                    market_state=beliefs.market_states[symbol],
                    indicators=beliefs.indicators[symbol],
                    portfolio=beliefs.portfolio,
                    cash_balance=beliefs.cash_balance,
                    risk_rules=intentions.risk_management_rules
                )
                
                # Process strategy actions
                if actions:
                    if not isinstance(actions, list):
                        actions = [actions]
                        
                    for action in actions:
                        # Execute action
                        if isinstance(action, TradeAction):
                            result = self._execute_trade_action(action)
                            if result:
                                executed_actions.append(action)
                                self.execution_stats["successful_orders"] += 1
                                
            except Exception as e:
                logging.error(f"Strategy execution failed for {strategy_name} on {symbol}: {e}")
                self.execution_stats["failed_orders"] += 1
        
        return executed_actions
    
    def _execute_order(self, order: Dict) -> Dict:
        """Execute a single order through the exchange connector"""
        start_time = time.time()
        
        # Execute order through exchange
        result = self.exchange.place_order(
            symbol=order['symbol'],
            side=order['action'],
            order_type=order['order_type'],
            quantity=order['quantity'],
            price=order.get('price'),
            time_in_force=order.get('time_in_force', 'GTC'),
            stop_price=order.get('stop_price'),
            client_order_id=order.get('client_order_id')
        )
        
        # Calculate execution metrics
        execution_time = time.time() - start_time
        
        # Update average execution time
        count = self.execution_stats["successful_orders"]
        self.execution_stats["avg_execution_time"] = (
            (self.execution_stats["avg_execution_time"] * count + execution_time) / (count + 1)
        )
        
        # Calculate slippage if applicable
        if 'expected_price' in order and 'executed_price' in result:
            slippage = abs(result['executed_price'] - order['expected_price']) / order['expected_price']
            self.execution_stats["slippage_total"] += slippage
            result['slippage'] = slippage
            
        # Track order
        self.active_orders[result['order_id']] = {
            'order': order,
            'result': result,
            'status': 'OPEN',
            'executed_time': time.time()
        }
        
        # Add to executed trades history
        self.executed_trades.append({
            'symbol': order['symbol'],
            'action': order['action'],
            'quantity': order['quantity'],
            'price': result.get('executed_price'),
            'order_type': order['order_type'],
            'timestamp': time.time(),
            'strategy_id': order.get('strategy_id', '')
        })
        
        return result
    
    def _execute_trade_action(self, action: TradeAction) -> Dict:
        """Execute a trade action"""
        order = {
            'symbol': action.symbol,
            'action': action.action,
            'quantity': action.quantity,
            'order_type': action.order_type,
            'time_in_force': action.time_in_force,
            'strategy_id': action.strategy_id
        }
        
        if action.price:
            order['price'] = action.price
            
        try:
            result = self._execute_order(order)
            return result
        except Exception as e:
            logging.error(f"Failed to execute trade action: {e}")
            return None
    
    def update_order_status(self):
        """Update status of all active orders"""
        for order_id, order_info in list(self.active_orders.items()):
            try:
                status = self.exchange.get_order_status(order_id)
                order_info['status'] = status['status']
                
                # Remove from active orders if completed
                if status['status'] in ['FILLED', 'CANCELED', 'REJECTED', 'EXPIRED']:
                    self.active_orders.pop(order_id)
                    
            except Exception as e:
                logging.error(f"Failed to update order status: {e}")


# ======== TRADING STRATEGIES ========

class TradingStrategies:
    """
    Collection of trading strategies that can be used by the agent.
    Each strategy analyzes market data and returns trade actions.
    """
    
    @staticmethod
    def trend_following(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """Trend following strategy using moving averages"""
        actions = []
        
        # Check if we have enough indicators
        if not all(k in indicators for k in ['sma_20', 'sma_50']):
            return actions
            
        sma_20 = indicators['sma_20'][-2:]
        sma_50 = indicators['sma_50'][-2:]
        
        if len(sma_20) < 2 or len(sma_50) < 2:
            return actions
        
        current_price = market_state.price
        
        # Calculate position size based on risk rules
        max_position_value = (risk_rules['max_position_size_pct'] / 100) * sum([
            amount * market_state.price if symbol in portfolio else 0 
            for symbol, amount in portfolio.items()
        ]) + cash_balance
        
        position_size = max_position_value / current_price
        
        # Current position in portfolio
        current_position = portfolio.get(symbol, 0)
        
        # Check for golden cross (sma_20 crosses above sma_50)
        if sma_20[-2] <= sma_50[-2] and sma_20[-1] > sma_50[-1]:
            # Golden cross - BUY signal
            if current_position == 0:  # Only buy if we don't have a position
                # Calculate buy quantity
                buy_quantity = position_size
                
                # Create buy action
                actions.append(TradeAction(
                    symbol=symbol,
                    action="BUY",
                    quantity=buy_quantity,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="trend_following"
                ))
                
        # Check for death cross (sma_20 crosses below sma_50)
        elif sma_20[-2] >= sma_50[-2] and sma_20[-1] < sma_50[-1]:
            # Death cross - SELL signal
            if current_position > 0:  # Only sell if we have a position
                # Sell entire position
                actions.append(TradeAction(
                    symbol=symbol,
                    action="SELL",
                    quantity=current_position,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="trend_following"
                ))
        
        return actions
    
    @staticmethod
    def mean_reversion(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """Mean reversion strategy using Bollinger Bands"""
        actions = []
        
        # Check if we have enough indicators
        if not all(k in indicators for k in ['bb_upper', 'bb_lower', 'bb_middle']):
            return actions
            
        bb_upper = indicators['bb_upper'][-1]
        bb_lower = indicators['bb_lower'][-1]
        bb_middle = indicators['bb_middle'][-1]
        
        current_price = market_state.price
        
        # Calculate position size based on risk rules
        max_position_value = (risk_rules['max_position_size_pct'] / 100) * sum([
            amount * market_state.price if symbol in portfolio else 0 
            for symbol, amount in portfolio.items()
        ]) + cash_balance
        
        position_size = max_position_value / current_price
        
        # Current position in portfolio
        current_position = portfolio.get(symbol, 0)
        
        # Price below lower band - Buy signal
        if current_price < bb_lower:
            if current_position == 0:  # Only buy if we don't have a position
                # Calculate buy quantity
                buy_quantity = position_size
                
                # Create buy action
                actions.append(TradeAction(
                    symbol=symbol,
                    action="BUY",
                    quantity=buy_quantity,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="mean_reversion"
                ))
                
        # Price above upper band - Sell signal
        elif current_price > bb_upper:
            if current_position > 0:  # Only sell if we have a position
                # Sell entire position
                actions.append(TradeAction(
                    symbol=symbol,
                    action="SELL",
                    quantity=current_position,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="mean_reversion"
                ))
                
        # Price crosses middle band from below - Take profit on half position
        elif (current_price > bb_middle and 
              market_state.price > indicators['bb_middle'][-2] and
              current_position > 0):
            
            # Sell half position when crossing the middle band
            sell_quantity = current_position / 2
            
            actions.append(TradeAction(
                symbol=symbol,
                action="SELL",
                quantity=sell_quantity,
                price=None,  # Market order
                order_type="MARKET",
                strategy_id="mean_reversion"
            ))
        
        return actions
    
    @staticmethod
    def volatility_breakout(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """Volatility breakout strategy"""
        actions = []
        
        # Calculate Average True Range (ATR) for volatility measurement
        if not all(k in indicators for k in ['atr_14']):
            # ATR not available, try to compute it
            if 'high' in market_state.historical_data and 'low' in market_state.historical_data:
                high = market_state.historical_data['high'][-14:]
                low = market_state.historical_data['low'][-14:]
                close = market_state.historical_data['close'][-15:-1]  # Previous closes
                
                if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
                    # Calculate TR (True Range)
                    tr = []
                    for i in range(len(high)):
                        hl = high[i] - low[i]
                        hc = abs(high[i] - close[i])
                        lc = abs(low[i] - close[i])
                        tr.append(max(hl, hc, lc))
                    
                    atr = sum(tr) / len(tr)
                else:
                    return actions  # Not enough data
            else:
                return actions  # Not enough data
        else:
            atr = indicators['atr_14'][-1]
        
        current_price = market_state.price
        
        # Calculate position size based on risk rules
        max_position_value = (risk_rules['max_position_size_pct'] / 100) * sum([
            amount * market_state.price if symbol in portfolio else 0 
            for symbol, amount in portfolio.items()
        ]) + cash_balance
        
        position_size = max_position_value / current_price
        
        # Current position in portfolio
        current_position = portfolio.get(symbol, 0)
        
        # Get daily high and lows
        if 'high' in market_state.historical_data and 'low' in market_state.historical_data:
            prev_high = market_state.historical_data['high'][-2]
            prev_low = market_state.historical_data['low'][-2]
            
            # Breakout above previous high
            if current_price > prev_high + (atr * 0.5):  # Using half ATR as confirmation
                if current_position == 0:  # Only buy if we don't have a position
                    # Calculate buy quantity
                    buy_quantity = position_size
                    
                    # Calculate stop loss based on ATR
                    stop_loss = current_price - (atr * 2)
                    
                    # Create buy action
                    actions.append(TradeAction(
                        symbol=symbol,
                        action="BUY",
                        quantity=buy_quantity,
                        price=None,  # Market order
                        order_type="MARKET",
                        strategy_id="volatility_breakout"
                    ))
                    
            # Breakdown below previous low
            elif current_price < prev_low - (atr * 0.5):  # Using half ATR as confirmation
                if current_position > 0:  # Only sell if we have a position
                    # Sell entire position
                    actions.append(TradeAction(
                        symbol=symbol,
                        action="SELL",
                        quantity=current_position,
                        price=None,  # Market order
                        order_type="MARKET",
                        strategy_id="volatility_breakout"
                    ))
        
        return actions
    
    @staticmethod
    def oversold_reversal(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """RSI oversold reversal strategy"""
        actions = []
        
        # Check if we have RSI indicator
        if 'rsi' not in indicators:
            return actions
            
        rsi = indicators['rsi'][-2:]  # Get last two RSI values
        
        if len(rsi) < 2:
            return actions
            
        current_price = market_state.price
        
        # Calculate position size based on risk rules
        max_position_value = (risk_rules['max_position_size_pct'] / 100) * sum([
            amount * market_state.price if symbol in portfolio else 0 
            for symbol, amount in portfolio.items()
        ]) + cash_balance
        
        position_size = max_position_value / current_price
        
        # Current position in portfolio
        current_position = portfolio.get(symbol, 0)
        
        # Check for RSI reversal from oversold
        if rsi[-2] < 30 and rsi[-1] > rsi[-2]:  # RSI was below 30 and is now rising
            if current_position == 0:  # Only buy if we don't have a position
                # Calculate buy quantity
                buy_quantity = position_size
                
                # Create buy action
                actions.append(TradeAction(
                    symbol=symbol,
                    action="BUY",
                    quantity=buy_quantity,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="oversold_reversal"
                ))
                
        # Take profit when RSI goes above 50
        elif rsi[-1] > 50 and current_position > 0:
            # Sell entire position
            actions.append(TradeAction(
                symbol=symbol,
                action="SELL",
                quantity=current_position,
                price=None,  # Market order
                order_type="MARKET",
                strategy_id="oversold_reversal"
            ))
        
        return actions
    
    @staticmethod
    def overbought_reversal(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """RSI overbought reversal strategy - for shorting or selling"""
        actions = []
        
        # Check if we have RSI indicator
        if 'rsi' not in indicators:
            return actions
            
        rsi = indicators['rsi'][-2:]  # Get last two RSI values
        
        if len(rsi) < 2:
            return actions
            
        # Current position in portfolio
        current_position = portfolio.get(symbol, 0)
        
        # Check for RSI reversal from overbought - sell if we have a position
        if rsi[-2] > 70 and rsi[-1] < rsi[-2]:  # RSI was above 70 and is now falling
            if current_position > 0:
                # Sell entire position
                actions.append(TradeAction(
                    symbol=symbol,
                    action="SELL",
                    quantity=current_position,
                    price=None,  # Market order
                    order_type="MARKET",
                    strategy_id="overbought_reversal"
                ))
        
        return actions
    
    @staticmethod
    def adaptive(symbol, market_state, indicators, portfolio, cash_balance, risk_rules):
        """Adaptive strategy that combines multiple strategies based on market conditions"""
        actions = []
        
        # Determine market condition
        if 'rsi' in indicators and 'sma_20' in indicators and 'sma_50' in indicators:
            rsi = indicators['rsi'][-1]
            sma_20 = indicators['sma_20'][-1]
            sma_50 = indicators['sma_50'][-1]
            price = market_state.price
            
            # Oversold condition
            if rsi < 30:
                # Use oversold reversal strategy
                return TradingStrategies.oversold_reversal(
                    symbol, market_state, indicators, portfolio, cash_balance, risk_rules
                )
                
            # Overbought condition
            elif rsi > 70:
                # Use overbought reversal strategy
                return TradingStrategies.overbought_reversal(
                    symbol, market_state, indicators, portfolio, cash_balance, risk_rules
                )
                
            # Trending market
            elif abs((sma_20 - sma_50) / sma_50) > 0.02:
                # Use trend following
                return TradingStrategies.trend_following(
                    symbol, market_state, indicators, portfolio, cash_balance, risk_rules
                )
                
            # Ranging market
            else:
                # Use mean reversion
                return TradingStrategies.mean_reversion(
                    symbol, market_state, indicators, portfolio, cash_balance, risk_rules
                )
                
        return actions


# ======== CRYPTO TRADING AGENT ========

class CryptoTradingAgent:
    """
    Main agent class that orchestrates the BDI components and manages the trading system.
    """
    
    def __init__(self, 
                symbols: List[str],
                exchange_connector, # This should be an instance of your exchange connector class
                use_gpu: bool = True,
                config: Optional[Dict] = None):
        
        self.symbols = symbols
        self.exchange = exchange_connector
        self.use_gpu = use_gpu and HAS_GPU
        self.config = config if config is not None else {}
        
        # Initialize BDI components
        self.belief_system = BeliefSystem(symbols, use_gpu=self.use_gpu)
        
        self.desire_system = DesireSystem(
            target_return_pct=self.config.get('target_return_pct', 20.0),
            max_drawdown_pct=self.config.get('max_drawdown_pct', 10.0), # This is a desire/limit
            risk_reward_ratio=self.config.get('risk_reward_ratio', 2.0),
            diversification_target=self.config.get('diversification_target', 5),
            preferred_timeframe=self.config.get('preferred_timeframe', '1h')
        )
        
        # Initialize trading strategies
        self.strategies = {
            "trend_following": TradingStrategies.trend_following,
            "mean_reversion": TradingStrategies.mean_reversion,
            "volatility_breakout": TradingStrategies.volatility_breakout,
            "oversold_reversal": TradingStrategies.oversold_reversal,
            "overbought_reversal": TradingStrategies.overbought_reversal,
            "adaptive": TradingStrategies.adaptive
        }
        
        self.intention_system = IntentionSystem(self.strategies)
        self.action_system = ActionSystem(exchange_connector)
        
        # Performance tracking
        self.performance_metrics = {
            "start_balance": 0.0,
            "current_balance": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "win_rate": 0.0, # Placeholder, requires detailed trade logging
            "peak_balance_for_drawdown": 0.0, # Tracks peak balance for drawdown calculation
            "max_drawdown_pct": 0.0,      # Tracks the maximum drawdown experienced
            "current_drawdown_pct": 0.0,  # Current drawdown from peak
            "sharpe_ratio": 0.0, # Placeholder
            "trades_history": [], # List of TradeAction objects for executed trades
            "balance_history": [] # History of total balance for drawdown and charts
        }
        
        self.running = False
        self.last_update_time = 0.0 # Renamed from last_update to avoid conflict if a property 'last_update' exists
        self.update_interval = self.config.get('update_interval', 60)  # seconds
        self.base_currency = self.config.get('base_currency', 'USDT') # e.g., USDT, USD

        self.agent_thread = None # Initialize agent_thread
        
    def start(self):
        """Start the agent"""
        if self.running:
            logging.info("Agent is already running.")
            return

        self.running = True
        self.last_update_time = 0 # Reset last update time
        
        # Initialize portfolio and starting balance
        self._update_portfolio_data() # Fetch initial portfolio
        start_bal = self._calculate_total_balance()
        self.performance_metrics["start_balance"] = start_bal
        self.performance_metrics["current_balance"] = start_bal
        self.performance_metrics["peak_balance_for_drawdown"] = start_bal
        self.performance_metrics["balance_history"] = [start_bal]
        
        logging.info(f"Agent started with initial balance: {start_bal:.2f} {self.base_currency}")
        
        # Start main loop in a separate thread
        self.agent_thread = threading.Thread(target=self._run_loop, name="CryptoAgentThread")
        self.agent_thread.daemon = True # Allow main program to exit even if thread is running
        self.agent_thread.start()
        
    def stop(self):
        """Stop the agent"""
        self.running = False
        if self.agent_thread and self.agent_thread.is_alive():
            logging.info("Stopping agent thread...")
            self.agent_thread.join(timeout=10) # Wait for the thread to finish
            if self.agent_thread.is_alive():
                logging.warning("Agent thread did not stop in time.")
        logging.info("Agent stopped")
        
    def get_status(self):
        """Get current agent status"""
        # Ensure belief system attributes are accessed safely if not yet populated
        belief_market_regime = "unknown"
        belief_symbols_count = 0
        if hasattr(self.belief_system, 'overall_market_regime'):
            belief_market_regime = self.belief_system.overall_market_regime
        if self.belief_system.market_states:
            belief_symbols_count = len([s for s in self.belief_system.market_states.values() if s is not None])

        return {
            "running": self.running,
            "last_update_timestamp": self.last_update_time,
            "last_update_readable": datetime.fromtimestamp(self.last_update_time).isoformat() if self.last_update_time > 0 else "N/A",
            "performance": self.performance_metrics,
            "actions_stats": self.action_system.execution_stats,
            "active_orders_count": len(self.action_system.active_orders),
            "beliefs": {
                "market_regime": belief_market_regime,
                "symbols_count": belief_symbols_count,
                "cash_balance": self.belief_system.cash_balance,
                "portfolio": self.belief_system.portfolio,
            },
            "desires": self.desire_system.get_current_desires().__dict__ if self.desire_system else {}
        }
        
    def _run_loop(self):
        """Main agent loop"""
        logging.info("Agent run loop started.")
        while self.running:
            current_time = time.time()
            
            if current_time - self.last_update_time >= self.update_interval:
                logging.info(f"Starting update cycle at {datetime.fromtimestamp(current_time).isoformat()}")
                try:
                    # Step 1: Update market data (prices, klines)
                    self._update_market_data_for_all_symbols()
                    
                    # Step 2: Update portfolio data (balances)
                    self._update_portfolio_data()
                    
                    # Step 3: Perform BDI cycle (reasoning and acting)
                    self._perform_bdi_cycle()
                    
                    # Step 4: Update performance metrics
                    self._update_performance_metrics()
                    
                    self.last_update_time = current_time
                    logging.info(f"Update cycle completed. Next update in {self.update_interval}s.")
                    
                except Exception as e:
                    logging.exception(f"Error in agent's main loop: {e}") # Use logging.exception to include stack trace
                    # Potentially implement a cooldown or retry mechanism here
                    time.sleep(self.update_interval / 2) # Wait a bit before retrying after an error
            
            try:
                # Sleep to reduce CPU usage, but wake up often enough to check self.running
                # Check self.running frequently to allow faster shutdown
                for _ in range(int(min(self.update_interval, 5))): # Check every second, up to 5s or update_interval
                    if not self.running:
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt received in run loop, stopping agent.")
                self.running = False # Ensure flag is set to break loop
                break # Exit loop immediately

        logging.info("Agent run loop finished.")
            
    def _update_market_data_for_all_symbols(self):
        """Update market data (ticker, klines) for all relevant symbols."""
        logging.info("Updating market data for all symbols...")
        # Using ThreadPoolExecutor for fetching data in parallel
        # Adjust max_workers based on API rate limits and system resources
        max_workers = min(len(self.symbols), 5) # Example: up to 5 concurrent requests
        if max_workers <= 0: return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._fetch_and_process_market_data_for_symbol, symbol): symbol
                for symbol in self.symbols
            }
            for future in future_to_symbol:
                symbol = future_to_symbol[future]
                try:
                    future.result() # Wait for completion and raise exceptions if any
                except Exception as e:
                    logging.error(f"Failed to update market data for {symbol}: {e}")
        logging.info("Market data update for all symbols complete.")

    def _fetch_and_process_market_data_for_symbol(self, symbol: str):
        """Fetch and process market data for a single symbol."""
        try:
            # Get current ticker data
            ticker = self.exchange.get_ticker(symbol)
            if not ticker:
                logging.warning(f"No ticker data received for {symbol}")
                return

            # Get recent klines/candlesticks
            # Ensure timeframe is valid for the exchange
            klines = self.exchange.get_klines(
                symbol, 
                interval=self.desire_system.preferred_timeframe, 
                limit=200 + 50 # Fetch a bit more for indicator stability (e.g. 200 period SMA)
            )
            if not klines:
                logging.warning(f"No klines data received for {symbol}")
                # Keep existing historical data if new data fails, or clear it
                # self.belief_system.historical_data.pop(symbol, None) # Optional: clear if fetch fails
                return

            # Update belief system with new ticker data
            self.belief_system.update_market_data({
                symbol: {
                    'price': float(ticker['last_price']),
                    'timestamp': float(ticker['timestamp']) / 1000.0, # Assuming ms from exchange -> s
                    'volume_24h': float(ticker['volume_24h']),
                    'high_24h': float(ticker['high_24h']),
                    'low_24h': float(ticker['low_24h']),
                    'bid': float(ticker['bid']) if ticker.get('bid') is not None else None,
                    'ask': float(ticker['ask']) if ticker.get('ask') is not None else None,
                    'order_book': ticker.get('order_book') # Optional
                }
            })
            
            # Process klines into DataFrame
            # Standard columns: 0: timestamp, 1: open, 2: high, 3: low, 4: close, 5: volume
            # Verify based on your exchange_connector's get_klines output
            df_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] # Common columns
            # If your exchange returns more (e.g., quote_asset_volume, num_trades, etc.), add them or select these.
            
            processed_klines = []
            for kline_data in klines:
                # Ensure kline_data has enough elements for standard columns
                if len(kline_data) >= len(df_columns):
                    processed_klines.append(kline_data[:len(df_columns)]) 
                else:
                    logging.warning(f"Skipping malformed kline for {symbol}: {kline_data}")
            
            if not processed_klines:
                logging.warning(f"No valid klines processed for {symbol} after filtering.")
                return

            df = pd.DataFrame(processed_klines, columns=df_columns)
            
            # Convert columns to numeric, handling potential errors
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp (assuming exchange provides ms timestamps for klines)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Drop rows with NaNs that might have been introduced by 'coerce'
            df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

            if df.empty:
                logging.warning(f"Historical DataFrame for {symbol} is empty after processing.")
                return

            self.belief_system.update_historical_data(symbol, df)
            logging.debug(f"Updated historical data for {symbol} with {len(df)} rows.")

        except Exception as e:
            logging.error(f"Error processing market data for {symbol}: {e}")
            # Re-raise if you want the ThreadPoolExecutor to catch it in the main update method
            raise


    def _update_portfolio_data(self):
        """Update portfolio and cash balance from the exchange"""
        logging.info("Updating portfolio data...")
        try:
            # Assuming exchange.get_balances() returns a dict like:
            # {'BTC': {'free': '1.0', 'locked': '0.1'}, 'USDT': {'free': '10000.0', 'locked': '0.0'}, ...}
            balances_data = self.exchange.get_balances()
            if not balances_data:
                logging.warning("Received no balance data from exchange.")
                return

            new_portfolio = {}
            new_cash_balance = 0.0
            
            for asset, details in balances_data.items():
                try:
                    free_amount = float(details.get('free', 0.0))
                except ValueError:
                    logging.warning(f"Could not parse 'free' amount for asset {asset}: {details.get('free')}. Skipping.")
                    continue

                if asset.upper() == self.base_currency.upper():
                    new_cash_balance = free_amount
                else:
                    # Define a minimum threshold for an asset to be in portfolio (e.g., > $1 value)
                    # For simplicity, any non-zero free amount for non-base currency is part of portfolio.
                    if free_amount > 1e-8: # Threshold to avoid dust
                        new_portfolio[asset.upper()] = free_amount 
            
            self.belief_system.portfolio = new_portfolio
            self.belief_system.cash_balance = new_cash_balance
            logging.info(
                f"Portfolio updated: Cash ({self.base_currency}): {new_cash_balance:.2f}, "
                f"Assets: { {k: round(v,5) for k,v in new_portfolio.items()} }"
            )
            
        except Exception as e:
            logging.exception(f"Failed to update portfolio data: {e}")

    def _calculate_total_balance(self) -> float:
        """Calculate total portfolio value in the base currency."""
        total_balance = self.belief_system.cash_balance
        
        for asset_symbol, amount in self.belief_system.portfolio.items():
            if amount == 0:
                continue

            # Construct market pair symbol, e.g., BTCUSDT if asset is BTC and base is USDT
            market_pair_symbol = f"{asset_symbol.upper()}{self.base_currency.upper()}"
            
            if market_pair_symbol in self.belief_system.market_states and \
               self.belief_system.market_states[market_pair_symbol] is not None:
                
                price = self.belief_system.market_states[market_pair_symbol].price # type: ignore
                if price > 0:
                    total_balance += amount * price
                else:
                    logging.warning(f"Price for {market_pair_symbol} is zero or negative, cannot value asset {asset_symbol}.")
            else:
                # Fallback for assets that don't have a direct pair with base_currency (e.g. ALT/BTC, and base is USDT)
                # This requires more complex logic (e.g., ALT/BTC price * BTC/USDT price)
                # For now, we only value assets with a direct pair to base_currency.
                # Or, if the asset IS the base currency (already handled by cash_balance if keys are distinct).
                if asset_symbol.upper() != self.base_currency.upper(): # Check to avoid double counting if base is in portfolio keys
                    logging.warning(
                        f"No market state or price found for {market_pair_symbol} to value asset {asset_symbol}. "
                        f"It will not be included in total balance calculation unless it's the base currency."
                    )
        return total_balance

    def _perform_bdi_cycle(self):
        """Perform one BDI cycle: Beliefs -> Desires -> Intentions -> Actions"""
        logging.info("Performing BDI cycle...")
        
        # 1. Update Beliefs internal states (indicators, regime, correlations)
        # Ensure historical data is loaded before calculating indicators
        if not self.belief_system.historical_data:
            logging.warning("No historical data available in BeliefSystem. Skipping indicator calculation.")
        else:
            self.belief_system.calculate_indicators(parallel=True) # parallel default is True

        current_beliefs = self.belief_system.get_current_beliefs()
        logging.debug(f"Current Beliefs: Market Regime: {current_beliefs.market_regime}, Cash: {current_beliefs.cash_balance}, Portfolio: {current_beliefs.portfolio}")

        # 2. Update Desires
        self.desire_system.adjust_desires(current_beliefs, self.performance_metrics)
        current_desires = self.desire_system.get_current_desires()
        logging.debug(f"Current Desires: Target Return: {current_desires.target_return_pct}%, Max Drawdown Limit: {current_desires.max_drawdown_pct}%")

        # 3. Form Intentions
        current_intentions = self.intention_system.form_intentions(current_beliefs, current_desires)
        # Validate intentions (optional, as strategies should already consider risk rules)
        # current_intentions = self.intention_system.validate_intention(current_intentions, current_beliefs)
        logging.debug(f"Formed Intentions: Watchlist: {current_intentions.watchlist}, Active Strategies: {current_intentions.active_strategies}")
        
        # 4. Execute Actions
        # Strategies are called from within action_system.execute_actions
        executed_trade_actions = self.action_system.execute_actions(
            intentions=current_intentions,
            beliefs=current_beliefs,
            strategies=self.strategies 
        )
        
        if executed_trade_actions:
            logging.info(f"BDI cycle resulted in {len(executed_trade_actions)} executed trade actions:")
            for trade_action in executed_trade_actions:
                log_msg = (f"  - {trade_action.action} {trade_action.quantity:.6f} {trade_action.symbol} "
                           f"@ {trade_action.price if trade_action.price is not None else 'MARKET'} "
                           f"(Strategy: {trade_action.strategy_id})")
                logging.info(log_msg)
                self.performance_metrics["trades_history"].append(trade_action)
        else:
            logging.info("BDI cycle completed. No new trade actions executed.")
        
        # Update status of any open/active orders
        self.action_system.update_order_status()
        
        logging.info("BDI cycle finished.")

    def _update_performance_metrics(self):
        """Update performance metrics based on current state."""
        logging.info("Updating performance metrics...")
        
        current_total_balance = self._calculate_total_balance()
        self.performance_metrics["current_balance"] = current_total_balance
        
        start_balance = self.performance_metrics["start_balance"]
        if start_balance > 0:
            pnl = current_total_balance - start_balance
            pnl_pct = (pnl / start_balance) * 100
            self.performance_metrics["total_pnl"] = pnl
            self.performance_metrics["total_pnl_pct"] = pnl_pct
        else: # Avoid division by zero if start_balance is 0
            self.performance_metrics["total_pnl"] = 0.0
            self.performance_metrics["total_pnl_pct"] = 0.0

        # Balance history for drawdown
        self.performance_metrics["balance_history"].append(current_total_balance)
        
        # Update peak balance for drawdown calculation
        current_peak = self.performance_metrics.get("peak_balance_for_drawdown", start_balance)
        if current_total_balance > current_peak:
            self.performance_metrics["peak_balance_for_drawdown"] = current_total_balance
            current_peak = current_total_balance # Update for current calculation
        
        # Current Drawdown
        if current_peak > 0: # Avoid division by zero
            drawdown = (current_peak - current_total_balance) / current_peak
            self.performance_metrics["current_drawdown_pct"] = drawdown * 100
            
            # Max Drawdown Experienced
            if self.performance_metrics["current_drawdown_pct"] > self.performance_metrics["max_drawdown_pct"]:
                self.performance_metrics["max_drawdown_pct"] = self.performance_metrics["current_drawdown_pct"]
        else:
            self.performance_metrics["current_drawdown_pct"] = 0.0

        # Win Rate and Sharpe Ratio are more complex and typically require a dedicated trade logging/analysis component
        # For now, they remain placeholders.
        # self.performance_metrics["win_rate"] = ... 
        # self.performance_metrics["sharpe_ratio"] = ...

        logging.info(
            f"Performance Update: Current Balance: {current_total_balance:.2f} {self.base_currency}, "
            f"Total P&L: {self.performance_metrics['total_pnl']:.2f} ({self.performance_metrics['total_pnl_pct']:.2f}%), "
            f"Current Drawdown: {self.performance_metrics['current_drawdown_pct']:.2f}%, "
            f"Max Drawdown: {self.performance_metrics['max_drawdown_pct']:.2f}%"
        )