"""
CUDA Kernel Optimization for Financial Trading
Custom CUDA kernels optimized for financial computations and trading algorithms.
"""

import cupy as cp
import numpy as np
from numba import cuda, float32, int32, boolean
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from typing import Tuple, Optional, Any, Dict, List
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)


class CUDAKernelManager:
    """Manages optimized CUDA kernels for financial computations."""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA kernel manager."""
        self.device_id = device_id
        self.context = cuda.select_device(device_id)
        self.stream = cuda.stream()
        self.kernel_cache = {}
        
    def get_optimal_grid_size(self, n_elements: int, threads_per_block: int = 256) -> Tuple[int, int]:
        """Calculate optimal grid size for given number of elements."""
        blocks_per_grid = (n_elements + threads_per_block - 1) // threads_per_block
        return blocks_per_grid, threads_per_block


# Technical Indicator Kernels
@cuda.jit
def moving_average_kernel(prices, window_size, result):
    """CUDA kernel for computing moving averages."""
    idx = cuda.grid(1)
    if idx >= len(result):
        return
    
    if idx < window_size - 1:
        result[idx] = float32(0.0)
        return
    
    sum_val = float32(0.0)
    for i in range(window_size):
        sum_val += prices[idx - i]
    
    result[idx] = sum_val / float32(window_size)


@cuda.jit
def exponential_moving_average_kernel(prices, alpha, result):
    """CUDA kernel for computing exponential moving averages."""
    idx = cuda.grid(1)
    if idx >= len(result):
        return
    
    if idx == 0:
        result[idx] = prices[idx]
        return
    
    result[idx] = alpha * prices[idx] + (1.0 - alpha) * result[idx - 1]


@cuda.jit
def bollinger_bands_kernel(prices, window_size, num_std, upper_band, lower_band, middle_band):
    """CUDA kernel for computing Bollinger Bands."""
    idx = cuda.grid(1)
    if idx >= len(prices) or idx < window_size - 1:
        return
    
    # Calculate moving average
    sum_val = float32(0.0)
    for i in range(window_size):
        sum_val += prices[idx - i]
    mean = sum_val / float32(window_size)
    
    # Calculate standard deviation
    variance = float32(0.0)
    for i in range(window_size):
        diff = prices[idx - i] - mean
        variance += diff * diff
    std_dev = math.sqrt(variance / float32(window_size))
    
    middle_band[idx] = mean
    upper_band[idx] = mean + num_std * std_dev
    lower_band[idx] = mean - num_std * std_dev


@cuda.jit
def rsi_kernel(prices, window_size, result):
    """CUDA kernel for computing Relative Strength Index (RSI)."""
    idx = cuda.grid(1)
    if idx >= len(result) or idx < window_size:
        return
    
    gains = float32(0.0)
    losses = float32(0.0)
    
    for i in range(1, window_size + 1):
        diff = prices[idx - i + 1] - prices[idx - i]
        if diff > 0:
            gains += diff
        else:
            losses += abs(diff)
    
    if losses == 0:
        result[idx] = float32(100.0)
    else:
        rs = gains / losses
        result[idx] = float32(100.0) - (float32(100.0) / (float32(1.0) + rs))


# Trading Strategy Kernels
@cuda.jit
def momentum_strategy_kernel(prices, returns, lookback_period, momentum_threshold, signals):
    """CUDA kernel for momentum strategy signals."""
    idx = cuda.grid(1)
    if idx >= len(signals) or idx < lookback_period:
        return
    
    # Calculate momentum
    momentum = prices[idx] / prices[idx - lookback_period] - 1.0
    
    # Generate signals
    if momentum > momentum_threshold:
        signals[idx] = int32(1)  # Buy signal
    elif momentum < -momentum_threshold:
        signals[idx] = int32(-1)  # Sell signal
    else:
        signals[idx] = int32(0)  # Hold


@cuda.jit
def mean_reversion_kernel(prices, window_size, z_threshold, signals):
    """CUDA kernel for mean reversion strategy."""
    idx = cuda.grid(1)
    if idx >= len(signals) or idx < window_size:
        return
    
    # Calculate rolling mean and std
    sum_val = float32(0.0)
    for i in range(window_size):
        sum_val += prices[idx - i]
    mean = sum_val / float32(window_size)
    
    variance = float32(0.0)
    for i in range(window_size):
        diff = prices[idx - i] - mean
        variance += diff * diff
    std_dev = math.sqrt(variance / float32(window_size))
    
    # Calculate z-score
    if std_dev > 0:
        z_score = (prices[idx] - mean) / std_dev
        
        if z_score > z_threshold:
            signals[idx] = int32(-1)  # Sell (price too high)
        elif z_score < -z_threshold:
            signals[idx] = int32(1)   # Buy (price too low)
        else:
            signals[idx] = int32(0)   # Hold
    else:
        signals[idx] = int32(0)


@cuda.jit
def swing_trading_kernel(prices, short_ma, long_ma, rsi, rsi_oversold, rsi_overbought, signals):
    """CUDA kernel for swing trading strategy."""
    idx = cuda.grid(1)
    if idx >= len(signals):
        return
    
    # Moving average crossover
    ma_signal = int32(0)
    if short_ma[idx] > long_ma[idx]:
        ma_signal = int32(1)
    elif short_ma[idx] < long_ma[idx]:
        ma_signal = int32(-1)
    
    # RSI signals
    rsi_signal = int32(0)
    if rsi[idx] < rsi_oversold:
        rsi_signal = int32(1)   # Oversold - buy
    elif rsi[idx] > rsi_overbought:
        rsi_signal = int32(-1)  # Overbought - sell
    
    # Combine signals
    if ma_signal == 1 and rsi_signal == 1:
        signals[idx] = int32(1)   # Strong buy
    elif ma_signal == -1 and rsi_signal == -1:
        signals[idx] = int32(-1)  # Strong sell
    elif ma_signal == 1 or rsi_signal == 1:
        signals[idx] = int32(1)   # Weak buy
    elif ma_signal == -1 or rsi_signal == -1:
        signals[idx] = int32(-1)  # Weak sell
    else:
        signals[idx] = int32(0)   # Hold


# Portfolio Optimization Kernels
@cuda.jit
def portfolio_returns_kernel(prices, weights, returns):
    """CUDA kernel for calculating portfolio returns."""
    idx = cuda.grid(1)
    if idx >= len(returns):
        return
    
    portfolio_return = float32(0.0)
    for i in range(len(weights)):
        if idx > 0:
            asset_return = (prices[idx, i] - prices[idx - 1, i]) / prices[idx - 1, i]
            portfolio_return += weights[i] * asset_return
    
    returns[idx] = portfolio_return


@cuda.jit
def risk_metrics_kernel(returns, window_size, var_confidence, metrics):
    """CUDA kernel for calculating risk metrics (VaR, CVaR, etc.)."""
    idx = cuda.grid(1)
    if idx >= len(metrics) or idx < window_size:
        return
    
    # Extract returns window
    window_returns = cuda.local.array(256, float32)  # Assuming max window size of 256
    if window_size > 256:
        return
    
    for i in range(window_size):
        window_returns[i] = returns[idx - window_size + 1 + i]
    
    # Sort returns (simple bubble sort for small arrays)
    for i in range(window_size):
        for j in range(window_size - 1 - i):
            if window_returns[j] > window_returns[j + 1]:
                temp = window_returns[j]
                window_returns[j] = window_returns[j + 1]
                window_returns[j + 1] = temp
    
    # Calculate VaR
    var_index = int(var_confidence * window_size)
    if var_index >= window_size:
        var_index = window_size - 1
    
    var = window_returns[var_index]
    
    # Calculate CVaR (Expected Shortfall)
    cvar_sum = float32(0.0)
    cvar_count = 0
    for i in range(var_index + 1):
        cvar_sum += window_returns[i]
        cvar_count += 1
    
    cvar = cvar_sum / float32(max(cvar_count, 1))
    
    # Calculate volatility
    mean_return = float32(0.0)
    for i in range(window_size):
        mean_return += window_returns[i]
    mean_return /= float32(window_size)
    
    variance = float32(0.0)
    for i in range(window_size):
        diff = window_returns[i] - mean_return
        variance += diff * diff
    volatility = math.sqrt(variance / float32(window_size))
    
    # Store metrics: [VaR, CVaR, Volatility, Mean Return]
    metrics[idx, 0] = var
    metrics[idx, 1] = cvar
    metrics[idx, 2] = volatility
    metrics[idx, 3] = mean_return


# Monte Carlo Simulation Kernels
@cuda.jit
def monte_carlo_price_simulation_kernel(initial_price, drift, volatility, dt, n_steps, 
                                      random_states, simulated_prices):
    """CUDA kernel for Monte Carlo price simulation."""
    idx = cuda.grid(1)
    if idx >= simulated_prices.shape[0]:
        return
    
    # Initialize price path
    price = initial_price
    simulated_prices[idx, 0] = price
    
    # Generate price path
    for step in range(1, n_steps):
        # Generate random number
        random_val = xoroshiro128p_uniform_float32(random_states, idx)
        
        # Box-Muller transformation for normal distribution
        if step % 2 == 1:
            # Store the random value for next iteration
            continue
        
        # Convert uniform to normal
        z = math.sqrt(-2.0 * math.log(random_val)) * math.cos(2.0 * math.pi * random_val)
        
        # Apply geometric Brownian motion
        price_change = drift * dt + volatility * math.sqrt(dt) * z
        price *= math.exp(price_change)
        
        simulated_prices[idx, step] = price


@cuda.jit
def option_pricing_kernel(spot_prices, strike, risk_free_rate, time_to_expiry, 
                         volatility, option_type, option_prices):
    """CUDA kernel for Black-Scholes option pricing."""
    idx = cuda.grid(1)
    if idx >= len(option_prices):
        return
    
    S = spot_prices[idx]
    K = strike
    r = risk_free_rate
    T = time_to_expiry
    sigma = volatility
    
    # Black-Scholes formula
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    # Approximation of cumulative normal distribution
    def norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    
    if option_type == 1:  # Call option
        option_prices[idx] = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:  # Put option
        option_prices[idx] = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# Backtesting Kernels
@cuda.jit
def backtest_strategy_kernel(prices, signals, initial_capital, transaction_cost, 
                           portfolio_values, positions, trades):
    """CUDA kernel for strategy backtesting."""
    idx = cuda.grid(1)
    if idx >= len(signals):
        return
    
    if idx == 0:
        portfolio_values[idx] = initial_capital
        positions[idx] = float32(0.0)
        trades[idx] = int32(0)
        return
    
    # Previous values
    prev_portfolio = portfolio_values[idx - 1]
    prev_position = positions[idx - 1]
    
    # Current signal
    signal = signals[idx]
    price = prices[idx]
    
    # Execute trades
    new_position = prev_position
    trade_executed = int32(0)
    
    if signal == 1 and prev_position <= 0:  # Buy signal
        shares_to_buy = (prev_portfolio * 0.95) / price  # Use 95% of capital
        new_position = shares_to_buy
        trade_cost = shares_to_buy * price * transaction_cost
        trade_executed = int32(1)
    elif signal == -1 and prev_position > 0:  # Sell signal
        trade_cost = prev_position * price * transaction_cost
        new_position = float32(0.0)
        trade_executed = int32(-1)
    
    # Calculate portfolio value
    if new_position > 0:
        portfolio_values[idx] = new_position * price
    else:
        portfolio_values[idx] = prev_portfolio
    
    positions[idx] = new_position
    trades[idx] = trade_executed


class OptimizedCUDAOperations:
    """High-level interface for optimized CUDA operations."""
    
    def __init__(self, kernel_manager: CUDAKernelManager):
        """Initialize CUDA operations."""
        self.kernel_manager = kernel_manager
        self.random_states = None
        
    def initialize_random_states(self, n_states: int = 10000):
        """Initialize random states for Monte Carlo simulations."""
        self.random_states = create_xoroshiro128p_states(n_states, seed=42)
        
    def compute_moving_average(self, prices: cp.ndarray, window_size: int) -> cp.ndarray:
        """Compute moving average using CUDA kernel."""
        result = cp.zeros_like(prices, dtype=cp.float32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        moving_average_kernel[blocks_per_grid, threads_per_block](
            prices, window_size, result
        )
        
        return result
    
    def compute_bollinger_bands(self, prices: cp.ndarray, window_size: int = 20, 
                              num_std: float = 2.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        """Compute Bollinger Bands using CUDA kernel."""
        upper_band = cp.zeros_like(prices, dtype=cp.float32)
        lower_band = cp.zeros_like(prices, dtype=cp.float32)
        middle_band = cp.zeros_like(prices, dtype=cp.float32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        bollinger_bands_kernel[blocks_per_grid, threads_per_block](
            prices, window_size, num_std, upper_band, lower_band, middle_band
        )
        
        return upper_band, lower_band, middle_band
    
    def compute_rsi(self, prices: cp.ndarray, window_size: int = 14) -> cp.ndarray:
        """Compute RSI using CUDA kernel."""
        result = cp.zeros_like(prices, dtype=cp.float32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        rsi_kernel[blocks_per_grid, threads_per_block](
            prices, window_size, result
        )
        
        return result
    
    def generate_momentum_signals(self, prices: cp.ndarray, returns: cp.ndarray,
                                lookback_period: int = 20, 
                                momentum_threshold: float = 0.02) -> cp.ndarray:
        """Generate momentum trading signals using CUDA kernel."""
        signals = cp.zeros(len(prices), dtype=cp.int32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        momentum_strategy_kernel[blocks_per_grid, threads_per_block](
            prices, returns, lookback_period, momentum_threshold, signals
        )
        
        return signals
    
    def generate_mean_reversion_signals(self, prices: cp.ndarray, window_size: int = 30,
                                      z_threshold: float = 2.0) -> cp.ndarray:
        """Generate mean reversion signals using CUDA kernel."""
        signals = cp.zeros(len(prices), dtype=cp.int32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        mean_reversion_kernel[blocks_per_grid, threads_per_block](
            prices, window_size, z_threshold, signals
        )
        
        return signals
    
    def generate_swing_signals(self, prices: cp.ndarray, short_ma: cp.ndarray, 
                             long_ma: cp.ndarray, rsi: cp.ndarray,
                             rsi_oversold: float = 30.0, 
                             rsi_overbought: float = 70.0) -> cp.ndarray:
        """Generate swing trading signals using CUDA kernel."""
        signals = cp.zeros(len(prices), dtype=cp.int32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        swing_trading_kernel[blocks_per_grid, threads_per_block](
            prices, short_ma, long_ma, rsi, rsi_oversold, rsi_overbought, signals
        )
        
        return signals
    
    def run_monte_carlo_simulation(self, initial_price: float, drift: float, 
                                 volatility: float, n_steps: int, 
                                 n_simulations: int, dt: float = 1/252) -> cp.ndarray:
        """Run Monte Carlo price simulation using CUDA kernel."""
        if self.random_states is None:
            self.initialize_random_states(n_simulations)
        
        simulated_prices = cp.zeros((n_simulations, n_steps), dtype=cp.float32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(n_simulations)
        
        monte_carlo_price_simulation_kernel[blocks_per_grid, threads_per_block](
            initial_price, drift, volatility, dt, n_steps, 
            self.random_states, simulated_prices
        )
        
        return simulated_prices
    
    def backtest_strategy(self, prices: cp.ndarray, signals: cp.ndarray,
                        initial_capital: float = 100000.0,
                        transaction_cost: float = 0.001) -> Dict[str, cp.ndarray]:
        """Backtest trading strategy using CUDA kernel."""
        portfolio_values = cp.zeros_like(prices, dtype=cp.float32)
        positions = cp.zeros_like(prices, dtype=cp.float32)
        trades = cp.zeros(len(prices), dtype=cp.int32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(prices))
        
        backtest_strategy_kernel[blocks_per_grid, threads_per_block](
            prices, signals, initial_capital, transaction_cost,
            portfolio_values, positions, trades
        )
        
        return {
            'portfolio_values': portfolio_values,
            'positions': positions,
            'trades': trades
        }
    
    def calculate_risk_metrics(self, returns: cp.ndarray, window_size: int = 252,
                             var_confidence: float = 0.05) -> cp.ndarray:
        """Calculate risk metrics using CUDA kernel."""
        metrics = cp.zeros((len(returns), 4), dtype=cp.float32)
        
        blocks_per_grid, threads_per_block = self.kernel_manager.get_optimal_grid_size(len(returns))
        
        risk_metrics_kernel[blocks_per_grid, threads_per_block](
            returns, window_size, var_confidence, metrics
        )
        
        return metrics


def create_cuda_operations(device_id: int = 0) -> OptimizedCUDAOperations:
    """Create optimized CUDA operations instance."""
    kernel_manager = CUDAKernelManager(device_id)
    return OptimizedCUDAOperations(kernel_manager)


# Performance benchmarking
def benchmark_cuda_kernels(data_sizes: List[int] = None) -> Dict[str, Any]:
    """Benchmark CUDA kernel performance."""
    if data_sizes is None:
        data_sizes = [1000, 10000, 100000, 1000000]
    
    cuda_ops = create_cuda_operations()
    results = {}
    
    for size in data_sizes:
        # Generate test data
        prices = cp.random.rand(size).astype(cp.float32) * 100 + 50
        
        # Benchmark moving average
        start_time = cp.cuda.Event()
        end_time = cp.cuda.Event()
        
        start_time.record()
        ma = cuda_ops.compute_moving_average(prices, 20)
        end_time.record()
        end_time.synchronize()
        
        ma_time = cp.cuda.get_elapsed_time(start_time, end_time)
        
        # Benchmark RSI
        start_time.record()
        rsi = cuda_ops.compute_rsi(prices, 14)
        end_time.record()
        end_time.synchronize()
        
        rsi_time = cp.cuda.get_elapsed_time(start_time, end_time)
        
        results[f'size_{size}'] = {
            'moving_average_ms': ma_time,
            'rsi_ms': rsi_time,
            'throughput_ma': size / (ma_time / 1000),
            'throughput_rsi': size / (rsi_time / 1000)
        }
    
    return results


if __name__ == "__main__":
    # Test CUDA kernels
    logger.info("Testing CUDA kernels...")
    
    # Create test data
    n_points = 10000
    prices = cp.random.rand(n_points).astype(cp.float32) * 100 + 50
    
    # Test operations
    cuda_ops = create_cuda_operations()
    
    # Test moving average
    ma = cuda_ops.compute_moving_average(prices, 20)
    logger.info(f"Moving average computed for {n_points} points")
    
    # Test RSI
    rsi = cuda_ops.compute_rsi(prices, 14)
    logger.info(f"RSI computed for {n_points} points")
    
    # Test momentum signals
    returns = cp.diff(prices) / prices[:-1]
    momentum_signals = cuda_ops.generate_momentum_signals(prices[1:], returns, 20, 0.02)
    logger.info(f"Momentum signals generated for {len(momentum_signals)} points")
    
    # Benchmark
    benchmark_results = benchmark_cuda_kernels()
    logger.info(f"Benchmark results: {benchmark_results}")
    
    print("CUDA kernels tested successfully!")