"""
Numba JIT-compiled kernels for CPU acceleration of QBMIA computations.
"""

import numpy as np
import numba as nb
from numba import jit, njit, prange, vectorize, cuda
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Check if CUDA is available through Numba
NUMBA_CUDA_AVAILABLE = cuda.is_available()

if NUMBA_CUDA_AVAILABLE:
    logger.info("Numba CUDA support is available")

@njit(fastmath=True, parallel=True, cache=True)
def quantum_gate_application_cpu(state_real: np.ndarray, state_imag: np.ndarray,
                                gate_real: np.ndarray, gate_imag: np.ndarray,
                                target_qubit: int, n_qubits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply single-qubit gate to quantum state (CPU optimized).

    Args:
        state_real: Real part of state vector
        state_imag: Imaginary part of state vector
        gate_real: Real part of gate matrix
        gate_imag: Imaginary part of gate matrix
        target_qubit: Target qubit index
        n_qubits: Total number of qubits

    Returns:
        Updated state vector (real and imaginary parts)
    """
    state_size = 1 << n_qubits
    qubit_mask = 1 << target_qubit

    new_state_real = state_real.copy()
    new_state_imag = state_imag.copy()

    for i in prange(state_size // 2):
        # Calculate indices for |0> and |1> components
        idx0 = (i >> target_qubit) << (target_qubit + 1) | (i & ((1 << target_qubit) - 1))
        idx1 = idx0 | qubit_mask

        # Get current amplitudes
        amp0_real = state_real[idx0]
        amp0_imag = state_imag[idx0]
        amp1_real = state_real[idx1]
        amp1_imag = state_imag[idx1]

        # Apply gate matrix
        # new_amp0 = gate[0,0] * amp0 + gate[0,1] * amp1
        new_state_real[idx0] = (gate_real[0, 0] * amp0_real - gate_imag[0, 0] * amp0_imag +
                               gate_real[0, 1] * amp1_real - gate_imag[0, 1] * amp1_imag)
        new_state_imag[idx0] = (gate_real[0, 0] * amp0_imag + gate_imag[0, 0] * amp0_real +
                               gate_real[0, 1] * amp1_imag + gate_imag[0, 1] * amp1_real)

        # new_amp1 = gate[1,0] * amp0 + gate[1,1] * amp1
        new_state_real[idx1] = (gate_real[1, 0] * amp0_real - gate_imag[1, 0] * amp0_imag +
                               gate_real[1, 1] * amp1_real - gate_imag[1, 1] * amp1_imag)
        new_state_imag[idx1] = (gate_real[1, 0] * amp0_imag + gate_imag[1, 0] * amp0_real +
                               gate_real[1, 1] * amp1_imag + gate_imag[1, 1] * amp1_real)

    return new_state_real, new_state_imag

@njit(fastmath=True, parallel=True, cache=True)
def nash_equilibrium_update_parallel(strategies: np.ndarray,
                                   payoff_matrix: np.ndarray,
                                   learning_rate: float,
                                   temperature: float) -> np.ndarray:
    """
    Parallel Nash equilibrium strategy update.

    Args:
        strategies: Current strategy profile
        payoff_matrix: Game payoff matrix
        learning_rate: Learning rate
        temperature: Softmax temperature

    Returns:
        Updated strategies
    """
    n_players, n_actions = strategies.shape
    new_strategies = np.zeros_like(strategies)

    for player in prange(n_players):
        # Calculate expected payoffs for each action
        expected_payoffs = np.zeros(n_actions)

        for action in range(n_actions):
            payoff = 0.0
            for opp in range(n_players):
                if opp != player:
                    for opp_action in range(n_actions):
                        payoff += strategies[opp, opp_action] * payoff_matrix[player, opp, action, opp_action]
            expected_payoffs[action] = payoff

        # Softmax with temperature
        max_payoff = np.max(expected_payoffs)
        exp_payoffs = np.exp((expected_payoffs - max_payoff) / temperature)
        softmax_probs = exp_payoffs / np.sum(exp_payoffs)

        # Update strategy with learning rate
        new_strategies[player] = strategies[player] + learning_rate * (softmax_probs - strategies[player])

    return new_strategies

@njit(fastmath=True, parallel=True, cache=True)
def batch_cosine_similarity(patterns: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between multiple patterns and queries.

    Args:
        patterns: Array of patterns (n_patterns, feature_dim)
        queries: Array of queries (n_queries, feature_dim)

    Returns:
        Similarity matrix (n_patterns, n_queries)
    """
    n_patterns, _ = patterns.shape
    n_queries, feature_dim = queries.shape

    similarities = np.zeros((n_patterns, n_queries))

    for i in prange(n_patterns):
        pattern_norm = 0.0
        for k in range(feature_dim):
            pattern_norm += patterns[i, k] * patterns[i, k]
        pattern_norm = np.sqrt(pattern_norm)

        for j in range(n_queries):
            query_norm = 0.0
            dot_product = 0.0

            for k in range(feature_dim):
                dot_product += patterns[i, k] * queries[j, k]
                query_norm += queries[j, k] * queries[j, k]

            query_norm = np.sqrt(query_norm)

            if pattern_norm > 0 and query_norm > 0:
                similarities[i, j] = dot_product / (pattern_norm * query_norm)

    return similarities

@njit(fastmath=True, parallel=True, cache=True)
def manipulation_detection_kernel(order_flow: np.ndarray,
                                price_changes: np.ndarray,
                                window_size: int) -> np.ndarray:
    """
    Detect market manipulation patterns using parallel processing.

    Args:
        order_flow: Order flow data (size, price, time, cancelled)
        price_changes: Price change series
        window_size: Detection window size

    Returns:
        Manipulation scores
    """
    n_orders = len(order_flow)
    manipulation_scores = np.zeros(n_orders)

    for i in prange(window_size, n_orders):
        window_start = i - window_size

        # Spoofing detection
        large_order_threshold = np.percentile(order_flow[window_start:i, 0], 90)
        large_orders = 0
        cancelled_large = 0

        for j in range(window_start, i):
            if order_flow[j, 0] > large_order_threshold:
                large_orders += 1
                if order_flow[j, 3] > 0:  # Cancelled flag
                    cancelled_large += 1

        spoofing_score = cancelled_large / max(1, large_orders)

        # Layering detection
        price_levels = np.unique(order_flow[window_start:i, 1])
        if len(price_levels) > 5:
            # Check for similar order sizes across levels
            size_variance = np.var(order_flow[window_start:i, 0])
            size_mean = np.mean(order_flow[window_start:i, 0])
            cv = size_variance / (size_mean * size_mean + 1e-8)

            layering_score = (1.0 - cv) if cv < 1.0 else 0.0
        else:
            layering_score = 0.0

        # Pump and dump detection
        price_volatility = np.std(price_changes[max(0, i-window_size):i])
        price_trend = np.mean(price_changes[max(0, i-window_size//2):i])

        pump_score = 0.0
        if price_trend > 0.01 and i + window_size//2 < len(price_changes):
            future_trend = np.mean(price_changes[i:i+window_size//2])
            if future_trend < -0.01:
                pump_score = abs(price_trend) + abs(future_trend)

        # Combined score
        manipulation_scores[i] = 0.4 * spoofing_score + 0.3 * layering_score + 0.3 * pump_score

    return manipulation_scores

@njit(fastmath=True, parallel=True, cache=True)
def volatility_garch_estimation(returns: np.ndarray,
                              omega: float,
                              alpha: float,
                              beta: float) -> np.ndarray:
    """
    GARCH(1,1) volatility estimation.

    Args:
        returns: Return series
        omega: Constant term
        alpha: ARCH coefficient
        beta: GARCH coefficient

    Returns:
        Conditional volatility series
    """
    n = len(returns)
    volatility = np.zeros(n)

    # Initialize with unconditional volatility
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))

    for t in range(1, n):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 +
                               beta * volatility[t-1]**2)

    return volatility

@vectorize(['float64(float64, float64)'], target='parallel')
def fast_option_payoff(spot_price: float, strike_price: float) -> float:
    """
    Vectorized option payoff calculation.

    Args:
        spot_price: Current asset price
        strike_price: Option strike price

    Returns:
        Option payoff
    """
    return max(0.0, spot_price - strike_price)

@njit(fastmath=True, cache=True)
def portfolio_var_calculation(weights: np.ndarray,
                            returns: np.ndarray,
                            confidence_level: float = 0.95) -> float:
    """
    Calculate portfolio Value at Risk (VaR).

    Args:
        weights: Portfolio weights
        returns: Historical returns matrix
        confidence_level: VaR confidence level

    Returns:
        Portfolio VaR
    """
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)

    # Sort returns
    sorted_returns = np.sort(portfolio_returns)

    # Calculate VaR
    var_index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[var_index]

    return var

# CUDA kernels if available
if NUMBA_CUDA_AVAILABLE:

    @cuda.jit
    def matrix_multiply_cuda(A, B, C):
        """
        CUDA kernel for matrix multiplication.

        Args:
            A: First matrix
            B: Second matrix
            C: Output matrix
        """
        row, col = cuda.grid(2)

        if row < C.shape[0] and col < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp

    @cuda.jit
    def monte_carlo_option_pricing_cuda(paths, S0, K, r, sigma, T, dt, prices):
        """
        CUDA kernel for Monte Carlo option pricing.

        Args:
            paths: Random normal samples
            S0: Initial price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            dt: Time step
            prices: Output option prices
        """
        idx = cuda.grid(1)
        if idx < paths.shape[0]:
            S = S0
            n_steps = paths.shape[1]

            for i in range(n_steps):
                S = S * cuda.exp((r - 0.5 * sigma * sigma) * dt +
                                sigma * cuda.sqrt(dt) * paths[idx, i])

            prices[idx] = max(0.0, S - K) * cuda.exp(-r * T)

# Utility functions for kernel selection
def get_optimal_kernel(operation: str, data_size: int,
                      use_gpu: bool = False) -> callable:
    """
    Select optimal kernel based on operation and data size.

    Args:
        operation: Operation type
        data_size: Size of data
        use_gpu: Whether to use GPU if available

    Returns:
        Optimal kernel function
    """
    if use_gpu and NUMBA_CUDA_AVAILABLE and data_size > 10000:
        kernel_map = {
            'matrix_multiply': matrix_multiply_cuda,
            'option_pricing': monte_carlo_option_pricing_cuda
        }
        return kernel_map.get(operation)

    # CPU kernels
    kernel_map = {
        'quantum_gate': quantum_gate_application_cpu,
        'nash_update': nash_equilibrium_update_parallel,
        'pattern_matching': batch_cosine_similarity,
        'manipulation_detection': manipulation_detection_kernel,
        'volatility_estimation': volatility_garch_estimation,
        'portfolio_var': portfolio_var_calculation
    }

    return kernel_map.get(operation)
