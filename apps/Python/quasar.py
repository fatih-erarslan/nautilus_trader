#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUASAR: Quantum Unified Star Agentic Reasoning

A hybrid system combining Q*-River with Quantum Agentic Reasoning for 
sophisticated financial decision making. Designed to integrate with
the Panarchy Adaptive Decision System (PADS).

Features:
- Q* reinforcement learning with quantum-inspired enhancements
- River ML drift and anomaly detection
- QAR quantum circuit-based decision making
- LMSR probability aggregation
- Numba and Catalyst optimization for high performance
- Panarchy theory for market regime identification

Author: Claude AI
"""

import os
import time
import pickle
import logging
import threading
import warnings
import uuid
from datetime import datetime
from pathlib import Path
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# Optimization libraries
try:
    import numba
    from numba import njit, prange, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator for graceful fallback
    def njit(*args, **kwargs):
        if callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    
    def prange(*args):
        return range(*args)

# JAX for acceleration when available
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, grad, random
    JAX_AVAILABLE = True
    # Enable 64-bit for better precision in financial calculations
    jax.config.update("jax_enable_x64", True)
    # Default to CPU to avoid CUDA initialization issues
    os.environ['JAX_PLATFORM_NAME'] = os.environ.get('JAX_PLATFORM_NAME', 'cpu')
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # Fallback to regular numpy

# Quantum computing with PennyLane
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    # Check for PennyLane Catalyst for speedups
    try:
        from pennylane import catalyst
        CATALYST_AVAILABLE = True
        # Use qjit for quantum circuit acceleration
        qjit = catalyst.qjit
    except ImportError:
        CATALYST_AVAILABLE = False
        # Create dummy decorator if Catalyst is not available
        def qjit(*args, **kwargs):
            if callable(args[0]):
                return args[0]
            else:
                def decorator(func):
                    return func
                return decorator
    
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    qnp = np  # Fallback to standard numpy
    # Create dummy decorator
    def qjit(*args, **kwargs):
        if callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator

# River for online learning
try:
    import river
    from river import drift, anomaly, preprocessing, metrics, stats, feature_selection
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("quasar.log"), logging.StreamHandler()],
)

logger = logging.getLogger("QUASAR")

# Import the hardware manager and other components
from hardware_manager import HardwareManager

# Import from qar.py and logarithmic_market_scoring_rule.py
from qar import (
    DecisionType, MarketPhase, TradingDecision, 
    CircuitCache, QuantumAgenticReasoning
)
from enhanced_lmsr import (
    LogarithmicMarketScoringRule, LMSRConfig, 
    ProbabilityConversionMethod, AggregationMethod
)

# Import from qstar_river.py (assuming this is a module based on what we developed)
from pulsar import (
    QStarLearningAgent, ExperienceBuffer, 
    TradingAction as QStarAction, CerebellumSNN, QuantumOptimizer
)

from river_ml import RiverOnlineML

# Enhanced Trading Action with integration with both QStar and QAR decision types
class UnifiedTradingAction:
    """
    Unified trading action definitions for the QUASAR system.
    
    Integrates action types from both QStar and QAR systems.
    """
    
    # Core actions
    BUY = 0
    SELL = 1
    HOLD = 2
    
    # Advanced actions
    DECREASE = 3
    INCREASE = 4
    HEDGE = 5
    EXIT = 6
    
    @staticmethod
    def get_num_actions() -> int:
        """Get number of possible actions."""
        return 7
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """Get human-readable action name."""
        actions = {
            UnifiedTradingAction.BUY: "BUY",
            UnifiedTradingAction.SELL: "SELL",
            UnifiedTradingAction.HOLD: "HOLD",
            UnifiedTradingAction.DECREASE: "DECREASE",
            UnifiedTradingAction.INCREASE: "INCREASE",
            UnifiedTradingAction.HEDGE: "HEDGE",
            UnifiedTradingAction.EXIT: "EXIT"
        }
        return actions.get(action, "UNKNOWN")
    
    @staticmethod
    def action_to_signal(action: int) -> float:
        """Convert action to normalized signal value (-1 to 1)."""
        signals = {
            UnifiedTradingAction.BUY: 1.0,
            UnifiedTradingAction.INCREASE: 0.5,
            UnifiedTradingAction.HOLD: 0.0,
            UnifiedTradingAction.DECREASE: -0.25,
            UnifiedTradingAction.HEDGE: -0.5,
            UnifiedTradingAction.SELL: -0.75,
            UnifiedTradingAction.EXIT: -1.0
        }
        return signals.get(action, 0.0)
    
    @staticmethod
    def from_qstar_action(qstar_action: int) -> int:
        """Convert QStar action to unified action."""
        mapping = {
            QStarAction.BUY: UnifiedTradingAction.BUY,
            QStarAction.SELL: UnifiedTradingAction.SELL,
            QStarAction.HOLD: UnifiedTradingAction.HOLD,
            QStarAction.DECREASE: UnifiedTradingAction.DECREASE,
            QStarAction.INCREASE: UnifiedTradingAction.INCREASE,
            QStarAction.HEDGE: UnifiedTradingAction.HEDGE
        }
        return mapping.get(qstar_action, UnifiedTradingAction.HOLD)
    
    @staticmethod
    def from_qar_decision(qar_decision: DecisionType) -> int:
        """Convert QAR decision type to unified action."""
        mapping = {
            DecisionType.BUY: UnifiedTradingAction.BUY,
            DecisionType.SELL: UnifiedTradingAction.SELL,
            DecisionType.HOLD: UnifiedTradingAction.HOLD,
            DecisionType.EXIT: UnifiedTradingAction.EXIT,
            DecisionType.INCREASE: UnifiedTradingAction.INCREASE,
            DecisionType.DECREASE: UnifiedTradingAction.DECREASE
        }
        return mapping.get(qar_decision, UnifiedTradingAction.HOLD)


@dataclass
class MarketRegimeState:
    """
    Market regime state information based on Panarchy theory.
    
    Contains information about the current market phase, characteristics,
    and transition probabilities.
    """
    phase: MarketPhase = MarketPhase.UNKNOWN
    volatility: float = 0.5
    trend_strength: float = 0.5
    resilience: float = 0.5
    transition_probabilities: Dict[MarketPhase, float] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Initialize transition probabilities if empty."""
        if not self.transition_probabilities:
            # Default uniform transition probabilities
            phases = [phase for phase in MarketPhase if phase != MarketPhase.UNKNOWN]
            self.transition_probabilities = {
                phase: 1.0 / len(phases) for phase in phases
            }


@dataclass
class QUASARState:
    """System state tracking for QUASAR."""
    is_initialized: bool = False
    is_trained: bool = False
    qstar_ready: bool = False
    qar_ready: bool = False
    last_update_time: float = field(default_factory=time.time)
    last_decision: Optional[TradingDecision] = None
    last_prediction: Dict[str, Any] = field(default_factory=dict)
    market_regime: MarketRegimeState = field(default_factory=MarketRegimeState)
    

@dataclass
class QUASARConfig:
    """Configuration for the QUASAR system."""
    # General settings
    quantum_enabled: bool = True
    use_cerebellum: bool = True
    use_numba: bool = True
    use_jax: bool = True
    use_catalyst: bool = True
    log_level: int = logging.INFO
    
    # Component weights
    qstar_weight: float = 0.4
    qar_weight: float = 0.4
    lmsr_weight: float = 0.2
    
    # Decision parameters
    decision_threshold: float = 0.6
    anomaly_threshold: float = 0.75
    drift_sensitivity: float = 0.05
    
    # QStar settings
    qstar_learning_rate: float = 0.05
    qstar_discount_factor: float = 0.97
    qstar_batch_size: int = 64
    qstar_num_states: int = 200
    
    # QAR settings
    qar_memory_length: int = 50
    qar_num_factors: int = 8
    qar_cache_size: int = 100
    
    # LMSR settings
    lmsr_liquidity_parameter: float = 100.0
    lmsr_min_probability: float = 0.001
    lmsr_max_probability: float = 0.999
    
    # Hardware settings
    max_workers: int = 4
    device: str = "default.qubit"
    shots: int = 1000
    


class OptimizedLMSR(LogarithmicMarketScoringRule):
    """
    Optimized implementation of LMSR using Numba for performance.
    
    This adapts the LogarithmicMarketScoringRule implementation with
    performance optimizations for critical functions.
    """
    
    def __init__(self, config: Optional[LMSRConfig] = None):
        """Initialize the optimized LMSR system."""
        self.config = config if config is not None else LMSRConfig()
        self.lmsr = LogarithmicMarketScoringRule(config=self.config)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Optimize methods if numba is available
        if NUMBA_AVAILABLE:
            self._optimize_methods()
    
    def _optimize_methods(self):
        """Apply Numba optimization to critical methods."""
        # Don't modify original LMSR - create optimized versions
        self._fast_normalize_probability = njit(cache=True)(self._create_fast_normalize_probability())
        self._fast_to_log_odds = njit(cache=True)(self._create_fast_to_log_odds())
        self._fast_from_log_odds = njit(cache=True)(self._create_fast_from_log_odds())
        self._fast_sigmoid = njit(cache=True)(self._create_fast_sigmoid())
        
        if cuda.is_available():
            # Create CUDA versions for large batch operations
            self._cuda_normalize_probabilities = cuda.jit(self._create_cuda_normalize_probabilities())
            self._cuda_to_log_odds = cuda.jit(self._create_cuda_to_log_odds())
            self._cuda_from_log_odds = cuda.jit(self._create_cuda_from_log_odds())
    
    def _create_fast_normalize_probability(self):
        """Create a Numba-optimized version of normalize_probability."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def fast_normalize_probability(prob):
            """Normalize probability to valid range."""
            if np.isnan(prob) or np.isinf(prob):
                return 0.5
            return max(min(float(prob), max_prob), min_prob)
        
        return fast_normalize_probability
    
    def _create_fast_to_log_odds(self):
        """Create a Numba-optimized version of to_log_odds."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def fast_to_log_odds(prob):
            """Convert probability to log-odds."""
            prob = max(min(prob, max_prob), min_prob)
            return np.log(prob / (1.0 - prob))
        
        return fast_to_log_odds
    
    def _create_fast_from_log_odds(self):
        """Create a Numba-optimized version of from_log_odds."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def fast_from_log_odds(log_odds):
            """Convert log-odds to probability."""
            if log_odds > 709:
                return max_prob
            elif log_odds < -709:
                return min_prob
            
            prob = 1.0 / (1.0 + np.exp(-log_odds))
            return max(min(prob, max_prob), min_prob)
        
        return fast_from_log_odds
    
    def _create_fast_sigmoid(self):
        """Create a Numba-optimized sigmoid function."""
        def fast_sigmoid(x):
            """Fast sigmoid implementation."""
            return 1.0 / (1.0 + np.exp(-x))
        
        return fast_sigmoid
    
    def _create_cuda_normalize_probabilities(self):
        """Create a CUDA kernel for batch probability normalization."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def cuda_normalize_probabilities(probs, result):
            """CUDA kernel for normalizing probabilities."""
            i = cuda.grid(1)
            if i < probs.size:
                if np.isnan(probs[i]) or np.isinf(probs[i]):
                    result[i] = 0.5
                else:
                    result[i] = max(min(probs[i], max_prob), min_prob)
        
        return cuda_normalize_probabilities
    
    def _create_cuda_to_log_odds(self):
        """Create a CUDA kernel for batch probability to log-odds conversion."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def cuda_to_log_odds(probs, result):
            """CUDA kernel for converting probabilities to log-odds."""
            i = cuda.grid(1)
            if i < probs.size:
                p = max(min(probs[i], max_prob), min_prob)
                result[i] = np.log(p / (1.0 - p))
        
        return cuda_to_log_odds
    
    def _create_cuda_from_log_odds(self):
        """Create a CUDA kernel for batch log-odds to probability conversion."""
        min_prob = self.config.min_probability
        max_prob = self.config.max_probability
        
        def cuda_from_log_odds(log_odds, result):
            """CUDA kernel for converting log-odds to probabilities."""
            i = cuda.grid(1)
            if i < log_odds.size:
                lo = log_odds[i]
                if lo > 709:
                    result[i] = max_prob
                elif lo < -709:
                    result[i] = min_prob
                else:
                    p = 1.0 / (1.0 + np.exp(-lo))
                    result[i] = max(min(p, max_prob), min_prob)
        
        return cuda_from_log_odds
    
    def normalize_probability(self, prob: float) -> float:
        """Normalize probability with optimized implementation."""
        if NUMBA_AVAILABLE:
            return self._fast_normalize_probability(prob)
        else:
            return self.lmsr.normalize_probability(prob)
    
    def to_log_odds(self, probability: float) -> float:
        """Convert probability to log-odds with optimized implementation."""
        if NUMBA_AVAILABLE:
            return self._fast_to_log_odds(probability)
        else:
            return self.lmsr.to_log_odds(probability)
    
    def from_log_odds(self, log_odds: float) -> float:
        """Convert log-odds to probability with optimized implementation."""
        if NUMBA_AVAILABLE:
            return self._fast_from_log_odds(log_odds)
        else:
            return self.lmsr.from_log_odds(log_odds)
    
    def indicator_to_probability(self, 
                               value: float, 
                               min_val: float, 
                               max_val: float, 
                               center: Optional[float] = None,
                               steepness: float = 1.0,
                               method: Optional[Union[str, ProbabilityConversionMethod]] = None) -> float:
        """Convert indicator value to probability using LMSR."""
        return self.lmsr.indicator_to_probability(
            value, min_val, max_val, center, steepness, method
        )
    
    def aggregate_probabilities(self, 
                              probabilities: List[float],
                              method: Optional[Union[str, AggregationMethod]] = None) -> float:
        """Aggregate multiple probabilities."""
        return self.lmsr.aggregate_probabilities(probabilities, method)
    
    def weighted_aggregate(self, 
                          probabilities: List[float], 
                          weights: List[float]) -> float:
        """Perform weighted aggregation of probabilities."""
        return self.lmsr.weighted_aggregate(probabilities, weights)
    
    def update_with_evidence(self, 
                           prior_probability: float, 
                           evidence_probability: float) -> float:
        """Update prior probability with new evidence using Bayes' rule."""
        return self.lmsr.update_with_evidence(prior_probability, evidence_probability)
    
    def cost_function(self, quantities: List[float]) -> float:
        """Calculate LMSR cost function."""
        return self.lmsr.cost_function(quantities)
    
    def get_market_probability(self, quantities: List[float], index: int) -> float:
        """Derive implicit probability from current market state."""
        return self.lmsr.get_market_probability(quantities, index)
    
    def get_all_market_probabilities(self, quantities: List[float]) -> List[float]:
        """Calculate probabilities for all outcomes from market state."""
        return self.lmsr.get_all_market_probabilities(quantities)
    
    def calculate_cost_to_move(self, 
                             current_quantities: List[float],
                             target_probability: float,
                             outcome_index: int) -> float:
        """Calculate cost to move market to target probability."""
        return self.lmsr.calculate_cost_to_move(
            current_quantities, target_probability, outcome_index
        )
    
    def calculate_information_gain(self, 
                                 prior_probabilities: List[float],
                                 posterior_probabilities: List[float]) -> float:
        """Calculate information gain (KL divergence) between distributions."""
        return self.lmsr.calculate_information_gain(prior_probabilities, posterior_probabilities)
    
    def batch_process_indicators(self,
                               indicators: Dict[str, List[float]],
                               indicator_configs: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
        """Process multiple indicators in batch mode."""
        return self.lmsr.batch_process_indicators(indicators, indicator_configs)


class EnhancedQuantumCircuits:
    """
    Enhanced quantum circuits for QUASAR with Catalyst optimization.
    
    Provides optimized quantum circuits for decision making, feature
    extraction, and quantum annealing processes.
    """
    
    def __init__(self, device_name: str = "lightning.kokkos", shots: int = 1000):
        """Initialize the quantum circuit factory."""
        if not QUANTUM_AVAILABLE:
            raise ImportError("PennyLane is required for quantum circuits")
        
        self.device_name = device_name
        self.shots = shots
        
        # Initialize default device
        self.default_device = qml.device(device_name, shots=shots)
        
        # Circuit cache for reuse
        self.circuit_cache = {}
        
        # Initialize with random seed
        np.random.seed(42)
        
        # Initialize phase trackers
        self.phases = np.random.uniform(0, 2*np.pi, 10)
        
        # Apply Catalyst optimization if available
        self.use_catalyst = CATALYST_AVAILABLE
    
    @qjit
    def state_preparation_circuit(self, features: np.ndarray, n_qubits: int) -> qml.QNode:
        """
        Create an optimized circuit for state preparation based on features.
        
        Args:
            features: Feature vector to encode
            n_qubits: Number of qubits to use
            
        Returns:
            Quantum circuit for state preparation
        """
        # Ensure consistent device usage
        dev = qml.device(self.device_name, wires=n_qubits, shots=self.shots)
        
        # Cache key
        cache_key = f"state_prep_{n_qubits}"
        
        if cache_key not in self.circuit_cache:
            @qml.qnode(dev)
            def state_prep_circuit(features):
                # Normalize features
                norm_features = np.array(features) / np.sqrt(np.sum(np.array(features)**2) + 1e-8)
                
                # Amplitude encoding
                qml.templates.AmplitudeEmbedding(
                    norm_features, wires=range(n_qubits), normalize=True
                )
                
                # Create entanglement
                qml.templates.StronglyEntanglingLayers(
                    weights=np.random.uniform(0, np.pi, (2, n_qubits, 3)),
                    wires=range(n_qubits)
                )
                
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            # Store in cache
            self.circuit_cache[cache_key] = state_prep_circuit
        
        return self.circuit_cache[cache_key]
    
    @qjit
    def decision_circuit(self, features: np.ndarray, n_decision_qubits: int = 3) -> qml.QNode:
        """
        Create a quantum circuit for decision making.
        
        Args:
            features: Feature vector
            n_decision_qubits: Number of decision qubits
            
        Returns:
            Quantum circuit for decision making
        """
        # Determine total qubits needed
        n_feature_qubits = min(8, len(features))
        n_qubits = n_feature_qubits + n_decision_qubits
        
        # Create device
        dev = qml.device(self.device_name, wires=n_qubits, shots=self.shots)
        
        # Cache key
        cache_key = f"decision_{n_feature_qubits}_{n_decision_qubits}"
        
        if cache_key not in self.circuit_cache:
            @qml.qnode(dev)
            def decision_circuit(features, phases):
                # Normalize features
                features_array = np.array(features)
                norm_features = features_array / np.sqrt(np.sum(features_array**2) + 1e-8)
                
                # Encode features
                for i in range(min(n_feature_qubits, len(norm_features))):
                    qml.RY(norm_features[i] * np.pi, wires=i)
                
                # Apply phases
                for i in range(min(n_feature_qubits, len(phases))):
                    qml.RZ(phases[i], wires=i)
                
                # Feature qubits to decision qubits
                for i in range(n_feature_qubits):
                    for j in range(n_decision_qubits):
                        qml.CNOT(wires=[i, n_feature_qubits + j])
                        qml.RZ(norm_features[i % len(norm_features)] * np.pi, wires=n_feature_qubits + j)
                
                # Apply Hadamard to decision qubits for superposition
                for i in range(n_decision_qubits):
                    qml.Hadamard(wires=n_feature_qubits + i)
                
                # Final entanglement
                for i in range(n_decision_qubits - 1):
                    qml.CNOT(wires=[n_feature_qubits + i, n_feature_qubits + i + 1])
                
                # Measure decision qubits
                return [qml.expval(qml.PauliZ(n_feature_qubits + i)) for i in range(n_decision_qubits)]
            
            # Store in cache
            self.circuit_cache[cache_key] = decision_circuit
        
        return self.circuit_cache[cache_key]
    
    @qjit
    def regime_detection_circuit(self, market_indicators: np.ndarray, n_regimes: int = 4) -> qml.QNode:
        """
        Create a quantum circuit for market regime detection.
        
        Args:
            market_indicators: Market indicator values
            n_regimes: Number of possible regimes/states
            
        Returns:
            Quantum circuit for regime detection
        """
        # Determine qubit count
        n_indicator_qubits = min(6, len(market_indicators))
        n_regime_qubits = max(2, int(np.ceil(np.log2(n_regimes))))
        n_qubits = n_indicator_qubits + n_regime_qubits
        
        # Create device
        dev = qml.device(self.device_name, wires=n_qubits, shots=self.shots)
        
        # Cache key
        cache_key = f"regime_{n_indicator_qubits}_{n_regime_qubits}"
        
        if cache_key not in self.circuit_cache:
            @qml.qnode(dev)
            def regime_circuit(indicators, phases):
                # Normalize indicators
                indicators_array = np.array(indicators)
                norm_indicators = indicators_array / np.sqrt(np.sum(indicators_array**2) + 1e-8)
                
                # Encode indicators
                for i in range(min(n_indicator_qubits, len(norm_indicators))):
                    qml.RY(norm_indicators[i] * np.pi, wires=i)
                
                # Apply phases for quantum interference
                for i in range(min(n_indicator_qubits, len(phases))):
                    qml.RZ(phases[i], wires=i)
                
                # Create entanglement between indicators
                for i in range(n_indicator_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                
                # Connect indicators to regime qubits
                for i in range(n_indicator_qubits):
                    for j in range(n_regime_qubits):
                        qml.CNOT(wires=[i, n_indicator_qubits + j])
                        qml.RZ(norm_indicators[i % len(norm_indicators)] * np.pi / 2, 
                               wires=n_indicator_qubits + j)
                
                # Apply Hadamard to regime qubits
                for i in range(n_regime_qubits):
                    qml.Hadamard(wires=n_indicator_qubits + i)
                
                # Final controlled operations
                for i in range(n_regime_qubits - 1):
                    qml.CNOT(wires=[n_indicator_qubits + i, n_indicator_qubits + i + 1])
                
                # Measure regime qubits
                return [qml.expval(qml.PauliZ(n_indicator_qubits + i)) for i in range(n_regime_qubits)]
                
            # Store in cache
            self.circuit_cache[cache_key] = regime_circuit
        
        return self.circuit_cache[cache_key]
    
    def update_phases(self, delta: float = 0.1) -> None:
        """Update quantum phases for interference."""
        self.phases += np.random.uniform(-delta, delta, self.phases.shape)
    
    def get_decision(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Get decision based on quantum circuit.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (action_index, confidence)
        """
        if not QUANTUM_AVAILABLE:
            # Fallback to classical method
            action = np.random.randint(0, 7)  # 7 possible actions
            confidence = np.random.random()
            return action, confidence
        
        try:
            # Get decision circuit
            circuit = self.decision_circuit(features)
            
            # Run circuit
            measurements = circuit(features, self.phases)
            
            # Convert measurements to probabilities
            probs = [(m + 1) / 2 for m in measurements]
            
            # Map to action
            # First bit determines buy/sell direction
            # Second bit determines action strength
            # Third bit determines position action
            direction_bit = int(probs[0] > 0.5)
            strength_bit = int(probs[1] > 0.5)
            position_bit = int(probs[2] > 0.5)
            
            # Logic for mapping to actions
            if direction_bit == 0:  # Sell/reduce side
                if strength_bit == 0:  # Lower strength
                    if position_bit == 0:
                        action = UnifiedTradingAction.DECREASE
                    else:
                        action = UnifiedTradingAction.HEDGE
                else:  # Higher strength
                    if position_bit == 0:
                        action = UnifiedTradingAction.SELL
                    else:
                        action = UnifiedTradingAction.EXIT
            else:  # Buy/increase side
                if strength_bit == 0:  # Lower strength
                    if position_bit == 0:
                        action = UnifiedTradingAction.INCREASE
                    else:
                        action = UnifiedTradingAction.HOLD
                else:  # Higher strength
                    action = UnifiedTradingAction.BUY
            
            # Calculate confidence based on measurement certainty
            # High certainty = measurements far from 0.5
            certainty = np.mean([abs(p - 0.5) * 2 for p in probs])
            confidence = certainty
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Quantum decision error: {e}")
            # Fallback to classical
            action = UnifiedTradingAction.HOLD
            confidence = 0.3
            return action, confidence
    
    def get_market_regime(self, indicators: np.ndarray) -> Tuple[int, Dict[int, float]]:
        """
        Detect market regime using quantum circuit.
        
        Args:
            indicators: Market indicators
            
        Returns:
            Tuple of (regime_index, regime_probabilities)
        """
        if not QUANTUM_AVAILABLE:
            # Fallback to classical method
            regime = np.random.randint(0, 4)  # 4 possible regimes
            probs = {i: np.random.random() for i in range(4)}
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            return regime, probs
        
        try:
            # Get regime circuit
            circuit = self.regime_detection_circuit(indicators)
            
            # Run circuit
            measurements = circuit(indicators, self.phases)
            
            # Convert to regime
            # Use binary encoding of regime qubits
            n_regime_qubits = len(measurements)
            regime_bits = [(m + 1) / 2 > 0.5 for m in measurements]
            
            # Convert bits to integer (binary encoding)
            regime = 0
            for i, bit in enumerate(regime_bits):
                if bit:
                    regime += 2 ** i
            
            # Calculate probabilities for each regime
            probs = {}
            for i in range(2 ** n_regime_qubits):
                # Calculate probability for this regime
                prob = 1.0
                for j in range(n_regime_qubits):
                    bit_val = (i >> j) & 1
                    if bit_val == 1:
                        prob *= (measurements[j] + 1) / 2
                    else:
                        prob *= 1 - (measurements[j] + 1) / 2
                probs[i] = prob
            
            # Normalize probabilities
            total = sum(probs.values())
            if total > 0:
                probs = {k: v/total for k, v in probs.items()}
            
            return regime, probs
            
        except Exception as e:
            logger.error(f"Quantum regime detection error: {e}")
            # Fallback to classical
            regime = 0  # Default to unknown
            probs = {i: 0.25 for i in range(4)}
            return regime, probs

@njit
def calculate_state_index(features):
    """Calculate state index from feature vector."""
    # Simple hash function
    hash_value = 0
    for i in range(len(features)):
        # Discretize feature into 10 levels
        level = max(0, min(9, int((features[i] + 1) * 5)))
        hash_value += level * (10 ** i)
    
    # Modulo to get within state space
    return hash_value % 200  # 200 states

class QUASAR:
    """
    QUASAR: Quantum Unified Star Agentic Reasoning
    
    A hybrid system that combines Q*-River reinforcement learning with 
    Quantum Agentic Reasoning (QAR) for sophisticated financial trading
    decisions. Designed to integrate with PADS (Panarchy Adaptive 
    Decision System).
    
    Features:
    - Quantum-enhanced reinforcement learning
    - Online drift and anomaly detection
    - Market regime identification with Panarchy theory
    - LMSR probability aggregation
    - Hardware-accelerated computation
    """
    
    def __init__(self, config: Optional[QUASARConfig] = None):
        """Initialize the QUASAR system."""
        self.config = config if config is not None else QUASARConfig()
        
        # Configure logging
        self._configure_logging()
        
        # System state
        self.state = QUASARState()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Component initialization status
        self._initialized_components = set()
        
        # Initialize decision history
        self.decision_history = deque(maxlen=1000)
                
        # Initialize hardware manager
        self._initialize_hardware_manager()
        
        
        # Initialize components
        self._initialize_components()
    
    def _configure_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
    def _initialize_hardware_manager(self):
        """Initialize hardware manager for quantum operations."""
        try:
            # Create hardware manager without any assumptions about its interface
            self.hardware_manager = HardwareManager()
            
            # Check what attributes and methods are available
            hw_dir = dir(self.hardware_manager)
            
            # Log available methods for debugging
            self.logger.debug(f"Hardware manager methods: {hw_dir}")
            
            # Use whatever device is already configured in the hardware manager
            self.logger.info(f"Hardware manager initialized")
            self._initialized_components.add("hardware_manager")
            
        except Exception as e:
            self.logger.error(f"Error initializing hardware manager: {e}")
            # Create basic fallback hardware manager
            self.hardware_manager = HardwareManager()
            
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # QStar agent
            self._initialize_qstar()
            
            # QAR system
            self._initialize_qar()
            
            # LMSR system
            self._initialize_lmsr()
            
            # Quantum circuits
            self._initialize_quantum_circuits()
            
            # Mark system as initialized
            self.state.is_initialized = (
                "qstar" in self._initialized_components and
                "qar" in self._initialized_components and
                "lmsr" in self._initialized_components
            )
            
            self.logger.info(f"QUASAR initialization status: {self.state.is_initialized}")
            self.logger.info(f"Initialized components: {self._initialized_components}")
            
        except Exception as e:
            self.logger.error(f"Error initializing QUASAR: {e}")
            self.state.is_initialized = False
    
    def _initialize_qstar(self):
        """Initialize QStar components."""
        try:
            # QStar agent
            self.qstar_agent = QStarLearningAgent(
                states=self.config.qstar_num_states,
                actions=UnifiedTradingAction.get_num_actions(),
                learning_rate=self.config.qstar_learning_rate,
                discount_factor=self.config.qstar_discount_factor,
                batch_size=self.config.qstar_batch_size,
                use_quantum_representation=self.config.quantum_enabled
            )
            
            # River ML adapter
            self.river_ml = RiverOnlineML(
                drift_detector_type='adwin',
                anomaly_detector_type='hst',
                feature_window=50,
                drift_sensitivity=self.config.drift_sensitivity,
                anomaly_threshold=self.config.anomaly_threshold
            )
            
            # Mark as initialized
            self._initialized_components.add("qstar")
            self.state.qstar_ready = True
            self.logger.info("QStar components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing QStar: {e}")
            self.state.qstar_ready = False
    
    def _initialize_qar(self):
        """Initialize QAR components."""
        try:
            # Create QAR instance using the hardware manager
            self.qar = QuantumAgenticReasoning(
                hardware_manager=self.hardware_manager if QUANTUM_AVAILABLE else None,
                memory_length=self.config.qar_memory_length,
                decision_threshold=self.config.decision_threshold,
                num_factors=self.config.qar_num_factors,
                cache_size=self.config.qar_cache_size,
                log_level=self.config.log_level,
                use_classical=not self.config.quantum_enabled
            )
            
            # Mark as initialized
            self._initialized_components.add("qar")
            self.state.qar_ready = True
            self.logger.info("QAR components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing QAR: {e}")
            self.state.qar_ready = False
    
    def _initialize_lmsr(self):
        """Initialize LMSR components."""
        try:
            # Create LMSR config
            lmsr_config = LMSRConfig(
                liquidity_parameter=self.config.lmsr_liquidity_parameter,
                min_probability=self.config.lmsr_min_probability,
                max_probability=self.config.lmsr_max_probability,
                #liquidity_parameter=100.0,  # Controls price sensitivity
                #min_probability=self.min_probability,
                #max_probability=self.max_probability,
                use_numba=True,                     # Enable Numba acceleration
                #use_vectorization=enable_vectorization,  # Enable vectorized operations
                batch_size=1024,                    # Efficient batch size for large datasets
                hardware_aware_parallelism=True,    # Optimize for available hardware
                enable_parallel=True,               # Enable parallel processing
                max_workers=4,                      # Adjust based on system capabilities
                log_level=logging.INFO
            )
            # Create optimized LMSR
            self.lmsr = OptimizedLMSR(config=lmsr_config)
            
            # Mark as initialized
            self._initialized_components.add("lmsr")
            self.logger.info("LMSR components initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing LMSR: {e}")
    
    def _initialize_quantum_circuits(self):
        """Initialize quantum circuits."""
        try:
            if QUANTUM_AVAILABLE and self.config.quantum_enabled:
                self.quantum_circuits = EnhancedQuantumCircuits(
                    device_name=self.config.device,
                    shots=self.config.shots
                )
                self._initialized_components.add("quantum_circuits")
                self.logger.info("Quantum circuits initialized")
            else:
                self.quantum_circuits = None
                self.logger.info("Quantum circuits not available or disabled")
            
        except Exception as e:
            self.logger.error(f"Error initializing quantum circuits: {e}")
            self.quantum_circuits = None


    def _initialize_river_ml(self):
        """Initialize River ML components with robust error handling."""
        try:
            self.feature_scaler = preprocessing.StandardScaler()
            self.anomaly_detector = anomaly.HalfSpaceTrees(
                n_trees=50,
                height=8,
                window_size=50,
                seed=42
            )
            self.logger.info("River ML components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing River ML components: {e}")
            # Create dummy components that handle missing methods
            self.feature_scaler = type('DummyScaler', (), {
                'learn_one': lambda self, x: self,
                'transform_one': lambda self, x: x
            })()
            self.anomaly_detector = type('DummyDetector', (), {
                'score_one': lambda self, x: 0.0,
                'learn_one': lambda self, x: None
            })()

    
    def _calculate_state_index(self, features):
        """Call the JIT-optimized standalone function."""
        return calculate_state_index(features)
    
    def register_indicator(self, name: str, weight: float = 1.0) -> None:
        """
        Register a market indicator with the QAR subsystem.
        
        Args:
            name: Indicator name
            weight: Initial weight
        """
        if not self.state.qar_ready:
            self.logger.warning("QAR not initialized, cannot register indicator")
            return
        
        self.qar.register_factor(name, weight)
        self.logger.debug(f"Registered indicator {name} with weight {weight}")
    
    def extract_features(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract features from market data.
        
        Args:
            dataframe: Market data dataframe
            
        Returns:
            Dictionary with extracted features
        """
        # Check if dataframe is valid
        if dataframe is None or len(dataframe) == 0:
            return {}
        
        try:
            # Get the most recent data
            recent_data = dataframe.iloc[-50:] if len(dataframe) >= 50 else dataframe
            
            # Extract price data
            close_prices = recent_data['close'].values if 'close' in recent_data.columns else None
            high_prices = recent_data['high'].values if 'high' in recent_data.columns else None
            low_prices = recent_data['low'].values if 'low' in recent_data.columns else None
            
            if close_prices is None:
                # Try to find suitable numeric column
                numeric_cols = recent_data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    close_prices = recent_data[numeric_cols[0]].values
                else:
                    return {}
            
            # Calculate basic features
            features = {}
            
            # Price features
            features['close'] = close_prices[-1]
            
            # Returns
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                features['return'] = returns[-1]
                features['return_5'] = np.mean(returns[-5:]) if len(returns) >= 5 else returns[-1]
                features['return_10'] = np.mean(returns[-10:]) if len(returns) >= 10 else returns[-1]
                features['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                features['return'] = 0.0
                features['return_5'] = 0.0
                features['return_10'] = 0.0
                features['volatility'] = 0.1  # Default
            
            # Add common technical indicators if available
            indicator_columns = [
                'rsi', 'rsi_14', 'adx', 'adx_14', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'atr_14',
                'cci', 'cci_20', 'stoch_k', 'stoch_d'
            ]
            
            for indicator in indicator_columns:
                if indicator in recent_data.columns:
                    features[indicator] = recent_data[indicator].iloc[-1]
            
            # Add market regime indicators
            features['trend'] = features['return_10'] / features['volatility'] if features['volatility'] > 0 else 0
            features['momentum'] = np.sum(np.sign(returns[-10:])) / 10 if len(returns) >= 10 else 0
            
            # Calculate momentum strength
            if len(close_prices) > 20:
                sma_5 = np.mean(close_prices[-5:])
                sma_20 = np.mean(close_prices[-20:])
                features['momentum_strength'] = (sma_5 / sma_20) - 1
            else:
                features['momentum_strength'] = 0
                
            # Extract features for QAR
            qar_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    qar_features[key] = value
            
            # Extract features for QUASAR as vector
            vector_features = np.array([
                features.get('return', 0),
                features.get('return_5', 0),
                features.get('return_10', 0),
                features.get('volatility', 0.1),
                features.get('trend', 0),
                features.get('momentum', 0),
                features.get('momentum_strength', 0),
                features.get('rsi_14', 50) / 100 if 'rsi_14' in features else 0.5,
                features.get('adx_14', 15) / 100 if 'adx_14' in features else 0.15,
                features.get('macd', 0) * 10 if 'macd' in features else 0
            ])
            
            # Determine market phase
            market_phase = self._determine_market_phase(features)
            
            return {
                'scalar_features': features,
                'qar_features': qar_features,
                'vector_features': vector_features,
                'market_phase': market_phase
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def _determine_market_phase(self, features: Dict[str, float]) -> MarketPhase:
        """
        Determine market phase based on Panarchy theory.
        
        Args:
            features: Market features
            
        Returns:
            Market phase
        """
        # Default to unknown
        phase = MarketPhase.UNKNOWN
        
        # Get key features
        volatility = features.get('volatility', 0.2)
        trend = features.get('trend', 0)
        momentum = features.get('momentum', 0)
        rsi = features.get('rsi_14', 50)
        
        # Simple rules for market phases
        if trend > 0.5 and momentum > 0.3:
            # Strong uptrend with momentum - Growth phase
            phase = MarketPhase.GROWTH
        elif trend > 0 and momentum < 0.3 and rsi > 70:
            # Slowing momentum, extended trend - Conservation phase
            phase = MarketPhase.CONSERVATION
        elif trend < -0.5 and momentum < -0.3:
            # Strong downtrend - Release phase
            phase = MarketPhase.RELEASE
        elif trend < 0 and momentum > -0.3 and rsi < 30:
            # Slowing downtrend, oversold - Reorganization phase
            phase = MarketPhase.REORGANIZATION
        
        # Update market regime state
        self.state.market_regime.phase = phase
        self.state.market_regime.volatility = volatility
        self.state.market_regime.trend_strength = abs(trend)
        
        return phase
    
    def make_decision(self, 
                      dataframe: pd.DataFrame, 
                      position_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate trading decision based on market data.
        
        Args:
            dataframe: Market data
            position_state: Current position information
            
        Returns:
            Decision with action, confidence, and explanation
        """
        if not self.state.is_initialized:
            return {
                'action': UnifiedTradingAction.HOLD,
                'action_name': 'HOLD',
                'confidence': 0.0,
                'explanation': 'System not initialized'
            }
        
        try:
            with self._lock:
                # Extract features
                feature_data = self.extract_features(dataframe)
                
                if not feature_data:
                    return {
                        'action': UnifiedTradingAction.HOLD,
                        'action_name': 'HOLD',
                        'confidence': 0.0,
                        'explanation': 'Failed to extract features'
                    }
                
                # Get individual feature sets
                qar_features = feature_data['qar_features']
                scalar_features = feature_data['scalar_features']
                vector_features = feature_data['vector_features']
                market_phase = feature_data['market_phase']
                
                # Extract position information
                position_open, position_direction = self._extract_position_state(position_state)
                
                # Process with River ML for drift and anomaly detection
                river_results = self.river_ml.detect_anomalies(qar_features)
                
                # Check for anomalies and drift
                anomaly_detected = river_results.get('is_anomaly', False)
                drift_detected = river_results.get('drift_detected', False)
                
                # Get decision from QStar agent
                qstar_decision = self._get_qstar_decision(vector_features, position_open, position_direction)
                
                # Get decision from QAR
                qar_decision = self._get_qar_decision(qar_features, scalar_features, position_open, position_direction)
                
                # Get decision from quantum circuits
                quantum_decision = self._get_quantum_circuit_decision(vector_features)
                
                # Calculate final decision with LMSR
                final_decision = self._blend_decisions(
                    qstar_decision, qar_decision, quantum_decision,
                    anomaly_detected, drift_detected, market_phase
                )
                
                # Create complete decision result
                result = {
                    'action': final_decision.decision_type,
                    'action_name': UnifiedTradingAction.get_action_name(final_decision.decision_type),
                    'confidence': final_decision.confidence,
                    'reasoning': final_decision.reasoning,
                    'timestamp': datetime.now().isoformat(),
                    'qstar_decision': qstar_decision,
                    'qar_decision': qar_decision,
                    'quantum_decision': quantum_decision,
                    'anomaly_detected': anomaly_detected,
                    'drift_detected': drift_detected,
                    'market_phase': market_phase.name,
                    'actionable': final_decision.confidence >= self.config.decision_threshold
                }
                
                # Store decision in history
                self.decision_history.append(final_decision)
                
                # Update state
                self.state.last_decision = final_decision
                self.state.last_prediction = result
                self.state.last_update_time = time.time()
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error in make_decision: {e}")
            return {
                'action': UnifiedTradingAction.HOLD,
                'action_name': 'HOLD',
                'confidence': 0.0,
                'explanation': f'Error in decision making: {str(e)}'
            }
    
    def _extract_position_state(self, position_state: Optional[Dict[str, Any]]) -> Tuple[bool, int]:
        """
        Extract position information.
        
        Args:
            position_state: Position state dictionary
            
        Returns:
            Tuple of (position_open, position_direction)
        """
        position_open = False
        position_direction = 0
        
        if position_state:
            try:
                position_open = bool(position_state.get('position_open', False))
                position_direction = position_state.get('position_direction', 0)
                
                # Normalize direction to -1, 0, 1
                if position_direction > 0:
                    position_direction = 1
                elif position_direction < 0:
                    position_direction = -1
                else:
                    position_direction = 0
            except Exception as e:
                self.logger.warning(f"Error extracting position state: {e}")
        
        return position_open, position_direction
 
    

    def _get_qar_decision(self,
                         qar_features: Dict[str, float], # Features specifically for QAR
                         scalar_features: Dict[str, float], # Full feature dictionary (includes non-numeric potentially)
                         position_open: bool,
                         position_direction: int) -> TradingDecision:
        """
        Get trading decision from Quantum Agentic Reasoning (QAR) subsystem.

        Args:
            qar_features: Dictionary of numerical features suitable for QAR input.
            scalar_features: Dictionary of all scalar features (can include non-numeric).
            position_open: Whether a position is currently open.
            position_direction: Direction of the current position (-1, 0, 1).

        Returns:
            TradingDecision: Decision from the QAR component.
        """
        # Default decision in case of errors or if QAR is not ready
        default_decision = TradingDecision(
                decision_type=UnifiedTradingAction.HOLD,
                confidence=0.1,
                reasoning="QAR component error or not ready.",
                timestamp=datetime.now()
            )

        # Check if the QAR component is ready
        if not self.state.qar_ready or not hasattr(self, 'qar'):
            self.logger.warning("QAR component not ready or available during decision request.")
            return default_decision

        try:
            # --- Prepare Inputs for QAR ---
            # Prepare position state dictionary
            position_state_input = {
                'position_open': position_open,
                'position_direction': position_direction
                # Add other relevant position info if needed by QAR, e.g., entry price
            }

            # Prepare market data dictionary (select relevant fields from scalar_features)
            market_data_input = {
                'close': scalar_features.get('close', 0.0),
                'volume': scalar_features.get('volume', 0.0), # If volume feature exists
                'timestamp': time.time(),
                # Include features relevant to QAR's internal logic/regime detection
                'trend': scalar_features.get('trend', 0.0),
                'volatility': scalar_features.get('volatility', 0.5),
                'momentum': scalar_features.get('momentum', 0.0),
                # Pass the externally determined market phase
                'panarchy_phase': self.state.market_regime.phase.value if self.state.market_regime else MarketPhase.UNKNOWN.value
            }

            # Ensure QAR features are suitable (e.g., handle NaNs if QAR requires it)
            qar_input_features_clean = {k: v for k, v in qar_features.items() if np.isfinite(v)}
            if not qar_input_features_clean:
                 self.logger.warning("No finite features available for QAR input.")
                 # Return default or handle as QAR prefers (maybe HOLD?)
                 return default_decision

            # --- Make Decision with QAR ---
            # QAR's make_decision might raise errors if inputs are wrong type/shape
            qar_result = self.qar.make_decision(qar_input_features_clean, market_data_input, position_state_input)

            # --- Process QAR Result ---
            # Extract decision safely using .get()
            decision_type_str = qar_result.get('decision_type', 'HOLD') # Default to HOLD string
            confidence = qar_result.get('confidence', 0.3) # Default confidence
            explanation = qar_result.get('explanation', "QAR provided no explanation.")

            # Convert decision type string (from QAR) to QAR's internal DecisionType enum
            try:
                # Ensure DecisionType enum is available in this scope (imported from qar.py)
                qar_decision_type_enum = DecisionType[decision_type_str.upper()]
            except KeyError:
                self.logger.warning(f"QAR returned unknown decision type string '{decision_type_str}'. Defaulting to HOLD.")
                qar_decision_type_enum = DecisionType.HOLD
            except NameError:
                 self.logger.error("QAR DecisionType enum not defined or imported. Cannot map decision.")
                 return default_decision # Critical failure if enum missing

            # Convert QAR's DecisionType enum to QUASAR's UnifiedTradingAction integer
            unified_action = UnifiedTradingAction.from_qar_decision(qar_decision_type_enum)

            # Validate confidence
            if not isinstance(confidence, (float, int)) or not (0 <= confidence <= 1):
                 self.logger.warning(f"QAR returned invalid confidence {confidence}. Clamping to [0,1].")
                 confidence = np.clip(confidence, 0.0, 1.0)


            return TradingDecision(
                decision_type=unified_action,
                confidence=float(confidence),
                reasoning=explanation,
                timestamp=datetime.now()
            )

        except AttributeError as e_attr:
             # Catch attribute errors on the QAR object
             self.logger.error(f"AttributeError in QAR decision (likely QAR object issue): {e_attr}", exc_info=True)
             default_decision.reasoning = f"QAR decision failed: Missing attribute '{e_attr}'"
             return default_decision
        except Exception as e:
            # Catch any other unexpected errors during QAR processing
            self.logger.error(f"Unexpected error getting QAR decision: {e}", exc_info=True)
            default_decision.reasoning = f"QAR decision unexpected error: {str(e)}"
            return default_decision



    def _get_qstar_decision(self,
                           vector_features: np.ndarray,
                           position_open: bool,
                           position_direction: int) -> TradingDecision:
        """
        Get trading decision from QStar subsystem, ensuring NumPy arrays
        are used for calculations where appropriate.

        Args:
            vector_features: Feature vector (NumPy array)
            position_open: Whether position is open
            position_direction: Position direction

        Returns:
            TradingDecision: Decision from the QStar agent
        """
        # Default decision in case of errors
        default_decision = TradingDecision(
                decision_type=UnifiedTradingAction.HOLD,
                confidence=0.1, # Low confidence on error
                reasoning="QStar component error or not ready.",
                timestamp=datetime.now()
            )

        # Check if the QStar component is ready
        if not self.state.qstar_ready or not hasattr(self, 'qstar_agent'):
            self.logger.warning("QStar agent not ready or not available during decision request.")
            return default_decision

        try:
            # Calculate state index (assuming _calculate_state_index handles potential errors)
            state = self._calculate_state_index(vector_features)
            # Validate state index before using it
            if not (0 <= state < self.qstar_agent.states):
                 self.logger.error(f"Invalid state index {state} calculated. Max states: {self.qstar_agent.states}")
                 return default_decision # Return default if state index is invalid

            # --- Check if agent has choose_action ---
            if not hasattr(self.qstar_agent, 'choose_action') or not callable(self.qstar_agent.choose_action):
                 self.logger.error("QStar agent object missing callable 'choose_action' method.")
                 # This indicates a more severe problem, potentially during initialization
                 # You might want to raise an error or attempt reinitialization here.
                 return default_decision

            # --- Get action from Q* agent ---
            action = self.qstar_agent.choose_action(state)
            decision_type = int(action) # Assuming direct mapping

            # --- Calculate confidence from Q-values ---
            confidence = 0.1 # Default low confidence
            try:
                # Check if q_table exists and state is valid before accessing
                if hasattr(self.qstar_agent, 'q_table') and state < self.qstar_agent.q_table.shape[0]:
                    # Explicitly convert the relevant Q-table slice to NumPy array
                    q_values = np.asarray(self.qstar_agent.q_table[state])

                    # Check for non-finite values before calculations
                    q_values_finite = q_values[np.isfinite(q_values)]

                    if len(q_values_finite) > 1: # Need at least 2 values for range
                        max_q = np.max(q_values_finite)
                        min_q = np.min(q_values_finite)
                        mean_q_abs = np.abs(np.mean(q_values_finite)) + 1e-6 # Avoid div by zero

                        # Confidence based on normalized range
                        q_range = max_q - min_q
                        calculated_confidence = np.clip(q_range / mean_q_abs, 0.1, 0.9) # Clamp range
                        confidence = float(calculated_confidence)
                    elif len(q_values_finite) == 1:
                        confidence = 0.5 # If only one valid Q-value, medium confidence?
                    else:
                        self.logger.warning(f"All Q-values non-finite or empty for state {state}. Confidence remains low.")
                        # confidence remains 0.1
                else:
                    self.logger.error(f"Q-table not found or state index {state} out of bounds. Q-table shape: {getattr(self.qstar_agent, 'q_table', 'N/A')}")
                    # confidence remains 0.1

            except Exception as e_conf:
                 self.logger.error(f"Error calculating QStar confidence for state {state}: {e_conf}", exc_info=True)
                 # confidence remains 0.1

            # --- Generate Reasoning ---
            action_name = UnifiedTradingAction.get_action_name(decision_type)
            reasoning = f"QStar suggests {action_name} based on state {state} with confidence {confidence:.2f}"

            return TradingDecision(
                decision_type=decision_type,
                confidence=confidence, # Already float
                reasoning=reasoning,
                timestamp=datetime.now()
            )

        except AttributeError as e_attr:
            # Catch attribute errors on the agent itself
             self.logger.error(f"AttributeError in QStar decision (likely agent issue): {e_attr}", exc_info=True)
             default_decision.reasoning = f"QStar decision failed: Missing attribute '{e_attr}'"
             return default_decision
        except Exception as e:
            self.logger.error(f"Unexpected error in QStar decision: {e}", exc_info=True)
            default_decision.reasoning = f"QStar decision unexpected error: {str(e)}"
            return default_decision


    def _get_quantum_circuit_decision(self, vector_features: np.ndarray) -> TradingDecision:
        """
        Get trading decision from quantum circuits.
        
        Args:
            vector_features: Feature vector
            
        Returns:
            Trading decision
        """
        if self.quantum_circuits is None or not QUANTUM_AVAILABLE:
            # Default to HOLD with low confidence
            return TradingDecision(
                decision_type=UnifiedTradingAction.HOLD,
                confidence=0.3,
                reasoning="Quantum circuits not available",
                timestamp=datetime.now()
            )
        
        try:
            # Get decision from quantum circuit
            action, confidence = self.quantum_circuits.get_decision(vector_features)
            
            # Generate reasoning
            action_name = UnifiedTradingAction.get_action_name(action)
            reasoning = f"Quantum circuit recommends {action_name} with confidence {confidence:.2f}"
            
            return TradingDecision(
                decision_type=action,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in quantum circuit decision: {e}")
            return TradingDecision(
                decision_type=UnifiedTradingAction.HOLD,
                confidence=0.1,
                reasoning=f"Quantum circuit error: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _blend_decisions(self, 
                         qstar_decision: TradingDecision,
                         qar_decision: TradingDecision,
                         quantum_decision: TradingDecision,
                         anomaly_detected: bool,
                         drift_detected: bool,
                         market_phase: MarketPhase) -> TradingDecision:
        """
        Blend decisions from multiple subsystems using LMSR.
        
        Args:
            qstar_decision: Decision from QStar
            qar_decision: Decision from QAR
            quantum_decision: Decision from quantum circuits
            anomaly_detected: Whether anomaly was detected
            drift_detected: Whether drift was detected
            market_phase: Current market phase
            
        Returns:
            Blended decision
        """
        # Default weights
        weights = {
            'qstar': self.config.qstar_weight,
            'qar': self.config.qar_weight,
            'quantum': 1.0 - self.config.qstar_weight - self.config.qar_weight
        }
        
        # Adjust weights based on market conditions
        if anomaly_detected:
            # Increase QAR weight during anomalies
            weights['qar'] += 0.2
            weights['qstar'] -= 0.1
            weights['quantum'] -= 0.1
        
        if drift_detected:
            # Increase QStar weight during drift (more adaptive)
            weights['qstar'] += 0.2
            weights['qar'] -= 0.1
            weights['quantum'] -= 0.1
        
        # Adjust based on market phase
        if market_phase == MarketPhase.GROWTH:
            # In growth phase, increase quantum weight
            weights['quantum'] += 0.1
            weights['qstar'] -= 0.05
            weights['qar'] -= 0.05
        elif market_phase == MarketPhase.CONSERVATION:
            # In conservation phase, increase QAR weight
            weights['qar'] += 0.1
            weights['qstar'] -= 0.05
            weights['quantum'] -= 0.05
        elif market_phase == MarketPhase.RELEASE:
            # In release phase, increase QStar weight
            weights['qstar'] += 0.1
            weights['qar'] -= 0.05
            weights['quantum'] -= 0.05
        elif market_phase == MarketPhase.REORGANIZATION:
            # In reorganization phase, balance weights
            avg_weight = sum(weights.values()) / 3
            weights = {k: avg_weight for k in weights}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get decision probabilities for each action
        action_probs = {action: 0.0 for action in range(UnifiedTradingAction.get_num_actions())}
        
        # Add QStar probabilities
        qstar_action = qstar_decision.decision_type
        qstar_confidence = qstar_decision.confidence
        action_probs[qstar_action] += weights['qstar'] * qstar_confidence
        
        # Add QAR probabilities
        qar_action = qar_decision.decision_type
        qar_confidence = qar_decision.confidence
        action_probs[qar_action] += weights['qar'] * qar_confidence
        
        # Add quantum probabilities
        quantum_action = quantum_decision.decision_type
        quantum_confidence = quantum_decision.confidence
        action_probs[quantum_action] += weights['quantum'] * quantum_confidence
        
        # Find best action
        best_action = max(action_probs.items(), key=lambda x: x[1])
        action = best_action[0]
        raw_confidence = best_action[1]
        
        # Calculate overall confidence
        # Scale by the number of systems that agree
        systems_agreement = len(set([qstar_action, qar_action, quantum_action]))
        if systems_agreement == 1:
            # All systems agree
            confidence = raw_confidence * 1.2  # Boost confidence
        elif systems_agreement == 2:
            # Two systems agree
            confidence = raw_confidence * 1.0
        else:
            # All systems disagree
            confidence = raw_confidence * 0.8  # Reduce confidence
        
        # Cap confidence
        confidence = min(0.95, max(0.05, confidence))
        
        # Special case: HOLD with high confidence if anomaly is extreme
        if anomaly_detected and self.config.anomaly_threshold > 0.9:
            action = UnifiedTradingAction.HOLD
            confidence = 0.9
        
        # Generate reasoning
        action_name = UnifiedTradingAction.get_action_name(action)
        reasoning = (
            f"QUASAR recommends {action_name} with confidence {confidence:.2f} based on: "
            f"QStar ({weights['qstar']:.2f}): {UnifiedTradingAction.get_action_name(qstar_action)}, "
            f"QAR ({weights['qar']:.2f}): {UnifiedTradingAction.get_action_name(qar_action)}, "
            f"Quantum ({weights['quantum']:.2f}): {UnifiedTradingAction.get_action_name(quantum_action)}. "
        )
        
        if anomaly_detected:
            reasoning += f"Anomaly detected with score {self.config.anomaly_threshold:.2f}. "
        
        if drift_detected:
            reasoning += f"Drift detected in market conditions. "
        
        reasoning += f"Current market phase: {market_phase.name}."
        
        return TradingDecision(
            decision_type=action,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
    
    def train(self, historical_data: pd.DataFrame, epochs: int = 5) -> Dict[str, Any]:
        """
        Train the QUASAR system with historical data.
        
        Args:
            historical_data: Historical market data
            epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        if not self.state.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            self.logger.info(f"Training QUASAR on {len(historical_data)} data points for {epochs} epochs")
            
            # Create training metrics
            metrics = {}
            
            # Train QStar if available
            if self.state.qstar_ready:
                self.logger.info("Training QStar component...")
                
                # Create trading environment
                from qstar_learning import TradingEnvironment
                
                env = TradingEnvironment(
                    price_data=historical_data,
                    price_column='close',
                    window_size=50
                )
                
                # Train QStar agent
                converged, q_episodes = self.qstar_agent.train(env)
                
                # Evaluate agent
                qstar_eval = self.qstar_agent.evaluate(env)
                
                # Add to metrics
                metrics['qstar'] = {
                    'converged': converged,
                    'episodes': q_episodes,
                    'evaluation': qstar_eval
                }
                
                self.logger.info(f"QStar training completed: {converged}, episodes: {q_episodes}")
            
            # Train QAR regime-specific weights
            if self.state.qar_ready:
                self.logger.info("Training QAR component...")
                
                # Configure regime-specific weights
                regime_weights = {}
                
                # Growth phase - emphasize trend and momentum
                regime_weights[MarketPhase.GROWTH] = {
                    'trend': 0.3,
                    'momentum': 0.3,
                    'volatility': 0.1,
                    'rsi_14': 0.1,
                    'adx': 0.2
                }
                
                # Conservation phase - emphasize volatility and overbought
                regime_weights[MarketPhase.CONSERVATION] = {
                    'trend': 0.1,
                    'momentum': 0.1,
                    'volatility': 0.3,
                    'rsi_14': 0.3,
                    'adx': 0.2
                }
                
                # Release phase - emphasize volatility and momentum
                regime_weights[MarketPhase.RELEASE] = {
                    'trend': 0.2,
                    'momentum': 0.3,
                    'volatility': 0.3,
                    'rsi_14': 0.1,
                    'adx': 0.1
                }
                
                # Reorganization phase - balance factors
                regime_weights[MarketPhase.REORGANIZATION] = {
                    'trend': 0.2,
                    'momentum': 0.2,
                    'volatility': 0.2,
                    'rsi_14': 0.2,
                    'adx': 0.2
                }
                
                self.qar.configure_regime_weights(regime_weights)
                
                # Add to metrics
                metrics['qar'] = {
                    'regime_weights_configured': True
                }
                
                self.logger.info("QAR regime weights configured")
            
            # Update quantum phases
            if self.quantum_circuits is not None:
                self.quantum_circuits.update_phases()
            
            # Mark as trained
            self.state.is_trained = True
            
            return {
                'success': True,
                'epochs': epochs,
                'data_points': len(historical_data),
                'metrics': metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error in training: {e}")
            return {'error': str(e)}
    
    def provide_feedback(self, 
                        decision_id: str, 
                        outcome: str, 
                        profit_loss: float) -> bool:
        """
        Provide feedback on a previous decision.
        
        Args:
            decision_id: ID of the decision
            outcome: 'success' or 'failure'
            profit_loss: Realized profit/loss
            
        Returns:
            Success status
        """
        if not self.state.is_initialized:
            return False
        
        try:
            # Provide feedback to QAR
            if self.state.qar_ready:
                self.qar.provide_feedback(decision_id, outcome, profit_loss)
            
            # TODO: Implement feedback for QStar
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")
            return False
    
    def save_state(self, filepath: str) -> bool:
        """
        Save QUASAR state to file.
        
        Args:
            filepath: Path to save state
            
        Returns:
            Success status
        """
        if not self.state.is_initialized:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            # Compile state dictionary
            state_dict = {
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'is_initialized': self.state.is_initialized,
                'is_trained': self.state.is_trained,
                'config': vars(self.config)
            }
            
            # Save QStar state if available
            if self.state.qstar_ready:
                qstar_path = f"{filepath}_qstar.pkl"
                self.qstar_agent.save(qstar_path)
                state_dict['qstar_path'] = qstar_path
            
            # Save QAR state if available
            if self.state.qar_ready:
                qar_path = f"{filepath}_qar.json"
                self.qar.save_state(qar_path)
                state_dict['qar_path'] = qar_path
            
            # Save combined state
            with open(filepath, 'wb') as f:
                pickle.dump(state_dict, f)
            
            self.logger.info(f"QUASAR state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
            return False
    
    @classmethod
    def load_state(cls, filepath: str) -> 'QUASAR':
        """
        Load QUASAR state from file.
        
        Args:
            filepath: Path to load state from
            
        Returns:
            QUASAR instance
        """
        try:
            # Load state dictionary
            with open(filepath, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Create config from saved settings
            config = QUASARConfig()
            for key, value in state_dict.get('config', {}).items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            # Create QUASAR instance
            quasar = cls(config=config)
            
            # Load QStar state if available
            qstar_path = state_dict.get('qstar_path')
            if qstar_path and os.path.exists(qstar_path):
                quasar.qstar_agent = QStarLearningAgent.load(qstar_path)
                quasar.state.qstar_ready = True
            
            # Load QAR state if available
            qar_path = state_dict.get('qar_path')
            if qar_path and os.path.exists(qar_path):
                quasar.qar.load_state(qar_path)
                quasar.state.qar_ready = True
            
            # Set state flags
            quasar.state.is_initialized = state_dict.get('is_initialized', False)
            quasar.state.is_trained = state_dict.get('is_trained', False)
            
            return quasar
            
        except Exception as e:
            logger.error(f"Error loading QUASAR state: {e}")
            raise


# Integration function for PADS
def initialize_quasar_for_pads(
    pads_config: Optional[Dict[str, Any]] = None,
    data_directory: Optional[str] = None,
    load_existing: bool = True
) -> QUASAR:
    """
    Initialize QUASAR system for integration with PADS.
    
    Args:
        pads_config: PADS configuration settings
        data_directory: Directory for model data
        load_existing: Whether to load existing model if available
        
    Returns:
        Initialized QUASAR instance
    """
    # Create data directory if specified and doesn't exist
    if data_directory:
        os.makedirs(data_directory, exist_ok=True)
    
    # Default model path
    model_path = os.path.join(data_directory, "quasar_model.pkl") if data_directory else "quasar_model.pkl"
    
    # Try to load existing model
    if load_existing and os.path.exists(model_path):
        try:
            logger.info(f"Loading existing QUASAR model from {model_path}")
            quasar = QUASAR.load_state(model_path)
            logger.info("QUASAR model loaded successfully")
            return quasar
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}")
    
    # Create new model
    logger.info("Creating new QUASAR model")
    
    # Configure from PADS settings if available
    config = QUASARConfig()
    
    if pads_config:
        # Map PADS settings to QUASAR configuration
        if 'quantum_enabled' in pads_config:
            config.quantum_enabled = pads_config['quantum_enabled']
        
        if 'decision_threshold' in pads_config:
            config.decision_threshold = pads_config['decision_threshold']
        
        if 'log_level' in pads_config:
            config.log_level = pads_config['log_level']
    
    # Create QUASAR instance
    quasar = QUASAR(config=config)
    
    # Register standard indicators
    if quasar.state.qar_ready:
        quasar.register_indicator('trend', 0.25)
        quasar.register_indicator('momentum', 0.25)
        quasar.register_indicator('volatility', 0.2)
        quasar.register_indicator('rsi_14', 0.15)
        quasar.register_indicator('adx', 0.15)
    
    return quasar


# Example usage
if __name__ == "__main__":
    # Create QUASAR instance
    quasar = QUASAR()
    
    # Generate sample data
    import numpy as np
    
    # Generate sample price data
    dates = pd.date_range(start='2023-01-01', periods=1000)
    price = 100
    prices = []
    
    for _ in range(1000):
        change = np.random.normal(0, 1) / 100
        price *= (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame({
        'open': prices,
        'high': np.array(prices) * (1 + np.random.uniform(0, 0.02, 1000)),
        'low': np.array(prices) * (1 - np.random.uniform(0, 0.02, 1000)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000) * 1000
    }, index=dates)
    
    # Add some technical indicators
    df['rsi_14'] = np.random.uniform(30, 70, 1000)
    df['adx'] = np.random.uniform(15, 40, 1000)
    
    # Train the system
    quasar.train(df)
    
    # Make a trading decision
    decision = quasar.make_decision(df)
    
    # Print decision
    print("Available keys:", decision.keys())
    print(f"QUASAR decision: {decision['action_name']} with confidence {decision['confidence']:.2f}")
    explanation = decision.get('explanation', decision.get('reasoning', 'Not available'))
    print(f"Explanation: {explanation}")
    print(f"Explanation: {decision.get('explanation', 'Not available')}")
    # Save model
    quasar.save_state("quasar_model.pkl")
