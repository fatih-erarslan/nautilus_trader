#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STDP Weight Optimizer for CDFA Extensions

Provides biologically-inspired weight optimization using Spike-Timing-Dependent Plasticity:
- Dynamic weight updates based on temporal correlations between signals
- Adaptive learning based on market regimes
- Unsupervised feature discovery
- Automatic pruning for sparse representations
- Homeostatic plasticity for system stability

Author: Created on May 6, 2025
"""

import logging
import time
import numpy as np
import pandas as pd
import threading
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import warnings
import queue
import os
from datetime import datetime, timedelta
import uuid
import math
from collections import defaultdict

# Import from cdfa_extensions
from .hw_acceleration import HardwareAccelerator

# ---- Optional dependencies with graceful fallbacks ----

# PyTorch for tensor operations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Tensor operations will be limited.", DeprecationWarning, DeprecationWarning)

# Numba for JIT acceleration
try:
    import numba as nb
    from numba import njit, prange, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. JIT acceleration will be limited.", DeprecationWarning, DeprecationWarning)
    
    # Define dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
        
    prange = range
    float64 = int64 = lambda x: x

class STDPMode(Enum):
    """Modes for STDP learning."""
    STANDARD = auto()      # Standard STDP (LTP/LTD)
    REWARD = auto()        # Reward-modulated STDP
    DOPAMINE = auto()      # Dopamine-like reward signal
    TRIPLET = auto()       # Triplet-based STDP
    ASYMMETRIC = auto()    # Asymmetric STDP (different A+/A-)
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'STDPMode':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown STDPMode: {s}")

class HomeostaticMode(Enum):
    """Modes for homeostatic plasticity."""
    NONE = auto()          # No homeostatic plasticity
    SCALING = auto()       # Synaptic scaling
    THRESHOLD = auto()     # Threshold adjustment
    METAPLASTICITY = auto() # Metaplasticity
    
    def __str__(self) -> str:
        return self.name.lower()
    
    @classmethod
    def from_string(cls, s: str) -> 'HomeostaticMode':
        """Create enum from string representation."""
        s_upper = s.upper()
        for item in cls:
            if item.name == s_upper:
                return item
        for item in cls:
            if item.name.startswith(s_upper):
                return item
        raise ValueError(f"Unknown HomeostaticMode: {s}")

@dataclass
class STDPParameters:
    """Parameters for STDP learning."""
    a_plus: float = 0.01      # LTP amplitude
    a_minus: float = 0.01     # LTD amplitude
    tau_plus: float = 20.0    # LTP time constant (ms)
    tau_minus: float = 20.0   # LTD time constant (ms)
    w_min: float = 0.0        # Minimum weight
    w_max: float = 1.0        # Maximum weight
    nearest_neighbor: bool = True  # Nearest neighbor or all-to-all
    mode: STDPMode = STDPMode.STANDARD  # STDP mode
    learning_rate: float = 0.001  # Learning rate
    reward_factor: float = 1.0  # Reward modulation factor
    homeostatic_mode: HomeostaticMode = HomeostaticMode.SCALING  # Homeostatic plasticity mode
    target_rate: float = 0.1  # Target firing rate (for homeostasis)
    homeostatic_rate: float = 0.001  # Rate of homeostatic adjustment
    pruning_threshold: float = 0.01  # Weight threshold for pruning
    initial_weight_sigma: float = 0.1  # Std dev for initial weights
    weight_decay: float = 0.0  # Weight decay factor
    conduction_delay: float = 1.0  # Conduction delay (ms)

@dataclass
class STDPResult:
    """Result of STDP learning."""
    weights: np.ndarray      # Updated weights
    weight_changes: np.ndarray  # Weight changes
    pre_activity: np.ndarray  # Presynaptic activity
    post_activity: np.ndarray  # Postsynaptic activity
    pruned_synapses: int     # Number of pruned synapses
    total_update: float      # Total weight update magnitude
    correlation_matrix: Optional[np.ndarray] = None  # Correlation matrix
    metadata: Dict[str, Any] = field(default_factory=dict)

class STDPOptimizer:
    """
    Weight optimizer using biologically-inspired Spike-Timing-Dependent Plasticity.
    
    Provides adaptive weight optimization for the CDFA fusion system using principles
    from neuroplasticity, including:
    - STDP learning for temporal correlation detection
    - Homeostatic plasticity for system stability
    - Reward-modulated learning for adaptive responses
    - Synaptic pruning for sparse representations
    """
    
    def __init__(self, hw_accelerator: Optional[HardwareAccelerator] = None,
                config: Optional[Dict[str, Any]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the STDP optimizer.
        
        Args:
            hw_accelerator: Optional hardware accelerator
            config: Configuration parameters
            log_level: Logging level
        """
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Initialize hardware accelerator
        self.hw_accelerator = hw_accelerator if hw_accelerator is not None else HardwareAccelerator()
        
        # Default configuration
        self.default_config = {
            # STDP parameters
            "default_a_plus": 0.01,
            "default_a_minus": 0.01,
            "default_tau_plus": 20.0,   # ms
            "default_tau_minus": 20.0,  # ms
            "default_w_min": 0.0,
            "default_w_max": 1.0,
            "default_nearest_neighbor": True,
            "default_stdp_mode": "standard",
            "default_learning_rate": 0.001,
            
            # Reward modulation
            "default_reward_factor": 1.0,
            "use_adaptive_reward": True,
            "reward_smoothing": 0.95,   # Exponential smoothing factor for rewards
            "reward_baseline": 0.0,     # Baseline reward
            
            # Homeostatic plasticity
            "default_homeostatic_mode": "scaling",
            "default_target_rate": 0.1,  # Target firing rate
            "default_homeostatic_rate": 0.001,
            "use_adaptive_homeostasis": True,
            
            # Weight initialization
            "default_initial_weight_mean": 0.5,
            "default_initial_weight_sigma": 0.1,
            
            # Structural plasticity
            "default_pruning_threshold": 0.01,
            "use_structural_plasticity": True,
            "rewiring_probability": 0.001,  # Probability of creating new synapse
            "consolidation_threshold": 0.9,  # Threshold for consolidating synapses
            
            # Weight decay
            "default_weight_decay": 0.0,
            "use_adaptive_decay": True,
            
            # Time parameters
            "default_time_step": 1.0,  # ms
            "default_conduction_delay": 1.0,  # ms
            
            # Performance parameters
            "use_numba": True,
            "use_torch": True,
            "parallel_threshold": 1000,  # Threshold for parallel processing
            "cache_results": True,
            "cache_ttl": 3600,  # 1 hour
            
            # STDP specialization
            "use_regime_specific_stdp": True,
            "regime_stdp_params": {
                "growth": {
                    "a_plus": 0.02,
                    "a_minus": 0.01,
                    "tau_plus": 25.0,
                    "tau_minus": 20.0
                },
                "conservation": {
                    "a_plus": 0.01,
                    "a_minus": 0.01,
                    "tau_plus": 20.0,
                    "tau_minus": 20.0
                },
                "release": {
                    "a_plus": 0.005,
                    "a_minus": 0.02,
                    "tau_plus": 15.0,
                    "tau_minus": 25.0
                },
                "reorganization": {
                    "a_plus": 0.015,
                    "a_minus": 0.015,
                    "tau_plus": 15.0,
                    "tau_minus": 15.0
                }
            }
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize state
        self._lock = threading.RLock()
        self._weight_cache = {}  # (source, target) -> (weights, timestamp)
        self._learning_state = {}  # (source, target) -> state
        self._reward_history = {}  # (source, target) -> [rewards]
        
        # Check available backends
        self.has_torch = TORCH_AVAILABLE
        self.has_numba = NUMBA_AVAILABLE
        
        # Initialize default STDP parameters
        self.default_params = STDPParameters(
            a_plus=self.config["default_a_plus"],
            a_minus=self.config["default_a_minus"],
            tau_plus=self.config["default_tau_plus"],
            tau_minus=self.config["default_tau_minus"],
            w_min=self.config["default_w_min"],
            w_max=self.config["default_w_max"],
            nearest_neighbor=self.config["default_nearest_neighbor"],
            mode=STDPMode.from_string(self.config["default_stdp_mode"]),
            learning_rate=self.config["default_learning_rate"],
            reward_factor=self.config["default_reward_factor"],
            homeostatic_mode=HomeostaticMode.from_string(self.config["default_homeostatic_mode"]),
            target_rate=self.config["default_target_rate"],
            homeostatic_rate=self.config["default_homeostatic_rate"],
            pruning_threshold=self.config["default_pruning_threshold"],
            initial_weight_sigma=self.config["default_initial_weight_sigma"],
            weight_decay=self.config["default_weight_decay"],
            conduction_delay=self.config["default_conduction_delay"]
        )
        
        self.logger.info("STDPOptimizer initialized")
    
    def _get_cached_weights(self, source: str, target: str) -> Optional[np.ndarray]:
        """
        Get cached weights if valid.
        
        Args:
            source: Source identifier
            target: Target identifier
            
        Returns:
            Cached weights or None if not found or expired
        """
        if not self.config["cache_results"]:
            return None
            
        with self._lock:
            # Check if weights are in cache
            cache_key = (source, target)
            cache_entry = self._weight_cache.get(cache_key)
            
            if cache_entry is None:
                return None
                
            weights, timestamp = cache_entry
            
            # Check if expired
            current_time = time.time()
            if current_time - timestamp > self.config["cache_ttl"]:
                # Remove from cache
                self._weight_cache.pop(cache_key, None)
                return None
                
            return weights.copy()  # Return copy to prevent modification
            
    def _cache_weights(self, source: str, target: str, weights: np.ndarray):
        """
        Cache weights for future use.
        
        Args:
            source: Source identifier
            target: Target identifier
            weights: Weights matrix
        """
        if not self.config["cache_results"]:
            return
            
        with self._lock:
            cache_key = (source, target)
            self._weight_cache[cache_key] = (weights.copy(), time.time())
            
    def _get_learning_state(self, source: str, target: str) -> Dict[str, Any]:
        """
        Get learning state for a connection.
        
        Args:
            source: Source identifier
            target: Target identifier
            
        Returns:
            Learning state dictionary
        """
        with self._lock:
            cache_key = (source, target)
            state = self._learning_state.get(cache_key)
            
            if state is None:
                # Initialize new learning state
                state = {
                    "pre_trace": None,
                    "post_trace": None,
                    "pre_activity": None,
                    "post_activity": None,
                    "reward_history": [],
                    "reward_baseline": self.config["reward_baseline"],
                    "iteration": 0,
                    "total_updates": 0.0,
                    "timestamp": time.time()
                }
                self._learning_state[cache_key] = state
                
            return state
            
    def _update_learning_state(self, source: str, target: str, updates: Dict[str, Any]):
        """
        Update learning state for a connection.
        
        Args:
            source: Source identifier
            target: Target identifier
            updates: Dictionary of state updates
        """
        with self._lock:
            cache_key = (source, target)
            state = self._learning_state.get(cache_key)
            
            if state is None:
                state = self._get_learning_state(source, target)
                
            # Update state
            state.update(updates)
            state["timestamp"] = time.time()
            
            self._learning_state[cache_key] = state
    
    def initialize_weights(self, n_pre: int, n_post: int, 
                         params: Optional[STDPParameters] = None) -> np.ndarray:
        """
        Initialize weight matrix for STDP learning.
        
        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            params: STDP parameters (or use defaults)
            
        Returns:
            Initialized weight matrix
        """
        if params is None:
            params = self.default_params
            
        # Initialize with random normal distribution
        mean = self.config["default_initial_weight_mean"]
        sigma = params.initial_weight_sigma
        
        weights = np.random.normal(mean, sigma, (n_pre, n_post))
        
        # Clip to weight range
        weights = np.clip(weights, params.w_min, params.w_max)
        
        return weights
    
    def apply_stdp(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                 weights: np.ndarray, params: Optional[STDPParameters] = None,
                 reward: Optional[float] = None, source: str = "", target: str = "") -> STDPResult:
        """
        Apply STDP learning rule to update weights.
        
        Args:
            pre_spikes: Presynaptic spike trains [batch, time, n_pre] or [time, n_pre]
            post_spikes: Postsynaptic spike trains [batch, time, n_post] or [time, n_post]
            weights: Weight matrix [n_pre, n_post]
            params: STDP parameters (or use defaults)
            reward: Optional reward signal for reward-modulated STDP
            source: Source identifier for state tracking
            target: Target identifier for state tracking
            
        Returns:
            STDP learning result
        """
        # Use default parameters if not provided
        if params is None:
            params = self.default_params
            
        # Choose implementation based on hardware and data size
        if self.has_torch and self.config["use_torch"] and pre_spikes.size > self.config["parallel_threshold"]:
            return self._apply_stdp_torch(pre_spikes, post_spikes, weights, params, reward, source, target)
        elif self.has_numba and self.config["use_numba"]:
            return self._apply_stdp_numba(pre_spikes, post_spikes, weights, params, reward, source, target)
        else:
            return self._apply_stdp_numpy(pre_spikes, post_spikes, weights, params, reward, source, target)
    
    def _apply_stdp_numpy(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                       weights: np.ndarray, params: STDPParameters,
                       reward: Optional[float], source: str, target: str) -> STDPResult:
        """
        Apply STDP learning rule using NumPy.
        
        Args:
            pre_spikes: Presynaptic spike trains
            post_spikes: Postsynaptic spike trains
            weights: Weight matrix
            params: STDP parameters
            reward: Optional reward signal
            source: Source identifier
            target: Target identifier
            
        Returns:
            STDP learning result
        """
        # Handle batch dimension if present
        if len(pre_spikes.shape) == 3:
            # Batch data - average across batch dimension for simplicity
            pre_spikes = np.mean(pre_spikes, axis=0)
            post_spikes = np.mean(post_spikes, axis=0)
            
        # Get dimensions
        n_timesteps = pre_spikes.shape[0]
        n_pre = pre_spikes.shape[1]
        n_post = post_spikes.shape[1]
        
        # Ensure weights match dimensions
        if weights.shape != (n_pre, n_post):
            raise ValueError(f"Weight matrix shape {weights.shape} doesn't match spike train dimensions: {n_pre} x {n_post}")
            
        # Get learning state
        learning_state = self._get_learning_state(source, target)
        
        # Initialize traces if needed
        if learning_state["pre_trace"] is None or learning_state["pre_trace"].shape != (n_pre,):
            learning_state["pre_trace"] = np.zeros(n_pre)
            
        if learning_state["post_trace"] is None or learning_state["post_trace"].shape != (n_post,):
            learning_state["post_trace"] = np.zeros(n_post)
            
        # Get current trace values
        pre_trace = learning_state["pre_trace"]
        post_trace = learning_state["post_trace"]
        
        # Initialize weight changes
        weight_changes = np.zeros_like(weights)
        
        # Time constants
        tau_plus = params.tau_plus
        tau_minus = params.tau_minus
        
        # Learning rate and amplitudes
        lr = params.learning_rate
        a_plus = params.a_plus
        a_minus = params.a_minus
        
        # Process reward modulation
        if params.mode in (STDPMode.REWARD, STDPMode.DOPAMINE) and reward is not None:
            # Update reward history
            learning_state["reward_history"].append(reward)
            if len(learning_state["reward_history"]) > 100:
                learning_state["reward_history"] = learning_state["reward_history"][-100:]
                
            # Calculate reward baseline (running average)
            alpha = self.config["reward_smoothing"]
            baseline = learning_state["reward_baseline"]
            baseline = alpha * baseline + (1 - alpha) * reward
            learning_state["reward_baseline"] = baseline
            
            # Calculate modulated reward
            if params.mode == STDPMode.DOPAMINE:
                # Dopamine-like modulation (reward prediction error)
                reward_mod = reward - baseline
            else:
                # Standard reward modulation
                reward_mod = reward
                
            # Scale reward
            reward_factor = params.reward_factor
        else:
            # No reward modulation
            reward_mod = 1.0
            reward_factor = 1.0
            
        # Calculate pre and post activity (average firing rate)
        pre_activity = np.mean(pre_spikes, axis=0)
        post_activity = np.mean(post_spikes, axis=0)
        
        # Store activity for homeostatic plasticity
        learning_state["pre_activity"] = pre_activity
        learning_state["post_activity"] = post_activity
        
        # Process each time step
        for t in range(n_timesteps):
            # Get spikes at this time step
            pre_spike = pre_spikes[t]
            post_spike = post_spikes[t]
            
            # Update traces
            # Pre-synaptic trace (decays and gets incremented by new spikes)
            pre_trace = pre_trace * np.exp(-1.0 / tau_plus)
            pre_trace += pre_spike
            
            # Post-synaptic trace (decays and gets incremented by new spikes)
            post_trace = post_trace * np.exp(-1.0 / tau_minus)
            post_trace += post_spike
            
            # Calculate weight updates for LTP (pre -> post) and LTD (post -> pre)
            for i in range(n_pre):
                for j in range(n_post):
                    # LTP: pre spike followed by post spike
                    if post_spike[j] > 0:
                        # Pre-trace indicates how recently pre-neuron spiked
                        weight_changes[i, j] += lr * reward_mod * reward_factor * a_plus * pre_trace[i]
                        
                    # LTD: post spike followed by pre spike
                    if pre_spike[i] > 0:
                        # Post-trace indicates how recently post-neuron spiked
                        weight_changes[i, j] -= lr * reward_mod * reward_factor * a_minus * post_trace[j]
                        
        # Apply weight decay if enabled
        if params.weight_decay > 0:
            weight_decay = params.weight_decay
            weight_changes -= weight_decay * weights
            
        # Apply homeostatic plasticity if enabled
        if params.homeostatic_mode != HomeostaticMode.NONE:
            weight_changes = self._apply_homeostasis(
                weight_changes, weights, pre_activity, post_activity, params
            )
            
        # Store updated traces in learning state
        learning_state["pre_trace"] = pre_trace
        learning_state["post_trace"] = post_trace
        learning_state["iteration"] += 1
        
        # Update total weight changes
        total_update = np.sum(np.abs(weight_changes))
        learning_state["total_updates"] += total_update
        
        # Apply weight changes
        new_weights = weights + weight_changes
        
        # Clip weights to allowed range
        new_weights = np.clip(new_weights, params.w_min, params.w_max)
        
        # Apply pruning if enabled
        pruned_synapses = 0
        if self.config["use_structural_plasticity"] and params.pruning_threshold > 0:
            # Identify weak synapses
            weak_mask = new_weights < params.pruning_threshold
            
            # Count pruned synapses
            pruned_synapses = np.sum(weak_mask)
            
            # Prune weak synapses
            new_weights[weak_mask] = 0.0
            
            # Apply rewiring (create new random synapses)
            if self.config["rewiring_probability"] > 0:
                rewire_prob = self.config["rewiring_probability"]
                
                # Generate random mask for creating new synapses
                rewire_mask = np.random.rand(*new_weights.shape) < rewire_prob
                
                # Only rewire where weights are currently zero
                rewire_mask = rewire_mask & (new_weights == 0)
                
                # Initialize new synapses with small weights
                new_synapse_weight = params.w_max * 0.1
                new_weights[rewire_mask] = new_synapse_weight
                
        # Update learning state
        self._update_learning_state(source, target, learning_state)
        
        # Cache updated weights
        self._cache_weights(source, target, new_weights)
        
        # Create result
        result = STDPResult(
            weights=new_weights,
            weight_changes=weight_changes,
            pre_activity=pre_activity,
            post_activity=post_activity,
            pruned_synapses=pruned_synapses,
            total_update=total_update,
            metadata={
                "iteration": learning_state["iteration"],
                "reward": reward,
                "reward_modulation": reward_mod if "reward_mod" in locals() else 1.0,
                "reward_baseline": learning_state["reward_baseline"],
                "timestamp": time.time()
            }
        )
        
        return result
    
    @staticmethod
    @njit(cache=True)
    def _stdp_numba_core(pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                      weights: np.ndarray, weight_changes: np.ndarray,
                      pre_trace: np.ndarray, post_trace: np.ndarray,
                      lr: float, a_plus: float, a_minus: float,
                      tau_plus: float, tau_minus: float,
                      reward_mod: float, reward_factor: float,
                      nearest_neighbor: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core STDP update algorithm accelerated with Numba.
        
        Args:
            pre_spikes: Presynaptic spike trains [time, n_pre]
            post_spikes: Postsynaptic spike trains [time, n_post]
            weights: Weight matrix [n_pre, n_post]
            weight_changes: Weight changes matrix [n_pre, n_post]
            pre_trace: Presynaptic trace [n_pre]
            post_trace: Postsynaptic trace [n_post]
            lr: Learning rate
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            tau_plus: LTP time constant
            tau_minus: LTD time constant
            reward_mod: Reward modulation
            reward_factor: Reward factor
            nearest_neighbor: Whether to use nearest neighbor STDP
            
        Returns:
            Updated weight_changes, pre_trace, post_trace
        """
        n_timesteps = pre_spikes.shape[0]
        n_pre = pre_spikes.shape[1]
        n_post = post_spikes.shape[1]
        
        # Process each time step
        for t in range(n_timesteps):
            # Get spikes at this time step
            pre_spike = pre_spikes[t]
            post_spike = post_spikes[t]
            
            # Update traces
            # Pre-synaptic trace (decays and gets incremented by new spikes)
            for i in range(n_pre):
                pre_trace[i] = pre_trace[i] * np.exp(-1.0 / tau_plus)
                if nearest_neighbor and pre_spike[i] > 0:
                    # Reset trace for nearest neighbor mode
                    pre_trace[i] = 0.0
                pre_trace[i] += pre_spike[i]
                
            # Post-synaptic trace (decays and gets incremented by new spikes)
            for j in range(n_post):
                post_trace[j] = post_trace[j] * np.exp(-1.0 / tau_minus)
                if nearest_neighbor and post_spike[j] > 0:
                    # Reset trace for nearest neighbor mode
                    post_trace[j] = 0.0
                post_trace[j] += post_spike[j]
                
            # Calculate weight updates for LTP (pre -> post) and LTD (post -> pre)
            for i in range(n_pre):
                for j in range(n_post):
                    # LTP: pre spike followed by post spike
                    if post_spike[j] > 0:
                        # Pre-trace indicates how recently pre-neuron spiked
                        weight_changes[i, j] += lr * reward_mod * reward_factor * a_plus * pre_trace[i]
                        
                    # LTD: post spike followed by pre spike
                    if pre_spike[i] > 0:
                        # Post-trace indicates how recently post-neuron spiked
                        weight_changes[i, j] -= lr * reward_mod * reward_factor * a_minus * post_trace[j]
                        
        return weight_changes, pre_trace, post_trace
    
    def _apply_stdp_numba(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                       weights: np.ndarray, params: STDPParameters,
                       reward: Optional[float], source: str, target: str) -> STDPResult:
        """
        Apply STDP learning rule using Numba acceleration.
        
        Args:
            pre_spikes: Presynaptic spike trains
            post_spikes: Postsynaptic spike trains
            weights: Weight matrix
            params: STDP parameters
            reward: Optional reward signal
            source: Source identifier
            target: Target identifier
            
        Returns:
            STDP learning result
        """
        # Handle batch dimension if present
        if len(pre_spikes.shape) == 3:
            # Batch data - average across batch dimension for simplicity
            pre_spikes = np.mean(pre_spikes, axis=0)
            post_spikes = np.mean(post_spikes, axis=0)
            
        # Get dimensions
        n_timesteps = pre_spikes.shape[0]
        n_pre = pre_spikes.shape[1]
        n_post = post_spikes.shape[1]
        
        # Ensure weights match dimensions
        if weights.shape != (n_pre, n_post):
            raise ValueError(f"Weight matrix shape {weights.shape} doesn't match spike train dimensions: {n_pre} x {n_post}")
            
        # Get learning state
        learning_state = self._get_learning_state(source, target)
        
        # Initialize traces if needed
        if learning_state["pre_trace"] is None or learning_state["pre_trace"].shape != (n_pre,):
            learning_state["pre_trace"] = np.zeros(n_pre)
            
        if learning_state["post_trace"] is None or learning_state["post_trace"].shape != (n_post,):
            learning_state["post_trace"] = np.zeros(n_post)
            
        # Get current trace values
        pre_trace = learning_state["pre_trace"].copy()
        post_trace = learning_state["post_trace"].copy()
        
        # Initialize weight changes
        weight_changes = np.zeros_like(weights)
        
        # Process reward modulation
        if params.mode in (STDPMode.REWARD, STDPMode.DOPAMINE) and reward is not None:
            # Update reward history
            learning_state["reward_history"].append(reward)
            if len(learning_state["reward_history"]) > 100:
                learning_state["reward_history"] = learning_state["reward_history"][-100:]
                
            # Calculate reward baseline (running average)
            alpha = self.config["reward_smoothing"]
            baseline = learning_state["reward_baseline"]
            baseline = alpha * baseline + (1 - alpha) * reward
            learning_state["reward_baseline"] = baseline
            
            # Calculate modulated reward
            if params.mode == STDPMode.DOPAMINE:
                # Dopamine-like modulation (reward prediction error)
                reward_mod = reward - baseline
            else:
                # Standard reward modulation
                reward_mod = reward
                
            # Scale reward
            reward_factor = params.reward_factor
        else:
            # No reward modulation
            reward_mod = 1.0
            reward_factor = 1.0
            
        # Calculate pre and post activity (average firing rate)
        pre_activity = np.mean(pre_spikes, axis=0)
        post_activity = np.mean(post_spikes, axis=0)
        
        # Store activity for homeostatic plasticity
        learning_state["pre_activity"] = pre_activity
        learning_state["post_activity"] = post_activity
        
        # Call Numba optimized core function
        weight_changes, pre_trace, post_trace = self._stdp_numba_core(
            pre_spikes, post_spikes, weights, weight_changes,
            pre_trace, post_trace, params.learning_rate,
            params.a_plus, params.a_minus, params.tau_plus, params.tau_minus,
            reward_mod, reward_factor, params.nearest_neighbor
        )
        
        # Apply weight decay if enabled
        if params.weight_decay > 0:
            weight_decay = params.weight_decay
            weight_changes -= weight_decay * weights
            
        # Apply homeostatic plasticity if enabled
        if params.homeostatic_mode != HomeostaticMode.NONE:
            weight_changes = self._apply_homeostasis(
                weight_changes, weights, pre_activity, post_activity, params
            )
            
        # Store updated traces in learning state
        learning_state["pre_trace"] = pre_trace
        learning_state["post_trace"] = post_trace
        learning_state["iteration"] += 1
        
        # Update total weight changes
        total_update = np.sum(np.abs(weight_changes))
        learning_state["total_updates"] += total_update
        
        # Apply weight changes
        new_weights = weights + weight_changes
        
        # Clip weights to allowed range
        new_weights = np.clip(new_weights, params.w_min, params.w_max)
        
        # Apply pruning if enabled
        pruned_synapses = 0
        if self.config["use_structural_plasticity"] and params.pruning_threshold > 0:
            # Identify weak synapses
            weak_mask = new_weights < params.pruning_threshold
            
            # Count pruned synapses
            pruned_synapses = np.sum(weak_mask)
            
            # Prune weak synapses
            new_weights[weak_mask] = 0.0
            
            # Apply rewiring (create new random synapses)
            if self.config["rewiring_probability"] > 0:
                rewire_prob = self.config["rewiring_probability"]
                
                # Generate random mask for creating new synapses
                rewire_mask = np.random.rand(*new_weights.shape) < rewire_prob
                
                # Only rewire where weights are currently zero
                rewire_mask = rewire_mask & (new_weights == 0)
                
                # Initialize new synapses with small weights
                new_synapse_weight = params.w_max * 0.1
                new_weights[rewire_mask] = new_synapse_weight
                
        # Update learning state
        self._update_learning_state(source, target, learning_state)
        
        # Cache updated weights
        self._cache_weights(source, target, new_weights)
        
        # Create result
        result = STDPResult(
            weights=new_weights,
            weight_changes=weight_changes,
            pre_activity=pre_activity,
            post_activity=post_activity,
            pruned_synapses=pruned_synapses,
            total_update=total_update,
            metadata={
                "iteration": learning_state["iteration"],
                "reward": reward,
                "reward_modulation": reward_mod,
                "reward_baseline": learning_state["reward_baseline"],
                "timestamp": time.time()
            }
        )
        
        return result
    
    def _apply_stdp_torch(self, pre_spikes: np.ndarray, post_spikes: np.ndarray, 
                       weights: np.ndarray, params: STDPParameters,
                       reward: Optional[float], source: str, target: str) -> STDPResult:
        """
        Apply STDP learning rule using PyTorch acceleration.
        
        Args:
            pre_spikes: Presynaptic spike trains
            post_spikes: Postsynaptic spike trains
            weights: Weight matrix
            params: STDP parameters
            reward: Optional reward signal
            source: Source identifier
            target: Target identifier
            
        Returns:
            STDP learning result
        """
        if not self.has_torch:
            return self._apply_stdp_numpy(pre_spikes, post_spikes, weights, params, reward, source, target)
            
        try:
            import torch
            
            # Get device
            device = self.hw_accelerator.get_torch_device()
            
            # Handle batch dimension if present
            if len(pre_spikes.shape) == 3:
                # Batch data
                batch_size = pre_spikes.shape[0]
                batch_processing = True
            else:
                # Add batch dimension
                pre_spikes = np.expand_dims(pre_spikes, 0)
                post_spikes = np.expand_dims(post_spikes, 0)
                batch_size = 1
                batch_processing = False
                
            # Get dimensions
            n_timesteps = pre_spikes.shape[1]
            n_pre = pre_spikes.shape[2]
            n_post = post_spikes.shape[2]
            
            # Ensure weights match dimensions
            if weights.shape != (n_pre, n_post):
                raise ValueError(f"Weight matrix shape {weights.shape} doesn't match spike train dimensions: {n_pre} x {n_post}")
                
            # Get learning state
            learning_state = self._get_learning_state(source, target)
            
            # Initialize traces if needed
            if learning_state["pre_trace"] is None or learning_state["pre_trace"].shape != (n_pre,):
                learning_state["pre_trace"] = np.zeros(n_pre)
                
            if learning_state["post_trace"] is None or learning_state["post_trace"].shape != (n_post,):
                learning_state["post_trace"] = np.zeros(n_post)
                
            # Get current trace values
            pre_trace = torch.tensor(learning_state["pre_trace"], dtype=torch.float32, device=device)
            post_trace = torch.tensor(learning_state["post_trace"], dtype=torch.float32, device=device)
            
            # Convert to PyTorch tensors
            pre_spikes_t = torch.tensor(pre_spikes, dtype=torch.float32, device=device)
            post_spikes_t = torch.tensor(post_spikes, dtype=torch.float32, device=device)
            weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
            
            # Initialize weight changes
            weight_changes_t = torch.zeros_like(weights_t)
            
            # Time constants
            tau_plus = params.tau_plus
            tau_minus = params.tau_minus
            
            # Learning rate and amplitudes
            lr = params.learning_rate
            a_plus = params.a_plus
            a_minus = params.a_minus
            
            # Process reward modulation
            if params.mode in (STDPMode.REWARD, STDPMode.DOPAMINE) and reward is not None:
                # Update reward history
                learning_state["reward_history"].append(reward)
                if len(learning_state["reward_history"]) > 100:
                    learning_state["reward_history"] = learning_state["reward_history"][-100:]
                    
                # Calculate reward baseline (running average)
                alpha = self.config["reward_smoothing"]
                baseline = learning_state["reward_baseline"]
                baseline = alpha * baseline + (1 - alpha) * reward
                learning_state["reward_baseline"] = baseline
                
                # Calculate modulated reward
                if params.mode == STDPMode.DOPAMINE:
                    # Dopamine-like modulation (reward prediction error)
                    reward_mod = reward - baseline
                else:
                    # Standard reward modulation
                    reward_mod = reward
                    
                # Scale reward
                reward_factor = params.reward_factor
            else:
                # No reward modulation
                reward_mod = 1.0
                reward_factor = 1.0
                
            # Calculate pre and post activity (average firing rate)
            pre_activity = torch.mean(pre_spikes_t, dim=1)  # [batch, n_pre]
            post_activity = torch.mean(post_spikes_t, dim=1)  # [batch, n_post]
            
            # Store activity for homeostatic plasticity (average over batch)
            learning_state["pre_activity"] = pre_activity.mean(dim=0).cpu().numpy()
            learning_state["post_activity"] = post_activity.mean(dim=0).cpu().numpy()
            
            # Process each time step
            for t in range(n_timesteps):
                # Get spikes at this time step (for all batches)
                pre_spike = pre_spikes_t[:, t, :]  # [batch, n_pre]
                post_spike = post_spikes_t[:, t, :]  # [batch, n_post]
                
                # Update traces
                # Pre-synaptic trace (decays and gets incremented by new spikes)
                pre_trace = pre_trace * torch.exp(torch.tensor(-1.0 / tau_plus, device=device))
                
                if params.nearest_neighbor:
                    # Reset trace for spikes in nearest neighbor mode
                    pre_trace = pre_trace * (1.0 - pre_spike[0])
                    
                pre_trace = pre_trace + pre_spike[0]
                
                # Post-synaptic trace (decays and gets incremented by new spikes)
                post_trace = post_trace * torch.exp(torch.tensor(-1.0 / tau_minus, device=device))
                
                if params.nearest_neighbor:
                    # Reset trace for spikes in nearest neighbor mode
                    post_trace = post_trace * (1.0 - post_spike[0])
                    
                post_trace = post_trace + post_spike[0]
                
                # For batch processing, we'll calculate updates for each example
                for b in range(batch_size):
                    # LTP: pre spike followed by post spike
                    weight_changes_b = torch.outer(pre_trace, post_spike[b])
                    weight_changes_t += lr * reward_mod * reward_factor * a_plus * weight_changes_b
                    
                    # LTD: post spike followed by pre spike
                    weight_changes_b = torch.outer(pre_spike[b], post_trace)
                    weight_changes_t -= lr * reward_mod * reward_factor * a_minus * weight_changes_b
                    
            # Apply weight decay if enabled
            if params.weight_decay > 0:
                weight_decay = params.weight_decay
                weight_changes_t -= weight_decay * weights_t
                
            # Convert back to NumPy
            weight_changes = weight_changes_t.cpu().numpy()
            
            # Apply homeostatic plasticity if enabled
            if params.homeostatic_mode != HomeostaticMode.NONE:
                weight_changes = self._apply_homeostasis(
                    weight_changes, weights, 
                    learning_state["pre_activity"], 
                    learning_state["post_activity"], 
                    params
                )
                
            # Store updated traces in learning state
            learning_state["pre_trace"] = pre_trace.cpu().numpy()
            learning_state["post_trace"] = post_trace.cpu().numpy()
            learning_state["iteration"] += 1
            
            # Update total weight changes
            total_update = float(np.sum(np.abs(weight_changes)))
            learning_state["total_updates"] += total_update
            
            # Apply weight changes
            new_weights = weights + weight_changes
            
            # Clip weights to allowed range
            new_weights = np.clip(new_weights, params.w_min, params.w_max)
            
            # Apply pruning if enabled
            pruned_synapses = 0
            if self.config["use_structural_plasticity"] and params.pruning_threshold > 0:
                # Identify weak synapses
                weak_mask = new_weights < params.pruning_threshold
                
                # Count pruned synapses
                pruned_synapses = int(np.sum(weak_mask))
                
                # Prune weak synapses
                new_weights[weak_mask] = 0.0
                
                # Apply rewiring (create new random synapses)
                if self.config["rewiring_probability"] > 0:
                    rewire_prob = self.config["rewiring_probability"]
                    
                    # Generate random mask for creating new synapses
                    rewire_mask = np.random.rand(*new_weights.shape) < rewire_prob
                    
                    # Only rewire where weights are currently zero
                    rewire_mask = rewire_mask & (new_weights == 0)
                    
                    # Initialize new synapses with small weights
                    new_synapse_weight = params.w_max * 0.1
                    new_weights[rewire_mask] = new_synapse_weight
                    
            # Update learning state
            self._update_learning_state(source, target, learning_state)
            
            # Cache updated weights
            self._cache_weights(source, target, new_weights)
            
            # Get activity for result (average over batch if needed)
            if batch_processing:
                pre_act = pre_activity.mean(dim=0).cpu().numpy()
                post_act = post_activity.mean(dim=0).cpu().numpy()
            else:
                pre_act = pre_activity[0].cpu().numpy()
                post_act = post_activity[0].cpu().numpy()
                
            # Create result
            result = STDPResult(
                weights=new_weights,
                weight_changes=weight_changes,
                pre_activity=pre_act,
                post_activity=post_act,
                pruned_synapses=pruned_synapses,
                total_update=total_update,
                metadata={
                    "iteration": learning_state["iteration"],
                    "reward": reward,
                    "reward_modulation": reward_mod,
                    "reward_baseline": learning_state["reward_baseline"],
                    "timestamp": time.time()
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in PyTorch STDP implementation: {e}")
            
            # Fallback to NumPy implementation
            return self._apply_stdp_numpy(pre_spikes, post_spikes, weights, params, reward, source, target)
    
    def _apply_homeostasis(self, weight_changes: np.ndarray, weights: np.ndarray,
                         pre_activity: np.ndarray, post_activity: np.ndarray,
                         params: STDPParameters) -> np.ndarray:
        """
        Apply homeostatic plasticity to weight changes.
        
        Args:
            weight_changes: Weight change matrix
            weights: Current weight matrix
            pre_activity: Presynaptic activity
            post_activity: Postsynaptic activity
            params: STDP parameters
            
        Returns:
            Modified weight changes
        """
        homeostatic_mode = params.homeostatic_mode
        target_rate = params.target_rate
        homeostatic_rate = params.homeostatic_rate
        
        if homeostatic_mode == HomeostaticMode.SCALING:
            # Synaptic scaling: scale weights to maintain target firing rate
            for j in range(post_activity.shape[0]):
                # Calculate scaling factor based on activity
                if post_activity[j] > 0:
                    rate_ratio = target_rate / post_activity[j]
                    
                    # Apply scaling to weight changes
                    scale_factor = np.exp(homeostatic_rate * np.log(rate_ratio))
                    
                    # Ensure scaling is reasonable
                    scale_factor = np.clip(scale_factor, 0.9, 1.1)
                    
                    # Apply to all incoming weights
                    weight_changes[:, j] *= scale_factor
                    
        elif homeostatic_mode == HomeostaticMode.THRESHOLD:
            # Threshold adjustment: strengthen or weaken all weights based on activity
            for j in range(post_activity.shape[0]):
                # Calculate activity difference
                activity_diff = post_activity[j] - target_rate
                
                # Apply threshold adjustment
                adjustment = -homeostatic_rate * activity_diff
                
                # Apply to all incoming weights
                weight_changes[:, j] += adjustment
                
        elif homeostatic_mode == HomeostaticMode.METAPLASTICITY:
            # Metaplasticity: adjust learning rates based on activity history
            for j in range(post_activity.shape[0]):
                # Calculate activity difference
                activity_diff = post_activity[j] - target_rate
                
                # Calculate metaplastic factor
                meta_factor = np.exp(-homeostatic_rate * activity_diff)
                
                # Ensure factor is reasonable
                meta_factor = np.clip(meta_factor, 0.5, 2.0)
                
                # Apply asymmetrically - reduce potentiation and increase depression
                # for high activity, and vice versa for low activity
                if activity_diff > 0:
                    # High activity: reduce potentiation, increase depression
                    potentiation_mask = weight_changes[:, j] > 0
                    depression_mask = weight_changes[:, j] < 0
                    
                    weight_changes[:, j][potentiation_mask] /= meta_factor
                    weight_changes[:, j][depression_mask] *= meta_factor
                else:
                    # Low activity: increase potentiation, reduce depression
                    potentiation_mask = weight_changes[:, j] > 0
                    depression_mask = weight_changes[:, j] < 0
                    
                    weight_changes[:, j][potentiation_mask] *= meta_factor
                    weight_changes[:, j][depression_mask] /= meta_factor
                    
        return weight_changes
    
    def get_regime_specific_parameters(self, regime: str, base_params: Optional[STDPParameters] = None) -> STDPParameters:
        """
        Get regime-specific STDP parameters.
        
        Args:
            regime: Market regime (growth, conservation, release, reorganization)
            base_params: Base parameters to modify (default: self.default_params)
            
        Returns:
            STDP parameters adapted for the regime
        """
        if not self.config["use_regime_specific_stdp"]:
            return base_params or self.default_params
            
        if base_params is None:
            base_params = self.default_params
            
        # Get regime parameters
        regime_params = self.config["regime_stdp_params"].get(regime.lower())
        
        if regime_params is None:
            # Default to conservation if regime not recognized
            regime_params = self.config["regime_stdp_params"].get("conservation", {})
            
        # Create new parameters
        params = STDPParameters(
            a_plus=regime_params.get("a_plus", base_params.a_plus),
            a_minus=regime_params.get("a_minus", base_params.a_minus),
            tau_plus=regime_params.get("tau_plus", base_params.tau_plus),
            tau_minus=regime_params.get("tau_minus", base_params.tau_minus),
            w_min=regime_params.get("w_min", base_params.w_min),
            w_max=regime_params.get("w_max", base_params.w_max),
            nearest_neighbor=regime_params.get("nearest_neighbor", base_params.nearest_neighbor),
            mode=STDPMode.from_string(regime_params.get("mode", str(base_params.mode))),
            learning_rate=regime_params.get("learning_rate", base_params.learning_rate),
            reward_factor=regime_params.get("reward_factor", base_params.reward_factor),
            homeostatic_mode=HomeostaticMode.from_string(regime_params.get("homeostatic_mode", str(base_params.homeostatic_mode))),
            target_rate=regime_params.get("target_rate", base_params.target_rate),
            homeostatic_rate=regime_params.get("homeostatic_rate", base_params.homeostatic_rate),
            pruning_threshold=regime_params.get("pruning_threshold", base_params.pruning_threshold),
            initial_weight_sigma=regime_params.get("initial_weight_sigma", base_params.initial_weight_sigma),
            weight_decay=regime_params.get("weight_decay", base_params.weight_decay),
            conduction_delay=regime_params.get("conduction_delay", base_params.conduction_delay)
        )
        
        return params
    
    def optimize_signal_weights(self, signal_values: Dict[str, np.ndarray], 
                             current_weights: Optional[Dict[str, float]] = None,
                             target: str = "fusion", regime: Optional[str] = None,
                             params: Optional[STDPParameters] = None,
                             reward: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize signal weights using STDP learning.
        
        Args:
            signal_values: Dictionary of signal name to value array
            current_weights: Current weights (default: equal weights)
            target: Target identifier (for state tracking)
            regime: Market regime (for regime-specific parameters)
            params: STDP parameters (or use defaults for regime)
            reward: Optional reward signal
            
        Returns:
            Dictionary of optimized weights
        """
        if not signal_values:
            return {}
            
        # Get default current weights if not provided
        if current_weights is None:
            # Initialize with equal weights
            current_weights = {name: 1.0 / len(signal_values) for name in signal_values}
            
        # Get parameters
        if params is None and regime is not None:
            # Use regime-specific parameters
            params = self.get_regime_specific_parameters(regime)
        elif params is None:
            # Use default parameters
            params = self.default_params
            
        # Convert signals to spike trains
        pre_signals = []
        signal_names = []
        
        for name, values in signal_values.items():
            pre_signals.append(values)
            signal_names.append(name)
            
        # Convert to spike-like representation
        try:
            # Stack signals (time, signals)
            stacked_signals = np.column_stack(pre_signals)
            
            # Normalize to [0, 1] for each signal
            min_vals = np.min(stacked_signals, axis=0, keepdims=True)
            max_vals = np.max(stacked_signals, axis=0, keepdims=True)
            
            # Ensure non-zero range
            ranges = np.maximum(max_vals - min_vals, 1e-8)
            
            # Normalize
            normalized = (stacked_signals - min_vals) / ranges
            
            # Convert to spike-like format (threshold crossing)
            # Use rate coding: probability of spike proportional to signal value
            spike_threshold = 0.7  # High threshold to get sparse activity
            
            # Generate random values
            random_vals = np.random.rand(*normalized.shape)
            
            # Generate spikes (1 where normalized value exceeds random value)
            pre_spikes = (normalized > random_vals).astype(float)
            
            # Create "fusion" signal from weighted sum of inputs
            weights_array = np.array([current_weights.get(name, 1.0 / len(signal_names)) 
                                  for name in signal_names])
            
            # Normalize weights
            weights_array = weights_array / np.sum(weights_array)
            
            # Calculate weighted sum
            weighted_sum = np.dot(stacked_signals, weights_array)
            
            # Normalize weighted sum
            weighted_min = np.min(weighted_sum)
            weighted_max = np.max(weighted_sum)
            weighted_range = max(weighted_max - weighted_min, 1e-8)
            
            weighted_normalized = (weighted_sum - weighted_min) / weighted_range
            
            # Convert to spikes (use same method as for inputs)
            random_vals = np.random.rand(len(weighted_normalized))
            post_spikes = (weighted_normalized > random_vals).astype(float)
            
            # Reshape to compatible dimensions for STDP
            pre_spikes_reshaped = pre_spikes.reshape(pre_spikes.shape[0], pre_spikes.shape[1])
            post_spikes_reshaped = post_spikes.reshape(post_spikes.shape[0], 1)
            
            # Get weight matrix from current weights
            n_pre = len(signal_names)
            n_post = 1  # Single fusion output
            
            weight_matrix = np.zeros((n_pre, n_post))
            for i, name in enumerate(signal_names):
                weight_matrix[i, 0] = current_weights.get(name, 1.0 / n_pre)
                
            # Create source identifier
            source = "signals:" + "-".join(signal_names)
            
            # Apply STDP
            stdp_result = self.apply_stdp(
                pre_spikes_reshaped, post_spikes_reshaped, 
                weight_matrix, params, reward, source, target
            )
            
            # Extract optimized weights
            optimized_weights = stdp_result.weights.flatten()
            
            # Ensure weights are positive
            optimized_weights = np.maximum(optimized_weights, 0.0)
            
            # Normalize weights to sum to 1
            weight_sum = np.sum(optimized_weights)
            
            if weight_sum > 0:
                optimized_weights = optimized_weights / weight_sum
            else:
                # Fallback to uniform weights
                optimized_weights = np.ones(n_pre) / n_pre
                
            # Create result dictionary
            optimized_dict = {name: float(optimized_weights[i]) 
                           for i, name in enumerate(signal_names)}
            
            return optimized_dict
            
        except Exception as e:
            self.logger.error(f"Error optimizing signal weights: {e}")
            
            # Fallback to current weights
            return current_weights
    
    def optimize_fusion_weights(self, fusion_systems: Dict[str, Dict[str, float]],
                             system_outputs: Dict[str, np.ndarray],
                             current_system_weights: Optional[Dict[str, float]] = None,
                             target: str = "meta_fusion", regime: Optional[str] = None,
                             params: Optional[STDPParameters] = None,
                             reward: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize fusion system weights using STDP learning.
        
        Args:
            fusion_systems: Dictionary of system name to signal weights
            system_outputs: Dictionary of system name to output values
            current_system_weights: Current system weights (default: equal weights)
            target: Target identifier (for state tracking)
            regime: Market regime (for regime-specific parameters)
            params: STDP parameters (or use defaults for regime)
            reward: Optional reward signal
            
        Returns:
            Dictionary of optimized system weights
        """
        # Similar to optimize_signal_weights but for meta-fusion
        # This method optimizes the weights for combining multiple fusion systems
        
        if not fusion_systems or not system_outputs:
            return {}
            
        # Get default current weights if not provided
        if current_system_weights is None:
            # Initialize with equal weights
            current_system_weights = {name: 1.0 / len(fusion_systems) for name in fusion_systems}
            
        # Get parameters
        if params is None and regime is not None:
            # Use regime-specific parameters
            params = self.get_regime_specific_parameters(regime)
        elif params is None:
            # Use default parameters
            params = self.default_params
            
        # Convert system outputs to spike trains
        system_names = list(system_outputs.keys())
        
        try:
            # Get system outputs as array (time, systems)
            system_array = np.column_stack([system_outputs[name] for name in system_names])
            
            # Normalize to [0, 1] for each system
            min_vals = np.min(system_array, axis=0, keepdims=True)
            max_vals = np.max(system_array, axis=0, keepdims=True)
            
            # Ensure non-zero range
            ranges = np.maximum(max_vals - min_vals, 1e-8)
            
            # Normalize
            normalized = (system_array - min_vals) / ranges
            
            # Convert to spike-like format (threshold crossing with randomness)
            spike_threshold = 0.7  # High threshold to get sparse activity
            
            # Generate random values
            random_vals = np.random.rand(*normalized.shape)
            
            # Generate spikes (1 where normalized value exceeds random value)
            pre_spikes = (normalized > random_vals).astype(float)
            
            # Create "meta fusion" signal from weighted sum of system outputs
            weights_array = np.array([current_system_weights.get(name, 1.0 / len(system_names)) 
                                  for name in system_names])
            
            # Normalize weights
            weights_array = weights_array / np.sum(weights_array)
            
            # Calculate weighted sum
            weighted_sum = np.dot(system_array, weights_array)
            
            # Normalize weighted sum
            weighted_min = np.min(weighted_sum)
            weighted_max = np.max(weighted_sum)
            weighted_range = max(weighted_max - weighted_min, 1e-8)
            
            weighted_normalized = (weighted_sum - weighted_min) / weighted_range
            
            # Convert to spikes (use same method as for inputs)
            random_vals = np.random.rand(len(weighted_normalized))
            post_spikes = (weighted_normalized > random_vals).astype(float)
            
            # Reshape to compatible dimensions for STDP
            pre_spikes_reshaped = pre_spikes.reshape(pre_spikes.shape[0], pre_spikes.shape[1])
            post_spikes_reshaped = post_spikes.reshape(post_spikes.shape[0], 1)
            
            # Get weight matrix from current weights
            n_pre = len(system_names)
            n_post = 1  # Single fusion output
            
            weight_matrix = np.zeros((n_pre, n_post))
            for i, name in enumerate(system_names):
                weight_matrix[i, 0] = current_system_weights.get(name, 1.0 / n_pre)
                
            # Create source identifier
            source = "systems:" + "-".join(system_names)
            
            # Apply STDP
            stdp_result = self.apply_stdp(
                pre_spikes_reshaped, post_spikes_reshaped, 
                weight_matrix, params, reward, source, target
            )
            
            # Extract optimized weights
            optimized_weights = stdp_result.weights.flatten()
            
            # Ensure weights are positive
            optimized_weights = np.maximum(optimized_weights, 0.0)
            
            # Normalize weights to sum to 1
            weight_sum = np.sum(optimized_weights)
            
            if weight_sum > 0:
                optimized_weights = optimized_weights / weight_sum
            else:
                # Fallback to uniform weights
                optimized_weights = np.ones(n_pre) / n_pre
                
            # Create result dictionary
            optimized_dict = {name: float(optimized_weights[i]) 
                           for i, name in enumerate(system_names)}
            
            return optimized_dict
            
        except Exception as e:
            self.logger.error(f"Error optimizing fusion weights: {e}")
            
            # Fallback to current weights
            return current_system_weights
