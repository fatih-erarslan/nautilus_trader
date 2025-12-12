#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 00:32:15 2025

@author: ashina
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QStar Learning Trading System with GPU Acceleration and Freqtrade Integration

This module implements the Q* learning algorithm with GPU acceleration for cryptocurrency
trading, leveraging Freqtrade's historical data and backtesting infrastructure.

Key features:
- GPU-accelerated Q* Learning implementation
- Integration with Freqtrade's data storage
- Multi-pair and multi-timeframe support
- Advanced technical indicators for market state representation
- Performance tracking and visualization
- Model persistence and management
"""

import os
import sys
import logging
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime
import json
import pickle
import threading
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QStar-Freqtrade")

# Intelligent path setup for Freqtrade environment


def setup_freqtrade_paths():
    """
    Set up paths for Freqtrade environment, handling both direct calls and strategy imports.
    
    Returns:
        Dict: Dictionary containing important paths
    """
    # Determine script directory
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
    else:
        script_dir = Path.cwd()
    
    # Find user_data directory (handle both strategy dir and direct script execution)
    if script_dir.name == "strategies" or script_dir.parent.name == "strategies":
        # We're likely in a strategy directory
        if script_dir.name == "strategies":
            user_data_dir = script_dir.parent
            strategy_dir = script_dir
        else:  # We're in a subdirectory of strategies
            user_data_dir = script_dir.parent.parent
            strategy_dir = script_dir.parent
    else:
        # Try to find user_data directory by looking for common subdirectories
        current = script_dir
        user_data_dir = None
        while current != current.parent:  # Stop at filesystem root
            if (current / "data").exists() and (current / "strategies").exists():
                user_data_dir = current
                strategy_dir = current / "strategies"
                break
            current = current.parent
        
        # If still not found, assume current directory and warn
        if user_data_dir is None:
            logger.warning("Could not locate user_data directory. Using current directory.")
            user_data_dir = script_dir
            strategy_dir = script_dir
    
    # Define key paths
    paths = {
        "user_data_dir": user_data_dir,
        "strategy_dir": strategy_dir,
        "data_dir": user_data_dir / "data",
        "models_dir": user_data_dir / "models"
    }
    
    # Ensure paths exist
    for name, path in paths.items():
        if name != "models_dir":  # Don't check models dir as it might not exist yet
            if not path.exists():
                logger.warning(f"Path {name} ({path}) does not exist.")
    
    # Create models directory if it doesn't exist
    paths["models_dir"].mkdir(exist_ok=True, parents=True)
    
    # Add paths to sys.path for imports if needed
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    
    logger.info(f"Freqtrade paths configured. User data: {paths['user_data_dir']}")
    return paths

# Path setup
PATHS = setup_freqtrade_paths()

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not available, using NumPy for computations")
    
# At the top of your file
try:
    import pennylane as qml
    from pennylane import numpy as qnp
    PENNYLANE_AVAILABLE = True
    
    # Try to import and use Catalyst
    try:
        from pennylane import catalyst
        # Check if Catalyst can use hardware acceleration
        catalyst_device = qml.device("lightning.gpu", wires=2)
        CATALYST_GPU_AVAILABLE = True
        logger.info("PennyLane Catalyst with GPU acceleration available")
    except Exception as e:
        CATALYST_GPU_AVAILABLE = False
        logger.info(f"PennyLane Catalyst GPU acceleration not available: {e}")
        
except ImportError:
    PENNYLANE_AVAILABLE = False
    CATALYST_GPU_AVAILABLE = False
    
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.info("JAX not available, falling back to other methods")

try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("TA-Lib not available, using basic indicators")

# ---------------------------------
# GPU Utilities
# ---------------------------------

def get_tensor_lib():
    """Get the tensor library to use for computations."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch
    elif JAX_AVAILABLE:
        # Just use JAX regardless of GPU detection
        return jnp
    else:
        import numpy as np
        return np

def is_gpu_available() -> bool:
    """Check if GPU is available for computation."""
    # Check PyTorch first
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return True
    
    # Then check PennyLane
    if PENNYLANE_AVAILABLE:
        try:
            # Try lightning.gpu device which works with AMD
            device = qml.device("lightning.gpu", wires=2)
            return True
        except Exception:
            pass
            
    # Finally check JAX
    if JAX_AVAILABLE:
        try:
            # Try to get GPU devices
            return len(jax.devices('gpu')) > 0
        except RuntimeError:
            # JAX couldn't find GPU backends
            try:
                # Alternative check - look for GPUs in all devices
                devices = jax.devices()
                return any('gpu' in str(d).lower() for d in devices)
            except Exception:
                pass
    
    return False

def to_tensor(data, device='auto'):
    """Convert array-like data to a tensor on the specified device."""
    tensor_lib = get_tensor_lib()
    
    # Already a tensor - just move to device if needed
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        if device == 'auto':
            if torch.cuda.is_available():
                return data.cuda()
            else:
                return data.cpu()
        elif device == 'cuda' or device == 'gpu':
            return data.cuda()
        else:
            return data.cpu()
    elif JAX_AVAILABLE and hasattr(data, 'device_buffer'):
        if device == 'auto' or device == 'gpu':
            return jax.device_put(data, jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])
        else:
            return jax.device_put(data, jax.devices('cpu')[0])
    
    # Convert to tensor based on available libraries
    if tensor_lib is torch:
        result = torch.tensor(data, dtype=torch.float32)
        if device == 'auto':
            if torch.cuda.is_available():
                return result.cuda()
            else:
                return result
        elif device == 'cuda' or device == 'gpu':
            if torch.cuda.is_available():
                return result.cuda()
            return result
        else:
            return result
    elif tensor_lib is jnp:
        result = jnp.array(data, dtype=jnp.float32)
        if device == 'auto' or device == 'gpu':
            return jax.device_put(result, jax.devices('gpu')[0] if jax.devices('gpu') else jax.devices('cpu')[0])
        else:
            return jax.device_put(result, jax.devices('cpu')[0])
    else:
        return np.array(data, dtype=np.float32)

def to_numpy(tensor):
    """Convert a tensor to a NumPy array."""
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    elif JAX_AVAILABLE and hasattr(tensor, 'device_buffer'):
        return np.array(tensor)
    return np.array(tensor)

# ---------------------------------
# Data Loading and Processing
# ---------------------------------

def find_available_data(exchange_name: str = 'binance') -> Dict[str, List[str]]:
    """
    Find all available pairs and timeframes in Freqtrade data directory.
    
    Args:
        exchange_name: Name of the exchange
        
    Returns:
        Dict: Dictionary of pairs and their available timeframes
    """
    data_dir = PATHS["data_dir"] / exchange_name
    if not data_dir.exists():
        logger.error(f"Exchange directory not found: {data_dir}")
        return {}
    
    available_data = {}
    
    # Get all feather files in the directory
    feather_files = list(data_dir.glob("*.feather"))
    
    # Parse pair names and timeframes from filenames
    for filepath in feather_files:
        filename = filepath.stem
        parts = filename.split('-')
        
        if len(parts) >= 2:
            # Last part is timeframe
            timeframe = parts[-1]
            # Everything before the last dash is the pair name
            pair = '-'.join(parts[:-1])
            
            # Convert from filename format to exchange format
            pair = pair.replace('_', '/')
            
            # Add to available data
            if pair not in available_data:
                available_data[pair] = []
            available_data[pair].append(timeframe)
    
    # Sort timeframes for each pair
    for pair in available_data:
        available_data[pair].sort()
    
    return available_data

def load_freqtrade_data(pair: str, timeframe: str, exchange: str = 'binance') -> pd.DataFrame:
    """
    Load historical data from Freqtrade's data directory.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '5m')
        exchange: Exchange name
        
    Returns:
        DataFrame: Historical price data with OHLCV columns
    """
    # Convert pair to filename format
    pair_file = pair.replace('/', '_')
    filepath = PATHS["data_dir"] / exchange / f"{pair_file}-{timeframe}.feather"
    
    logger.info(f"Loading data from: {filepath}")
    
    if not filepath.exists():
        logger.error(f"Data file not found: {filepath}")
        logger.error(f"Please download data using: freqtrade download-data --exchange {exchange} --pairs {pair} --timeframes {timeframe}")
        return pd.DataFrame()
    
    try:
        # Load feather file
        df = pd.read_feather(filepath)
        
        # Validate required columns
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            logger.error(f"Feather file {filepath} missing required columns: {missing}")
            return pd.DataFrame()
        
        # Convert date to datetime index
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        
        # Ensure numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean data
        df.dropna(subset=['close'], inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} candles from {filepath}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}", exc_info=True)
        return pd.DataFrame()

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for trading features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame: Original data with indicators added
    """
    if df.empty:
        return df
    
    logger.info("Calculating technical indicators...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    try:
        # Trend indicators
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Momentum indicators
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility indicators
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR for volatility
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # Volume indicators
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # Trend strength
        df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        
        # Custom indicators for Q* environment
        # Normalized returns
        df['returns'] = df['close'].pct_change()
        df['returns_z'] = (df['returns'] - df['returns'].rolling(30).mean()) / df['returns'].rolling(30).std()
        
        # Volatility regime
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
        df['volatility_regime'] = (df['volatility'] - df['volatility'].rolling(100).min()) / \
                               (df['volatility'].rolling(100).max() - df['volatility'].rolling(100).min() + 1e-10)
        
        # Q* specific features
        df['qerc_trend'] = df['returns'].rolling(10).mean() * 10
        df['qerc_momentum'] = df['close'].pct_change(5)
        df['iqad_score'] = df['returns'].rolling(20).std() / df['returns'].abs().rolling(20).mean()
        df['performance_metric'] = df['returns'].rolling(20).mean()
        
        # Clean up NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        logger.info(f"Calculated {len(df.columns) - 6} indicators")  # -6 for OHLCV + date
        return df
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        # Return original DataFrame if error occurs
        return df

# ---------------------------------
# Experience Buffer with GPU Support
# ---------------------------------

class GPUExperienceBuffer:
    """Memory buffer for experience replay with prioritized sampling and GPU acceleration."""

    def __init__(self, max_size: int = 10000, alpha: float = 0.6, use_gpu: bool = True):
        """
        Initialize experience buffer with GPU support.

        Args:
            max_size: Maximum buffer size
            alpha: Priority exponent (0 = uniform sampling)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.max_size = max_size
        self.alpha = alpha
        self.epsilon = 1e-6  # Small constant to avoid zero priority
        self.use_gpu = use_gpu and (TORCH_AVAILABLE or JAX_AVAILABLE)
        
        # Use lists for initial storage before conversion to tensors
        self.buffer = []
        self.priorities = []
        
        # Tensor versions for GPU
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._priorities = None
        
        # Track current size and position
        self.size = 0
        self.position = 0
        
        # Get tensor library
        self.tensor_lib = get_tensor_lib()
        
        logger.debug(f"Initialized GPUExperienceBuffer with max_size={max_size}, use_gpu={self.use_gpu}")

    def add(self, experience: Tuple, error: float = None):
        """
        Add experience to buffer with GPU-aware storage.

        Args:
            experience: (state, action, reward, next_state) tuple
            error: TD error for prioritization (if None, max priority is used)
        """
        state, action, reward, next_state = experience
        
        # Set priority (either from error or max existing priority)
        if error is None:
            priority = max(self.priorities) if self.priorities else 1.0
        else:
            priority = abs(error) + self.epsilon
            
        # Initial fill phase - use regular lists
        if self.size < self.max_size:
            if self.size < len(self.buffer):
                # Overwrite existing entry
                self.buffer[self.position] = experience
                self.priorities[self.position] = priority
            else:
                # Add new entry
                self.buffer.append(experience)
                self.priorities.append(priority)
            
            self.size = max(self.size, self.position + 1)
        else:
            # Buffer is full, use circular indexing
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        # Update position
        self.position = (self.position + 1) % self.max_size
        
        # Invalidate tensor cache when enough new data has been added
        if self._states is not None and self.position % 100 == 0:
            self._invalidate_tensor_cache()

    def _invalidate_tensor_cache(self):
        """Invalidate tensor cache to force regeneration."""
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._priorities = None

    def _ensure_tensors(self):
        """Ensure tensor versions of buffer data are available and up-to-date."""
        if self._states is not None:
            return  # Already converted
            
        if self.size == 0:
            return  # Nothing to convert
            
        # Extract individual components for better GPU performance
        states, actions, rewards, next_states = zip(*self.buffer[:self.size])
        
        # Convert to tensors based on available frameworks
        if self.use_gpu:
            self._states = to_tensor(np.array(states), device='auto')
            self._actions = to_tensor(np.array(actions), device='auto')
            self._rewards = to_tensor(np.array(rewards), device='auto')
            self._next_states = to_tensor(np.array(next_states), device='auto')
            self._priorities = to_tensor(np.array(self.priorities[:self.size]), device='auto')
        else:
            # Use NumPy for CPU-only processing
            self._states = np.array(states)
            self._actions = np.array(actions)
            self._rewards = np.array(rewards)
            self._next_states = np.array(next_states)
            self._priorities = np.array(self.priorities[:self.size])

    def sample(self, batch_size: int) -> Tuple[List[Tuple], List[int], np.ndarray]:
        """
        Sample batch from buffer with prioritization and GPU acceleration.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple containing experiences, indices, and importance sampling weights
        """
        if self.size < batch_size:
            batch_size = self.size
            
        if self.size == 0:
            return [], [], np.array([])
            
        # Ensure tensors are available
        self._ensure_tensors()
        
        # Compute sampling probabilities
        if self.use_gpu:
            if TORCH_AVAILABLE and isinstance(self._priorities, torch.Tensor):
                priorities_alpha = self._priorities ** self.alpha
                prob = priorities_alpha / priorities_alpha.sum()
                prob_np = to_numpy(prob)
            elif JAX_AVAILABLE and hasattr(self._priorities, 'device_buffer'):
                priorities_alpha = self._priorities ** self.alpha
                prob = priorities_alpha / jnp.sum(priorities_alpha)
                prob_np = to_numpy(prob)
            else:
                priorities_alpha = self._priorities ** self.alpha
                prob_np = priorities_alpha / np.sum(priorities_alpha)
        else:
            priorities = np.array(self.priorities[:self.size]) ** self.alpha
            prob_np = priorities / np.sum(priorities)
            
        # Sample indices based on prioritization probabilities
        indices = np.random.choice(self.size, batch_size, p=prob_np, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * prob_np[indices]) ** (-0.4)  # Beta=0.4
        weights = weights / np.max(weights)  # Normalize
        
        # Extract experiences - mix of tensor and list operations for efficiency
        if self.use_gpu:
            # Creating a list of tensors is more efficient than slicing in some cases
            if TORCH_AVAILABLE and isinstance(self._states, torch.Tensor):
                batch_states = self._states[indices]
                batch_actions = self._actions[indices]
                batch_rewards = self._rewards[indices]
                batch_next_states = self._next_states[indices]
                
                # Convert to numpy for experience tuples
                states_np = to_numpy(batch_states)
                actions_np = to_numpy(batch_actions)
                rewards_np = to_numpy(batch_rewards)
                next_states_np = to_numpy(batch_next_states)
                
                experiences = [
                    (states_np[i], actions_np[i], rewards_np[i], next_states_np[i])
                    for i in range(batch_size)
                ]
            else:
                # JAX or CPU fallback
                experiences = [self.buffer[i] for i in indices]
        else:
            experiences = [self.buffer[i] for i in indices]
            
        return experiences, indices.tolist(), weights

    def update_priorities(self, indices: List[int], errors: List[float]):
        """
        Update priorities for experiences with GPU acceleration if available.

        Args:
            indices: Indices of experiences to update
            errors: TD errors for prioritization
        """
        for i, error in zip(indices, errors):
            if i < self.size:
                priority = abs(error) + self.epsilon
                self.priorities[i] = priority
                
                # Update tensor version if it exists
                if self._priorities is not None and self.use_gpu:
                    if TORCH_AVAILABLE and isinstance(self._priorities, torch.Tensor) and i < len(self._priorities):
                        self._priorities[i] = priority
        
        # If a large number of priorities were updated, invalidate the cache
        if len(indices) > self.size // 10:
            self._invalidate_tensor_cache()

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size
    
    def get_batch_tensors(self, indices: List[int]) -> Tuple:
        """
        Get batch of experiences as tensors for efficient GPU processing.
        
        Args:
            indices: Indices of experiences to retrieve
            
        Returns:
            Tuple of (states, actions, rewards, next_states) tensors
        """
        self._ensure_tensors()
        
        if self.use_gpu:
            # Direct tensor indexing for GPU acceleration
            if TORCH_AVAILABLE and isinstance(self._states, torch.Tensor):
                idx_tensor = torch.tensor(indices, device=self._states.device)
                return (
                    torch.index_select(self._states, 0, idx_tensor),
                    torch.index_select(self._actions, 0, idx_tensor),
                    torch.index_select(self._rewards, 0, idx_tensor),
                    torch.index_select(self._next_states, 0, idx_tensor)
                )
            elif JAX_AVAILABLE and hasattr(self._states, 'device_buffer'):
                # JAX indexing
                return (
                    self._states[indices],
                    self._actions[indices],
                    self._rewards[indices],
                    self._next_states[indices]
                )
        
        # CPU fallback
        states = np.array([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        rewards = np.array([self.buffer[i][2] for i in indices])
        next_states = np.array([self.buffer[i][3] for i in indices])
        
        return states, actions, rewards, next_states

# ---------------------------------
# Market State and Environment
# ---------------------------------

class MarketState:
    """
    Market state representation combining technical indicators and features.
    
    This class handles the conversion of raw market data (OHLCV + indicators)
    into a state representation suitable for reinforcement learning.
    """
    
    def __init__(self, num_states: int = 200, feature_keys=None, history_size=100, feature_window: int = 50):
        """
        Initialize market state with default features.
        
        Args:
            num_states: Number of discrete states
            feature_keys: List of feature keys to use (None for defaults)
            history_size: Size of feature history to maintain
            feature_window: Window size for feature aggregation
        """
        self.logger = logging.getLogger("MarketState")
        self.feature_keys = feature_keys or [
            'qerc_trend', 'volatility_regime', 'qerc_momentum', 
            'rsi_14', 'adx', 'macd', 'bb_width', 'atr'
        ]
        self.history_size = history_size
        self.num_states = num_states
        self.feature_window = feature_window
        self.feature_history = deque(maxlen=feature_window)
        
        # Initialize with defaults for all common features
        self.normalized_features = {k: 0.5 for k in self.feature_keys}
        self.normalized_features.update({
            'trend': 0.5,
            'volume': 0.5,
            'close': 0.5,
            'high': 0.5,
            'low': 0.5,
            'open': 0.5
        })
        
        self.feature_mins = {}
        self.feature_maxs = {}
        
        # Start initialized with defaults
        self._initialized = True
        
    def update(self, features: Dict[str, float]) -> None:
        """
        Update state with new market features.
        
        Args:
            features: Dictionary of feature values
        """
        # Handle empty features case
        if not features:
            self.logger.warning("Empty features dictionary provided to MarketState")
            return
            
        # Store original features for reference
        self.feature_history.append(features.copy())
        
        # Filter out non-numeric values
        valid_features = {}
        for key, value in features.items():
            try:
                # Convert to float and check for NaN/inf
                float_val = float(value)
                if not np.isnan(float_val) and not np.isinf(float_val):
                    valid_features[key] = float_val
            except (ValueError, TypeError):
                continue
        
        # Check if we have any valid features after filtering
        if not valid_features:
            self.logger.warning("No valid numeric features available after filtering")
            return
            
        # Update min/max values for normalization
        for key, value in valid_features.items():
            if key not in self.feature_mins or value < self.feature_mins[key]:
                self.feature_mins[key] = value
            if key not in self.feature_maxs or value > self.feature_maxs[key]:
                self.feature_maxs[key] = value
                
        # Update normalized features (preserve existing ones)
        for key, value in valid_features.items():
            if key in self.feature_mins and key in self.feature_maxs:
                min_val = self.feature_mins[key]
                max_val = self.feature_maxs[key]
                if max_val > min_val:
                    self.normalized_features[key] = (value - min_val) / (max_val - min_val)
                else:
                    self.normalized_features[key] = 0.5
            else:
                self.normalized_features[key] = 0.5
                
        # Set initialization flag once we have normalized features
        if not self._initialized and self.normalized_features:
            self._initialized = True
                    
    def get_state_index(self) -> int:
        """
        Map current normalized features to a discrete state index.
        
        Returns:
            State index (0 to num_states-1)
        """
        if not self._initialized or not self.normalized_features:
            self.logger.warning("No normalized features available, returning default state 0")
            return 0
            
        try:
            # Select key features for state determination
            state_features = {}
            for key in self.feature_keys:
                if key in self.normalized_features:
                    state_features[key] = self.normalized_features[key]
                else:
                    state_features[key] = 0.5  # Default if not available
            
            # Create a hash from discretized feature values
            hash_value = 0
            bins = 5  # Number of bins per feature
            for i, (key, value) in enumerate(state_features.items()):
                # Ensure value is valid
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    binned_value = min(bins - 1, max(0, int(value * bins)))
                else:
                    binned_value = 0
                hash_value += binned_value * (bins ** i)
            
            # Map hash to state index
            state_index = hash_value % self.num_states
            return state_index
        except Exception as e:
            self.logger.error(f"Error in get_state_index: {e}")
            return 0  # Return default state on error

class TradingAction:
    """Trading action definitions and utilities."""
    
    # Action definitions
    BUY = 0
    SELL = 1
    HOLD = 2
    REDUCE = 3  # Reduce position
    INCREASE = 4  # Increase position
    
    @staticmethod
    def get_num_actions() -> int:
        """Get number of possible actions."""
        return 5
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """
        Get human-readable action name.
        
        Args:
            action: Action index
            
        Returns:
            Action name
        """
        actions = {
            TradingAction.BUY: "BUY",
            TradingAction.SELL: "SELL",
            TradingAction.HOLD: "HOLD",
            TradingAction.REDUCE: "REDUCE",
            TradingAction.INCREASE: "INCREASE"
        }
        return actions.get(action, "UNKNOWN")

class TradingEnvironment:
    """
    Trading environment for Q* Learning agent.
    
    This environment simulates a trading scenario using historical market data,
    allowing an agent to learn optimal trading strategies.
    """
    
    def __init__(self, price_data=None, window_size=50, 
                 initial_balance=10000.0, transaction_fee=0.001, 
                 reward_scaling=0.01, use_position_limits=True, 
                 max_position_size=1.0):
        """
        Initialize trading environment.
        
        Args:
            price_data: DataFrame with price data and indicators
            window_size: Window size for feature aggregation
            initial_balance: Initial account balance
            transaction_fee: Transaction fee as a fraction
            reward_scaling: Reward scaling factor
            use_position_limits: Whether to use position size limits
            max_position_size: Maximum position size as fraction of portfolio
        """
        # Initialize logger
        self.logger = logging.getLogger("TradingEnvironment")
        
        # Set consistent number of states and actions
        self.num_states = 200
        self.num_actions = TradingAction.get_num_actions()
        
        # Initialize parameters
        self.price_data = price_data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_position_limits = use_position_limits
        self.max_position_size = max_position_size
        
        # Market state
        self.market_state = MarketState(num_states=self.num_states, feature_window=window_size)
        
        # Trading state
        self.balance = initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = initial_balance
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        
        # Market data
        self.current_price = 0.0
        self.current_idx = 0
        self.prices = []
        
        if price_data is not None:
            self.prices = price_data['close'].values
        
        # Tracking variables
        self.returns = []
        self.positions = []
        self.portfolio_values = []
        self.actions_taken = []
        
        # Performance metrics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        self.logger.info(f"Trading environment initialized with {self.num_states} states and {self.num_actions} actions")
        
    def reset(self) -> int:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state index
        """
        # Reset trading state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.initial_balance
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        
        # Reset market position
        self.current_idx = self.window_size
        if self.price_data is not None and len(self.prices) > 0:
            self.current_price = self.prices[self.current_idx]
        else:
            self.current_price = 100.0  # Default starting price
            
        # Reset history
        self.returns = []
        self.positions = []
        self.portfolio_values = []
        self.actions_taken = []
        
        # Reset metrics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_pnl = 0.0
        
        # Update market state
        self._update_market_state()
        
        # Return initial state
        return self.market_state.get_state_index()
        
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        try:
            # Track the action
            self.actions_taken.append(action)
            
            # Execute action
            reward = self._execute_action(action)
            
            # Move to next time step
            done = self._next_time_step()
            
            # If we're done (reached end of data), return immediately
            if done:
                # Return the last valid state for consistency
                return self.market_state.get_state_index(), reward, done
            
            # Update market state with new data
            self._update_market_state()
            
            # Get new state
            next_state = self.market_state.get_state_index()
            
            # Return step results
            return next_state, reward, done
        
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}", exc_info=True)
            
            # Return current state, no reward, and done=True to terminate episode
            return self.market_state.get_state_index(), 0.0, True
        
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action and calculate reward.
        
        Args:
            action: Action to take
            
        Returns:
            Reward
        """
        previous_portfolio_value = self.portfolio_value
        reward = 0.0
        
        # Execute based on action type
        if action == TradingAction.BUY and self.balance > 0:
            # Full buy - use all available balance
            buy_amount = self.balance
            if self.use_position_limits:
                # Limit position size based on settings
                max_buy = self.portfolio_value * self.max_position_size - self.position_value
                buy_amount = min(buy_amount, max(0, max_buy))
                
            if buy_amount > 0:
                # Calculate position after fees
                fee = buy_amount * self.transaction_fee
                buy_amount_after_fee = buy_amount - fee
                new_position = buy_amount_after_fee / self.current_price
                
                # Update state
                self.balance -= buy_amount
                self.position += new_position
                self.position_value = self.position * self.current_price
                self.last_buy_price = self.current_price
                self.total_trades += 1
                self.logger.debug(f"BUY: {new_position:.4f} units at {self.current_price:.2f}")
                
        elif action == TradingAction.SELL and self.position > 0:
            # Full sell - liquidate entire position
            sell_amount = self.position
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L
            position_cost = self.position_value
            pnl = sell_value_after_fee - position_cost
            
            # Update state
            self.balance += sell_value_after_fee
            self.position = 0.0
            self.position_value = 0.0
            self.last_sell_price = self.current_price
            self.total_trades += 1
            self.total_pnl += pnl
            
            if pnl > 0:
                self.profitable_trades += 1
                
            self.logger.debug(f"SELL: {sell_amount:.4f} units at {self.current_price:.2f}, PnL: {pnl:.2f}")
            
        elif action == TradingAction.REDUCE and self.position > 0:
            # Reduce position by 50%
            sell_amount = self.position * 0.5
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L for the sold portion
            position_cost = self.position_value * 0.5
            pnl = sell_value_after_fee - position_cost
            
            # Update state
            self.balance += sell_value_after_fee
            self.position -= sell_amount
            self.position_value = self.position * self.current_price
            self.last_sell_price = self.current_price
            self.total_trades += 1
            self.total_pnl += pnl
            
            if pnl > 0:
                self.profitable_trades += 1
                
            self.logger.debug(f"REDUCE: {sell_amount:.4f} units at {self.current_price:.2f}, PnL: {pnl:.2f}")
            
        elif action == TradingAction.INCREASE and self.balance > 0:
            # Increase position by 50% of available balance
            buy_amount = self.balance * 0.5
            if self.use_position_limits:
                # Limit position size based on settings
                max_buy = self.portfolio_value * self.max_position_size - self.position_value
                buy_amount = min(buy_amount, max(0, max_buy))
                
            if buy_amount > 0:
                # Calculate position after fees
                fee = buy_amount * self.transaction_fee
                buy_amount_after_fee = buy_amount - fee
                new_position = buy_amount_after_fee / self.current_price
                
                # Update state
                self.balance -= buy_amount
                self.position += new_position
                self.position_value = self.position * self.current_price
                self.last_buy_price = self.current_price
                self.total_trades += 1
                self.logger.debug(f"INCREASE: {new_position:.4f} units at {self.current_price:.2f}")
        
        # For HOLD action, do nothing
        
        # Calculate current portfolio value
        self.position_value = self.position * self.current_price
        self.portfolio_value = self.balance + self.position_value
        
        # Track positions and portfolio value
        self.positions.append(self.position)
        self.portfolio_values.append(self.portfolio_value)
        
        # Calculate reward based on portfolio change
        portfolio_return = (self.portfolio_value / previous_portfolio_value) - 1.0
        self.returns.append(portfolio_return)
        
        # Calculate reward - scaled return with penalty for excessive trading
        reward = portfolio_return * self.reward_scaling
        
        return reward
        
    def _next_time_step(self) -> bool:
        """
        Move to next time step.
        
        Returns:
            Whether the episode is done
        """
        self.current_idx += 1
        
        # Check if we've reached the end of available data
        if self.price_data is not None:
            if self.current_idx >= len(self.prices):
                # Important: Set the done flag and return True
                self.logger.debug(f"Reached end of price data at index {self.current_idx}, total length: {len(self.prices)}")
                return True
            
            # Update current price
            self.current_price = self.prices[self.current_idx]
        else:
            # Simulate random price movement if no data provided
            price_change = np.random.normal(0, 0.01)  # 1% standard deviation
            self.current_price *= (1 + price_change)
            
            # End after 200 steps if no data provided
            if self.current_idx > 200:
                return True
                
        return False
            
    def _update_market_state(self) -> None:
        """Update market state with latest data and features."""
        if self.price_data is None:
            # Create synthetic features
            features = {
                'price': self.current_price,
                'volatility_regime': 0.5,
                'qerc_trend': 0.5,
                'qerc_momentum': 0.5,
                'iqad_score': 0.0,
                'performance_metric': 0.5
            }
        else:
            # Check if current_idx is within bounds
            if self.current_idx >= len(self.price_data):
                self.logger.warning(f"Current index {self.current_idx} exceeds price data length {len(self.price_data)}. Using last valid index.")
                self.current_idx = len(self.price_data) - 1
                
            # Get the current row of data
            current_row = self.price_data.iloc[self.current_idx]
            
            # Create features dictionary from current row
            features = current_row.to_dict()
            
            # Ensure essential features are included
            if 'close' not in features:
                features['close'] = self.current_price
                
            # Extract additional features from window if needed
            # This is optional but can provide more context
            if len(self.price_data) > self.window_size:
                start_idx = max(0, self.current_idx - self.window_size)
                window_data = self.price_data.iloc[start_idx:self.current_idx + 1]
                
                # Add trend feature based on window
                if len(window_data) > 1:
                    price_change = (window_data['close'].iloc[-1] / window_data['close'].iloc[0]) - 1
                    features['trend'] = price_change
        
        # Update market state
        self.market_state.update(features)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current episode.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns metrics
        returns = np.array(self.returns)
        
        metrics = {
            "total_return": (self.portfolio_value / self.initial_balance) - 1.0,
            "sharpe_ratio": np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if len(returns) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(),
            "win_rate": self.profitable_trades / max(1, self.total_trades),
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "final_portfolio": self.portfolio_value
        }
        
        return metrics
        
    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Maximum drawdown as positive percentage
        """
        if not self.portfolio_values:
            return 0.0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdowns = (running_max - self.portfolio_values) / running_max
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered image or None
        """
        if mode == 'human':
            print(f"Step: {self.current_idx}, Price: {self.current_price:.2f}, "
                 f"Balance: {self.balance:.2f}, Position: {self.position:.6f}, "
                 f"Portfolio: {self.portfolio_value:.2f}")
            return None
        else:
            return None

# ---------------------------------
# QStar Learning Agent
# ---------------------------------

class QStarLearningAgent:
    """
    Advanced Q* Learning Agent with GPU acceleration for trading.
    
    This agent implements the Q* learning algorithm with various enhancements:
    - GPU-accelerated tensor operations
    - Experience replay with prioritized sampling
    - Adaptive learning rates
    - Quantum-inspired state representation
    - Performance metrics tracking
    """

    def __init__(self, states: int, actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 min_exploration_rate: float = 0.01,
                 exploration_decay_rate: float = 0.995,
                 use_adaptive_learning_rate: bool = True,
                 use_experience_replay: bool = True,
                 experience_buffer_size: int = 10000,
                 batch_size: int = 32,
                 max_episodes: int = 1000,
                 max_steps_per_episode: int = 10000,
                 use_quantum_representation: bool = False,
                 use_gpu: bool = None,
                 mixed_precision: bool = False):
        """
        Initialize the Q* learning agent.
        
        Args:
            states: Number of states
            actions: Number of actions
            learning_rate: Initial learning rate
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate
            min_exploration_rate: Minimum exploration rate
            exploration_decay_rate: Exploration rate decay per episode
            use_adaptive_learning_rate: Whether to use adaptive learning rate
            use_experience_replay: Whether to use experience replay
            experience_buffer_size: Size of experience replay buffer
            batch_size: Batch size for experience replay updates
            max_episodes: Maximum number of episodes for training
            max_steps_per_episode: Maximum steps per episode
            use_quantum_representation: Whether to use quantum-inspired representation
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
            mixed_precision: Whether to use mixed precision (FP16/BF16)
        """
        self.states = states
        self.actions = actions
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Advanced features
        self.use_adaptive_learning_rate = use_adaptive_learning_rate
        self.use_experience_replay = use_experience_replay
        self.batch_size = batch_size
        self.use_quantum_representation = use_quantum_representation
        
        # GPU configuration
        if use_gpu is None:
            # Auto-detect GPU availability
            self.use_gpu = is_gpu_available()
        else:
            self.use_gpu = use_gpu and is_gpu_available()
        
        # Mixed precision setting
        self.mixed_precision = mixed_precision and self.use_gpu and TORCH_AVAILABLE
        
        # Get tensor library
        self.tensor_lib = get_tensor_lib()

        # Training parameters
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # Initialize Q-table on appropriate device
        self._init_q_table()

        # Initialize experience replay buffer if enabled
        if use_experience_replay:
            self.experience_buffer = GPUExperienceBuffer(
                max_size=experience_buffer_size,
                alpha=0.6,
                use_gpu=self.use_gpu
            )
        else:
            self.experience_buffer = None

        # State for adaptive learning
        self.avg_q_change = 0.0
        self.episode_rewards = []
        self.convergence_history = []

        # Quantum representation (if enabled)
        if use_quantum_representation:
            if self.use_gpu:
                self.quantum_phases = to_tensor(
                    np.random.uniform(0, 2*np.pi, (states, actions)),
                    device='auto'
                )
            else:
                self.quantum_phases = np.random.uniform(0, 2*np.pi, (states, actions))
        else:
            self.quantum_phases = None

        # Metrics tracking
        self.episode_rewards = []
        self.q_value_changes = []
        self.exploration_rates = []
        self.learning_rates = []
        self.convergence_measures = []
        self.episode_lengths = []
        self.execution_times = []
        self.gpu_utilization = []

        logger.info(f"Initialized Q* Learning Agent with {states} states and {actions} actions")
        logger.info(f"Advanced features: adaptive_lr={use_adaptive_learning_rate}, "
                   f"experience_replay={use_experience_replay}, "
                   f"quantum_representation={use_quantum_representation}, "
                   f"use_gpu={self.use_gpu}, mixed_precision={self.mixed_precision}")
        
        # Report hardware configuration
        if self.use_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU acceleration with: {device_name}")
            elif JAX_AVAILABLE and len(jax.devices('gpu')) > 0:
                device_name = str(jax.devices('gpu')[0])
                logger.info(f"Using JAX GPU acceleration with: {device_name}")

    def _init_q_table(self):
        """Initialize Q-table on the appropriate device with proper data type."""
        # Determine the appropriate data type based on mixed precision setting
        if self.mixed_precision and self.tensor_lib is torch:
            dtype = torch.float16
        elif self.mixed_precision and self.tensor_lib is jnp:
            dtype = jnp.float16
        elif self.tensor_lib is torch:
            dtype = torch.float32
        elif self.tensor_lib is jnp:
            dtype = jnp.float32
        else:
            dtype = np.float32
            
        # Create the Q-table on the appropriate device
        if self.tensor_lib is torch:
            if self.use_gpu and torch.cuda.is_available():
                self.q_table = torch.zeros((self.states, self.actions), dtype=dtype, device='cuda')
            else:
                self.q_table = torch.zeros((self.states, self.actions), dtype=dtype)
        elif self.tensor_lib is jnp:
            if self.use_gpu and len(jax.devices('gpu')) > 0:
                self.q_table = jax.device_put(
                    jnp.zeros((self.states, self.actions), dtype=dtype),
                    jax.devices('gpu')[0]
                )
            else:
                self.q_table = jnp.zeros((self.states, self.actions), dtype=dtype)
        else:
            self.q_table = np.zeros((self.states, self.actions), dtype=dtype)
            
        logger.debug(f"Initialized Q-table with shape {self.q_table.shape}, "
                    f"on {'GPU' if self.use_gpu else 'CPU'}, "
                    f"using {self.tensor_lib.__name__ if hasattr(self.tensor_lib, '__name__') else 'numpy'}")

    def choose_action(self, state: int) -> int:
        """
        Choose an action based on the current state with exploration-exploitation balance.

        Args:
            state: Current state index

        Returns:
            Selected action index
        """
        # Ensure state is within bounds
        state = self._validate_state(state)

        # Exploration-exploitation tradeoff
        if np.random.random() < self.exploration_rate:
            # Exploration: choose random action
            return np.random.randint(0, self.actions)
        else:
            # Exploitation: choose best known action
            if self.use_quantum_representation:
                # Quantum-inspired action selection
                return self._quantum_action_selection(state)
            else:
                # Classical action selection: highest Q-value
                if self.tensor_lib is torch:
                    return torch.argmax(self.q_table[state, :]).item()
                elif self.tensor_lib is jnp:
                    return int(jnp.argmax(self.q_table[state, :]))
                else:
                    return np.argmax(self.q_table[state, :])

    def _quantum_action_selection(self, state: int) -> int:
        """
        Select action using quantum-inspired probabilistic approach with hardware acceleration.
        
        This method applies quantum-inspired principles to action selection by:
        1. Treating Q-values as probability amplitudes
        2. Applying phase shifts based on quantum_phases
        3. Calculating resulting probability distribution
        4. Sampling an action according to these probabilities
        
        The implementation automatically uses available hardware acceleration
        (GPU via PyTorch, PennyLane, or JAX) or falls back to NumPy.
        
        Args:
            state: Current state index
            
        Returns:
            Selected action index
        """
        # Get Q-values for current state
        q_values = self.q_table[state, :]
        
        # Convert to NumPy for consistent processing
        if TORCH_AVAILABLE and isinstance(q_values, torch.Tensor):
            q_values_np = q_values.cpu().detach().numpy()
        elif JAX_AVAILABLE and hasattr(q_values, 'device_buffer'):
            q_values_np = np.array(q_values)
        else:
            q_values_np = np.array(q_values)
        
        # Handle cases where all Q-values are the same
        if np.all(q_values_np == q_values_np[0]):
            return np.random.randint(0, len(q_values_np))
        
        # Normalize Q-values to create valid amplitudes (non-negative)
        q_min = np.min(q_values_np)
        if q_min < 0:
            q_values_np = q_values_np - q_min
        
        sum_q = np.sum(q_values_np)
        if sum_q <= 0:
            # If all Q-values are zero/negative, use uniform distribution
            return np.random.randint(0, len(q_values_np))
        
        # Create amplitude values (square root of probabilities)
        amplitudes = np.sqrt(q_values_np / sum_q)
        
        # Get phase values for this state
        if hasattr(self, 'quantum_phases') and self.quantum_phases is not None:
            if TORCH_AVAILABLE and isinstance(self.quantum_phases, torch.Tensor):
                phases = self.quantum_phases[state, :].cpu().detach().numpy()
            elif JAX_AVAILABLE and hasattr(self.quantum_phases, 'device_buffer'):
                phases = np.array(self.quantum_phases[state, :])
            else:
                phases = self.quantum_phases[state, :]
        else:
            # Generate random phases if not previously defined
            phases = np.random.uniform(0, 2*np.pi, len(q_values_np))
        
        # Try to use PennyLane for quantum simulation if available
        try:
            if PENNYLANE_AVAILABLE:
                import pennylane as qml
                num_actions = len(amplitudes)
                
                # Create appropriate device - try GPU first, fall back to CPU
                try:
                    # Try GPU device first
                    dev = qml.device("lightning.gpu", wires=num_actions)
                except Exception:
                    try:
                        # Then try kokkos for AMD GPU
                        dev = qml.device("lightning.kokkos", wires=num_actions)
                    except Exception:
                        # Fall back to CPU
                        dev = qml.device("lightning.qubit", wires=num_actions)
                
                # Define quantum circuit for probability calculation
                @qml.qnode(dev)
                def quantum_circuit():
                    # Prepare initial state with amplitudes
                    qml.AmplitudeEmbedding(amplitudes, wires=range(num_actions), normalize=True)
                    
                    # Apply phase shifts
                    for i in range(num_actions):
                        qml.PhaseShift(phases[i], wires=i)
                    
                    # Return measurement probabilities
                    return qml.probs(wires=range(num_actions))
                
                # Get probabilities and sample
                probs = quantum_circuit()
                probs = np.array(probs)
                return np.random.choice(len(probs), p=probs)
        
        except Exception as e:
            # Fall back to direct calculation if PennyLane fails
            pass
        
        # Direct calculation fallback using numpy or available tensor library
        try:
            # Calculate complex amplitudes
            complex_amplitudes = amplitudes * np.exp(1j * phases)
            
            # Calculate probabilities from complex amplitudes
            probs = np.abs(complex_amplitudes)**2
            probs = probs / np.sum(probs)  # Normalize
            
            # Sample action based on probabilities
            return np.random.choice(len(probs), p=probs)
        
        except Exception as e:
            # Ultimate fallback - just return action with highest Q-value
            self.logger.warning(f"Error in quantum action selection: {e}, falling back to greedy")
            return np.argmax(q_values_np)

    def learn(self, state: int, action: int, reward: float, next_state: int) -> float:
        """
        Update the Q-table based on experience.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: New state

        Returns:
            Q-value change magnitude
        """
        # Ensure states are within bounds
        state = self._validate_state(state)
        next_state = self._validate_state(next_state)

        # Store experience if using replay buffer
        if self.use_experience_replay:
            self.experience_buffer.add((state, action, reward, next_state))

        # For PyTorch
        if self.tensor_lib is torch:
            # Calculate current prediction and target Q-value
            predict = self.q_table[state, action].item()
            
            # Get max q-value for next state
            with torch.no_grad():
                next_q_values = self.q_table[next_state, :]
                max_next_q = torch.max(next_q_values).item()
                
            target = reward + self.discount_factor * max_next_q
            
            # Update Q-value with current learning rate
            self.q_table[state, action] += self.learning_rate * (target - predict)
            
        # For JAX
        elif self.tensor_lib is jnp:
            # JAX arrays are immutable, so we need to create a new array
            predict = float(self.q_table[state, action])
            max_next_q = float(jnp.max(self.q_table[next_state, :]))
            target = reward + self.discount_factor * max_next_q
            
            # Create updated Q-table
            update = jnp.zeros_like(self.q_table)
            update = update.at[state, action].set(self.learning_rate * (target - predict))
            self.q_table += update
            
        # For NumPy
        else:
            predict = self.q_table[state, action]
            target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
            
            # Update Q-value with current learning rate
            self.q_table[state, action] += self.learning_rate * (target - predict)

        # Return magnitude of update for monitoring
        return abs(target - predict)

    def replay_experiences(self) -> float:
        """
        Learn from stored experiences using prioritized replay.

        Returns:
            Average Q-value change magnitude
        """
        if not self.use_experience_replay or len(self.experience_buffer) < self.batch_size:
            return 0.0

        # Sample batch of experiences with priorities
        experiences, indices, weights = self.experience_buffer.sample(self.batch_size)

        total_change = 0.0
        td_errors = []

        # PyTorch version - more efficient batch processing
        if self.tensor_lib is torch and self.use_gpu:
            # Get batch tensors for efficient processing
            states, actions, rewards, next_states = self.experience_buffer.get_batch_tensors(indices)
            
            # Convert to tensors if not already
            if not torch.is_tensor(states):
                states = torch.tensor(states, device=self.q_table.device, dtype=torch.int64)
            if not torch.is_tensor(actions):
                actions = torch.tensor(actions, device=self.q_table.device, dtype=torch.int64)
            if not torch.is_tensor(rewards):
                rewards = torch.tensor(rewards, device=self.q_table.device, dtype=torch.float32)
            if not torch.is_tensor(next_states):
                next_states = torch.tensor(next_states, device=self.q_table.device, dtype=torch.int64)
            
            # Convert weights to tensor
            weights_tensor = torch.tensor(weights, device=self.q_table.device, dtype=torch.float32)
            
            # Get current Q values
            q_values = self.q_table[states, actions]
            
            # Calculate target Q values
            with torch.no_grad():
                next_q_values = torch.amax(self.q_table[next_states], dim=1)
                targets = rewards + self.discount_factor * next_q_values
                
            # Calculate TD errors
            td_error = targets - q_values
            
            # Apply importance sampling weights
            weighted_td_error = td_error * weights_tensor
            
            # Update Q values
            self.q_table[states, actions] += self.learning_rate * weighted_td_error
            
            # Convert TD errors to numpy for priority update
            td_errors_np = td_error.cpu().detach().numpy()
            total_change = float(torch.sum(torch.abs(td_error)).item())
            
            # Update priorities
            self.experience_buffer.update_priorities(indices, td_errors_np)
            
        # JAX version - functional approach
        elif self.tensor_lib is jnp and self.use_gpu:
            try:
                # Get batch tensors
                states, actions, rewards, next_states = self.experience_buffer.get_batch_tensors(indices)
                
                # Convert to JAX arrays
                states = jnp.array(states, dtype=jnp.int32)
                actions = jnp.array(actions, dtype=jnp.int32)
                rewards = jnp.array(rewards, dtype=jnp.float32)
                next_states = jnp.array(next_states, dtype=jnp.int32)
                weights_jax = jnp.array(weights, dtype=jnp.float32)
                
                # Get current Q values (functional indexing)
                q_values = self.q_table[states, actions]
                
                # Calculate target Q values
                next_q_max = jnp.max(self.q_table[next_states], axis=1)
                targets = rewards + self.discount_factor * next_q_max
                
                # Calculate TD errors
                td_error = targets - q_values
                weighted_td_error = td_error * weights_jax
                
                # Create update array
                update = jnp.zeros_like(self.q_table)
                for i in range(len(states)):
                    update = update.at[states[i], actions[i]].set(
                        self.learning_rate * weighted_td_error[i]
                    )
                
                # Apply update
                self.q_table += update
                
                # Process TD errors
                td_errors_np = np.array(td_error)
                total_change = float(jnp.sum(jnp.abs(td_error)))
                
                # Update priorities
                self.experience_buffer.update_priorities(indices, td_errors_np)
                
            except Exception as e:
                logger.error(f"Error in JAX experience replay: {e}")
                # Fallback to standard processing
                for i, (state, action, reward, next_state) in enumerate(experiences):
                    predict = float(self.q_table[state, action])
                    target = reward + self.discount_factor * float(jnp.max(self.q_table[next_state, :]))
                    td_error = target - predict
                    td_errors.append(td_error)
                    weighted_update = self.learning_rate * td_error * weights[i]
                    
                    # Update Q-value
                    update = jnp.zeros_like(self.q_table)
                    update = update.at[state, action].set(weighted_update)
                    self.q_table += update
                    
                    total_change += abs(td_error)
                
                # Update priorities
                self.experience_buffer.update_priorities(indices, td_errors)
                
        # Standard NumPy/CPU processing
        else:
            # Learn from each experience in batch
            for i, (state, action, reward, next_state) in enumerate(experiences):
                # Calculate current prediction and target
                predict = self.q_table[state, action]
                target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
                
                # Get TD error for prioritization
                td_error = target - predict
                td_errors.append(td_error)
                
                # Apply importance sampling weight to update
                weighted_update = self.learning_rate * td_error * weights[i]
                self.q_table[state, action] += weighted_update
                
                total_change += abs(td_error)
                
            # Update priorities based on new TD errors
            self.experience_buffer.update_priorities(indices, td_errors)

        return total_change / self.batch_size

    def update_exploration_rate(self) -> None:
        """Decrease exploration rate according to decay schedule."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay_rate
        )

    def update_learning_rate(self, episode: int, avg_q_change: float) -> None:
        """
        Update learning rate adaptively based on progress.

        Args:
            episode: Current episode number
            avg_q_change: Average Q-value change magnitude
        """
        if not self.use_adaptive_learning_rate:
            return

        # Adaptive learning rate strategies
        if avg_q_change < 0.01:
            # Small Q-value changes might indicate convergence or getting stuck
            # Increase learning rate slightly to escape local minima
            self.learning_rate = min(0.5, self.learning_rate * 1.05)
        elif avg_q_change > 0.5:
            # Large Q-value changes might indicate instability
            # Decrease learning rate to stabilize learning
            self.learning_rate = max(0.01, self.learning_rate * 0.95)
        else:
            # Schedule-based decay
            self.learning_rate = self.initial_learning_rate * (1.0 - episode / self.max_episodes)

    def has_converged(self, threshold: float = 0.005, window_size: int = 100) -> Tuple[bool, float]:
        """
        Check if the Q-values have converged.

        Args:
            threshold: Convergence threshold for Q-value changes
            window_size: Window size for convergence check

        Returns:
            Tuple of (has_converged, convergence_measure)
        """
        # Check if we have enough history
        if len(self.convergence_history) < window_size:
            return False, 1.0

        # Calculate average change over recent episodes
        recent_changes = np.mean(self.convergence_history[-window_size:])

        # Compare to threshold
        converged = recent_changes < threshold

        return converged, recent_changes

    def _validate_state(self, state: int) -> int:
        """
        Ensure state is within bounds of Q-table.

        Args:
            state: State index

        Returns:
            Valid state index
        """
        return max(0, min(state, self.states - 1))

    def resize_q_table(self, new_states: int, new_actions: int) -> None:
        """
        Resize the Q-table and quantum phases array for expanded state/action spaces.

        Args:
            new_states: New number of states
            new_actions: New number of actions
        """
        # Store old dimensions
        old_states = self.states
        old_actions = self.actions

        if new_states <= old_states and new_actions <= old_actions:
            logger.warning(f"Skipping resize: New dimensions ({new_states}, {new_actions}) not larger than current ({old_states}, {old_actions})")
            return

        logger.info(f"Resizing Q-table from ({old_states}, {old_actions}) to ({new_states}, {new_actions})")

        # PyTorch resizing
        if self.tensor_lib is torch:
            device = self.q_table.device
            dtype = self.q_table.dtype
            
            # Create new tensor
            new_q_table = torch.zeros((new_states, new_actions), dtype=dtype, device=device)
            
            # Copy old values
            new_q_table[:old_states, :old_actions] = self.q_table
            
            # Replace old table
            self.q_table = new_q_table
            
            # Resize quantum phases if used
            if self.use_quantum_representation:
                new_phases = torch.rand((new_states, new_actions), device=device) * (2 * np.pi)
                if self.quantum_phases is not None:
                    new_phases[:old_states, :old_actions] = self.quantum_phases
                self.quantum_phases = new_phases
                
        # JAX resizing
        elif self.tensor_lib is jnp:
            # JAX arrays are immutable, need to create new array and copy values
            if self.use_gpu and len(jax.devices('gpu')) > 0:
                device = jax.devices('gpu')[0]
            else:
                device = jax.devices('cpu')[0]
                
            # Create new array
            new_q_table = jnp.zeros((new_states, new_actions), dtype=self.q_table.dtype)
            
            # Copy old values
            new_q_table = new_q_table.at[:old_states, :old_actions].set(self.q_table)
            
            # Replace old table
            self.q_table = jax.device_put(new_q_table, device)
            
            # Resize quantum phases if used
            if self.use_quantum_representation:
                new_phases = jnp.random.uniform(0, 2*np.pi, (new_states, new_actions))
                if self.quantum_phases is not None:
                    new_phases = new_phases.at[:old_states, :old_actions].set(self.quantum_phases)
                self.quantum_phases = jax.device_put(new_phases, device)
                
        # NumPy resizing
        else:
            # Resize Q-table
            new_q_table = np.zeros((new_states, new_actions), dtype=self.q_table.dtype)
            new_q_table[:old_states, :old_actions] = self.q_table
            self.q_table = new_q_table

            # Resize quantum phases if used
            if self.use_quantum_representation:
                new_phases = np.random.uniform(0, 2*np.pi, (new_states, new_actions))
                if self.quantum_phases is not None:
                    new_phases[:old_states, :old_actions] = self.quantum_phases
                self.quantum_phases = new_phases

        # Update state/action counts
        self.states = new_states
        self.actions = new_actions

        logger.info(f"Q-table resized to {self.q_table.shape}")
        if self.use_quantum_representation:
            logger.info(f"Quantum phases resized to {self.quantum_phases.shape}")

    def train(self, environment: TradingEnvironment) -> Tuple[bool, int]:
        """
        Train the agent in the provided environment.

        Args:
            environment: Trading environment to train in

        Returns:
            Tuple of (converged, episodes_used)
        """
        if not isinstance(environment, TradingEnvironment):
            logger.error("Environment must be a TradingEnvironment instance")
            return False, 0

        logger.info(f"Starting training for up to {self.max_episodes} episodes")

        # Initialize metrics
        episodes_completed = 0
        total_q_changes = []

        try:
            # For each episode
            for episode in range(self.max_episodes):
                # Reset environment for new episode
                state = environment.reset()
                episode_reward = 0
                episode_steps = 0
                episode_q_changes = []
                
                # Track execution time
                start_time = time.time()

                # Reset quantum phases slightly for this episode (if using quantum representation)
                if self.use_quantum_representation and episode % 10 == 0:
                    if self.tensor_lib is torch:
                        # Add small random perturbations to phases
                        noise = torch.rand_like(self.quantum_phases) * 0.2 - 0.1
                        self.quantum_phases += noise
                    elif self.tensor_lib is jnp:
                        # JAX version - create new array with noise
                        noise = jnp.random.uniform(-0.1, 0.1, self.quantum_phases.shape)
                        self.quantum_phases += noise
                    else:
                        # NumPy version
                        self.quantum_phases += np.random.uniform(-0.1, 0.1, (self.states, self.actions))

                # Episode loop
                for step in range(self.max_steps_per_episode):
                    # Choose and take action
                    action = self.choose_action(state)
                    next_state, reward, done = environment.step(action)

                    # Learn from this experience
                    q_change = self.learn(state, action, reward, next_state)
                    episode_q_changes.append(q_change)

                    # Accumulate reward
                    episode_reward += reward
                    episode_steps += 1

                    # Move to next state
                    state = next_state

                    # Check if episode is done
                    if done:
                        break

                # After episode actions

                # Learn from experience replay
                if self.use_experience_replay and len(self.experience_buffer) >= self.batch_size:
                    replay_q_change = self.replay_experiences()
                    episode_q_changes.append(replay_q_change)

                # Update rates
                self.update_exploration_rate()

                # Calculate average Q-change for this episode
                avg_q_change = np.mean(episode_q_changes) if episode_q_changes else 0
                self.convergence_history.append(avg_q_change)

                # Update learning rate adaptively
                self.update_learning_rate(episode, avg_q_change)

                # Store metrics
                self.episode_rewards.append(episode_reward)
                self.q_value_changes.append(avg_q_change)
                self.exploration_rates.append(self.exploration_rate)
                self.learning_rates.append(self.learning_rate)
                
                # Calculate episode timing
                execution_time = time.time() - start_time
                self.execution_times.append(execution_time)
                
                # Get GPU utilization if available
                gpu_utilization = None
                if self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        # This is not available on all platforms/CUDA versions
                        # gpu_utilization = torch.cuda.utilization()
                        gpu_utilization = 0  # Placeholder
                    except:
                        pass
                self.gpu_utilization.append(gpu_utilization if gpu_utilization is not None else 0)

                # Check for convergence
                converged, convergence_value = self.has_converged()
                self.convergence_measures.append(convergence_value)
                self.episode_lengths.append(episode_steps)

                # Log progress
                if episode % 10 == 0:
                    logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, "
                               f"Steps={episode_steps}, Exploration={self.exploration_rate:.4f}, "
                               f"Convergence={convergence_value:.6f}")

                # Check for convergence
                if converged:
                    logger.info(f"Converged after {episode+1} episodes")
                    episodes_completed = episode + 1
                    break
                    
                # Force garbage collection every few episodes if using GPU to prevent memory leaks
                if self.use_gpu and episode % 50 == 0:
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            if episodes_completed == 0:
                episodes_completed = self.max_episodes
                logger.info(f"Maximum episodes ({self.max_episodes}) reached without convergence")

            # Log training summary
            logger.info(f"Training completed: {episodes_completed}/{self.max_episodes} episodes")
            self._log_training_summary()
            
            # Final GPU cleanup
            if self.use_gpu:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Return convergence status and episodes used
            return converged, episodes_completed
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            return False, episode if 'episode' in locals() else 0

    def _log_training_summary(self):
        """Log training summary statistics."""
        if not self.episode_rewards:
            logger.info("No training data to summarize")
            return
            
        summary = {
            "avg_reward": np.mean(self.episode_rewards[-100:]),
            "max_reward": np.max(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "avg_episode_length": np.mean(self.episode_lengths[-100:]),
            "final_convergence": self.convergence_measures[-1] if self.convergence_measures else None,
            "final_exploration_rate": self.exploration_rates[-1] if self.exploration_rates else None,
            "total_episodes": len(self.episode_rewards)
        }
        
        # Add performance metrics if available
        if self.execution_times:
            summary["avg_execution_time"] = np.mean(self.execution_times[-100:])
            summary["min_execution_time"] = np.min(self.execution_times)
            summary["max_execution_time"] = np.max(self.execution_times)
            
        if any(u > 0 for u in self.gpu_utilization):
            summary["avg_gpu_utilization"] = np.mean([u for u in self.gpu_utilization if u > 0])
            
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")

    def evaluate(self, environment: TradingEnvironment, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the agent in the environment without learning.

        Args:
            environment: Environment to evaluate in
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")

        rewards = []
        episode_lengths = []
        portfolio_returns = []
        execution_times = []

        # Save original exploration rate and set to 0 for evaluation
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0

        try:
            for episode in range(num_episodes):
                state = environment.reset()
                episode_reward = 0
                episode_steps = 0
                
                # Track execution time
                start_time = time.time()

                for step in range(self.max_steps_per_episode):
                    # Render if requested
                    if render:
                        environment.render()

                    # Choose and take action (no learning)
                    action = self.choose_action(state)
                    next_state, reward, done = environment.step(action)

                    # Accumulate reward
                    episode_reward += reward
                    episode_steps += 1

                    # Move to next state
                    state = next_state

                    # Check if episode is done
                    if done:
                        break
                        
                # Record execution time
                execution_time = time.time() - start_time
                execution_times.append(execution_time)

                # Record episode results
                rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                
                # Get portfolio metrics
                metrics = environment.get_performance_metrics()
                portfolio_returns.append(metrics["total_return"])
                
                logger.info(f"Evaluation Episode {episode+1}: Return={metrics['total_return']:.4f}, "
                           f"Reward={episode_reward:.4f}, Steps={episode_steps}")

            # Restore exploration rate
            self.exploration_rate = original_exploration_rate

            # Calculate evaluation metrics
            eval_metrics = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_episode_length": np.mean(episode_lengths),
                "mean_portfolio_return": np.mean(portfolio_returns),
                "max_portfolio_return": np.max(portfolio_returns),
                "min_portfolio_return": np.min(portfolio_returns),
                "mean_execution_time": np.mean(execution_times)
            }

            logger.info(f"Evaluation results: Mean portfolio return: {eval_metrics['mean_portfolio_return']:.4f}")
            
            return eval_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            self.exploration_rate = original_exploration_rate
            return {"error": str(e)}

    def save(self, filepath: str) -> None:
        """
        Save the agent to a file.

        Args:
            filepath: Path to save the agent
        """
        # Convert tensors to numpy arrays for serialization
        if self.tensor_lib is torch:
            q_table_np = self.q_table.cpu().detach().numpy()
            quantum_phases_np = self.quantum_phases.cpu().detach().numpy() if self.quantum_phases is not None else None
        elif self.tensor_lib is jnp:
            q_table_np = np.array(self.q_table)
            quantum_phases_np = np.array(self.quantum_phases) if self.quantum_phases is not None else None
        else:
            q_table_np = self.q_table
            quantum_phases_np = self.quantum_phases
            
        agent_data = {
            'q_table': q_table_np,
            'states': self.states,
            'actions': self.actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'min_exploration_rate': self.min_exploration_rate,
            'exploration_decay_rate': self.exploration_decay_rate,
            'use_adaptive_learning_rate': self.use_adaptive_learning_rate,
            'use_experience_replay': self.use_experience_replay,
            'use_quantum_representation': self.use_quantum_representation,
            'quantum_phases': quantum_phases_np,
            
            # Training metrics
            'episode_rewards': self.episode_rewards,
            'q_value_changes': self.q_value_changes,
            'exploration_rates': self.exploration_rates,
            'learning_rates': self.learning_rates,
            'convergence_measures': self.convergence_measures,
            'episode_lengths': self.episode_lengths,
            'execution_times': self.execution_times,
            'gpu_utilization': self.gpu_utilization,
            
            # Configuration
            'timestamp': datetime.now().isoformat(),
            'use_gpu': self.use_gpu,
            'mixed_precision': self.mixed_precision
        }

        try:
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(filepath, 'wb') as f:
                pickle.dump(agent_data, f)

            logger.info(f"Agent saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent: {str(e)}", exc_info=True)

    @classmethod
    def load(cls, filepath: str, use_gpu: bool = None) -> 'QStarLearningAgent':
        """
        Load an agent from a file.
    
        Args:
            filepath: Path to load the agent from
            use_gpu: Whether to use GPU acceleration, overriding saved setting
    
        Returns:
            Loaded agent
        """
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
                
            # Check if GPU setting was provided
            if use_gpu is None:
                # Use saved setting if available, otherwise auto-detect
                use_gpu = agent_data.get('use_gpu', None)
                
            # Create agent with saved configuration
            agent = cls(
                states=agent_data['states'],
                actions=agent_data['actions'],
                learning_rate=agent_data['learning_rate'],
                discount_factor=agent_data['discount_factor'],
                exploration_rate=agent_data['exploration_rate'],
                min_exploration_rate=agent_data['min_exploration_rate'],
                exploration_decay_rate=agent_data['exploration_decay_rate'],
                use_adaptive_learning_rate=agent_data['use_adaptive_learning_rate'],
                use_experience_replay=agent_data['use_experience_replay'],
                use_quantum_representation=agent_data['use_quantum_representation'],
                use_gpu=use_gpu,
                mixed_precision=agent_data.get('mixed_precision', False)
            )
    
            # Restore Q-table to appropriate device
            q_table_np = agent_data['q_table']
            if agent.use_gpu:
                if agent.tensor_lib is torch:
                    agent.q_table = torch.tensor(
                        q_table_np, 
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        dtype=torch.float16 if agent.mixed_precision else torch.float32
                    )
                elif agent.tensor_lib is jnp:
                    if len(jax.devices('gpu')) > 0:
                        agent.q_table = jax.device_put(
                            jnp.array(q_table_np, dtype=jnp.float16 if agent.mixed_precision else jnp.float32),
                            jax.devices('gpu')[0]
                        )
                    else:
                        agent.q_table = jnp.array(q_table_np)
            else:
                agent.q_table = q_table_np
    
            # Restore quantum phases if using quantum representation
            if agent.use_quantum_representation and 'quantum_phases' in agent_data:
                quantum_phases_np = agent_data['quantum_phases']
                if quantum_phases_np is not None:
                    if agent.use_gpu:
                        if agent.tensor_lib is torch:
                            agent.quantum_phases = torch.tensor(
                                quantum_phases_np, 
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                        elif agent.tensor_lib is jnp:
                            if len(jax.devices('gpu')) > 0:
                                agent.quantum_phases = jax.device_put(
                                    jnp.array(quantum_phases_np),
                                    jax.devices('gpu')[0]
                                )
                            else:
                                agent.quantum_phases = jnp.array(quantum_phases_np)
                    else:
                        agent.quantum_phases = quantum_phases_np
    
            # Restore training metrics if available
            for metric_name in ['episode_rewards', 'q_value_changes', 'exploration_rates',
                               'learning_rates', 'convergence_measures', 'episode_lengths',
                               'execution_times', 'gpu_utilization']:
                if metric_name in agent_data:
                    setattr(agent, metric_name, agent_data[metric_name])
    
            # Restore history for convergence checking
            agent.convergence_history = agent_data.get('convergence_measures', [])
    
            logger.info(f"Agent loaded from {filepath} with GPU={agent.use_gpu}")
            return agent
            
        except Exception as e:
            logger.error(f"Error loading agent: {str(e)}", exc_info=True)
            raise

    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot training metrics for analysis.
        
        Args:
            save_path: Path to save the plot or None to display
        """
        if not self.episode_rewards:
            logger.warning("No training metrics to plot")
            return
            
        # Determine number of plots based on available metrics
        has_perf_metrics = len(self.execution_times) > 0 or any(g > 0 for g in self.gpu_utilization)
        num_rows = 4 if has_perf_metrics else 3
        
        fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 4))
        
        # Rewards plot
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True)
        
        # Q-value changes
        axes[1, 0].plot(self.q_value_changes)
        axes[1, 0].set_title('Q-Value Changes')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Avg Change')
        axes[1, 0].grid(True)
        
        # Exploration rate
        axes[1, 1].plot(self.exploration_rates)
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].grid(True)
        
        # Learning rate
        axes[2, 0].plot(self.learning_rates)
        axes[2, 0].set_title('Learning Rate')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Rate')
        axes[2, 0].grid(True)
        
        # Convergence
        axes[2, 1].plot(self.convergence_measures)
        axes[2, 1].set_title('Convergence Measure')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Value')
        axes[2, 1].grid(True)
        
        # Performance metrics (if available)
        if has_perf_metrics:
            if self.execution_times:
                axes[3, 0].plot(self.execution_times)
                axes[3, 0].set_title('Execution Time per Episode')
                axes[3, 0].set_xlabel('Episode')
                axes[3, 0].set_ylabel('Time (s)')
                axes[3, 0].grid(True)
            
            if any(g > 0 for g in self.gpu_utilization):
                axes[3, 1].plot([g for g in self.gpu_utilization if g > 0])
                axes[3, 1].set_title('GPU Utilization')
                axes[3, 1].set_xlabel('Episode')
                axes[3, 1].set_ylabel('Utilization (%)')
                axes[3, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Metrics plot saved to {save_path}")
        else:
            plt.show()

    def plot_backtest_results(self, environment: TradingEnvironment, save_path: Optional[str] = None):
        """
        Plot backtest results for the agent in a trading environment.
        
        Args:
            environment: Environment to run backtest in
            save_path: Path to save the plot or None to display
        """
        if not isinstance(environment, TradingEnvironment):
            logger.error("Environment must be a TradingEnvironment instance")
            return
            
        # Save original exploration rate
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0
        
        try:
            # Run a single episode to collect data
            state = environment.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = environment.step(action)
                state = next_state
            
            # Get performance metrics from environment
            metrics = environment.get_performance_metrics()
            
            # Get backtest data
            portfolio_values = environment.portfolio_values
            positions = environment.positions
            actions = environment.actions_taken
            
            # Create figure
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Create x-axis (time steps)
            time_steps = list(range(len(portfolio_values)))
            
            # Plot portfolio value
            axes[0].plot(time_steps, portfolio_values, label='Portfolio Value')
            axes[0].set_title(f'Portfolio Value (Return: {metrics["total_return"]:.2%})')
            axes[0].set_ylabel('Value ($)')
            axes[0].grid(True)
            axes[0].legend()
            
            # Add buy/sell markers
            for i, action in enumerate(actions):
                if action == TradingAction.BUY:
                    axes[0].scatter(i, portfolio_values[i], color='green', marker='^', alpha=0.7)
                elif action == TradingAction.SELL:
                    axes[0].scatter(i, portfolio_values[i], color='red', marker='v', alpha=0.7)
                elif action == TradingAction.INCREASE:
                    axes[0].scatter(i, portfolio_values[i], color='lime', marker='^', alpha=0.5)
                elif action == TradingAction.REDUCE:
                    axes[0].scatter(i, portfolio_values[i], color='orange', marker='v', alpha=0.5)
            
            # Plot positions
            axes[1].plot(time_steps, positions, label='Position Size')
            axes[1].set_title('Position Size')
            axes[1].set_ylabel('Units')
            axes[1].grid(True)
            axes[1].legend()
            
            # Plot actions as color-coded areas
            cmap = plt.cm.get_cmap('viridis', TradingAction.get_num_actions())
            for i, action in enumerate(actions):
                color = cmap(action)
                axes[2].axvspan(i, i+1, facecolor=color, alpha=0.3)
            
            # Add action legend
            action_colors = [cmap(i) for i in range(TradingAction.get_num_actions())]
            action_labels = [TradingAction.get_action_name(i) for i in range(TradingAction.get_num_actions())]
            action_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3) for color in action_colors]
            axes[2].legend(action_patches, action_labels, loc='upper right')
            
            axes[2].set_title('Trading Actions')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Action')
            axes[2].set_yticks([])
            
            # Add performance metrics as text
            metrics_text = (
                f"Total Return: {metrics['total_return']:.2%}\n"
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
                f"Win Rate: {metrics['win_rate']:.2%}\n"
                f"Total Trades: {metrics['total_trades']}"
            )
            
            plt.figtext(0.01, 0.01, metrics_text, fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Backtest plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting backtest results: {str(e)}", exc_info=True)
            
        finally:
            # Restore exploration rate
            self.exploration_rate = original_exploration_rate

# ---------------------------------
# Agent Refinement
# ---------------------------------

def refine_agent(agent: QStarLearningAgent, environment: TradingEnvironment,
                max_refinements: int = 3, evaluation_episodes: int = 10) -> QStarLearningAgent:
    """
    Refine the agent by progressively expanding state-action space and training.
    
    Args:
        agent: Agent to refine
        environment: Environment to train in
        max_refinements: Maximum number of refinement steps
        evaluation_episodes: Number of episodes for evaluation between refinements
        
    Returns:
        Refined agent
    """
    logger.info(f"Beginning refinement process with up to {max_refinements} steps")

    best_performance = float('-inf')
    best_agent_path = None
    refinement_count = 0
    
    for ref_step in range(max_refinements):
        logger.info(f"--- Refinement Step {ref_step + 1}/{max_refinements} ---")
        
        # Adjust training parameters for refinement phase
        agent.max_episodes += 2000  # Add more episodes per refinement
        agent.max_steps_per_episode += 50
        agent.learning_rate = max(0.01, agent.learning_rate * 0.9)  # Slightly decrease LR
        agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * 0.9)  # Decrease exploration faster

        # Train the agent further
        logger.info(f"Refinement Training (Max Ep: {agent.max_episodes})...")
        converged, episodes = agent.train(environment)

        # Evaluate the agent
        logger.info(f"Refinement Evaluation ({evaluation_episodes} episodes)...")
        eval_results = agent.evaluate(environment, num_episodes=evaluation_episodes)
        current_performance = eval_results['mean_portfolio_return']

        logger.info(f"Refinement Step {ref_step + 1}: Performance={current_performance:.4f}, Best={best_performance:.4f}, Episodes={episodes}")

        # Check if this is the best performing agent so far
        if current_performance > best_performance:
            best_performance = current_performance
            # Save the current best agent state
            try:
                if best_agent_path:  # Remove previous best temp file
                    if os.path.exists(best_agent_path):
                        os.remove(best_agent_path)
                best_agent_path = f"temp_refined_agent_{ref_step+1}.pkl"
                agent.save(best_agent_path)
                logger.info(f"Saved new best agent state to {best_agent_path}")
            except Exception as e_save:
                logger.error(f"Error saving temporary refined agent state: {e_save}")
                best_agent_path = None  # Reset if save failed
                
        # Force garbage collection
        if agent.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Load the best agent state found during refinement
    if best_agent_path and os.path.exists(best_agent_path):
        logger.info(f"Loading best agent found during refinement from: {best_agent_path}")
        try:
            # Use the agent's load method
            best_agent = QStarLearningAgent.load(best_agent_path)
            
            # Clean up ALL temp files
            for i in range(max_refinements):
                temp_path = f"temp_refined_agent_{i+1}.pkl"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            return best_agent
            
        except Exception as e_load:
            logger.error(f"Error loading best refined agent state: {e_load}. Returning last agent state.")
            # Fallback: clean up any temp file we created but couldn't load
            if os.path.exists(best_agent_path):
                os.remove(best_agent_path)
            return agent
    else:
        logger.warning("No better agent state saved during refinement. Returning last agent state.")
        # Clean up any temp files that might exist
        for i in range(max_refinements):
            temp_path = f"temp_refined_agent_{i+1}.pkl"
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return agent

# ---------------------------------
# Training Functions
# ---------------------------------

def train_agent(pair: str, timeframe: str, exchange: str = 'binance',
               use_gpu: bool = None, epochs: int = 100, batch_size: int = 64,
               initial_balance: float = 10000.0, transaction_fee: float = 0.001,
               use_quantum: bool = True, refine: bool = True,
               max_refinements: int = 3, save_path: Optional[str] = None) -> QStarLearningAgent:
    """
    Train a QStar agent on historical data for a trading pair.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        timeframe: Timeframe (e.g., '1h')
        exchange: Exchange name
        use_gpu: Whether to use GPU (None for auto-detect)
        epochs: Number of training epochs
        batch_size: Batch size for experience replay
        initial_balance: Initial balance for trading environment
        transaction_fee: Transaction fee rate
        use_quantum: Whether to use quantum representation
        refine: Whether to refine the agent after training
        max_refinements: Maximum number of refinement steps
        save_path: Path to save the trained agent
        
    Returns:
        Trained QStar agent
    """
    logger.info(f"Training agent for {pair} on {timeframe} timeframe")
    
    # Load historical data
    df = load_freqtrade_data(pair, timeframe, exchange)
    if df.empty:
        logger.error(f"No data available for {pair} on {timeframe}")
        return None
        
    # Calculate indicators
    df_indicators = calculate_indicators(df)
    
    # Create trading environment
    env = TradingEnvironment(
        price_data=df_indicators,
        window_size=50,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )
    
    # Initialize agent
    agent = QStarLearningAgent(
        states=env.num_states,
        actions=env.num_actions,
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=1.0,
        min_exploration_rate=0.05,
        exploration_decay_rate=0.997,
        use_adaptive_learning_rate=True,
        use_experience_replay=True,
        experience_buffer_size=50000,
        batch_size=batch_size,
        max_episodes=epochs,
        max_steps_per_episode=len(df) - 50,  # Max steps is data length - window size
        use_quantum_representation=use_quantum,
        use_gpu=use_gpu
    )
    
    # Train the agent
    logger.info(f"Starting agent training for {epochs} episodes...")
    start_time = time.time()
    converged, episodes_completed = agent.train(env)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds. Episodes: {episodes_completed}")
    
    # Generate plots
    plots_dir = PATHS["models_dir"] / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_plot_path = plots_dir / f"{pair.replace('/', '_')}_{timeframe}_metrics.png"
    agent.plot_metrics(str(metrics_plot_path))
    
    backtest_plot_path = plots_dir / f"{pair.replace('/', '_')}_{timeframe}_backtest.png"
    agent.plot_backtest_results(env, str(backtest_plot_path))
    
    # Refine agent if requested
    if refine and episodes_completed >= 10:  # Only refine if we've had some training
        logger.info("Refining agent...")
        agent = refine_agent(agent, env, max_refinements=max_refinements)
        
        # Generate plots after refinement
        refined_metrics_path = plots_dir / f"{pair.replace('/', '_')}_{timeframe}_refined_metrics.png"
        agent.plot_metrics(str(refined_metrics_path))
        
        refined_backtest_path = plots_dir / f"{pair.replace('/', '_')}_{timeframe}_refined_backtest.png"
        agent.plot_backtest_results(env, str(refined_backtest_path))
    
    # Save agent if path provided
    if save_path:
        agent.save(save_path)
        logger.info(f"Agent saved to {save_path}")
    
    return agent

def evaluate_agent(agent: QStarLearningAgent, pair: str, timeframe: str,
                  exchange: str = 'binance', initial_balance: float = 10000.0,
                  transaction_fee: float = 0.001, episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate a trained agent on historical data.
    
    Args:
        agent: Trained agent
        pair: Trading pair
        timeframe: Timeframe
        exchange: Exchange name
        initial_balance: Initial balance
        transaction_fee: Transaction fee rate
        episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Evaluating agent for {pair} on {timeframe} timeframe")
    
    # Load historical data
    df = load_freqtrade_data(pair, timeframe, exchange)
    if df.empty:
        logger.error(f"No data available for {pair} on {timeframe}")
        return {}
        
    # Calculate indicators
    df_indicators = calculate_indicators(df)
    
    # Create trading environment
    env = TradingEnvironment(
        price_data=df_indicators,
        window_size=50,
        initial_balance=initial_balance,
        transaction_fee=transaction_fee
    )
    
    # Evaluate agent
    metrics = agent.evaluate(env, num_episodes=episodes)
    
    # Generate backtest plot
    plots_dir = PATHS["models_dir"] / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    backtest_plot_path = plots_dir / f"{pair.replace('/', '_')}_{timeframe}_evaluation.png"
    agent.plot_backtest_results(env, str(backtest_plot_path))
    
    return metrics

def run_training(pairs: List[str] = None, timeframes: List[str] = None,
                exchange: str = 'binance', epochs: int = 100, use_gpu: bool = None,
                refined: bool = True) -> Dict[str, Any]:
    """
    Run training on multiple pairs and timeframes.
    
    Args:
        pairs: List of trading pairs (None for auto-detect)
        timeframes: List of timeframes (None for auto-detect)
        exchange: Exchange name
        epochs: Number of training epochs
        use_gpu: Whether to use GPU
        refined: Whether to refine agents
        
    Returns:
        Dictionary of trained agents and metrics
    """
    # Auto-detect available data if not specified
    available_data = find_available_data(exchange)
    
    if not available_data:
        logger.error(f"No data found for exchange {exchange}")
        return {}
    
    if pairs is None:
        pairs = list(available_data.keys())
        
    if not pairs:
        logger.error("No pairs specified or detected")
        return {}
        
    # For each pair, use available timeframes if not specified
    results = {}
    
    for pair in pairs:
        if pair not in available_data:
            logger.warning(f"No data available for {pair}, skipping")
            continue
            
        pair_timeframes = timeframes if timeframes else available_data[pair]
        
        for tf in pair_timeframes:
            if tf not in available_data[pair]:
                logger.warning(f"No data available for {pair} on {tf} timeframe, skipping")
                continue
                
            # Define save path
            save_filename = f"qstar_agent_{pair.replace('/', '_')}_{tf}.pkl"
            save_path = PATHS["models_dir"] / save_filename
            
            # Train agent
            logger.info(f"Training agent for {pair} on {tf} timeframe")
            try:
                agent = train_agent(
                    pair=pair,
                    timeframe=tf,
                    exchange=exchange,
                    use_gpu=use_gpu,
                    epochs=epochs,
                    refine=refined,
                    save_path=str(save_path)
                )
                
                if agent:
                    # Save results
                    key = f"{pair}_{tf}"
                    results[key] = {
                        'agent': agent,
                        'agent_path': str(save_path)
                    }
                
            except Exception as e:
                logger.error(f"Error training agent for {pair} on {tf}: {str(e)}", exc_info=True)
    
    return results

# ---------------------------------
# Example Usage
# ---------------------------------

def main():
    """Main function for running the QStar Trading system."""
    print("QStar Trading System with Freqtrade Integration")
    print("=" * 50)
    
    # Check for available hardware
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {gpu_name}")
    elif JAX_AVAILABLE:
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) > 0:
                print(f"GPU available: {gpu_devices[0]}")
            else:
                print("No GPU detected with JAX")
        except RuntimeError:
            print("JAX GPU backends not available, using CPU")
            try:
                all_devices = jax.devices()
                print(f"Available JAX devices: {all_devices}")
            except Exception:
                pass
    else:
        print("No GPU detected, using CPU")
    
    # Show available data
    print("\nChecking for available data...")
    available_data = find_available_data()
    
    if not available_data:
        print("No data found. Please download data using:")
        print("freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT --timeframes 1h 4h 1d")
        return
    
    print("\nAvailable data:")
    for pair, timeframes in available_data.items():
        print(f"  {pair}: {', '.join(timeframes)}")
    
    # Select pair and timeframe
    if len(available_data) > 0:
        default_pair = list(available_data.keys())[0]
        default_tf = available_data[default_pair][0] if available_data[default_pair] else None
    else:
        print("No data available.")
        return
    
    # Ask user for training parameters
    print("\nTraining Configuration:")
    print("-" * 50)
    
    pair_input = input(f"Trading pair ({default_pair}): ").strip()
    pair = pair_input if pair_input else default_pair
    
    available_tfs = available_data.get(pair, [])
    if not available_tfs:
        print(f"No data available for {pair}")
        return
    
    default_tf = available_tfs[0]
    tf_input = input(f"Timeframe ({default_tf}): ").strip()
    timeframe = tf_input if tf_input else default_tf
    
    if timeframe not in available_tfs:
        print(f"Timeframe {timeframe} not available for {pair}")
        return
    
    epochs_input = input("Training epochs (100): ").strip()
    epochs = int(epochs_input) if epochs_input.isdigit() else 100
    
    use_gpu_input = input("Use GPU if available (y/n, default: y): ").strip().lower()
    use_gpu = use_gpu_input != 'n'
    
    refine_input = input("Refine agent after training (y/n, default: y): ").strip().lower()
    refine = refine_input != 'n'
    
    # Train the agent
    print("\nStarting training...")
    save_filename = f"qstar_agent_{pair.replace('/', '_')}_{timeframe}.pkl"
    save_path = PATHS["models_dir"] / save_filename
    
    agent = train_agent(
        pair=pair,
        timeframe=timeframe,
        exchange='binance',
        use_gpu=use_gpu,
        epochs=epochs,
        refine=refine,
        save_path=str(save_path)
    )
    
    if agent:
        print("\nTraining completed!")
        print(f"Agent saved to: {save_path}")
        
        # Evaluate on the same data
        print("\nEvaluating agent...")
        metrics = evaluate_agent(agent, pair, timeframe)
        
        print("\nEvaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()