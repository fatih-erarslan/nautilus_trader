#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QStar-River Hybrid Trading System with Numba & Catalyst Optimization

A sophisticated hybrid trading system that combines Q* reinforcement learning,
River ML online learning, cerebellum-inspired neural networks, and quantum
circuits for advanced decision making.

Features:
- Numba njit for numerical operations
- PennyLane Catalyst qjit for quantum circuits
- Vectorized implementations for performance
- Hardware manager integration
- Rockpool-based cerebellar SNN
"""

import os
import time
import logging
import threading
import pickle
import gc
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto

from hardware_manager import HardwareManager
from river_ml import RiverOnlineML
# Configure logging

# Logger setup
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pulsar.log"), logging.StreamHandler()],
)

logger = logging.getLogger("Pulsar")


# from qstar_multitrainer
try:
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("TA library not available. Installing basic TA library would improve indicator quality.")
    
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
    
##end import
    
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    logger.info("JAX not available, falling back to other methods")


# Numba for optimization
try:
    import numba
    from numba import njit, prange, jit
    NUMBA_AVAILABLE = True
    
    # Create optimized numba config
    NUMBA_CACHE = True
    NUMBA_PARALLEL = True
    
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators for graceful fallback
    def njit(*args, **kwargs):
        if callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    
    def prange(*args):
        return range(*args)
    
    def jit(*args, **kwargs):
        if callable(args[0]):
            return args[0]
        else:
            def decorator(func):
                return func
            return decorator
    
    NUMBA_CACHE = False
    NUMBA_PARALLEL = False

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

# Import Rockpool for spiking neural networks if available
try:
    import rockpool
    import rockpool.nn as rnn
    import rockpool.parameters as rp
    import rockpool.training as rt
    ROCKPOOL_AVAILABLE = True
except ImportError:
    ROCKPOOL_AVAILABLE = False

# River ML for online learning
try:
    import river
    from river import drift, anomaly, preprocessing, metrics, stats, feature_selection
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("QStarHybrid")

# Import hardware manager
from hardware_manager import HardwareManager
from river_ml import RiverOnlineML
# Optional imports from existing modules if available
try:
    from qar import DecisionType, MarketPhase, TradingDecision
    QAR_AVAILABLE = True
except ImportError:
    QAR_AVAILABLE = False
    
    # Define trading action types
    class TradingAction:
        """Trading action definitions with six distinct action types."""
        
        # Core actions
        BUY = 0     # Full buy with available capital
        SELL = 1    # Full sell of current position
        HOLD = 2    # No action
        
        # Advanced actions
        REDUCE = 3  # Reduce position by percentage
        INCREASE = 4  # Increase position by percentage
        HEDGE = 5   # Create hedge position
        
        @staticmethod
        def get_num_actions() -> int:
            """Get number of possible actions."""
            return 6
        
        @staticmethod
        def get_action_name(action: int) -> str:
            """Get human-readable action name."""
            actions = {
                TradingAction.BUY: "BUY",
                TradingAction.SELL: "SELL",
                TradingAction.HOLD: "HOLD",
                TradingAction.REDUCE: "REDUCE",
                TradingAction.INCREASE: "INCREASE",
                TradingAction.HEDGE: "HEDGE"
            }
            return actions.get(action, "UNKNOWN")
        
        @staticmethod
        def action_to_signal(action: int) -> float:
            """Convert action to normalized signal value for technical indicators."""
            signals = {
                TradingAction.BUY: 1.0,
                TradingAction.INCREASE: 0.5,
                TradingAction.HOLD: 0.0,
                TradingAction.REDUCE: -0.5,
                TradingAction.SELL: -1.0,
                TradingAction.HEDGE: -0.25  # Slightly bearish
            }
            return signals.get(action, 0.0)
    
    # Create simplified versions for standalone operation
    class MarketPhase(Enum):
        UNKNOWN = 0
        GROWTH = 1
        CONSERVATION = 2
        RELEASE = 3
        REORGANIZATION = 4
    
    class DecisionType(Enum):
        BUY = 0
        SELL = 1
        HOLD = 2
        INCREASE = 3
        DECREASE = 4
        HEDGE = 5
        EXIT = 6

    @dataclass
    class TradingDecision:
        decision_type: int
        confidence: float
        reasoning: str
        timestamp: datetime = field(default_factory=datetime.now)

try:
    from logarithmic_market_scoring_rule import (
        LogarithmicMarketScoringRule, LMSRConfig,
        ProbabilityConversionMethod, AggregationMethod
    )
    LMSR_AVAILABLE = True
except ImportError:
    LMSR_AVAILABLE = False

# imported from qstar_multitrainer

# ---------------------------------
# GPU Utilities
# ---------------------------------
def configure_amd_gpu():
    """Configure environment for AMD GPU with ROCm."""
    try:
        # Check if we have an AMD GPU with PyTorch
        if TORCH_AVAILABLE:
            # Set environment variables for ROCm performance
            os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # For gfx1030 architecture (RX 6800 XT)
            os.environ['HIP_VISIBLE_DEVICES'] = '0'  # Use first GPU
            os.environ['GPU_MAX_HEAP_SIZE'] = '100'  # % of GPU memory to use
            os.environ['GPU_MAX_ALLOC_PERCENT'] = '100'
            
            # Force PyTorch to recognize ROCm device
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                if "AMD" in device_name or "Radeon" in device_name:
                    logger.info(f"Configured for AMD GPU: {device_name}")
                    return True
            
            # Try direct HIP check (ROCm's CUDA equivalent)
            if hasattr(torch, 'hip') and torch.hip.is_available():
                logger.info("AMD ROCm HIP support detected")
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error configuring AMD GPU: {e}")
        return False

# Call this function early
AMD_GPU_CONFIGURED = configure_amd_gpu()

def get_tensor_lib():
    """Get the tensor library to use for computations."""
    # Check for PyTorch - handles both NVIDIA and AMD
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            logger.info(f"Using PyTorch with GPU: {torch.cuda.get_device_name(0)}")
            return torch
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            logger.info("Using PyTorch with AMD HIP")
            return torch
    
    # Don't try JAX GPU at all - we know it doesn't work
    if JAX_AVAILABLE:
        logger.info("Using JAX (CPU only)")
        return jnp
        
    # Fallback to NumPy
    logger.info("Using NumPy (CPU only)")
    return np

def is_gpu_available() -> bool:
    """Check if GPU is available with special handling for AMD GPUs."""
    # Check if we've already configured AMD GPU
    if AMD_GPU_CONFIGURED:
        return True
        
    # Check PyTorch CUDA (works for NVIDIA or ROCm-enabled PyTorch)
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device_name}")
            return True
        # For AMD: also check HIP directly (ROCm's CUDA equivalent)
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            logger.info("AMD ROCm HIP support detected")
            return True
    
    # Don't check JAX GPU - it doesn't work with your setup
    return False

def test_gpu_availability():
    """Run a diagnostic test to check if GPU is usable."""
    print("\nGPU Availability Test:")
    print("-" * 40)
    
    # Test PyTorch
    if TORCH_AVAILABLE:
        print("PyTorch is available")
        
        # Check CUDA/ROCm availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"PyTorch can access {device_count} GPU(s)")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {device_name}")
                
            # Run a simple computation to verify GPU works
            try:
                x = torch.rand(1000, 1000, device='cuda')
                y = torch.rand(1000, 1000, device='cuda')
                start = time.time()
                z = torch.matmul(x, y)
                torch.cuda.synchronize()  # Wait for GPU operation to complete
                end = time.time()
                print(f"  GPU Matrix multiplication time: {(end-start)*1000:.2f} ms")
                del x, y, z  # Free GPU memory
                torch.cuda.empty_cache()
                print("  ✓ GPU computation successful")
            except Exception as e:
                print(f"  ✗ GPU computation failed: {e}")
        
        # Test for AMD HIP support
        elif hasattr(torch, 'hip') and torch.hip.is_available():
            print("PyTorch has HIP (ROCm) support for AMD GPUs")
            try:
                x = torch.rand(1000, 1000, device='cuda')  # 'cuda' is still the device name even with ROCm
                print("  ✓ AMD GPU recognized")
                del x
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ AMD GPU test failed: {e}")
        else:
            print("PyTorch does not detect any GPU")
            print("For AMD GPUs, make sure PyTorch is built with ROCm support")
            print("Try: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6")
    else:
        print("PyTorch is not installed")
    
    # Return overall status
    return torch.cuda.is_available() if TORCH_AVAILABLE else False

# Add this to your main function right after loading config
gpu_available = test_gpu_availability()
if not gpu_available:
    print("\nWARNING: No GPU detected. Training will be slow on CPU.")
    print("For AMD GPUs, ensure PyTorch has ROCm support.\n")
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


def load_training_config(config_path: Union[str, Path]) -> Dict:
    """
    Load training configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded configuration from {config_path}")
    return config

def load_data_from_config(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load data based on configuration settings.
    This function handles both individual pair files and combined files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping pair names to their respective dataframes
    """
    data_config = config['data_config']
    mode = data_config.get('mode', 'individual_pairs')
    
    logger.info(f"Loading data in {mode} mode")
    
    if mode == 'combined_file':
        return load_combined_file_data(data_config['combined_file'])
    elif mode == 'individual_pairs':
        return load_individual_pair_data(data_config['individual_pairs'])
    else:
        raise ValueError(f"Unknown data loading mode: {mode}")

def load_combined_file_data(combined_config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load data from a combined file containing multiple pairs.
    
    Args:
        combined_config: Configuration for the combined file
        
    Returns:
        Dictionary mapping pair names to their respective dataframes
    """
    filepath = combined_config['filepath']
    delimiter = combined_config.get('delimiter', ' ')
    has_header = combined_config.get('has_header', False)
    
    logger.info(f"Loading combined data from: {filepath}")
    
    # Read data with appropriate settings
    if has_header:
        df = pd.read_csv(filepath, delimiter=delimiter)
    else:
        # Use column mapping if no header
        column_mapping = combined_config.get('column_mapping', {})
        if column_mapping:
            # Create header names from mapping
            header = [None] * (max(map(int, column_mapping.values())) + 1)
            for name, index in column_mapping.items():
                header[index] = name
            
            # Read with generated header
            df = pd.read_csv(filepath, delimiter=delimiter, header=None, names=header)
        else:
            # No mapping provided, use numbered columns
            df = pd.read_csv(filepath, delimiter=delimiter, header=None)
    
    # Get pair column name
    pair_col = 'pair'
    for name, index in combined_config.get('column_mapping', {}).items():
        if index == 1:  # Assuming pair is always in column 1
            pair_col = name
            break
    
    # Filter to selected pairs if specified
    selected_pairs = combined_config.get('selected_pairs')
    if selected_pairs:
        df = df[df[pair_col].isin(selected_pairs)]
        logger.info(f"Filtered to {len(selected_pairs)} selected pairs")
    
    # Split data by pair
    pair_data = {}
    for pair in df[pair_col].unique():
        pair_df = df[df[pair_col] == pair].copy()
        
        # Handle date formatting if specified
        date_format = combined_config.get('date_format')
        if date_format:
            date_col = combined_config.get('date_column', 0)
            pair_df['date'] = pd.to_datetime(pair_df.iloc[:, date_col], format=date_format)
            pair_df.set_index('date', inplace=True)
        
        # Sort by index if not using dates
        else:
            index_col = combined_config.get('index_column', 0)
            pair_df.sort_values(pair_df.columns[index_col], inplace=True)
        
        # Store in result dictionary
        pair_data[pair] = pair_df
        logger.info(f"Loaded {len(pair_df)} rows for {pair}")
    
    return pair_data

def load_individual_pair_data(individual_config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Load data from individual files for each pair.
    
    Args:
        individual_config: Configuration for individual pair data
        
    Returns:
        Dictionary mapping pair names to their respective dataframes
    """
    pairs = individual_config['pairs']
    timeframe = individual_config['timeframe']
    exchange = individual_config.get('exchange', 'binance')
    data_dir = individual_config.get('data_dir', 'user_data/data')
    
    logger.info(f"Loading individual data for {len(pairs)} pairs with timeframe {timeframe}")
    
    pair_data = {}
    for pair in pairs:
        # Convert pair to filename format
        pair_file = pair.replace('/', '_')
        filepath = Path(data_dir) / exchange / f"{pair_file}-{timeframe}.feather"
        
        if not filepath.exists():
            logger.warning(f"Data file not found: {filepath}")
            continue
        
        try:
            # Load Freqtrade feather file
            df = pd.read_feather(filepath)
            
            # Check required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {filepath}")
                continue
            
            # Set date index
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], utc=True)
            df.set_index('date', inplace=True)
            
            # Calculate indicators
            df = calculate_indicators(df)
            
            # Store in result dictionary
            pair_data[pair] = df
            logger.info(f"Loaded {len(df)} rows for {pair}")
            
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {e}")
    
    return pair_data
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
    """Calculate technical indicators for trading features."""
    if df.empty:
        return df
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    try:
        if TA_AVAILABLE:
            # Use TA-Lib if available
            from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
            from ta.momentum import RSIIndicator, StochasticOscillator
            from ta.volatility import BollingerBands, AverageTrueRange
            from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

            df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator() # Added for consistency
            
            ema_12 = EMAIndicator(close=df['close'], window=12)
            ema_26 = EMAIndicator(close=df['close'], window=26)
            df['ema_12'] = ema_12.ema_indicator()
            df['ema_26'] = ema_26.ema_indicator()
            
            # MACD
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Momentum indicators
            df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3, fillna=True)
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volatility indicators
            bb = BollingerBands(close=df['close'], window=20, window_dev=2, fillna=True)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband() # Use wband for width
            
            # ATR for volatility
            df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True).average_true_range()
            
            # Volume indicators
            df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'], fillna=True).on_balance_volume()
            
            # Trend strength
            df['adx'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True).adx()

        else:
            # Basic calculations fallback (No TA-Lib used here)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # RSI (manual)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() # ensure loss is positive
            
            # Handle division by zero for rs
            rs = gain / (loss + 1e-10) # Add epsilon to prevent division by zero
            df['rsi_14'] = 100 - (100 / (1 + rs))
            df['rsi_14'] = df['rsi_14'].fillna(50) # Fill NaNs that can occur at the beginning or if loss is consistently zero

            # Stochastic Oscillator (manual) - More complex, simplified here or omit for else
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            df['stoch_k'] = df['stoch_k'].fillna(50)
            df['stoch_d'] = df['stoch_d'].fillna(50)

            # Bollinger Bands (manual)
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (2 * std_dev)
            df['bb_lower'] = df['bb_middle'] - (2 * std_dev)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
            df['bb_width'] = df['bb_width'].fillna(0)


            # ATR (manual) - Simplified, true ATR is more complex
            df['tr0'] = abs(df['high'] - df['low'])
            df['tr1'] = abs(df['high'] - df['close'].shift())
            df['tr2'] = abs(df['low'] - df['close'].shift())
            df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            # --- MODERNIZED LINE ---
            df['atr'] = df['atr'].bfill().fillna(0) # Backfill then zero for start
            # --- END MODERNIZED LINE ---
            df.drop(columns=['tr0', 'tr1', 'tr2', 'tr'], inplace=True, errors='ignore')

            # OBV (manual)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

            # ADX (manual) - Very complex, often simplified or omitted in fallbacks
            # For simplicity, we'll use a placeholder or a simpler trend indicator
            df['adx'] = (df['close'] / df['close'].rolling(14).mean() - 1) * 100 # Simple trend strength proxy
            df['adx'] = df['adx'].fillna(0)
            
        # Custom indicators for Q* environment (common to both TA_AVAILABLE and not)
        df['returns'] = df['close'].pct_change()
        df['returns_z'] = (df['returns'] - df['returns'].rolling(30).mean()) / (df['returns'].rolling(30).std() + 1e-10)
        
        # Volatility regime
        df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)  # Annualized
        df['volatility_regime'] = (df['volatility'] - df['volatility'].rolling(100).min()) / \
                               (df['volatility'].rolling(100).max() - df['volatility'].rolling(100).min() + 1e-10)
        
        # Q* specific features
        df['qerc_trend'] = df['returns'].rolling(10).mean() * 10
        df['qerc_momentum'] = df['close'].pct_change(5)
        df['iqad_score'] = df['returns'].rolling(20).std() / (df['returns'].abs().rolling(20).mean() + 1e-10)
        df['performance_metric'] = df['returns'].rolling(20).mean()
        
        # Clean up NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        # Prioritize forward fill for time series, then backfill, then zero
        df = df.ffill().bfill().fillna(0)       
            
        logger.info(f"Calculated {len(df.columns) - 6} indicators")  # -6 for OHLCV + date (original)
        return df
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        # Return original DataFrame if error occurs, after standard cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        return df.ffill().bfill().fillna(0)
    
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

##end import

#######################################################
# STANDALONE NUMBA-OPTIMIZED COMPUTATION FUNCTIONS
#######################################################

@njit(cache=NUMBA_CACHE)
def calculate_state_index(features, num_states=200):
    """
    Calculate state index from feature vector.
    Numba-optimized standalone function.
    
    Args:
        features: Feature vector
        num_states: Number of discrete states
        
    Returns:
        State index (0 to num_states-1)
    """
    # Simple hash function
    hash_value = 0
    for i in range(len(features)):
        # Discretize feature into 10 levels
        feature_val = features[i]
        level = max(0, min(9, int((feature_val + 1) * 5)))
        hash_value += level * (10 ** i)
    
    # Modulo to get within state space
    return hash_value % num_states

@njit(cache=NUMBA_CACHE)
def update_q_value(q_table, state, action, target, learning_rate):
    """
    Update Q-value with Numba optimization.
    
    Args:
        q_table: Q-value table
        state: State index
        action: Action index
        target: Target Q-value
        learning_rate: Learning rate
        
    Returns:
        Magnitude of the update
    """
    # Extract current prediction
    current = q_table[state, action]
    # Calculate delta
    delta = learning_rate * (target - current)
    # Update the specific Q-value
    q_table[state, action] = current + delta
    return abs(delta)

@njit(cache=True)
def calculate_reward(old_value, new_value, reward_scaling=0.01):
    """Calculate reward based on portfolio change."""
    return ((new_value / old_value) - 1.0) * reward_scaling

@njit(cache=True)
def normalize_probabilities(probs, min_prob=0.001, max_prob=0.999):
    """Normalize probabilities to valid range."""
    result = np.zeros_like(probs)
    for i in range(len(probs)):
        if np.isnan(probs[i]) or np.isinf(probs[i]):
            result[i] = 0.5
        else:
            result[i] = max(min(probs[i], max_prob), min_prob)
    return result

@njit(cache=NUMBA_CACHE)
def quantum_action_selection(q_values_arr, phases_arr, seed):
    """
    Select action using quantum interference principles. Inputs assumed NumPy arrays.
    Explicit dtype added for Numba type unification. Numba-optimized.
    """
    np.random.seed(seed)
    # Explicitly define expected dtype (float64 is safer for precision)
    TARGET_DTYPE = np.float64

    # Ensure inputs are the target dtype (might be redundant if caller ensures it, but safer)
    q_values_conv = q_values_arr.astype(TARGET_DTYPE)
    phases_conv = phases_arr.astype(TARGET_DTYPE)
    num_actions = len(q_values_conv)

    # Normalize q-values
    q_values_finite = q_values_conv[np.isfinite(q_values_conv)]
    if len(q_values_finite) == 0:
        return np.random.randint(num_actions)

    q_min = np.min(q_values_finite) # Use min of finite values
    q_values_shifted = q_values_conv - q_min if q_min < 0 else q_values_conv
    q_sum = np.sum(q_values_shifted[np.isfinite(q_values_shifted)])

    # Calculate amplitudes and probabilities
    if q_sum > 1e-9:
        shifted_positive = np.maximum(0.0, q_values_shifted)
        amplitudes = np.sqrt(shifted_positive / (q_sum + 1e-9)).astype(TARGET_DTYPE) # Cast sqrt result

        # Ensure phases_final has the correct length and dtype
        valid_phases = phases_conv[np.isfinite(phases_conv)]
        if len(valid_phases) != num_actions:
             phases_final = np.random.uniform(0, 2*np.pi, num_actions).astype(TARGET_DTYPE) # Cast random phases
        else:
             phases_final = valid_phases # Already correct type

        # Calculate complex results
        real_parts = amplitudes * np.cos(phases_final)
        imag_parts = amplitudes * np.sin(phases_final)
        probs = (real_parts**2 + imag_parts**2).astype(TARGET_DTYPE) # Cast probs

        # Ensure positive and sums to 1, handle NaNs
        probs[np.isnan(probs)] = 0.0
        probs = np.maximum(0.0, probs)
        probs_sum = np.sum(probs)
        if probs_sum > 1e-9:
            probs = probs / probs_sum
        else:
             probs = np.ones(num_actions, dtype=TARGET_DTYPE) / num_actions # Use target dtype
    else:
        probs = np.ones(num_actions, dtype=TARGET_DTYPE) / num_actions # Use target dtype

    # Manual weighted choice
    cum_probs = np.cumsum(probs)
    r = np.random.rand()
    action = np.searchsorted(cum_probs, r, side='right')
    action = min(action, num_actions - 1)
    action = max(action, 0)

    return action
@njit(cache=NUMBA_CACHE, parallel=NUMBA_PARALLEL)
def compute_batch_targets(q_table, states, actions, rewards, next_states, dones, discount_factor):
    """
    Compute target Q-values for a batch of experiences.
    Optimized with Numba for parallel processing.
    
    Args:
        q_table: Q-value table
        states: Array of state indices
        actions: Array of action indices
        rewards: Array of rewards
        next_states: Array of next state indices
        dones: Array of done flags
        discount_factor: Discount factor for future rewards
        
    Returns:
        Array of target Q-values
    """
    batch_size = len(states)
    targets = np.zeros(batch_size, dtype=np.float32)
    
    for i in prange(batch_size):
        if dones[i]:
            targets[i] = rewards[i]
        else:
            # Find maximum Q-value for next state
            next_max_q = np.max(q_table[next_states[i]])
            targets[i] = rewards[i] + discount_factor * next_max_q
    
    return targets

@njit(cache=NUMBA_CACHE)
def normalize_features(features):
    """
    Normalize feature vector to zero mean and unit variance.
    
    Args:
        features: Feature vector
        
    Returns:
        Normalized features
    """
    if len(features) == 0:
        return features
        
    # Calculate mean and standard deviation
    mean = np.mean(features)
    std = np.std(features)
    
    # Avoid division by zero
    if std < 1e-10:
        std = 1.0
    
    # Normalize features
    return (features - mean) / std

@njit(cache=NUMBA_CACHE)
def compute_drawdown(portfolio_values):
    """
    Calculate maximum drawdown from portfolio value history.
    
    Args:
        portfolio_values: Array of portfolio values over time
        
    Returns:
        Maximum drawdown as a positive percentage
    """
    # Calculate running maximum
    running_max = np.zeros_like(portfolio_values)
    running_max[0] = portfolio_values[0]
    
    for i in range(1, len(portfolio_values)):
        running_max[i] = max(running_max[i-1], portfolio_values[i])
    
    # Calculate drawdowns
    drawdowns = (running_max - portfolio_values) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    return max_drawdown

@njit(cache=NUMBA_CACHE)
def calculate_sharpe_ratio(returns):
    """
    Calculate Sharpe ratio from returns. Numba-optimized.

    Args:
        returns: Array or list of period returns

    Returns:
        Annualized Sharpe ratio
    """
    # Convert list to numpy array for Numba compatibility
    # Ensure it handles potential non-numeric types gracefully if list isn't clean
    # Best practice is to ensure `returns` is already a clean list/array of floats
    try:
        # Explicitly cast to float64 for consistency
        returns_arr = np.asarray(returns, dtype=np.float64)
        # Remove NaNs or Infs that might crash Numba functions
        returns_arr = returns_arr[np.isfinite(returns_arr)]
    except:
        # Fallback if conversion fails (e.g., list contains non-numerics)
        return 0.0 # Or raise an error

    if len(returns_arr) < 2:
        return 0.0

    # Use the numpy array for calculations
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr)

    # Avoid division by zero or near-zero standard deviation
    if std_return < 1e-10:
         return 0.0 # Return 0 for constant returns

    # Annualize (assuming daily returns with 252 trading days)
    # Ensure 252.0 is float for sqrt and potential division
    sharpe = mean_return / std_return * np.sqrt(252.0)

    # Handle potential NaN result if mean_return was NaN (though filtered above)
    if np.isnan(sharpe):
        return 0.0

    return sharpe

#######################################################
# QUANTUM CIRCUIT DEFINITIONS WITH CATALYST
#######################################################

@qjit
def quantum_decision_circuit(features, phases):
    """
    Quantum circuit for trading decisions.
    Optimized with PennyLane Catalyst.
    
    Args:
        features: Normalized feature vector
        phases: Phase angles for quantum interference
        
    Returns:
        Measurement results from quantum circuit
    """
    # Use CPU device by default for reliability
    dev = qml.device("default.qubit", wires=8)
    
    # Define quantum circuit
    @qml.qnode(dev)
    def circuit(features, phases):
        # Amplitude encoding of features
        max_features = min(len(features), 5)
        for i in range(max_features):
            qml.RY(features[i] * np.pi, wires=i)
        
        # Apply phase rotations
        max_phases = min(len(phases), 5)
        for i in range(max_phases):
            qml.RZ(phases[i], wires=i)
        
        # Create entanglement
        for i in range(4):
            qml.CNOT(wires=[i, i+1])
        
        # Add quantum interference
        qml.CNOT(wires=[0, 5])
        qml.CNOT(wires=[1, 6])
        qml.CNOT(wires=[2, 7])
        
        # Final Hadamard layer
        for i in range(5, 8):
            qml.Hadamard(wires=i)
        
        # Measure decision qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(5, 8)]
    
    # Execute circuit
    return circuit(features, phases)

@qjit
def quantum_market_regime_circuit(indicators, phases):
    """
    Quantum circuit for market regime detection.
    Optimized with PennyLane Catalyst.
    
    Args:
        indicators: Market indicators
        phases: Phase angles for quantum interference
        
    Returns:
        Measurement results for regime detection
    """
    # Use CPU device by default for reliability
    dev = qml.device("default.qubit", wires=6)
    
    # Define quantum circuit
    @qml.qnode(dev)
    def circuit(indicators, phases):
        # Encode market indicators
        max_indicators = min(len(indicators), 4)
        for i in range(max_indicators):
            qml.RY(indicators[i] * np.pi, wires=i)
        
        # Apply phases
        max_phases = min(len(phases), 4)
        for i in range(max_phases):
            qml.RZ(phases[i], wires=i)
        
        # Create entanglement
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
        
        # Connect to regime qubits
        qml.CNOT(wires=[0, 4])
        qml.CNOT(wires=[2, 5])
        
        # Final Hadamard layer
        qml.Hadamard(wires=4)
        qml.Hadamard(wires=5)
        
        # Measure regime qubits
        return [qml.expval(qml.PauliZ(4)), qml.expval(qml.PauliZ(5))]
    
    # Execute circuit
    return circuit(indicators, phases)


#######################################################
# CORE COMPONENTS
#######################################################

logger = logging.getLogger("TradingEnvironment")


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    
    This environment implements a financial trading simulation with:
    - Portfolio management
    - Transaction costs
    - Market data integration
    - Reward based on portfolio performance
    """
    
    def __init__(
        self, 
        price_data: pd.DataFrame = None, 
        price_column: str = 'close', 
        window_size: int = 50,
        initial_capital: float = 10000.0, 
        transaction_fee: float = 0.001,
        reward_scaling: float = 0.01,
        use_position_limits: bool = True,
        max_position_size: float = 1.0
    ):
        """Initialize trading environment with proper attribute initialization."""
        # Environment parameters
        self.price_data = price_data
        self.price_column = price_column
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.reward_scaling = reward_scaling
        self.use_position_limits = use_position_limits
        self.max_position_size = max_position_size
        
        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Environment dimensions
        self.num_states = 200
        self.num_actions = 7  # 7 actions in unified action space
        
        # Extract prices
        if price_data is not None and price_column in price_data.columns:
            self.prices = price_data[price_column].values
        else:
            # Default prices for testing
            self.prices = np.linspace(100, 110, 1000) + np.random.normal(0, 1, 1000)
        
        # Initialize state
        self.reset()
    
    def reset(self) -> int:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state index
        """
        # Reset trading state
        self.balance = self.initial_capital
        self.position = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.initial_capital
        self.last_buy_price = 0.0
        self.last_sell_price = 0.0
        
        # Reset market position
        self.current_idx = self.window_size
        if self.current_idx < len(self.prices):
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
        
        # Calculate initial state
        initial_state = self._calculate_state()
        
        return initial_state
    
    def _calculate_state(self) -> int:
        """
        Calculate current state index.
        
        Returns:
            State index
        """
        # Extract window of price data
        start_idx = max(0, self.current_idx - self.window_size + 1)
        end_idx = self.current_idx
        if end_idx >= len(self.prices):
            end_idx = len(self.prices) - 1
        
        price_window = self.prices[start_idx:end_idx+1]
        
        # Calculate features
        features = np.zeros(10)
        
        if len(price_window) > 1:
            # Calculate returns
            returns = np.diff(price_window) / price_window[:-1]
            
            # Momentum features
            features[0] = np.mean(returns[-5:]) if len(returns) >= 5 else 0
            features[1] = np.mean(returns[-10:]) if len(returns) >= 10 else 0
            
            # Volatility
            features[2] = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.1
            
            # Trend
            features[3] = (price_window[-1] / price_window[0] - 1) if len(price_window) > 1 else 0
            
            # Position information
            features[4] = self.position_value / (self.portfolio_value + 1e-10)  # Current position size
            features[5] = 1.0 if self.position > 0 else 0.0  # Has position
            
            # Profit/loss if position exists
            if self.position > 0 and self.last_buy_price > 0:
                features[6] = (self.current_price / self.last_buy_price) - 1.0
            else:
                features[6] = 0.0
            
            # Technical indicators (simplified versions)
            if len(price_window) >= 14:
                # Simple RSI
                up_moves = np.zeros(min(14, len(returns)))
                down_moves = np.zeros(min(14, len(returns)))
                
                for i in range(min(14, len(returns))):
                    if returns[-(i+1)] > 0:
                        up_moves[i] = returns[-(i+1)]
                    else:
                        down_moves[i] = -returns[-(i+1)]
                
                avg_up = np.mean(up_moves)
                avg_down = np.mean(down_moves)
                
                if avg_down != 0:
                    rs = avg_up / avg_down
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100 if avg_up > 0 else 50
                
                features[7] = rsi / 100  # Normalize to [0, 1]
            else:
                features[7] = 0.5  # Default
            
            # Moving averages
            if len(price_window) >= 20:
                sma_5 = np.mean(price_window[-5:])
                sma_20 = np.mean(price_window[-20:])
                features[8] = sma_5 / sma_20 - 1
            else:
                features[8] = 0
        
        # Market data state (placeholder)
        features[9] = 0.5
        
        # Use standalone function to calculate state index
        state_index = calculate_state_index(features, self.num_states)
        
        return state_index
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        try:
            # Validate action
            if not 0 <= action < 7:
                self.logger.warning(f"Invalid action: {action}, defaulting to HOLD")
                action = 2  # Default to HOLD
            
            # Track the action
            self.actions_taken.append(action)
            
            # Store current portfolio value for reward calculation
            old_portfolio_value = self.portfolio_value
            
            # Execute action
            self._execute_action(action)
            
            # Move to next time step
            done = self._next_time_step()
            
            # If we're done, return immediately
            if done:
                return self._calculate_state(), 0.0, done
            
            # Update portfolio value
            self.position_value = self.position * self.current_price
            self.portfolio_value = self.balance + self.position_value
            
            # Track positions and portfolio value
            self.positions.append(self.position)
            self.portfolio_values.append(self.portfolio_value)
            
            # Calculate return
            portfolio_return = (self.portfolio_value / old_portfolio_value) - 1.0
            self.returns.append(portfolio_return)
            
            # Calculate reward - scaled return
            reward = portfolio_return * self.reward_scaling
            
            # Calculate next state
            next_state = self._calculate_state()
            
            return next_state, reward, done
        
        except Exception as e:
            self.logger.error(f"Error in step: {str(e)}")
            
            # Return current state, no reward, and done=True to terminate episode
            return self._calculate_state(), 0.0, True
    
    def _execute_action(self, action: int) -> None:
        """
        Execute trading action.
        
        Args:
            action: Action to take (0-6)
        """
        # BUY: Full buy with available balance
        if action == 0 and self.balance > 0:
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
                
        # SELL: Full sell of current position
        elif action == 1 and self.position > 0:
            # Full sell - liquidate entire position
            sell_amount = self.position
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L
            position_cost = self.position * self.last_buy_price if self.last_buy_price > 0 else 0
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
                
        # HOLD: No action
        elif action == 2:
            pass
            
        # REDUCE: Partially reduce position
        elif action == 3 and self.position > 0:
            # Reduce position by 50%
            sell_amount = self.position * 0.5
            sell_value = sell_amount * self.current_price
            fee = sell_value * self.transaction_fee
            sell_value_after_fee = sell_value - fee
            
            # Calculate P&L for the sold portion
            position_cost = sell_amount * self.last_buy_price if self.last_buy_price > 0 else 0
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
                
        # INCREASE: Add to position
        elif action == 4 and self.balance > 0:
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
                
        # HEDGE: Create a hedge position
        elif action == 5 and self.position > 0 and self.balance > 0:
            # Use 20% of available balance for hedge
            hedge_amount = self.balance * 0.2
            
            # Calculate the hedge ratio based on position value
            hedge_ratio = min(0.5, hedge_amount / (self.position_value + 1e-10))
            
            # Simplified: just reduce effective exposure by moving to cash
            self.balance -= hedge_amount
            
        # EXIT: Exit the market completely
        elif action == 6:
            if self.position > 0:
                # Sell everything
                sell_amount = self.position
                sell_value = sell_amount * self.current_price
                fee = sell_value * self.transaction_fee
                sell_value_after_fee = sell_value - fee
                
                # Calculate P&L
                position_cost = self.position * self.last_buy_price if self.last_buy_price > 0 else 0
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
    
    def _next_time_step(self) -> bool:
        """
        Move to next time step.
        
        Returns:
            Whether the episode is done
        """
        self.current_idx += 1
        
        # Check if we've reached the end of available data
        if self.current_idx >= len(self.prices):
            return True
        
        # Update current price
        self.current_price = self.prices[self.current_idx]
        
        return False

# imported from qstar_multitrainer

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
        
##end import
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current episode.
        
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate key metrics
            metrics = {
                "total_return": (self.portfolio_value / self.initial_capital) - 1.0,
                "win_rate": self.profitable_trades / max(1, self.total_trades),
                "total_trades": self.total_trades,
                "total_pnl": self.total_pnl,
                "final_portfolio": self.portfolio_value
            }
            
            # Calculate Sharpe ratio if enough returns
            if len(self.returns) > 1:
                metrics["sharpe_ratio"] = calculate_sharpe_ratio(self.returns)
            else:
                metrics["sharpe_ratio"] = 0.0
            
            # Calculate max drawdown
            metrics["max_drawdown"] = compute_drawdown(np.array(self.portfolio_values))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0,
                "total_pnl": 0.0,
                "final_portfolio": self.initial_capital,
                "error": str(e)
            }

# imported from qstar_multitrainer

logger = logging.getLogger("MultiPairTradingEnvironment")

class MultiPairTradingEnvironment:
    """
    Trading environment that handles multiple pairs simultaneously.
    Compatible with both combined file data and individual pair data.
    """
    
    def __init__(self, pair_data: Dict[str, pd.DataFrame], window_size=50, 
                 initial_capital=10000.0, transaction_fee=0.001, use_gpu=None):
        """
        Initialize trading environment with multiple pairs.
        
        Args:
            pair_data: Dictionary mapping pair names to their DataFrames
            window_size: Size of the rolling window for features
            initial_capital: Initial account balance
            transaction_fee: Trading fee as fraction
            use_gpu: Whether to use GPU acceleration
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store configurations
        self.pair_data = pair_data
        self.pairs = list(pair_data.keys())
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            self.use_gpu = is_gpu_available()
        else:
            self.use_gpu = use_gpu
        
        # Validate data
        if not self.pairs:
            raise ValueError("No pairs provided in pair_data")
        
        # Prepare price data for each pair
        self.pair_prices = {}
        for pair, df in pair_data.items():
            if 'close' in df.columns:
                self.pair_prices[pair] = df['close'].values
            else:
                # Try to find close price column
                close_cols = [col for col in df.columns if 'close' in str(col).lower()]
                if close_cols:
                    self.pair_prices[pair] = df[close_cols[0]].values
                else:
                    raise ValueError(f"No close price column found for {pair}")
        
        # Find minimum length across all pairs
        self.max_steps = min([len(prices) for prices in self.pair_prices.values()]) - window_size
        
        if self.max_steps <= 0:
            raise ValueError("Insufficient data length for the specified window size")
        
        # Initialize state
        self.reset()
        
        # State and action space
        self.num_states = 500  # Larger state space for multi-pair interactions
        self.num_actions = len(self.pairs) * 3  # Buy/Sell/Hold for each pair
        
        self.logger.info(f"Multi-pair environment initialized with {len(self.pairs)} pairs")
        self.logger.info(f"State space: {self.num_states}, Action space: {self.num_actions}")
        self.logger.info(f"Maximum steps: {self.max_steps}")
    
    def reset(self) -> int:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state index
        """
        # Reset trading state
        self.balance = self.initial_capital
        self.positions = {pair: 0.0 for pair in self.pairs}
        self.position_values = {pair: 0.0 for pair in self.pairs}
        self.total_pnl = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Reset market position
        self.current_idx = self.window_size
        
        # Initialize current prices
        self.current_prices = {
            pair: self.pair_prices[pair][self.current_idx] 
            for pair in self.pairs
        }
        
        # Initialize market states for each pair
        self.market_states = {
            pair: MarketState(num_states=200, feature_window=self.window_size)
            for pair in self.pairs
        }
        
        # Update market states with initial data
        self._update_market_states()
        
        # Track portfolio history
        self.portfolio_values = [self.get_portfolio_value()]
        self.actions_taken = []
        
        # Return initial state
        return self.get_portfolio_state()
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Current portfolio value
        """
        position_total = sum(self.position_values.values())
        return self.balance + position_total
    
    def get_portfolio_state(self) -> int:
        """
        Get a unified state representation for the portfolio.
        
        Returns:
            State index
        """
        # Get individual states for each pair
        pair_states = [
            self.market_states[pair].get_state_index()
            for pair in self.pairs
        ]
        
        # Get allocation percentages
        portfolio_value = self.get_portfolio_value()
        allocations = [
            self.position_values.get(pair, 0) / portfolio_value if portfolio_value > 0 else 0
            for pair in self.pairs
        ]
        
        # Combine states and allocations using hash function
        state_components = tuple(pair_states + allocations)
        combined_state = hash(state_components)
        
        # Map to state index space
        return abs(combined_state) % self.num_states
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Track the action
        self.actions_taken.append(action)
        
        # Execute action
        reward = self._execute_action(action)
        
        # Move to next time step
        done = self._next_time_step()
        
        # If we're done (reached end of data), return immediately
        if done:
            return self.get_portfolio_state(), reward, done
        
        # Update market states with new data
        self._update_market_states()
        
        # Get new state
        next_state = self.get_portfolio_state()
        
        # Track portfolio value history
        self.portfolio_values.append(self.get_portfolio_value())
        
        # Return step results
        return next_state, reward, done
    
    def _execute_action(self, action: int) -> float:
        """
        Execute trading action and calculate reward.
        
        Args:
            action: Action index
            
        Returns:
            Reward
        """
        previous_portfolio_value = self.get_portfolio_value()
        
        # Determine which pair this action affects
        pair_index = action // 3
        pair_action = action % 3  # 0=Buy, 1=Sell, 2=Hold
        
        # Make sure pair_index is valid
        if pair_index >= len(self.pairs):
            return 0.0  # No reward for invalid action
        
        # Get the pair for this action
        pair = self.pairs[pair_index]
        
        # Execute the action
        if pair_action == 0:  # Buy
            self._execute_buy(pair)
        elif pair_action == 1:  # Sell
            self._execute_sell(pair)
        # Hold action (2) does nothing
        
        # Calculate reward based on portfolio change
        new_portfolio_value = self.get_portfolio_value()
        portfolio_return = (new_portfolio_value / previous_portfolio_value) - 1.0
        
        # Scale reward to make it more meaningful for learning
        reward = portfolio_return * 10.0  # Scale factor can be adjusted
        
        return reward
    
    def _execute_buy(self, pair: str) -> None:
        """
        Execute buy action for a specific pair.
        
        Args:
            pair: Trading pair to buy
        """
        if self.balance <= 0:
            return  # No balance to buy with
        
        # Use all available balance
        buy_amount = self.balance
        
        # Calculate fees
        fee = buy_amount * self.transaction_fee
        buy_amount_after_fee = buy_amount - fee
        
        # Calculate position size
        price = self.current_prices[pair]
        new_position = buy_amount_after_fee / price
        
        # Update state
        self.balance = 0  # Used all balance
        self.positions[pair] += new_position
        self.position_values[pair] = self.positions[pair] * price
        self.total_trades += 1
        
        self.logger.debug(f"BUY {pair}: {new_position:.4f} units at {price:.2f}")
    
    def _execute_sell(self, pair: str) -> None:
        """
        Execute sell action for a specific pair.
        
        Args:
            pair: Trading pair to sell
        """
        position = self.positions.get(pair, 0)
        if position <= 0:
            return  # No position to sell
        
        # Calculate sell value
        price = self.current_prices[pair]
        sell_value = position * price
        
        # Calculate fees
        fee = sell_value * self.transaction_fee
        sell_value_after_fee = sell_value - fee
        
        # Calculate P&L
        previous_value = self.position_values[pair]
        pnl = sell_value_after_fee - previous_value
        
        # Update state
        self.balance += sell_value_after_fee
        self.positions[pair] = 0
        self.position_values[pair] = 0
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.profitable_trades += 1
        
        self.logger.debug(f"SELL {pair}: {position:.4f} units at {price:.2f}, PnL: {pnl:.2f}")
    
    def _next_time_step(self) -> bool:
        """
        Move to next time step.
        
        Returns:
            Whether the episode is done
        """
        self.current_idx += 1
        
        # Check if we've reached the end of available data
        if self.current_idx >= self.max_steps + self.window_size:
            return True
        
        # Update current prices
        self._update_current_prices()
        
        # Update position values
        self._update_position_values()
        
        return False
    
    def _update_current_prices(self) -> None:
        """Update current prices for all pairs."""
        for pair in self.pairs:
            self.current_prices[pair] = self.pair_prices[pair][self.current_idx]
    
    def _update_position_values(self) -> None:
        """Update position values based on current prices."""
        for pair in self.pairs:
            position = self.positions.get(pair, 0)
            price = self.current_prices.get(pair, 0)
            self.position_values[pair] = position * price
    
    def _update_market_states(self) -> None:
        """Update market states for all pairs with current data."""
        for pair, market_state in self.market_states.items():
            # Get current data for this pair
            df = self.pair_data[pair]
            
            # Get window of data up to current index
            if self.current_idx < len(df):
                # If we're using dataframes with time indices
                current_row = df.iloc[self.current_idx]
                
                # Extract features from current row
                features = current_row.to_dict()
                
                # Update market state
                market_state.update(features)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the current episode.
        
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns metrics
        portfolio_values = np.array(self.portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        metrics = {
            "total_return": (portfolio_values[-1] / portfolio_values[0]) - 1.0 if len(portfolio_values) > 0 else 0,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": self.profitable_trades / max(1, self.total_trades),
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "final_portfolio": self.get_portfolio_value()
        }
        
        return metrics

class ExperienceBuffer:
    """
    Experience replay buffer for reinforcement learning.
    Optimized for performance with vectorized operations.
    """
    
    def __init__(self, max_size: int = 10000, batch_size: int = 64, alpha: float = 0.6):
        """Initialize experience buffer with prioritized replay."""
        self.max_size = max_size
        self.batch_size = batch_size
        self.alpha = alpha  # Priority exponent
        self.beta = 0.4     # Initial importance sampling weight
        self.beta_increment = 0.001  # Annealing rate for beta
        
        # Use efficient numpy arrays for storage
        self.states = np.zeros(max_size, dtype=np.int32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros(max_size, dtype=np.int32)
        self.dones = np.zeros(max_size, dtype=np.bool_)
        self.priorities = np.ones(max_size, dtype=np.float32)
        
        # Position tracking
        self.position = 0
        self.size = 0
    
    def add(self, state: int, action: int, reward: float, next_state: int, done: bool = False) -> None:
        """Add experience to buffer with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        # Store experience
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.priorities[self.position] = max_priority
        
        # Update position and size
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch using prioritized experience replay."""
        if self.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # Anneal beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, min(self.batch_size, self.size), p=probabilities, replace=False)
        
        # Calculate importance-sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Retrieve experiences
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            weights,
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for experiences based on TD errors."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < self.size:
                self.priorities[idx] = max(1e-6, priority)  # Ensure positive priority
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return self.size


class QLearningMetrics:
    """Tracks and analyzes Q* learning performance metrics."""
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.episode_rewards = []
        self.q_value_changes = []
        self.exploration_rates = []
        self.learning_rates = []
        self.convergence_measures = []
        self.episode_lengths = []
        self.timestamps = []
    
    def add_episode_data(self, reward: float, length: int, q_change: float,
                        exploration_rate: float, learning_rate: float,
                        convergence: float) -> None:
        """Add metrics from a completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.q_value_changes.append(q_change)
        self.exploration_rates.append(exploration_rate)
        self.learning_rates.append(learning_rate)
        self.convergence_measures.append(convergence)
        self.timestamps.append(time.time())
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of metrics."""
        if not self.episode_rewards:
            return {}
        
        # Calculate summary metrics
        return {
            "avg_reward": float(np.mean(self.episode_rewards[-100:])),
            "max_reward": float(np.max(self.episode_rewards)),
            "min_reward": float(np.min(self.episode_rewards)),
            "avg_episode_length": float(np.mean(self.episode_lengths[-100:])),
            "final_convergence": float(self.convergence_measures[-1]) if self.convergence_measures else None,
            "final_exploration_rate": float(self.exploration_rates[-1]) if self.exploration_rates else None,
            "total_episodes": len(self.episode_rewards)
        }

def run_training_from_config_file(config_path):
    """Run training using a configuration file."""
    agent = train_with_config(config_path)
    
    if agent is None:
        logger.error("Training failed")
        return
    
    logger.info("Training completed successfully")
    
    # You could add additional reporting or backtesting here
    return agent

## end import

logger = logging.getLogger("Q*LearningAgent")

class QStarLearningAgent:
    """
    Advanced Q* Learning implementation with Numba optimization.
    
    Features:
    - Quantum-inspired state representation
    - Adaptive learning rates and exploration
    - Experience replay with prioritization
    - Advanced convergence metrics
    - Hardware-accelerated operations
    """
    
    def __init__(
        self,
        states: int,
        actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        exploration_rate: float = 1.0,
        min_exploration_rate: float = 0.05,
        exploration_decay_rate: float = 0.995,
        use_adaptive_learning_rate: bool = True,
        use_experience_replay: bool = True,
        experience_buffer_size: int = 10000,
        batch_size: int = 64,
        max_episodes: int = 1000,
        max_steps_per_episode: int = 10000,
        use_quantum_representation: bool = True,
        use_gpu: bool = None,
        mixed_precision: bool = False
    ):
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
        # === Basic Parameters & Logging ===
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.states = states
        self.actions = actions
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode

        # === Feature Flags ===
        self.use_adaptive_learning_rate = use_adaptive_learning_rate
        self.use_experience_replay = use_experience_replay
        self.use_quantum_representation = use_quantum_representation

        # === Hardware/Backend Configuration (Assign BEFORE _init_q_table) ===
        if use_gpu is None:
            self.use_gpu = is_gpu_available() # Auto-detect
        else:
            self.use_gpu = use_gpu and is_gpu_available() # Use user preference only if GPU available

        self.tensor_lib = get_tensor_lib() # Determine tensor library based on availability/preference

        # Critical: Assign mixed_precision BEFORE calling _init_q_table
        # Check TORCH_AVAILABLE necessary for torch mixed precision
        self.mixed_precision = mixed_precision and self.use_gpu and (TORCH_AVAILABLE or JAX_AVAILABLE)
        # Log the final decision for mixed precision
        if mixed_precision and not self.mixed_precision:
             self.logger.warning(f"Mixed precision requested but not enabled (GPU available: {self.use_gpu}, Torch/JAX Available: {TORCH_AVAILABLE or JAX_AVAILABLE})")
        elif self.mixed_precision:
            self.logger.info("Mixed precision training enabled.")


        # === Q-Table Initialization (Call AFTER dependencies are set) ===
        self.q_table = None # Initialize to None before calling init
        self._init_q_table()

        # === Experience Replay Buffer ===
        self.batch_size = batch_size
        if self.use_experience_replay:
            # Use the appropriate buffer (GPUExperienceBuffer or simple ExperienceBuffer)
            self.experience_buffer = ExperienceBuffer( # Or GPUExperienceBuffer if using that version
                max_size=experience_buffer_size,
                batch_size=self.batch_size
                # alpha=0.6, # If using prioritization
                # use_gpu=self.use_gpu # If GPU buffer exists
            )
        else:
            self.experience_buffer = None

        # === Metrics Tracking ===
        self.metrics = QLearningMetrics()
        self.episode_rewards = []
        self.q_value_changes = []
        self.exploration_rates = []
        self.learning_rates = []
        self.execution_times = []
        self.gpu_utilization = []
        self.convergence_measures = []
        self.episode_lengths = []
        self.convergence_history = [] # For has_converged check
        self.avg_q_change = 0.0 # For adaptive LR

        # === Quantum Phases ===
        # Initialize AFTER states/actions/backend are known
        if self.use_quantum_representation:
            self.quantum_phases = None # Initialize to None
            self._init_quantum_phases() # Create a helper if needed or init here
        else:
            self.quantum_phases = None

        self.logger.info(f"Q* Agent Initialized: States={self.states}, Actions={self.actions}, UseGPU={self.use_gpu}, MixedPrec={self.mixed_precision}, TensorLib={'torch' if self.tensor_lib is torch else 'jnp' if self.tensor_lib is jnp else 'numpy'}")

    # Add this helper method or include logic in __init__
    def _init_quantum_phases(self):
        """Initialize quantum phases array on the correct device."""
        if not self.use_quantum_representation:
            self.quantum_phases = None
            return

        shape = (self.states, self.actions)
        if self.tensor_lib is torch:
             device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
             self.quantum_phases = torch.rand(shape, device=device) * (2 * np.pi)
        elif self.tensor_lib is jnp:
             key = jax.random.PRNGKey(int(time.time())) # Use a different key
             phases_data = jax.random.uniform(key, shape=shape, minval=0, maxval=2*np.pi)
             if self.use_gpu and len(jax.devices('gpu')) > 0:
                 self.quantum_phases = jax.device_put(phases_data, jax.devices('gpu')[0])
             else:
                 self.quantum_phases = phases_data # Already on CPU JAX device
        else: # NumPy
             self.quantum_phases = np.random.uniform(0, 2*np.pi, shape)
        logger.debug(f"Initialized quantum phases with shape {self.quantum_phases.shape}")

##imported
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
            self.q_table = jnp.zeros((self.states, self.actions), dtype=dtype)
            if self.use_gpu and len(jax.devices('gpu')) > 0:
                try:
                    # Try to move to GPU
                    self.q_table = jax.device_put(self.q_table, jax.devices('gpu')[0])
                except:
                    # Fallback if GPU access fails
                    pass
        else:
            self.q_table = np.zeros((self.states, self.actions), dtype=dtype)
            
        logger.debug(f"Initialized Q-table with shape {self.q_table.shape}, "
                    f"on {'GPU' if self.use_gpu else 'CPU'}, "
                    f"using {self.tensor_lib.__name__ if hasattr(self.tensor_lib, '__name__') else 'numpy'}")

    
    def choose_action(self, state: int) -> int:
        """
        Choose action based on current state and exploration strategy.
        Ensures NumPy arrays are passed to Numba function when needed.
        """
        state = max(0, min(state, self.states - 1))

        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.actions)

        # --- Exploitation Path ---
        if self.use_quantum_representation and self.quantum_phases is not None:
            q_values = self.q_table[state]
            phases = self.quantum_phases[state]
            seed = int(time.time() * 1000 + state) % 10000 # Vary seed slightly

            # Check tensor library being used
            if self.tensor_lib is torch:
                q_values_np = q_values.cpu().numpy() # Convert Torch to NumPy
                phases_np = phases.cpu().numpy()     # Convert Torch to NumPy
            elif self.tensor_lib is jnp:
                q_values_np = np.asarray(q_values) # Convert JAX to NumPy
                phases_np = np.asarray(phases)     # Convert JAX to NumPy
            else: # Assume already NumPy
                q_values_np = q_values
                phases_np = phases

            # Call Numba function with guaranteed NumPy arrays
            return quantum_action_selection(q_values_np, phases_np, seed)
        else:
            # Classical: find max Q-value
             if self.tensor_lib is torch:
                 return int(torch.argmax(self.q_table[state]).item())
             elif self.tensor_lib is jnp:
                 return int(jnp.argmax(self.q_table[state]))
             else:
                 return int(np.argmax(self.q_table[state]))
        
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
    

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool = False) -> float:
        """
        Update the Q-table based on experience, supporting different backends
        (PyTorch, JAX, NumPy) and handling terminal states correctly.

        Args:
            state: Current state index
            action: Action taken index
            reward: Received reward
            next_state: Resulting state index
            done: Boolean indicating if the episode ended after this step

        Returns:
            Absolute TD error (magnitude of Q-value update)
        """
        # --- 1. State Validation ---
        state = max(0, min(state, self.states - 1))
        next_state = max(0, min(next_state, self.states - 1))

        # --- 2. Experience Replay ---
        if self.use_experience_replay and hasattr(self, 'experience_buffer') and self.experience_buffer is not None:
            self.experience_buffer.add(state, action, reward, next_state, done)

        # --- 3. Target Calculation (Handles 'done' flag) ---
        target = 0.0
        predict = 0.0
        next_max_q = 0.0

        # Get current Q prediction
        try:
            if self.tensor_lib is torch:
                predict = self.q_table[state, action].item()
            elif self.tensor_lib is jnp:
                predict = float(self.q_table[state, action]) # Convert JAX scalar to float
            else:
                predict = self.q_table[state, action]
        except IndexError:
            self.logger.error(f"IndexError accessing Q-table PREDICT at state={state}, action={action}. Shape: {self.q_table.shape}")
            return 0.0

        # Calculate next state max Q if not done
        if not done:
            try:
                if self.tensor_lib is torch:
                    with torch.no_grad():
                        next_max_q = torch.max(self.q_table[next_state, :]).item()
                elif self.tensor_lib is jnp:
                    next_max_q = float(jnp.max(self.q_table[next_state, :])) # Convert JAX scalar to float
                else:
                    next_max_q = np.max(self.q_table[next_state, :])
            except IndexError:
                 self.logger.error(f"IndexError accessing Q-table NEXT_MAX_Q at next_state={next_state}. Shape: {self.q_table.shape}")
                 next_max_q = 0.0 # Default if next state out of bounds

        # Final Target
        target = reward + self.discount_factor * next_max_q

        # --- 4. Q-Value Update (Backend-specific) ---
        try:
            update_value = self.learning_rate * (target - predict)
            if self.tensor_lib is torch:
                # Standard PyTorch in-place update
                self.q_table[state, action] += update_value
            elif self.tensor_lib is jnp:
                 # Correct JAX immutable update
                 # Make sure update_value has compatible dtype (convert if needed)
                 update_value_jax = jnp.array(update_value, dtype=self.q_table.dtype)
                 self.q_table = self.q_table.at[state, action].add(update_value_jax)
            else: # Assume NumPy
                # Standard NumPy in-place update
                self.q_table[state, action] += update_value
        except IndexError:
             # Logged above
             self.logger.error(f"Skipping Q-table update due to IndexError for state={state}, action={action}.")
             return 0.0
        except Exception as e:
             self.logger.error(f"Error during Q-table update (state={state}, action={action}): {e}", exc_info=True)
             return 0.0

        # --- 5. Return Value (Absolute TD Error) ---
        td_error = abs(target - predict)
        return td_error
    
    # Inside QStarLearningAgent class (pulsar.py)
    def replay_experiences(self) -> float:
        """
        Learn from stored experiences using prioritized replay with
        backend-specific (Torch/JAX/NumPy) implementations.

        Returns:
            Average magnitude of Q-value updates for the batch.
        """
        if not self.use_experience_replay or self.experience_buffer is None:
            return 0.0

        buffer_len = len(self.experience_buffer)
        if buffer_len < self.batch_size:
            return 0.0

        # Sample batch (indices and weights are NumPy arrays from buffer)
        # Ensure the buffer sample returns NumPy arrays as expected
        states_np, actions_np, rewards_np, next_states_np, dones_np, weights_np, indices = self.experience_buffer.sample()

        # Check if sampling returned anything
        if len(indices) == 0:
            return 0.0

        total_td_error = 0.0 # Use TD error for priority update

        # --- Backend Specific Processing ---
        try:
            if self.tensor_lib is torch:
                # --- PyTorch Path ---
                # Convert sampled NumPy arrays to Torch tensors on the correct device
                device = self.q_table.device
                dtype = self.q_table.dtype
                states = torch.tensor(states_np, device=device, dtype=torch.long)
                actions = torch.tensor(actions_np, device=device, dtype=torch.long)
                rewards = torch.tensor(rewards_np, device=device, dtype=dtype)
                next_states = torch.tensor(next_states_np, device=device, dtype=torch.long)
                dones = torch.tensor(dones_np, device=device, dtype=torch.bool)
                weights = torch.tensor(weights_np, device=device, dtype=dtype)

                # Get Q-values for next states
                with torch.no_grad():
                    next_q_values_all = self.q_table[next_states] # Shape: (batch_size, num_actions)
                    next_max_q, _ = torch.max(next_q_values_all, dim=1) # Max along action dimension

                # Calculate targets (handle terminal states)
                targets = torch.where(dones, rewards, rewards + self.discount_factor * next_max_q)

                # Get current Q-values for the actions taken
                current_q_values = self.q_table[states, actions]

                # Calculate TD errors
                td_errors = targets - current_q_values

                # Calculate updates scaled by learning rate and importance weights
                updates = self.learning_rate * td_errors * weights

                # Apply updates IN-PLACE to the Q-table
                self.q_table[states, actions] += updates

                # Convert TD errors back to NumPy for priority update
                td_errors_np = td_errors.cpu().numpy()
                total_td_error = np.sum(np.abs(td_errors_np))

            elif self.tensor_lib is jnp:
                # --- JAX Path (as derived previously) ---
                # Ensure sampled data are JAX arrays if needed (depends on how Q-table slicing works with NumPy indices)
                # JAX usually handles NumPy indices fine.
                next_q_values_all = self.q_table[next_states_np, :]
                next_max_q = jnp.max(next_q_values_all, axis=1)

                targets = jnp.where(dones_np, rewards_np, rewards_np + self.discount_factor * next_max_q)

                current_q_values = self.q_table[states_np, actions_np]
                td_errors = targets - current_q_values

                weights_jax = jnp.array(weights_np, dtype=targets.dtype)
                updates = self.learning_rate * td_errors * weights_jax

                # IMMUTABLE UPDATE
                self.q_table = self.q_table.at[states_np, actions_np].add(updates)

                # Convert TD errors back to NumPy
                td_errors_np = np.asarray(td_errors)
                total_td_error = np.sum(np.abs(td_errors_np))

            else:
                # --- NumPy Path ---
                next_q_values_all = self.q_table[next_states_np, :]
                next_max_q = np.max(next_q_values_all, axis=1)

                targets = np.where(dones_np, rewards_np, rewards_np + self.discount_factor * next_max_q)

                current_q_values = self.q_table[states_np, actions_np]
                td_errors = targets - current_q_values

                updates = self.learning_rate * td_errors * weights_np

                # In-place update for NumPy
                self.q_table[states_np, actions_np] += updates

                td_errors_np = td_errors # Already NumPy
                total_td_error = np.sum(np.abs(td_errors_np))

            # --- Update Priorities (Common Logic) ---
            # Use absolute TD errors for priorities
            abs_td_errors = np.abs(td_errors_np)
            self.experience_buffer.update_priorities(indices, abs_td_errors + 1e-6) # Add epsilon

            # Return average absolute TD error
            avg_td_error = total_td_error / len(indices) if len(indices) > 0 else 0.0
            return float(avg_td_error)

        except Exception as e:
            self.logger.error(f"Error during replay_experiences (tensor_lib={self.tensor_lib}): {e}", exc_info=True)
            return 0.0 # Return 0 error on failure to avoid breaking training loop    
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
            avg_q_change: Average Q-value change in recent episodes
        """
        if not self.use_adaptive_learning_rate:
            return
        
        # Decrease learning rate over time with adaptive adjustments
        if avg_q_change < 0.01:
            # Small changes might indicate convergence or getting stuck
            # Increase learning rate slightly to escape local minima
            self.learning_rate = min(0.5, self.learning_rate * 1.05)
        elif avg_q_change > 0.5:
            # Large changes might indicate instability
            # Decrease learning rate to stabilize learning
            self.learning_rate = max(0.01, self.learning_rate * 0.95)
        else:
            # Standard decay schedule
            progress = min(1.0, episode / self.max_episodes)
            self.learning_rate = self.initial_learning_rate * (1.0 - 0.9 * progress)
    
    def has_epoch_converged(self, threshold: float = 0.01, window_size: int = 100) -> Tuple[bool, float]:
        """
        Check if learning has epoch_converged based on recent Q-value changes.
        
        Args:
            threshold: Convergence threshold
            window_size: Number of recent episodes to consider
            
        Returns:
            Tuple of (epoch_converged, convergence_value)
        """
        if len(self.convergence_history) < window_size:
            return False, 1.0
        
        # Calculate average change over recent episodes
        recent_changes = np.mean(self.convergence_history[-window_size:])
        
        # Check if below threshold
        epoch_converged = recent_changes < threshold
        
        return epoch_converged, recent_changes
    
#imported    
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
            
##end import
            
# Assuming necessary imports like time, np, torch, jax, gc are present
# And assuming the class has attributes like self.logger, self.use_gpu, self.tensor_lib, etc.
# defined in its __init__ method.



    def train(self, environment) -> Tuple[bool, int]:
        """
        Train the agent in the provided environment, combining robustness and detailed tracking.
    
        Args:
            environment: Training environment with reset() and step() methods
    
        Returns:
            Tuple of (converged, episodes_completed)
        """
        # --- Pre-Training Setup (from V1 & V2) ---
        start_time_training = time.time() # Overall training timer
    
        # GPU Check and Initialization (from V1)
        if self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
            try:
                # Run a small operation to initialize GPU context if needed
                test_tensor = torch.ones(1, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"GPU initialization check failed: {e}. Proceeding...")
        else:
            logger.info("Training on CPU or GPU not configured/available.")
    
        # Environment Check (from V2, enhanced)
        if not (hasattr(environment, 'reset') and callable(environment.reset) and
                hasattr(environment, 'step') and callable(environment.step)):
            logger.error("Environment must implement reset() and step() methods")
            # Consider returning a specific error or raising ValueError
            return False, 0 # Indicate failure
    
        # Environment Type Check (from V1, optional but good practice)
        # Assuming you have a specific environment class like TradingEnvironment
        # if hasattr(self, 'TradingEnvironment') and not isinstance(environment, self.TradingEnvironment):
        #     logger.warning(f"Expected TradingEnvironment, got {type(environment)}. Compatibility not guaranteed.")
    
        logger.info(f"Starting training for up to {self.max_episodes} episodes...")
    
        # Initialize metrics tracking (keep V2's pattern, but use V1's lists)
        episodes_completed = 0
        converged = False # Final convergence status
    
        # --- Training Loop (Structure from V2, content from V1 & V2) ---
        for episode in range(self.max_episodes):
            episode_start_time = time.time() # V1 detailed timing
            try:
                # Reset environment and episode metrics (from V2)
                state = environment.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_q_changes = [] # To calculate avg_q_change for the episode
    
                # Update quantum phases slightly (Combined V1 & V2)
                if self.use_quantum_representation and episode % 10 == 0 and self.quantum_phases is not None:
                     logger.debug(f"Perturbing quantum phases for episode {episode}")
                     try:
                         if self.tensor_lib is torch:
                             noise = torch.rand_like(self.quantum_phases) * 0.2 - 0.1
                             self.quantum_phases.add_(noise) # Use inplace add_ for tensors if appropriate
                         elif self.tensor_lib is jnp and JAX_AVAILABLE: # Check JAX_AVAILABLE flag
                             key = jax.random.PRNGKey(int(time.time()) + episode) # Use episode for varying key
                             noise = jax.random.uniform(key, shape=self.quantum_phases.shape, minval=-0.1, maxval=0.1)
                             self.quantum_phases = self.quantum_phases + noise # Immutable update
                         else: # Fallback to NumPy
                             noise = np.random.uniform(-0.1, 0.1, self.quantum_phases.shape)
                             self.quantum_phases += noise
                     except Exception as e_phase:
                         logger.warning(f"Could not update quantum phases: {e_phase}")
    
                # --- Episode Step Loop (from V2) ---
            # --- Episode Step Loop ---
                done = False
                while not done and episode_steps < self.max_steps_per_episode:
                    # --- Choose Action ---
                    # This needs to handle the agent's internal backend (Torch/JAX/NumPy)
                    action = self.choose_action(state) # Should now be robust
    
                    try:
                        # --- Interact with Environment ---
                        # The environment (potentially from qstar_river) returns standard types (int, float, bool)
                        next_state, reward, done = environment.step(action)
    
                        # --- Learn ---
                        # The learn method handles different backends internally
                        q_change = self.learn(state, action, reward, next_state, done)
                        if q_change is not None: # learn might return None on error
                             episode_q_changes.append(q_change)
    
                        # Update state etc. (standard types)
                        episode_reward += float(reward) # Ensure float
                        episode_steps += 1
                        state = int(next_state) # Ensure int
    
                    except Exception as e_step:
                         self.logger.error(f"Error in environment step {episode_steps}: {e_step}", exc_info=True)
                         done = True
                         # Use the last valid state, default reward/q_change
                         next_state = state # Assume state unchanged on error
                         reward = -1.0 # Penalty
                         episode_q_changes.append(0.0) # No change
                # --- End Episode Step Loop ---
    
                # === After Episode Actions ===
    
                # Learn from experience replay (V2 structure)
                if self.use_experience_replay and hasattr(self, 'experience_buffer') and self.experience_buffer is not None and len(self.experience_buffer) >= self.batch_size:
                     logger.debug(f"Episode {episode}: Performing experience replay.")
                     replay_q_change = self.replay_experiences()
                     if replay_q_change is not None: episode_q_changes.append(replay_q_change)
    
                # Update exploration rate (V2 structure)
                self.update_exploration_rate()
    
                # Calculate average Q-change for this episode (V2 name, V1 storage)
                avg_q_change = np.mean(episode_q_changes) if episode_q_changes else 0.0
                # Add to V1's convergence history list for checking
                if hasattr(self, 'convergence_history'):
                     self.convergence_history.append(avg_q_change)
    
                # Update learning rate adaptively (V2 structure, requires V2's method)
                if hasattr(self, 'update_learning_rate'):
                     self.update_learning_rate(episode, avg_q_change)
    
                # Store metrics (Using V1's individual lists for direct access)
                if hasattr(self, 'episode_rewards'): self.episode_rewards.append(episode_reward)
                if hasattr(self, 'q_value_changes'): self.q_value_changes.append(avg_q_change) # V1 name
                if hasattr(self, 'exploration_rates'): self.exploration_rates.append(self.exploration_rate)
                if hasattr(self, 'learning_rates'): self.learning_rates.append(self.learning_rate)
                if hasattr(self, 'episode_lengths'): self.episode_lengths.append(episode_steps)
    
                # Calculate and store execution time (from V1)
                execution_time = time.time() - episode_start_time
                if hasattr(self, 'execution_times'): self.execution_times.append(execution_time)
    
                # Get and store GPU utilization (from V1)
                gpu_utilization = 0.0 # Default
                if self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        # Placeholder - Direct utilization often unavailable. Monitor memory instead.
                        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**2) # MB
                        gpu_utilization = gpu_memory_allocated # Use memory as proxy metric
                    except Exception as e_gpu_util:
                        logger.debug(f"Could not get GPU memory metric: {e_gpu_util}")
                if hasattr(self, 'gpu_utilization'): self.gpu_utilization.append(gpu_utilization)
    
                # Check for convergence (using V1's method and history)
                converged_episode, convergence_value = False, float('nan')
                if hasattr(self, 'has_converged') and callable(self.has_converged):
                    converged_episode, convergence_value = self.has_converged()
                    if hasattr(self, 'convergence_measures'): self.convergence_measures.append(convergence_value)
    
                # Log progress (from V1)
                if (episode + 1) % 10 == 0 or converged_episode: # Log every 10 or on convergence
                    logger.info(f"Episode {episode+1}/{self.max_episodes}: Reward={episode_reward:.2f}, "
                               f"Steps={episode_steps}, Explore={self.exploration_rate:.4f}, LR={self.learning_rate:.5f}, "
                               f"AvgQChg={avg_q_change:.6f}, ConvergVal={convergence_value:.6f}, "
                               f"Time={execution_time:.2f}s")
                    # Log GPU memory (more informative than utilization usually)
                    if self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
                         logger.info(f"  GPU Memory Allocated: {gpu_utilization:.2f} MB")
    
    
                # Break if converged (from V1)
                if converged_episode:
                    logger.info(f"Converged after {episode+1} episodes.")
                    episodes_completed = episode + 1
                    converged = True # Set overall convergence flag
                    break
    
                # GPU Cleanup (from V1)
                if self.use_gpu and episode > 0 and episode % 50 == 0:
                     if TORCH_AVAILABLE and torch.cuda.is_available():
                         logger.debug(f"Running GPU cache clearing at episode {episode}")
                         torch.cuda.empty_cache()
                     if 'gc' in globals(): # Check if gc imported
                         gc.collect()
    
            except KeyboardInterrupt:
                 logger.warning("Training interrupted by user.")
                 episodes_completed = episode + 1
                 converged = False
                 break # Exit outer loop on interrupt
            except Exception as e:
                logger.error(f"Error during training episode {episode}: {e}", exc_info=True)
                # Optionally break or continue? Continuing might hide issues. Let's break.
                episodes_completed = episode + 1
                converged = False
                break
        # --- End Training Loop ---
    
        # Set final completed episodes if loop finished naturally (from V2)
        if episodes_completed == 0:
            episodes_completed = self.max_episodes
            if not converged: # Only log if it didn't converge earlier
                logger.info(f"Maximum episodes ({self.max_episodes}) reached without convergence.")
    
        # --- Post-Training Summary and Cleanup ---
        training_time = time.time() - start_time_training
        logger.info(f"Training finished. Completed: {episodes_completed}/{self.max_episodes} episodes in {training_time:.2f} seconds.")
    
        # Final Log Summary (from V1)
        if hasattr(self, '_log_training_summary') and callable(self._log_training_summary):
            self._log_training_summary()
    
        # Final GPU cleanup (from V1)
        if self.use_gpu:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if 'gc' in globals():
                gc.collect()
    
        # Return convergence status and episodes used
        return converged, episodes_completed
    
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
            
    def evaluate(self, environment, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the agent in the environment without learning.
        
        Args:
            environment: Evaluation environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        # Disable exploration during evaluation
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0
        
        metrics = {
            "rewards": [],
            "steps": [],
            "success": 0
        }
        
        for episode in range(num_episodes):
            try:
                state = environment.reset()
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done and episode_steps < self.max_steps_per_episode:
                    # Choose action without exploration
                    action = self.choose_action(state)
                    
                    try:
                        next_state, reward, done = environment.step(action)
                    except Exception as e:
                        logger.error(f"Error in environment step during evaluation: {e}")
                        done = True
                        next_state = state
                        reward = 0
                    
                    # Track metrics
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                
                metrics["rewards"].append(episode_reward)
                metrics["steps"].append(episode_steps)
                
                # Track successful episodes (positive reward)
                if episode_reward > 0:
                    metrics["success"] += 1
                    
            except Exception as e:
                logger.error(f"Error during evaluation episode {episode}: {e}")
                # Continue to next episode
        
        # Calculate summary metrics
        evaluation_metrics = {
            "mean_reward": float(np.mean(metrics["rewards"])) if metrics["rewards"] else 0,
            "std_reward": float(np.std(metrics["rewards"])) if metrics["rewards"] else 0,
            "min_reward": float(np.min(metrics["rewards"])) if metrics["rewards"] else 0,
            "max_reward": float(np.max(metrics["rewards"])) if metrics["rewards"] else 0,
            "mean_steps": float(np.mean(metrics["steps"])) if metrics["steps"] else 0,
            "success_rate": float(metrics["success"] / max(1, num_episodes))
        }
        
        # Restore exploration rate
        self.exploration_rate = original_exploration_rate
        
        return evaluation_metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            filepath: Path to save file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        data = {
            "q_table": self.q_table,
            "states": self.states,
            "actions": self.actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "min_exploration_rate": self.min_exploration_rate,
            "exploration_decay_rate": self.exploration_decay_rate,
            "use_adaptive_learning_rate": self.use_adaptive_learning_rate,
            "use_experience_replay": self.use_experience_replay,
            "batch_size": self.batch_size,
            "use_quantum_representation": self.use_quantum_representation,
            "quantum_phases": self.quantum_phases,
            "metrics": self.metrics.__dict__,
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
            'use_gpu': self.use_gpu,
            'mixed_precision': self.mixed_precision,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, use_gpu: bool = None) -> 'QStarLearningAgent':
        """
        Load an agent from a file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded QStarLearningAgent
        """
        try:
            with open(filepath, 'rb') as f:
                agent_data = pickle.load(f)
                
            # Check if GPU setting was provided
            if use_gpu is None:
                # Use saved setting if available, otherwise auto-detect
                use_gpu = agent_data.get('use_gpu', None)
            
            # Create agent with saved parameters
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
                batch_size=agent_data.get("batch_size", 64),
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
        """ Plots backtest results, ensuring index safety. """
        # ... (check environment type) ...
    
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0
    
        try:
            state = environment.reset()
            done = False
            # Explicitly re-initialize lists for this run
            environment.actions_taken = []
            environment.positions = []
            environment.portfolio_values = [environment.initial_capital] # Start with initial capital
    
            # Reset internal state if necessary? Depends on environment.reset()
            environment.total_trades = 0
            environment.profitable_trades = 0
            environment.total_pnl = 0.0
    
            step_count = 0
            # Use a large number or environment specific length if available
            max_sim_steps = len(environment.prices) - environment.window_size if environment.prices is not None else 10000
    
            while not done and step_count < max_sim_steps:
                action = self.choose_action(state)
                # Action is chosen BUT step might fail below
    
                try:
                    next_state, reward, done = environment.step(action)
                    # If step SUCCEEDS, action is valid and state/metrics are updated
                    state = next_state
                    # Environment's step should handle appending to its internal lists
                except Exception as e:
                    self.logger.error(f"Error during backtest step {step_count}: {e}", exc_info=True)
                    done = True # Stop simulation on error
    
                step_count += 1
    
            # --- Plotting ---
            metrics = environment.get_performance_metrics()
            # Use the lengths of the lists *collected by the environment*
            portfolio_values = environment.portfolio_values
            positions = environment.positions
            actions = environment.actions_taken # Action list might be longer if last step failed
    
            # Determine the number of steps for plotting based on portfolio_values length
            num_plot_steps = len(portfolio_values)
            if num_plot_steps == 0:
                 logger.warning("No portfolio values recorded for plotting.")
                 return # Cannot plot
    
            time_steps = list(range(num_plot_steps))
    
            fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
            # Plot portfolio value (safe indexing)
            axes[0].plot(time_steps, portfolio_values, label='Portfolio Value')
            axes[0].set_title(f'Portfolio Value (Return: {metrics.get("total_return", 0):.2%})')
            axes[0].set_ylabel('Value ($)')
            axes[0].grid(True)
            axes[0].legend()
    
            # Add buy/sell markers (safe indexing)
            # Iterate only up to the number of PLOTTED steps (length of portfolio_values)
            for i in range(min(len(actions), num_plot_steps)):
                 action = actions[i]
                 pv_value = portfolio_values[i] # Use safe index 'i'
                 if action == TradingAction.BUY:
                     axes[0].scatter(i, pv_value, color='green', marker='^', alpha=0.7, label='Buy' if i == 0 else "") # Label once
                 elif action == TradingAction.SELL:
                     axes[0].scatter(i, pv_value, color='red', marker='v', alpha=0.7, label='Sell' if i == 0 else "")
                 elif action == TradingAction.INCREASE:
                     axes[0].scatter(i, pv_value, color='lime', marker='^', alpha=0.5, label='Increase' if i == 0 else "")
                 elif action == TradingAction.REDUCE:
                     axes[0].scatter(i, pv_value, color='orange', marker='v', alpha=0.5, label='Reduce' if i == 0 else "")
                 elif action == TradingAction.EXIT:
                     axes[0].scatter(i, pv_value, color='black', marker='x', alpha=0.7, label='Exit' if i == 0 else "")
                 elif action == TradingAction.HEDGE:
                     axes[0].scatter(i, pv_value, color='purple', marker='s', alpha=0.5, label='Hedge' if i == 0 else "")
    
            # Plot positions with length correction - IMPORTANT FIX
            if len(positions) == 0:
                # If no positions, create a zero array
                padded_positions = np.zeros(num_plot_steps)
            elif len(positions) < num_plot_steps:
                # If shorter than portfolio_values, pad with zeros at the beginning
                padding_length = num_plot_steps - len(positions)
                padded_positions = np.zeros(num_plot_steps)
                padded_positions[padding_length:] = positions
            else:
                # Truncate if longer (shouldn't happen, but just in case)
                padded_positions = positions[:num_plot_steps]
                
            # Now plot with matching lengths
            axes[1].plot(time_steps, padded_positions, label='Position Size')
            axes[1].set_title('Position Size')
            axes[1].set_ylabel('Units')
            axes[1].grid(True)
            axes[1].legend()
    
            # Plot actions heatmap/areas with safety checks
            valid_actions = []
            if len(actions) > 0:
                valid_actions = actions[:num_plot_steps]
                
            num_actions = TradingAction.get_num_actions() # Get total possible actions
            cmap = plt.get_cmap('viridis', num_actions) # Use total actions for consistent coloring
    
            for i in range(num_plot_steps): # Loop up to plot length
                if i < len(valid_actions):
                    action = valid_actions[i]
                    color = cmap(action / (num_actions - 1) if num_actions > 1 else 0.5)
                    axes[2].axvspan(i, i+1, facecolor=color, alpha=0.3)
                else:
                    # Use neutral color for steps without actions
                    axes[2].axvspan(i, i+1, facecolor='gray', alpha=0.1)
    
            # Add action legend
            all_action_names = [TradingAction.get_action_name(i) for i in range(num_actions)]
            all_action_colors = [cmap(i / (num_actions-1) if num_actions > 1 else 0.5) for i in range(num_actions)]
            action_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3) for color in all_action_colors]
            axes[2].legend(action_patches, all_action_names, loc='upper right', fontsize='small')
    
            axes[2].set_title('Trading Actions')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Action Type')
            axes[2].set_yticks([]) # Remove numerical y-ticks
            
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

    def stop(self):
        """Called when strategy is stopped."""
        try:
            # Save models state
            if hasattr(self, 'qstar_predictor') and self.qstar_predictor is not None:
                self.qstar_predictor.save_state(force=True)  # Force save regardless of timing
                
            self.logger.info("QStar strategy models saved")
        except Exception as e:
            self.logger.error(f"Error saving models during shutdown: {e}")
        
        # Call parent method
        super().stop()
        
class RiverOnlineMLAdapter:
    """
    Adapter for River online learning components.
    
    Provides:
    - Drift detection
    - Anomaly detection
    - Online feature selection
    - Preprocessing
    """
    
    def __init__(
        self,
        drift_detector_type: str = 'adwin',
        anomaly_detector_type: str = 'hst',
        feature_window: int = 50,
        drift_sensitivity: float = 0.05,
        anomaly_threshold: float = 0.75,
        enable_feature_selection: bool = False
    ):
        """Initialize River online ML components."""
        if not RIVER_AVAILABLE:
            logger.warning("River is not available, using dummy components")
            self._river_available = False
        else:
            self._river_available = True
        
        self.drift_detector_type = drift_detector_type
        self.anomaly_detector_type = anomaly_detector_type
        self.feature_window = feature_window
        self.drift_sensitivity = drift_sensitivity
        self.anomaly_threshold = anomaly_threshold
        self.enable_feature_selection = enable_feature_selection
        
        # Initialize components
        self._initialize_components()
        
        # Statistics
        self.drift_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=100)
        self.feature_importance = {}
    
    def _initialize_components(self):
        """Initialize River online learning components."""
        if not self._river_available:
            # Create dummy components
            self.drift_detector = type('DummyDriftDetector', (), {
                'update': lambda x: None,
                'drift_detected': False
            })()
            
            self.anomaly_detector = type('DummyAnomalyDetector', (), {
                'score_one': lambda x: 0.0,
                'learn_one': lambda x: None
            })()
            
            self.feature_scaler = type('DummyScaler', (), {
                'learn_one': lambda x: self,
                'transform_one': lambda x: x
            })()
            
            self.feature_selector = None
            return
        
        try:
            # Initialize drift detector
            if self.drift_detector_type == 'adwin':
                self.drift_detector = drift.ADWIN(delta=self.drift_sensitivity)
            elif self.drift_detector_type == 'page_hinkley':
                self.drift_detector = drift.PageHinkley(
                    min_instances=30, 
                    delta=self.drift_sensitivity,
                    threshold=20, 
                    alpha=0.9999
                )
            else:
                # Default to ensemble of detectors
                self.drift_detector = drift.EnsembleDriftDetector([
                    drift.ADWIN(delta=self.drift_sensitivity),
                    drift.PageHinkley(
                        min_instances=30,
                        delta=self.drift_sensitivity,
                        threshold=20,
                        alpha=0.9999
                    )
                ], vote_threshold=0.5)
            
            # Initialize anomaly detector
            if self.anomaly_detector_type == 'hst':
                self.anomaly_detector = anomaly.HalfSpaceTrees(
                    n_trees=50,
                    height=8,
                    window_size=self.feature_window,
                    seed=42
                )
            elif self.anomaly_detector_type == 'robust_svm':
                self.anomaly_detector = river.compose.Pipeline(
                    preprocessing.StandardScaler(),
                    anomaly.RobustOneClassSVM(nu=0.1)
                )
            else:
                # Default
                self.anomaly_detector = anomaly.HalfSpaceTrees(
                    n_trees=50,
                    height=8,
                    window_size=self.feature_window,
                    seed=42
                )
            
            # Initialize feature scaler
            self.feature_scaler = preprocessing.StandardScaler()
            
            # Initialize feature selector if enabled
            if self.enable_feature_selection:
                self.feature_selector = feature_selection.VarianceThreshold(
                    threshold=0.05
                )
            else:
                self.feature_selector = None
                
        except Exception as e:
            logger.error(f"Error initializing River components: {e}")
            # Create dummy components as fallback
            self.drift_detector = type('DummyDriftDetector', (), {
                'update': lambda x: None,
                'drift_detected': False
            })()
            
            self.anomaly_detector = type('DummyAnomalyDetector', (), {
                'score_one': lambda x: 0.0,
                'learn_one': lambda x: None
            })()
            
            self.feature_scaler = type('DummyScaler', (), {
                'learn_one': lambda x: self,
                'transform_one': lambda x: x
            })()
            
            self.feature_selector = None
    
    def detect_drift(self, value: float) -> Dict[str, Any]:
        """
        Detect concept drift in data stream.
        
        Args:
            value: New data point
            
        Returns:
            Drift detection results
        """
        try:
            # Update drift detector
            prev_drift = self.drift_detector.drift_detected
            self.drift_detector.update(value)
            
            # Check for drift
            drift_detected = self.drift_detector.drift_detected
            
            # Track if new drift was detected
            new_drift = drift_detected and not prev_drift
            
            # Store drift event
            self.drift_history.append((drift_detected, time.time()))
            
            # Calculate drift rate
            drift_rate = sum(1 for d, _ in self.drift_history if d) / len(self.drift_history) if self.drift_history else 0
            
            return {
                'drift_detected': drift_detected,
                'new_drift_detected': new_drift,
                'drift_rate': drift_rate
            }
        
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return {
                'drift_detected': False,
                'new_drift_detected': False,
                'drift_rate': 0.0,
                'error': str(e)
            }
    
    def detect_anomalies(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect anomalies in feature vector.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Anomaly detection results
        """
        try:
            # Preprocess features
            preprocessed = {}
            for k, v in features.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    preprocessed[k] = float(v)
            
            if not preprocessed:
                return {
                    'anomaly_score': 0.0,
                    'is_anomaly': False,
                    'error': 'No valid features'
                }
            
            # Apply standard scaling
            try:
                scaled_features = self.feature_scaler.learn_one(preprocessed).transform_one(preprocessed)
            except Exception as e:
                logger.warning(f"Error in feature scaling: {e}")
                scaled_features = preprocessed
            
            # Apply feature selection if enabled
            selected_features = scaled_features
            if self.enable_feature_selection and self.feature_selector is not None:
                try:
                    self.feature_selector.learn_one(scaled_features)
                    selected_features = self.feature_selector.transform_one(scaled_features)
                except Exception as e:
                    logger.warning(f"Error in feature selection: {e}")
            
            # Calculate anomaly score
            try:
                anomaly_score = self.anomaly_detector.score_one(selected_features)
                self.anomaly_detector.learn_one(selected_features)
            except Exception as e:
                logger.warning(f"Error in anomaly detection: {e}")
                anomaly_score = 0.0
            
            # Determine if anomalous
            is_anomaly = anomaly_score > self.anomaly_threshold
            
            # Track anomaly
            self.anomaly_history.append((is_anomaly, time.time()))
            
            # Calculate anomaly rate
            anomaly_rate = sum(1 for a, _ in self.anomaly_history if a) / len(self.anomaly_history) if self.anomaly_history else 0
            
            return {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': is_anomaly,
                'anomaly_rate': anomaly_rate,
                'feature_count': len(selected_features)
            }
        
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {
                'anomaly_score': 0.0,
                'is_anomaly': False,
                'anomaly_rate': 0.0,
                'error': str(e)
            }
    
    def update_feature_importance(self, features: Dict[str, float], target: float) -> Dict[str, float]:
        """
        Update feature importance based on correlation with target.
        
        Args:
            features: Feature dictionary
            target: Target value
            
        Returns:
            Feature importance scores
        """
        try:
            # Initialize feature importance tracking if needed
            for feature in features:
                if feature not in self.feature_importance:
                    self.feature_importance[feature] = {
                        'correlation': stats.RollingPearson(self.feature_window),
                        'importance': 0.0
                    }
            
            # Update correlation statistics
            for feature, value in features.items():
                if isinstance(value, (int, float)) and np.isfinite(value) and feature in self.feature_importance:
                    self.feature_importance[feature]['correlation'].update(value, target)
            
            # Calculate importance scores
            for feature in self.feature_importance:
                try:
                    corr = abs(self.feature_importance[feature]['correlation'].get())
                    if np.isnan(corr) or np.isinf(corr):
                        corr = 0.0
                    self.feature_importance[feature]['importance'] = corr
                except Exception as e:
                    logger.warning(f"Error calculating correlation for {feature}: {e}")
                    self.feature_importance[feature]['importance'] = 0.0
            
            # Return current importance scores
            return {f: data['importance'] for f, data in self.feature_importance.items()}
            
        except Exception as e:
            logger.error(f"Feature importance update error: {e}")
            return {}
    
    def preprocess_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Preprocess features for prediction.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Preprocessed features
        """
        try:
            # Filter out non-numeric values
            preprocessed = {}
            for k, v in features.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    preprocessed[k] = float(v)
            
            if not preprocessed:
                return {}
            
            # Apply scaling
            try:
                scaled = self.feature_scaler.transform_one(preprocessed)
            except Exception as e:
                logger.warning(f"Error in feature scaling: {e}")
                scaled = preprocessed
            
            # Apply feature selection if enabled
            if self.enable_feature_selection and self.feature_selector is not None:
                try:
                    return self.feature_selector.transform_one(scaled)
                except Exception as e:
                    logger.warning(f"Error in feature selection: {e}")
            
            return scaled
            
        except Exception as e:
            logger.error(f"Feature preprocessing error: {e}")
            return {}


class CerebellumSNN:
    """
    Cerebellum-inspired spiking neural network using Rockpool.
    
    This implements a biologically-inspired neural network based on
    the cerebellar circuitry for pattern recognition and prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int = 1, 
        n_granule: int = 1000, 
        n_purkinje: int = 100, 
        time_steps: int = 50, 
        dt: float = 1e-3, 
        learning_rate: float = 1e-4
    ):
        """Initialize cerebellar SNN."""
        if not ROCKPOOL_AVAILABLE:
            logger.warning("Rockpool is not available, SNN functionality will be limited")
            self._rockpool_available = False
        else:
            self._rockpool_available = True
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_granule = n_granule
        self.n_purkinje = n_purkinje
        self.n_dcn = output_dim  # One DCN per output
        self.time_steps = time_steps
        self.dt = dt
        self.learning_rate = learning_rate
        
        # Initialize network if Rockpool is available
        if self._rockpool_available:
            self._initialize_network()
        
        # State
        self._is_trained = False
    
    def _initialize_network(self):
        """Initialize the cerebellar network using Rockpool."""
        try:
            # Create neuron layers with appropriate time constants
            # Granule cells: fast response, pattern separation
            self.granule_layer = rnn.aLIFTorch(
                shape=(self.n_granule,),
                tau_mem=rp.Constant(10e-3),  # 10ms membrane time constant
                tau_syn=rp.Constant(5e-3),   # 5ms synaptic time constant
                threshold=rp.Constant(0.8),  # Firing threshold
                dt=self.dt
            )
            
            # Purkinje cells: integrative function
            self.purkinje_layer = rnn.aLIFJax(
                shape=(self.n_purkinje,),
                tau_mem=rp.Constant(15e-3),  # 15ms membrane time constant
                tau_syn=rp.Constant(7e-3),   # 7ms synaptic time constant
                threshold=rp.Constant(0.9),  # Higher threshold
                dt=self.dt
            )
            
            # Deep cerebellar nuclei: output
            self.dcn_layer = rnn.aLIFTorch(
                shape=(self.n_dcn,),
                tau_mem=rp.Constant(20e-3),  # 20ms membrane time constant
                tau_syn=rp.Constant(10e-3),  # 10ms synaptic time constant
                threshold=rp.Constant(0.7),  # Lower threshold
                dt=self.dt
            )
            
            # Initialize connection weights
            # Mossy fibers (input) to granule cells: expansive, random
            self.conn_mf_gc = rnn.LinearTorch(
                shape=(self.input_dim,),
                out_features=self.n_granule,
                weight=rp.Constant(
                    np.random.randn(self.input_dim, self.n_granule) * 0.1
                ),
                has_bias=False
            )
            
            # Granule cells to Purkinje cells: learnable
            self.conn_gc_pc = rnn.LinearTorch(
                shape=(self.n_granule,),
                out_features=self.n_purkinje,
                weight=rp.Trainable(
                    np.random.randn(self.n_granule, self.n_purkinje) * 0.05
                ),
                has_bias=False
            )
            
            # Purkinje cells to DCN: inhibitory, fixed
            self.conn_pc_dcn = rnn.LinearTorch(
                shape=(self.n_purkinje,),
                out_features=self.n_dcn,
                weight=rp.Constant(
                    -np.abs(np.random.randn(self.n_purkinje, self.n_dcn) * 0.2)
                ),
                has_bias=False
            )
            
            # Mossy fibers to DCN: excitatory, direct pathway
            self.conn_mf_dcn = rnn.LinearTorch(
                shape=(self.input_dim,),
                out_features=self.n_dcn,
                weight=rp.Constant(
                    np.abs(np.random.randn(self.input_dim, self.n_dcn) * 0.3)
                ),
                has_bias=False
            )
            
            # Build the network as a sequential model with multiple paths
            self.network = rnn.Sequential([
                # Input processing
                {
                    "to_granule": rnn.Sequential([self.conn_mf_gc, self.granule_layer]),
                    "to_dcn_direct": self.conn_mf_dcn  # Direct pathway
                },
                # Cerebellar processing
                {
                    "granule_to_purkinje": rnn.Sequential([self.conn_gc_pc, self.purkinje_layer])
                },
                # Output
                {
                    "purkinje_to_dcn": rnn.Sequential([self.conn_pc_dcn, self.dcn_layer])
                }
            ])
            
            self._initialized = True
            logger.info("Cerebellar network initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cerebellar network: {e}")
            self._initialized = False
            self._rockpool_available = False
    
    def _encode_input(self, X):
        """Encode input data as currents for SNN."""
        # Normalize input to appropriate range
        X_mean = np.mean(X)
        X_std = np.std(X) + 1e-6
        X_norm = np.clip((X - X_mean) / X_std, -3, 3)
        
        # Scale for neuron activation
        return X_norm * 0.5
    
    def _decode_output(self, spikes):
        """Decode output spikes to predictions."""
        if not self._rockpool_available or not self._initialized:
            # Return zeros as fallback
            return np.zeros(self.output_dim)
            
        try:
            # Extract DCN spike rates
            dcn_spikes = spikes["purkinje_to_dcn.spikes"]
            
            # Calculate firing rates
            firing_rates = np.mean(dcn_spikes[-10:], axis=0)  # Use last 10 timesteps
            
            # Normalize to [0, 1] range
            max_rate = np.max(firing_rates) if np.max(firing_rates) > 0 else 1
            predictions = firing_rates / max_rate
            
            return predictions
        except Exception as e:
            logger.error(f"Error decoding SNN output: {e}")
            return np.zeros(self.output_dim)
    
    def predict(self, X):
        """Generate predictions with the cerebellar SNN."""
        if not self._rockpool_available or not self._initialized:
            # Fallback to basic prediction if Rockpool not available
            return np.random.random(self.output_dim)
        
        try:
            # Ensure X has correct shape
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Process each sample
            predictions = []
            for sample in X:
                # Encode input
                encoded = self._encode_input(sample)
                
                # Repeat for time steps (static input)
                input_sequence = np.tile(encoded, (self.time_steps, 1))
                
                # Run network simulation
                self.network.reset_state()
                outputs, _ = self.network(input_sequence)
                
                # Decode predictions
                pred = self._decode_output(outputs)
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"Error in cerebellar prediction: {e}")
            return np.zeros((len(X), self.output_dim))
    
    def fit(self, X, y, epochs=10):
        """Train the cerebellar network."""
        if not self._rockpool_available or not self._initialized:
            logger.warning("Rockpool not available, cannot train SNN")
            return self
        
        try:
            # Create simple optimizer
            optimizer = rt.Adam(self.learning_rate)
            
            # Define loss function
            def loss_fn(model, x_batch, y_batch):
                # Forward pass
                model.reset_state()
                outputs, _ = model(x_batch)
                
                # Decode predictions
                preds = self._decode_output(outputs)
                
                # MSE loss
                return np.mean((preds - y_batch) ** 2)
            
            # Train for specified epochs
            for epoch in range(epochs):
                epoch_loss = 0
                
                # Process each sample
                for i, (x, y) in enumerate(zip(X, y)):
                    # Encode input
                    encoded_x = self._encode_input(x)
                    input_sequence = np.tile(encoded_x, (self.time_steps, 1))
                    
                    # Compute loss and update
                    loss, gradients = rt.loss_and_gradient(
                        loss_fn, self.network, input_sequence, y, dt=self.dt
                    )
                    
                    # Apply gradients
                    optimizer.update(gradients)
                    
                    epoch_loss += loss
                
                # Log progress
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X):.6f}")
            
            self._is_trained = True
            
        except Exception as e:
            logger.error(f"Error training cerebellar network: {e}")
        
        return self
    
    @property
    def is_trained(self):
        """Check if model is trained."""
        return self._is_trained

logger = logging.getLogger("QuantumOptimizer")
class QuantumOptimizer:
    """
    Quantum-inspired optimization for model parameters.
    
    This implements a quantum-inspired optimization algorithm:
    - Simulates quantum annealing effects
    - Uses interference patterns for efficient parameter space exploration
    - Finds global optima in complex parameter spaces
    """
    
    def __init__(
        self,
        param_count: int = 5,
        iterations: int = 100,
        temperature_schedule: Optional[List[float]] = None,
        use_catalyst: bool = True
    ):
        """Initialize quantum optimizer."""
        self.param_count = param_count
        self.iterations = iterations
        self.use_catalyst = use_catalyst and CATALYST_AVAILABLE
        
        # Set temperature schedule for simulated annealing
        if temperature_schedule is None:
            # Default cooling schedule
            self.temperature_schedule = np.linspace(2.0, 0.01, iterations)
        else:
            self.temperature_schedule = temperature_schedule
        
        # Initialize quantum phases for interference
        self.phases = np.random.uniform(0, 2*np.pi, param_count)
        
        # Initialize hardware manager if available
        try:
            self.hardware_manager = HardwareManager()
            self._has_hardware_manager = True
        except:
            logger.warning("Hardware manager not available")
            self.hardware_manager = None
            self._has_hardware_manager = False
    
    def _quantum_update(self, params, gradient, temperature, phases):
        """
        Apply quantum-inspired update to parameters.
        
        Args:
            params: Current parameters
            gradient: Parameter gradients
            temperature: Current annealing temperature
            phases: Quantum phases
            
        Returns:
            Updated parameters
        """
        # Calculate quantum interference
        interference = np.exp(1j * phases)
        
        # Apply to gradient with temperature
        scaled_gradient = gradient * temperature
        
        # Apply quantum effects to gradient
        quantum_gradient = scaled_gradient * np.abs(interference)
        
        # Update parameters
        new_params = params - quantum_gradient
        
        return new_params
    
    def _calculate_energy(self, cost_fn, params):
        """
        Calculate energy (cost) of parameters.
        
        Args:
            cost_fn: Cost function
            params: Parameters to evaluate
            
        Returns:
            Cost value
        """
        try:
            return cost_fn(params)
        except Exception as e:
            logger.error(f"Error in cost function: {e}")
            return float('inf')
    
    def _quantum_annealing_step(self, cost_fn, params, temperature, phases):
        """
        Perform one step of quantum annealing.
        
        Args:
            cost_fn: Cost function
            params: Current parameters
            temperature: Current temperature
            phases: Quantum phases
            
        Returns:
            Updated parameters and cost
        """
        # Calculate current cost
        current_cost = self._calculate_energy(cost_fn, params)
        
        # Quantum tunneling effect (randomly jump in parameter space)
        if np.random.random() < 0.1:
            # Random tunneling
            tunnel_distance = temperature * 0.1
            params_tunnel = params + np.random.uniform(-tunnel_distance, tunnel_distance, params.shape)
            
            # Evaluate tunneled position
            tunnel_cost = self._calculate_energy(cost_fn, params_tunnel)
            
            # Accept tunneling if better
            if tunnel_cost < current_cost:
                return params_tunnel, tunnel_cost
        
        # Calculate numerical gradient
        gradient = np.zeros_like(params)
        delta = 1e-6
        
        for i in range(len(params)):
            # Perturb parameter
            params_plus = params.copy()
            params_plus[i] += delta
            
            # Calculate gradient
            cost_plus = self._calculate_energy(cost_fn, params_plus)
            gradient[i] = (cost_plus - current_cost) / delta
        
        # Apply quantum-inspired update
        new_params = self._quantum_update(params, gradient, temperature, phases)
        
        # Calculate new cost
        new_cost = self._calculate_energy(cost_fn, new_params)
        
        return new_params, new_cost
    
    def optimize(self, cost_fn: Callable, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Optimize parameters using quantum-inspired annealing.
        
        Args:
            cost_fn: Function that takes parameters and returns cost (lower is better)
            initial_params: Starting parameter values
            
        Returns:
            Dict with optimized parameters and metadata
        """
        # Ensure correct dimensions
        params = np.array(initial_params).flatten()
        if len(params) != self.param_count:
            raise ValueError(f"Expected {self.param_count} parameters, got {len(params)}")
        
        # Initialize tracking
        best_params = params.copy()
        best_cost = self._calculate_energy(cost_fn, params)
        costs = [best_cost]
        
        # Run annealing process
        for i in range(self.iterations):
            # Get current temperature
            temperature = self.temperature_schedule[i]
            
            # Perform annealing step
            params, current_cost = self._quantum_annealing_step(
                cost_fn, params, temperature, self.phases
            )
            
            # Record cost
            costs.append(current_cost)
            
            # Update best parameters
            if current_cost < best_cost:
                best_params = params.copy()
                best_cost = current_cost
            
            # Update phases
            self.phases += np.random.uniform(-0.1, 0.1, self.phases.shape)
            
            # Log progress periodically
            if (i + 1) % 20 == 0:
                logger.info(f"Iteration {i+1}/{self.iterations}, Cost: {current_cost:.6f}, Best: {best_cost:.6f}")
        
        # Return results
        return {
            'params': best_params,
            'cost': best_cost,
            'cost_history': costs,
            'initial_cost': costs[0],
            'improvement': (costs[0] - best_cost) / (abs(costs[0]) + 1e-10)
        }

logger = logging.getLogger("Q*RiverPredictor")

class QStarRiverPredictor:
    """
    QStar-River trading predictor with Numba and Catalyst optimization.
    
    This class integrates Q* reinforcement learning, River ML online learning,
    spiking neural networks, and quantum computing for advanced trading predictions.
    """
    
    def __init__(
        self,
        cerebellum_snn: CerebellumSNN = None,
        use_quantum_representation: bool = True,
        initial_states: int = 200,
        initial_actions: int = 7,
        experience_buffer_size: int = 20000,
        learning_rate: float = 0.05,
        discount_factor: float = 0.95,
        training_episodes: int = 200,
        batch_size: int = 64,
        river_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize QStar-River predictor.
        
        Args:
            river_ml: RiverML instance for online learning
            cerebellum_snn: Cerebellum SNN instance
            use_quantum_representation: Whether to use quantum representation
            initial_states: Initial number of states
            initial_actions: Initial number of actions
            experience_buffer_size: Size of experience buffer
            learning_rate: Initial learning rate
            discount_factor: Discount factor for future rewards
            training_episodes: Maximum training episodes
            batch_size: Batch size for experience replay
            river_config (Optional[Dict[str, Any]]): Configuration dictionary
                                      for RiverOnlineML.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize state flags
        self._is_initializing = True
        self._is_trained = False
        
        # Initialize components
        self.river_ml = None
        if RiverOnlineML.RIVER_AVAILABLE: # Check static flag on the imported class
            try:
                # Pass the config dict, use {} if None
                self.river_ml = RiverOnlineML(**(river_config or {}))
                # Check if initialization within RiverOnlineML was successful
                if not self.river_ml.is_initialized:
                     self.logger.warning("External RiverOnlineML initialized but failed internal setup. Disabling River features.")
                     self.river_ml = None # Set back to None if init failed
                else:
                     self.logger.info("External RiverOnlineML component initialized successfully.")
            except Exception as e:
                 self.logger.error(f"Failed to instantiate external RiverOnlineML: {e}", exc_info=True)
                 self.river_ml = None # Ensure it's None on instantiation error
                 
        self.cerebellum_snn = cerebellum_snn
        
        # Initialize hardware manager for quantum operations
        try:
            self.hardware_manager = HardwareManager()
            self._has_hardware_manager = True
        except:
            self.logger.warning("Hardware manager not available")
            self.hardware_manager = None
            self._has_hardware_manager = False
        
        # Initialize Q* Learning agent
        self.agent = QStarLearningAgent(
            states=initial_states,
            actions=initial_actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=1.0,
            min_exploration_rate=0.05,
            exploration_decay_rate=0.99,
            use_adaptive_learning_rate=True,
            use_experience_replay=True,
            experience_buffer_size=experience_buffer_size,
            batch_size=batch_size,
            max_episodes=training_episodes,
            max_steps_per_episode=10000,  # Long episodes for backtesting
            use_quantum_representation=use_quantum_representation
        )
        
        # Trading environment
        self.env = None
        self.river_ml = None # Example: Placeholder for River ML components object
        self.hyperparameters = {}
        self.feature_columns = []
        self.feature_stats = {} # For normalization
        self.market_embeddings = {}
        self.quantum_params = {}
        self.performance_metrics = {}
        self.preprocessors = {}
        self.model_state = {} # General model state bucket
        self._is_trained = False
        self.quantum_phases = None # From original save
        self.backtest_results = None # From original save
        # Phases for quantum circuits
        self.quantum_phases = np.random.uniform(0, 2*np.pi, 10)
        
        # Initialize model state
        self.model_state = {}
        
        # Set initialization complete
        self._is_initializing = False
        
        # For backward compatibility
        self.is_trained = self._is_trained
        self.backtest_results = None
        
        self.logger.info(f"QStar-River Predictor initialized. River ML {'Enabled' if self.river_ml else 'Disabled'}.")
    
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
            
    
    def train(self, price_data: pd.DataFrame, window_size: int = 50,
             initial_capital: float = 10000.0, transaction_fee: float = 0.001,
             epochs: int = 3) -> Dict[str, Any]:
        """
        Train the QStar-River agent, preserving structure and handling epoch errors.
        Saves and loads the best agent state found across epochs.
        Args:
            price_data: Historical price data
            window_size: Window size for features
            initial_capital: Initial capital
            transaction_fee: Transaction fee
            epochs: Number of training epochs
        Returns:
            Dictionary of performance metrics from the evaluation of the best agent.
        """
        self.logger.info(f"Starting QStar-River Training: Data Points={len(price_data)}, Epochs={epochs}")

        # --- Initial Setup & Validation ---
        if 'close' not in price_data.columns:
            self.logger.error("Price data must contain 'close' column")
            return {'error': "Price data must contain 'close' column"}
        if len(price_data) <= window_size:
            self.logger.error(f"Not enough data points. Need > {window_size}, got {len(price_data)}")
            return {'error': f"Insufficient data points"}

        # --- Variables to track best state across epochs ---
        best_epoch_info = {
            'epoch': -1,
            'agent_path': None,
            'return': -float('inf'),
            'metrics': None # Store full metrics dict of the best epoch
        }
        temp_files_to_clean = [] # List to hold temp file paths for cleanup
        final_metrics = {'error': 'Training did not complete successfully or find a best agent.'} # Default return
        was_trained_flag = False # Track if any epoch succeeded and loaded best agent

        try:
            # --- Environment Setup ---
            # Wrap env creation in try-except as well
            try:
                self.env = TradingEnvironment(
                    price_data=price_data, price_column='close', window_size=window_size,
                    initial_capital=initial_capital, transaction_fee=transaction_fee
                )
                self.logger.info("Trading environment created successfully.")
                # Optional: Agent dimension check/resize here if needed
            except Exception as e:
                 self.logger.error(f"Failed to create TradingEnvironment: {e}", exc_info=True)
                 return {'error': f"Environment creation failed: {e}"}


            # --- Train Cerebellum (Optional, placeholder) ---
            # if self.cerebellum_snn is not None:
            #     self.logger.info("Attempting Cerebellum pre-training...")
            #     # ... Cerebellum training logic ...

            # --- Epoch Training Loop ---
            for epoch in range(epochs):
                self.logger.info(f"--- Starting Training Epoch {epoch + 1}/{epochs} ---")
                epoch_epoch_converged = False # Reset convergence flag for the epoch

                try:
                    # --- Q* Agent Training ---
                    # Ensure the agent's train method itself handles its internal errors
                    # The Numba fix in quantum_action_selection is critical here
                    epoch_epoch_converged, epoch_episodes = self.agent.train(self.env)
                    self.logger.info(f"Epoch {epoch + 1} Q* training attempted: epoch_converged={epoch_epoch_converged}, Episodes Run={epoch_episodes}")

                    # --- Evaluate Current Agent Performance ---
                    # Run a clean evaluation using the agent state *after* training this epoch
                    current_eval_metrics = self.agent.evaluate(self.env, num_episodes=1) # Quick eval
                    current_env_metrics = self.env.get_performance_metrics()
                    current_return = current_env_metrics.get("total_return", -float('inf')) # Get return safely

                    self.logger.info(f"Epoch {epoch+1} Post-Train Eval: Return={current_return:.4f}, "
                                     f"WinRate={current_env_metrics.get('win_rate', 0):.2f}, "
                                     f"Trades={current_env_metrics.get('total_trades', 0)}")

                    # --- Save Agent if Best Performance So Far ---
                    if current_return > best_epoch_info['return']:
                        self.logger.info(f"*** New Best Performance Found in Epoch {epoch + 1} (Return: {current_return:.4f}) ***")
                        best_epoch_info['return'] = current_return
                        best_epoch_info['epoch'] = epoch + 1
                        best_epoch_info['metrics'] = current_env_metrics # Store the metrics dict

                        # Save agent state to a unique temporary file
                        import tempfile
                        import os
                        fd, temp_path = tempfile.mkstemp(suffix='.pkl', prefix=f'best_agent_epoch_{epoch+1}_')
                        os.close(fd)
                        try:
                            self.agent.save(temp_path) # Save the current agent state
                            self.logger.info(f"Agent state saved temporarily to {temp_path}")
                            # If there was a previous best file, plan to remove it later
                            if best_epoch_info['agent_path'] and best_epoch_info['agent_path'] != temp_path:
                                 temp_files_to_clean.append(best_epoch_info['agent_path'])
                            best_epoch_info['agent_path'] = temp_path # Update the path to the new best
                        except Exception as save_err:
                             self.logger.error(f"Failed to save agent state for epoch {epoch+1}: {save_err}", exc_info=True)
                             # Continue without saving this state as best

                except Exception as e:
                    # This catches errors within the epoch's train/eval/save block
                    # Critical Numba/JAX errors in agent.train might land here
                    self.logger.error(f"Error during training epoch {epoch + 1}: {e}", exc_info=True)
                    # Continue to the next epoch

            # --- Post-Training: Load the Best Saved Agent State ---
            if best_epoch_info['agent_path'] and os.path.exists(best_epoch_info['agent_path']):
                try:
                    self.logger.info(f"Loading best agent state from {best_epoch_info['agent_path']} (Epoch {best_epoch_info['epoch']})")
                    # --- CRITICAL: Load the agent state ---
                    self.agent = QStarLearningAgent.load(best_epoch_info['agent_path'])
                    # --- Mark predictor as trained ---
                    self._is_trained = True
                    self.is_trained = True # Keep for compatibility if checked elsewhere
                    was_trained_flag = True
                    # Assign the stored metrics from the best epoch as the final result
                    final_metrics = best_epoch_info['metrics']
                    if final_metrics is None: # Should not happen if save logic worked
                         self.logger.warning("Best agent loaded, but metrics were not stored. Re-evaluating...")
                         final_eval = self.agent.evaluate(self.env, num_episodes=1)
                         final_metrics = self.env.get_performance_metrics()
                         final_metrics['eval_reward'] = final_eval.get('mean_reward', 0)
                    # Add info about which epoch was best
                    final_metrics['best_epoch_num'] = best_epoch_info['epoch']
                    self.backtest_results = final_metrics # Store final metrics


                    self.logger.info("Best agent state loaded successfully.")
                except Exception as e:
                    self.logger.error(f"Error loading best agent state from {best_epoch_info['agent_path']}: {e}", exc_info=True)
                    # Keep _is_trained as False, return default error metrics
                    final_metrics = {'error': f'Failed to load best agent state: {e}'}
            else:
                 self.logger.warning("No best agent state was saved or found after training.")
                 self._is_trained = False
                 self.is_trained = False
                 # Return default error metrics


            # --- Cleanup ---
            # Add the loaded best agent file to the cleanup list IF it was successfully loaded
            if was_trained_flag and best_epoch_info['agent_path']:
                 temp_files_to_clean.append(best_epoch_info['agent_path'])
            # Clean up all tracked temporary files
            for f_path in temp_files_to_clean:
                 try:
                     if os.path.exists(f_path):
                         os.unlink(f_path)
                         self.logger.debug(f"Cleaned up temporary file: {f_path}")
                 except OSError as e:
                     self.logger.warning(f"Could not delete temporary agent file {f_path}: {e}")

            self.logger.info(f"Training process completed. Model is_trained: {self._is_trained}")
            return final_metrics, epoch_converged, episodes_completed# Return metrics dictionary

        except Exception as e:
            # Catch errors during the overall setup (e.g., env creation)
            self.logger.error(f"Fatal error during training setup: {e}", exc_info=True)
            import traceback
            return {"error": f"Fatal training setup error: {e}", "details": traceback.format_exc()}

#from qstar_river
    def _run_backtest(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on historical data with improved error handling.
        
        Args:
            price_data: Historical price data
            
        Returns:
            Backtest metrics
        """
        # Default return value in case of error
        default_results = {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "final_portfolio": self.env.initial_capital if hasattr(self.env, 'initial_capital') else 10000.0,
            "actions": [],
            "positions": [],
            "portfolio_values": []
        }
        
        try:
            # Reset environment for clean backtest
            self.env.reset()
            
            # Set agent to no exploration for backtest
            original_exploration = self.agent.exploration_rate
            self.agent.exploration_rate = 0.0
            
            # Tracking variables
            state = self.env.reset()
            done = False
            actions = []
            positions = []
            portfolio_values = []
            
            # Safety counter to prevent infinite loops
            max_steps = len(price_data) * 2
            step_count = 0
            
            while not done and step_count < max_steps:
                # Choose action from trained policy
                action = self.agent.choose_action(state)
                actions.append(action)
                
                # Take step in environment with error handling
                try:
                    next_state, reward, done = self.env.step(action)
                    
                    # Track positions and portfolio
                    positions.append(self.env.position)
                    portfolio_values.append(self.env.portfolio_value)
                    
                    # Move to next state
                    state = next_state
                except Exception as e:
                    self.logger.error(f"Error during backtest step: {str(e)}", exc_info=True)
                    done = True
                
                step_count += 1
            
            # Restore exploration rate
            self.agent.exploration_rate = original_exploration
            
            # Get performance metrics
            try:
                metrics = self.env.get_performance_metrics()
            except Exception as e:
                self.logger.error(f"Error getting performance metrics: {str(e)}", exc_info=True)
                metrics = default_results
            
            # Add actions and positions to results
            metrics["actions"] = actions
            metrics["positions"] = positions
            metrics["portfolio_values"] = portfolio_values
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}", exc_info=True)
            return default_results

            ##end if import

    def _extract_features(self, df_window: pd.DataFrame) -> np.ndarray:
        """
        Extract features from price window.
        
        Args:
            df_window: Price data window
            
        Returns:
            Feature vector
        """
        try:
            # Check if we have close prices
            if 'close' not in df_window.columns or len(df_window) < 2:
                return np.array([])
            
            # Extract price data
            close_prices = df_window['close'].values
            
            # Calculate returns
            returns = np.diff(close_prices) / close_prices[:-1]
            
            # Basic features
            features = np.zeros(10)
            
            # Return features
            features[0] = np.mean(returns[-5:]) if len(returns) >= 5 else 0  # 5-period return
            features[1] = np.mean(returns[-10:]) if len(returns) >= 10 else 0  # 10-period return
            
            # Volatility
            features[2] = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.1  # Annualized volatility
            
            # Price momentum
            features[3] = (close_prices[-1] / close_prices[0] - 1) if len(close_prices) > 1 else 0  # Overall return
            
            # Trend features
            if len(close_prices) >= 20:
                sma_5 = np.mean(close_prices[-5:])
                sma_20 = np.mean(close_prices[-20:])
                features[4] = sma_5 / sma_20 - 1  # SMA ratio
            else:
                features[4] = 0
            
            # Volume features if available
            if 'volume' in df_window.columns:
                volume = df_window['volume'].values
                features[5] = np.mean(volume[-5:]) / np.mean(volume) if len(volume) > 0 and np.mean(volume) > 0 else 1.0  # Recent volume
            else:
                features[5] = 1.0
            
            # Technical indicators if available
            # RSI (simplified)
            if len(returns) >= 14:
                up_moves = np.array([max(0, r) for r in returns[-14:]])
                down_moves = np.array([max(0, -r) for r in returns[-14:]])
                
                avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0.0001
                avg_down = np.mean(down_moves) if len(down_moves) > 0 else 0.0001
                
                rs = avg_up / avg_down
                rsi = 100 - (100 / (1 + rs))
                
                features[6] = rsi / 100  # Normalize to [0, 1]
            else:
                features[6] = 0.5
            
            # Placeholder for more features
            features[7] = 0.5
            features[8] = 0.5
            features[9] = 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return np.zeros(10)  # Return zeros as fallback

    # Inside QStarRiverPredictor class (pulsar.py)
    def predict(self, df_window: pd.DataFrame) -> dict:
        """
        Generate prediction based on market data, fixing errors while preserving
        original structure and variable availability.

        Args:
            df_window: Market data window

        Returns:
            Prediction dictionary with action, confidence, and component outputs.
        """
        # --- Default values for all expected return keys ---
        # Initialize structure to match expected output, prevents KeyErrors downstream
        qstar_action = 2 # Default HOLD
        cerebellum_prediction = None
        quantum_prediction = None # Assuming this was intended
        is_anomaly = False
        anomaly_score = 0.0
        drift_detected = False # IMPORTANT: Add the missing drift_detected key
        confidence = 0.0
        explanation = "Prediction failed or model not ready." # Default explanation
        final_action = 2 # Default HOLD
        action_name = 'HOLD'
        error_log = [] # To collect non-fatal errors

        try:
            # --- Stage 1: Check Training Status (Corrected Check) ---
            # Use the predictor's own flag
            if not self._is_trained:
                explanation = 'Model not trained'
                error_log.append(explanation)
                # Return the full dictionary structure with default values
                return {
                    'action': final_action, 'action_name': action_name, 'confidence': confidence,
                    'explanation': explanation, 'qstar_action': qstar_action,
                    'cerebellum_prediction': cerebellum_prediction, 'quantum_prediction': quantum_prediction,
                    'anomaly_detected': is_anomaly, 'anomaly_score': anomaly_score,
                    'drift_detected': drift_detected, 'error_log': error_log # Add error log
                }

            # --- Stage 2: Feature Extraction ---
            try:
                features = self._extract_features(df_window)
                if len(features) == 0:
                    explanation = 'Failed to extract features.'
                    error_log.append(explanation)
                    # Return default state, but preserving structure
                    return {
                        'action': final_action, 'action_name': action_name, 'confidence': confidence,
                        'explanation': explanation, 'qstar_action': qstar_action,
                        'cerebellum_prediction': cerebellum_prediction, 'quantum_prediction': quantum_prediction,
                        'anomaly_detected': is_anomaly, 'anomaly_score': anomaly_score,
                        'drift_detected': drift_detected, 'error_log': error_log
                    }
            except Exception as e:
                self.logger.error(f"Error during feature extraction: {e}", exc_info=True)
                explanation = f'Feature extraction error: {e}'
                error_log.append(explanation)
                # Return default state
                return {
                    'action': final_action, 'action_name': action_name, 'confidence': confidence,
                    'explanation': explanation, 'qstar_action': qstar_action,
                    'cerebellum_prediction': cerebellum_prediction, 'quantum_prediction': quantum_prediction,
                    'anomaly_detected': is_anomaly, 'anomaly_score': anomaly_score,
                    'drift_detected': drift_detected, 'error_log': error_log
                }

            # --- Stage 3: River ML Processing ---
            # Initialize intermediate results to defaults
            river_results_dict = {'is_anomaly': False, 'anomaly_score': 0.0, 'drift_detected': False}
            try:
                if self.river_ml:
                    feature_dict = {f'feature_{i}': v for i, v in enumerate(features)}
                    # Assume anomaly detection provides drift info or call detect_drift separately
                    anomaly_results = self.river_ml.detect_anomalies(feature_dict) # This might fail if scaler is None
                    river_results_dict['is_anomaly'] = anomaly_results.get('is_anomaly', False)
                    river_results_dict['anomaly_score'] = anomaly_results.get('anomaly_score', 0.0)
                    # Placeholder: Update drift_detected if your River adapter provides it
                    # drift_info = self.river_ml.detect_drift(...)
                    # river_results_dict['drift_detected'] = drift_info.get('drift_detected', False)
                else:
                     self.logger.warning("RiverML component not available or not initialized during predict.")

                # Assign results to top-level variables AFTER successful processing
                is_anomaly = river_results_dict['is_anomaly']
                anomaly_score = river_results_dict['anomaly_score']
                drift_detected = river_results_dict['drift_detected']

            except Exception as e:
                 # Log error but continue with default River values (False/0.0)
                 self.logger.error(f"Error during RiverML processing: {e}", exc_info=True)
                 error_log.append(f"RiverML Error: {e}")
                 # is_anomaly, anomaly_score, drift_detected retain defaults


            # --- Stage 4: State Calculation ---
            try:
                state = calculate_state_index(features, self.agent.states)
            except Exception as e:
                 self.logger.error(f"Error calculating state index: {e}", exc_info=True)
                 explanation = f'State calculation error: {e}'
                 error_log.append(explanation)
                 # Return default state, critical failure
                 return {
                     'action': final_action, 'action_name': action_name, 'confidence': confidence,
                     'explanation': explanation, 'qstar_action': qstar_action,
                     'cerebellum_prediction': cerebellum_prediction, 'quantum_prediction': quantum_prediction,
                     'anomaly_detected': is_anomaly, 'anomaly_score': anomaly_score,
                     'drift_detected': drift_detected, 'error_log': error_log
                 }

            # --- Stage 5: Q* Agent Action and Confidence ---
            try:
                qstar_action = self.agent.choose_action(state) # Assume this doesn't crash after Numba fix
                # Calculate confidence (using numpy conversion for safety if JAX is used)
                q_values = np.asarray(self.agent.q_table[state])
                q_max = np.max(q_values)
                q_min = np.min(q_values)
                qstar_confidence = np.clip((q_max - q_min) / (np.abs(q_max) + np.abs(q_min) + 1e-6), 0, 1)
                # Assign to main confidence variable (will be potentially blended/overridden later)
                confidence = float(qstar_confidence)
            except Exception as e:
                 self.logger.error(f"Error getting Q* action/confidence: {e}", exc_info=True)
                 error_log.append(f"Q* Error: {e}")
                 # Keep default qstar_action and confidence(0.0), but allow processing to continue


            # --- Stage 6: Cerebellum SNN Prediction (Optional) ---
            # cerebellum_prediction remains None unless successfully updated
            try:
                if self.cerebellum_snn is not None and self.cerebellum_snn.is_trained:
                    cerebellum_output = self.cerebellum_snn.predict(features.reshape(1, -1))[0]
                    c_action = np.argmax(cerebellum_output)
                    c_conf = np.max(cerebellum_output)
                    cerebellum_prediction = { # Assign to the variable if successful
                        'action': int(c_action),
                        'confidence': float(np.clip(c_conf, 0, 1)),
                        'scores': cerebellum_output.tolist()
                    }
            except Exception as e:
                 self.logger.error(f"Error during Cerebellum prediction: {e}", exc_info=True)
                 error_log.append(f"Cerebellum Error: {e}")
                 # cerebellum_prediction remains None


            # --- Stage 7: Quantum Prediction (Optional) ---
            # quantum_prediction remains None unless successfully updated
            try:
                 if self._has_hardware_manager and QUANTUM_AVAILABLE:
                    # ... (Actual quantum logic would go here) ...
                    # If successful:
                    # quantum_prediction = {'action': ..., 'confidence': ...}
                    pass
            except Exception as e:
                 self.logger.error(f"Error during Quantum prediction: {e}", exc_info=True)
                 error_log.append(f"Quantum Error: {e}")
                 # quantum_prediction remains None


            # --- Stage 8: Blending and Final Decision ---
            # Start with Q* action and its confidence
            final_action = qstar_action
            # Blending logic needs to use qstar_confidence, cerebellum_prediction['confidence'] etc.
            # Example: Simple override logic (keeping original variable names)
            explanation_parts = [f"Q*->{TradingAction.get_action_name(final_action)}({confidence:.2f})"]
            if cerebellum_prediction and cerebellum_prediction['confidence'] > 0.7 and cerebellum_prediction['confidence'] > confidence:
                 explanation_parts.append(f"Cereb->{TradingAction.get_action_name(cerebellum_prediction['action'])}({cerebellum_prediction['confidence']:.2f}) overriding")
                 final_action = cerebellum_prediction['action']
                 confidence = cerebellum_prediction['confidence'] # Update confidence if overridden
            # Add quantum blending if quantum_prediction is not None...

            # --- Anomaly Override ---
            anomaly_threshold_override = 0.85
            if is_anomaly and anomaly_score > anomaly_threshold_override:
                explanation_parts.append(f"Anomaly OV->HOLD (Score:{anomaly_score:.2f})")
                final_action = 2  # HOLD
                confidence = 0.95 # Assign high confidence

            action_name = TradingAction.get_action_name(final_action)
            explanation = ". ".join(explanation_parts)
            if drift_detected: explanation += ". Drift Detected"


            # --- Final Assembly of Return Dictionary ---
            # Update the dictionary with the final computed values
            return_dict = {
                'action': int(final_action),
                'action_name': action_name,
                'confidence': float(confidence),
                'explanation': explanation,
                'qstar_action': int(qstar_action), # The original Q* proposal
                # Keep the component predictions as they are (can be None)
                'cerebellum_prediction': cerebellum_prediction,
                'quantum_prediction': quantum_prediction,
                'anomaly_detected': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'drift_detected': drift_detected, # Ensure this key is present
            }
            # Include error log if any non-fatal errors occurred
            if error_log:
                 return_dict['error_log'] = error_log

            return return_dict

        except Exception as e:
            # Catch unexpected fatal errors in the main flow
            self.logger.error(f"Unhandled error in predict method: {e}", exc_info=True)
            # Return the default structure but update explanation/error
            return {
                'action': 2, 'action_name': 'HOLD', 'confidence': 0.0,
                'explanation': f"Prediction failed: {e}",
                'qstar_action': 2, 'cerebellum_prediction': None, 'quantum_prediction': None,
                'anomaly_detected': False, 'anomaly_score': 0.0, 'drift_detected': False,
                'error_log': [f"Unhandled predict error: {e}"]
            }
    
    def save(self, filepath: str, custom_filename: Optional[str] = None) -> bool:
        """
        Save the comprehensive model state to a single pickle file.

        Args:
            filepath: Base directory or full file path for saving.
                      If a directory, a standard filename will be used.
                      If a full path, that path will be used.
            custom_filename: DEPRECATED. Included for signature compatibility,
                             but `filepath` should specify the full desired path.

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # --- Determine Final Save Path ---
            save_path = Path(filepath).resolve()
            if save_path.is_dir():
                # If filepath is a directory, create standard filename
                models_dir = save_path
                filename = models_dir / "qstar_model_state.pkl" # Standard name
                self.logger.warning(f"Provided filepath '{filepath}' is a directory. Saving to standard file: {filename}")
            elif custom_filename:
                 # If custom_filename is still provided, prioritize filepath but log warning
                 filename = save_path
                 self.logger.warning(f"Both filepath and custom_filename provided. Using filepath: {filename}")
            else:
                # filepath is treated as the full desired path
                filename = save_path
                models_dir = filename.parent

            # Create directory if it doesn't exist
            models_dir.mkdir(parents=True, exist_ok=True)

            # --- Create State Dictionary ---
            current_time = datetime.now()
            state_dict = {
                'timestamp': current_time.strftime("%Y%m%d_%H%M%S"),
                'version': getattr(self, 'VERSION', '1.0.0'), # Add version if exists
                'metadata': {
                    'saved_at': current_time.isoformat(),
                    'description': 'Comprehensive QStar Model State', # Updated description
                },
                # Include attributes from original 'save' metadata
                '_is_trained': getattr(self, '_is_trained', False),
                'quantum_phases': getattr(self, 'quantum_phases', None),
                'backtest_results': getattr(self, 'backtest_results', None),
                'model_state': getattr(self, 'model_state', {}), # Also present in V2
            }

            # Include attributes from 'save_state' (checking existence)
            if hasattr(self, 'river_ml') and self.river_ml is not None:
                river_state = {}
                # Save River components carefully
                for attr_name in ['regression_models', 'classification_models',
                                  'drift_detectors', 'anomaly_detectors', 'feature_selectors',
                                  'statistics', 'feature_window', 'drift_sensitivity',
                                  'anomaly_threshold']:
                     if hasattr(self.river_ml, attr_name):
                         river_state[attr_name] = getattr(self.river_ml, attr_name)
                state_dict['river_ml_state'] = river_state

            if hasattr(self, 'agent'):
                # Save agent state directly in the dictionary
                if hasattr(self.agent, 'q_table'):
                    state_dict['q_table'] = self.agent.q_table
                agent_params_to_save = {}
                for param in ['learning_rate', 'discount_factor', 'exploration_rate',
                              'min_exploration_rate', 'exploration_decay_rate']:
                    if hasattr(self.agent, param):
                        agent_params_to_save[param] = getattr(self.agent, param)
                if agent_params_to_save:
                     state_dict['agent_params'] = agent_params_to_save

            # Save other common attributes
            for attr_name in ['hyperparameters', 'feature_columns', 'feature_stats',
                              'market_embeddings', 'quantum_params',
                              'performance_metrics', 'preprocessors']:
                 if hasattr(self, attr_name):
                     state_dict[attr_name] = getattr(self, attr_name)

            # --- Safe Write (Temp File + Rename) ---
            temp_filename = filename.with_suffix(".tmp")
            with open(temp_filename, 'wb') as f:
                pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol

            os.replace(temp_filename, filename) # Atomic replace on Unix-like systems

            self.logger.info(f"Comprehensive model state successfully saved to {filename}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving comprehensive model state to {filename}: {e}", exc_info=True)
            return False

    # --- MERGED LOAD METHOD ---
    @classmethod
    def load(cls, filepath: str): # Use filepath as in original load
        """
        Load comprehensive model state from a single pickle file.

        Args:
            filepath: Path to the saved model state file.

        Returns:
            Loaded model instance of cls.
        """
        filename = Path(filepath).resolve()
        if not filename.exists():
            # Log using the class's logger if available, otherwise use global logger
            class_logger = logging.getLogger(f"{__name__}.{cls.__name__}")
            class_logger.error(f"Model state file not found: {filename}")
            raise FileNotFoundError(f"Model state file not found: {filename}")

        try:
            # Load the state dictionary
            with open(filename, 'rb') as f:
                state_dict = pickle.load(f)

            # Create a new instance of the class
            # Pass config if it's stored or if cls requires it (needs adjustment based on your __init__)
            # Assuming basic init works, or load config from state_dict if saved
            model_config = state_dict.get('config', None) # Example if config was saved
            model = cls(config=model_config)

            # --- Restore State Attributes Safely ---
            model.logger.info(f"Attempting to load model state from {filename}...")
            saved_at = state_dict.get('metadata', {}).get('saved_at', 'N/A')
            model.logger.info(f"  State saved at: {saved_at}")

            # Attributes from original 'load' metadata
            model._is_trained = state_dict.get('_is_trained', False)
            # Ensure is_trained property reflects _is_trained if it exists
            if hasattr(model, 'is_trained'):
                 model.is_trained = model._is_trained
            model.quantum_phases = state_dict.get('quantum_phases', None)
            model.backtest_results = state_dict.get('backtest_results', None)
            model.model_state = state_dict.get('model_state', {})

            # Attributes from 'load_state' logic
            # River ML state
            if 'river_ml_state' in state_dict and hasattr(model, 'river_ml') and model.river_ml is not None:
                model.logger.debug("  Restoring River ML state...")
                river_state = state_dict['river_ml_state']
                for attr_name in ['regression_models', 'classification_models',
                                  'drift_detectors', 'anomaly_detectors', 'feature_selectors',
                                  'statistics', 'feature_window', 'drift_sensitivity',
                                  'anomaly_threshold']:
                    if attr_name in river_state and hasattr(model.river_ml, attr_name):
                        setattr(model.river_ml, attr_name, river_state[attr_name])

            # Agent state
            if hasattr(model, 'agent') and model.agent is not None:
                 model.logger.debug("  Restoring Agent state...")
                 if 'q_table' in state_dict and hasattr(model.agent, 'q_table'):
                     model.agent.q_table = state_dict['q_table']
                 if 'agent_params' in state_dict:
                     for key, value in state_dict['agent_params'].items():
                         if hasattr(model.agent, key):
                             setattr(model.agent, key, value)
            elif 'q_table' in state_dict or 'agent_params' in state_dict:
                 model.logger.warning("  Agent state found in file, but 'self.agent' is None or missing.")


            # Other common attributes
            for attr_name in ['hyperparameters', 'feature_columns', 'feature_stats',
                              'market_embeddings', 'quantum_params',
                              'performance_metrics', 'preprocessors']:
                 if attr_name in state_dict and hasattr(model, attr_name):
                      model.logger.debug(f"  Restoring {attr_name}...")
                      setattr(model, attr_name, state_dict[attr_name])

            model.logger.info(f"Model loaded successfully from {filename}")
            return model

        except FileNotFoundError:
             # Log using class logger if possible
             class_logger = logging.getLogger(f"{__name__}.{cls.__name__}")
             class_logger.error(f"File not found during load: {filename}")
             raise # Re-raise the specific error
        except pickle.UnpicklingError as e:
            class_logger = logging.getLogger(f"{__name__}.{cls.__name__}")
            class_logger.error(f"Error unpickling state file {filename}: {e}")
            raise RuntimeError(f"Failed to unpickle state file: {filename}") from e
        except Exception as e:
            class_logger = logging.getLogger(f"{__name__}.{cls.__name__}")
            class_logger.error(f"Unexpected error loading model state from {filename}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model state: {filename}") from e

    # Expose _is_trained via property if desired
    @property
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return getattr(self, '_is_trained', False)

#    @is_trained.setter
#    def is_trained(self, value: bool):
#       """Setter for is_trained to potentially sync with _is_trained"""
#        self._is_trained = bool(value)
        
    def _cleanup_old_state_files(self, models_dir: str, keep_count: int = 5) -> None:
        """
        Clean up old state files, keeping only the most recent ones.
        
        Args:
            models_dir: Directory containing model files
            keep_count: Number of recent files to keep
        """
        try:
            # Find all QStar River state files
            state_files = []
            
            # Check if directory exists before attempting to list files
            if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
                self.logger.warning(f"Models directory not found for cleanup: {models_dir}")
                return
                
            for filename in os.listdir(models_dir):
                if filename.startswith("qstar_river_state_") and filename.endswith(".pkl"):
                    filepath = os.path.join(models_dir, filename)
                    state_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            state_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove older files beyond keep_count
            if len(state_files) > keep_count:
                for filepath, _ in state_files[keep_count:]:
                    os.remove(filepath)
                    self.logger.debug(f"Removed old QStar River state file: {filepath}")
                    
        except Exception as e:
            self.logger.warning(f"Error cleaning up old state files: {e}")
        
    def plot_backtest_results(self, environment: TradingEnvironment, save_path: Optional[str] = None):
        """ Plots backtest results, ensuring index safety. """
        original_exploration_rate = self.exploration_rate
        self.exploration_rate = 0.0
    
        try:
            state = environment.reset()
            done = False
            # Explicitly re-initialize lists for this run
            environment.actions_taken = []
            environment.positions = []
            environment.portfolio_values = [environment.initial_capital] # Start with initial capital
    
            # Reset internal state if necessary
            environment.total_trades = 0
            environment.profitable_trades = 0
            environment.total_pnl = 0.0
    
            step_count = 0
            # Use a large number or environment specific length if available
            max_sim_steps = len(environment.prices) - environment.window_size if environment.prices is not None else 10000
    
            while not done and step_count < max_sim_steps:
                action = self.choose_action(state)
    
                try:
                    next_state, reward, done = environment.step(action)
                    state = next_state
                except Exception as e:
                    self.logger.error(f"Error during backtest step {step_count}: {e}", exc_info=True)
                    done = True # Stop simulation on error
    
                step_count += 1
    
            # --- Plotting ---
            metrics = environment.get_performance_metrics()
            
            # Get data for plotting
            portfolio_values = environment.portfolio_values
            positions = environment.positions
            actions = environment.actions_taken
            
            # Safety check - if no data to plot
            if len(portfolio_values) == 0:
                logger.warning("No portfolio values recorded for plotting.")
                return
                
            # CRITICAL FIX: Handle length mismatches by finding the minimum length that's safe to plot
            min_length = min(len(portfolio_values), len(positions) if positions else 0)
            
            # If positions is empty or too short, we can't plot it
            can_plot_positions = len(positions) > 0
            
            # Create time steps of the appropriate length
            time_steps = list(range(len(portfolio_values)))
            position_time_steps = list(range(len(positions))) if can_plot_positions else []
            
            # Create the plot with appropriate number of subplots
            num_subplots = 3 if can_plot_positions else 2
            fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 15), sharex=True)
            
            # Plot portfolio value
            axes[0].plot(time_steps, portfolio_values, label='Portfolio Value')
            axes[0].set_title(f'Portfolio Value (Return: {metrics.get("total_return", 0):.2%})')
            axes[0].set_ylabel('Value ($)')
            axes[0].grid(True)
            axes[0].legend()
    
            # Add buy/sell markers
            for i in range(min(len(actions), len(portfolio_values))):
                action = actions[i]
                pv_value = portfolio_values[i] 
                if action == TradingAction.BUY:
                    axes[0].scatter(i, pv_value, color='green', marker='^', alpha=0.7, label='Buy' if i == 0 else "")
                elif action == TradingAction.SELL:
                    axes[0].scatter(i, pv_value, color='red', marker='v', alpha=0.7, label='Sell' if i == 0 else "")
                elif action == TradingAction.INCREASE:
                    axes[0].scatter(i, pv_value, color='lime', marker='^', alpha=0.5, label='Increase' if i == 0 else "")
                elif action == TradingAction.REDUCE:
                    axes[0].scatter(i, pv_value, color='orange', marker='v', alpha=0.5, label='Reduce' if i == 0 else "")
                elif action == TradingAction.EXIT:
                    axes[0].scatter(i, pv_value, color='black', marker='x', alpha=0.7, label='Exit' if i == 0 else "")
                elif action == TradingAction.HEDGE:
                    axes[0].scatter(i, pv_value, color='purple', marker='s', alpha=0.5, label='Hedge' if i == 0 else "")
    
            # Plot positions ONLY if we have position data
            if can_plot_positions:
                axes[1].plot(position_time_steps, positions, label='Position Size')
                axes[1].set_title('Position Size')
                axes[1].set_ylabel('Units')
                axes[1].grid(True)
                axes[1].legend()
                
                # Adjust action plot index if positions are plotted
                action_plot_index = 2
            else:
                # Skip position plot, adjust action plot index
                action_plot_index = 1
    
            # Plot actions heatmap/areas
            valid_actions = actions[:len(portfolio_values)]  # Ensure we don't go out of bounds
            num_actions = TradingAction.get_num_actions()
            cmap = plt.get_cmap('viridis', num_actions)
    
            for i in range(len(valid_actions)):
                action = valid_actions[i]
                color = cmap(action / (num_actions -1) if num_actions > 1 else 0.5)
                axes[action_plot_index].axvspan(i, i+1, facecolor=color, alpha=0.3)
    
            # Add action legend
            all_action_names = [TradingAction.get_action_name(i) for i in range(num_actions)]
            all_action_colors = [cmap(i / (num_actions-1) if num_actions > 1 else 0.5) for i in range(num_actions)]
            action_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3) for color in all_action_colors]
            axes[action_plot_index].legend(action_patches, all_action_names, loc='upper right', fontsize='small')
    
            axes[action_plot_index].set_title('Trading Actions')
            axes[action_plot_index].set_xlabel('Time Steps')
            axes[action_plot_index].set_ylabel('Action Type')
            axes[action_plot_index].set_yticks([])
            
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

    def stop(self):
        """Called when strategy is stopped."""
        try:
            # Save models state
            if hasattr(self, 'qstar_predictor') and self.qstar_predictor is not None:
                self.qstar_predictor.save_state(force=True)  # Force save regardless of timing
                
            self.logger.info("QStar strategy models saved")
        except Exception as e:
            self.logger.error(f"Error saving models during shutdown: {e}")
        
        # Call parent method
        super().stop()
        
#######################################################
# UNIFIED TRADING ACTIONS
#######################################################

logger = logging.getLogger("TradingAction")
class TradingAction:
    """
    Trading action definitions with six distinct action types.
    Optimized for use with QStar-River hybrid model.
    """
    
    # Core actions
    BUY = 0     # Full buy with available capital
    SELL = 1    # Full sell of current position
    HOLD = 2    # No action
    
    # Advanced actions
    REDUCE = 3  # Reduce position by percentage
    INCREASE = 4  # Increase position by percentage
    HEDGE = 5   # Create hedge position
    EXIT = 6    # Exit the market completely
    
    @staticmethod
    def get_num_actions() -> int:
        """Get number of possible actions."""
        return 7
    
    @staticmethod
    def get_action_name(action: int) -> str:
        """Get human-readable action name."""
        actions = {
            TradingAction.BUY: "BUY",
            TradingAction.SELL: "SELL",
            TradingAction.HOLD: "HOLD",
            TradingAction.REDUCE: "REDUCE",
            TradingAction.INCREASE: "INCREASE",
            TradingAction.HEDGE: "HEDGE",
            TradingAction.EXIT: "EXIT"
        }
        return actions.get(action, "UNKNOWN")
    
    @staticmethod
    def action_to_signal(action: int) -> float:
        """Convert action to normalized signal value for technical indicators."""
        signals = {
            TradingAction.BUY: 1.0,
            TradingAction.INCREASE: 0.5,
            TradingAction.HOLD: 0.0,
            TradingAction.REDUCE: -0.5,
            TradingAction.SELL: -1.0,
            TradingAction.HEDGE: -0.25,  # Slightly bearish
            TradingAction.EXIT: -0.75    # Mostly bearish
        }
        return signals.get(action, 0.0)

# ---------------------------------
# Agent Refinement
# ---------------------------------

# Inside pulsar.py
def refine_agent(agent: QStarLearningAgent, environment: TradingEnvironment,
                max_refinements: int = 3, evaluation_episodes: int = 10) -> QStarLearningAgent:
    """
    Refine the agent by progressively training further.
    Compares performance based on evaluation mean reward.

    Args:
        agent: Agent to refine
        environment: Environment to train in
        max_refinements: Maximum number of refinement steps
        evaluation_episodes: Number of episodes for evaluation between refinements

    Returns:
        Refined agent (the best one found during refinement)
    """
    logger.info(f"Beginning refinement process with up to {max_refinements} steps")

    best_performance = float('-inf') # Initialize best performance found so far
    best_agent_path = None
    temp_files = [] # Keep track of temp files created

    # Store the initial state of the agent just in case refinement doesn't improve performance
    initial_agent_state = agent # Keep a reference to the initial agent

    for ref_step in range(max_refinements):
        logger.info(f"--- Refinement Step {ref_step + 1}/{max_refinements} ---")

        # Adjust training parameters for refinement phase (optional, based on your strategy)
        # Example: Increase max episodes for this refinement training run
        original_max_episodes = agent.max_episodes
        agent.max_episodes = agent.max_episodes + 500 # Add 500 episodes for refinement training
        agent.learning_rate = max(0.005, agent.learning_rate * 0.95) # Reduce LR further
        agent.exploration_rate = max(agent.min_exploration_rate, agent.exploration_rate * 0.98) # Decrease explore


        # Train the agent further
        logger.info(f"Refinement Training (Current Max Ep: {agent.max_episodes}, LR: {agent.learning_rate:.5f})...")
        epoch_converged, episodes = agent.train(environment) # Train again
        agent.max_episodes = original_max_episodes # Restore original setting for next potential refinement loop

        # Evaluate the agent
        logger.info(f"Refinement Evaluation ({evaluation_episodes} episodes)...")
        eval_results = agent.evaluate(environment, num_episodes=evaluation_episodes)

        # <<< FIX HERE: Use 'mean_reward' as the performance metric >>>
        current_performance = eval_results.get('mean_reward', float('-inf')) # Use .get for safety
        # Ensure performance is float
        current_performance = float(current_performance) if np.isfinite(current_performance) else float('-inf')

        logger.info(f"Refinement Step {ref_step + 1}: Current Mean Reward={current_performance:.4f}, Previous Best={best_performance:.4f}")

        # Check if this is the best performing agent so far
        if current_performance > best_performance:
            best_performance = current_performance
            # Save the current best agent state to a NEW temp file
            import tempfile
            import os
            try:
                fd, temp_path = tempfile.mkstemp(suffix='.pkl', prefix=f'refined_agent_{ref_step+1}_')
                os.close(fd)
                agent.save(temp_path)
                logger.info(f"Saved NEW best agent state to {temp_path}")
                # Add this new file to track
                temp_files.append(temp_path)
                # Update the path for the *current* best agent
                best_agent_path = temp_path
            except Exception as e_save:
                logger.error(f"Error saving temporary refined agent state: {e_save}")
                # Keep the previous best_agent_path if save fails


        # Force garbage collection (optional but can help in long runs)
        gc.collect()
        if agent.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


    # --- Post Refinement ---
    # Load the best agent state found during refinement
    final_agent = initial_agent_state # Default to the agent passed in initially
    if best_agent_path and os.path.exists(best_agent_path):
        logger.info(f"Loading best agent found during refinement (Performance: {best_performance:.4f}) from: {best_agent_path}")
        try:
            final_agent = QStarLearningAgent.load(best_agent_path) # Load the best one
        except Exception as e_load:
            logger.error(f"Error loading best refined agent state: {e_load}. Returning agent from last refinement step (or initial).")
            final_agent = agent # Fallback to the very last state if load fails
    elif best_performance <= float('-inf'):
         logger.warning("Refinement did not improve performance. Returning the initial agent passed to refine_agent.")
         final_agent = initial_agent_state # Return the original agent if no improvement
    else:
         logger.warning("No best agent path recorded during refinement, returning agent from last refinement step.")
         final_agent = agent # Return the agent from the last iteration


    # Clean up ALL temporary agent files created during refinement
    for temp_path in temp_files:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary refinement file: {temp_path}")
        except OSError as e:
            logger.warning(f"Could not delete temporary refinement file {temp_path}: {e}")

    return final_agent # Return the best loaded agent (or fallback)

# ---------------------------------
# Training Functions
# ---------------------------------
def train_with_config(config_path: Union[str, Path]) -> QStarLearningAgent:
    """
    Train QStar agent using configuration settings.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Trained QStar agent
    """
    # Load configuration
    config = load_training_config(config_path)
    
    # Load data according to configuration
    pair_data = load_data_from_config(config)
    
    if not pair_data:
        logger.error("No valid data loaded")
        return None
    
    # Get training parameters
    training_config = config['training_config']
    window_size = training_config.get('window_size', 50)
    initial_capital = training_config.get('initial_capital', 10000.0)
    transaction_fee = training_config.get('transaction_fee', 0.001)
    epochs = training_config.get('epochs', 200)
    batch_size = training_config.get('batch_size', 64)
    learning_rate = training_config.get('learning_rate', 0.05)
    discount_factor = training_config.get('discount_factor', 0.95)
    use_quantum = training_config.get('use_quantum_representation', True)
    use_gpu = training_config.get('use_gpu', True)
    save_path = training_config.get('save_path')
    
    # Create multi-pair environment
    env = MultiPairTradingEnvironment(
        pair_data=pair_data,
        window_size=window_size,
        initial_capital=initial_capital,
        transaction_fee=transaction_fee,
        use_gpu=use_gpu
    )
    
    # Create agent
    agent = QStarLearningAgent(
        states=env.num_states,
        actions=env.num_actions,
        learning_rate=0.03,  # Lower learning rate for stability
        discount_factor=0.97,  # Higher discount for long-term rewards
        exploration_rate=1.0,
        min_exploration_rate=0.05,
        exploration_decay_rate=0.998,  # Slower decay
        use_adaptive_learning_rate=True,
        use_experience_replay=True,
        batch_size=128,  # Larger batch size
        experience_buffer_size=100000,  # Larger buffer
        max_episodes=500,  # More episodes
        use_quantum_representation=True,
        use_gpu=True
    )
        
    # Train the agent
    logger.info(f"Starting training for {epochs} epochs with {len(pair_data)} pairs")
    epoch_converged, episodes = agent.train(env)
    
    # Save the trained agent if path specified
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        agent.save(save_path)
        logger.info(f"Agent saved to {save_path}")
    
    return agent


def train_agent(pair: str, timeframe: str, exchange: str = 'binance',
               use_gpu: bool = None, epochs: int = 100, batch_size: int = 64,
               initial_capital: float = 10000.0, transaction_fee: float = 0.001,
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
        initial_capital: Initial balance for trading environment
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
        initial_capital=initial_capital,
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
    epoch_converged, episodes_completed = agent.train(env)
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
                  exchange: str = 'binance', initial_capital: float = 10000.0,
                  transaction_fee: float = 0.001, episodes: int = 10) -> Dict[str, Any]:
    """
    Evaluate a trained agent on historical data.
    
    Args:
        agent: Trained agent
        pair: Trading pair
        timeframe: Timeframe
        exchange: Exchange name
        initial_capital: Initial balance
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
        initial_capital=initial_capital,
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
###### TRAINING  ##################
# ---------------------------------

def main():
    """Main function for running the QStar Trading system."""
    print("QStar Trading System with Freqtrade Integration")
    print("=" * 50)
    
    # Check for command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train QStar agent with optional configuration file")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # If config file provided, use it for training
    if args.config:
        print(f"\nUsing configuration from: {args.config}")
        agent = run_training_from_config_file(args.config)
        if agent:
            print("\nTraining completed!")
        return
    
    # Otherwise fall back to the original user-interactive flow
    
    # Check for available hardware
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU available: {gpu_name}")
    elif JAX_AVAILABLE:
        try:
            # Try to get GPU devices with error handling
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) > 0:
                print(f"GPU available: {gpu_devices[0]}")
            else:
                print("No GPU detected with JAX")
        except RuntimeError:
            # JAX couldn't find 'gpu' backend
            print("JAX GPU backends not available, using CPU")
            # Show all available devices
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
    


