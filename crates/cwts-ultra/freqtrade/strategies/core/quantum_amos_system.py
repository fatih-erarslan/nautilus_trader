"""
Quantum AMOS System Integration
-------------------------------

This module integrates the Quantum AMOS agent with other quantum systems:
- Quantum Annealing Regression for forecasting
- Immune-inspired Quantum Anomaly Detection
- Quantum-Enhanced Reservoir Computing
- Neuromorphic Quantum Optimizer
- QStar-River reinforcement learning system

Author: Cascade AI
Version: 1.0.0
"""

import os
import sys
import logging
import json
import time
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import time
import signal
from contextlib import contextmanager
from pathlib import Path
from hardware_manager import HardwareManager
from cdfa_extensions.hw_acceleration import HardwareAccelerator, AcceleratorType, MemoryMode, quantum_accelerated

# Add the correct paths to sys.path
sys.path.append('/home/ashina/freqtrade')
sys.path.append('/home/ashina/freqtrade/user_data/strategies/core')

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)



# Import quantum components
try:
    from quantum_amos import QuantumAmosAgent
    from qerc import QuantumEnhancedReservoirComputing
    from quantum_annealing_regression import QuantumAnnealingRegression
    from iqad import ImmuneQuantumAnomalyDetector
    from nqo import NeuromorphicQuantumOptimizer
    from pulsar import QStarRiverPredictor, QStarLearningAgent, RiverOnlineMLAdapter
    # Import resource scheduler
    from resource_scheduler import QuantumResourceScheduler, ResourceType, ResourcePriority
    # Import hardware acceleration
    # Import RiverOnlineML directly from river_ml module
    try:
        from river_ml import RiverOnlineML
    except ImportError as e:
        logging.warning(f"Could not import RiverOnlineML: {e}")
        # Create a mock class if unavailable
        class RiverOnlineML:
            RIVER_AVAILABLE = False
            def __init__(self, *args, **kwargs):
                self.is_initialized = False
    from hardware_manager import HardwareManager
except ImportError as e:
    logging.error(f"Error importing quantum components: {e}")
    logging.warning("Some system components may not be available")



# Constants for system configuration
DEFAULT_PENNYLANE_DEVICE = "lightning.kokkos"
DEFAULT_SHOTS = 1000
DEFAULT_BATCH_SIZE = 32
DEFAULT_BUFFER_SIZE = 10000
DEFAULT_LEARNING_RATE = 0.01

@dataclass
class QuantumSystemConfig:
    """Configuration for the Quantum AMOS System"""
    
    # Hardware configuration
    use_gpu: bool = True
    use_quantum_hardware: bool = False
    pennylane_device: str = DEFAULT_PENNYLANE_DEVICE
    shots: int = DEFAULT_SHOTS
    mixed_precision: bool = False
    
    # Integration configuration
    enable_quantum_annealing: bool = True
    enable_iqad: bool = True
    enable_qerc: bool = True
    enable_nqo: bool = True
    enable_rl: bool = True
    
    # Reinforcement learning configuration
    rl_batch_size: int = DEFAULT_BATCH_SIZE
    rl_buffer_size: int = DEFAULT_BUFFER_SIZE
    rl_learning_rate: float = DEFAULT_LEARNING_RATE
    rl_discount_factor: float = 0.95
    rl_exploration_rate: float = 1.0
    rl_min_exploration_rate: float = 0.05
    rl_exploration_decay: float = 0.995
    
    # System parameters
    log_level: str = "INFO"
    save_model_interval: int = 100
    model_dir: str = "models"
    
    def __post_init__(self):
        """Validate configuration"""
        # Convert log level string to actual logging level
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            logger.warning(f"Invalid log level: {self.log_level}, using INFO")
            self.log_level = "INFO"
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)



try:
    # Assumes panarchy_analyzer.py is importable from qar.py's location
    from analyzers.panarchy_analyzer import MarketPhase
    logging.debug(f"QAR.PY: Successfully imported MarketPhase: {MarketPhase}")
except ImportError:
    logging.error("CRITICAL: Failed to import MarketPhase from panarchy_analyzer. Using fallback enum (No UNKNOWN phase).")
    class MarketPhase(Enum):
         GROWTH="growth"
         CONSERVATION="conservation"
         RELEASE="release"
         REORGANIZATION="reorganization"

         @classmethod
         def from_string(cls, phase_str: str):
             phase_str = str(phase_str).lower()
             for phase in cls:
                 if phase.value == phase_str: return phase
             logging.warning(f"Invalid phase string '{phase_str}' received in fallback enum. Defaulting to CONSERVATION.")
             return cls.CONSERVATION

class MarketRegime(Enum):
    """Market regime types for adaptive decision making"""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING_LOW_VOL = auto()
    RANGING_HIGH_VOL = auto()
    BREAKOUT = auto()
    REVERSAL = auto()
    UNKNOWN = auto()

class QuantumAmosSystem:
    """
    Quantum AMOS System that integrates all quantum components
    and reinforcement learning capabilities.
    
    This system connects the following components:
    1. Quantum AMOS Agent - Core decision making
    2. Quantum Annealing Regression - Time series forecasting
    3. IQAD - Anomaly detection
    4. QERC - Feature extraction and enhancement
    5. NQO - Parameter optimization
    6. QStar-River - Reinforcement learning
    """
    
    def __init__(self, config: QuantumSystemConfig = None):
        """Initialize the Quantum AMOS System with strategic on-demand component loading"""
        # Use default config if none provided
        self.config = config or QuantumSystemConfig()
        
        # Configure logging
        self._configure_logging()
        
        # Track component states
        self._initialized_components = set()  # Components that have been initialized
        self._components_status = {}          # Status of each component
        self._active_components = set()       # Currently active components
        self._registered_components = set()   # Components available to be loaded
        self._component_last_used = {}        # When each component was last used
        self._component_usage_count = {}      # How many times each component has been used
        
        # On-demand loading parameters
        self._max_active_components = 3       # Maximum number of active quantum components
        self._next_forecast_time = None       # Next time to run forecasting (hourly)
        
        # Track system and market state
        self._is_initialized = False
        self._is_trained = False
        self.current_regime = MarketRegime.UNKNOWN
        self.current_volatility = 0.0
        self.current_volume = 0.0
        self.entropy = 0.0
        self._last_optimization_time = time.time() - 86400  # Start ready for optimization
        self._previous_regime = None
        self._avg_reward = 0.0
        
        # System state variables - create metrics before component initialization
        self.metrics = {
            "decisions_count": 0,
            "anomalies_detected": 0,
            "regime_changes": 0,
            "rl_episodes": 0,
            "optimization_runs": 0,
            "component_activations": 0,     # Track component activation count
            "component_unloads": 0          # Track component unloading count
        }
        
        # Tracking history for analysis
        self.decision_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=100)
        self.forecast_history = deque(maxlen=100)
        self.feature_history = deque(maxlen=100)
        self.regime_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Initialize hardware manager
        self._initialize_hardware_manager()
        
        # Initialize quantum resource scheduler
        self._initialize_resource_scheduler()
        
        # Register components for on-demand loading (doesn't load them yet)
        self._register_components()
        
        # Only initialize core AMOS agent upfront
        self._initialize_quantum_amos_agent()
        
        logger.info(f"Quantum AMOS System initialized with core agent and {len(self._registered_components)} registered components for on-demand loading")
    
    def _configure_logging(self):
        """Configure logging for the system"""
        numeric_level = getattr(logging, self.config.log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        
        # Log configuration
        logger.info(f"Quantum AMOS System logging configured at {self.config.log_level} level")
    
    def _initialize_hardware_manager(self):
        """Initialize hardware manager for quantum operations"""
        try:
            self.hardware_manager = HardwareManager()
            
            # Initialize hardware accelerator for GPU optimizations
            self.hw_accel = HardwareAccelerator(
                enable_gpu=True,          # Enable GPU acceleration
                prefer_cuda=False,        # Don't prefer CUDA over ROCm for AMD GPUs
                device=None,              # Auto-select the best device
                log_level=logging.INFO,   # Match our logging level
                memory_mode=MemoryMode.DYNAMIC,  # Dynamic memory management
                optimization_level="performance"  # Prioritize performance
            )
            # Use a try block to get device info as method may have different name
            try:
                if hasattr(self.hw_accel, 'get_active_device'):
                    device_info = self.hw_accel.get_active_device()
                elif hasattr(self.hw_accel, 'get_device'):
                    device_info = self.hw_accel.get_device()
                else:
                    device_info = "unknown device"
                logger.info(f"Hardware accelerator initialized with device: {device_info}")
            except Exception as e:
                logger.warning(f"Could not get device info: {e}")
            
            self._initialized_components.add("hardware_manager")
            self._components_status["hardware_manager"] = "initialized"
            logger.info("Hardware manager initialized")
            
            # Set environment variables for optimal performance
            os.environ["OMP_PROC_BIND"] = "true"
            os.environ["KMP_BLOCKTIME"] = "0"
            
            # For AMD GPUs (if detected at runtime)
            if hasattr(self.hardware_manager, "get_manager") and self.hardware_manager.get_manager():
                os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
                
            # Apply recommended settings from hardware accelerator
            if self.hw_accel.get_accelerator_type():
                accel_type = self.hw_accel.get_accelerator_type()
                if accel_type == AcceleratorType.ROCM:
                    logger.info("Configuring for AMD GPU acceleration")
                    # AMD-specific optimizations
                    os.environ["HIP_VISIBLE_DEVICES"] = "0"
                    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
                elif accel_type == AcceleratorType.CUDA:
                    logger.info("Configuring for NVIDIA GPU acceleration")
                    # NVIDIA-specific optimizations
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    
            # Register HW accelerator as an initialized component
            self._initialized_components.add("hw_accelerator")
            self._components_status["hw_accelerator"] = "initialized"
        except Exception as e:
            logger.error(f"Error initializing hardware manager: {e}")
            self._components_status["hardware_manager"] = f"error: {str(e)}"
            
            # Try to initialize hardware accelerator even if hardware_manager fails
            try:
                self.hw_accel = HardwareAccelerator(
                    enable_gpu=True,
                    prefer_cuda=False,
                    log_level=logging.INFO,
                    memory_mode=MemoryMode.DYNAMIC
                )
                # Use a try block to get device info as method may have different name
                try:
                    if hasattr(self.hw_accel, 'get_active_device'):
                        device_info = self.hw_accel.get_active_device()
                    elif hasattr(self.hw_accel, 'get_device'):
                        device_info = self.hw_accel.get_device()
                    else:
                        device_info = "unknown device"
                    logger.info(f"Hardware accelerator initialized separately: {device_info}")
                except Exception as e:
                    logger.warning(f"Could not get device info: {e}")
                self._initialized_components.add("hw_accelerator")
                self._components_status["hw_accelerator"] = "initialized"
            except Exception as hw_err:
                logger.error(f"Error initializing hardware accelerator: {hw_err}")
                self._components_status["hw_accelerator"] = f"error: {str(hw_err)}"
            
            # Create a minimal fallback hardware manager
            class FallbackHardwareManager:
                def __init__(self):
                    pass
                    
                def get_device(self):
                    return "cpu"
                    
                def optimize_for_device(self, model):
                    return model
                    
            self.hardware_manager = FallbackHardwareManager()
            logger.warning("Using fallback hardware manager with CPU only")
            
    def _initialize_resource_scheduler(self):
        """Initialize the quantum resource scheduler"""
        try:
            # Create the resource scheduler with appropriate configuration
            self.resource_scheduler = QuantumResourceScheduler(
                max_concurrent_quantum_tasks=1,  # Limit to 1 quantum task at a time
                max_concurrent_gpu_tasks=2,      # Allow 2 GPU tasks simultaneously
                max_cpu_threads=None,           # Use auto-detection
                enable_profiling=True,
                hardware_manager=self.hardware_manager
            )
            
            # Start the scheduler
            self.resource_scheduler.start()
            
            # Register the system itself
            self.resource_scheduler.register_component("quantum_amos_system")
            
            self._initialized_components.add("resource_scheduler")
            self._components_status["resource_scheduler"] = "initialized"
            logger.info("Quantum resource scheduler initialized and started")
            
        except Exception as e:
            logger.error(f"Error initializing resource scheduler: {e}")
            self._components_status["resource_scheduler"] = f"error: {str(e)}"
            # Create a no-op context manager as a fallback
            class NoOpResourceScheduler:
                def register_component(self, *args, **kwargs): pass
                def acquire(self, *args, **kwargs):
                    class NoOpContextManager:
                        def __enter__(self): return None
                        def __exit__(self, *args): pass
                    return NoOpContextManager()
                def request_resources(self, *args, **kwargs): return "dummy-id"
                def release_resources(self, *args, **kwargs): pass
                def stop(self): pass
            
            self.resource_scheduler = NoOpResourceScheduler()
            logger.warning("Using no-op resource scheduler fallback")

    @contextmanager
    def _timeout(self, seconds, operation_name="quantum operation"):
        """Context manager for handling timeout for quantum operations
        
        Args:
            seconds: Timeout in seconds
            operation_name: Name of the operation for logging purposes
        """
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Quantum {operation_name} timed out after {seconds} seconds")
            
        # Set the timeout handler
        original_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            # Set the alarm
            signal.alarm(seconds)
            yield
        finally:
            # Cancel the alarm and restore the original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
            
    def _register_components(self):
        """Register components for on-demand loading but don't initialize them yet"""
        # Register quantum annealing regression (hourly forecasting)
        if self.config.enable_quantum_annealing:
            self._registered_components.add("quantum_annealing_regression")
            self._components_status["quantum_annealing_regression"] = "registered"
        
        # Register IQAD (for market anomalies and whale activity detection)
        if self.config.enable_iqad:
            self._registered_components.add("iqad")
            self._components_status["iqad"] = "registered"
        
        # Register QERC (lightweight continuous operation)
        if self.config.enable_qerc:
            self._registered_components.add("qerc")
            self._components_status["qerc"] = "registered"
            # Initialize QERC immediately since it's lightweight (4 qubits)
            # and provides valuable real-time quantum indicators
            self._initialize_qerc()
        
        # Register NQO (for parameter optimization)
        if self.config.enable_nqo:
            self._registered_components.add("nqo")
            self._components_status["nqo"] = "registered"
        
        # Register Q*-River for reinforcement learning
        if self.config.enable_rl:
            self._registered_components.add("qstar_river")
            self._components_status["qstar_river"] = "registered"
        
        # Set the next forecast time 1 hour from now
        self._next_forecast_time = time.time() + 3600
        
        logger.info(f"Registered {len(self._registered_components)} components for on-demand loading")
    
    def _manage_active_components(self, component_to_load):
        """Manage active components to ensure we don't exceed the maximum (3)
        
        Args:
            component_to_load: Name of the component we want to activate
            
        Returns:
            True if the component can be loaded, False otherwise
        """
        # If component is already active, just update its usage stats
        if component_to_load in self._active_components:
            self._component_last_used[component_to_load] = time.time()
            self._component_usage_count[component_to_load] = self._component_usage_count.get(component_to_load, 0) + 1
            return True
        
        # If we have room for another component, allow loading
        if len(self._active_components) < self._max_active_components:
            return True
        
        # We need to unload a component to make room
        # Find the least recently used component that can be unloaded
        lru_component = None
        oldest_time = float('inf')
        
        for component in self._active_components:
            # Never unload the core quantum AMOS agent
            if component == "quantum_amos_agent":
                continue
            
            # Skip QERC if requested as it's lightweight and provides valuable continuous indicators
            if component == "qerc" and self.config.enable_qerc:
                continue
            
            # Find the least recently used component
            last_used = self._component_last_used.get(component, 0)
            if last_used < oldest_time:
                oldest_time = last_used
                lru_component = component
        
        # If we found a component to unload, do so
        if lru_component:
            logger.info(f"Unloading component {lru_component} to make room for {component_to_load}")
            self._unload_component(lru_component)
            return True
        
        # We couldn't find a component to unload
        logger.warning(f"Cannot load {component_to_load}: maximum active components reached and no components can be unloaded")
        return False
    
    def _unload_component(self, component_name):
        """Unload a component to free up resources
        
        Args:
            component_name: Name of the component to unload
        """
        try:
            # Remove from active components set
            if component_name in self._active_components:
                self._active_components.remove(component_name)
            
            # Call cleanup method if it exists, then delete the reference
            if component_name == "quantum_annealing_regression" and hasattr(self, "qar"):
                if hasattr(self.qar, "cleanup") and callable(self.qar.cleanup):
                    self.qar.cleanup()
                delattr(self, "qar")
            
            elif component_name == "iqad" and hasattr(self, "iqad"):
                if hasattr(self.iqad, "cleanup") and callable(self.iqad.cleanup):
                    self.iqad.cleanup()
                delattr(self, "iqad")
            
            elif component_name == "qerc" and hasattr(self, "qerc"):
                if hasattr(self.qerc, "cleanup") and callable(self.qerc.cleanup):
                    self.qerc.cleanup()
                delattr(self, "qerc")
            
            elif component_name == "nqo" and hasattr(self, "nqo"):
                if hasattr(self.nqo, "cleanup") and callable(self.nqo.cleanup):
                    self.nqo.cleanup()
                delattr(self, "nqo")
            
            elif component_name == "qstar_river" and hasattr(self, "qstar_river"):
                if hasattr(self.qstar_river, "cleanup") and callable(self.qstar_river.cleanup):
                    self.qstar_river.cleanup()
                delattr(self, "qstar_river")
            
            # Update component status and metrics
            self._components_status[component_name] = "inactive"
            self.metrics["component_unloads"] += 1
            
            logger.info(f"Component {component_name} unloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error unloading component {component_name}: {e}")
            return False
    
    def _should_load_component(self, component_name, market_data=None):
        """Determine if a component should be loaded based on strategic criteria
        
        Args:
            component_name: Name of the component to evaluate
            market_data: Optional market data for condition evaluation
            
        Returns:
            True if component should be loaded, False otherwise
        """
        # Quantum Annealing Regression - load hourly
        if component_name == "quantum_annealing_regression":
            # If next_forecast_time is None or we've passed it, load for forecasting
            if self._next_forecast_time is None or time.time() >= self._next_forecast_time:
                # Schedule next forecast 1 hour from now
                self._next_forecast_time = time.time() + 3600
                return True
            return False
        
        # IQAD - load only during high entropy/volatility/volume
        elif component_name == "iqad":
            # Check market conditions
            if market_data:
                # Get volatility from market data or use stored value
                volatility = market_data.get("volatility", self.current_volatility)
                
                # Check volume conditions
                volume = market_data.get("volume", 0.0)
                avg_volume = market_data.get("avg_volume", 0.0)
                volume_spike = False
                if avg_volume > 0:
                    volume_spike = volume > (avg_volume * 1.5)  # 50% above average
                
                # Check entropy/disorder in the market
                entropy = market_data.get("entropy", self.entropy)
                
                # Load IQAD if any conditions indicate potential anomalies
                return volatility > 0.3 or volume_spike or entropy > 0.7
            return False
        
        # QERC - lightweight, can run continuously
        elif component_name == "qerc":
            return True  # Always available as it's lightweight
        
        # NQO - load only when hyperparameter tuning is needed
        elif component_name == "nqo":
            # Check if regime has changed (requiring parameter optimization)
            regime_changed = False
            if market_data and "regime" in market_data:
                current_regime = market_data["regime"]
                if hasattr(self, "_previous_regime") and self._previous_regime is not None:
                    regime_changed = current_regime != self._previous_regime
                self._previous_regime = current_regime
            
            # Also check if it's time for periodic optimization (every 24 hours)
            time_for_periodic = (time.time() - self._last_optimization_time) > 86400  # 24 hours
            
            # Or if volatility has changed significantly
            volatility_change = False
            if market_data and "volatility" in market_data:
                new_volatility = market_data["volatility"]
                volatility_change = abs(new_volatility - self.current_volatility) > 0.2
                self.current_volatility = new_volatility
            
            return regime_changed or time_for_periodic or volatility_change
        
        # QStar River - load when learning thresholds are met
        elif component_name == "qstar_river":
            # Need sufficient data for learning
            enough_data = len(self.decision_history) >= 20
            
            # Poor performance indicates need for learning
            poor_performance = self._avg_reward < 0.4 if hasattr(self, "_avg_reward") else False
            
            # Has sufficient decisions been made since last learning?
            decisions_since_training = self.metrics["decisions_count"] % 50 == 0
            
            return (enough_data and (poor_performance or decisions_since_training)) or not self._is_trained
        
        # Default case
        return False
            
    def _initialize_quantum_amos_agent(self):
        """Initialize the core Quantum AMOS agent"""
        try:
            # Create the agent with required parameters
            self.agent = QuantumAmosAgent(
                name="QuantumAmosSystem",  # Provide a name for the agent
                weights={
                    "forecast": 0.4,
                    "trend": 0.3,
                    "volatility": 0.2,
                    "volume": 0.1
                },  # Default weights for decision factors
                desire=0.5  # Default desire value as a float, not a string
            )
            
            if hasattr(self.agent, "configure") and callable(self.agent.configure):
                self.agent.configure(use_gpu=self.config.use_gpu)
                
            # Add to active and initialized components
            self._initialized_components.add("quantum_amos_agent")
            self._active_components.add("quantum_amos_agent")
            self._components_status["quantum_amos_agent"] = "active"
            self._component_last_used["quantum_amos_agent"] = time.time()
            self._component_usage_count["quantum_amos_agent"] = 1
            
            # System is initialized if the core agent is available
            self._is_initialized = True
            
            logger.info("Quantum AMOS Agent initialized and activated")
        except Exception as e:
            logger.error(f"Error initializing Quantum AMOS Agent: {e}")
            self._components_status["quantum_amos_agent"] = f"error: {str(e)}"
            self._is_initialized = False
    
    def _initialize_quantum_annealing_regression(self):
        """Initialize the Quantum Annealing Regression component on-demand"""
        # First check if already active
        if "quantum_annealing_regression" in self._active_components:
            # Just update usage timestamp
            self._component_last_used["quantum_annealing_regression"] = time.time()
            return True
        
        # Check if we have room or need to unload something
        if not self._manage_active_components("quantum_annealing_regression"):
            logger.warning("Cannot initialize Quantum Annealing Regression: component limit reached")
            return False
            
        try:
            # Create QAR with default parameters - we'll adapt to its interface
            self.qar = QuantumAnnealingRegression()
            
            # If there's a configure method, try to use it for hardware optimization
            if hasattr(self.qar, "configure") and callable(self.qar.configure):
                self.qar.configure(use_gpu=self.config.use_gpu)
            
            # Add to active and initialized components
            self._initialized_components.add("quantum_annealing_regression")
            self._active_components.add("quantum_annealing_regression")
            self._components_status["quantum_annealing_regression"] = "active"
            self._component_last_used["quantum_annealing_regression"] = time.time()
            self._component_usage_count["quantum_annealing_regression"] = self._component_usage_count.get("quantum_annealing_regression", 0) + 1
            
            # Track metrics
            self.metrics["component_activations"] += 1
            
            logger.info("Quantum Annealing Regression initialized and activated")
            return True
        except Exception as e:
            logger.error(f"Error initializing Quantum Annealing Regression: {e}")
            self._components_status["quantum_annealing_regression"] = f"error: {str(e)}"
            return False
    
    def _initialize_iqad(self):
        """Initialize the Immune Quantum Anomaly Detector on-demand"""
        # First check if already active
        if "iqad" in self._active_components:
            # Just update usage timestamp
            self._component_last_used["iqad"] = time.time()
            return True
        
        # Check if we have room or need to unload something
        if not self._manage_active_components("iqad"):
            logger.warning("Cannot initialize IQAD: component limit reached")
            return False
        
        try:
            # Create with default parameters
            self.iqad = ImmuneQuantumAnomalyDetector()
            
            # Try to configure if method exists
            if hasattr(self.iqad, "configure_hardware") and callable(self.iqad.configure_hardware):
                self.iqad.configure_hardware(device="GPU" if self.config.use_gpu else "CPU")
            
            # Add to active and initialized components
            self._initialized_components.add("iqad")
            self._active_components.add("iqad")
            self._components_status["iqad"] = "active"
            self._component_last_used["iqad"] = time.time()
            self._component_usage_count["iqad"] = self._component_usage_count.get("iqad", 0) + 1
            
            # Track metrics
            self.metrics["component_activations"] += 1
            
            logger.info("Immune Quantum Anomaly Detector initialized and activated")
            return True
        except Exception as e:
            logger.error(f"Error initializing Immune Quantum Anomaly Detector: {e}")
            self._components_status["iqad"] = f"error: {str(e)}"
            return False
    
    def _initialize_qerc(self):
        """Initialize Quantum Enhanced Reservoir Computing component
        
        We need this to be lightweight so it can run continuously.
        This component is optimized for feature extraction and regime detection.
        """
        # Check if QERC components are available
        #if not QUANTUM_COMPONENTS_AVAILABLE:
        #    logger.warning("QERC components not available - some quantum features will be disabled")
        #    return False
        
        # First check if already active
        if "qerc" in self._active_components:
            # Just update usage timestamp
            self._component_last_used["qerc"] = time.time()
            return True
        
        # QERC is lightweight (4 qubits) and can run continuously, so we prioritize it
        # Check if we have room or need to unload something
        if not self._manage_active_components("qerc"):
            logger.warning("Cannot initialize QERC: component limit reached")
            return False
            
        try:
            # Create with optimized parameters for quantum indicators
            self.qerc = QuantumEnhancedReservoirComputing(
                reservoir_size=500,  # Larger reservoir for better pattern recognition
                quantum_kernel_size=8,  # Updated from 4 to 8 qubits for quantum indicators
                spectral_radius=0.95,
                leaking_rate=0.3,
                temporal_windows=[5, 15, 30, 60],  # Multiple timeframes for better market analysis
                input_dimensionality=16  # Match with quantum kernel calculations
            )
            
            # Configure hardware acceleration using new quantum_accelerated decorators
            if hasattr(self.qerc, "set_hardware_acceleration") and callable(self.qerc.set_hardware_acceleration):
                self.qerc.set_hardware_acceleration(
                    enabled=self.config.use_gpu,
                    use_quantum_hardware=self.config.use_quantum_hardware,
                    device_type=self.config.pennylane_device,
                    shots=self.config.shots
                )
            
            # Verify quantum indicators are properly loaded
            self._verify_quantum_indicators()
            
            # Add to active and initialized components
            self._initialized_components.add("qerc")
            self._active_components.add("qerc")
            self._components_status["qerc"] = "active"
            self._component_last_used["qerc"] = time.time()
            self._component_usage_count["qerc"] = self._component_usage_count.get("qerc", 0) + 1
            
            # Track metrics
            self.metrics["component_activations"] += 1
            
            logger.info("Quantum Enhanced Reservoir Computing initialized with optimized quantum indicators")
            return True
        except Exception as e:
            logger.error(f"Error initializing Quantum Enhanced Reservoir Computing: {e}")
            self._components_status["qerc"] = f"error: {str(e)}"
            return False
    
    def _verify_quantum_indicators(self):
        """Verify that all quantum indicators are available in the QERC instance"""
        if not hasattr(self, "qerc") or self.qerc is None:
            logger.warning("QERC not initialized, cannot verify quantum indicators")
            return False
        
        required_indicators = [
            "quantum_phase_transition_detector",
            "quantum_entropy_analyzer",
            "quantum_momentum_oscillator",
            "quantum_fractal_dimension_estimator",
            "quantum_correlation_network"
        ]
        
        missing_indicators = []
        for indicator in required_indicators:
            if not hasattr(self.qerc, indicator) or not callable(getattr(self.qerc, indicator)):
                missing_indicators.append(indicator)
        
        if missing_indicators:
            logger.warning(f"Some quantum indicators are unavailable: {missing_indicators}")
            return False
        
        logger.info("All quantum indicators verified and available")
        return True
    
    def _initialize_nqo(self):
        """Initialize the Neuromorphic Quantum Optimizer (only for hyperparameter optimization)"""
        # First check if already active
        if "nqo" in self._active_components:
            # Just update usage timestamp
            self._component_last_used["nqo"] = time.time()
            return True
        
        # Check if we have room or need to unload something
        if not self._manage_active_components("nqo"):
            logger.warning("Cannot initialize NQO: component limit reached")
            return False
            
        try:
            # Create with default parameters
            self.nqo = NeuromorphicQuantumOptimizer()
            
            # Configure GPU usage if method exists
            if hasattr(self.nqo, "enable_gpu") and callable(self.nqo.enable_gpu):
                if self.config.use_gpu:
                    self.nqo.enable_gpu()
            
            # Add to active and initialized components
            self._initialized_components.add("nqo")
            self._active_components.add("nqo")
            self._components_status["nqo"] = "active"
            self._component_last_used["nqo"] = time.time()
            self._component_usage_count["nqo"] = self._component_usage_count.get("nqo", 0) + 1
            
            # Update last optimization time
            self._last_optimization_time = time.time()
            
            # Track metrics
            self.metrics["component_activations"] += 1
            
            logger.info("Neuromorphic Quantum Optimizer initialized and activated")
            return True
        except Exception as e:
            logger.error(f"Error initializing Neuromorphic Quantum Optimizer: {e}")
            self._components_status["nqo"] = f"error: {str(e)}"
            return False
    
    def _initialize_rl_system(self):
        """Initialize the QStar-River reinforcement learning system when learning thresholds are met"""
        # First check if already active
        if "qstar_river" in self._active_components:
            # Just update usage timestamp
            self._component_last_used["qstar_river"] = time.time()
            return True
        
        # Check if we have room or need to unload something
        if not self._manage_active_components("qstar_river"):
            logger.warning("Cannot initialize QStar-River: component limit reached")
            return False
            
        try:
            # The error 'RiverOnlineML' has no attribute 'RIVER_AVAILABLE' suggests
            # we need a direct approach that doesn't rely on this attribute
            
            # Create a direct patch for the RiverOnlineML.RIVER_AVAILABLE check issue
            # Set this attribute directly on the class before attempting initialization
            # This is a safe monkey patch since the application will check this attribute
            if not hasattr(RiverOnlineML, 'RIVER_AVAILABLE'):
                # Set it to False first to avoid any attempted initialization
                setattr(RiverOnlineML, 'RIVER_AVAILABLE', False)
                logger.info("Added RIVER_AVAILABLE attribute to RiverOnlineML class")
                
                # Attempt to check if river is actually available
                try:
                    # Try to import River as a quick check
                    try:
                        import river
                        # If we get here, river is available
                        setattr(RiverOnlineML, 'RIVER_AVAILABLE', True)
                        logger.info("Updated RIVER_AVAILABLE=True after importing river")
                    except ImportError:
                        # Keep it False
                        logger.info("Confirmed river module is not available")
                        pass
                except Exception:
                    # Any exception, keep it False
                    pass
            
            # Now initialize the predictor with sensible defaults
            # Use a more direct approach with minimal required params
            self.qstar_river = QStarRiverPredictor(
                cerebellum_snn=None,  # No SNN to start with
                use_quantum_representation=True
            )
                
            # Add to active and initialized components
            self._initialized_components.add("qstar_river")
            self._active_components.add("qstar_river")
            self._components_status["qstar_river"] = "active"
            self._component_last_used["qstar_river"] = time.time()
            self._component_usage_count["qstar_river"] = self._component_usage_count.get("qstar_river", 0) + 1
            
            # Set training status
            self._is_trained = True
            
            # Track metrics
            self.metrics["component_activations"] += 1
            self.metrics["rl_episodes"] += 1
            
            logger.info("QStar-River reinforcement learning system initialized and activated")
            return True
        except Exception as e:
            logger.error(f"Error initializing QStar-River system: {e}")
            self._components_status["qstar_river"] = f"error: {str(e)}"
            return False

    @quantum_accelerated(use_hw_accel=True, hw_batch_size=32)
    def integrate_forecasting(self, market_data: Dict[str, Any]):
        """
        Integrate forecasting capabilities using Quantum Annealing Regression
        GPU-accelerated via the quantum_accelerated decorator
        Components are loaded on-demand based on hourly schedule
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary with forecast results
        """
        # Check if component should be loaded based on scheduled timing
        load_for_forecasting = self._should_load_component("quantum_annealing_regression", market_data)
        
        # If component isn't active but should be loaded, initialize it
        if load_for_forecasting and "quantum_annealing_regression" not in self._active_components:
            logger.info("Loading Quantum Annealing Regression on-demand for scheduled forecasting")
            if not self._initialize_quantum_annealing_regression():
                logger.warning("Failed to load Quantum Annealing Regression component")
                return {"expected_outcome": 0.0, "probability": 0.5, "forecast_available": False, "error": "Component initialization failed"}
        
        # Check if component is active, if not, return default values
        if "quantum_annealing_regression" not in self._active_components:
            logger.info("Skipping forecasting: not scheduled for current time period")
            return {"expected_outcome": 0.0, "probability": 0.5, "forecast_available": False, "scheduled": False}
        
        # Get market data suitable for forecasting
        timeframe = market_data.get("timeframe", "1h")
        steps = market_data.get("forecast_steps", 5)
        
        # Update component usage stats
        self._component_last_used["quantum_annealing_regression"] = time.time()
        self._component_usage_count["quantum_annealing_regression"] = self._component_usage_count.get("quantum_annealing_regression", 0) + 1
        
        # Register this forecast operation with the resource scheduler, with high priority
        try:
            # Acquire quantum resources with appropriate priority
            with self.resource_scheduler.acquire(
                component_id="quantum_annealing_regression",
                resource_type=ResourceType.QUANTUM_DEVICE,
                priority=ResourcePriority.HIGH,  # HIGH priority for forecasting
                timeout=30.0,  # Allow up to 30 seconds for forecasting
                description="QAR forecasting"
            ) as allocation_id:
                # Log the resource acquisition
                if allocation_id:
                    logger.info(f"Acquired quantum resources for forecasting: {allocation_id}")
                
                # Use a timeout context manager to prevent hanging in quantum operations
                with self._timeout(seconds=25, operation_name="forecasting"):
                    # Execute forecasting with allocated resources
                    forecast_results = self.qar.forecast(
                        market_data, 
                        timeframe=timeframe, 
                        steps=steps
                    )
                
                # Store forecast in history
                self.forecast_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "forecast": forecast_results
                })
            
            # Check if we should unload the component after use
            # For forecasting, we typically unload after each use as it's scheduled hourly
            if "quantum_annealing_regression" in self._active_components and load_for_forecasting:
                logger.info("Unloading Quantum Annealing Regression after forecasting completion")
                self._unload_component("quantum_annealing_regression")
            
            return forecast_results
        except Exception as e:
            logger.error(f"Error in forecasting integration: {e}")
            return {"expected_outcome": 0.0, "probability": 0.5, "forecast_available": False, "error": str(e)}
    
    @quantum_accelerated(use_hw_accel=True, hw_batch_size=16)
    def integrate_anomaly_detection(self, market_data: Dict[str, Any]):
        """
        Integrate anomaly detection capabilities using IQAD
        GPU-accelerated via the quantum_accelerated decorator
        Components are loaded on-demand based on market conditions
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Update market state information
        if market_data.get("volatility"):
            self.current_volatility = market_data["volatility"]
        if market_data.get("volume"):
            self.current_volume = market_data["volume"]
        if market_data.get("entropy"):
            self.entropy = market_data["entropy"]
        
        # Check if component should be loaded based on market conditions
        load_for_detection = self._should_load_component("iqad", market_data)
        
        # If component isn't active but should be loaded, initialize it
        if load_for_detection and "iqad" not in self._active_components:
            logger.info("Loading IQAD on-demand due to high volatility or entropy conditions")
            if not self._initialize_iqad():
                logger.warning("Failed to load IQAD component")
                return {"anomalies": [], "anomaly_score": 0.0, "detection_available": False, "error": "Component initialization failed"}
        
        # Check if component is active, if not, return default values
        if "iqad" not in self._active_components:
            logger.info("Skipping anomaly detection: market conditions don't require it")
            return {"anomalies": [], "anomaly_score": 0.0, "detection_available": False, "required": False}

        # Extract features for anomaly detection
        features = market_data.get("features", {})
        expected_behavior = market_data.get("expected_behavior", None)
        
        # Update component usage stats
        self._component_last_used["iqad"] = time.time()
        self._component_usage_count["iqad"] = self._component_usage_count.get("iqad", 0) + 1
        
        try:
            # Register anomaly detection with the resource scheduler
            # Lower priority than forecasting since it's less time-critical
            with self.resource_scheduler.acquire(
                component_id="iqad",
                resource_type=ResourceType.QUANTUM_DEVICE,
                priority=ResourcePriority.MEDIUM,  # MEDIUM priority for anomaly detection
                timeout=20.0,  # Allow reasonable time for completion
                description="IQAD anomaly detection"
            ) as allocation_id:
                # Log the resource acquisition
                if allocation_id:
                    logger.info(f"Acquired quantum resources for anomaly detection: {allocation_id}")
                
                # Use a timeout context manager to prevent hanging in quantum operations
                with self._timeout(seconds=15, operation_name="anomaly detection"):
                    # Use IQAD for anomaly detection with allocated resources
                    anomaly_results = self.iqad.detect_anomalies(features, expected_behavior)
                
                # Update metrics if anomaly detected
                if anomaly_results.get("anomaly_detected", False):
                    self.metrics["anomalies_detected"] += 1
                    logger.info(f"Anomaly detected with score: {anomaly_results.get('anomaly_score', 0.0)}")
            
            # Store anomaly in history
            self.anomaly_history.append({
                "timestamp": datetime.now().isoformat(),
                "anomaly": anomaly_results
            })
            
            # If market conditions have normalized, consider unloading the component
            # Only unload if it was loaded due to temporary market conditions
            if not self._should_load_component("iqad", market_data) and "iqad" in self._active_components:
                logger.info("Unloading IQAD after detection as market conditions have normalized")
                self._unload_component("iqad")
            
            return anomaly_results
        except Exception as e:
            logger.error(f"Error in anomaly detection integration: {e}")
            return {"anomalies": [], "anomaly_score": 0.0, "detection_available": False, "error": str(e)}
    
    @quantum_accelerated(use_hw_accel=True, hw_batch_size=24)
    def _convert_numpy_to_python(self, obj):
        """
        Recursively convert NumPy arrays and data types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert (can be dict, list, numpy array, or scalar)
            
        Returns:
            Object with all NumPy types converted to native Python types
        """
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_to_python(item) for item in obj]
        return obj
        
    def integrate_feature_extraction(self, market_data: Dict[str, Any]):
        """
        Integrate feature extraction capabilities using QERC
        GPU-accelerated via the quantum_accelerated decorator
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Dictionary with enhanced features (all NumPy arrays converted to lists for JSON serialization)
        """
        if "qerc" not in self._initialized_components:
            logger.warning("QERC not initialized, skipping feature extraction")
            return {"enhanced_features": {}, "extraction_available": False}
        
        # Extract data for feature extraction
        features = market_data.get("raw_features", [])
        
        # Validate and debug the features
        if not features or len(features) == 0:
            logger.warning("No raw_features found in market_data. Attempting to generate features from close prices.")
            # Try to use close prices as fallback
            close_prices = market_data.get("close_prices", [])
            # Try other potential price keys if close_prices isn't found
            if not close_prices and "ohlc" in market_data:
                ohlc_data = market_data.get("ohlc", {})
                if "close" in ohlc_data:
                    close_prices = ohlc_data["close"]
                    logger.info(f"Found close prices in OHLC data: {len(close_prices)} points")
            # If we have price data for backtesting scenarios
            if not close_prices and "price_history" in market_data:
                price_history = market_data.get("price_history", [])
                if price_history and len(price_history) > 0:
                    # Extract close prices from price history
                    if isinstance(price_history[0], dict) and "close" in price_history[0]:
                        close_prices = [p["close"] for p in price_history]
                        logger.info(f"Extracted {len(close_prices)} close prices from price history")
            
            # For testing/debugging - generate synthetic prices if nothing else is available
            if not close_prices or len(close_prices) <= 5:
                logger.warning("Creating synthetic price data for testing")
                import numpy as np
                # Generate 100 synthetic prices with some pattern for testing
                t = np.linspace(0, 4*np.pi, 100)
                close_prices = 100 + 10*np.sin(t) + t + np.random.normal(0, 1, 100)
                close_prices = close_prices.tolist()
                
            if close_prices and len(close_prices) > 10:
                # Convert to feature format - use close prices as the feature
                features = np.array(close_prices).reshape(-1, 1)
                logger.info(f"Generated features from {len(close_prices)} close prices: shape={features.shape}")
            else:
                logger.error("No alternative feature sources available")
                return {"enhanced_features": {}, "extraction_available": False, "error": "No valid feature data"}
        else:
            # Convert to numpy array if not already
            if not isinstance(features, np.ndarray):
                features = np.array(features)
                
            # Ensure proper shape - QERC expects a 2D array
            if features.ndim == 1:
                features = features.reshape(-1, 1)
                logger.info(f"Reshaped 1D features to 2D: shape={features.shape}")
                
            logger.info(f"Feature extraction processing with shape={features.shape}, dtype={features.dtype}")
            
        freq_components = market_data.get("frequency_components", None)
        
        try:
            # Resource allocation for feature extraction
            # Lower priority than forecasting and anomaly detection
            with self.resource_scheduler.acquire(
                component_id="qerc",
                resource_type=ResourceType.GPU,  # QERC typically works well on GPU
                priority=ResourcePriority.LOW,  # Lower priority as this can be deferred
                timeout=15.0,  # Allow up to 15 seconds for processing
                description="QERC feature extraction"
            ) as allocation_id:
                # Log the resource acquisition
                if allocation_id:
                    logger.info(f"Acquired resources for feature extraction: {allocation_id}")
                
                # Use QERC for feature extraction with allocated resources
                processed_results = self.qerc.process(features, freq_components)
                
                # Check if we received an error from QERC
                if 'error' in processed_results:
                    logger.error(f"QERC process returned error: {processed_results['error']}")
                    return {"enhanced_features": {}, "extraction_available": False, "error": processed_results['error']}
                
                # Enhance with quantum indicators if available
                if hasattr(self, "qerc") and all(hasattr(self.qerc, indicator) for indicator in 
                                            ["quantum_phase_transition_detector", "quantum_entropy_analyzer"]):
                    # Get price data for quantum indicators
                    price_data = market_data.get("close_prices", [])
                    if len(price_data) > 30:  # Ensure we have enough data points
                        price_data = np.array(price_data)
                        
                        # Add quantum indicator results to processed results
                        quantum_indicators = self._analyze_with_quantum_indicators(price_data, market_data)
                        processed_results.update(quantum_indicators)
                
                # Convert all NumPy arrays to Python lists for JSON serialization
                serializable_results = self._convert_numpy_to_python(processed_results)
                
                # Store features in history
                self.feature_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "features": serializable_results
                })
                
                return serializable_results
        except Exception as e:
            logger.error(f"Error in feature extraction integration: {e}")
            return {"enhanced_features": {}, "extraction_available": False, "error": str(e)}
    

    def strategic_quantum_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive strategic analysis using quantum indicators
        
        This high-level method integrates all quantum indicators to produce market insights,
        trading signals, risk metrics, and position sizing recommendations. It uses the resource
        scheduler to manage quantum computing resources efficiently.
        
        Args:
            market_data: Dictionary containing market data including:
                - close_prices: List of close prices
                - ohlcv_data: OHLCV data if available
                - asset_prices: Dictionary of price data for multiple assets if available
            
        Returns:
            Dictionary containing strategic analysis results including:
                - regime: Current market regime
                - indicators: All quantum indicator results
                - signals: Trading signals derived from quantum analysis
                - insights: Key market insights from quantum indicators
                - correlation_insights: Asset correlation insights if available
        """
        # Ensure QERC is initialized
        if "qerc" not in self._initialized_components:
            self._initialize_qerc()
            if "qerc" not in self._initialized_components:
                logger.error("Cannot perform quantum analysis: QERC initialization failed")
                return {"error": "QERC initialization failed", "available": False}
        
        # Extract price data
        close_prices = market_data.get("close_prices", [])
        if not close_prices:
            # Try extracting from OHLCV data if available
            ohlcv_data = market_data.get("ohlcv_data", [])
            if ohlcv_data and len(ohlcv_data) > 0 and len(ohlcv_data[0]) >= 4:
                close_prices = [candle[4] for candle in ohlcv_data]  # Close is typically at index 4
        
        if len(close_prices) < 30:
            logger.warning("Insufficient price data for quantum analysis (need at least 30 data points)")
            return {"error": "Insufficient data", "available": False}
        
        # Convert to numpy array
        price_data = np.array(close_prices)
        
        try:
            # Use resource scheduler for this important analysis
            with self.resource_scheduler.acquire(
                component_id="qerc",
                resource_type=ResourceType.GPU,
                priority=ResourcePriority.HIGH,  # Strategic analysis gets high priority
                timeout=30.0,  # Allow more time for comprehensive analysis
                description="QERC strategic quantum analysis"
            ) as allocation_id:
                # 1. Perform comprehensive analysis with all quantum indicators
                quantum_indicators = self._analyze_with_quantum_indicators(price_data, market_data)
                
                # 2. Determine market regime based on phase transitions and entropy
                phase_transitions = quantum_indicators.get("phase_transitions", {})
                entropy_results = quantum_indicators.get("quantum_entropy", {})
                momentum_results = quantum_indicators.get("momentum", {})
                fractal_results = quantum_indicators.get("fractal_dimension", {})
                
                regime = self._determine_market_regime(
                    phase_transitions.get("transition_probability", 0.0),
                    entropy_results.get("entropy", 0.5),
                    momentum_results.get("momentum", 0.0),
                    fractal_results.get("complexity", 0.5)
                )
                
                # 3. Generate trading signals based on quantum analysis
                signals = self._generate_quantum_signals(
                    regime,
                    phase_transitions,
                    entropy_results,
                    momentum_results,
                    fractal_results,
                    price_data
                )
                
                # 4. Extract market insights and key observations
                market_insights = self._extract_market_insights(quantum_indicators)
                
                # 5. Risk assessment and position sizing recommendations
                risk_metrics = self._assess_risk_with_quantum_metrics(quantum_indicators, price_data)
                
                # 6. Prepare correlation insights if available
                correlation_insights = {}
                if "correlation_network" in quantum_indicators:
                    correlation_insights = self._analyze_correlation_network(
                        quantum_indicators["correlation_network"]
                    )
                
                # 7. Return comprehensive analysis results
                results = {
                    "available": True,
                    "regime": regime,
                    "signals": signals,
                    "insights": market_insights,
                    "risk": risk_metrics,
                    "indicators": quantum_indicators,
                    "correlation_insights": correlation_insights,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Strategic quantum analysis completed. Market regime: {regime}")
                return results
                
        except Exception as e:
            logger.error(f"Error in strategic quantum analysis: {str(e)}", exc_info=True)
            return {"available": False, "error": str(e)}
    
    def _determine_market_regime(self, transition_probability: float, entropy: float, 
                               momentum: float, complexity: float) -> str:
        """
        Determine the current market regime using quantum indicators
        
        Args:
            transition_probability: Probability of phase transition (0-1)
            entropy: Quantum entropy value measuring market complexity (0-1) 
            momentum: Quantum momentum oscillator value (-1 to 1)
            complexity: Fractal dimension complexity measure (0-2)
            
        Returns:
            String identifying the market regime
        """
        # High transition probability indicates potential regime change
        if transition_probability > 0.7:
            if momentum > 0.3:
                return "BREAKOUT_UP"
            elif momentum < -0.3:
                return "BREAKOUT_DOWN"
            else:
                return "REGIME_CHANGE_IMMINENT"
                
        # Established trends    
        if abs(momentum) > 0.6:
            if momentum > 0:
                return "TRENDING_UP_STRONG"
            else:
                return "TRENDING_DOWN_STRONG"
        elif abs(momentum) > 0.3:
            if momentum > 0:
                return "TRENDING_UP"
            else:
                return "TRENDING_DOWN"
                
        # Ranging/consolidation markets
        if entropy < 0.3:
            return "RANGING_LOW_VOL"
        elif entropy > 0.7:
            return "RANGING_HIGH_VOL"
            
        # Complex/choppy markets
        if complexity > 1.7:
            return "COMPLEX_STRUCTURE"
        elif complexity > 1.5:
            return "FRACTAL_PATTERN"
            
        # Default - mixed/transitional regime
        return "MIXED_SIGNALS"
    
    def _generate_quantum_signals(self, regime: str, phase_transitions: Dict[str, Any],
                                entropy_results: Dict[str, Any], momentum_results: Dict[str, Any],
                                fractal_results: Dict[str, Any], price_data: np.ndarray) -> Dict[str, Any]:
        """
        Generate trading signals based on quantum analysis
        
        Args:
            regime: Identified market regime
            phase_transitions: Results from phase transition detector
            entropy_results: Results from entropy analyzer
            momentum_results: Results from momentum oscillator
            fractal_results: Results from fractal dimension estimator
            price_data: Price time series for context
            
        Returns:
            Dictionary with entry/exit signals, probabilities, and recommendations
        """
        # Initialize signal containers
        entry_signals = {"long": False, "short": False, "probability": 0.5, "strength": 0.0}
        exit_signals = {"long": False, "short": False, "probability": 0.5}
        
        # Extract key metrics
        momentum = momentum_results.get("momentum", 0.0)
        entropy = entropy_results.get("entropy", 0.5)
        transition_prob = phase_transitions.get("transition_probability", 0.0)
        complexity = fractal_results.get("complexity", 0.5)
        
        # Signal generation logic based on regime
        if regime.startswith("TRENDING_UP"):
            entry_signals["long"] = True
            entry_signals["probability"] = 0.7 if "STRONG" in regime else 0.6
            entry_signals["strength"] = min(0.9, abs(momentum) * 1.2)  # Scale momentum as signal strength
            
        elif regime.startswith("TRENDING_DOWN"):
            entry_signals["short"] = True 
            entry_signals["probability"] = 0.7 if "STRONG" in regime else 0.6
            entry_signals["strength"] = min(0.9, abs(momentum) * 1.2)
            
        elif regime == "BREAKOUT_UP":
            entry_signals["long"] = True
            entry_signals["probability"] = 0.75
            entry_signals["strength"] = min(0.95, transition_prob * 1.3)
            
        elif regime == "BREAKOUT_DOWN":
            entry_signals["short"] = True
            entry_signals["probability"] = 0.75 
            entry_signals["strength"] = min(0.95, transition_prob * 1.3)
            
        elif "RANGING" in regime:
            # Ranging markets - lower probability mean-reversion signals
            current_position = self._calculate_relative_position(price_data)
            
            if current_position > 0.8:  # Near top of range
                entry_signals["short"] = True
                entry_signals["probability"] = 0.55
                entry_signals["strength"] = 0.3
            elif current_position < 0.2:  # Near bottom of range
                entry_signals["long"] = True
                entry_signals["probability"] = 0.55
                entry_signals["strength"] = 0.3
        
        # Calculate recommended position size (0-1) based on signal strength and risk
        risk_adjustment = self._calculate_risk_adjustment(regime, entropy, complexity)
        recommended_size = entry_signals["strength"] * risk_adjustment
        
        return {
            "entry": entry_signals,
            "exit": exit_signals,
            "risk_adjustment": risk_adjustment,
            "recommended_position_size": recommended_size
        }
    
    def _calculate_relative_position(self, price_data: np.ndarray, lookback: int = 20) -> float:
        """
        Calculate current price position relative to recent range
        
        Args:
            price_data: Array of price data
            lookback: Number of periods to consider for range
            
        Returns:
            Float between 0-1 indicating position in range (0=bottom, 1=top)
        """
        if len(price_data) < lookback:
            return 0.5  # Not enough data, return middle
            
        # Get recent price data
        recent_prices = price_data[-lookback:]
        
        # Find range
        price_min = np.min(recent_prices)
        price_max = np.max(recent_prices)
        price_range = price_max - price_min
        
        if price_range < 1e-9:  # Avoid division by zero
            return 0.5
            
        # Calculate current position in range
        current_price = price_data[-1]
        position = (current_price - price_min) / price_range
        
        return position
    
    def _calculate_risk_adjustment(self, regime: str, entropy: float, complexity: float) -> float:
        """
        Calculate risk adjustment factor for position sizing
        
        Args:
            regime: Market regime identification
            entropy: Market entropy/uncertainty measure
            complexity: Market complexity/fractal measure
            
        Returns:
            Float between 0-1 indicating how much to adjust position size based on risk
        """
        # Base risk adjustment starts at 0.8 (moderately conservative)
        risk_adjustment = 0.8
        
        # Adjust based on regime
        regime_risk_factors = {
            "TRENDING_UP_STRONG": 0.2,      # +0.2 (lower risk in strong uptrend)
            "TRENDING_UP": 0.1,           # +0.1 (slightly lower risk in uptrend)
            "TRENDING_DOWN": -0.1,        # -0.1 (slightly higher risk in downtrend)
            "TRENDING_DOWN_STRONG": -0.15, # -0.15 (higher risk in strong downtrend)
            "BREAKOUT_UP": 0.05,          # +0.05 (slightly lower risk in upward breakout)
            "BREAKOUT_DOWN": -0.1,        # -0.1 (higher risk in downward breakout)
            "RANGING_LOW_VOL": 0.15,      # +0.15 (lower risk in quiet range)
            "RANGING_HIGH_VOL": -0.25,     # -0.25 (much higher risk in volatile range)
            "REGIME_CHANGE_IMMINENT": -0.3, # -0.3 (very high risk during regime change)
            "COMPLEX_STRUCTURE": -0.2,     # -0.2 (higher risk in complex markets)
            "FRACTAL_PATTERN": -0.15,      # -0.15 (higher risk in fractal markets)
            "MIXED_SIGNALS": -0.1          # -0.1 (slightly higher risk with mixed signals)
        }
        
        # Apply regime-specific adjustment
        risk_adjustment += regime_risk_factors.get(regime, 0)
        
        # Further adjust based on entropy (uncertainty)
        # High entropy = more uncertainty = higher risk
        risk_adjustment -= (entropy - 0.5) * 0.3  # Subtract up to 0.15 for high entropy
        
        # Further adjust based on complexity
        # Higher complexity = higher risk
        if complexity > 1.5:
            risk_adjustment -= (complexity - 1.5) * 0.2  # Subtract up to 0.1 for very complex markets
        
        # Ensure risk adjustment stays within bounds
        return max(0.1, min(1.0, risk_adjustment))
        

    
    def _detect_price_pattern(self, price_data: np.ndarray) -> Optional[str]:
        """
        Detect common price patterns in the data
        
        Args:
            price_data: Array of price data
            
        Returns:
            String describing the detected pattern, or None if no pattern detected
        """
        # Need sufficient data for pattern detection
        if len(price_data) < 30:
            return None
            
        # Get recent price data for pattern detection
        recent_prices = price_data[-30:]
        
        # Calculate returns
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Simple pattern detection
        # Double top pattern
        if self._is_double_top(recent_prices):
            return "Double Top"
            
        # Double bottom pattern
        if self._is_double_bottom(recent_prices):
            return "Double Bottom"
            
        # Head and shoulders (simplified)
        if self._is_head_and_shoulders(recent_prices):
            return "Head and Shoulders"
            
        # Check for other patterns...
        
        return None
        
    def _is_double_top(self, prices: np.ndarray) -> bool:
        """Simplified double top detection"""
        if len(prices) < 20:
            return False
            
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prices, distance=5)
        
        if len(peaks) >= 2:
            peak1 = prices[peaks[-2]]
            peak2 = prices[peaks[-1]]
            
            # Check if peaks are similar in height
            if abs(peak1 - peak2) / peak1 < 0.03 and peak1 > np.mean(prices) and peak2 > np.mean(prices):
                return True
                
        return False
        
    def _is_double_bottom(self, prices: np.ndarray) -> bool:
        """Simplified double bottom detection"""
        if len(prices) < 20:
            return False
            
        # Find valleys (invert prices to find peaks in the inverted series)
        from scipy.signal import find_peaks
        inverted = -prices
        valleys, _ = find_peaks(inverted, distance=5)
        
        if len(valleys) >= 2:
            valley1 = prices[valleys[-2]]
            valley2 = prices[valleys[-1]]
            
            # Check if valleys are similar in height
            if abs(valley1 - valley2) / valley1 < 0.03 and valley1 < np.mean(prices) and valley2 < np.mean(prices):
                return True
                
        return False
        
    def _is_head_and_shoulders(self, prices: np.ndarray) -> bool:
        """Simplified head and shoulders detection"""
        if len(prices) < 25:
            return False
            
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(prices, distance=3)
        
        if len(peaks) >= 3:
            # Get last three peaks
            peak1 = prices[peaks[-3]]
            peak2 = prices[peaks[-2]]
            peak3 = prices[peaks[-1]]
            
            # Check for head and shoulders pattern
            if peak2 > peak1 and peak2 > peak3 and abs(peak1 - peak3) / peak1 < 0.05:
                return True
                
        return False
    
    def _assess_risk_with_quantum_metrics(self, quantum_indicators: Dict[str, Any], 
                                       price_data: np.ndarray) -> Dict[str, Any]:
        """
        Assess market risk using quantum indicators
        
        Args:
            quantum_indicators: Results from all quantum indicators
            price_data: Price time series for context
            
        Returns:
            Dictionary with risk assessment metrics
        """
        # Initialize risk assessment
        risk_assessment = {
            "overall_risk": 0.5,     # 0-1 scale (higher = more risky)
            "volatility_risk": 0.5,  # 0-1 scale (higher = more volatile)
            "trend_risk": 0.5,      # 0-1 scale (higher = more trend reversal risk)
            "complexity_risk": 0.5,  # 0-1 scale (higher = more complex structure risk)
            "risk_factors": []
        }
        
        # Extract values from indicators
        entropy = quantum_indicators.get("quantum_entropy", {}).get("entropy", 0.5)
        transition_prob = quantum_indicators.get("phase_transitions", {}).get("transition_probability", 0.0)
        momentum = quantum_indicators.get("momentum", {}).get("momentum", 0.0)
        complexity = quantum_indicators.get("fractal_dimension", {}).get("complexity", 0.5)
        
        # Calculate volatility risk from entropy and recent price volatility
        std_dev = np.std(price_data[-20:]) / np.mean(price_data[-20:]) if len(price_data) >= 20 else 0.01
        scaled_std = min(1.0, std_dev * 20)  # Scale to 0-1 range
        risk_assessment["volatility_risk"] = 0.4 * entropy + 0.6 * scaled_std
        
        # Calculate trend risk based on momentum and transition probability
        momentum_reversal_risk = 1.0 - abs(momentum)  # Lower absolute momentum = higher reversal risk
        risk_assessment["trend_risk"] = 0.7 * transition_prob + 0.3 * momentum_reversal_risk
        
        # Calculate complexity risk directly from fractal dimension
        if complexity > 1.0:
            # Scale from 1.0-2.0 range to 0-1 range
            risk_assessment["complexity_risk"] = min(1.0, (complexity - 1.0))
        else:
            risk_assessment["complexity_risk"] = 0.0
        
        # Calculate overall risk as weighted average of components
        risk_assessment["overall_risk"] = (
            0.4 * risk_assessment["volatility_risk"] +
            0.4 * risk_assessment["trend_risk"] +
            0.2 * risk_assessment["complexity_risk"]
        )
        
        # Identify key risk factors
        risk_factors = []
        
        if risk_assessment["volatility_risk"] > 0.7:
            risk_factors.append(f"High volatility risk ({risk_assessment['volatility_risk']:.2f})")
        if risk_assessment["trend_risk"] > 0.7:
            risk_factors.append(f"High trend reversal risk ({risk_assessment['trend_risk']:.2f})")
        if risk_assessment["complexity_risk"] > 0.7:
            risk_factors.append(f"High complexity risk ({risk_assessment['complexity_risk']:.2f})")
        if transition_prob > 0.6:
            risk_factors.append(f"High regime change probability ({transition_prob:.2f})")
        
        risk_assessment["risk_factors"] = risk_factors
        
        return risk_assessment
        
    def _analyze_with_quantum_indicators(self, price_data: np.ndarray, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data with quantum indicators
        
        This method uses the optimized quantum indicators for in-depth market analysis:
        1. Phase transitions for regime change detection
        2. Quantum entropy for market complexity measurement
        3. Momentum oscillator for trend strength
        4. Fractal dimension for market structure complexity
        5. Correlation network for multi-asset relationships
        
        Args:
            price_data: Array of price data
            market_data: Dictionary containing market data including multiple assets if available
            
        Returns:
            Dictionary of quantum indicator results
        """
        results = {}
        
        try:
            # 1. Detect phase transitions for regime changes
            phase_results = self.qerc.quantum_phase_transition_detector(price_data)
            results["phase_transitions"] = {
                "critical_points": phase_results.get("critical_points", []),
                "transition_probability": phase_results.get("latest_transition_probability", 0),
                "latest_criticality": phase_results.get("latest_criticality", 0),
                "is_critical_point": phase_results.get("is_critical_point", False)
            }
            
            # 2. Analyze quantum entropy for market complexity
            entropy_results = self.qerc.quantum_entropy_analyzer(price_data)
            results["quantum_entropy"] = {
                "entropy": entropy_results.get("latest_quantum_entropy", 0),
                "entropy_trend": entropy_results.get("entropy_trend", 0),
                "complexity": entropy_results.get("latest_complexity", 0),
                "is_high_entropy": entropy_results.get("is_high_entropy", False)
            }
            
            # 3. Calculate quantum momentum oscillator
            momentum_results = self.qerc.quantum_momentum_oscillator(price_data, window_size=30)
            results["momentum"] = {
                "oscillator": momentum_results.get("latest_oscillator", 50),
                "momentum": momentum_results.get("latest_momentum", 0),
                "overbought": momentum_results.get("overbought", False),
                "oversold": momentum_results.get("oversold", False),
                "divergences": momentum_results.get("divergences", [])
            }
            
            # 4. Estimate fractal dimension
            fractal_results = self.qerc.quantum_fractal_dimension_estimator(price_data, window_size=30)
            results["fractal_dimension"] = {
                "dimension": fractal_results.get("latest_fractal_dimension", 1.5),
                "complexity": fractal_results.get("latest_complexity", 0.5),
                "increasing": fractal_results.get("complexity_increasing", False)
            }
            
            # 5. Analyze correlation network if multi-asset data is available
            if "asset_prices" in market_data and isinstance(market_data["asset_prices"], dict):
                asset_prices = market_data["asset_prices"]
                if len(asset_prices) >= 2:  # Need at least 2 assets for correlation
                    network_results = self.qerc.quantum_correlation_network(asset_prices, window_size=20)
                    results["correlation_network"] = {
                        "entropy": network_results.get("latest_correlation_entropy", 0),
                        "average_correlation": network_results.get("latest_average_correlation", 0),
                        "central_asset": network_results.get("central_asset", None),
                        "key_relationships": network_results.get("key_relationships", [])
                    }
            
            # Update system state based on quantum indicators
            self._update_market_regime_from_quantum_indicators(results)
            
        except Exception as e:
            logger.error(f"Error running quantum indicators: {e}")
        
        return results
    
    def _update_market_regime_from_quantum_indicators(self, indicator_results: Dict[str, Any]) -> None:
        """Update the market regime based on quantum indicator results
        
        This method uses quantum indicators to determine the current market regime,
        helping the system adapt to changing market conditions.
        
        Args:
            indicator_results: Results from quantum indicators
        """
        # Store previous regime for change detection
        self._previous_regime = self.current_regime
        
        # Extract key indicators
        is_critical = indicator_results.get("phase_transitions", {}).get("is_critical_point", False)
        transition_prob = indicator_results.get("phase_transitions", {}).get("transition_probability", 0)
        entropy = indicator_results.get("quantum_entropy", {}).get("entropy", 0)
        entropy_trend = indicator_results.get("quantum_entropy", {}).get("entropy_trend", 0)
        momentum = indicator_results.get("momentum", {}).get("momentum", 0)
        oscillator = indicator_results.get("momentum", {}).get("oscillator", 50)
        fractal_dim = indicator_results.get("fractal_dimension", {}).get("dimension", 1.5)
        complexity = indicator_results.get("fractal_dimension", {}).get("complexity", 0.5)
        
        # Update entropy (class variable)
        self.entropy = entropy
        
        # Determine market regime based on quantum indicators
        # High transition probability and critical point suggests regime change
        if is_critical and transition_prob > 0.7:
            if momentum > 0.3 and oscillator > 60:
                self.current_regime = MarketRegime.TRENDING_UP
                logger.info("Quantum indicators detected TRENDING_UP regime")
            elif momentum < -0.3 and oscillator < 40:
                self.current_regime = MarketRegime.TRENDING_DOWN
                logger.info("Quantum indicators detected TRENDING_DOWN regime")
            elif entropy > 0.6 and entropy_trend > 0:
                self.current_regime = MarketRegime.BREAKOUT
                logger.info("Quantum indicators detected BREAKOUT regime")
            elif fractal_dim > 1.8 and complexity > 0.7:
                self.current_regime = MarketRegime.REVERSAL
                logger.info("Quantum indicators detected REVERSAL regime")
        # No critical transition but other indicators can still identify the regime
        else:
            if entropy < 0.3 and complexity < 0.4:
                self.current_regime = MarketRegime.RANGING_LOW_VOL
                logger.info("Quantum indicators detected RANGING_LOW_VOL regime")
            elif entropy > 0.5 and complexity > 0.6:
                self.current_regime = MarketRegime.RANGING_HIGH_VOL
                logger.info("Quantum indicators detected RANGING_HIGH_VOL regime")
            # Keep current regime if no clear signals
        
        # Detect regime changes
        if self._previous_regime is not None and self._previous_regime != self.current_regime:
            logger.info(f"Market regime change: {self._previous_regime} -> {self.current_regime}")
            # Update metrics
            self.metrics["regime_changes"] += 1

    def _derive_trading_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Derive trading signals from quantum indicators
        
        Args:
            indicators: Dictionary containing quantum indicator results
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": {
                "long": False,
                "short": False,
                "probability": 0.0,
                "strength": 0.0
            },
            "exit": {
                "long": False,
                "short": False,
                "probability": 0.0,
                "strength": 0.0
            },
            "risk_adjustment": 1.0,  # Normal risk
            "opportunity_quality": 0.0,  # 0-1 scale
            "recommended_position_size": 0.0  # 0-1 scale
        }
        
        # Extract key parameters for signal generation
        entropy = indicators.get("quantum_entropy", {}).get("entropy", 0)
        momentum = indicators.get("momentum", {}).get("momentum", 0)
        oscillator = indicators.get("momentum", {}).get("oscillator", 50)
        is_oversold = indicators.get("momentum", {}).get("oversold", False)
        is_overbought = indicators.get("momentum", {}).get("overbought", False)
        fractal_complexity = indicators.get("fractal_dimension", {}).get("complexity", 0.5)
        is_critical = indicators.get("phase_transitions", {}).get("is_critical_point", False)
        
        # Calculate signal probabilities based on indicator combinations
        long_probability = 0.0
        short_probability = 0.0
        
        # Long entry signals
        if is_oversold and momentum > 0.1 and self.current_regime != MarketRegime.TRENDING_DOWN:
            # Oversold with positive momentum is a long entry signal
            long_probability = 0.7 + min(momentum, 0.3)  # Max 1.0
            signals["entry"]["long"] = True
            signals["entry"]["strength"] = long_probability
        
        # Short entry signals
        if is_overbought and momentum < -0.1 and self.current_regime != MarketRegime.TRENDING_UP:
            # Overbought with negative momentum is a short entry signal
            short_probability = 0.7 + min(abs(momentum), 0.3)  # Max 1.0
            signals["entry"]["short"] = True
            signals["entry"]["strength"] = short_probability
        
        # Exit signals
        if signals["entry"]["long"] and is_overbought:
            signals["exit"]["long"] = True
            signals["exit"]["probability"] = 0.8
        elif signals["entry"]["short"] and is_oversold:
            signals["exit"]["short"] = True
            signals["exit"]["probability"] = 0.8
        
        # Adjust risk based on market complexity and entropy
        if fractal_complexity > 0.7 or entropy > 0.6:
            # High complexity/entropy means higher uncertainty - reduce risk
            signals["risk_adjustment"] = max(0.3, 1.0 - fractal_complexity)
        elif is_critical:
            # Critical points can be volatile - reduce risk
            signals["risk_adjustment"] = 0.5
        elif self.current_regime in [MarketRegime.RANGING_LOW_VOL, MarketRegime.TRENDING_UP]:
            # More predictable regimes - can increase risk slightly
            signals["risk_adjustment"] = min(1.2, 1.0 + momentum * 0.5)
        
        # Determine overall opportunity quality
        if signals["entry"]["long"] or signals["entry"]["short"]:
            signal_strength = max(signals["entry"]["strength"], 0.1)
            regime_quality = 0.8 if self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] else 0.5
            signals["opportunity_quality"] = signal_strength * regime_quality * (1.0 - entropy * 0.5)
            
            # Calculate recommended position size
            signals["recommended_position_size"] = signals["opportunity_quality"] * signals["risk_adjustment"]
        
        # Set highest probability
        signals["entry"]["probability"] = max(long_probability, short_probability)
        
        return signals
    
    def _extract_market_insights(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key market insights from quantum indicators
        
        This method analyzes quantum indicator results to generate actionable market insights,
        including market stability, trend strength, and potential reversals or breakouts.
        It combines the functionality of both the original and enhanced versions.
        
        Args:
            indicators: Dictionary containing quantum indicator results including:
                - quantum_entropy: Entropy analysis results
                - momentum: Momentum oscillator values
                - fractal_dimension: Market complexity metrics
                - phase_transitions: Regime change probabilities
                
        Returns:
            Dictionary containing market insights with the following structure:
            {
                'market_stability': float (0-1),
                'trend_strength': float (0-1),
                'reversal_probability': float (0-1),
                'breakout_probability': float (0-1),
                'market_complexity': float (0-1),
                'key_observations': List[str]
            }
        """
        insights = {
            "market_stability": 0.5,  # 0-1 scale (0=unstable, 1=stable)
            "trend_strength": 0.5,   # 0-1 scale
            "reversal_probability": 0.5,  # 0-1 scale
            "breakout_probability": 0.5,  # 0-1 scale
            "market_complexity": 0.5,  # 0-1 scale (0=simple, 1=complex)
            "key_observations": []
        }
        
        try:
            # Extract relevant indicators with safe defaults
            entropy = indicators.get("quantum_entropy", {}).get("entropy", 0.5)
            momentum = indicators.get("momentum", {}).get("momentum", 0.0)
            oscillator = indicators.get("momentum", {}).get("oscillator", 50.0)
            is_oversold = indicators.get("momentum", {}).get("oversold", False)
            is_overbought = indicators.get("momentum", {}).get("overbought", False)
            fractal_complexity = indicators.get("fractal_dimension", {}).get("complexity", 0.5)
            is_critical = indicators.get("phase_transitions", {}).get("is_critical_point", False)
            transition_prob = indicators.get("phase_transitions", {}).get("transition_probability", 0.0)
            fractal_dim = indicators.get("fractal_dimension", {}).get("dimension", 1.5)
            divergences = indicators.get("momentum", {}).get("divergences", [])
            
            # 1. Calculate market stability (inverse of entropy and transition probability)
            insights["market_stability"] = max(0.0, min(1.0, 1.0 - (entropy * 0.5 + transition_prob * 0.5)))
            
            # 2. Calculate trend strength (based on momentum and oscillator)
            insights["trend_strength"] = min(1.0, abs(momentum) * 1.5)
            if is_overbought or is_oversold:
                insights["trend_strength"] = min(1.0, insights["trend_strength"] + 0.2)
            
            # 3. Calculate reversal probability
            insights["reversal_probability"] = min(1.0, transition_prob * 0.7 + (fractal_dim - 1.5) * 0.5)
            if divergences:
                insights["reversal_probability"] = min(1.0, insights["reversal_probability"] + 0.3)
            if hasattr(self, 'current_regime') and self.current_regime == MarketRegime.REVERSAL:
                insights["reversal_probability"] = min(1.0, insights["reversal_probability"] + 0.4)
            
            # 4. Calculate breakout probability (from both versions)
            # From version 1:
            breakout_prob_v1 = min(1.0, entropy * 0.5 + transition_prob * 0.5)
            if hasattr(self, 'current_regime') and self.current_regime == MarketRegime.BREAKOUT:
                breakout_prob_v1 = min(1.0, breakout_prob_v1 + 0.4)
            
            # From version 2:
            breakout_prob_v2 = min(1.0, (entropy - 0.5) * 2.0)  # Higher entropy suggests potential breakout
            if is_critical:
                breakout_prob_v2 = min(1.0, breakout_prob_v2 + 0.4)
            
            # Combine both approaches
            insights["breakout_probability"] = max(breakout_prob_v1, breakout_prob_v2)
            
            # 5. Calculate market complexity (from both versions)
            # From version 1:
            complexity_v1 = fractal_complexity
            
            # From version 2:
            complexity_v2 = min(1.0, max(0.0, (fractal_dim - 1.3) * 1.5))  # 1.3-1.97 maps to ~0-1
            
            # Combine both approaches
            insights["market_complexity"] = max(complexity_v1, complexity_v2)
            
            # 6. Generate key observations (combining both versions)
            observations = []
            
            # Market stability observations
            if insights["market_stability"] < 0.3:
                observations.append("Market shows high instability, consider reducing position sizes.")
            elif insights["market_stability"] > 0.8:
                observations.append("Market shows high stability, favorable for trend-following strategies.")
                
            # Trend strength observations
            if insights["trend_strength"] > 0.7:
                direction = "up" if momentum > 0 else "down"
                observations.append(f"Strong {direction} trend detected, consider trend-following strategies.")
            
            # Reversal probability observations
            if insights["reversal_probability"] > 0.7:
                observations.append("High probability of market reversal, consider taking profits or tightening stops.")
            
            # Breakout observations
            if insights["breakout_probability"] > 0.7:
                observations.append("Potential breakout detected, watch for confirmation.")
            
            # Market complexity observations
            if insights["market_complexity"] > 0.7:
                observations.append("High market complexity detected, consider reducing position sizes.")
            
            # Add divergence observations if present
            if divergences:
                observations.append(f"Found {len(divergences)} momentum divergence(s), potential reversal signals.")
            
            # Add critical point observation if available
            if is_critical:
                observations.append("Critical market point detected, high probability of significant move.")
            
            # Add oscillator extremes if available
            if is_overbought:
                observations.append("Market is overbought, watch for potential pullback.")
            elif is_oversold:
                observations.append("Market is oversold, watch for potential bounce.")
            
            # Add classic observations from version 1 if not already covered
            if not any("high instability" in obs for obs in observations) and insights["market_stability"] < 0.3:
                observations.append("Market showing high instability, reduce position sizes.")
                
            if not any("trend detected" in obs for obs in observations) and insights["trend_strength"] > 0.7:
                observations.append("Strong trend detected, trend-following strategies favored.")
                
            if not any("reversal" in obs.lower() for obs in observations) and insights["reversal_probability"] > 0.7:
                observations.append("High probability of market reversal, prepare for trend change.")
                
            if not any("breakout" in obs.lower() for obs in observations) and insights["breakout_probability"] > 0.7:
                observations.append("Breakout likely, watch for volatility expansion.")
                
            if not any("complex" in obs.lower() for obs in observations) and insights["market_complexity"] > 0.7:
                observations.append("Complex market structure, reduce trade frequency.")
            
            # Limit to top 10 most important observations
            insights["key_observations"] = observations[:10]
            
        except Exception as e:
            logger.error(f"Error extracting market insights: {str(e)}", exc_info=True)
            insights["key_observations"].append("Error generating market insights. Using default values.")
        
        return insights
    
    def _analyze_correlation_network(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze asset correlation network for insights
        
        Args:
            correlation_data: Dictionary with correlation network information
            
        Returns:
            Dictionary with correlation insights
        """
        insights = {
            "diversification_score": 0.0,  # 0-1 scale (0=highly correlated, 1=well diversified)
            "central_influence": 0.0,     # 0-1 scale (influence of central asset)
            "key_relationships": [],      # Important asset relationships
            "portfolio_recommendations": []
        }
        
        # Extract correlation data
        avg_correlation = correlation_data.get("average_correlation", 0)
        central_asset = correlation_data.get("central_asset", None)
        key_relationships = correlation_data.get("key_relationships", [])
        
        # Calculate diversification score (inverse of average correlation)
        insights["diversification_score"] = max(0, 1.0 - avg_correlation)
        
        # Determine central influence if a central asset exists
        if central_asset:
            # Central asset with strong correlations has high influence
            insights["central_influence"] = min(1.0, avg_correlation * 1.5)
            insights["key_relationships"].append(f"{central_asset} is central to the correlation network")
        
        # Analyze key relationships
        for rel in key_relationships[:5]:  # Limit to top 5
            if rel.get("correlation", 0) > 0.8:
                insights["key_relationships"].append(
                    f"Strong positive correlation ({rel.get('correlation', 0):.2f}) between {rel.get('asset1', '')} and {rel.get('asset2', '')}"
                )
            elif rel.get("correlation", 0) < -0.8:
                insights["key_relationships"].append(
                    f"Strong negative correlation ({rel.get('correlation', 0):.2f}) between {rel.get('asset1', '')} and {rel.get('asset2', '')}"
                )
        
        # Generate portfolio recommendations
        if insights["diversification_score"] < 0.3:
            insights["portfolio_recommendations"].append("Portfolio highly correlated, consider adding uncorrelated assets.")
        if insights["central_influence"] > 0.7 and central_asset:
            insights["portfolio_recommendations"].append(f"{central_asset} has dominant influence on portfolio, consider reducing exposure.")
        if len(key_relationships) > 0:
            pos_correlations = [r for r in key_relationships if r.get("correlation", 0) > 0.5]
            neg_correlations = [r for r in key_relationships if r.get("correlation", 0) < -0.5]
            if len(pos_correlations) > len(key_relationships) * 0.7:
                insights["portfolio_recommendations"].append("Many assets moving together, portfolio vulnerable to systematic risk.")
            if len(neg_correlations) > 2:
                insights["portfolio_recommendations"].append("Found negatively correlated assets, good for hedging strategies.")
        
        return insights

    def integrate_parameter_optimization(self, parameters: Dict[str, float], objective_fn: Callable, market_data: Dict[str, Any] = None):
        """
        Integrate parameter optimization capabilities using NQO
        Components are loaded on-demand based on regime changes and periodic tuning needs
        
        Args:
            parameters: Dictionary of current parameters
            objective_fn: Function to evaluate parameter quality
            market_data: Optional market data for determining optimization needs
            
        Returns:
            Dictionary with optimized parameters
        """
        # Check if component should be loaded based on optimization criteria
        load_for_optimization = self._should_load_component("nqo", market_data)
        
        # If component isn't active but should be loaded, initialize it
        if load_for_optimization and "nqo" not in self._active_components:
            logger.info("Loading NQO on-demand for parameter optimization due to regime change or time threshold")
            if not self._initialize_nqo():
                logger.warning("Failed to load NQO component")
                return {"optimized_parameters": parameters, "optimization_available": False, "error": "Component initialization failed"}
        
        # Check if component is active, if not, return default values
        if "nqo" not in self._active_components:
            logger.info("Skipping parameter optimization: not scheduled for current time period")
            return {"optimized_parameters": parameters, "optimization_available": False, "required": False}
        
        # Update component usage stats
        self._component_last_used["nqo"] = time.time()
        self._component_usage_count["nqo"] = self._component_usage_count.get("nqo", 0) + 1
        
        try:
            # Acquire resources for parameter optimization with appropriate priority
            with self.resource_scheduler.acquire(
                component_id="nqo",
                resource_type=ResourceType.QUANTUM_DEVICE, 
                priority=ResourcePriority.MEDIUM, 
                timeout=30.0,
                description="NQO parameter optimization"
            ) as resource_id:
                logger.info(f"Acquired quantum resources for parameter optimization: {resource_id}")
                
                # Define a wrapper for the cost function to avoid reference issues
                def cost_wrapper(params_array):
                    # Convert numpy array back to dictionary format for the objective function
                    params_dict = {}
                    param_names = list(parameters.keys())
                    for i, name in enumerate(param_names):
                        if i < len(params_array):
                            params_dict[name] = float(params_array[i])
                        else:
                            params_dict[name] = parameters[name]
                    
                    # Call the objective function and return the cost
                    return objective_fn(params_dict)
                
                # Convert parameters to numpy array for optimization
                params_array = np.array(list(parameters.values()))
                param_names = list(parameters.keys())
                
                # Use the best available method in NQO for optimization
                if hasattr(self.nqo, 'optimize_parameters'):
                    # Standard method for parameter optimization
                    optimization_results = self.nqo.optimize_parameters(
                        params_array,
                        cost_wrapper,
                        learning_rate=self.config.rl_learning_rate,
                        iterations=100
                    )
                    
                    # Convert the results back to dictionary format
                    optimized_params = {}
                    if isinstance(optimization_results, dict) and "optimized_parameters" in optimization_results:
                        for i, key in enumerate(param_names):
                            optimized_params[key] = float(optimization_results["optimized_parameters"][i])
                    elif isinstance(optimization_results, np.ndarray):
                        for i, key in enumerate(param_names):
                            optimized_params[key] = float(optimization_results[i])
                    else:
                        optimized_params = parameters
                        
                    # Update metrics
                    self.metrics["optimization_runs"] += 1
                    
                    # Construct result with the optimized parameters
                    result = {
                        "optimized_parameters": optimized_params,
                        "optimization_available": True,
                        "improvement": optimization_results.get("improvement", 0.0),
                        "iterations": optimization_results.get("iterations", 100)
                    }
                    
                    # Unload the NQO component after use since we don't need it continuously
                    if "nqo" in self._active_components and load_for_optimization:
                        logger.info("Unloading NQO after parameter optimization completion")
                        self._unload_component("nqo")
                        
                    return result
                    
                elif hasattr(self.nqo, 'optimize'):
                    # Alternative method that might be available
                    optimization_results = self.nqo.optimize(
                        parameters=parameters,
                        cost_function=objective_fn,
                        max_iterations=100
                    )
                    
                    # Update metrics
                    self.metrics["optimization_runs"] += 1
                    
                    # Construct result
                    result = {
                        "optimized_parameters": optimization_results.get("parameters", parameters),
                        "optimization_available": True,
                        "improvement": optimization_results.get("improvement", 0.0),
                        "iterations": optimization_results.get("iterations", 100)
                    }
                    
                    # Unload the NQO component after use
                    if "nqo" in self._active_components and load_for_optimization:
                        logger.info("Unloading NQO after parameter optimization completion")
                        self._unload_component("nqo")
                        
                    return result
                
                else:
                    # No suitable method found
                    logger.warning("NQO does not have optimization methods available")
                    
                    # Unload the component since it's not useful
                    if "nqo" in self._active_components and load_for_optimization:
                        logger.info("Unloading NQO as no compatible optimization method found")
                        self._unload_component("nqo")
                    
                    return {
                        "optimized_parameters": parameters,
                        "optimization_available": False,
                        "reason": "No compatible optimization method"
                    }
        
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            
            # Make sure to unload the component even on error
            if "nqo" in self._active_components and load_for_optimization:
                logger.info("Unloading NQO after parameter optimization error")
                self._unload_component("nqo")
                
            return {
                "optimized_parameters": parameters,
                "optimization_available": False,
                "error": str(e)
            }


    @quantum_accelerated(use_hw_accel=True, hw_batch_size=8)
    def optimize_parameters(self, parameters: Dict[str, float], cost_function: Callable) -> Dict[str, Any]:
        """
        Optimize parameters using the Neuromorphic Quantum Optimizer
        GPU-accelerated via the quantum_accelerated decorator
        
        Args:
            parameters: Dictionary of parameters to optimize
            cost_function: Function that evaluates parameters and returns a cost (lower is better)
            
        Returns:
            Optimization results
        """
        if "nqo" not in self._initialized_components:
            logger.warning("NQO not available for parameter optimization")
            return {"optimized": False, "parameters": parameters}
            
        try:
            # Acquire resources with high priority for this important operation
            with self.resource_scheduler.acquire(ResourceType.QUANTUM_ANNEALER, "nqo", 
                                               priority=ResourcePriority.HIGH,
                                               timeout=60) as resource_id:
                logger.info(f"Acquired quantum resources for parameter optimization: {resource_id}")
                
                # Define a wrapper for the cost function to avoid reference issues
                def cost_wrapper(params):
                    # Convert numpy array to dictionary if needed
                    if isinstance(params, np.ndarray):
                        params_dict = dict(zip(parameters.keys(), params))
                    else:
                        params_dict = params
                    return cost_function(params_dict)
                
                # Determine budget allocation based on complexity
                complexity = len(parameters)
                budget_allocation = max(100, complexity * 20)  # Scale budget with parameter count
                
                # Set optimization constraints
                optimization_constraints = {
                    "max_iterations": budget_allocation,
                    "convergence_threshold": 1e-4
                }
                
                # Use timeout to prevent long-running operations
                with self._timeout(30, "parameter optimization"):
                    # Apply resource constraints based on hardware capabilities
                    max_memory = 1024 * 1024 * 128  # 128 MB
                    max_time = 30000  # 30 seconds in ms
                    
                    # Check available methods in the NQO object
                    if hasattr(self.nqo, 'optimize'):
                        optimization_result = self.nqo.optimize(parameters, budget=budget_allocation, constrains=optimization_constraints)
                    elif hasattr(self.nqo, 'optimize_parameters'):
                        # Convert parameters to required format
                        params_array = np.array(list(parameters.values()))
                        param_names = list(parameters.keys())
                        
                        optimization_result = self.nqo.optimize_parameters(
                            params_array, 
                            cost_wrapper, 
                            learning_rate=self.config.rl_learning_rate,
                            max_iterations=optimization_constraints['max_iterations']
                        )
                        
                        # Convert optimized parameters back to dictionary format
                        optimized_params = {}
                        for i, key in enumerate(parameters.keys()):
                            if isinstance(optimization_result, dict) and "parameters" in optimization_result:
                                optimized_params[key] = float(optimization_result["parameters"][i])
                            elif isinstance(optimization_result, np.ndarray):
                                optimized_params[key] = float(optimization_result[i])
                            else:
                                optimized_params[key] = parameters[key]
                        
                        optimization_result = {
                            "optimized": True,
                            "parameters": optimized_params,
                            "improvement": 0.0,  # Default values since we don't have this info
                            "iterations": optimization_constraints['max_iterations']
                        }
                    else:
                        # If method not available, use a mock implementation
                        logger.warning("NQO optimization method not available, using fallback")
                        optimization_result = {"optimized": False, "parameters": parameters}
        
                # Convert optimized parameters to return format
                result = {
                    "optimized": optimization_result.get("optimized", False),
                    "parameters": optimization_result.get("parameters", parameters),
                    "improvement": optimization_result.get("improvement", 0.0),
                    "iterations": optimization_result.get("iterations", 0),
                    "final_cost": optimization_result.get("final_cost", 0.0)
                }
                
                # Update metrics
                self.metrics["optimization_runs"] += 1
                
                return result
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return {"optimized": False, "parameters": parameters, "error": str(e)}
        
    
    def detect_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Detect current market regime using multiple quantum components
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Detected market regime
        """
        # Get forecasts, anomalies and features
        forecast = self.integrate_forecasting(market_data)
        anomalies = self.integrate_anomaly_detection(market_data)
        features = self.integrate_feature_extraction(market_data)
        
        # Determine regime based on forecasts, anomalies, and features
        prev_regime = self.current_regime
        
        # Start with unknown regime
        detected_regime = MarketRegime.UNKNOWN
        
        # Extract basic trend and volatility information
        trend = features.get("trend", 0)
        volatility = features.get("volatility", 0)
        trend_strength = features.get("trend_strength", 0)
        
        # Detect regime based on trend, volatility, and anomalies
        if anomalies.get("is_anomaly", False) and anomalies.get("anomaly_score", 0) > 0.7:
            # High anomaly score could indicate breakout or reversal
            if trend > 0:
                detected_regime = MarketRegime.BREAKOUT
            else:
                detected_regime = MarketRegime.REVERSAL
        else:
            # Normal regime detection based on trend and volatility
            if trend_strength > 0.6:
                if trend > 0:
                    detected_regime = MarketRegime.TRENDING_UP
                else:
                    detected_regime = MarketRegime.TRENDING_DOWN
            else:
                if volatility > 0.5:
                    detected_regime = MarketRegime.RANGING_HIGH_VOL
                else:
                    detected_regime = MarketRegime.RANGING_LOW_VOL
        # Use anomaly detection and feature extraction to identify regime
        anomalies = self.integrate_anomaly_detection(market_data)
        features = self.integrate_feature_extraction(market_data)
        
        # Extract key metrics for regime detection
        volatility = features.get("features", {}).get("volatility", 0.0)
        trend_strength = features.get("features", {}).get("trend_strength", 0.0)
        momentum = features.get("features", {}).get("momentum", 0.0)
        has_anomaly = anomalies.get("detected", False)
        
        # Determine regime based on extracted features
        if has_anomaly and volatility > 0.15:
            # High volatility with anomalies suggests breakout or reversal
            regime = MarketRegime.BREAKOUT if momentum > 0 else MarketRegime.REVERSAL
        elif trend_strength > 0.7:
            # Strong trend detected
            regime = MarketRegime.TRENDING_UP if momentum > 0 else MarketRegime.TRENDING_DOWN
        else:
            # Ranging market
            regime = MarketRegime.RANGING_HIGH_VOL if volatility > 0.1 else MarketRegime.RANGING_LOW_VOL
        
        # Check for regime change
        if self.current_regime != regime:
            logger.info(f"Market regime changed from {self.current_regime} to {regime}")
            self.metrics["regime_changes"] += 1
            self.current_regime = regime
            
        # Store in history
        self.regime_history.append({
            "timestamp": datetime.now(),
            "regime": regime,
            "volatility": volatility,
            "trend_strength": trend_strength
        })
        
        # Update metrics if regime changed
        if detected_regime != prev_regime:
            self.metrics["regime_changes"] += 1
            
        # Update current regime and history
        self.current_regime = detected_regime
        self.regime_history.append(detected_regime)
        
        return detected_regime
    
    

    def _state_to_index(self, state: Dict[str, Any]) -> int:
        """Convert state dictionary to state index for RL"""
        # This is a simplified conversion - in production you'd want something more sophisticated
        try:
            # Extract key values
            trend = state.get("trend", 0)
            volatility = state.get("volatility", 0)
            volume = state.get("volume", 0)
            momentum = state.get("momentum", 0)
            regime_id = self._get_regime_id(state.get("regime", MarketRegime.UNKNOWN))
            
            # Combine into a hash
            hash_val = int(trend * 1000) + int(volatility * 100) + int(volume * 10) + regime_id
            # Map to state range (0-199 for default 200 states)
            return hash_val % 200
            
        except Exception as e:
            logger.error(f"Error converting state to index: {e}")
            return 0
    
    def _get_regime_id(self, regime: MarketRegime) -> int:
        """Convert market regime to numeric ID"""
        regime_map = {
            MarketRegime.TRENDING_UP: 1,
            MarketRegime.TRENDING_DOWN: 2,
            MarketRegime.RANGING_LOW_VOL: 3,
            MarketRegime.RANGING_HIGH_VOL: 4,
            MarketRegime.BREAKOUT: 5,
            MarketRegime.REVERSAL: 6,
            MarketRegime.UNKNOWN: 0
        }
        return regime_map.get(regime, 0)
    
    
        
    def analyze_market_regime(self, market_data: Dict[str, Any]):
        """
        Analyze market data to identify the current market regime
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Identified market regime
        """
        try:
            # Extract relevant features for regime detection
            prices = market_data.get("close", [])
            volumes = market_data.get("volume", [])
            volatility = market_data.get("volatility", None)
            trend_strength = market_data.get("trend_strength", None)
            
            if len(prices) < 2:
                logger.warning("Insufficient price data for regime detection")
                return MarketRegime.UNKNOWN
            
            # Detect trend using last N prices
            price_trend = np.mean(np.diff(prices[-20:]))
            
            # Calculate recent volatility if not provided
            if volatility is None and len(prices) > 20:
                volatility = np.std(np.diff(prices[-20:]) / prices[-21:-1])
            else:
                volatility = 0.01  # Default low volatility
            
            # Detect volume pattern
            volume_trend = np.mean(np.diff(volumes[-20:])) if len(volumes) >= 20 else 0
            volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
            
            # Determine market regime
            if price_trend > 0.001:  # Positive trend
                if volume_ratio > 1.2:  # Increasing volume
                    regime = MarketRegime.TRENDING_UP
                elif volatility > 0.02:  # High volatility
                    regime = MarketRegime.BREAKOUT
                else:
                    regime = MarketRegime.TRENDING_UP
            elif price_trend < -0.001:  # Negative trend
                if volume_ratio > 1.2:  # Increasing volume
                    regime = MarketRegime.TRENDING_DOWN
                elif volatility > 0.02:  # High volatility
                    regime = MarketRegime.REVERSAL
                else:
                    regime = MarketRegime.TRENDING_DOWN
            else:  # Sideways
                if volatility > 0.015:
                    regime = MarketRegime.RANGING_HIGH_VOL
                else:
                    regime = MarketRegime.RANGING_LOW_VOL
            
            # Track regime changes
            if self.current_regime != regime:
                self.metrics["regime_changes"] += 1
                self.regime_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "previous_regime": self.current_regime.name,
                    "new_regime": regime.name
                })
                self.current_regime = regime
            
            return regime
        except Exception as e:
            logger.error(f"Error in market regime analysis: {e}")
            return MarketRegime.UNKNOWN

    def enhanced_decision(self, market_data: Dict[str, Any]):
        """
        Make enhanced trading decision using all integrated components
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Trading decision with confidence and reasoning
        """
        # Skip if core agent is not initialized
        if "quantum_amos_agent" not in self._initialized_components:
            logger.warning("Quantum AMOS agent not initialized, cannot make decision")
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": "System not initialized"}
        
        try:
            # Initialize clean_market_data and reasoning_components
            clean_market_data = {} 
            reasoning_components = []
            expected_outcome = 0.0 # Default
            probability = 0.5    # Default
            reasoning_source = "Default" # To track where outcome/prob came from

            # Populate clean_market_data with basic numeric market_data first
            if isinstance(market_data, dict):
                for key, value in market_data.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool): # bools are instances of int
                        clean_market_data[key] = float(value)
            
            # 1. Analyze market regime (uses basic features, should be safe)
            # Ensure market_data for analyze_market_regime has what it needs (e.g. 'close', 'volume')
            # If they come from ohlcv, ensure ohlcv is processed into market_data first
            if "ohlcv" in market_data and isinstance(market_data["ohlcv"], np.ndarray) and market_data["ohlcv"].shape[0] > 0:
                if "close" not in market_data and market_data["ohlcv"].shape[1] > 3:
                    market_data["close"] = market_data["ohlcv"][:, 3].tolist() # Pass list for consistency with example data
                if "volume" not in market_data and market_data["ohlcv"].shape[1] > 4:
                    market_data["volume"] = market_data["ohlcv"][:, 4].tolist()

            regime = self.analyze_market_regime(market_data) # Pass original market_data
            
            # 2. Get market forecasts
            forecast = self.integrate_forecasting(market_data) # Pass original market_data
            
            # 3. Detect anomalies
            anomalies = self.integrate_anomaly_detection(market_data) # Pass original market_data
            
            # 4. Extract enhanced features
            # Create a copy for feature extraction if it modifies input, or ensure it doesn't
            feature_input_data = dict(market_data) if isinstance(market_data, dict) else {}
            if "ohlcv" in feature_input_data and isinstance(feature_input_data["ohlcv"], np.ndarray) and feature_input_data["ohlcv"].shape[0] > 0:
                 if "close_prices" not in feature_input_data and feature_input_data["ohlcv"].shape[1] > 3:
                    feature_input_data["close_prices"] = feature_input_data["ohlcv"][:, 3].tolist()

            features = self.integrate_feature_extraction(feature_input_data) # Pass potentially augmented market_data

            # Start building enhanced_market_data for agent.decide()
            # It's distinct from clean_market_data at this stage
            enhanced_market_data = dict(market_data) if isinstance(market_data, dict) else {}
            enhanced_market_data["regime"] = regime.name if hasattr(regime, "name") else str(regime)
            enhanced_market_data["forecast"] = forecast if isinstance(forecast, dict) else {}
            enhanced_market_data["anomalies"] = anomalies if isinstance(anomalies, dict) else {}
            # Ensure 'enhanced_features' key from 'features' is used
            enhanced_market_data["enhanced_features"] = features.get("enhanced_features", features.get("temporal_features", {})) if isinstance(features, dict) else {}


            # 5. Combine all information for decision (This uses enhanced_market_data to derive signals)
            # Prepare clean_market_data for self.agent.decide()
            # At this point, clean_market_data may only have initial numeric values from market_data.
            # We need to add more specific features.
            
            # Populate 'close_prices' in enhanced_market_data if available from ohlcv
            if "ohlcv" in enhanced_market_data and isinstance(enhanced_market_data["ohlcv"], np.ndarray) and enhanced_market_data["ohlcv"].shape[0] > 0:
                if "close_prices" not in enhanced_market_data or not enhanced_market_data.get("close_prices"): # check if empty or not present
                    if enhanced_market_data["ohlcv"].shape[1] > 3: # Check if 'close' column exists
                        enhanced_market_data["close_prices"] = enhanced_market_data["ohlcv"][:, 3].tolist()
                        logger.info(f"Populated 'close_prices' from 'ohlcv' data for strategic analysis in enhanced_decision.")
                    else:
                         logger.warning("'ohlcv' data is missing 'close' column, cannot populate 'close_prices'.")
                         enhanced_market_data["close_prices"] = [] # Ensure it's an empty list
            elif "close_prices" not in enhanced_market_data:
                 enhanced_market_data["close_prices"] = [] # Ensure it's an empty list if not from ohlcv

            # 6. Perform strategic quantum analysis if QERC is available
            strategic_insights = None
            
            if "qerc" in self._initialized_components:
                logger.info("Using quantum indicators for strategic analysis")
                # Pass enhanced_market_data which should now contain 'close_prices' if available
                strategic_insights = self.strategic_quantum_analysis(enhanced_market_data) 
                
                if strategic_insights and strategic_insights.get("available", False):
                    regime_name = strategic_insights.get("regime", "UNKNOWN")
                    # clean_market_data["quantum_regime"] = regime_name # Not a direct numeric input for AMOS agent
                    reasoning_components.append(f"Strategic Quantum Regime: {regime_name}")
                    
                    market_insights = strategic_insights.get("insights", {})
                    if market_insights:
                        insights_observations = market_insights.get('key_observations', [])
                        if insights_observations:
                             reasoning_components.extend(insights_observations)
                        
                        # Add specific numeric insights to clean_market_data for the agent
                        if "market_stability" in market_insights: clean_market_data["market_stability"] = float(market_insights["market_stability"])
                        if "trend_strength" in market_insights: clean_market_data["trend_strength"] = float(market_insights["trend_strength"])
                        if "market_complexity" in market_insights: clean_market_data["complexity"] = float(market_insights["market_complexity"]) # Renamed from 'complexity' to 'market_complexity' in prev plan, keeping 'complexity' if it's an existing factor

                    signal_data = strategic_insights.get("signals", {}).get("entry", {}) # Corrected from 'signals' to 'signal_data'
                    if signal_data.get("long") or signal_data.get("short"):
                        expected_outcome = signal_data.get("strength", 0.0) if signal_data.get("long") else -signal_data.get("strength", 0.0)
                        probability = signal_data.get("probability", 0.5)
                        reasoning_source = "Strategic Quantum Indicators"
                        logger.info(f"Using outcome/prob from Strategic Quantum Indicators: outcome={expected_outcome}, prob={probability}")
                        reasoning_components.append(f"Strategic Signal: {'Long' if signal_data.get('long') else 'Short'} (Str: {signal_data.get('strength',0.0):.2f}, Prob: {probability:.2f})")
                    
                    # Add recommended position size to clean_market_data if it exists and is numeric
                    recommended_pos_size = strategic_insights.get("signals", {}).get("recommended_position_size")
                    if recommended_pos_size is not None:
                        clean_market_data["position_size_recommendation"] = float(recommended_pos_size)

            # Fallback to QAR forecast if strategic insights didn't provide outcome/prob
            if reasoning_source == "Default" and forecast and forecast.get("forecast_available", False):
                expected_outcome = forecast.get("expected_outcome", 0.0)
                probability = forecast.get("probability", 0.5)
                reasoning_source = "Quantum Annealing Regression"
                logger.info(f"Using outcome/prob from QAR: outcome={expected_outcome}, prob={probability}")
            
            if reasoning_source == "Default":
                 logger.info(f"Using default outcome/prob: outcome={expected_outcome}, prob={probability}")


            # Populate clean_market_data with features from QERC if available
            # This assumes 'features' dict from QERC process method contains keys that AMOS agent expects
            if isinstance(features, dict):
                # If QERC produced 'temporal_features' or 'enhanced_features'
                qerc_features_dict = features.get("temporal_features", features.get("enhanced_features", {}))
                if isinstance(qerc_features_dict, dict):
                    for key, value in qerc_features_dict.items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            clean_market_data[key] = float(value)
            
            # Ensure standard factors are present in clean_market_data for the agent
            standard_factors = [
                "trend", "volatility", "momentum", "sentiment", 
                "liquidity", "correlation", "cycle", "anomaly"
            ]
            
            for factor in standard_factors:
                if factor not in clean_market_data:
                    # Use meaningful defaults based on available data
                    if factor == "trend":
                        if "close_prices" in enhanced_market_data and enhanced_market_data["close_prices"] and len(enhanced_market_data["close_prices"]) > 5:
                            prices_for_trend = enhanced_market_data["close_prices"][-5:]
                            trend_val = (prices_for_trend[-1] - prices_for_trend[0]) / prices_for_trend[0] if prices_for_trend[0] > 0 else 0.0
                            clean_market_data[factor] = min(max(trend_val, -1.0), 1.0)
                        else:
                            clean_market_data[factor] = 0.0
                    elif factor == "sentiment":
                        regime_name_for_sentiment = enhanced_market_data.get("regime", "UNKNOWN")
                        regime_sentiment_map = {
                            "TRENDING_UP": 0.7, "TRENDING_DOWN": -0.7,
                            "RANGING_LOW_VOL": 0.2, "RANGING_HIGH_VOL": -0.2,
                            "BREAKOUT": 0.5, "REVERSAL": -0.3, "UNKNOWN": 0.0
                        }
                        clean_market_data[factor] = regime_sentiment_map.get(regime_name_for_sentiment, 0.0)
                    else:
                        clean_market_data[factor] = 0.0 # Neutral default for other factors
            
            # Make sure common keys like 'price' and 'volume' are present if AMOS expects them.
            if "price" not in clean_market_data and "close_prices" in enhanced_market_data and enhanced_market_data["close_prices"]:
                clean_market_data["price"] = float(enhanced_market_data["close_prices"][-1])
            elif "price" not in clean_market_data:
                 clean_market_data["price"] = 0.0

            if "volume" not in clean_market_data and "volume_data" in enhanced_market_data and enhanced_market_data["volume_data"]: # Assuming 'volume_data' might exist
                clean_market_data["volume"] = float(enhanced_market_data["volume_data"][-1])
            elif "volume" not in clean_market_data:
                clean_market_data["volume"] = 0.0


            # Call agent.decide with the fully prepared clean_market_data
            amos_output = self.agent.decide(
                clean_market_data, 
                expected_outcome=expected_outcome,
                probability=probability
            )
            
            decision_enum = amos_output.get("decision", "HOLD") # Default to HOLD if key missing
            amos_confidence = amos_output.get("confidence", 0.5)
            agent_reasoning = amos_output.get("reasoning", "")
            if agent_reasoning: # Add agent's own reasoning only if it's not empty
                reasoning_components.insert(0, agent_reasoning) # Put agent's reasoning first

            # Construct final decision
            final_decision_str = decision_enum.name if hasattr(decision_enum, "name") else str(decision_enum)
            final_confidence = amos_confidence

            if strategic_insights and strategic_insights.get("available", False):
                signal_data_final = strategic_insights.get("signals", {}).get("entry", {}) # Renamed to avoid conflict
                signal_prob_final = signal_data_final.get("probability", 0.0) # Renamed to avoid conflict
                signal_strength_final = signal_data_final.get("strength", 0.0) # Renamed to avoid conflict

                if signal_data_final.get("long") or signal_data_final.get("short"):
                    strategic_signal_confidence = signal_prob_final * signal_strength_final
                    final_confidence = (amos_confidence * 0.4) + (strategic_signal_confidence * 0.6)
                    reasoning_components.append(f"Strategic Signal Conf: {strategic_signal_confidence:.2f}")
            elif forecast and forecast.get("forecast_available", False):
                 forecast_prob = forecast.get('probability',0.5)
                 final_confidence = (amos_confidence + forecast_prob) / 2.0
                 reasoning_components.append(f"Forecast Outcome: {forecast.get('expected_outcome',0.0):.2f}, Prob: {forecast_prob:.2f}")
            
            final_reasoning = " | ".join(reasoning_components)
            if not final_reasoning:
                 final_reasoning = f"Default decision based on agent output."

            decision = {
                "decision": final_decision_str,
                "confidence": round(final_confidence, 3),
                "reasoning": final_reasoning
            }
            
            # 7. Update metrics
            self.metrics["decisions_count"] += 1
            
            # 8. Store decision in history
            self.decision_history.append({
                "timestamp": datetime.now().isoformat(),
                "decision": dict(decision) # Store a copy
            })
            
            return decision
        except Exception as e:
            logger.error(f"Error in enhanced decision making: {str(e)}", exc_info=True)
            return {"decision": "HOLD", "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
        
     
    def rl_enhanced_decide(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a trading decision enhanced by reinforcement learning
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Enhanced decision dictionary
        """
        if "qstar_river" not in self._initialized_components or "quantum_amos_agent" not in self._initialized_components:
            logger.warning("Required components not available for RL-enhanced decision")
            if "quantum_amos_agent" in self._initialized_components:
                # Fall back to standard AMOS decision
                return self.agent.decide(market_data)
            else:
                return {"decision": "HOLD", "confidence": 0.0, "enhanced": False}
                
        try:
            # 1. Extract state from market data
            features = self._extract_features_from_market_data(market_data)
            
            # 2. Detect current market regime
            regime = self.detect_market_regime(market_data)
            
            # 3. Get forecast and anomaly detection results
            forecast = self.integrate_forecasting(market_data)
            anomalies = self.integrate_anomaly_detection(market_data)
            
            # 4. Extract enhanced features
            enhanced_features = self.integrate_feature_extraction(market_data)
            
            # 5. Create state for RL agent
            state = {
                "features": features,
                "regime": self._get_regime_id(regime),
                "forecast": forecast.get("expected_outcome", 0.0),
                "anomaly": int(anomalies.get("detected", False)),
                "enhanced_features": enhanced_features.get("enhanced_features", {})
            }
            
            # 6. Get action from RL agent
            # Check attribute name for qstar agent - could be qstar_agent, qstar_river or qstar_learning_agent
            if hasattr(self, 'qstar_agent'):
                rl_action = self.qstar_agent.get_action(state)
            elif hasattr(self, 'qstar_river'):
                rl_action = self.qstar_river.get_action(state)
            elif hasattr(self, 'qstar_learning_agent'):
                rl_action = self.qstar_learning_agent.get_action(state)
            else:
                # Create a default action if RL agent not available
                rl_action = {"action": 1, "action_name": "HOLD", "confidence": 0.5}
            
            # Get standard AMOS decision - passing the required parameters
            expected_outcome = forecast.get("expected_outcome", 0.0)
            probability = forecast.get("probability", 0.5)
            amos_decision = self.agent.decide(
                market_data,
                expected_outcome=expected_outcome,
                probability=probability
            )
            
            # Combine decisions (with RL having higher weight in training)
            if self._is_trained:
                # RL system is trained, give it more weight
                rl_weight = 0.7
                amos_weight = 0.3
            else:
                # RL system is not trained, give AMOS more weight
                rl_weight = 0.3
                amos_weight = 0.7
                
            # Determine final decision based on weighted combination
            combined_confidence = (rl_weight * rl_action.get("confidence", 0.5) + 
                                  amos_weight * amos_decision.get("confidence", 0.5))
                                  
            # Map RL action to decision string
            rl_decision_type = rl_action.get("action_name", "HOLD")
            
            # If both agree, use that decision with higher confidence
            if rl_decision_type == amos_decision.get("decision", "HOLD"):
                final_decision = rl_decision_type
                final_confidence = max(rl_action.get("confidence", 0.5), amos_decision.get("confidence", 0.5))
            else:
                # If they disagree, use weighted decision
                final_decision = rl_decision_type if rl_weight > amos_weight else amos_decision.get("decision", "HOLD")
                final_confidence = combined_confidence
                
            # Create enhanced decision
            enhanced_decision = {
                "decision": final_decision,
                "confidence": final_confidence,
                "enhanced": True,
                "rl_contribution": rl_weight,
                "amos_contribution": amos_weight,
                "regime": regime.name,
                "forecast": forecast.get("expected_outcome", 0.0),
                "has_anomaly": anomalies.get("detected", False)
            }
            
            # Update decision history
            self.decision_history.append({
                "timestamp": datetime.now(),
                "decision": enhanced_decision,
                "regime": regime
            })
            
            # Update metrics
            self.metrics["decisions_count"] += 1
            
            return enhanced_decision
            
        except Exception as e:
            logger.error(f"Error in RL-enhanced decision making: {e}")
            # Fallback to standard decision
            if "quantum_amos_agent" in self._initialized_components:
                # Extract forecast info needed for the quantum_amos_agent.decide() call
                try:
                    forecast_info = self.integrate_forecasting(market_data)
                    expected_outcome = forecast_info.get("expected_outcome", 0.0)
                    probability = forecast_info.get("probability", 0.5)
                    return self.agent.decide(
                        market_data,
                        expected_outcome=expected_outcome,
                        probability=probability
                    )
                except Exception as e2:
                    logger.error(f"Error in fallback decision: {e2}")
                    return {"decision": "HOLD", "confidence": 0.0, "enhanced": False, "error": f"{e}, {e2}"}
            else:
                return {"decision": "HOLD", "confidence": 0.0, "enhanced": False, "error": str(e)}
    
    def _combine_decisions(self, standard_decision: Dict[str, Any], rl_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine standard decision with RL recommendation
        
        Args:
            standard_decision: Decision from Quantum AMOS
            rl_action: Decision from QStar-River
            
        Returns:
            Combined decision
        """
        # Start with standard decision
        combined = standard_decision.copy()
        
        # Add RL information
        combined["rl_action"] = rl_action.get("action", "NOOP")
        combined["rl_confidence"] = rl_action.get("confidence", 0.0)
        combined["rl_weight"] = 0.4  # Weight of RL contribution
        combined["quantum_weight"] = 0.6  # Weight of quantum model contribution
        
        # Combine confidences
        standard_confidence = standard_decision.get("confidence", 0.5)
        rl_confidence = rl_action.get("confidence", 0.5)
        
        combined["combined_confidence"] = (
            combined["quantum_weight"] * standard_confidence +
            combined["rl_weight"] * rl_confidence
        )
        
        # Determine final decision based on weighted combination
        if standard_decision.get("decision") == rl_action.get("action"):
            # Both agree, use standard decision
            combined["final_decision"] = standard_decision.get("decision")
        else:
            # Disagreement, use the one with higher weighted confidence
            if standard_confidence * combined["quantum_weight"] > rl_confidence * combined["rl_weight"]:
                combined["final_decision"] = standard_decision.get("decision")
            else:
                combined["final_decision"] = rl_action.get("action")
        
        return combined
    
    def _merge_decisions(self, amos_decision: Dict[str, Any], rl_action: int) -> Dict[str, Any]:
        """
        Merge AMOS decision with RL action
        
        Args:
            amos_decision: Decision from Quantum AMOS
            rl_action: Action from RL agent
            
        Returns:
            Merged decision
        """
        # Map RL action to decision
        action_map = {
            0: "buy",      # Buy
            1: "sell",     # Sell
            2: "hold",     # Hold
            3: "reduce",   # Reduce position
            4: "increase", # Increase position
            5: "hedge",    # Apply hedging
            6: "exit"      # Exit all positions
        }
        
        rl_decision = action_map.get(rl_action, "hold")
        
        # Default to AMOS decision
        final_decision = amos_decision.copy()
        
        # If RL confidence is high (exploration rate is low), use RL decision
        if hasattr(self.qstar_river.agent, "exploration_rate"):
            rl_confidence = 1.0 - self.qstar_river.agent.exploration_rate
            if rl_confidence > 0.7:
                final_decision["action"] = rl_decision
                final_decision["rl_influence"] = rl_confidence
                
        return final_decision
    
    def _extract_basic_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic features from market data"""
        features = {}
        
        try:
            # Extract OHLCV
            ohlcv = self._extract_ohlcv(market_data)
            
            # Simple features
            if len(ohlcv) > 0:
                # Calculate basic indicators
                close = ohlcv[:, 3]  # Close price column
                
                # Trend (simple moving average difference)
                if len(close) >= 20:
                    sma5 = np.mean(close[-5:])
                    sma20 = np.mean(close[-20:])
                    features["trend"] = (sma5 / sma20) - 1
                    
                # Volatility (standard deviation)
                if len(close) >= 20:
                    features["volatility"] = np.std(close[-20:]) / np.mean(close[-20:])
                    
                # Volume
                if ohlcv.shape[1] > 4:  # If volume exists
                    volume = ohlcv[:, 4]  # Volume column
                    if len(volume) >= 20:
                        features["volume"] = volume[-1] / np.mean(volume[-20:])
                        
                # Momentum (RSI-like)
                if len(close) >= 15:
                    diff = np.diff(close[-15:])
                    if len(diff) > 0:
                        pos_sum = np.sum(diff[diff > 0])
                        neg_sum = np.abs(np.sum(diff[diff < 0]))
                        if neg_sum > 0:
                            rs = pos_sum / neg_sum
                            features["momentum"] = rs / (1 + rs)
                        else:
                            features["momentum"] = 1.0
                    else:
                        features["momentum"] = 0.5
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            return {"trend": 0, "volatility": 0, "volume": 0, "momentum": 0.5}
    
    def _extract_ohlcv(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract OHLCV data from market data"""
        try:
            if "ohlcv" in market_data:
                return market_data["ohlcv"]
                
            if "dataframe" in market_data:
                df = market_data["dataframe"]
                return df[["open", "high", "low", "close", "volume"]].values
                
            # Try to build from individual arrays
            ohlcv = []
            if all(k in market_data for k in ["open", "high", "low", "close"]):
                ohlcv = np.column_stack([
                    market_data["open"],
                    market_data["high"],
                    market_data["low"],
                    market_data["close"],
                ])
                
                # Add volume if available
                if "volume" in market_data:
                    ohlcv = np.column_stack([ohlcv, market_data["volume"]])
                    
            return np.array(ohlcv)      
        except Exception as e:
            logger.error(f"Error receiving data")
            return df
        

    def train_reinforcement_learning(self, environment, episodes: int = 100):
        """
        Train the reinforcement learning component using the provided environment
        
        Args:
            environment: Trading environment for training
            episodes: Number of training episodes
            
        Returns:
            Training metrics
        """
        if "qstar_river" not in self._initialized_components:
            logger.warning("QStar-River not initialized, skipping training")
            return {"trained": False, "reason": "QStar-River not initialized"}
        
        try:
            # Train the agent
            converged, episodes_completed = self.qstar_river.agent.train(environment)
            
            # Update metrics
            self.metrics["rl_episodes"] += episodes_completed
            
            # Mark system as trained
            self._is_trained = True
            
            # Return training results
            return {
                "trained": True,
                "converged": converged,
                "episodes_completed": episodes_completed,
                "metrics": self.qstar_river.agent.metrics.get_summary()
            }
        except Exception as e:
            logger.error(f"Error training reinforcement learning: {e}")
            return {"trained": False, "reason": str(e)}
    
    def provide_feedback(self, decision: Dict[str, Any], actual_outcome: float, reward: float) -> None:
        """
        Provide feedback to the reinforcement learning system
        
        Args:
            decision: Decision dictionary returned by rl_enhanced_decide
            actual_outcome: Actual outcome value (e.g., profit/loss)
            reward: Reward value for reinforcement learning
        """
        if "qstar_river" not in self._initialized_components:
            logger.warning("QStar-River system not available for feedback")
            return
            
        try:
            # Record reward
            self.reward_history.append({
                "timestamp": datetime.now(),
                "decision": decision.get("decision", "HOLD"),
                "expected_outcome": decision.get("forecast", 0.0),
                "actual_outcome": actual_outcome,
                "reward": reward
            })
            
            # Update RL system with feedback
            state = decision.get("state_index", None)
            action = decision.get("action_index", None)
            
            if state is not None and action is not None:
                # Update Q-values directly
                self.qstar_river.agent.learn(state, action, reward, state, False)
                
            # If we have accumulated enough feedback, consider retraining
            if len(self.reward_history) >= 50 and "qstar_river" in self._initialized_components:
                avg_reward = np.mean([r.get("reward", 0.0) for r in list(self.reward_history)[-50:]])
                logger.info(f"Average reward over last 50 decisions: {avg_reward:.4f}")
                
        except Exception as e:
            logger.error(f"Error providing feedback to RL system: {e}")
    

    def save_system_state(self, filepath: str = None):
        """
        Save the current system state to a file
        
        Args:
            filepath: Path to save file (default: models/quantum_amos_system_<timestamp>.json)
            
        Returns:
            Path to saved file
        """
        try:
            # Create default filepath if not provided
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.config.model_dir, f"quantum_amos_system_{timestamp}.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Create state dictionary
            state = {
                "metrics": self.metrics,
                "components_status": self._components_status,
                "initialized_components": list(self._initialized_components),
                "current_regime": self.current_regime.name,
                "is_trained": self._is_trained,
                "config": {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}
            }
            
            # Save state to file
            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"System state saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            return None
    
    def load_system_state(self, filepath: str):
        """
        Load system state from a file
        
        Args:
            filepath: Path to load file
            
        Returns:
            Success status
        """
        try:
            # Load state from file
            with open(filepath, "r") as f:
                state = json.load(f)
            
            # Update metrics
            self.metrics = state.get("metrics", self.metrics)
            
            # Update components status
            self._components_status = state.get("components_status", self._components_status)
            
            # Update initialized components
            self._initialized_components = set(state.get("initialized_components", []))
            
            # Update current regime
            regime_name = state.get("current_regime", MarketRegime.UNKNOWN.name)
            self.current_regime = MarketRegime[regime_name]
            
            # Update trained status
            self._is_trained = state.get("is_trained", self._is_trained)
            
            logger.info(f"System state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading system state: {e}")
            return False
    
    def get_system_status(self):
        """
        Get current system status summary
        
        Returns:
            Dictionary with system status
        """
        return {
            "initialized": self._is_initialized,
            "trained": self._is_trained,
            "components": self._components_status,
            "metrics": self.metrics,
            "current_regime": self.current_regime.name
        }

