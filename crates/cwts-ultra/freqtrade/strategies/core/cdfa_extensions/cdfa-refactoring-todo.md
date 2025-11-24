# CDFA Suite Comprehensive Refactoring TODO

## Phase 1: Immediate Stability Improvements

### 1. Standardize Detector Interfaces
- Create a base `BaseDetector` abstract class in `detectors/base_detector.py` with standard methods:
  ```python
  from abc import ABC, abstractmethod
  import pandas as pd
  import numpy as np
  import logging
  from typing import Dict, Any, Union, Optional

  class BaseDetector(ABC):
      """Base abstract class for all detectors"""
      
      def __init__(self, log_level: int = logging.INFO):
          """Initialize the detector"""
          self.logger = logging.getLogger(f"{self.__class__.__name__}")
          self.logger.setLevel(log_level)
      
      @abstractmethod
      def detect(self, data: pd.DataFrame) -> pd.Series:
          """
          Detect patterns or anomalies in the given data
          
          Args:
              data: Input DataFrame with OHLCV data
              
          Returns:
              Binary indicator series (1 for detection, 0 for no detection)
          """
          pass
          
      @abstractmethod
      def calculate_probability(self, data: pd.DataFrame) -> pd.Series:
          """
          Calculate probability of pattern or anomaly
          
          Args:
              data: Input DataFrame with OHLCV data
              
          Returns:
              Probability series (0.0-1.0)
          """
          pass
          
      def visualize(self, data: pd.DataFrame, detections: pd.Series, 
                   title: Optional[str] = None) -> Dict[str, Any]:
          """
          Visualize detected patterns
          
          Args:
              data: Input DataFrame with OHLCV data
              detections: Detection results
              title: Plot title
              
          Returns:
              Visualization configuration
          """
          # Default implementation returns empty dict
          return {}
  ```

- Update all detectors to inherit from BaseDetector:
  - BlackSwanDetector
  - WhaleDetector  
  - FibonacciPatternDetector
  - PatternRecognizer
  - AccumulationDetector
  - DistributionDetector
  - BubbleDetector
  - ConfluenceAreaDetector

### 2. Fix Remaining Numba Compatibility Issues
- Audit all Numba-decorated functions and fix issues similar to `_detect_fractality_numba`:
  - Check for Python list usage and replace with Numba arrays
  - Ensure all NumPy functions used are compatible
  - Add manual implementations for unsupported NumPy functions

- Add a Numba utility module `utils/numba_utils.py`:
  ```python
  import numpy as np
  import numba as nb
  from numba import njit, float64, int64, boolean
  from numba.typed import List, Dict
  
  @njit(float64[:](float64[:], float64), cache=True)
  def numba_rolling_mean(arr, window):
      """Calculate rolling mean using Numba"""
      result = np.zeros_like(arr)
      for i in range(len(arr)):
          if i < window - 1:
              window_sum = 0
              count = 0
              for j in range(max(0, i - window + 1), i + 1):
                  window_sum += arr[j]
                  count += 1
              result[i] = window_sum / count if count > 0 else np.nan
          else:
              window_sum = 0
              for j in range(i - window + 1, i + 1):
                  window_sum += arr[j]
              result[i] = window_sum / window
      return result
  
  @njit(float64[:](float64[:], float64), cache=True)
  def numba_rolling_std(arr, window):
      """Calculate rolling standard deviation using Numba"""
      result = np.zeros_like(arr)
      means = numba_rolling_mean(arr, window)
      for i in range(len(arr)):
          if i < window - 1:
              window_sum_sq = 0
              count = 0
              for j in range(max(0, i - window + 1), i + 1):
                  window_sum_sq += (arr[j] - means[i]) ** 2
                  count += 1
              result[i] = np.sqrt(window_sum_sq / count) if count > 0 else np.nan
          else:
              window_sum_sq = 0
              for j in range(i - window + 1, i + 1):
                  window_sum_sq += (arr[j] - means[i]) ** 2
              result[i] = np.sqrt(window_sum_sq / window)
      return result
  ```

### 3. Add Basic Error Handling
- Create an error handling module `utils/error_handling.py`:
  ```python
  import logging
  import traceback
  import functools
  import time
  from typing import Any, Callable, Dict, Optional, TypeVar, cast

  logger = logging.getLogger("error_handling")

  T = TypeVar('T')

  class CDFAError(Exception):
      """Base class for all CDFA exceptions"""
      pass

  class HardwareError(CDFAError):
      """Hardware-related errors"""
      pass

  class DataError(CDFAError):
      """Data-related errors"""
      pass

  class ConfigurationError(CDFAError):
      """Configuration-related errors"""
      pass

  class ExternalServiceError(CDFAError):
      """External service errors (Redis, etc.)"""
      pass

  def safe_execute(default_return: Any = None, log_errors: bool = True,
                  num_retries: int = 0, retry_delay: float = 1.0) -> Callable:
      """
      Decorator for safely executing functions with error handling
      
      Args:
          default_return: Default value to return on error
          log_errors: Whether to log errors
          num_retries: Number of retry attempts
          retry_delay: Delay between retries (seconds)
          
      Returns:
          Decorated function
      """
      def decorator(func: Callable[..., T]) -> Callable[..., T]:
          @functools.wraps(func)
          def wrapper(*args: Any, **kwargs: Any) -> T:
              retries = 0
              while True:
                  try:
                      return func(*args, **kwargs)
                  except Exception as e:
                      if log_errors:
                          logger.error(f"Error in {func.__name__}: {e}")
                          logger.debug(traceback.format_exc())
                      
                      retries += 1
                      if retries <= num_retries:
                          logger.warning(f"Retrying {func.__name__} ({retries}/{num_retries})...")
                          time.sleep(retry_delay)
                      else:
                          return cast(T, default_return)
          return wrapper
      return decorator
  ```

- Apply error handling to critical paths in Redis and hardware acceleration:
  - Add `@safe_execute` decorator to Redis connection methods
  - Add error handling to GPU acceleration functions

### 4. Improve Core Documentation
- Add comprehensive docstrings to key components, focusing on:
  - Class and method purpose
  - Parameters and return values
  - Examples

## Phase 2: System Robustness Improvements

### 1. Component Lifecycle Management
- Create a base `Component` class in `core/component.py`:
  ```python
  from abc import ABC, abstractmethod
  import logging
  from typing import Dict, Any, List, Optional

  class Component(ABC):
      """Base class for all system components"""
      
      def __init__(self, name: str, log_level: int = logging.INFO):
          """Initialize the component"""
          self.name = name
          self.logger = logging.getLogger(f"{name}")
          self.logger.setLevel(log_level)
          self.initialized = False
          self.dependencies: List[Component] = []
          
      def add_dependency(self, component: 'Component') -> None:
          """Add a dependency component"""
          self.dependencies.append(component)
          
      @abstractmethod
      def initialize(self) -> bool:
          """Initialize the component"""
          return True
          
      @abstractmethod
      def shutdown(self) -> bool:
          """Shutdown the component"""
          return True
          
      def check_health(self) -> Dict[str, Any]:
          """Check component health"""
          return {"status": "healthy", "name": self.name}
  ```

- Update key components to inherit from Component:
  - HardwareAccelerator
  - RedisConnector
  - PulsarConnector
  - PADSReporter

### 2. Testing Framework
- Create a basic testing module `tests/test_utils.py`:
  ```python
  import numpy as np
  import pandas as pd
  from typing import Dict, Any, Optional, List, Tuple
  
  def generate_test_ohlcv(length: int = 100, trend: str = "random", 
                         volatility: float = 0.02) -> pd.DataFrame:
      """
      Generate test OHLCV data
      
      Args:
          length: Length of the DataFrame
          trend: Trend type ('up', 'down', 'random', 'cycle')
          volatility: Volatility level
          
      Returns:
          DataFrame with OHLCV data
      """
      np.random.seed(42)  # For reproducibility
      
      # Generate close prices based on trend
      if trend == "up":
          close = np.linspace(100, 200, length) + np.random.normal(0, volatility * 100, length)
      elif trend == "down":
          close = np.linspace(200, 100, length) + np.random.normal(0, volatility * 100, length)
      elif trend == "cycle":
          x = np.linspace(0, 4 * np.pi, length)
          close = 150 + 50 * np.sin(x) + np.random.normal(0, volatility * 100, length)
      else:  # random
          close = 100 + np.cumsum(np.random.normal(0, volatility * 10, length))
      
      # Generate other OHLCV data
      daily_volatility = close * volatility
      high = close + np.random.uniform(0.5, 1.5, length) * daily_volatility
      low = close - np.random.uniform(0.5, 1.5, length) * daily_volatility
      open_price = low + np.random.uniform(0, 1, length) * (high - low)
      volume = np.random.uniform(100000, 1000000, length)
      
      # Create DataFrame
      df = pd.DataFrame({
          'open': open_price,
          'high': high,
          'low': low,
          'close': close,
          'volume': volume
      })
      
      # Add datetime index
      df.index = pd.date_range(start='2023-01-01', periods=length, freq='D')
      
      return df
  
  def generate_test_patterns(df: pd.DataFrame, pattern_types: List[str] = ["up", "down", "flat"],
                           num_patterns: int = 5, pattern_length: int = 10) -> Dict[str, List[Tuple[int, int]]]:
      """
      Generate test patterns in data
      
      Args:
          df: DataFrame with OHLCV data
          pattern_types: Types of patterns to generate
          num_patterns: Number of patterns per type
          pattern_length: Length of each pattern
          
      Returns:
          Dictionary mapping pattern types to lists of (start, end) indices
      """
      result = {}
      for pattern_type in pattern_types:
          patterns = []
          for _ in range(num_patterns):
              start = np.random.randint(0, len(df) - pattern_length)
              end = start + pattern_length
              patterns.append((start, end))
          result[pattern_type] = patterns
      return result
  ```

### 3. Configuration System
- Create a unified configuration module `utils/config_manager.py`:
  ```python
  import yaml
  import json
  import os
  import logging
  from typing import Dict, Any, List, Optional, Union
  from dataclasses import dataclass, field, asdict
  
  logger = logging.getLogger("config_manager")
  
  @dataclass
  class ComponentConfig:
      """Base configuration for components"""
      name: str
      enabled: bool = True
      log_level: int = logging.INFO
      
  @dataclass
  class HardwareConfig(ComponentConfig):
      """Hardware acceleration configuration"""
      use_gpu: bool = True
      prefer_cuda: bool = True
      device: Optional[str] = None
      num_threads: int = 8
      
  @dataclass
  class RedisConfig(ComponentConfig):
      """Redis configuration"""
      host: str = "localhost"
      port: int = 6379
      db: int = 0
      password: str = ""
      socket_timeout: float = 5.0
      
  @dataclass
  class AnalysisConfig(ComponentConfig):
      """Analysis configuration"""
      default_window: int = 100
      default_lookback: int = 200
      default_correlation_method: str = "pearson"
      regime_detection_method: str = "wavelet"
      
  @dataclass
  class SystemConfig:
      """Overall system configuration"""
      hardware: HardwareConfig = field(default_factory=lambda: HardwareConfig(name="hardware"))
      redis: RedisConfig = field(default_factory=lambda: RedisConfig(name="redis"))
      analysis: AnalysisConfig = field(default_factory=lambda: AnalysisConfig(name="analysis"))
      
      @classmethod
      def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
          """Create from dictionary"""
          hardware_dict = config_dict.get("hardware", {})
          redis_dict = config_dict.get("redis", {})
          analysis_dict = config_dict.get("analysis", {})
          
          return cls(
              hardware=HardwareConfig(name="hardware", **hardware_dict),
              redis=RedisConfig(name="redis", **redis_dict),
              analysis=AnalysisConfig(name="analysis", **analysis_dict)
          )
          
      @classmethod
      def from_file(cls, file_path: str) -> 'SystemConfig':
          """Load configuration from file"""
          if not os.path.exists(file_path):
              logger.warning(f"Config file {file_path} not found, using defaults")
              return cls()
              
          with open(file_path, 'r') as f:
              if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                  config_dict = yaml.safe_load(f)
              elif file_path.endswith('.json'):
                  config_dict = json.load(f)
              else:
                  raise ValueError(f"Unsupported file format: {file_path}")
                  
          return cls.from_dict(config_dict)
          
      def to_dict(self) -> Dict[str, Any]:
          """Convert to dictionary"""
          return asdict(self)
          
      def save(self, file_path: str) -> None:
          """Save configuration to file"""
          config_dict = self.to_dict()
          
          with open(file_path, 'w') as f:
              if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                  yaml.dump(config_dict, f, default_flow_style=False)
              elif file_path.endswith('.json'):
                  json.dump(config_dict, f, indent=2)
              else:
                  raise ValueError(f"Unsupported file format: {file_path}")
  ```

## Phase 3: Architectural Improvements

### 1. Event-Driven Architecture
- Create an event bus module `core/event_bus.py`:
  ```python
  import logging
  import threading
  import queue
  import time
  from typing import Dict, Any, List, Callable, Optional, Set, Tuple
  from dataclasses import dataclass
  
  logger = logging.getLogger("event_bus")
  
  @dataclass
  class Event:
      """Event message"""
      type: str
      source: str
      data: Dict[str, Any]
      timestamp: float = None
      
      def __post_init__(self):
          if self.timestamp is None:
              self.timestamp = time.time()
  
  class EventBus:
      """Simple event bus implementation"""
      
      def __init__(self, max_queue_size: int = 1000):
          """Initialize the event bus"""
          self.subscribers: Dict[str, List[Callable[[Event], None]]] = {}
          self.queue = queue.Queue(maxsize=max_queue_size)
          self.running = False
          self.worker_thread = None
          
      def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
          """Subscribe to an event type"""
          if event_type not in self.subscribers:
              self.subscribers[event_type] = []
          self.subscribers[event_type].append(callback)
          
      def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
          """Unsubscribe from an event type"""
          if event_type in self.subscribers and callback in self.subscribers[event_type]:
              self.subscribers[event_type].remove(callback)
              
      def publish(self, event: Event) -> None:
          """Publish an event"""
          try:
              self.queue.put(event, block=False)
          except queue.Full:
              logger.warning(f"Event queue full, dropping event: {event.type}")
              
      def publish_sync(self, event: Event) -> None:
          """Publish an event synchronously (for testing/critical events)"""
          self._dispatch_event(event)
              
      def start(self) -> None:
          """Start the event processing thread"""
          if self.running:
              return
              
          self.running = True
          self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
          self.worker_thread.start()
          
      def stop(self) -> None:
          """Stop the event processing thread"""
          self.running = False
          if self.worker_thread:
              self.worker_thread.join(timeout=5.0)
              
      def _process_events(self) -> None:
          """Process events from the queue"""
          while self.running:
              try:
                  event = self.queue.get(block=True, timeout=0.1)
                  self._dispatch_event(event)
                  self.queue.task_done()
              except queue.Empty:
                  continue
              except Exception as e:
                  logger.error(f"Error processing event: {e}")
                  
      def _dispatch_event(self, event: Event) -> None:
          """Dispatch an event to subscribers"""
          if event.type in self.subscribers:
              for callback in self.subscribers[event.type]:
                  try:
                      callback(event)
                  except Exception as e:
                      logger.error(f"Error in event callback: {e}")
  ```

### 2. Dependency Injection
- Create a dependency container module `core/container.py`:
  ```python
  import inspect
  from typing import Dict, Any, Type, Optional, TypeVar, cast

  T = TypeVar('T')

  class Container:
      """Simple dependency injection container"""
      
      def __init__(self):
          """Initialize the container"""
          self.instances: Dict[str, Any] = {}
          self.factories: Dict[str, Any] = {}
          
      def register(self, name: str, instance: Any) -> None:
          """Register an instance"""
          self.instances[name] = instance
          
      def register_factory(self, name: str, factory: Any) -> None:
          """Register a factory function or class"""
          self.factories[name] = factory
          
      def get(self, name: str) -> Any:
          """Get an instance by name"""
          if name in self.instances:
              return self.instances[name]
              
          if name in self.factories:
              factory = self.factories[name]
              if inspect.isclass(factory):
                  # Create class instance using container for dependencies
                  dependencies = {}
                  for param_name, param in inspect.signature(factory.__init__).parameters.items():
                      if param_name != 'self' and param.name in self.instances:
                          dependencies[param.name] = self.instances[param.name]
                  instance = factory(**dependencies)
              else:
                  # Call factory function
                  instance = factory()
                  
              # Register the instance
              self.instances[name] = instance
              return instance
              
          raise KeyError(f"No dependency registered for '{name}'")
          
      def get_typed(self, name: str, cls: Type[T]) -> T:
          """Get a typed instance"""
          instance = self.get(name)
          if not isinstance(instance, cls):
              raise TypeError(f"Instance '{name}' is not of type {cls.__name__}")
          return cast(T, instance)
  ```

### 3. Hardware Optimization
- Create a hardware detection module `hardware/detection.py`:
  ```python
  import platform
  import os
  import logging
  from typing import Dict, Any, List, Optional
  
  import torch
  
  logger = logging.getLogger("hardware_detection")
  
  def detect_hardware() -> Dict[str, Any]:
      """
      Detect available hardware resources
      
      Returns:
          Dictionary with hardware information
      """
      result = {
          "platform": platform.system(),
          "platform_version": platform.version(),
          "architecture": platform.machine(),
          "processor": platform.processor(),
          "cpu_count": os.cpu_count() or 1,
          "python_version": platform.python_version(),
          "torch_version": torch.__version__,
          "cuda_available": torch.cuda.is_available(),
          "cuda_devices": [],
          "rocm_available": False,
          "mps_available": False
      }
      
      # CUDA devices
      if torch.cuda.is_available():
          result["cuda_device_count"] = torch.cuda.device_count()
          for i in range(torch.cuda.device_count()):
              device_info = {
                  "name": torch.cuda.get_device_name(i),
                  "capability": torch.cuda.get_device_capability(i),
                  "total_memory": torch.cuda.get_device_properties(i).total_memory
              }
              result["cuda_devices"].append(device_info)
      
      # ROCm support (AMD)
      if hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available'):
          result["rocm_available"] = torch.hip.is_available()
      
      # MPS support (Apple Silicon)
      if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
          result["mps_available"] = torch.backends.mps.is_available()
      
      return result
  
  def select_best_device(prefer_cuda: bool = True) -> str:
      """
      Select the best available device
      
      Args:
          prefer_cuda: Whether to prefer CUDA over other options
          
      Returns:
          Device name ('cuda', 'mps', 'cpu')
      """
      if prefer_cuda and torch.cuda.is_available():
          return "cuda"
      
      # Check for MPS (Apple Silicon)
      if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
          if torch.backends.mps.is_available():
              return "mps"
      
      # Check for ROCm (AMD)
      if hasattr(torch, 'hip') and hasattr(torch.hip, 'is_available'):
          if torch.hip.is_available():
              return "hip"
      
      # Fall back to CPU
      return "cpu"
  ```

## Additional Notes

1. Implementation Strategy:
   - Start with Phase 1 to ensure immediate stability
   - Implement Phase 2 components as the foundation for larger improvements
   - Add Phase 3 architectural components without disrupting existing functionality

2. Testing Strategy:
   - Add simple test cases for each modified component
   - Use the test utilities to verify behavior
   - Create integration tests to ensure components work together

3. Documentation:
   - Add README files for each module
   - Update docstrings with comprehensive examples
   - Create a high-level architecture document
