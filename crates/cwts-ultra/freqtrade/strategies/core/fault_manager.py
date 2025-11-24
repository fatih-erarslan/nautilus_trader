import os
import time
from datetime import datetime
import logging
from collections import deque
import threading
import pandas as pd
import itertools
import logging
import threading
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from enum import Enum, auto

from hardware_manager import (
    HardwareManager
)

from ml_layer import H2OCatBoostHybrid

from usp import UniversalSignalProcessor as signal_pipeline

logger = logging.getLogger(__name__)

class FaultSeverity(Enum):
    """Fault severity levels."""
    INFO = auto()      # Informational, no impact
    WARNING = auto()   # Minor impact, can continue
    ERROR = auto()     # Significant impact, needs attention
    CRITICAL = auto()  # Severe impact, immediate action required
    FATAL = auto()     # System cannot function

class FaultCategory(Enum):
    """Fault categories."""
    HARDWARE = auto()         # Hardware-related issues
    QUANTUM = auto()          # Quantum device issues
    MEMORY = auto()           # Memory-related issues
    COMPUTATION = auto()      # Computation errors
    NETWORK = auto()          # Network-related issues
    RESOURCE = auto()         # Resource exhaustion
    EXTERNAL = auto()         # External service issues
    INTERNAL = auto()         # Internal logic errors
    UNKNOWN = auto()          # Unknown issues


class QuantumCircuitContext:
    """
    Context manager for quantum circuit execution with hardware-specific
    optimizations for AMD RX 6800XT and GTX 1080
    """

    def __init__(self, qhal, circuit_func, params=None, circuit_id=None):
        self.qhal = qhal
        self.circuit_func = circuit_func
        self.params = params
        self.circuit_id = circuit_id or f"circuit_{id(circuit_func) % 10000}"
        self.result = None

        # Track hardware type for optimizations
        if hasattr(qhal, "hardware_manager"):
            self.gpu_type = qhal.hardware_manager.gpu_type
        else:
            self.gpu_type = None

    def __enter__(self):
        """Context entry with hardware-specific preparation"""
        # Optimize memory before execution based on GPU type
        if self.gpu_type == "amd":
            # For AMD RX 6800XT - can be more aggressive with memory
            if hasattr(self.qhal, "hardware_manager"):
                self.qhal.hardware_manager.clear_gpu_memory(force=False)
        elif self.gpu_type == "nvidia":
            # For GTX 1080 - need to be more careful with limited 8GB VRAM
            if hasattr(self.qhal, "hardware_manager"):
                # Check current memory usage before execution
                try:
                    import torch

                    if torch.cuda.is_available():
                        current_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
                        if current_usage > 6:  # >6GB is high for 8GB card
                            self.qhal.hardware_manager.clear_gpu_memory(force=True)
                except:
                    pass

        return self

    def execute(self):
        """Execute quantum circuit with hardware-specific optimizations"""
        # Maximum retries (adjust based on GPU)
        if self.gpu_type == "amd":
            max_retries = 4  # More retries for 16GB AMD card
        else:
            max_retries = 3  # Fewer retries for 8GB NVIDIA card

        # Execute with retry logic
        for attempt in range(1, max_retries + 1):
            try:
                # Create qnode with device
                import pennylane as qml

                # Handle device initialization if needed
                if self.qhal.device is None:
                    if not self.qhal.initialize_device():
                        return self._classical_fallback()

                # Create and execute qnode
                qnode = qml.QNode(self.circuit_func, self.qhal.device)

                if self.params is None:
                    self.result = qnode()
                else:
                    self.result = qnode(self.params)

                # Cache result if caching is enabled
                if (
                    hasattr(self.qhal, "enable_caching")
                    and self.qhal.enable_caching
                    and self.params is not None
                ):
                    if hasattr(self.qhal, "_add_to_cache"):
                        self.qhal._add_to_cache(
                            self.circuit_id, self.params, self.result
                        )

                return self.result

            except Exception as e:
                logger.warning(f"Circuit {self.circuit_id} execution failed (attempt {attempt}/{max_retries}): {e}")

                # Hardware-specific recovery strategies
                if self.gpu_type == "amd":
                    # AMD-specific recovery - more aggressive memory management
                    if hasattr(self.qhal, "hardware_manage"):
                        self.qhal.hardware_manage.clear_gpu_memory(force=True)
                elif self.gpu_type == "nvidia":
                    # NVIDIA GTX 1080 recovery - careful with limited VRAM
                    if hasattr(self.qhal, "hardware_manage"):
                        self.qhal.hardware_manage.clear_gpu_memory(force=True)

                # Reinitialize device on final attempts
                if attempt == max_retries - 1:
                    logger.warning("Reinitializing quantum device for final attempt")
                    self.qhal.device = None
                    if not self.qhal.initialize_device():
                        return self._classical_fallback()

        # If all attempts failed, use fallbackclass FaultSeverity(Enum):
            """Fault severity levels."""
            INFO = auto()      # Informational, no impact
            WARNING = auto()   # Minor impact, can continue
            ERROR = auto()     # Significant impact, needs attention
            CRITICAL = auto()  # Severe impact, immediate action required
            FATAL = auto()     # System cannot function

        class FaultCategory(Enum):
            """Fault categories."""
            HARDWARE = auto()         # Hardware-related issues
            QUANTUM = auto()          # Quantum device issues
            MEMORY = auto()           # Memory-related issues
            COMPUTATION = auto()      # Computation errors
            NETWORK = auto()          # Network-related issues
            RESOURCE = auto()         # Resource exhaustion
            EXTERNAL = auto()         # External service issues
            INTERNAL = auto()         # Internal logic errors
            UNKNOWN = auto()          # Unknown issues

        return self._classical_fallback()

    def _classical_fallback(self):
        """Provide optimized classical fallback with circuit type detection"""
        import numpy as np

        circuit_id_str = str(self.circuit_id).lower()

        # Determine appropriate fallback based on circuit purpose
        if "regime" in circuit_id_str:
            # Neutral regime probabilities
            self.result = np.array([0.5, 0.3, 0.1, 0.1])
        elif "risk" in circuit_id_str:
            self.result = 0.5  # Neutral risk score
        elif "signal" in circuit_id_str:
            self.result = 0.5  # Neutral signal
        elif "momentum" in circuit_id_str:
            self.result = 0.0  # Neutral momentum
        elif "feature" in circuit_id_str:
            # Default feature output
            self.result = np.array([0.5, 0.5, 0.5, 0.5])
        elif "stoploss" in circuit_id_str:
            # Neutral stoploss values
            self.result = np.array([-0.5, -0.5, -0.5])
        else:
            # Generic fallback based on parameter count
            if self.params is not None:
                param_count = len(self.params) if hasattr(self.params, "__len__") else 1
                if param_count > 1:
                    self.result = np.ones(param_count) * 0.5
                else:
                    self.result = 0.5
            else:
                self.result = 0.5

        return self.result

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources after execution"""
        # Handle specific cleanup based on GPU type
        if self.gpu_type == "amd":
            # RX 6800XT has more VRAM, so less aggressive cleanup needed
            pass
        elif self.gpu_type == "nvidia":
            # GTX 1080 needs more aggressive cleanup with only 8GB VRAM
            if hasattr(self.qhal, "hardware_manager"):
                # Only force clear for larger circuits
                force_clear = (
                    "feature" in str(self.circuit_id).lower()
                    or "regime" in str(self.circuit_id).lower()
                )
                self.qhal.hardware_manager.clear_gpu_memory(force=force_clear)

        # Let exceptions propagate
        return False


class AsyncManager:
    """
    Thread-safe async event loop management with hardware awareness
    """

    def __init__(self, hardware_manager=None):
        self.hardware_manager = hardware_manager or HardwareManager()
        self.running_tasks = []
        self._is_shutting_down = False

        import threading

        self._lock = threading.RLock()

        # Configure based on hardware
        self._configure()

    def _configure(self):
        """Configure async parameters based on hardware"""
        # Determine max concurrent tasks based on CPU count
        import psutil

        cpu_count = psutil.cpu_count(logical=False) or 4

        if self.hardware_manager.system_ram_capacity >= 64:  # High-RAM system
            self.max_concurrent_tasks = max(8, cpu_count - 2)
        else:  # Test unit
            self.max_concurrent_tasks = max(4, cpu_count - 2)

        logger.info(
            f"Async manager configured for {cpu_count} CPUs: "
            f"max_concurrent_tasks={self.max_concurrent_tasks}"
        )

    def get_event_loop(self):
        """Get event loop for current thread with better error handling"""
        import asyncio
        import threading

        with self._lock:
            # Check if shutting down
            if self._is_shutting_down:
                logger.warning("Attempted to get event loop during shutdown")
                return None

            # Determine if we're in the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()

            try:
                # Try to get current loop
                return asyncio.get_event_loop()
            except RuntimeError:
                # No running loop
                if not self._is_shutting_down:
                    # Create new loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Set default executor with appropriate thread count
                    import concurrent.futures

                    loop.set_default_executor(
                        concurrent.futures.ThreadPoolExecutor(
                            max_workers=self.max_concurrent_tasks
                        )
                    )

                    return loop
                else:
                    logger.warning("Avoided creating event loop during shutdown")
                    return None

    async def run_with_timeout(self, coro, timeout=30):
        """Run coroutine with timeout and cleanup"""
        import asyncio

        try:
            result = await asyncio.wait_for(coro, timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Async operation timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Error in async operation: {e}")
            raise

    def run_coroutine(self, coro, timeout=30):
        """Run a coroutine in the event loop with proper error handling"""
        loop = self.get_event_loop()
        if not loop:
            logger.error("No event loop available")
            return None

        with self._lock:
            if self._is_shutting_down:
                logger.warning("Attempted to run coroutine during shutdown")
                return None

            # Check if we're at the concurrent task limit
            active_tasks = len([t for t in self.running_tasks if not t.done()])
            if active_tasks >= self.max_concurrent_tasks:
                logger.warning(f"Reached maximum concurrent tasks limit ({self.max_concurrent_tasks})")
                # Clean up completed tasks
                self.running_tasks = [t for t in self.running_tasks if not t.done()]

            try:
                # Create task with timeout
                task = loop.create_task(self.run_with_timeout(coro, timeout))
                self.running_tasks.append(task)

                # Run and return result
                return loop.run_until_complete(task)
            except Exception as e:
                logger.error(f"Error running coroutine: {e}")
                return None

    def shutdown(self):
        """Shutdown async manager and cancel all tasks"""
        with self._lock:
            self._is_shutting_down = True

            # Cancel all running tasks
            import asyncio

            for task in self.running_tasks:
                if not task.done():
                    task.cancel()

            # Clear task list
            self.running_tasks = []

            # Close event loop if in main thread
            import threading

            if threading.current_thread() is threading.main_thread():
                try:
                    loop = asyncio.get_event_loop()
                    if hasattr(loop, "shutdown_asyncgens"):
                        loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()
                except Exception as e:
                    logger.error(f"Error closing event loop: {e}")



# Configure logging
logger = logging.getLogger("quantum_trading.fault")

# Global singleton instance and lock
_fault_tolerance_manager_instance = None
_lock = threading.RLock()


class FaultToleranceManager:
    """
    Comprehensive fault tolerance and recovery system.

    Provides:
    - Component health monitoring
    - Automatic recovery strategies
    - Fault isolation and containment
    - Graceful degradation
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize fault tolerance manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.components = {}
        self.health_checks = {}
        self.recovery_strategies = {}
        self.fault_history = {}
        self.fault_counters = {}
        self.component_status = {}

        # Lock for thread safety
        self.lock = threading.RLock()

        # Initialize monitoring
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        self.monitoring_thread = None
        self.running = False

        # Circuit breaker configuration
        self.circuit_breaker_threshold = self.config.get('circuit_breaker_threshold', 3)
        self.circuit_breaker_reset_time = self.config.get('circuit_breaker_reset_time', 300)
        self.circuit_breakers = {}

        # Start monitoring if configured
        if self.config.get('auto_start', True):
            self.start_monitoring()

    def register_component(self, name: str, component: Any,
                          health_check: Callable[[], bool] = None,
                          recovery_strategy: Callable[[], bool] = None,
                          category: FaultCategory = FaultCategory.UNKNOWN,
                          dependencies: List[str] = None) -> None:
        """
        Register component for fault tolerance.

        Args:
            name: Component name
            component: Component instance
            health_check: Function to check component health
            recovery_strategy: Function to recover component
            category: Fault category
            dependencies: List of component dependencies
        """
        with self.lock:
            self.components[name] = component

            if health_check:
                self.health_checks[name] = health_check

            if recovery_strategy:
                self.recovery_strategies[name] = recovery_strategy

            # Initialize fault history
            self.fault_history[name] = []

            # Initialize fault counters for different severity levels
            self.fault_counters[name] = {
                severity: 0 for severity in FaultSeverity
            }

            # Initialize component status
            self.component_status[name] = {
                'status': 'unknown',
                'last_check': 0,
                'last_failure': 0,
                'consecutive_failures': 0,
                'category': category,
                'dependencies': dependencies or []
            }

            # Initialize circuit breaker
            self.circuit_breakers[name] = {
                'tripped': False,
                'trip_time': 0,
                'failure_count': 0
            }

            logger.info(f"Registered component '{name}' for fault tolerance")

    def start_monitoring(self) -> None:
        """Start fault monitoring thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True,
                    name="FaultMonitoringThread"
                )
                self.monitoring_thread.start()
                logger.info("Started fault monitoring thread")

    def stop_monitoring(self) -> None:
        """Stop fault monitoring thread."""
        with self.lock:
            self.running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=1.0)
                logger.info("Stopped fault monitoring thread")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._check_all_components()
                self._check_circuit_breakers()
            except Exception as e:
                logger.error(f"Error in fault monitoring: {e}")
                logger.debug(traceback.format_exc())

            # Sleep until next check
            time.sleep(self.monitoring_interval)

    def _check_all_components(self) -> None:
        """Check health of all registered components."""
        with self.lock:
            for name, component in self.components.items():
                # Skip if circuit breaker is tripped
                if self.circuit_breakers[name]['tripped']:
                    continue

                try:
                    if name in self.health_checks:
                        is_healthy = self.health_checks[name]()

                        # Update component status
                        self.component_status[name]['last_check'] = time.time()

                        if is_healthy:
                            # Component is healthy
                            self.component_status[name]['status'] = 'healthy'
                            self.component_status[name]['consecutive_failures'] = 0
                        else:
                            # Component is unhealthy
                            self._handle_fault(name, component)
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {e}")
                    logger.debug(traceback.format_exc())
                    self._handle_fault(name, component, exception=e)

    def _check_circuit_breakers(self) -> None:
        """Check circuit breakers for reset."""
        with self.lock:
            current_time = time.time()

            for name, breaker in self.circuit_breakers.items():
                if breaker['tripped']:
                    # Check if reset time has elapsed
                    if current_time - breaker['trip_time'] >= self.circuit_breaker_reset_time:
                        # Reset circuit breaker
                        breaker['tripped'] = False
                        breaker['failure_count'] = 0
                        logger.info(f"Circuit breaker reset for component '{name}'")

                        # Try to recover component
                        if name in self.recovery_strategies:
                            try:
                                logger.info(f"Attempting recovery for '{name}' after circuit breaker reset")
                                self.recovery_strategies[name]()
                            except Exception as e:
                                logger.error(f"Recovery failed for '{name}' after circuit breaker reset: {e}")

    def _handle_fault(self, name: str, component: Any,
                     exception: Exception = None,
                     severity: FaultSeverity = FaultSeverity.ERROR,
                     category: FaultCategory = None) -> None:
        """
        Handle component fault.

        Args:
            name: Component name
            component: Component instance
            exception: Exception that triggered fault
            severity: Fault severity
            category: Fault category
        """
        with self.lock:
            # Get component category if not provided
            if category is None:
                category = self.component_status[name]['category']

            # Record fault
            fault_info = {
                'timestamp': time.time(),
                'exception': str(exception) if exception else "Health check failed",
                'severity': severity,
                'category': category,
                'traceback': traceback.format_exc() if exception else None
            }

            # Update fault history
            self.fault_history[name].append(fault_info)

            # Limit history size
            max_history = self.config.get('max_fault_history', 100)
            if len(self.fault_history[name]) > max_history:
                self.fault_history[name] = self.fault_history[name][-max_history:]

            # Update fault counters
            self.fault_counters[name][severity] += 1

            # Update component status
            self.component_status[name]['status'] = 'unhealthy'
            self.component_status[name]['last_failure'] = fault_info['timestamp']
            self.component_status[name]['consecutive_failures'] += 1

            # Update circuit breaker
            breaker = self.circuit_breakers[name]
            breaker['failure_count'] += 1

            # Trip circuit breaker if threshold reached
            if breaker['failure_count'] >= self.circuit_breaker_threshold:
                breaker['tripped'] = True
                breaker['trip_time'] = time.time()
                logger.warning(f"Circuit breaker tripped for component '{name}'")

            # Log fault
            log_message = f"Fault detected in component '{name}'"
            if exception:
                log_message += f": {exception}"

            if severity == FaultSeverity.CRITICAL or severity == FaultSeverity.FATAL:
                logger.critical(log_message)
            else:
                logger.error(log_message)

            # Try recovery if strategy exists and circuit breaker not tripped
            if name in self.recovery_strategies and not breaker['tripped']:
                try:
                    logger.info(f"Attempting recovery for '{name}'")
                    recovery_success = self.recovery_strategies[name]()

                    if recovery_success:
                        logger.info(f"Recovery successful for '{name}'")
                        # Reset consecutive failures but keep history
                        self.component_status[name]['consecutive_failures'] = 0
                    else:
                        logger.warning(f"Recovery attempted but failed for '{name}'")
                except Exception as e:
                    logger.error(f"Recovery failed for '{name}': {e}")

    def report_fault(self, component_name: str, message: str,
                    exception: Exception = None,
                    severity: FaultSeverity = FaultSeverity.ERROR,
                    category: FaultCategory = None) -> None:
        """
        Report fault from external source.

        Args:
            component_name: Name of component with fault
            message: Fault description
            exception: Exception that caused the fault
            severity: Fault severity
            category: Fault category
        """
        with self.lock:
            # Check if component is registered
            if component_name not in self.components:
                logger.warning(f"Fault reported for unregistered component '{component_name}'")
                return

            # Create exception if not provided
            if exception is None and message:
                exception = Exception(message)

            # Handle fault
            self._handle_fault(
                component_name,
                self.components[component_name],
                exception,
                severity,
                category
            )

    def get_component_status(self, component_name: str = None) -> Dict[str, Any]:
        """
        Get status of components.

        Args:
            component_name: Name of specific component, or None for all

        Returns:
            dict: Component status information
        """
        with self.lock:
            if component_name:
                # Check if component exists
                if component_name not in self.component_status:
                    return {'error': f"Component '{component_name}' not found"}

                # Return specific component status
                return {
                    component_name: {
                        **self.component_status[component_name],
                        'faults': len(self.fault_history[component_name]),
                        'fault_counters': {
                            severity.name: count
                            for severity, count in self.fault_counters[component_name].items()
                        },
                        'circuit_breaker': self.circuit_breakers[component_name]
                    }
                }

            # Return all component statuses
            result = {}
            for name in self.component_status:
                result[name] = {
                    **self.component_status[name],
                    'faults': len(self.fault_history[name]),
                    'fault_counters': {
                        severity.name: count
                        for severity, count in self.fault_counters[name].items()
                    },
                    'circuit_breaker': self.circuit_breakers[name]
                }

            return result

    def get_fault_history(self, component_name: str = None,
                         limit: int = None,
                         severity: FaultSeverity = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get fault history.

        Args:
            component_name: Name of specific component, or None for all
            limit: Maximum number of faults to return
            severity: Filter by severity

        Returns:
            dict: Fault history by component
        """
        with self.lock:
            if component_name:
                # Check if component exists
                if component_name not in self.fault_history:
                    return {'error': f"Component '{component_name}' not found"}

                # Get faults for specific component
                faults = self.fault_history[component_name]

                # Filter by severity if requested
                if severity:
                    faults = [
                        fault for fault in faults
                        if fault['severity'] == severity
                    ]

                # Apply limit if requested
                if limit:
                    faults = faults[-limit:]

                return {component_name: faults}

            # Get faults for all components
            result = {}
            for name, faults in self.fault_history.items():
                # Filter by severity if requested
                if severity:
                    filtered_faults = [
                        fault for fault in faults
                        if fault['severity'] == severity
                    ]
                else:
                    filtered_faults = faults

                # Apply limit if requested
                if limit:
                    filtered_faults = filtered_faults[-limit:]

                result[name] = filtered_faults

            return result

    def reset_component(self, component_name: str) -> bool:
        """
        Reset component fault status and circuit breaker.

        Args:
            component_name: Name of component to reset

        Returns:
            bool: True if reset successful, False otherwise
        """
        with self.lock:
            # Check if component exists
            if component_name not in self.components:
                logger.warning(f"Cannot reset unknown component '{component_name}'")
                return False

            # Reset circuit breaker
            self.circuit_breakers[component_name] = {
                'tripped': False,
                'trip_time': 0,
                'failure_count': 0
            }

            # Reset component status
            self.component_status[component_name]['consecutive_failures'] = 0
            self.component_status[component_name]['status'] = 'unknown'

            logger.info(f"Reset component '{component_name}'")

            # Try to recover component
            if component_name in self.recovery_strategies:
                try:
                    logger.info(f"Attempting recovery for '{component_name}' after reset")
                    return self.recovery_strategies[component_name]()
                except Exception as e:
                    logger.error(f"Recovery failed for '{component_name}' after reset: {e}")
                    return False

            return True

    def shutdown(self) -> None:
        """Clean up resources and prepare for shutdown."""
        self.stop_monitoring()

        with self.lock:
            # Clear registered components
            self.components.clear()
            self.health_checks.clear()
            self.recovery_strategies.clear()

            logger.info("Fault tolerance manager shutting down")


def get_fault_tolerance_manager(config: Dict[str, Any] = None,
                               reset: bool = False) -> FaultToleranceManager:
    """
    Thread-safe singleton factory for FaultToleranceManager.

    Args:
        config: Optional configuration dictionary
        reset: Whether to force recreation of instance

    Returns:
        FaultToleranceManager: Singleton instance
    """
    global _fault_tolerance_manager_instance, _lock

    if _fault_tolerance_manager_instance is None or reset:
        with _lock:
            if _fault_tolerance_manager_instance is None or reset:
                # Clean up existing instance if needed
                if _fault_tolerance_manager_instance is not None:
                    _fault_tolerance_manager_instance.shutdown()

                # Create new instance with proper error handling
                try:
                    _fault_tolerance_manager_instance = FaultToleranceManager(config or {})
                except Exception as e:
                    logger.error(f"Failed to initialize FaultToleranceManager: {e}")
                    # Create minimal but functional instance
                    minimal_config = {'monitoring_interval': 60, 'auto_start': False}
                    _fault_tolerance_manager_instance = FaultToleranceManager(minimal_config)

    return _fault_tolerance_manager_instance

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class ComponentErrorBoundary:
    """Isolates component failures to prevent system-wide crashes"""

    def __init__(self, component_name):
        self.component_name = component_name
        self.failure_count = 0
        self.last_error = None
        self.max_failures = 3
        self.recovery_attempts = 0

    def execute(self, func, *args, **kwargs):
        """Execute function with error boundary protection"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.failure_count += 1
            self.last_error = e
            logger.warning(f"Component {self.component_name} failed: {e}")

            # Return safe default value based on function name
            if "populate" in func.__name__:
                # For indicator functions, return original dataframe
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        return arg
                return pd.DataFrame()  # Empty dataframe as last resort

            return None  # Safe default for other functions


class DataFrameProcessor:
    """
    Memory-efficient DataFrame processing optimized for large RAM systems
    with adaptive batch processing for different hardware configurations
    """

    def __init__(self, hardware_manager=None):
        self.hardware_manager = hardware_manager or HardwareManager()

        # Configure batch sizes based on available RAM
        self._configure_batch_sizes()

        # Track memory peaks
        self.peak_memory_usage = 0
        self.last_check_time = 0

    def _configure_batch_sizes(self):
        """Configure batch sizes based on available system RAM"""
        ram_gb = self.hardware_manager.system_ram_capacity

        if ram_gb >= 64:  # High RAM (96GB)
            self.large_dataframe_threshold = 1000000  # 1M rows
            self.default_batch_size = 100000  # 100K rows per batch
        elif ram_gb >= 24:  # Medium RAM (32GB)
            self.large_dataframe_threshold = 500000  # 500K rows
            self.default_batch_size = 50000  # 50K rows per batch
        else:  # Low RAM
            self.large_dataframe_threshold = 100000  # 100K rows
            self.default_batch_size = 10000  # 10K rows per batch

        logger.info(
            f"DataFrame processor configured for {ram_gb:.1f}GB RAM: "
            f"threshold={self.large_dataframe_threshold}, "
            f"batch_size={self.default_batch_size}"
        )

    def process_indicators(self, dataframe, metadata, indicator_func):
        """
        Process indicators with memory-efficient batching for large DataFrames

        Args:
                dataframe: Input DataFrame to process
                metadata: DataFrame metadata
                indicator_func: Function that computes indicators for a DataFrame batch

        Returns:
                DataFrame with computed indicators
        """
        import pandas as pd

        # Skip processing if DataFrame is empty or too small
        if dataframe.empty:
            return dataframe

        # For small DataFrames, process directly
        if len(dataframe) < self.large_dataframe_threshold:
            return indicator_func(dataframe, metadata)

        # For large DataFrames, use memory-efficient batch processing
        logger.info(
            f"Using batched processing for large DataFrame ({len(dataframe)} rows)"
        )

        # Split into batches
        batch_size = self.default_batch_size
        result_batches = []

        # Process in batches to limit memory usage
        for start_idx in range(0, len(dataframe), batch_size):
            end_idx = min(start_idx + batch_size, len(dataframe))
            batch = dataframe.iloc[start_idx:end_idx].copy()

            # Process batch
            try:
                batch_result = indicator_func(batch, metadata)
                result_batches.append(batch_result)

                # Check memory usage periodically
                import time

                current_time = time.time()
                if current_time - self.last_check_time > 5:  # Check every 5 seconds
                    self._check_memory_usage()
                    self.last_check_time = current_time

            except Exception as e:
                logger.error(f"Error processing batch {start_idx}-{end_idx}: {e}")
                # Add unprocessed batch to preserve data integrity
                result_batches.append(batch)

        # Combine results
        try:
            result = pd.concat(result_batches)

            # Force garbage collection after large operation
            import gc

            gc.collect()

            # Clear GPU memory if needed
            if hasattr(self.hardware_manager, "clear_gpu_memory"):
                self.hardware_manager.clear_gpu_memory(force=False)

            return result

        except Exception as e:
            logger.error(f"Error combining batched results: {e}")
            # Return original DataFrame as fallback
            return dataframe

    def _check_memory_usage(self):
        """Check current memory usage and take action if needed"""
        import psutil

        # Get current memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Update peak if current usage is higher
        self.peak_memory_usage = max(self.peak_memory_usage, memory_percent)

        # Take action if memory usage is too high
        if memory_percent > 85:
            logger.warning(f"High memory usage detected: {memory_percent:.1f}%")

            # Clear pandas caches if possible
            try:
                from pandas import _testing as tm

                tm.reset_testing_mode()
            except:
                pass

            # Force garbage collection
            import gc

            gc.collect()


class AdaptiveResourceManager:
    """Dynamically manages system resources based on load and available hardware"""

    def __init__(self, initial_resources=None):
        # Initialize with safe defaults
        self._resources = initial_resources or {
            "memory": {
                "total": 16384,  # Default to 16GB
                "available": 8192,
                "used": 0,
            },
            "gpu": {"available": False, "vram_mb": 0},
            "cpu": {"cores": os.cpu_count() or 4, "load": 0.0},
        }
        self.gpu_info = self._detect_gpus()
        self.cpu_count = os.cpu_count()
        self.memory_limit = self._get_memory_limit()
        self.resource_allocation = self._calculate_initial_allocation()

        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._resource_monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _detect_gpus(self):
        """Detect available GPUs and their capabilities"""
        # Implementation similar to what we did for H2OCatBoostHybrid
        # ...

    def _get_memory_limit(self):
        """Determine memory limits based on system"""
        import psutil

        total_memory = psutil.virtual_memory().total
        # Use 80% of total memory as limit
        return int(total_memory * 0.8)

    def _calculate_initial_allocation(self):
        """Calculate initial resource allocation for components"""
        # Divide resources among components
        return {
            "quantum_hardware": {
                "cpu_threads": max(1, self.cpu_count // 4),
                "gpu_memory_pct": 0.3 if self.gpu_info["gpu_available"] else 0,
            },
            "ml_processor": {
                "cpu_threads": max(1, self.cpu_count // 2),
                "gpu_memory_pct": 0.5 if self.gpu_info["gpu_available"] else 0,
            },
            # Other components
        }

    def _resource_monitor_loop(self):
        """Continuously monitor and adjust resource allocation"""
        import time

        import psutil

        while self.monitoring_active:
            try:
                # Check current CPU and memory usage
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent

                # Check GPU usage if available
                gpu_usage = self._check_gpu_usage()

                # Adjust allocations based on usage
                self._adjust_allocations(cpu_usage, memory_usage, gpu_usage)

                # Sleep for a bit
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                time.sleep(10)  # Longer sleep on error

    def get_allocation(self, component_name):
        """Get current resource allocation for a component"""
        return self.resource_allocation.get(component_name, {})
