"""
Error Handling and Fallback Mechanisms for Neural Forecasting.

This module provides comprehensive error handling, graceful degradation,
and fallback mechanisms for the NHITS forecasting system.
"""

import asyncio
import logging
import traceback
import functools
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FallbackStrategy(Enum):
    """Fallback strategy types."""
    SIMPLE_AVERAGE = "simple_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    LINEAR_TREND = "linear_trend"
    HISTORICAL_PATTERN = "historical_pattern"
    CACHED_PREDICTION = "cached_prediction"
    FAIL_SAFE = "fail_safe"


@dataclass
class ErrorEvent:
    """Error event record."""
    timestamp: str
    error_type: str
    severity: ErrorSeverity
    message: str
    component: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolution_strategy: Optional[str] = None


@dataclass
class FallbackResult:
    """Fallback prediction result."""
    prediction: List[float]
    confidence: float
    method: FallbackStrategy
    metadata: Dict[str, Any]
    timestamp: str
    success: bool


class NeuralForecastErrorHandler:
    """
    Comprehensive error handling system for neural forecasting.
    
    Features:
    - Multi-level error categorization
    - Intelligent fallback mechanisms
    - Circuit breaker pattern
    - Error recovery strategies
    - Performance degradation handling
    - Monitoring and alerting
    """
    
    def __init__(
        self,
        max_retry_attempts: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 300,  # 5 minutes
        enable_fallbacks: bool = True,
        enable_caching: bool = True,
        fallback_cache_ttl: int = 3600  # 1 hour
    ):
        """
        Initialize error handler.
        
        Args:
            max_retry_attempts: Maximum retry attempts for failed operations
            circuit_breaker_threshold: Number of failures to trigger circuit breaker
            circuit_breaker_timeout: Circuit breaker timeout in seconds
            enable_fallbacks: Enable fallback mechanisms
            enable_caching: Enable result caching for fallbacks
            fallback_cache_ttl: Fallback cache TTL in seconds
        """
        self.max_retry_attempts = max_retry_attempts
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.enable_fallbacks = enable_fallbacks
        self.enable_caching = enable_caching
        self.fallback_cache_ttl = fallback_cache_ttl
        
        # Error tracking
        self.error_history: List[ErrorEvent] = []
        self.component_failure_counts: Dict[str, int] = {}
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        
        # Fallback caching
        self.fallback_cache: Dict[str, Tuple[FallbackResult, datetime]] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            'response_times': [],
            'error_rates': [],
            'fallback_usage': []
        }
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        self.logger.info("Neural Forecast Error Handler initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def handle_errors(
        self,
        component: str,
        fallback_strategy: FallbackStrategy = FallbackStrategy.SIMPLE_AVERAGE,
        enable_retries: bool = True,
        enable_circuit_breaker: bool = True
    ):
        """
        Decorator for comprehensive error handling.
        
        Args:
            component: Component name for tracking
            fallback_strategy: Fallback strategy to use
            enable_retries: Enable retry mechanism
            enable_circuit_breaker: Enable circuit breaker
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Check circuit breaker
                if enable_circuit_breaker and self._is_circuit_breaker_open(component):
                    self.logger.warning(f"Circuit breaker open for {component}, using fallback")
                    return await self._execute_fallback(
                        component, fallback_strategy, *args, **kwargs
                    )
                
                # Attempt operation with retries
                last_error = None
                for attempt in range(self.max_retry_attempts if enable_retries else 1):
                    try:
                        start_time = time.time()
                        
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Track successful execution
                        execution_time = time.time() - start_time
                        self._track_success(component, execution_time)
                        
                        return result
                        
                    except Exception as e:
                        last_error = e
                        
                        # Categorize and record error
                        error_event = self._categorize_error(e, component, attempt)
                        self._record_error(error_event)
                        
                        # Update circuit breaker
                        if enable_circuit_breaker:
                            self._update_circuit_breaker(component)
                        
                        # Wait before retry (exponential backoff)
                        if attempt < self.max_retry_attempts - 1:
                            wait_time = (2 ** attempt) * 0.5  # 0.5, 1, 2 seconds
                            await asyncio.sleep(wait_time)
                            self.logger.info(f"Retrying {component} (attempt {attempt + 2})")
                        else:
                            self.logger.error(f"All retry attempts failed for {component}")
                
                # All attempts failed, use fallback
                if self.enable_fallbacks:
                    self.logger.warning(f"Using fallback strategy for {component}")
                    return await self._execute_fallback(
                        component, fallback_strategy, *args, **kwargs
                    )
                else:
                    # Re-raise the last error
                    raise last_error
            
            return wrapper
        return decorator
    
    def _categorize_error(self, error: Exception, component: str, attempt: int) -> ErrorEvent:
        """Categorize error and determine severity."""
        error_type = type(error).__name__
        message = str(error)
        
        # Determine severity based on error type and context
        if isinstance(error, (MemoryError, SystemError)):
            severity = ErrorSeverity.CRITICAL
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError)):
            severity = ErrorSeverity.MEDIUM
        else:
            severity = ErrorSeverity.LOW
        
        # GPU-specific errors
        if 'cuda' in message.lower() or 'gpu' in message.lower():
            severity = ErrorSeverity.HIGH
        
        # Model-specific errors
        if 'model' in message.lower() or 'forecast' in message.lower():
            severity = ErrorSeverity.MEDIUM
        
        return ErrorEvent(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            severity=severity,
            message=message,
            component=component,
            context={
                'attempt': attempt,
                'max_attempts': self.max_retry_attempts,
                'args_count': 0,  # Would need to be passed in real implementation
                'kwargs_count': 0
            },
            stack_trace=traceback.format_exc()
        )
    
    def _record_error(self, error_event: ErrorEvent):
        """Record error event for monitoring and analysis."""
        self.error_history.append(error_event)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # Update component failure counts
        component = error_event.component
        self.component_failure_counts[component] = self.component_failure_counts.get(component, 0) + 1
        
        # Log based on severity
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR in {component}: {error_event.message}")
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR in {component}: {error_event.message}")
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR in {component}: {error_event.message}")
        else:
            self.logger.info(f"Low severity error in {component}: {error_event.message}")
    
    def _track_success(self, component: str, execution_time: float):
        """Track successful execution."""
        self.performance_metrics['response_times'].append(execution_time)
        
        # Reset circuit breaker on success
        if component in self.circuit_breaker_states:
            self.circuit_breaker_states[component]['failure_count'] = 0
            self.circuit_breaker_states[component]['last_failure'] = None
        
        # Keep metrics manageable
        if len(self.performance_metrics['response_times']) > 1000:
            self.performance_metrics['response_times'] = self.performance_metrics['response_times'][-1000:]
    
    def _is_circuit_breaker_open(self, component: str) -> bool:
        """Check if circuit breaker is open for component."""
        if component not in self.circuit_breaker_states:
            return False
        
        state = self.circuit_breaker_states[component]
        failure_count = state.get('failure_count', 0)
        last_failure = state.get('last_failure')
        
        # Check if threshold exceeded
        if failure_count < self.circuit_breaker_threshold:
            return False
        
        # Check if timeout has passed
        if last_failure:
            time_since_failure = (datetime.now() - last_failure).total_seconds()
            if time_since_failure > self.circuit_breaker_timeout:
                # Reset circuit breaker
                state['failure_count'] = 0
                state['last_failure'] = None
                self.logger.info(f"Circuit breaker reset for {component}")
                return False
        
        return True
    
    def _update_circuit_breaker(self, component: str):
        """Update circuit breaker state after failure."""
        if component not in self.circuit_breaker_states:
            self.circuit_breaker_states[component] = {
                'failure_count': 0,
                'last_failure': None
            }
        
        state = self.circuit_breaker_states[component]
        state['failure_count'] += 1
        state['last_failure'] = datetime.now()
        
        if state['failure_count'] >= self.circuit_breaker_threshold:
            self.logger.warning(f"Circuit breaker triggered for {component}")
    
    async def _execute_fallback(
        self,
        component: str,
        strategy: FallbackStrategy,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute fallback strategy."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(component, strategy, args, kwargs)
            
            if self.enable_caching and cache_key in self.fallback_cache:
                cached_result, cache_time = self.fallback_cache[cache_key]
                if datetime.now() - cache_time < timedelta(seconds=self.fallback_cache_ttl):
                    self.logger.info(f"Using cached fallback for {component}")
                    return self._format_fallback_response(cached_result, True)
            
            # Execute fallback strategy
            fallback_result = await self._get_fallback_prediction(strategy, args, kwargs)
            
            # Cache result
            if self.enable_caching:
                self.fallback_cache[cache_key] = (fallback_result, datetime.now())
            
            # Track fallback usage
            self.performance_metrics['fallback_usage'].append(1)
            
            return self._format_fallback_response(fallback_result, True)
            
        except Exception as e:
            self.logger.error(f"Fallback strategy failed: {str(e)}")
            
            # Last resort: fail-safe fallback
            fail_safe_result = await self._fail_safe_fallback()
            return self._format_fallback_response(fail_safe_result, False)
    
    async def _get_fallback_prediction(
        self,
        strategy: FallbackStrategy,
        args: Tuple,
        kwargs: Dict
    ) -> FallbackResult:
        """Generate fallback prediction based on strategy."""
        try:
            # Extract data from arguments (assuming data is in first argument or 'data' kwarg)
            data = None
            if args:
                data = args[0]
            elif 'data' in kwargs:
                data = kwargs['data']
            
            if data is None:
                raise ValueError("No data found in arguments for fallback")
            
            # Convert to usable format
            if isinstance(data, dict):
                if 'y' in data:
                    values = data['y']
                elif 'prices' in data:
                    values = data['prices']
                elif 'close' in data:
                    values = data['close']
                else:
                    # Use first numeric list found
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], (int, float)):
                                values = value
                                break
                    else:
                        raise ValueError("No numeric data found for fallback")
            elif isinstance(data, (list, np.ndarray)):
                values = data
            elif isinstance(data, pd.DataFrame):
                if 'y' in data.columns:
                    values = data['y'].tolist()
                elif 'close' in data.columns:
                    values = data['close'].tolist()
                else:
                    # Use first numeric column
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        values = data[numeric_cols[0]].tolist()
                    else:
                        raise ValueError("No numeric columns found for fallback")
            else:
                raise ValueError(f"Unsupported data type for fallback: {type(data)}")
            
            # Ensure we have enough data
            if len(values) < 2:
                raise ValueError("Insufficient data for fallback prediction")
            
            # Generate prediction based on strategy
            if strategy == FallbackStrategy.SIMPLE_AVERAGE:
                prediction = await self._simple_average_fallback(values)
            elif strategy == FallbackStrategy.EXPONENTIAL_SMOOTHING:
                prediction = await self._exponential_smoothing_fallback(values)
            elif strategy == FallbackStrategy.LINEAR_TREND:
                prediction = await self._linear_trend_fallback(values)
            elif strategy == FallbackStrategy.HISTORICAL_PATTERN:
                prediction = await self._historical_pattern_fallback(values)
            else:
                # Default to simple average
                prediction = await self._simple_average_fallback(values)
            
            return FallbackResult(
                prediction=prediction,
                confidence=0.3,  # Low confidence for fallback
                method=strategy,
                metadata={
                    'data_points': len(values),
                    'last_value': values[-1],
                    'strategy_used': strategy.value
                },
                timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Fallback prediction failed: {str(e)}")
            return FallbackResult(
                prediction=[0.0],
                confidence=0.0,
                method=FallbackStrategy.FAIL_SAFE,
                metadata={'error': str(e)},
                timestamp=datetime.now().isoformat(),
                success=False
            )
    
    async def _simple_average_fallback(self, values: List[float]) -> List[float]:
        """Simple moving average fallback."""
        if len(values) == 0:
            return [0.0]
        
        # Use last 10 values or all if less than 10
        recent_values = values[-10:]
        avg = np.mean(recent_values)
        
        # Return single prediction
        return [float(avg)]
    
    async def _exponential_smoothing_fallback(self, values: List[float]) -> List[float]:
        """Exponential smoothing fallback."""
        if len(values) == 0:
            return [0.0]
        
        alpha = 0.3  # Smoothing parameter
        forecast = values[0]
        
        # Apply exponential smoothing
        for value in values[1:]:
            forecast = alpha * value + (1 - alpha) * forecast
        
        return [float(forecast)]
    
    async def _linear_trend_fallback(self, values: List[float]) -> List[float]:
        """Linear trend extrapolation fallback."""
        if len(values) < 2:
            return [values[0] if values else 0.0]
        
        # Use last 5 points for trend calculation
        recent_values = values[-5:]
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Calculate linear trend
        if len(recent_values) > 1:
            slope = np.polyfit(x, y, 1)[0]
            next_value = recent_values[-1] + slope
        else:
            next_value = recent_values[-1]
        
        return [float(next_value)]
    
    async def _historical_pattern_fallback(self, values: List[float]) -> List[float]:
        """Historical pattern-based fallback."""
        if len(values) < 7:
            return await self._simple_average_fallback(values)
        
        # Look for weekly patterns (last 7 values)
        pattern_length = min(7, len(values) // 2)
        recent_pattern = values[-pattern_length:]
        
        # Simple pattern continuation
        next_value = recent_pattern[0]  # Assume pattern repeats
        
        return [float(next_value)]
    
    async def _fail_safe_fallback(self) -> FallbackResult:
        """Last resort fail-safe fallback."""
        return FallbackResult(
            prediction=[0.0],
            confidence=0.0,
            method=FallbackStrategy.FAIL_SAFE,
            metadata={'message': 'Fail-safe fallback used'},
            timestamp=datetime.now().isoformat(),
            success=True
        )
    
    def _generate_cache_key(
        self,
        component: str,
        strategy: FallbackStrategy,
        args: Tuple,
        kwargs: Dict
    ) -> str:
        """Generate cache key for fallback results."""
        # Create a simple hash of component, strategy, and data characteristics
        key_parts = [component, strategy.value]
        
        # Add data characteristics if available
        if args and isinstance(args[0], (dict, list)):
            data = args[0]
            if isinstance(data, dict):
                key_parts.append(str(len(data)))
                if 'y' in data:
                    key_parts.append(f"y_{len(data['y'])}")
            elif isinstance(data, list):
                key_parts.append(f"list_{len(data)}")
        
        return "_".join(key_parts)
    
    def _format_fallback_response(self, result: FallbackResult, success: bool) -> Dict[str, Any]:
        """Format fallback result as standard response."""
        return {
            'success': success,
            'point_forecast': result.prediction,
            'prediction_intervals': {
                '80%': {
                    'lower': [p * 0.9 for p in result.prediction],
                    'upper': [p * 1.1 for p in result.prediction]
                }
            } if success else {},
            'confidence': result.confidence,
            'method': result.method.value,
            'fallback': True,
            'metadata': result.metadata,
            'timestamp': result.timestamp,
            'warning': 'Using fallback prediction due to model failure'
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)
            
            # Filter errors by time period
            recent_errors = [
                e for e in self.error_history
                if datetime.fromisoformat(e.timestamp) > last_hour
            ]
            
            daily_errors = [
                e for e in self.error_history
                if datetime.fromisoformat(e.timestamp) > last_day
            ]
            
            # Calculate error rates by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([
                    e for e in daily_errors if e.severity == severity
                ])
            
            # Component-wise error analysis
            component_errors = {}
            for error in daily_errors:
                component = error.component
                if component not in component_errors:
                    component_errors[component] = {'count': 0, 'severities': {}}
                component_errors[component]['count'] += 1
                
                severity = error.severity.value
                if severity not in component_errors[component]['severities']:
                    component_errors[component]['severities'][severity] = 0
                component_errors[component]['severities'][severity] += 1
            
            # Circuit breaker status
            circuit_breaker_status = {}
            for component, state in self.circuit_breaker_states.items():
                circuit_breaker_status[component] = {
                    'is_open': self._is_circuit_breaker_open(component),
                    'failure_count': state.get('failure_count', 0),
                    'last_failure': state.get('last_failure').isoformat() if state.get('last_failure') else None
                }
            
            return {
                'total_errors': len(self.error_history),
                'errors_last_hour': len(recent_errors),
                'errors_last_day': len(daily_errors),
                'error_rate_last_hour': len(recent_errors) / max(1, len(recent_errors) + 10),  # Rough estimate
                'severity_breakdown': severity_counts,
                'component_errors': component_errors,
                'circuit_breaker_status': circuit_breaker_status,
                'fallback_cache_size': len(self.fallback_cache),
                'performance_metrics': {
                    'avg_response_time': np.mean(self.performance_metrics['response_times'][-100:]) if self.performance_metrics['response_times'] else 0,
                    'fallback_usage_rate': len(self.performance_metrics['fallback_usage']) / max(1, len(self.performance_metrics['response_times']) + len(self.performance_metrics['fallback_usage']))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate error statistics: {str(e)}")
            return {'error': str(e)}
    
    def reset_circuit_breaker(self, component: str) -> bool:
        """Manually reset circuit breaker for a component."""
        try:
            if component in self.circuit_breaker_states:
                self.circuit_breaker_states[component] = {
                    'failure_count': 0,
                    'last_failure': None
                }
                self.logger.info(f"Circuit breaker manually reset for {component}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to reset circuit breaker: {str(e)}")
            return False
    
    def clear_error_history(self):
        """Clear error history (use with caution)."""
        self.error_history.clear()
        self.component_failure_counts.clear()
        self.fallback_cache.clear()
        self.logger.info("Error history cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of error handling system."""
        try:
            now = datetime.now()
            recent_errors = [
                e for e in self.error_history
                if datetime.fromisoformat(e.timestamp) > now - timedelta(minutes=30)
            ]
            
            # Check for critical issues
            critical_errors = [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            open_circuit_breakers = [
                comp for comp in self.circuit_breaker_states.keys()
                if self._is_circuit_breaker_open(comp)
            ]
            
            # Determine health status
            if critical_errors or len(open_circuit_breakers) > 2:
                status = 'critical'
            elif len(recent_errors) > 10 or open_circuit_breakers:
                status = 'warning'
            else:
                status = 'healthy'
            
            return {
                'status': status,
                'timestamp': now.isoformat(),
                'recent_errors': len(recent_errors),
                'critical_errors': len(critical_errors),
                'open_circuit_breakers': open_circuit_breakers,
                'fallback_cache_size': len(self.fallback_cache),
                'error_handler_enabled': True,
                'fallbacks_enabled': self.enable_fallbacks,
                'caching_enabled': self.enable_caching
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }