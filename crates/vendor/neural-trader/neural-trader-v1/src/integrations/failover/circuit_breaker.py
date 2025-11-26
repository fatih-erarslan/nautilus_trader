"""
Circuit Breaker and Failover System
Provides resilient API integration with automatic failover and recovery
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, List, Type
import random

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker activated (failing fast)
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for API resilience
    
    Monitors failures and automatically opens circuit when failure
    threshold is exceeded, preventing cascade failures.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        name: str = "default"
    ):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures to trigger open state
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to count as failure
            name: Name for logging purposes
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        self.success_count_in_half_open = 0
        self.required_successes_to_close = 3
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.last_success_time: Optional[datetime] = None
        self.state_changes: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized circuit breaker '{self.name}' "
            f"(threshold: {failure_threshold}, timeout: {recovery_timeout}s)"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
            Exception: Original exception from function
        """
        self.total_calls += 1
        
        # Check circuit state
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next attempt in {self._time_until_retry():.1f}s"
                )
        
        try:
            # Execute the function
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
        except Exception as e:
            # Don't count unexpected exceptions as circuit breaker failures
            logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
    
    def _time_until_retry(self) -> float:
        """Calculate seconds until next retry attempt"""
        if not self.last_failure_time:
            return 0.0
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return max(0.0, self.recovery_timeout - elapsed)
    
    def _on_success(self):
        """Handle successful function execution"""
        self.total_successes += 1
        self.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count_in_half_open += 1
            
            if self.success_count_in_half_open >= self.required_successes_to_close:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed function execution"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Any failure in half-open immediately goes back to open
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state"""
        self.state = CircuitState.OPEN
        self.success_count_in_half_open = 0
        
        self._record_state_change("OPEN", f"Failure threshold reached ({self.failure_count})")
        
        logger.warning(
            f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
        )
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.success_count_in_half_open = 0
        
        self._record_state_change("HALF_OPEN", "Attempting recovery")
        
        logger.info(f"Circuit breaker '{self.name}' attempting recovery (HALF_OPEN)")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count_in_half_open = 0
        
        self._record_state_change("CLOSED", "Recovery successful")
        
        logger.info(f"Circuit breaker '{self.name}' recovered (CLOSED)")
    
    def _record_state_change(self, new_state: str, reason: str):
        """Record state change for metrics"""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'from_state': self.state.value if hasattr(self.state, 'value') else str(self.state),
            'to_state': new_state,
            'reason': reason,
            'failure_count': self.failure_count,
            'total_failures': self.total_failures
        }
        
        self.state_changes.append(change_record)
        
        # Keep only last 50 state changes
        if len(self.state_changes) > 50:
            self.state_changes = self.state_changes[-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        success_rate = 0.0
        if self.total_calls > 0:
            success_rate = self.total_successes / self.total_calls
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'total_calls': self.total_calls,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'time_until_retry': self._time_until_retry() if self.state == CircuitState.OPEN else 0,
            'recovery_timeout': self.recovery_timeout,
            'recent_state_changes': self.state_changes[-10:]  # Last 10 changes
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count_in_half_open = 0
        
        self._record_state_change("CLOSED", "Manual reset")
        
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class APIFailoverManager:
    """
    Manages failover between multiple API providers with circuit breakers
    """
    
    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.provider_priorities: Dict[str, List[str]] = {}
        
        # Failover configuration
        self.max_retry_attempts = 3
        self.retry_delay_base = 1.0  # Base delay in seconds
        self.retry_delay_max = 30.0  # Maximum delay
        
        logger.info("Initialized API failover manager")
    
    def register_provider(
        self,
        provider_type: str,
        provider_name: str,
        provider_instance: Any,
        priority: int = 1,
        circuit_config: Dict[str, Any] = None
    ):
        """
        Register a provider with failover system
        
        Args:
            provider_type: Type of provider (e.g., 'news', 'market_data')
            provider_name: Unique name for provider
            provider_instance: Provider instance
            priority: Priority level (1 = highest)
            circuit_config: Circuit breaker configuration
        """
        if provider_type not in self.providers:
            self.providers[provider_type] = {}
            self.provider_priorities[provider_type] = []
        
        # Store provider
        self.providers[provider_type][provider_name] = {
            'instance': provider_instance,
            'priority': priority,
            'enabled': True,
            'last_used': None,
            'error_count': 0
        }
        
        # Create circuit breaker
        cb_config = circuit_config or {}
        circuit_breaker = CircuitBreaker(
            failure_threshold=cb_config.get('failure_threshold', 5),
            recovery_timeout=cb_config.get('recovery_timeout', 60),
            expected_exception=cb_config.get('expected_exception', Exception),
            name=f"{provider_type}_{provider_name}"
        )
        
        self.circuit_breakers[f"{provider_type}_{provider_name}"] = circuit_breaker
        
        # Update priority list
        self._update_priority_list(provider_type)
        
        logger.info(f"Registered provider {provider_name} for {provider_type} (priority: {priority})")
    
    def _update_priority_list(self, provider_type: str):
        """Update priority-sorted list for provider type"""
        providers = self.providers[provider_type]
        
        # Sort by priority (lower number = higher priority)
        sorted_providers = sorted(
            providers.items(),
            key=lambda x: x[1]['priority']
        )
        
        self.provider_priorities[provider_type] = [name for name, _ in sorted_providers]
    
    async def execute_with_failover(
        self,
        provider_type: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute method with automatic failover between providers
        
        Args:
            provider_type: Type of provider to use
            method_name: Method name to call on provider
            *args: Method arguments  
            **kwargs: Method keyword arguments
            
        Returns:
            Method result from successful provider
            
        Raises:
            Exception: If all providers fail
        """
        if provider_type not in self.providers:
            raise ValueError(f"No providers registered for type: {provider_type}")
        
        provider_names = self.provider_priorities[provider_type]
        last_exception = None
        
        for provider_name in provider_names:
            provider_info = self.providers[provider_type][provider_name]
            
            # Skip disabled providers
            if not provider_info['enabled']:
                continue
            
            circuit_breaker = self.circuit_breakers[f"{provider_type}_{provider_name}"]
            
            # Skip if circuit breaker is open and not ready for retry
            if circuit_breaker.state == CircuitState.OPEN and not circuit_breaker._should_attempt_reset():
                logger.debug(f"Skipping {provider_name} - circuit breaker open")
                continue
            
            try:
                # Get method from provider
                provider_instance = provider_info['instance']
                method = getattr(provider_instance, method_name)
                
                # Execute with circuit breaker protection
                result = await circuit_breaker.call(method, *args, **kwargs)
                
                # Update success metrics
                provider_info['last_used'] = datetime.now()
                provider_info['error_count'] = 0
                
                logger.debug(f"Successfully executed {method_name} on {provider_name}")
                return result
                
            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker prevented call to {provider_name}: {e}")
                last_exception = e
                continue
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                provider_info['error_count'] += 1
                last_exception = e
                
                # Add exponential backoff delay
                if provider_info['error_count'] > 1:
                    delay = min(
                        self.retry_delay_base * (2 ** (provider_info['error_count'] - 1)),
                        self.retry_delay_max
                    )
                    await asyncio.sleep(delay)
                
                continue
        
        # All providers failed
        raise Exception(f"All providers failed for {provider_type}.{method_name}. Last error: {last_exception}")
    
    def enable_provider(self, provider_type: str, provider_name: str):
        """Enable a provider"""
        if provider_type in self.providers and provider_name in self.providers[provider_type]:
            self.providers[provider_type][provider_name]['enabled'] = True
            logger.info(f"Enabled provider {provider_name} for {provider_type}")
    
    def disable_provider(self, provider_type: str, provider_name: str):
        """Disable a provider"""
        if provider_type in self.providers and provider_name in self.providers[provider_type]:
            self.providers[provider_type][provider_name]['enabled'] = False
            logger.info(f"Disabled provider {provider_name} for {provider_type}")
    
    def reset_circuit_breaker(self, provider_type: str, provider_name: str):
        """Reset circuit breaker for a provider"""
        cb_key = f"{provider_type}_{provider_name}"
        if cb_key in self.circuit_breakers:
            self.circuit_breakers[cb_key].reset()
            logger.info(f"Reset circuit breaker for {provider_name}")
    
    def get_provider_status(self, provider_type: str = None) -> Dict[str, Any]:
        """Get status of all providers or specific type"""
        status = {}
        
        provider_types = [provider_type] if provider_type else self.providers.keys()
        
        for ptype in provider_types:
            if ptype not in self.providers:
                continue
                
            status[ptype] = {}
            
            for provider_name, provider_info in self.providers[ptype].items():
                circuit_breaker = self.circuit_breakers[f"{ptype}_{provider_name}"]
                
                status[ptype][provider_name] = {
                    'enabled': provider_info['enabled'],
                    'priority': provider_info['priority'],
                    'last_used': provider_info['last_used'].isoformat() if provider_info['last_used'] else None,
                    'error_count': provider_info['error_count'],
                    'circuit_breaker': circuit_breaker.get_stats()
                }
        
        return status
    
    async def health_check_all(self) -> Dict[str, Dict[str, bool]]:
        """Perform health check on all providers"""
        results = {}
        
        for provider_type, providers in self.providers.items():
            results[provider_type] = {}
            
            for provider_name, provider_info in providers.items():
                if not provider_info['enabled']:
                    results[provider_type][provider_name] = False
                    continue
                
                try:
                    provider_instance = provider_info['instance']
                    
                    # Try to call health_check method if it exists
                    if hasattr(provider_instance, 'health_check'):
                        healthy = await provider_instance.health_check()
                        results[provider_type][provider_name] = healthy
                    else:
                        # Assume healthy if no health_check method
                        results[provider_type][provider_name] = True
                        
                except Exception as e:
                    logger.error(f"Health check failed for {provider_name}: {e}")
                    results[provider_type][provider_name] = False
        
        return results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive failover metrics"""
        metrics = {
            'provider_types': list(self.providers.keys()),
            'total_providers': sum(len(providers) for providers in self.providers.values()),
            'circuit_breakers': {},
            'provider_summary': {}
        }
        
        # Circuit breaker metrics
        for cb_name, circuit_breaker in self.circuit_breakers.items():
            metrics['circuit_breakers'][cb_name] = circuit_breaker.get_stats()
        
        # Provider summary
        for provider_type, providers in self.providers.items():
            enabled_count = sum(1 for p in providers.values() if p['enabled'])
            
            metrics['provider_summary'][provider_type] = {
                'total': len(providers),
                'enabled': enabled_count,
                'disabled': len(providers) - enabled_count,
                'providers': list(providers.keys())
            }
        
        return metrics