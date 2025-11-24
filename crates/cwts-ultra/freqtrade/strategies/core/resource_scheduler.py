#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum Resource Scheduler
-------------------------

An advanced resource scheduler and orchestrator for quantum computing tasks.
Manages GPU and CPU resources to optimize performance and prevent resource conflicts
between different quantum components.

Features:
- Priority-based scheduling for critical operations
- Resource reservation and release management
- Timeout handling for long-running quantum operations
- Fallback mechanisms when resources are constrained
- Dynamic performance profiling
"""

import os
import time
import logging
import threading
import contextlib
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Callable, Any, Set
from dataclasses import dataclass
from collections import deque
import queue
import signal
import weakref

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Resource priority levels
class ResourcePriority(Enum):
    CRITICAL = 0  # Essential operations, e.g., final trading decisions
    HIGH = 1      # Important computations, e.g., forecasting
    MEDIUM = 2    # Standard operations, e.g., feature extraction
    LOW = 3       # Background tasks, e.g., model updates
    BACKGROUND = 4  # Non-essential tasks, can be delayed indefinitely

# Resource types
class ResourceType(Enum):
    QUANTUM_DEVICE = auto()
    GPU = auto()
    CPU = auto()
    MEMORY = auto()
    SPECIALIZED_HARDWARE = auto()  # For specific quantum chips

@dataclass
class ResourceRequest:
    """Represents a request for computing resources"""
    component_id: str
    resource_type: ResourceType
    priority: ResourcePriority
    duration_estimate: float  # In seconds
    callback: Optional[Callable] = None
    timeout: float = 30.0  # Default timeout in seconds
    fallback_fn: Optional[Callable] = None
    created_at: float = time.time()
    resource_amount: float = 1.0  # Proportion of resource (0.0-1.0)
    
    def __post_init__(self):
        self.id = f"{self.component_id}_{int(time.time()*1000)}"

@dataclass
class ResourceAllocation:
    """Represents an active resource allocation"""
    request: ResourceRequest
    resource_type: ResourceType
    allocated_at: float = time.time()
    resource_amount: float = 1.0
    device_id: Optional[str] = None
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.allocated_at
    
    @property
    def timeout_exceeded(self) -> bool:
        return self.elapsed_time > self.request.timeout

class QuantumResourceScheduler:
    """
    Manages resources for quantum computing tasks with priority-based scheduling.
    Handles resource allocation, monitoring, and release.
    """
    
    def __init__(self, 
                 max_concurrent_quantum_tasks: int = 1,
                 max_concurrent_gpu_tasks: int = 2,
                 max_cpu_threads: int = None,
                 enable_profiling: bool = True,
                 hardware_manager = None):
        """
        Initialize the quantum resource scheduler.
        
        Args:
            max_concurrent_quantum_tasks: Maximum quantum tasks allowed simultaneously
            max_concurrent_gpu_tasks: Maximum GPU tasks allowed simultaneously
            max_cpu_threads: Maximum CPU threads to use (None = auto)
            enable_profiling: Whether to enable performance profiling
            hardware_manager: Optional hardware manager instance for resource discovery
        """
        self.logger = logging.getLogger("QuantumResourceScheduler")
        
        # Resource limits
        self.max_concurrent_quantum_tasks = max_concurrent_quantum_tasks
        self.max_concurrent_gpu_tasks = max_concurrent_gpu_tasks
        self.max_cpu_threads = max_cpu_threads or os.cpu_count()
        
        # Resource tracking
        self._active_allocations: Dict[str, ResourceAllocation] = {}
        self._request_queue: Dict[ResourceType, List[ResourceRequest]] = {
            resource_type: [] for resource_type in ResourceType
        }
        
        # Resource availability
        self._available_resources: Dict[ResourceType, float] = {
            ResourceType.QUANTUM_DEVICE: float(max_concurrent_quantum_tasks),
            ResourceType.GPU: float(max_concurrent_gpu_tasks),
            ResourceType.CPU: float(self.max_cpu_threads),
            ResourceType.MEMORY: 0.9,  # Start with 90% memory available
            ResourceType.SPECIALIZED_HARDWARE: 0.0,  # None by default
        }
        
        # Performance tracking
        self.enable_profiling = enable_profiling
        self._execution_times: Dict[str, deque] = {}
        self._component_performance: Dict[str, Dict[str, float]] = {}
        
        # Internal state
        self._lock = threading.RLock()
        self._scheduler_thread = None
        self._running = False
        self._event_queue = queue.Queue()
        
        # Hardware manager integration
        self.hardware_manager = hardware_manager
        if hardware_manager:
            self._update_resources_from_hw_manager()
            
        # Timeout handler
        self._timeout_thread = None
        
        # Register components
        self.registered_components: Set[str] = set()
        
        self.logger.info("Quantum Resource Scheduler initialized")
        
    def _update_resources_from_hw_manager(self):
        """Update available resources based on hardware manager information"""
        if not self.hardware_manager:
            return
        
        # Update GPU availability
        if hasattr(self.hardware_manager, 'num_gpus'):
            self._available_resources[ResourceType.GPU] = float(self.hardware_manager.num_gpus)
            
        # Update CPU availability
        if hasattr(self.hardware_manager, 'num_cpu_cores'):
            self._available_resources[ResourceType.CPU] = float(self.hardware_manager.num_cpu_cores)
            
        # Update memory availability
        if hasattr(self.hardware_manager, 'available_memory_gb'):
            total_memory = self.hardware_manager.total_memory_gb
            if total_memory > 0:
                self._available_resources[ResourceType.MEMORY] = self.hardware_manager.available_memory_gb / total_memory
                
        # Check for specialized quantum hardware
        if hasattr(self.hardware_manager, 'has_quantum_hardware') and self.hardware_manager.has_quantum_hardware:
            self._available_resources[ResourceType.SPECIALIZED_HARDWARE] = 1.0
            
        self.logger.info(f"Updated resources from hardware manager: {self._available_resources}")
        
    def start(self):
        """Start the scheduler threads"""
        with self._lock:
            if self._running:
                return
                
            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()
            
            self._timeout_thread = threading.Thread(target=self._monitor_timeouts, daemon=True)
            self._timeout_thread.start()
            
            self.logger.info("Resource scheduler started")
            
    def stop(self):
        """Stop the scheduler threads"""
        with self._lock:
            if not self._running:
                return
                
            self._running = False
            
            # Wait for threads to finish
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=2.0)
                
            if self._timeout_thread:
                self._timeout_thread.join(timeout=2.0)
                
            # Release all allocations
            for allocation_id in list(self._active_allocations.keys()):
                self.release_resources(allocation_id)
                
            self.logger.info("Resource scheduler stopped")
            
    def register_component(self, component_id: str, resource_profiles: Dict[str, Dict] = None):
        """
        Register a component with the scheduler.
        
        Args:
            component_id: Unique ID for the component
            resource_profiles: Dict of operation types and their resource profiles
        """
        with self._lock:
            self.registered_components.add(component_id)
            
            # Initialize performance tracking
            if self.enable_profiling:
                self._execution_times[component_id] = deque(maxlen=100)
                self._component_performance[component_id] = {
                    'avg_execution_time': 0.0,
                    'max_execution_time': 0.0,
                    'success_rate': 1.0,
                    'timeout_rate': 0.0
                }
                
            self.logger.debug(f"Component registered: {component_id}")
            
    def unregister_component(self, component_id: str):
        """Unregister a component from the scheduler"""
        with self._lock:
            if component_id in self.registered_components:
                self.registered_components.remove(component_id)
                
            # Clean up performance tracking
            if component_id in self._execution_times:
                del self._execution_times[component_id]
                
            if component_id in self._component_performance:
                del self._component_performance[component_id]
                
            # Release any active allocations
            allocations_to_release = [
                alloc_id for alloc_id, alloc in self._active_allocations.items()
                if alloc.request.component_id == component_id
            ]
            
            for alloc_id in allocations_to_release:
                self.release_resources(alloc_id)
                
            self.logger.debug(f"Component unregistered: {component_id}")
            
    def request_resources(self, component_id: str, resource_type: ResourceType, 
                         priority: ResourcePriority = ResourcePriority.MEDIUM,
                         duration_estimate: float = 10.0,
                         timeout: float = 30.0,
                         resource_amount: float = 1.0,
                         callback: Optional[Callable] = None,
                         fallback_fn: Optional[Callable] = None) -> str:
        """
        Request resources for a component operation.
        
        Args:
            component_id: ID of the component requesting resources
            resource_type: Type of resource being requested
            priority: Priority level of the request
            duration_estimate: Estimated duration of resource usage in seconds
            timeout: Maximum time to wait for resource allocation in seconds
            resource_amount: Proportion of resource needed (0.0-1.0)
            callback: Function to call when resources are allocated
            fallback_fn: Function to call if resource request times out
            
        Returns:
            Allocation ID that can be used to release the resources
        """
        # Create resource request
        request = ResourceRequest(
            component_id=component_id,
            resource_type=resource_type,
            priority=priority,
            duration_estimate=duration_estimate,
            callback=callback,
            timeout=timeout,
            fallback_fn=fallback_fn,
            resource_amount=resource_amount
        )
        
        with self._lock:
            # Check if component is registered
            if component_id not in self.registered_components:
                self.register_component(component_id)
                
            # Add to appropriate queue based on priority
            queue = self._request_queue[resource_type]
            
            # Insert request in priority order
            insertion_index = len(queue)
            for i, existing_request in enumerate(queue):
                if request.priority.value < existing_request.priority.value:
                    insertion_index = i
                    break
                    
            queue.insert(insertion_index, request)
            
            # Signal the scheduler that a new request has been added
            self._event_queue.put(("new_request", request.id))
            
            self.logger.debug(f"Resource request queued: {request.id} for {component_id} "
                             f"(type={resource_type.name}, priority={priority.name})")
            
            return request.id
            
    def check_allocation_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of a resource allocation request.
        
        Args:
            request_id: ID of the resource request
            
        Returns:
            Dictionary with status information
        """
        with self._lock:
            # Check if already allocated
            if request_id in self._active_allocations:
                allocation = self._active_allocations[request_id]
                return {
                    'status': 'allocated',
                    'allocation_id': request_id,
                    'resource_type': allocation.resource_type.name,
                    'allocated_at': allocation.allocated_at,
                    'elapsed_time': allocation.elapsed_time,
                    'device_id': allocation.device_id
                }
                
            # Check queues
            for resource_type, queue in self._request_queue.items():
                for i, request in enumerate(queue):
                    if request.id == request_id:
                        return {
                            'status': 'queued',
                            'queue_position': i,
                            'resource_type': resource_type.name,
                            'priority': request.priority.name,
                            'waiting_time': time.time() - request.created_at
                        }
                        
            # Not found
            return {
                'status': 'unknown',
                'request_id': request_id
            }
    
    def release_resources(self, allocation_id: str):
        """
        Release allocated resources.
        
        Args:
            allocation_id: ID of the allocation to release
        """
        with self._lock:
            if allocation_id not in self._active_allocations:
                self.logger.warning(f"Attempted to release non-existent allocation: {allocation_id}")
                return
                
            allocation = self._active_allocations[allocation_id]
            
            # Update performance metrics
            if self.enable_profiling:
                component_id = allocation.request.component_id
                if component_id in self._execution_times:
                    execution_time = allocation.elapsed_time
                    self._execution_times[component_id].append(execution_time)
                    
                    # Update performance metrics
                    times = list(self._execution_times[component_id])
                    if times:
                        self._component_performance[component_id]['avg_execution_time'] = sum(times) / len(times)
                        self._component_performance[component_id]['max_execution_time'] = max(times)
                        
            # Return resources to available pool
            resource_type = allocation.resource_type
            self._available_resources[resource_type] += allocation.resource_amount
            
            # Remove from active allocations
            del self._active_allocations[allocation_id]
            
            # Signal the scheduler
            self._event_queue.put(("resource_released", resource_type))
            
            self.logger.debug(f"Resources released: {allocation_id}")
            
    def get_performance_metrics(self, component_id: Optional[str] = None) -> Dict:
        """
        Get performance metrics for components.
        
        Args:
            component_id: Optional specific component to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        with self._lock:
            if not self.enable_profiling:
                return {"profiling_enabled": False}
                
            if component_id:
                if component_id in self._component_performance:
                    return {component_id: self._component_performance[component_id]}
                return {"component_not_found": component_id}
                
            return {
                "component_metrics": self._component_performance,
                "resource_utilization": {
                    resource_type.name: 1.0 - available_amount / self._get_total_resource(resource_type)
                    for resource_type, available_amount in self._available_resources.items()
                },
                "queue_lengths": {
                    resource_type.name: len(queue)
                    for resource_type, queue in self._request_queue.items()
                }
            }
            
    def _get_total_resource(self, resource_type: ResourceType) -> float:
        """Get the total amount of a specific resource"""
        if resource_type == ResourceType.QUANTUM_DEVICE:
            return float(self.max_concurrent_quantum_tasks)
        elif resource_type == ResourceType.GPU:
            return float(self.max_concurrent_gpu_tasks)
        elif resource_type == ResourceType.CPU:
            return float(self.max_cpu_threads)
        elif resource_type == ResourceType.MEMORY:
            return 1.0  # Full memory is 1.0 (100%)
        elif resource_type == ResourceType.SPECIALIZED_HARDWARE:
            # This might depend on specific quantum hardware
            return 1.0 if self._available_resources[resource_type] > 0 else 0.0
            
    def _scheduler_loop(self):
        """Main scheduler loop to process resource requests"""
        while self._running:
            try:
                # Process any events
                try:
                    event_type, event_data = self._event_queue.get(timeout=1.0)
                    self._event_queue.task_done()
                except queue.Empty:
                    # No events, continue with regular scheduling
                    pass
                    
                # Process resource allocations
                with self._lock:
                    for resource_type in ResourceType:
                        self._process_resource_queue(resource_type)
                        
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                
    def _process_resource_queue(self, resource_type: ResourceType):
        """Process the queue for a specific resource type"""
        queue = self._request_queue[resource_type]
        available_amount = self._available_resources[resource_type]
        
        # Nothing to process
        if not queue or available_amount <= 0:
            return
            
        # Try to allocate resources to requests in priority order
        requests_to_remove = []
        
        for request in queue:
            # Check if we have enough resources
            if request.resource_amount <= available_amount:
                # Allocate resources
                allocation = ResourceAllocation(
                    request=request,
                    resource_type=resource_type,
                    resource_amount=request.resource_amount,
                    device_id=self._get_device_id(resource_type)
                )
                
                # Update available resources
                available_amount -= request.resource_amount
                self._available_resources[resource_type] = available_amount
                
                # Add to active allocations
                self._active_allocations[request.id] = allocation
                
                # Mark request for removal from queue
                requests_to_remove.append(request)
                
                # Call the callback if provided
                if request.callback:
                    try:
                        request.callback(request.id)
                    except Exception as e:
                        self.logger.error(f"Error in resource allocation callback: {e}", exc_info=True)
                        
                self.logger.debug(f"Resources allocated: {request.id} for {request.component_id}")
                
                # Stop if we're out of resources
                if available_amount <= 0:
                    break
                    
        # Remove processed requests from queue
        for request in requests_to_remove:
            queue.remove(request)
            
    def _get_device_id(self, resource_type: ResourceType) -> Optional[str]:
        """Get a device ID for a specific resource type"""
        if self.hardware_manager:
            if resource_type == ResourceType.GPU and hasattr(self.hardware_manager, 'get_available_gpu'):
                return self.hardware_manager.get_available_gpu()
            elif resource_type == ResourceType.QUANTUM_DEVICE and hasattr(self.hardware_manager, 'get_quantum_device'):
                return self.hardware_manager.get_quantum_device()
        return None
        
    def _monitor_timeouts(self):
        """Monitor allocations for timeouts"""
        while self._running:
            try:
                time.sleep(1.0)  # Check every second
                
                with self._lock:
                    current_time = time.time()
                    
                    # Check active allocations for timeouts
                    timed_out_allocations = []
                    for allocation_id, allocation in self._active_allocations.items():
                        if allocation.elapsed_time > allocation.request.timeout:
                            timed_out_allocations.append(allocation_id)
                            
                    # Handle timed out allocations
                    for allocation_id in timed_out_allocations:
                        allocation = self._active_allocations[allocation_id]
                        
                        # Update performance metrics
                        if self.enable_profiling:
                            component_id = allocation.request.component_id
                            if component_id in self._component_performance:
                                self._component_performance[component_id]['timeout_rate'] = (
                                    self._component_performance[component_id].get('timeout_rate', 0) * 0.9 + 0.1
                                )
                                
                        # Call fallback function if provided
                        if allocation.request.fallback_fn:
                            try:
                                allocation.request.fallback_fn(allocation_id)
                            except Exception as e:
                                self.logger.error(f"Error in timeout fallback function: {e}", exc_info=True)
                                
                        # Release resources
                        self.release_resources(allocation_id)
                        
                        self.logger.warning(f"Resource allocation timed out: {allocation_id} "
                                          f"for {allocation.request.component_id}")
                                          
                    # Check queued requests for timeouts
                    for resource_type, queue in self._request_queue.items():
                        timed_out_requests = []
                        for request in queue:
                            if current_time - request.created_at > request.timeout:
                                timed_out_requests.append(request)
                                
                        # Handle timed out requests
                        for request in timed_out_requests:
                            queue.remove(request)
                            
                            # Call fallback function if provided
                            if request.fallback_fn:
                                try:
                                    request.fallback_fn(request.id)
                                except Exception as e:
                                    self.logger.error(f"Error in queue timeout fallback: {e}", exc_info=True)
                                    
                            # Update performance metrics
                            if self.enable_profiling and request.component_id in self._component_performance:
                                self._component_performance[request.component_id]['timeout_rate'] = (
                                    self._component_performance[request.component_id].get('timeout_rate', 0) * 0.9 + 0.1
                                )
                                
                            self.logger.warning(f"Resource request timed out in queue: {request.id} "
                                              f"for {request.component_id}")
                                              
            except Exception as e:
                self.logger.error(f"Error in timeout monitor: {e}", exc_info=True)
                
    @contextlib.contextmanager
    def acquire(self, component_id: str, resource_type: ResourceType,
               priority: ResourcePriority = ResourcePriority.MEDIUM,
               timeout: float = 30.0,
               resource_amount: float = 1.0,
               description: str = ""):
        """
        Context manager for acquiring resources.
        
        Example:
            with scheduler.acquire("my_component", ResourceType.GPU) as allocation_id:
                # Do GPU work here
                # Resources are automatically released when exiting the context
        """
        allocation_id = None
        resource_acquired = threading.Event()
        
        def on_resource_acquired(alloc_id):
            nonlocal allocation_id
            allocation_id = alloc_id
            resource_acquired.set()
            
        # Request the resource
        req_id = self.request_resources(
            component_id=component_id,
            resource_type=resource_type,
            priority=priority,
            timeout=timeout,
            resource_amount=resource_amount,
            callback=on_resource_acquired
        )
        
        try:
            # Wait for resource to be acquired
            acquired = resource_acquired.wait(timeout=timeout)
            if not acquired:
                raise TimeoutError(f"Timeout waiting for {resource_type.name} resources")
                
            self.logger.debug(f"Acquired {resource_type.name} resources for {component_id}: {description}")
            
            # Yield the allocation ID
            yield allocation_id
            
        finally:
            # Always release resources, even if an exception occurred
            if allocation_id:
                self.release_resources(allocation_id)
                self.logger.debug(f"Released {resource_type.name} resources for {component_id}: {description}")