"""
Resource-aware execution management for quantum components.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceRequest:
    """Resource request from QBMIA component."""
    request_id: str
    component: str
    operation: str
    resource_requirements: Dict[str, Any]
    priority: str = 'normal'  # 'low', 'normal', 'high', 'critical'
    timeout: float = 60.0  # seconds
    callback: Optional[Callable] = None

@dataclass
class ResourceAllocation:
    """Allocated resources for a request."""
    allocation_id: str
    request_id: str
    resources: Dict[str, Any]
    allocated_at: datetime
    expires_at: datetime

class ResourceManager:
    """
    Manages computational resources and coordinates with orchestrator.
    """

    def __init__(self, agent_id: str, hw_optimizer: Any):
        """
        Initialize resource manager.

        Args:
            agent_id: QBMIA agent identifier
            hw_optimizer: Hardware optimizer instance
        """
        self.agent_id = agent_id
        self.hw_optimizer = hw_optimizer

        # Resource tracking
        self.available_resources = self._init_resources()
        self.allocated_resources = {}
        self.pending_requests = asyncio.Queue()
        self.active_allocations = {}

        # Orchestrator interface
        self.orchestrator_connected = False
        self.orchestrator_endpoint = None

        # Resource monitoring
        self.monitor_interval = 1.0  # seconds
        self.resource_history = []
        self.max_history = 1000

        # Locks
        self.resource_lock = asyncio.Lock()

        # Start monitoring
        self.monitoring_task = None
        self.allocation_task = None
        self.running = True

    def _init_resources(self) -> Dict[str, Any]:
        """Initialize available resources based on hardware."""
        resources = {
            'cpu': {
                'cores': psutil.cpu_count(),
                'usage_percent': 0.0,
                'available': True
            },
            'memory': {
                'total_mb': psutil.virtual_memory().total // (1024 * 1024),
                'available_mb': psutil.virtual_memory().available // (1024 * 1024),
                'usage_percent': 0.0
            },
            'gpu': []
        }

        # Add GPU resources
        device_info = self.hw_optimizer.get_device_info()
        if 'cuda' in device_info or 'rocm' in device_info:
            # Get GPU memory info
            gpu_memory = self.hw_optimizer.get_memory_usage()

            resources['gpu'].append({
                'index': 0,
                'type': device_info.get('type', 'unknown'),
                'total_memory_mb': gpu_memory.get('total', 0) // (1024 * 1024),
                'available_memory_mb': gpu_memory.get('available', 0) // (1024 * 1024),
                'usage_percent': gpu_memory.get('usage_percent', 0.0)
            })

        return resources

    async def start(self):
        """Start resource manager tasks."""
        self.monitoring_task = asyncio.create_task(self._monitor_resources())
        self.allocation_task = asyncio.create_task(self._process_allocations())
        logger.info(f"Resource manager started for agent {self.agent_id}")

    async def stop(self):
        """Stop resource manager tasks."""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.allocation_task:
            self.allocation_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            self.monitoring_task,
            self.allocation_task,
            return_exceptions=True
        )

        logger.info("Resource manager stopped")

    async def _monitor_resources(self):
        """Monitor resource usage continuously."""
        while self.running:
            try:
                # Update CPU usage
                self.available_resources['cpu']['usage_percent'] = psutil.cpu_percent()

                # Update memory usage
                mem = psutil.virtual_memory()
                self.available_resources['memory']['available_mb'] = mem.available // (1024 * 1024)
                self.available_resources['memory']['usage_percent'] = mem.percent

                # Update GPU usage if available
                if self.available_resources['gpu']:
                    gpu_memory = self.hw_optimizer.get_memory_usage()
                    for i, gpu in enumerate(self.available_resources['gpu']):
                        gpu['available_memory_mb'] = gpu_memory.get('available', 0) // (1024 * 1024)
                        gpu['usage_percent'] = gpu_memory.get('usage_percent', 0.0)

                # Record history
                self._record_resource_snapshot()

                # Check for resource alerts
                await self._check_resource_alerts()

                await asyncio.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitor_interval)

    async def _process_allocations(self):
        """Process resource allocation requests."""
        while self.running:
            try:
                # Get pending request
                request = await asyncio.wait_for(
                    self.pending_requests.get(),
                    timeout=1.0
                )

                # Try to allocate resources
                allocation = await self._try_allocate(request)

                if allocation:
                    # Success - notify requester
                    if request.callback:
                        await request.callback(allocation)
                else:
                    # Failed - requeue or reject based on priority
                    if request.priority in ['high', 'critical']:
                        await self.pending_requests.put(request)
                    else:
                        logger.warning(f"Resource allocation failed for {request.request_id}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Allocation processing error: {e}")

    async def request_resources(self, requirements: Dict[str, Any]) -> ResourceAllocation:
        """
        Request resources for an operation.

        Args:
            requirements: Resource requirements

        Returns:
            Resource allocation
        """
        # Create request
        request = ResourceRequest(
            request_id=f"{self.agent_id}_{time.time()}",
            component=self.agent_id,
            operation=requirements.get('operation', 'unknown'),
            resource_requirements=requirements,
            priority=requirements.get('priority', 'normal'),
            timeout=requirements.get('timeout', 60.0)
        )

        # Try immediate allocation for critical requests
        if request.priority == 'critical':
            allocation = await self._try_allocate(request)
            if allocation:
                return allocation

        # Queue request
        await self.pending_requests.put(request)

        # Wait for allocation
        start_time = time.time()
        while time.time() - start_time < request.timeout:
            if request.request_id in self.active_allocations:
                return self.active_allocations[request.request_id]
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Resource allocation timeout for {request.request_id}")

    async def _try_allocate(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Try to allocate resources for a request."""
        async with self.resource_lock:
            required = request.resource_requirements

            # Check CPU requirements
            if 'cpu_cores' in required:
                if self.available_resources['cpu']['usage_percent'] > 80:
                    return None

            # Check memory requirements
            if 'memory_mb' in required:
                if self.available_resources['memory']['available_mb'] < required['memory_mb']:
                    return None

            # Check GPU requirements
            if 'gpu_memory_mb' in required and self.available_resources['gpu']:
                gpu_available = False
                for gpu in self.available_resources['gpu']:
                    if gpu['available_memory_mb'] >= required['gpu_memory_mb']:
                        gpu_available = True
                        break
                if not gpu_available:
                    return None

            # Create allocation
            allocation = ResourceAllocation(
                allocation_id=f"alloc_{request.request_id}",
                request_id=request.request_id,
                resources={
                    'cpu_cores': required.get('cpu_cores', 1),
                    'memory_mb': required.get('memory_mb', 1024),
                    'gpu_memory_mb': required.get('gpu_memory_mb', 0),
                    'gpu_index': 0 if required.get('gpu_memory_mb', 0) > 0 else None
                },
                allocated_at=datetime.utcnow(),
                expires_at=datetime.utcnow()
            )

            # Track allocation
            self.active_allocations[request.request_id] = allocation

            # Update available resources
            if 'memory_mb' in required:
                self.available_resources['memory']['available_mb'] -= required['memory_mb']

            if 'gpu_memory_mb' in required and self.available_resources['gpu']:
                self.available_resources['gpu'][0]['available_memory_mb'] -= required['gpu_memory_mb']

            logger.info(f"Allocated resources for {request.request_id}")
            return allocation

    async def release_resources(self, allocation: ResourceAllocation):
        """Release allocated resources."""
        async with self.resource_lock:
            if allocation.request_id in self.active_allocations:
                # Return resources
                resources = allocation.resources

                if 'memory_mb' in resources:
                    self.available_resources['memory']['available_mb'] += resources['memory_mb']

                if 'gpu_memory_mb' in resources and resources['gpu_memory_mb'] > 0:
                    if self.available_resources['gpu']:
                        self.available_resources['gpu'][0]['available_memory_mb'] += resources['gpu_memory_mb']

                # Remove allocation
                del self.active_allocations[allocation.request_id]

                logger.info(f"Released resources for {allocation.request_id}")

    def _record_resource_snapshot(self):
        """Record current resource state."""
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage': self.available_resources['cpu']['usage_percent'],
            'memory_available': self.available_resources['memory']['available_mb'],
            'memory_usage': self.available_resources['memory']['usage_percent']
        }

        if self.available_resources['gpu']:
            snapshot['gpu_memory_available'] = self.available_resources['gpu'][0]['available_memory_mb']
            snapshot['gpu_usage'] = self.available_resources['gpu'][0]['usage_percent']

        self.resource_history.append(snapshot)

        # Limit history size
        if len(self.resource_history) > self.max_history:
            self.resource_history = self.resource_history[-self.max_history:]

    async def _check_resource_alerts(self):
        """Check for resource constraint alerts."""
        alerts = []

        # CPU alert
        if self.available_resources['cpu']['usage_percent'] > 90:
            alerts.append({
                'type': 'cpu_high',
                'severity': 'warning',
                'message': f"CPU usage high: {self.available_resources['cpu']['usage_percent']:.1f}%"
            })

        # Memory alert
        if self.available_resources['memory']['usage_percent'] > 85:
            alerts.append({
                'type': 'memory_high',
                'severity': 'warning',
                'message': f"Memory usage high: {self.available_resources['memory']['usage_percent']:.1f}%"
            })

        # GPU memory alert
        if self.available_resources['gpu']:
            for gpu in self.available_resources['gpu']:
                if gpu['usage_percent'] > 85:
                    alerts.append({
                        'type': 'gpu_memory_high',
                        'severity': 'warning',
                        'message': f"GPU memory usage high: {gpu['usage_percent']:.1f}%"
                    })

        # Send alerts to orchestrator if connected
        if alerts and self.orchestrator_connected:
            await self._send_alerts_to_orchestrator(alerts)

    async def _send_alerts_to_orchestrator(self, alerts: List[Dict[str, Any]]):
        """Send resource alerts to orchestrator."""
        # Implementation depends on orchestrator protocol
        pass

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource summary."""
        return {
            'agent_id': self.agent_id,
            'resources': self.available_resources,
            'active_allocations': len(self.active_allocations),
            'pending_requests': self.pending_requests.qsize(),
            'resource_history': self.resource_history[-10:]  # Last 10 snapshots
        }

    def cleanup(self):
        """Clean up resource manager."""
        self.running = False
        logger.info("Resource manager cleanup complete")
