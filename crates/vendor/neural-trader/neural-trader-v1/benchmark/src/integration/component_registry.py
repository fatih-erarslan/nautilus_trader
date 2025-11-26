"""
Component Registry - Manages all system components and their lifecycle.

This module provides centralized component management including registration,
lifecycle control, dependency resolution, and health monitoring.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ComponentStatus(Enum):
    """Component operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ComponentType(Enum):
    """Component types for classification."""
    DATA_SOURCE = "data_source"
    PROCESSOR = "processor"
    SIMULATOR = "simulator"
    OPTIMIZER = "optimizer"
    MONITOR = "monitor"
    SERVICE = "service"


@dataclass
class ComponentMetrics:
    """Component-specific metrics."""
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    restart_count: int = 0
    error_count: int = 0
    last_health_check: Optional[float] = None
    health_status: bool = True
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class ComponentInterface(ABC):
    """Abstract interface for all manageable components."""
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the component."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the component."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check component health."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get component status information."""
        pass


@dataclass
class ComponentInfo:
    """Component registration information."""
    name: str
    instance: Any
    component_type: ComponentType
    dependencies: List[str] = field(default_factory=list)
    status: ComponentStatus = ComponentStatus.STOPPED
    metrics: ComponentMetrics = field(default_factory=ComponentMetrics)
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority starts first


class ComponentRegistry:
    """
    Centralized registry for all system components.
    
    Features:
    - Component lifecycle management
    - Dependency resolution and ordering
    - Health monitoring
    - Automatic recovery
    - Performance tracking
    """
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.logger = self._setup_logging()
        self.startup_timeout = 30  # seconds
        self.shutdown_timeout = 15  # seconds
        self.health_check_timeout = 5  # seconds
        
        # Startup and shutdown sequences
        self._startup_order: List[str] = []
        self._shutdown_order: List[str] = []
        
        self.logger.info("Component Registry initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup component registry logging."""
        logger = logging.getLogger('component_registry')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def register_component(
        self,
        name: str,
        instance: Any,
        component_type: ComponentType = ComponentType.SERVICE,
        dependencies: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> bool:
        """
        Register a component with the registry.
        
        Args:
            name: Unique component name
            instance: Component instance
            component_type: Type of component
            dependencies: List of dependency component names
            config: Component configuration
            priority: Startup priority (higher = earlier)
        
        Returns:
            bool: True if registration successful
        """
        if name in self.components:
            self.logger.warning(f"Component {name} already registered")
            return False
        
        dependencies = dependencies or []
        config = config or {}
        
        # Validate dependencies exist
        for dep in dependencies:
            if dep not in self.components:
                self.logger.warning(f"Dependency {dep} not found for component {name}")
        
        component_info = ComponentInfo(
            name=name,
            instance=instance,
            component_type=component_type,
            dependencies=dependencies,
            config=config,
            priority=priority
        )
        
        self.components[name] = component_info
        self.logger.info(f"Registered component: {name} (type: {component_type.value})")
        
        # Recalculate startup/shutdown order
        self._calculate_startup_order()
        
        return True
    
    async def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the registry.
        
        Args:
            name: Component name to unregister
        
        Returns:
            bool: True if unregistration successful
        """
        if name not in self.components:
            self.logger.warning(f"Component {name} not found for unregistration")
            return False
        
        component = self.components[name]
        
        # Stop component if running
        if component.status == ComponentStatus.RUNNING:
            await self.stop_component(name)
        
        # Check for dependent components
        dependents = [
            comp_name for comp_name, comp_info in self.components.items()
            if name in comp_info.dependencies
        ]
        
        if dependents:
            self.logger.warning(f"Component {name} has dependents: {dependents}")
            return False
        
        del self.components[name]
        self._calculate_startup_order()
        
        self.logger.info(f"Unregistered component: {name}")
        return True
    
    def _calculate_startup_order(self):
        """Calculate optimal startup order based on dependencies and priorities."""
        # Topological sort with priority consideration
        in_degree = {name: 0 for name in self.components}
        
        # Calculate in-degrees (number of dependencies)
        for name, info in self.components.items():
            for dep in info.dependencies:
                if dep in in_degree:
                    in_degree[name] += 1
        
        # Priority queue for components with no dependencies
        available = []
        for name, degree in in_degree.items():
            if degree == 0:
                # Use negative priority for max-heap behavior
                available.append((-self.components[name].priority, name))
        
        available.sort()  # Sort by priority
        startup_order = []
        
        while available:
            _, current = available.pop(0)
            startup_order.append(current)
            
            # Reduce in-degree for dependent components
            for name, info in self.components.items():
                if current in info.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        available.append((-info.priority, name))
                        available.sort()
        
        self._startup_order = startup_order
        self._shutdown_order = startup_order[::-1]  # Reverse order for shutdown
        
        self.logger.debug(f"Calculated startup order: {self._startup_order}")
    
    async def start_component(self, name: str) -> bool:
        """
        Start a specific component.
        
        Args:
            name: Component name to start
        
        Returns:
            bool: True if start successful
        """
        if name not in self.components:
            self.logger.error(f"Component {name} not found")
            return False
        
        component = self.components[name]
        
        if component.status == ComponentStatus.RUNNING:
            self.logger.warning(f"Component {name} already running")
            return True
        
        if component.status == ComponentStatus.STARTING:
            self.logger.warning(f"Component {name} already starting")
            return False
        
        # Check if dependencies are running
        for dep in component.dependencies:
            if dep in self.components:
                dep_status = self.components[dep].status
                if dep_status != ComponentStatus.RUNNING:
                    self.logger.error(f"Dependency {dep} not running for {name}")
                    return False
        
        component.status = ComponentStatus.STARTING
        component.metrics.start_time = time.time()
        
        try:
            self.logger.info(f"Starting component: {name}")
            
            # Start component with timeout
            if hasattr(component.instance, 'start'):
                result = await asyncio.wait_for(
                    component.instance.start(),
                    timeout=self.startup_timeout
                )
            else:
                result = True  # Assume success for components without start method
            
            if result:
                component.status = ComponentStatus.RUNNING
                self.logger.info(f"Component {name} started successfully")
                return True
            else:
                component.status = ComponentStatus.ERROR
                component.metrics.error_count += 1
                self.logger.error(f"Component {name} failed to start")
                return False
                
        except asyncio.TimeoutError:
            component.status = ComponentStatus.ERROR
            component.metrics.error_count += 1
            self.logger.error(f"Component {name} start timeout")
            return False
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.metrics.error_count += 1
            self.logger.error(f"Component {name} start error: {e}")
            return False
    
    async def stop_component(self, name: str) -> bool:
        """
        Stop a specific component.
        
        Args:
            name: Component name to stop
        
        Returns:
            bool: True if stop successful
        """
        if name not in self.components:
            self.logger.error(f"Component {name} not found")
            return False
        
        component = self.components[name]
        
        if component.status == ComponentStatus.STOPPED:
            self.logger.warning(f"Component {name} already stopped")
            return True
        
        if component.status == ComponentStatus.STOPPING:
            self.logger.warning(f"Component {name} already stopping")
            return False
        
        component.status = ComponentStatus.STOPPING
        
        try:
            self.logger.info(f"Stopping component: {name}")
            
            # Stop component with timeout
            if hasattr(component.instance, 'stop'):
                result = await asyncio.wait_for(
                    component.instance.stop(),
                    timeout=self.shutdown_timeout
                )
            else:
                result = True  # Assume success for components without stop method
            
            component.status = ComponentStatus.STOPPED
            component.metrics.stop_time = time.time()
            
            if result:
                self.logger.info(f"Component {name} stopped successfully")
                return True
            else:
                self.logger.warning(f"Component {name} stop returned False")
                return False
                
        except asyncio.TimeoutError:
            component.status = ComponentStatus.ERROR
            component.metrics.error_count += 1
            self.logger.error(f"Component {name} stop timeout")
            return False
        except Exception as e:
            component.status = ComponentStatus.ERROR
            component.metrics.error_count += 1
            self.logger.error(f"Component {name} stop error: {e}")
            return False
    
    async def restart_component(self, name: str) -> bool:
        """
        Restart a specific component.
        
        Args:
            name: Component name to restart
        
        Returns:
            bool: True if restart successful
        """
        if name not in self.components:
            self.logger.error(f"Component {name} not found")
            return False
        
        component = self.components[name]
        component.metrics.restart_count += 1
        
        self.logger.info(f"Restarting component: {name}")
        
        # Stop first
        if component.status != ComponentStatus.STOPPED:
            if not await self.stop_component(name):
                return False
        
        # Brief pause before restart
        await asyncio.sleep(1)
        
        # Start again
        return await self.start_component(name)
    
    async def start_all(self) -> bool:
        """
        Start all registered components in dependency order.
        
        Returns:
            bool: True if all components started successfully
        """
        self.logger.info("Starting all components...")
        
        success_count = 0
        for name in self._startup_order:
            if await self.start_component(name):
                success_count += 1
            else:
                self.logger.error(f"Failed to start component {name}, aborting startup")
                # Stop all successfully started components
                await self._stop_started_components(self._startup_order[:success_count])
                return False
        
        self.logger.info(f"All {success_count} components started successfully")
        return True
    
    async def stop_all(self) -> bool:
        """
        Stop all registered components in reverse dependency order.
        
        Returns:
            bool: True if all components stopped successfully
        """
        self.logger.info("Stopping all components...")
        
        success_count = 0
        for name in self._shutdown_order:
            if self.components[name].status == ComponentStatus.RUNNING:
                if await self.stop_component(name):
                    success_count += 1
                else:
                    self.logger.warning(f"Failed to stop component {name}")
        
        self.logger.info(f"Stopped {success_count} components")
        return True
    
    async def emergency_stop(self):
        """Emergency stop all components without waiting."""
        self.logger.warning("Initiating emergency stop...")
        
        tasks = []
        for name, component in self.components.items():
            if component.status == ComponentStatus.RUNNING:
                if hasattr(component.instance, 'stop'):
                    task = asyncio.create_task(
                        asyncio.wait_for(
                            component.instance.stop(),
                            timeout=5  # Shorter timeout for emergency
                        )
                    )
                    tasks.append((name, task))
        
        # Wait for all stops to complete or timeout
        if tasks:
            done, pending = await asyncio.wait(
                [task for _, task in tasks],
                timeout=10,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
        
        # Mark all as stopped
        for component in self.components.values():
            component.status = ComponentStatus.STOPPED
        
        self.logger.warning("Emergency stop completed")
    
    async def _stop_started_components(self, component_names: List[str]):
        """Stop components that were successfully started."""
        for name in reversed(component_names):
            await self.stop_component(name)
    
    async def health_check(self) -> Dict[str, ComponentStatus]:
        """
        Perform health check on all components.
        
        Returns:
            Dict mapping component names to their health status
        """
        health_results = {}
        
        for name, component in self.components.items():
            try:
                if component.status == ComponentStatus.RUNNING:
                    if hasattr(component.instance, 'health_check'):
                        health_ok = await asyncio.wait_for(
                            component.instance.health_check(),
                            timeout=self.health_check_timeout
                        )
                        if not health_ok:
                            component.status = ComponentStatus.ERROR
                    
                    component.metrics.last_health_check = time.time()
                    component.metrics.health_status = True
                
                health_results[name] = component.status
                
            except Exception as e:
                self.logger.error(f"Health check failed for {name}: {e}")
                component.status = ComponentStatus.ERROR
                component.metrics.error_count += 1
                component.metrics.health_status = False
                health_results[name] = ComponentStatus.ERROR
        
        return health_results
    
    async def get_failed_components(self) -> List[str]:
        """Get list of components in error state."""
        return [
            name for name, component in self.components.items()
            if component.status == ComponentStatus.ERROR
        ]
    
    def get_all_components(self) -> Dict[str, ComponentInfo]:
        """Get all registered components."""
        return self.components.copy()
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all components."""
        status = {}
        for name, component in self.components.items():
            status[name] = {
                'status': component.status.value,
                'type': component.component_type.value,
                'dependencies': component.dependencies,
                'metrics': {
                    'start_time': component.metrics.start_time,
                    'restart_count': component.metrics.restart_count,
                    'error_count': component.metrics.error_count,
                    'last_health_check': component.metrics.last_health_check,
                    'health_status': component.metrics.health_status
                }
            }
        return status
    
    def get_startup_order(self) -> List[str]:
        """Get the calculated startup order."""
        return self._startup_order.copy()
    
    def get_shutdown_order(self) -> List[str]:
        """Get the calculated shutdown order."""
        return self._shutdown_order.copy()