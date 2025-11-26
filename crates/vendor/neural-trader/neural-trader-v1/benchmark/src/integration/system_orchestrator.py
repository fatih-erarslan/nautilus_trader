"""
System Orchestrator - Main system coordinator for benchmark integration.

This module orchestrates all benchmark components including CLI, simulation,
real-time data, and optimization to ensure seamless operation.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from contextlib import asynccontextmanager

from .component_registry import ComponentRegistry, ComponentStatus
from .data_pipeline import DataPipeline
from .performance_monitor import PerformanceMonitor
from ..cli.config import Config
from ..data.realtime_manager import RealtimeManager
from ..simulation.simulator import Simulator
from ..optimization.optimizer import Optimizer


class SystemState(Enum):
    """System operational states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    processed_events: int = 0
    error_count: int = 0
    uptime: float = 0.0
    component_status: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class SystemOrchestrator:
    """
    Main system coordinator that orchestrates all benchmark components.
    
    Responsibilities:
    - Component lifecycle management
    - Resource coordination
    - Error handling and recovery
    - Performance monitoring
    - System health management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path) if config_path else Config()
        self.logger = self._setup_logging()
        
        # System state
        self.state = SystemState.STOPPED
        self.start_time: Optional[float] = None
        self.metrics = SystemMetrics()
        
        # Core components
        self.component_registry = ComponentRegistry()
        self.data_pipeline = DataPipeline(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Component instances
        self.realtime_manager: Optional[RealtimeManager] = None
        self.simulator: Optional[Simulator] = None
        self.optimizer: Optional[Optimizer] = None
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.shutdown_event = asyncio.Event()
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        self.logger.info("System Orchestrator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system-wide logging."""
        logger = logging.getLogger('system_orchestrator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def start(self) -> bool:
        """
        Start the complete system with all components.
        
        Returns:
            bool: True if system started successfully
        """
        if self.state != SystemState.STOPPED:
            self.logger.warning(f"Cannot start system in state: {self.state}")
            return False
        
        self.state = SystemState.STARTING
        self.start_time = time.time()
        
        try:
            self.logger.info("Starting System Orchestrator...")
            
            # Start performance monitoring
            await self.performance_monitor.start()
            
            # Initialize components in dependency order
            await self._initialize_components()
            
            # Start data pipeline
            await self.data_pipeline.start()
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            
            self.state = SystemState.RUNNING
            self.logger.info("System Orchestrator started successfully")
            
            # Emit system started event
            await self._emit_event('system_started', {
                'timestamp': time.time(),
                'components': list(self.component_registry.get_all_components().keys())
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.state = SystemState.ERROR
            await self._emergency_shutdown()
            return False
    
    async def stop(self) -> bool:
        """
        Gracefully stop the complete system.
        
        Returns:
            bool: True if system stopped successfully
        """
        if self.state == SystemState.STOPPED:
            return True
        
        self.state = SystemState.STOPPING
        self.logger.info("Stopping System Orchestrator...")
        
        try:
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop components in reverse dependency order
            await self._shutdown_components()
            
            # Stop data pipeline
            await self.data_pipeline.stop()
            
            # Stop performance monitoring
            await self.performance_monitor.stop()
            
            self.state = SystemState.STOPPED
            self.logger.info("System Orchestrator stopped successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the complete system."""
        self.logger.info("Restarting system...")
        if await self.stop():
            await asyncio.sleep(2)  # Brief pause
            return await self.start()
        return False
    
    async def _initialize_components(self):
        """Initialize all system components."""
        self.logger.info("Initializing system components...")
        
        # Initialize real-time data manager
        self.realtime_manager = RealtimeManager(self.config)
        await self.component_registry.register_component(
            'realtime_manager', 
            self.realtime_manager,
            dependencies=[]
        )
        
        # Initialize simulator
        self.simulator = Simulator(self.config)
        await self.component_registry.register_component(
            'simulator',
            self.simulator,
            dependencies=['realtime_manager']
        )
        
        # Initialize optimizer
        self.optimizer = Optimizer(self.config)
        await self.component_registry.register_component(
            'optimizer',
            self.optimizer,
            dependencies=['simulator']
        )
        
        # Start components in dependency order
        await self.component_registry.start_all()
    
    async def _shutdown_components(self):
        """Shutdown all components in reverse dependency order."""
        self.logger.info("Shutting down system components...")
        await self.component_registry.stop_all()
    
    async def _emergency_shutdown(self):
        """Emergency shutdown procedure."""
        self.logger.error("Initiating emergency shutdown...")
        
        try:
            # Force stop all components
            await self.component_registry.emergency_stop()
            
            # Stop monitoring
            if self.performance_monitor:
                await self.performance_monitor.stop()
            
            # Clean up resources
            await self._cleanup_resources()
            
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown: {e}")
    
    async def _health_check_loop(self):
        """Continuous health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
    
    async def _perform_health_check(self):
        """Perform comprehensive system health check."""
        try:
            # Check component health
            component_health = await self.component_registry.health_check()
            
            # Update metrics
            self.metrics.component_status = {
                name: status.value for name, status in component_health.items()
            }
            
            # Check system resources
            perf_metrics = await self.performance_monitor.get_current_metrics()
            self.metrics.cpu_usage = perf_metrics.get('cpu_usage', 0)
            self.metrics.memory_usage = perf_metrics.get('memory_usage', 0)
            
            # Calculate uptime
            if self.start_time:
                self.metrics.uptime = time.time() - self.start_time
            
            # Check for critical issues
            critical_issues = []
            
            # High resource usage
            if self.metrics.cpu_usage > 90:
                critical_issues.append("High CPU usage")
            if self.metrics.memory_usage > 90:
                critical_issues.append("High memory usage")
            
            # Component failures
            failed_components = [
                name for name, status in component_health.items()
                if status == ComponentStatus.ERROR
            ]
            if failed_components:
                critical_issues.append(f"Failed components: {failed_components}")
            
            # Handle critical issues
            if critical_issues:
                await self._handle_critical_issues(critical_issues)
            
            self.logger.debug(f"Health check completed. Issues: {len(critical_issues)}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
    
    async def _handle_critical_issues(self, issues: List[str]):
        """Handle critical system issues."""
        self.logger.warning(f"Critical issues detected: {issues}")
        
        # Emit alert event
        await self._emit_event('critical_alert', {
            'issues': issues,
            'timestamp': time.time(),
            'metrics': self.metrics.__dict__
        })
        
        # Implement recovery strategies
        for issue in issues:
            if "High CPU usage" in issue:
                await self._handle_high_cpu()
            elif "High memory usage" in issue:
                await self._handle_high_memory()
            elif "Failed components" in issue:
                await self._handle_component_failures()
    
    async def _handle_high_cpu(self):
        """Handle high CPU usage."""
        self.logger.info("Implementing CPU usage mitigation...")
        # Reduce processing frequency temporarily
        await self.data_pipeline.reduce_processing_rate(0.5)
    
    async def _handle_high_memory(self):
        """Handle high memory usage."""
        self.logger.info("Implementing memory usage mitigation...")
        # Clear caches and reduce memory footprint
        if self.realtime_manager:
            await self.realtime_manager.clear_cache()
    
    async def _handle_component_failures(self):
        """Handle component failures."""
        self.logger.info("Attempting component recovery...")
        failed_components = await self.component_registry.get_failed_components()
        
        for component_name in failed_components:
            try:
                await self.component_registry.restart_component(component_name)
                self.logger.info(f"Successfully restarted {component_name}")
            except Exception as e:
                self.logger.error(f"Failed to restart {component_name}: {e}")
    
    async def _cleanup_resources(self):
        """Clean up system resources."""
        self.logger.info("Cleaning up system resources...")
        
        # Clear temporary files
        temp_dir = Path("/tmp/benchmark_temp")
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit system event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'state': self.state.value,
            'uptime': self.metrics.uptime,
            'metrics': self.metrics.__dict__,
            'components': self.component_registry.get_component_status(),
            'data_pipeline': self.data_pipeline.get_status(),
            'performance': self.performance_monitor.get_summary()
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self.metrics
    
    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for managed system execution."""
        try:
            if not await self.start():
                raise RuntimeError("Failed to start system")
            yield self
        finally:
            await self.stop()


# Singleton instance for global access
_orchestrator_instance: Optional[SystemOrchestrator] = None

def get_orchestrator(config_path: Optional[str] = None) -> SystemOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = SystemOrchestrator(config_path)
    return _orchestrator_instance