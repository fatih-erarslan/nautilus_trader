"""
Integration module for the AI News Trading benchmark system.

This module provides system-level integration components that orchestrate
all benchmark subsystems including CLI, simulation, real-time data,
optimization, and monitoring.
"""

from .system_orchestrator import SystemOrchestrator, SystemState, get_orchestrator
from .component_registry import ComponentRegistry, ComponentStatus, ComponentType, ComponentInterface
from .data_pipeline import DataPipeline, PipelineStage, DataType, DataPacket
from .performance_monitor import PerformanceMonitor, MetricType, AlertLevel, Alert

__all__ = [
    # System Orchestrator
    'SystemOrchestrator',
    'SystemState', 
    'get_orchestrator',
    
    # Component Registry
    'ComponentRegistry',
    'ComponentStatus',
    'ComponentType',
    'ComponentInterface',
    
    # Data Pipeline
    'DataPipeline',
    'PipelineStage',
    'DataType',
    'DataPacket',
    
    # Performance Monitor
    'PerformanceMonitor',
    'MetricType',
    'AlertLevel',
    'Alert'
]

# Version information
__version__ = '1.0.0'
__author__ = 'AI Trading System Team'
__description__ = 'System integration layer for AI News Trading benchmark'