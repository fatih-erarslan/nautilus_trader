#!/usr/bin/env python3
"""
Performance Validation Suite for AI News Trading Platform.

This package provides comprehensive validation of all performance targets
and trading requirements for production deployment.

Modules:
    performance_validator: Main validation orchestrator
    latency_validator: Signal generation latency validation  
    throughput_validator: System throughput validation
    resource_validator: Resource usage validation
    strategy_validator: Strategy performance validation
"""

from .performance_validator import PerformanceValidator, ValidationStatus, PerformanceTarget, ValidationResult, ValidationSummary
from .latency_validator import LatencyValidator
from .throughput_validator import ThroughputValidator
from .resource_validator import ResourceValidator
from .strategy_validator import StrategyValidator

__all__ = [
    'PerformanceValidator',
    'LatencyValidator', 
    'ThroughputValidator',
    'ResourceValidator',
    'StrategyValidator',
    'ValidationStatus',
    'PerformanceTarget',
    'ValidationResult',
    'ValidationSummary'
]

__version__ = "1.0.0"