"""
Trading API Orchestrator Module

Provides intelligent routing, failover, and optimization for trading operations.
"""

from .api_selector import APISelector
from .execution_router import ExecutionRouter
from .failover_manager import FailoverManager

__all__ = ['APISelector', 'ExecutionRouter', 'FailoverManager']