"""
Polymarket Utilities

This module provides utility functions and classes for the Polymarket integration:
- Configuration management
- Authentication helpers
- Data validation
- Formatting utilities
- Performance monitoring
"""

from .config import PolymarketConfig, load_config, validate_config
from .auth import authenticate, sign_order, verify_signature
from .validation import (
    validate_order,
    validate_market_data,
    validate_price,
    validate_size,
    ValidationError as UtilValidationError,
)
from .formatting import (
    format_market_data,
    format_order_data,
    format_price,
    format_decimal,
    parse_datetime,
)
from .monitoring import (
    monitor_performance,
    PerformanceMonitor,
    MetricsCollector,
    AlertManager,
)

__all__ = [
    # Configuration
    "PolymarketConfig",
    "load_config",
    "validate_config",
    
    # Authentication
    "authenticate",
    "sign_order",
    "verify_signature",
    
    # Validation
    "validate_order",
    "validate_market_data",
    "validate_price",
    "validate_size",
    "UtilValidationError",
    
    # Formatting
    "format_market_data",
    "format_order_data",
    "format_price",
    "format_decimal",
    "parse_datetime",
    
    # Monitoring
    "monitor_performance",
    "PerformanceMonitor",
    "MetricsCollector",
    "AlertManager",
]