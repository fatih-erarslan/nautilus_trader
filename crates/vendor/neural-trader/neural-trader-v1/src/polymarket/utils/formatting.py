"""
Data formatting utilities for Polymarket integration

This module provides utilities for formatting market data, orders,
prices, and other Polymarket-specific data structures.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Union
import json


def format_market_data(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format market data for display"""
    formatted = market_data.copy()
    
    # Format prices
    if 'current_prices' in formatted:
        for outcome, price in formatted['current_prices'].items():
            formatted['current_prices'][outcome] = format_price(price)
    
    # Format dates
    if 'created_at' in formatted:
        formatted['created_at'] = format_datetime(formatted['created_at'])
    if 'updated_at' in formatted:
        formatted['updated_at'] = format_datetime(formatted['updated_at'])
    if 'end_date' in formatted:
        formatted['end_date'] = format_datetime(formatted['end_date'])
    
    return formatted


def format_order_data(order_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format order data for display"""
    formatted = order_data.copy()
    
    # Format price and size
    if 'price' in formatted and formatted['price'] is not None:
        formatted['price'] = format_price(formatted['price'])
    if 'size' in formatted:
        formatted['size'] = format_decimal(formatted['size'])
    
    # Format dates
    if 'created_at' in formatted:
        formatted['created_at'] = format_datetime(formatted['created_at'])
    
    return formatted


def format_price(price: Union[float, Decimal, str]) -> str:
    """Format price value for display"""
    if price is None:
        return "N/A"
    
    try:
        price_val = Decimal(str(price))
        return f"{price_val:.4f}"
    except (ValueError, TypeError):
        return str(price)


def format_decimal(value: Union[float, Decimal, str], places: int = 2) -> str:
    """Format decimal value for display"""
    if value is None:
        return "N/A"
    
    try:
        decimal_val = Decimal(str(value))
        return f"{decimal_val:.{places}f}"
    except (ValueError, TypeError):
        return str(value)


def parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """Parse datetime string into datetime object"""
    if not datetime_str:
        return None
    
    try:
        # Handle ISO format with 'Z' suffix
        if datetime_str.endswith('Z'):
            datetime_str = datetime_str[:-1] + '+00:00'
        
        return datetime.fromisoformat(datetime_str)
    except ValueError:
        return None


def format_datetime(dt: Union[datetime, str, None]) -> str:
    """Format datetime for display"""
    if dt is None:
        return "N/A"
    
    if isinstance(dt, str):
        dt = parse_datetime(dt)
        if dt is None:
            return "Invalid date"
    
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_percentage(value: Union[float, Decimal], places: int = 2) -> str:
    """Format percentage value for display"""
    try:
        percent_val = float(value) * 100
        return f"{percent_val:.{places}f}%"
    except (ValueError, TypeError):
        return "N/A"


def format_currency(value: Union[float, Decimal], symbol: str = "$") -> str:
    """Format currency value for display"""
    try:
        currency_val = float(value)
        return f"{symbol}{currency_val:,.2f}"
    except (ValueError, TypeError):
        return "N/A"


def truncate_string(text: str, max_length: int = 50) -> str:
    """Truncate string to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON string"""
    try:
        return json.dumps(data, indent=indent, default=str)
    except (TypeError, ValueError):
        return str(data)