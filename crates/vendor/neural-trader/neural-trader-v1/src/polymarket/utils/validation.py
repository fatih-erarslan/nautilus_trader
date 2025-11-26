"""
Data validation utilities for Polymarket integration

This module provides validation functions for orders, market data,
prices, and other Polymarket-specific data structures.
"""

from decimal import Decimal
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails"""
    pass


def validate_order(order_data: Dict[str, Any]) -> bool:
    """
    Validate order data
    
    Args:
        order_data: Order data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['market_id', 'outcome_id', 'side', 'type', 'size']
    
    for field in required_fields:
        if field not in order_data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate side
    if order_data['side'].lower() not in ['buy', 'sell']:
        raise ValidationError("Side must be 'buy' or 'sell'")
    
    # Validate order type
    if order_data['type'].lower() not in ['market', 'limit', 'stop', 'stop_limit']:
        raise ValidationError("Invalid order type")
    
    # Validate size
    try:
        size = float(order_data['size'])
        if size <= 0:
            raise ValidationError("Order size must be positive")
    except (ValueError, TypeError):
        raise ValidationError("Invalid order size")
    
    # Validate price for limit orders
    if order_data['type'].lower() in ['limit', 'stop_limit']:
        if 'price' not in order_data:
            raise ValidationError("Price required for limit orders")
        
        try:
            price = float(order_data['price'])
            if not (0 < price <= 1):
                raise ValidationError("Price must be between 0 and 1")
        except (ValueError, TypeError):
            raise ValidationError("Invalid price")
    
    return True


def validate_market_data(market_data: Dict[str, Any]) -> bool:
    """
    Validate market data
    
    Args:
        market_data: Market data dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    required_fields = ['id', 'question', 'outcomes', 'end_date', 'status']
    
    for field in required_fields:
        if field not in market_data:
            raise ValidationError(f"Missing required field: {field}")
    
    # Validate outcomes
    outcomes = market_data['outcomes']
    if not isinstance(outcomes, list) or len(outcomes) < 2:
        raise ValidationError("Market must have at least 2 outcomes")
    
    # Validate status
    valid_statuses = ['active', 'paused', 'closed', 'resolved', 'cancelled']
    if market_data['status'].lower() not in valid_statuses:
        raise ValidationError("Invalid market status")
    
    return True


def validate_price(price: Any) -> bool:
    """
    Validate price value
    
    Args:
        price: Price value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        price_val = float(price)
        if not (0 < price_val <= 1):
            raise ValidationError("Price must be between 0 and 1")
        return True
    except (ValueError, TypeError):
        raise ValidationError("Invalid price format")


def validate_size(size: Any) -> bool:
    """
    Validate order size
    
    Args:
        size: Size value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        size_val = float(size)
        if size_val <= 0:
            raise ValidationError("Size must be positive")
        return True
    except (ValueError, TypeError):
        raise ValidationError("Invalid size format")


def validate_market_id(market_id: str) -> bool:
    """
    Validate market ID format
    
    Args:
        market_id: Market ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(market_id, str) or not market_id.strip():
        raise ValidationError("Market ID must be a non-empty string")
    
    return True


def validate_outcome_id(outcome_id: str) -> bool:
    """
    Validate outcome ID format
    
    Args:
        outcome_id: Outcome ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(outcome_id, str) or not outcome_id.strip():
        raise ValidationError("Outcome ID must be a non-empty string")
    
    return True


def validate_user_id(user_id: str) -> bool:
    """
    Validate user ID format
    
    Args:
        user_id: User ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValidationError("User ID must be a non-empty string")
    
    return True


def validate_decimal_range(
    value: Any, 
    min_value: Optional[float] = None, 
    max_value: Optional[float] = None,
    field_name: str = "value"
) -> bool:
    """
    Validate decimal value within range
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Field name for error messages
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        decimal_val = Decimal(str(value))
        
        if min_value is not None and decimal_val < Decimal(str(min_value)):
            raise ValidationError(f"{field_name} must be >= {min_value}")
        
        if max_value is not None and decimal_val > Decimal(str(max_value)):
            raise ValidationError(f"{field_name} must be <= {max_value}")
        
        return True
        
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid {field_name} format")


def validate_confidence(confidence: Any) -> bool:
    """
    Validate confidence value (0.0 to 1.0)
    
    Args:
        confidence: Confidence value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_decimal_range(confidence, 0.0, 1.0, "confidence")


def validate_probability(probability: Any) -> bool:
    """
    Validate probability value (0.0 to 1.0)
    
    Args:
        probability: Probability value to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_decimal_range(probability, 0.0, 1.0, "probability")


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not api_key or not isinstance(api_key, str):
        raise ValidationError("API key must be a non-empty string")
    
    if len(api_key) < 10:
        raise ValidationError("API key too short")
    
    return True


def validate_signature(signature: Optional[str]) -> bool:
    """
    Validate signature format
    
    Args:
        signature: Signature to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    if not signature or not isinstance(signature, str):
        raise ValidationError("Signature must be a non-empty string")
    
    # Basic hex string validation
    try:
        int(signature, 16)
    except ValueError:
        raise ValidationError("Invalid signature format")
    
    return True