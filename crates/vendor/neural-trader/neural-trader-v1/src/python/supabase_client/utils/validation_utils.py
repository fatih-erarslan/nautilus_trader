"""
Validation Utility Functions
===========================

Input validation and sanitization utilities.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import html
import json

logger = logging.getLogger(__name__)

# Regular expressions for validation
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
SYMBOL_REGEX = re.compile(r'^[A-Z]{1,5}$')  # Stock symbols: 1-5 uppercase letters
UUID_REGEX = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

def validate_uuid(value: Union[str, UUID]) -> bool:
    """
    Validate UUID format.
    
    Args:
        value: UUID string or UUID object
        
    Returns:
        True if valid UUID
    """
    if isinstance(value, UUID):
        return True
    
    if isinstance(value, str):
        return bool(UUID_REGEX.match(value))
    
    return False

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid email format
    """
    if not isinstance(email, str):
        return False
    
    if len(email) > 254:  # RFC 5321 limit
        return False
    
    return bool(EMAIL_REGEX.match(email))

def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.
    
    Args:
        symbol: Symbol string to validate
        
    Returns:
        True if valid symbol format
    """
    if not isinstance(symbol, str):
        return False
    
    return bool(SYMBOL_REGEX.match(symbol))

def validate_symbols(symbols: List[str]) -> List[str]:
    """
    Validate and filter list of symbols.
    
    Args:
        symbols: List of symbols to validate
        
    Returns:
        List of valid symbols
    """
    valid_symbols = []
    for symbol in symbols:
        if validate_symbol(symbol):
            valid_symbols.append(symbol.upper())
        else:
            logger.warning(f"Invalid symbol format: {symbol}")
    
    return valid_symbols

def sanitize_input(value: Any, max_length: Optional[int] = None) -> str:
    """
    Sanitize input string for safe database storage.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if value is None:
        return ""
    
    # Convert to string
    if not isinstance(value, str):
        value = str(value)
    
    # HTML escape
    value = html.escape(value)
    
    # Remove control characters
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
    
    # Trim whitespace
    value = value.strip()
    
    # Enforce max length
    if max_length and len(value) > max_length:
        value = value[:max_length]
    
    return value

def validate_json(value: str) -> bool:
    """
    Validate JSON string format.
    
    Args:
        value: JSON string to validate
        
    Returns:
        True if valid JSON
    """
    try:
        json.loads(value)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> bool:
    """
    Validate numeric value is within range.
    
    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        True if value is within range
    """
    if not isinstance(value, (int, float)):
        return False
    
    if min_value is not None and value < min_value:
        return False
    
    if max_value is not None and value > max_value:
        return False
    
    return True

def validate_trading_account_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate trading account data.
    
    Args:
        data: Trading account data dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'account_name', 'account_type', 'broker']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate user_id
    if 'user_id' in data and not validate_uuid(data['user_id']):
        errors.append("Invalid user_id format")
    
    # Validate account_type
    valid_account_types = ['demo', 'live', 'paper']
    if 'account_type' in data and data['account_type'] not in valid_account_types:
        errors.append(f"Invalid account_type. Must be one of: {valid_account_types}")
    
    # Validate balance
    if 'balance' in data:
        if not validate_numeric_range(data['balance'], min_value=0):
            errors.append("Balance must be non-negative")
    
    # Validate account_name length
    if 'account_name' in data:
        if len(data['account_name']) > 100:
            errors.append("Account name must be 100 characters or less")
    
    return errors

def validate_neural_model_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate neural model data.
    
    Args:
        data: Neural model data dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'name', 'model_type']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate user_id
    if 'user_id' in data and not validate_uuid(data['user_id']):
        errors.append("Invalid user_id format")
    
    # Validate model_type
    valid_model_types = ['lstm', 'transformer', 'cnn', 'ensemble']
    if 'model_type' in data and data['model_type'] not in valid_model_types:
        errors.append(f"Invalid model_type. Must be one of: {valid_model_types}")
    
    # Validate name length
    if 'name' in data:
        if len(data['name']) > 100:
            errors.append("Model name must be 100 characters or less")
        if len(data['name']) < 1:
            errors.append("Model name must not be empty")
    
    # Validate symbols
    if 'symbols' in data and data['symbols']:
        valid_symbols = validate_symbols(data['symbols'])
        if len(valid_symbols) != len(data['symbols']):
            errors.append("Some symbols have invalid format")
    
    # Validate configuration
    if 'configuration' in data and data['configuration']:
        if not isinstance(data['configuration'], dict):
            errors.append("Configuration must be a dictionary")
    
    return errors

def validate_trading_bot_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate trading bot data.
    
    Args:
        data: Trading bot data dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'name', 'strategy', 'account_id']
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate UUIDs
    uuid_fields = ['user_id', 'account_id', 'neural_model_id']
    for field in uuid_fields:
        if field in data and data[field] and not validate_uuid(data[field]):
            errors.append(f"Invalid {field} format")
    
    # Validate strategy
    valid_strategies = [
        'mean_reversion', 'momentum', 'arbitrage', 'market_making',
        'neural_sentiment', 'neural_momentum', 'pairs_trading'
    ]
    if 'strategy' in data and data['strategy'] not in valid_strategies:
        errors.append(f"Invalid strategy. Must be one of: {valid_strategies}")
    
    # Validate name length
    if 'name' in data:
        if len(data['name']) > 100:
            errors.append("Bot name must be 100 characters or less")
        if len(data['name']) < 1:
            errors.append("Bot name must not be empty")
    
    # Validate symbols
    if 'symbols' in data and data['symbols']:
        valid_symbols = validate_symbols(data['symbols'])
        if len(valid_symbols) != len(data['symbols']):
            errors.append("Some symbols have invalid format")
    
    # Validate risk parameters
    if 'risk_params' in data and data['risk_params']:
        risk_params = data['risk_params']
        
        if 'max_position_size' in risk_params:
            if not validate_numeric_range(risk_params['max_position_size'], min_value=0, max_value=1):
                errors.append("max_position_size must be between 0 and 1")
        
        if 'stop_loss' in risk_params:
            if not validate_numeric_range(risk_params['stop_loss'], min_value=0, max_value=1):
                errors.append("stop_loss must be between 0 and 1")
        
        if 'take_profit' in risk_params:
            if not validate_numeric_range(risk_params['take_profit'], min_value=0):
                errors.append("take_profit must be non-negative")
    
    return errors

def validate_order_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate order data.
    
    Args:
        data: Order data dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['user_id', 'account_id', 'symbol', 'order_type', 'side', 'quantity']
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate UUIDs
    uuid_fields = ['user_id', 'account_id', 'bot_id']
    for field in uuid_fields:
        if field in data and data[field] and not validate_uuid(data[field]):
            errors.append(f"Invalid {field} format")
    
    # Validate symbol
    if 'symbol' in data and not validate_symbol(data['symbol']):
        errors.append("Invalid symbol format")
    
    # Validate order_type
    valid_order_types = ['market', 'limit', 'stop', 'stop_limit']
    if 'order_type' in data and data['order_type'] not in valid_order_types:
        errors.append(f"Invalid order_type. Must be one of: {valid_order_types}")
    
    # Validate side
    valid_sides = ['buy', 'sell']
    if 'side' in data and data['side'] not in valid_sides:
        errors.append(f"Invalid side. Must be one of: {valid_sides}")
    
    # Validate quantity
    if 'quantity' in data:
        if not validate_numeric_range(data['quantity'], min_value=0.01):
            errors.append("Quantity must be greater than 0")
    
    # Validate price (if provided)
    if 'price' in data and data['price'] is not None:
        if not validate_numeric_range(data['price'], min_value=0.01):
            errors.append("Price must be greater than 0")
    
    return errors

class ValidationError(Exception):
    """Custom validation error."""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []

def raise_if_invalid(data: Dict[str, Any], validator_func: callable, entity_name: str):
    """
    Raise ValidationError if data is invalid.
    
    Args:
        data: Data to validate
        validator_func: Validation function
        entity_name: Name of entity being validated
        
    Raises:
        ValidationError: If validation fails
    """
    errors = validator_func(data)
    if errors:
        raise ValidationError(f"Invalid {entity_name} data", errors)

# Pre-configured validators
def validate_create_account_request(data: Dict[str, Any]) -> List[str]:
    """Validate create account request."""
    return validate_trading_account_data(data)

def validate_create_model_request(data: Dict[str, Any]) -> List[str]:
    """Validate create model request."""
    return validate_neural_model_data(data)

def validate_create_bot_request(data: Dict[str, Any]) -> List[str]:
    """Validate create bot request."""
    return validate_trading_bot_data(data)

def validate_place_order_request(data: Dict[str, Any]) -> List[str]:
    """Validate place order request."""
    return validate_order_data(data)