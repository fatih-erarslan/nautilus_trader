"""Authentication module for Neural Trader"""

from .jwt_handler import (
    JWTHandler,
    get_current_user_optional,
    get_current_user_required,
    check_auth_optional,
    check_auth_required,
    get_auth_config,
    validate_token,
    AUTH_ENABLED,
    AUTH_USERNAME,
    AUTH_PASSWORD,
)

__all__ = [
    "JWTHandler",
    "get_current_user_optional",
    "get_current_user_required", 
    "check_auth_optional",
    "check_auth_required",
    "get_auth_config",
    "validate_token",
    "AUTH_ENABLED",
    "AUTH_USERNAME",
    "AUTH_PASSWORD",
]