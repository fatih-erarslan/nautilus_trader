"""
Configuration management for Polymarket integration

This module handles configuration loading, validation, and management
for the Polymarket trading system.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolymarketConfig:
    """Configuration for Polymarket integration"""
    
    # API Configuration
    api_key: str = ""
    private_key: str = ""
    clob_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    ws_url: str = "wss://ws.polymarket.com"
    
    # Rate Limiting
    rate_limit: int = 100  # requests per minute
    burst_limit: int = 10  # burst requests
    
    # Request Configuration
    timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # base delay in seconds
    
    # Cache Configuration
    cache_ttl: int = 300  # seconds
    cache_max_size: int = 1000
    
    # Trading Configuration
    default_slippage: float = 0.01  # 1%
    max_position_size: float = 1000.0
    min_order_size: float = 0.01
    
    # Environment
    environment: str = "production"  # production, staging, development
    debug: bool = False
    log_level: str = "INFO"
    
    # Feature Flags
    enable_websocket: bool = True
    enable_gpu_acceleration: bool = True
    enable_caching: bool = True
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._load_from_environment()
        self._validate()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'POLYMARKET_API_KEY': 'api_key',
            'POLYMARKET_PRIVATE_KEY': 'private_key',
            'POLYMARKET_CLOB_URL': 'clob_url',
            'POLYMARKET_GAMMA_URL': 'gamma_url',
            'POLYMARKET_WS_URL': 'ws_url',
            'POLYMARKET_RATE_LIMIT': ('rate_limit', int),
            'POLYMARKET_TIMEOUT': ('timeout', int),
            'POLYMARKET_MAX_RETRIES': ('max_retries', int),
            'POLYMARKET_ENVIRONMENT': 'environment',
            'POLYMARKET_DEBUG': ('debug', lambda x: x.lower() == 'true'),
            'POLYMARKET_LOG_LEVEL': 'log_level',
        }
        
        for env_var, config_field in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                if isinstance(config_field, tuple):
                    field_name, converter = config_field
                    try:
                        setattr(self, field_name, converter(env_value))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid value for {env_var}: {env_value}, error: {e}")
                else:
                    setattr(self, config_field, env_value)
    
    def _validate(self):
        """Validate configuration values"""
        errors = []
        
        # Required fields
        if not self.api_key and self.environment == "production":
            errors.append("API key is required for production environment")
        
        if not self.private_key and self.environment == "production":
            errors.append("Private key is required for production environment")
        
        # URL validation
        required_urls = [self.clob_url, self.gamma_url, self.ws_url]
        for url in required_urls:
            if not url.startswith(('http://', 'https://', 'ws://', 'wss://')):
                errors.append(f"Invalid URL format: {url}")
        
        # Numeric validations
        if self.rate_limit <= 0:
            errors.append("Rate limit must be positive")
        
        if self.timeout <= 0:
            errors.append("Timeout must be positive")
        
        if self.max_retries < 0:
            errors.append("Max retries cannot be negative")
        
        if self.cache_ttl < 0:
            errors.append("Cache TTL cannot be negative")
        
        if self.max_position_size <= 0:
            errors.append("Max position size must be positive")
        
        if self.min_order_size <= 0:
            errors.append("Min order size must be positive")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        logger.info(f"Configuration validated for {self.environment} environment")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PolymarketConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # Create instance with loaded data
            return cls(**data)
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise ValueError(f"Invalid configuration file: {e}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            raise ValueError(f"Failed to load configuration: {e}")
    
    def to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_data = {
            'clob_url': self.clob_url,
            'gamma_url': self.gamma_url,
            'ws_url': self.ws_url,
            'rate_limit': self.rate_limit,
            'burst_limit': self.burst_limit,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'cache_ttl': self.cache_ttl,
            'cache_max_size': self.cache_max_size,
            'default_slippage': self.default_slippage,
            'max_position_size': self.max_position_size,
            'min_order_size': self.min_order_size,
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'enable_websocket': self.enable_websocket,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_caching': self.enable_caching,
            'enable_metrics': self.enable_metrics,
        }
        
        # Don't save sensitive data
        # api_key and private_key should come from environment variables
        
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            raise ValueError(f"Failed to save configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'clob_url': self.clob_url,
            'gamma_url': self.gamma_url,
            'ws_url': self.ws_url,
            'rate_limit': self.rate_limit,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'environment': self.environment,
            'debug': self.debug,
            'features': {
                'websocket': self.enable_websocket,
                'gpu_acceleration': self.enable_gpu_acceleration,
                'caching': self.enable_caching,
                'metrics': self.enable_metrics,
            }
        }
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get common API headers"""
        return {
            'User-Agent': f'ai-news-trader-polymarket/1.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }


def load_config(config_path: Optional[str] = None) -> PolymarketConfig:
    """
    Load Polymarket configuration
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Loaded configuration
    """
    if config_path:
        return PolymarketConfig.from_file(config_path)
    
    # Try default locations
    default_paths = [
        os.path.join(os.getcwd(), 'polymarket.json'),
        os.path.join(os.path.expanduser('~'), '.polymarket', 'config.json'),
        os.path.join('/etc', 'polymarket', 'config.json'),
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            logger.info(f"Loading configuration from {path}")
            return PolymarketConfig.from_file(path)
    
    logger.info("No configuration file found, using defaults with environment variables")
    return PolymarketConfig()


def validate_config(config: PolymarketConfig) -> bool:
    """
    Validate configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    try:
        config._validate()
        return True
    except ValueError:
        raise


def create_default_config(config_path: str):
    """Create a default configuration file"""
    config = PolymarketConfig()
    config.to_file(config_path)
    logger.info(f"Created default configuration at {config_path}")


def get_config_template() -> Dict[str, Any]:
    """Get configuration template for documentation"""
    return {
        "clob_url": "https://clob.polymarket.com",
        "gamma_url": "https://gamma-api.polymarket.com", 
        "ws_url": "wss://ws.polymarket.com",
        "rate_limit": 100,
        "timeout": 30,
        "max_retries": 3,
        "environment": "production",
        "debug": False,
        "log_level": "INFO",
        "enable_websocket": True,
        "enable_gpu_acceleration": True,
        "enable_caching": True,
        "enable_metrics": True,
        "_comments": {
            "api_key": "Set via POLYMARKET_API_KEY environment variable",
            "private_key": "Set via POLYMARKET_PRIVATE_KEY environment variable",
            "environment": "Options: production, staging, development",
            "log_level": "Options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        }
    }