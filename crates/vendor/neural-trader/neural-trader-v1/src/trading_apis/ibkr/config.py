"""
Configuration management for IBKR integration

Provides configuration classes and utilities for managing
connection settings, performance tuning, and environment-specific
configurations.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Trading environment types"""
    PAPER = "paper"
    LIVE = "live"
    SIMULATION = "simulation"


class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class IBKRConfig:
    """
    Comprehensive IBKR configuration
    
    This class consolidates all configuration options for the IBKR integration,
    including connection settings, performance tuning, and feature flags.
    """
    
    # Connection settings
    host: str = "127.0.0.1"
    port: int = 7497  # Default to paper trading
    client_id: int = 1
    account: str = ""
    timeout: float = 10.0
    readonly: bool = False
    
    # Environment
    environment: Environment = Environment.PAPER
    
    # Reconnection settings
    auto_reconnect: bool = True
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    
    # Gateway settings
    gateway_host: str = "127.0.0.1"
    gateway_port: int = 4001
    use_gateway: bool = False
    backup_hosts: List[Tuple[str, int]] = None
    
    # Performance settings
    max_connections: int = 5
    load_balance: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    compression_enabled: bool = True
    
    # Data streaming settings
    buffer_size: int = 50000
    batch_size: int = 200
    batch_timeout_ms: float = 10.0
    max_symbols: int = 100
    tick_filtering: bool = True
    conflation_ms: float = 0.0  # 0 = no conflation
    snapshot_interval_ms: float = 1000.0
    use_native_parsing: bool = True
    pre_allocate_buffers: bool = True
    
    # Risk management
    max_position_size: int = 1000
    max_order_size: int = 500
    max_daily_trades: int = 1000
    max_daily_loss: float = 10000.0
    position_limits: Dict[str, int] = None
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    log_to_console: bool = True
    log_trades: bool = True
    log_market_data: bool = False
    
    # Feature flags
    enable_market_data: bool = True
    enable_trading: bool = True
    enable_options: bool = False
    enable_futures: bool = False
    enable_forex: bool = False
    enable_crypto: bool = False
    
    # Monitoring
    metrics_enabled: bool = True
    metrics_port: int = 8080
    health_check_interval: float = 30.0
    
    def __post_init__(self):
        """Post-initialization validation and defaults"""
        if self.backup_hosts is None:
            self.backup_hosts = []
        
        if self.position_limits is None:
            self.position_limits = {}
        
        # Set port based on environment
        if self.environment == Environment.PAPER:
            if self.port == 7497:  # Default wasn't changed
                self.port = 7497
            if self.gateway_port == 4001:  # Default wasn't changed
                self.gateway_port = 4002
        elif self.environment == Environment.LIVE:
            if self.port == 7497:  # Default wasn't changed
                self.port = 7496
            if self.gateway_port == 4001:  # Default wasn't changed
                self.gateway_port = 4001
        
        # Validate settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration settings"""
        if self.client_id < 0 or self.client_id > 32:
            raise ValueError("Client ID must be between 0 and 32")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        
        if self.batch_size <= 0 or self.batch_size > self.buffer_size:
            raise ValueError("Batch size must be positive and <= buffer size")
        
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")
        
        if self.max_order_size <= 0:
            raise ValueError("Max order size must be positive")
        
        # Warn about risky settings
        if self.environment == Environment.LIVE and self.readonly is False:
            logger.warning("Trading is enabled in LIVE environment - ensure this is intended")
        
        if self.conflation_ms > 0 and self.conflation_ms < 1.0:
            logger.warning("Conflation < 1ms may not be effective")
    
    @classmethod
    def from_env(cls) -> 'IBKRConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Connection settings
        config.host = os.getenv('IBKR_HOST', config.host)
        config.port = int(os.getenv('IBKR_PORT', str(config.port)))
        config.client_id = int(os.getenv('IBKR_CLIENT_ID', str(config.client_id)))
        config.account = os.getenv('IBKR_ACCOUNT', config.account)
        config.timeout = float(os.getenv('IBKR_TIMEOUT', str(config.timeout)))
        config.readonly = os.getenv('IBKR_READONLY', 'false').lower() == 'true'
        
        # Environment
        env_str = os.getenv('IBKR_ENVIRONMENT', 'paper').lower()
        if env_str == 'live':
            config.environment = Environment.LIVE
        elif env_str == 'simulation':
            config.environment = Environment.SIMULATION
        else:
            config.environment = Environment.PAPER
        
        # Gateway settings
        config.gateway_host = os.getenv('IBKR_GATEWAY_HOST', config.gateway_host)
        config.gateway_port = int(os.getenv('IBKR_GATEWAY_PORT', str(config.gateway_port)))
        config.use_gateway = os.getenv('IBKR_USE_GATEWAY', 'false').lower() == 'true'
        
        # Performance settings
        config.max_connections = int(os.getenv('IBKR_MAX_CONNECTIONS', str(config.max_connections)))
        config.load_balance = os.getenv('IBKR_LOAD_BALANCE', 'false').lower() == 'true'
        config.ssl_enabled = os.getenv('IBKR_SSL_ENABLED', 'false').lower() == 'true'
        config.ssl_cert_path = os.getenv('IBKR_SSL_CERT_PATH')
        
        # Data streaming
        config.buffer_size = int(os.getenv('IBKR_BUFFER_SIZE', str(config.buffer_size)))
        config.batch_size = int(os.getenv('IBKR_BATCH_SIZE', str(config.batch_size)))
        config.batch_timeout_ms = float(os.getenv('IBKR_BATCH_TIMEOUT_MS', str(config.batch_timeout_ms)))
        config.max_symbols = int(os.getenv('IBKR_MAX_SYMBOLS', str(config.max_symbols)))
        config.conflation_ms = float(os.getenv('IBKR_CONFLATION_MS', str(config.conflation_ms)))
        
        # Risk management
        config.max_position_size = int(os.getenv('IBKR_MAX_POSITION_SIZE', str(config.max_position_size)))
        config.max_order_size = int(os.getenv('IBKR_MAX_ORDER_SIZE', str(config.max_order_size)))
        config.max_daily_trades = int(os.getenv('IBKR_MAX_DAILY_TRADES', str(config.max_daily_trades)))
        config.max_daily_loss = float(os.getenv('IBKR_MAX_DAILY_LOSS', str(config.max_daily_loss)))
        
        # Logging
        log_level_str = os.getenv('IBKR_LOG_LEVEL', 'INFO').upper()
        config.log_level = LogLevel(log_level_str)
        config.log_file = os.getenv('IBKR_LOG_FILE')
        config.log_to_console = os.getenv('IBKR_LOG_TO_CONSOLE', 'true').lower() == 'true'
        
        # Feature flags
        config.enable_market_data = os.getenv('IBKR_ENABLE_MARKET_DATA', 'true').lower() == 'true'
        config.enable_trading = os.getenv('IBKR_ENABLE_TRADING', 'true').lower() == 'true'
        config.enable_options = os.getenv('IBKR_ENABLE_OPTIONS', 'false').lower() == 'true'
        config.enable_futures = os.getenv('IBKR_ENABLE_FUTURES', 'false').lower() == 'true'
        config.enable_forex = os.getenv('IBKR_ENABLE_FOREX', 'false').lower() == 'true'
        config.enable_crypto = os.getenv('IBKR_ENABLE_CRYPTO', 'false').lower() == 'true'
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'IBKRConfig':
        """Create configuration from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            config = cls()
            
            # Update config with file data
            for key, value in data.items():
                if hasattr(config, key):
                    # Handle enum conversions
                    if key == 'environment':
                        value = Environment(value)
                    elif key == 'log_level':
                        value = LogLevel(value)
                    
                    setattr(config, key, value)
            
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def to_file(self, file_path: str):
        """Save configuration to JSON file"""
        try:
            # Convert to dictionary
            data = {}
            for key, value in self.__dict__.items():
                if isinstance(value, Enum):
                    data[key] = value.value
                else:
                    data[key] = value
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise
    
    def get_connection_config(self):
        """Get connection configuration for IBKRClient"""
        from .ibkr_client import ConnectionConfig
        
        return ConnectionConfig(
            host=self.host,
            port=self.port,
            client_id=self.client_id,
            account=self.account,
            timeout=self.timeout,
            readonly=self.readonly,
            auto_reconnect=self.auto_reconnect,
            reconnect_interval=self.reconnect_interval,
            max_reconnect_attempts=self.max_reconnect_attempts
        )
    
    def get_gateway_config(self):
        """Get gateway configuration for IBKRGateway"""
        from .ibkr_gateway import GatewayConfig, ConnectionMode
        
        return GatewayConfig(
            primary_host=self.gateway_host,
            primary_port=self.gateway_port,
            backup_hosts=self.backup_hosts,
            connection_mode=ConnectionMode.SECURE if self.ssl_enabled else ConnectionMode.DIRECT,
            ssl_enabled=self.ssl_enabled,
            ssl_cert_path=self.ssl_cert_path,
            connection_timeout=self.timeout,
            max_connections=self.max_connections,
            load_balance=self.load_balance,
            compression_enabled=self.compression_enabled
        )
    
    def get_stream_config(self):
        """Get stream configuration for IBKRDataStream"""
        from .ibkr_data_stream import StreamConfig
        
        return StreamConfig(
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            batch_timeout_ms=self.batch_timeout_ms,
            max_symbols=self.max_symbols,
            tick_filtering=self.tick_filtering,
            conflation_ms=self.conflation_ms,
            snapshot_interval_ms=self.snapshot_interval_ms,
            use_native_parsing=self.use_native_parsing,
            pre_allocate_buffers=self.pre_allocate_buffers
        )
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        level = getattr(logging, self.log_level.value)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup root logger
        logger = logging.getLogger('ibkr')
        logger.setLevel(level)
        
        # Clear existing handlers
        logger.handlers = []
        
        # Add console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Set specific loggers
        if not self.log_market_data:
            logging.getLogger('ibkr.data_stream').setLevel(logging.WARNING)
        
        if not self.log_trades:
            logging.getLogger('ibkr.trading').setLevel(logging.WARNING)
    
    def get_risk_limits(self) -> Dict[str, Any]:
        """Get risk management limits"""
        return {
            'max_position_size': self.max_position_size,
            'max_order_size': self.max_order_size,
            'max_daily_trades': self.max_daily_trades,
            'max_daily_loss': self.max_daily_loss,
            'position_limits': self.position_limits.copy()
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        feature_map = {
            'market_data': self.enable_market_data,
            'trading': self.enable_trading,
            'options': self.enable_options,
            'futures': self.enable_futures,
            'forex': self.enable_forex,
            'crypto': self.enable_crypto
        }
        
        return feature_map.get(feature, False)
    
    def get_performance_profile(self) -> str:
        """Get performance profile description"""
        if self.conflation_ms == 0 and self.batch_timeout_ms <= 5:
            return "ultra_low_latency"
        elif self.conflation_ms <= 10 and self.batch_timeout_ms <= 20:
            return "low_latency"
        elif self.conflation_ms <= 100 and self.batch_timeout_ms <= 100:
            return "balanced"
        else:
            return "high_throughput"
    
    def __str__(self) -> str:
        """String representation"""
        return f"IBKRConfig(environment={self.environment.value}, " \
               f"host={self.host}:{self.port}, " \
               f"profile={self.get_performance_profile()})"


# Predefined configurations
PAPER_TRADING_CONFIG = IBKRConfig(
    environment=Environment.PAPER,
    host="127.0.0.1",
    port=7497,
    readonly=False,
    enable_trading=True,
    log_level=LogLevel.INFO
)

LIVE_TRADING_CONFIG = IBKRConfig(
    environment=Environment.LIVE,
    host="127.0.0.1",
    port=7496,
    readonly=False,
    enable_trading=True,
    log_level=LogLevel.WARNING,
    max_position_size=500,
    max_order_size=100
)

MARKET_DATA_CONFIG = IBKRConfig(
    environment=Environment.PAPER,
    host="127.0.0.1",
    port=7497,
    readonly=True,
    enable_trading=False,
    enable_market_data=True,
    buffer_size=100000,
    batch_size=500,
    batch_timeout_ms=1.0,
    log_level=LogLevel.INFO
)

HIGH_FREQUENCY_CONFIG = IBKRConfig(
    environment=Environment.PAPER,
    host="127.0.0.1",
    port=7497,
    readonly=False,
    enable_trading=True,
    enable_market_data=True,
    buffer_size=200000,
    batch_size=1000,
    batch_timeout_ms=0.5,
    conflation_ms=0,
    use_native_parsing=True,
    compression_enabled=False,
    log_level=LogLevel.WARNING
)