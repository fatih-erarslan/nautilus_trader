"""
Configuration for Canadian Trading Platform

Centralized configuration for all Canadian broker integrations.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class QuestradeConfig:
    """Questrade API configuration"""
    refresh_token: Optional[str] = None
    api_server: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_second: int = 30
    enable_streaming: bool = True
    paper_trading: bool = True  # Use practice account by default
    
    # Connection pool settings
    connection_pool_size: int = 100
    connections_per_host: int = 30
    
    # Cache settings
    symbol_cache_ttl: int = 3600  # 1 hour
    quote_cache_ttl: int = 5      # 5 seconds
    
    # Risk management
    max_position_size: float = 0.25  # 25% of portfolio
    max_daily_trades: int = 50
    max_order_value: float = 100000  # $100k per order
    
    def __post_init__(self):
        """Load from environment if not provided"""
        if not self.refresh_token:
            self.refresh_token = os.getenv("QUESTRADE_REFRESH_TOKEN")
        if not self.api_server:
            self.api_server = os.getenv("QUESTRADE_API_SERVER")


@dataclass
class IBKRConfig:
    """Interactive Brokers configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port
    client_id: int = 1
    account: Optional[str] = None
    
    # Production settings
    live_trading: bool = False
    live_port: int = 7496
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    
    # Risk management
    max_position_size: float = 0.25
    max_daily_volume: float = 100000
    
    def __post_init__(self):
        """Adjust settings based on mode"""
        if self.live_trading:
            self.port = self.live_port
            logger.warning("IBKR configured for LIVE TRADING")


@dataclass
class OANDAConfig:
    """OANDA configuration"""
    api_token: Optional[str] = None
    account_id: Optional[str] = None
    environment: str = "practice"  # "practice" or "live"
    
    # API endpoints
    practice_url: str = "https://api-fxpractice.oanda.com"
    live_url: str = "https://api-fxtrade.oanda.com"
    
    # Streaming endpoints
    practice_stream_url: str = "https://stream-fxpractice.oanda.com"
    live_stream_url: str = "https://stream-fxtrade.oanda.com"
    
    # Risk management
    max_position_units: int = 100000
    max_leverage: float = 30.0  # Canadian regulatory limit
    
    def __post_init__(self):
        """Load from environment"""
        if not self.api_token:
            self.api_token = os.getenv("OANDA_API_TOKEN")
        if not self.account_id:
            self.account_id = os.getenv("OANDA_ACCOUNT_ID")
    
    @property
    def api_url(self) -> str:
        """Get appropriate API URL based on environment"""
        return self.live_url if self.environment == "live" else self.practice_url
    
    @property
    def stream_url(self) -> str:
        """Get appropriate streaming URL based on environment"""
        return self.live_stream_url if self.environment == "live" else self.practice_stream_url


@dataclass
class ComplianceConfig:
    """CIRO compliance configuration"""
    # Position limits
    max_single_position_pct: float = 0.25  # 25% of portfolio
    max_sector_concentration: float = 0.40  # 40% in single sector
    max_leverage: float = 1.0               # No leverage for retail
    
    # Volume limits
    max_daily_trading_volume: float = 100000  # $100k daily
    max_order_size: float = 50000             # $50k per order
    
    # Time restrictions
    pre_market_trading: bool = False
    after_hours_trading: bool = False
    
    # Restricted securities
    restricted_symbols: list = None
    
    def __post_init__(self):
        """Initialize restricted list"""
        if self.restricted_symbols is None:
            self.restricted_symbols = []


@dataclass
class TradingConfig:
    """General trading configuration"""
    # Default settings
    default_broker: str = "questrade"
    enable_paper_trading: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_file: Optional[str] = "canadian_trading.log"
    
    # Performance
    enable_gpu: bool = True
    cache_enabled: bool = True
    
    # Data storage
    data_dir: Path = Path.home() / ".canadian_trading"
    token_dir: Path = None
    cache_dir: Path = None
    
    def __post_init__(self):
        """Setup directories"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if self.token_dir is None:
            self.token_dir = self.data_dir / "tokens"
            self.token_dir.mkdir(exist_ok=True)
        
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True)


class CanadianTradingConfig:
    """Main configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional JSON config file path
        """
        # Load default configurations
        self.trading = TradingConfig()
        self.questrade = QuestradeConfig()
        self.ibkr = IBKRConfig()
        self.oanda = OANDAConfig()
        self.compliance = ComplianceConfig()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        # Setup logging
        self._setup_logging()
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if "trading" in config_data:
                self.trading = TradingConfig(**config_data["trading"])
            if "questrade" in config_data:
                self.questrade = QuestradeConfig(**config_data["questrade"])
            if "ibkr" in config_data:
                self.ibkr = IBKRConfig(**config_data["ibkr"])
            if "oanda" in config_data:
                self.oanda = OANDAConfig(**config_data["oanda"])
            if "compliance" in config_data:
                self.compliance = ComplianceConfig(**config_data["compliance"])
            
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        config_data = {
            "trading": self.trading.__dict__,
            "questrade": {k: v for k, v in self.questrade.__dict__.items() 
                         if k != "refresh_token"},  # Don't save token
            "ibkr": self.ibkr.__dict__,
            "oanda": {k: v for k, v in self.oanda.__dict__.items() 
                     if k != "api_token"},  # Don't save token
            "compliance": self.compliance.__dict__
        }
        
        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        config_data = convert_paths(config_data)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved configuration to {config_file}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Trading overrides
        if os.getenv("CANADIAN_TRADING_PAPER_MODE"):
            self.trading.enable_paper_trading = os.getenv("CANADIAN_TRADING_PAPER_MODE").lower() == "true"
        
        # Broker selection
        if os.getenv("CANADIAN_TRADING_DEFAULT_BROKER"):
            self.trading.default_broker = os.getenv("CANADIAN_TRADING_DEFAULT_BROKER")
        
        # Risk limits
        if os.getenv("CANADIAN_TRADING_MAX_POSITION_PCT"):
            max_pos = float(os.getenv("CANADIAN_TRADING_MAX_POSITION_PCT"))
            self.compliance.max_single_position_pct = max_pos
            self.questrade.max_position_size = max_pos
            self.ibkr.max_position_size = max_pos
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler (if configured)
        handlers = [console_handler]
        if self.trading.log_file:
            file_handler = logging.FileHandler(
                self.trading.data_dir / self.trading.log_file
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.trading.log_level),
            handlers=handlers
        )
    
    def get_broker_config(self, broker: str) -> Any:
        """Get configuration for specific broker"""
        broker_configs = {
            "questrade": self.questrade,
            "ibkr": self.ibkr,
            "oanda": self.oanda
        }
        return broker_configs.get(broker.lower())
    
    def validate(self) -> bool:
        """Validate configuration"""
        valid = True
        
        # Check required tokens
        if self.trading.default_broker == "questrade" and not self.questrade.refresh_token:
            logger.error("Questrade refresh token not configured")
            valid = False
        
        if self.trading.default_broker == "oanda" and not self.oanda.api_token:
            logger.error("OANDA API token not configured")
            valid = False
        
        # Validate risk limits
        if self.compliance.max_single_position_pct > 0.5:
            logger.warning("Max position size exceeds 50% - this is risky!")
        
        return valid


# Global configuration instance
config = CanadianTradingConfig()


# Helper functions
def get_config() -> CanadianTradingConfig:
    """Get global configuration instance"""
    return config


def load_config(config_file: str) -> CanadianTradingConfig:
    """Load configuration from file"""
    global config
    config = CanadianTradingConfig(config_file)
    return config


def save_config(config_file: str):
    """Save current configuration to file"""
    config.save_to_file(config_file)