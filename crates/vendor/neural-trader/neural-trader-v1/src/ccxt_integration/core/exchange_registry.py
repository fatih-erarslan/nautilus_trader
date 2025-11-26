"""
Exchange Registry

Manages exchange configurations and provides a centralized registry
for all supported exchanges with their capabilities and settings.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExchangeCapabilities:
    """Capabilities and features supported by an exchange"""
    spot_trading: bool = True
    futures_trading: bool = False
    margin_trading: bool = False
    options_trading: bool = False
    websocket_support: bool = False
    sandbox_available: bool = False
    deposit_withdrawal: bool = False
    order_types: List[str] = field(default_factory=lambda: ['market', 'limit'])
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h', '1d'])
    max_request_rate: int = 10  # requests per second
    requires_api_key: bool = True
    requires_secret: bool = True
    requires_password: bool = False
    requires_uid: bool = False


@dataclass
class ExchangeMetadata:
    """Metadata about an exchange"""
    name: str
    display_name: str
    country: str
    url: str
    capabilities: ExchangeCapabilities
    trading_fees: Dict[str, float] = field(default_factory=dict)
    supported_quote_currencies: List[str] = field(default_factory=lambda: ['USDT', 'USD', 'BTC', 'ETH'])
    min_order_sizes: Dict[str, float] = field(default_factory=dict)
    active: bool = True
    notes: str = ""


class ExchangeRegistry:
    """
    Central registry for all exchange configurations and metadata.
    """
    
    # Default exchange configurations
    DEFAULT_EXCHANGES = {
        'binance': ExchangeMetadata(
            name='binance',
            display_name='Binance',
            country='Global',
            url='https://www.binance.com',
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                futures_trading=True,
                margin_trading=True,
                websocket_support=True,
                sandbox_available=True,
                deposit_withdrawal=True,
                order_types=['market', 'limit', 'stop', 'stop_limit', 'take_profit', 'take_profit_limit'],
                timeframes=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
                max_request_rate=20
            ),
            trading_fees={'maker': 0.001, 'taker': 0.001},
            supported_quote_currencies=['USDT', 'BUSD', 'BTC', 'ETH', 'BNB'],
            min_order_sizes={'BTC': 0.00001, 'ETH': 0.0001}
        ),
        
        'coinbase': ExchangeMetadata(
            name='coinbase',
            display_name='Coinbase Pro',
            country='USA',
            url='https://pro.coinbase.com',
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                websocket_support=True,
                sandbox_available=True,
                deposit_withdrawal=True,
                order_types=['market', 'limit', 'stop'],
                timeframes=['1m', '5m', '15m', '1h', '6h', '1d'],
                max_request_rate=10
            ),
            trading_fees={'maker': 0.005, 'taker': 0.005},
            supported_quote_currencies=['USD', 'USDC', 'EUR', 'GBP', 'BTC'],
            min_order_sizes={'BTC': 0.001, 'ETH': 0.01}
        ),
        
        'kraken': ExchangeMetadata(
            name='kraken',
            display_name='Kraken',
            country='USA',
            url='https://www.kraken.com',
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                futures_trading=True,
                margin_trading=True,
                websocket_support=True,
                deposit_withdrawal=True,
                order_types=['market', 'limit', 'stop-loss', 'take-profit', 'stop-loss-limit', 'take-profit-limit'],
                timeframes=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
                max_request_rate=15
            ),
            trading_fees={'maker': 0.0016, 'taker': 0.0026},
            supported_quote_currencies=['USD', 'EUR', 'GBP', 'CAD', 'JPY', 'BTC', 'ETH'],
            min_order_sizes={'BTC': 0.0001, 'ETH': 0.002}
        ),
        
        'bybit': ExchangeMetadata(
            name='bybit',
            display_name='Bybit',
            country='Singapore',
            url='https://www.bybit.com',
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                futures_trading=True,
                websocket_support=True,
                sandbox_available=True,
                order_types=['market', 'limit', 'stop', 'stop_limit'],
                timeframes=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'],
                max_request_rate=50
            ),
            trading_fees={'maker': 0.001, 'taker': 0.001},
            supported_quote_currencies=['USDT', 'USD', 'BTC', 'ETH'],
            min_order_sizes={'BTC': 0.0001, 'ETH': 0.001}
        ),
        
        'kucoin': ExchangeMetadata(
            name='kucoin',
            display_name='KuCoin',
            country='Seychelles',
            url='https://www.kucoin.com',
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                futures_trading=True,
                margin_trading=True,
                websocket_support=True,
                sandbox_available=True,
                deposit_withdrawal=True,
                order_types=['market', 'limit', 'stop', 'stop_limit'],
                timeframes=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w'],
                max_request_rate=30,
                requires_password=True
            ),
            trading_fees={'maker': 0.001, 'taker': 0.001},
            supported_quote_currencies=['USDT', 'BTC', 'ETH', 'KCS'],
            min_order_sizes={'BTC': 0.00001, 'ETH': 0.0001}
        )
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the exchange registry.
        
        Args:
            config_path: Optional path to additional exchange configurations
        """
        self.exchanges: Dict[str, ExchangeMetadata] = self.DEFAULT_EXCHANGES.copy()
        self.active_exchanges: Set[str] = set()
        self.config_path = config_path
        
        if config_path and config_path.exists():
            self._load_custom_configurations()
            
    def _load_custom_configurations(self) -> None:
        """Load custom exchange configurations from file."""
        if not self.config_path:
            return
            
        try:
            if self.config_path.suffix == '.json':
                with open(self.config_path, 'r') as f:
                    custom_configs = json.load(f)
            elif self.config_path.suffix in ['.yaml', '.yml']:
                with open(self.config_path, 'r') as f:
                    custom_configs = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {self.config_path}")
                return
                
            # Merge custom configurations
            for exchange_name, config in custom_configs.items():
                if exchange_name in self.exchanges:
                    # Update existing exchange
                    self._update_exchange_metadata(exchange_name, config)
                else:
                    # Add new exchange
                    self._add_custom_exchange(exchange_name, config)
                    
            logger.info(f"Loaded custom configurations from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading custom configurations: {str(e)}")
            
    def _update_exchange_metadata(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Update existing exchange metadata with custom configuration."""
        metadata = self.exchanges[exchange_name]
        
        for key, value in config.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
            elif hasattr(metadata.capabilities, key):
                setattr(metadata.capabilities, key, value)
                
    def _add_custom_exchange(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Add a new custom exchange from configuration."""
        capabilities_data = config.pop('capabilities', {})
        capabilities = ExchangeCapabilities(**capabilities_data)
        
        metadata = ExchangeMetadata(
            name=exchange_name,
            capabilities=capabilities,
            **config
        )
        
        self.exchanges[exchange_name] = metadata
        
    def get_exchange(self, exchange_name: str) -> Optional[ExchangeMetadata]:
        """
        Get metadata for a specific exchange.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            ExchangeMetadata or None if not found
        """
        return self.exchanges.get(exchange_name)
        
    def list_exchanges(self, filter_active: bool = True) -> List[str]:
        """
        List all available exchanges.
        
        Args:
            filter_active: Only return active exchanges
            
        Returns:
            List of exchange names
        """
        if filter_active:
            return [name for name, metadata in self.exchanges.items() if metadata.active]
        return list(self.exchanges.keys())
        
    def get_exchanges_by_capability(self, capability: str) -> List[str]:
        """
        Get exchanges that support a specific capability.
        
        Args:
            capability: Capability name (e.g., 'futures_trading', 'websocket_support')
            
        Returns:
            List of exchange names
        """
        result = []
        
        for name, metadata in self.exchanges.items():
            if metadata.active and hasattr(metadata.capabilities, capability):
                if getattr(metadata.capabilities, capability):
                    result.append(name)
                    
        return result
        
    def get_exchanges_by_quote_currency(self, currency: str) -> List[str]:
        """
        Get exchanges that support a specific quote currency.
        
        Args:
            currency: Quote currency (e.g., 'USDT', 'USD', 'BTC')
            
        Returns:
            List of exchange names
        """
        result = []
        
        for name, metadata in self.exchanges.items():
            if metadata.active and currency in metadata.supported_quote_currencies:
                result.append(name)
                
        return result
        
    def register_exchange(self, metadata: ExchangeMetadata) -> None:
        """
        Register a new exchange or update existing one.
        
        Args:
            metadata: Exchange metadata
        """
        self.exchanges[metadata.name] = metadata
        logger.info(f"Registered exchange: {metadata.name}")
        
    def activate_exchange(self, exchange_name: str) -> bool:
        """
        Mark an exchange as active.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            True if successful, False if exchange not found
        """
        if exchange_name in self.exchanges:
            self.exchanges[exchange_name].active = True
            self.active_exchanges.add(exchange_name)
            logger.info(f"Activated exchange: {exchange_name}")
            return True
        return False
        
    def deactivate_exchange(self, exchange_name: str) -> bool:
        """
        Mark an exchange as inactive.
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            True if successful, False if exchange not found
        """
        if exchange_name in self.exchanges:
            self.exchanges[exchange_name].active = False
            self.active_exchanges.discard(exchange_name)
            logger.info(f"Deactivated exchange: {exchange_name}")
            return True
        return False
        
    def get_best_exchange_for_pair(self, base: str, quote: str) -> Optional[str]:
        """
        Get the best exchange for trading a specific pair.
        
        Args:
            base: Base currency (e.g., 'BTC')
            quote: Quote currency (e.g., 'USDT')
            
        Returns:
            Exchange name or None
        """
        # Filter exchanges that support the quote currency
        candidates = self.get_exchanges_by_quote_currency(quote)
        
        if not candidates:
            return None
            
        # Sort by trading fees (lowest first)
        best_exchange = None
        lowest_fee = float('inf')
        
        for exchange_name in candidates:
            metadata = self.exchanges[exchange_name]
            taker_fee = metadata.trading_fees.get('taker', float('inf'))
            
            if taker_fee < lowest_fee:
                lowest_fee = taker_fee
                best_exchange = exchange_name
                
        return best_exchange
        
    def export_registry(self, output_path: Path) -> None:
        """
        Export the registry to a file.
        
        Args:
            output_path: Path to output file
        """
        data = {}
        
        for name, metadata in self.exchanges.items():
            data[name] = {
                'display_name': metadata.display_name,
                'country': metadata.country,
                'url': metadata.url,
                'active': metadata.active,
                'capabilities': asdict(metadata.capabilities),
                'trading_fees': metadata.trading_fees,
                'supported_quote_currencies': metadata.supported_quote_currencies,
                'min_order_sizes': metadata.min_order_sizes,
                'notes': metadata.notes
            }
            
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                
        logger.info(f"Exported registry to {output_path}")
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        total_exchanges = len(self.exchanges)
        active_exchanges = sum(1 for m in self.exchanges.values() if m.active)
        
        capabilities_count = {}
        for metadata in self.exchanges.values():
            if metadata.active:
                for attr in ['spot_trading', 'futures_trading', 'margin_trading', 
                           'options_trading', 'websocket_support', 'sandbox_available']:
                    if getattr(metadata.capabilities, attr):
                        capabilities_count[attr] = capabilities_count.get(attr, 0) + 1
                        
        return {
            'total_exchanges': total_exchanges,
            'active_exchanges': active_exchanges,
            'capabilities': capabilities_count,
            'exchanges_with_sandbox': self.get_exchanges_by_capability('sandbox_available'),
            'exchanges_with_websocket': self.get_exchanges_by_capability('websocket_support'),
            'exchanges_with_futures': self.get_exchanges_by_capability('futures_trading')
        }