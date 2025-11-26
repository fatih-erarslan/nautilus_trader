"""
Data normalizer for standardizing data from different sources
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re

from ..realtime_manager import DataPoint

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes data from different sources into standardized format"""
    
    # Symbol mappings for different formats
    SYMBOL_MAPPINGS = {
        # Crypto mappings
        'BTC-USD': 'BTC',
        'ETH-USD': 'ETH',
        'SOL-USD': 'SOL',
        'DOGE-USD': 'DOGE',
        
        # Index mappings
        '^GSPC': 'SPX',
        '^DJI': 'DJI',
        '^IXIC': 'NDX',
        
        # Treasury mappings
        '^TNX': '10Y',
        '^FVX': '5Y',
        '^TYX': '30Y',
        '^IRX': '3M',
    }
    
    # Source-specific normalization rules
    SOURCE_RULES = {
        'yahoo_finance': {
            'price_scale': 1.0,
            'volume_scale': 1.0,
            'timestamp_format': 'unix',
            'currency': 'USD'
        },
        'alpha_vantage': {
            'price_scale': 1.0,
            'volume_scale': 1.0,
            'timestamp_format': 'iso',
            'currency': 'USD'
        },
        'finnhub': {
            'price_scale': 1.0,
            'volume_scale': 1.0,
            'timestamp_format': 'unix_ms',
            'currency': 'USD'
        },
        'coinbase': {
            'price_scale': 1.0,
            'volume_scale': 1.0,
            'timestamp_format': 'iso',
            'currency': 'USD'
        }
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Normalization settings
        self.normalize_symbols = self.config.get('normalize_symbols', True)
        self.normalize_timestamps = self.config.get('normalize_timestamps', True)
        self.normalize_prices = self.config.get('normalize_prices', True)
        self.normalize_volumes = self.config.get('normalize_volumes', True)
        
        # Precision settings
        self.price_precision = self.config.get('price_precision', 4)
        self.volume_precision = self.config.get('volume_precision', 0)
        
        # Validation settings
        self.validate_data = self.config.get('validate_data', True)
        self.max_price_change_percent = self.config.get('max_price_change_percent', 50.0)
        
        # Metrics
        self.processed_count = 0
        self.normalized_count = 0
        self.validation_errors = 0
        
        # Cache for price history (for validation)
        self.price_history: Dict[str, List[float]] = {}
        self.max_history_size = 100
    
    def normalize_data_point(self, data_point: DataPoint) -> Optional[DataPoint]:
        """Normalize a single data point"""
        try:
            self.processed_count += 1
            
            # Validate input data if enabled
            if self.validate_data and not self._validate_data_point(data_point):
                self.validation_errors += 1
                return None
            
            # Create normalized copy
            normalized = DataPoint(
                source=self._normalize_source_name(data_point.source),
                symbol=self._normalize_symbol(data_point.symbol),
                timestamp=self._normalize_timestamp(data_point.timestamp),
                price=self._normalize_price(data_point.price, data_point.symbol),
                volume=self._normalize_volume(data_point.volume),
                bid=self._normalize_price(data_point.bid, data_point.symbol) if data_point.bid else None,
                ask=self._normalize_price(data_point.ask, data_point.symbol) if data_point.ask else None,
                latency_ms=data_point.latency_ms,
                sequence_id=data_point.sequence_id,
                metadata=self._normalize_metadata(data_point.metadata, data_point.source)
            )
            
            # Update price history for validation
            self._update_price_history(normalized.symbol, normalized.price)
            
            self.normalized_count += 1
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing data point: {e}")
            self.validation_errors += 1
            return None
    
    def normalize_batch(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Normalize a batch of data points"""
        normalized_points = []
        
        for data_point in data_points:
            normalized = self.normalize_data_point(data_point)
            if normalized:
                normalized_points.append(normalized)
        
        return normalized_points
    
    def _normalize_source_name(self, source: str) -> str:
        """Normalize source name to standard format"""
        # Remove version numbers and normalize
        source_clean = re.sub(r'_v\d+', '', source.lower())
        source_clean = source_clean.replace('-', '_')
        
        # Map common variations
        source_mappings = {
            'yahoo_finance': 'yahoo',
            'yahoo_realtime': 'yahoo',
            'alpha_vantage': 'alphavantage',
            'finnhub_client': 'finnhub',
            'coinbase_feed': 'coinbase',
            'coinbase_pro': 'coinbase'
        }
        
        return source_mappings.get(source_clean, source_clean)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol to standard format"""
        if not self.normalize_symbols:
            return symbol
        
        # Convert to uppercase
        symbol = symbol.upper().strip()
        
        # Apply symbol mappings
        if symbol in self.SYMBOL_MAPPINGS:
            return self.SYMBOL_MAPPINGS[symbol]
        
        # Handle crypto pairs
        if '-USD' in symbol:
            base = symbol.replace('-USD', '')
            if len(base) <= 5:  # Valid crypto symbol
                return base
        
        # Handle forex pairs
        if '=X' in symbol:
            return symbol.replace('=X', '')
        
        # Handle index symbols
        if symbol.startswith('^'):
            return symbol[1:]  # Remove ^ prefix
        
        return symbol
    
    def _normalize_timestamp(self, timestamp: datetime) -> datetime:
        """Normalize timestamp to UTC"""
        if not self.normalize_timestamps:
            return timestamp
        
        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            # Convert to UTC
            timestamp = timestamp.astimezone(timezone.utc)
        
        return timestamp
    
    def _normalize_price(self, price: Optional[float], symbol: str) -> Optional[float]:
        """Normalize price with appropriate precision"""
        if price is None or not self.normalize_prices:
            return price
        
        try:
            # Convert to Decimal for precise arithmetic
            decimal_price = Decimal(str(price))
            
            # Apply symbol-specific scaling if needed
            if symbol in ['BTC', 'ETH']:
                # Crypto prices - higher precision
                precision = 2
            elif symbol.endswith('Y') and len(symbol) <= 3:
                # Treasury yields - higher precision
                precision = 3
            else:
                # Stock prices - standard precision
                precision = self.price_precision
            
            # Round to specified precision
            quantizer = Decimal('0.1') ** precision
            normalized_price = decimal_price.quantize(quantizer)
            
            return float(normalized_price)
            
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"Could not normalize price {price} for {symbol}")
            return price
    
    def _normalize_volume(self, volume: Optional[int]) -> Optional[int]:
        """Normalize volume"""
        if volume is None or not self.normalize_volumes:
            return volume
        
        try:
            # Ensure non-negative
            volume = max(0, int(volume))
            
            # Apply volume precision (usually integers)
            if self.volume_precision == 0:
                return volume
            else:
                # Round to specified precision
                return round(volume, self.volume_precision)
                
        except (ValueError, TypeError):
            logger.warning(f"Could not normalize volume {volume}")
            return volume
    
    def _normalize_metadata(self, metadata: Optional[Dict[str, Any]], source: str) -> Optional[Dict[str, Any]]:
        """Normalize metadata based on source"""
        if metadata is None:
            return None
        
        normalized_metadata = {}
        
        # Source-specific metadata normalization
        if source in ['yahoo', 'yahoo_finance']:
            normalized_metadata.update(self._normalize_yahoo_metadata(metadata))
        elif source in ['alphavantage', 'alpha_vantage']:
            normalized_metadata.update(self._normalize_alphavantage_metadata(metadata))
        elif source == 'finnhub':
            normalized_metadata.update(self._normalize_finnhub_metadata(metadata))
        elif source == 'coinbase':
            normalized_metadata.update(self._normalize_coinbase_metadata(metadata))
        
        # Add common normalized fields
        normalized_metadata['source'] = source
        normalized_metadata['normalized_at'] = datetime.utcnow().isoformat()
        
        return normalized_metadata
    
    def _normalize_yahoo_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Yahoo Finance metadata"""
        normalized = {}
        
        # Map Yahoo-specific fields to standard names
        field_mappings = {
            'regularMarketChange': 'change',
            'regularMarketChangePercent': 'change_percent',
            'regularMarketVolume': 'volume',
            'regularMarketTime': 'market_time',
            'previousClose': 'previous_close'
        }
        
        for yahoo_field, standard_field in field_mappings.items():
            if yahoo_field in metadata:
                normalized[standard_field] = metadata[yahoo_field]
        
        return normalized
    
    def _normalize_alphavantage_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Alpha Vantage metadata"""
        normalized = {}
        
        # Alpha Vantage uses numbered keys
        field_mappings = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'previous_close': 'previous_close',
            'change': 'change',
            'change_percent': 'change_percent'
        }
        
        for av_field, standard_field in field_mappings.items():
            if av_field in metadata:
                normalized[standard_field] = metadata[av_field]
        
        return normalized
    
    def _normalize_finnhub_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Finnhub metadata"""
        normalized = {}
        
        # Finnhub fields
        if 'conditions' in metadata:
            normalized['trade_conditions'] = metadata['conditions']
        
        if 'trade_timestamp_ms' in metadata:
            normalized['trade_time'] = metadata['trade_timestamp_ms']
        
        return normalized
    
    def _normalize_coinbase_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Coinbase metadata"""
        normalized = {}
        
        # Coinbase-specific fields
        field_mappings = {
            'open_24h': 'open_24h',
            'high_24h': 'high_24h',
            'low_24h': 'low_24h',
            'volume_24h': 'volume_24h',
            'volume_30d': 'volume_30d',
            'best_bid_size': 'bid_size',
            'best_ask_size': 'ask_size',
            'side': 'trade_side',
            'trade_id': 'trade_id'
        }
        
        for cb_field, standard_field in field_mappings.items():
            if cb_field in metadata:
                normalized[standard_field] = metadata[cb_field]
        
        return normalized
    
    def _validate_data_point(self, data_point: DataPoint) -> bool:
        """Validate data point for basic consistency"""
        try:
            # Check required fields
            if not data_point.symbol or data_point.price is None:
                return False
            
            # Check price validity
            if data_point.price <= 0:
                return False
            
            # Check for extreme price changes
            if self._has_extreme_price_change(data_point.symbol, data_point.price):
                logger.warning(f"Extreme price change detected for {data_point.symbol}: {data_point.price}")
                return False
            
            # Check volume validity
            if data_point.volume is not None and data_point.volume < 0:
                return False
            
            # Check bid/ask validity
            if data_point.bid is not None and data_point.bid <= 0:
                return False
            
            if data_point.ask is not None and data_point.ask <= 0:
                return False
            
            # Check bid <= price <= ask if all present
            if (data_point.bid is not None and data_point.ask is not None and 
                data_point.bid > data_point.ask):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data point: {e}")
            return False
    
    def _has_extreme_price_change(self, symbol: str, price: float) -> bool:
        """Check if price change is extreme compared to history"""
        if symbol not in self.price_history or not self.price_history[symbol]:
            return False
        
        recent_prices = self.price_history[symbol][-10:]  # Last 10 prices
        avg_price = sum(recent_prices) / len(recent_prices)
        
        if avg_price > 0:
            change_percent = abs((price - avg_price) / avg_price) * 100
            return change_percent > self.max_price_change_percent
        
        return False
    
    def _update_price_history(self, symbol: str, price: float) -> None:
        """Update price history for validation"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Limit history size
        if len(self.price_history[symbol]) > self.max_history_size:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history_size:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get normalization metrics"""
        success_rate = self.normalized_count / self.processed_count if self.processed_count > 0 else 0
        
        return {
            'processed_count': self.processed_count,
            'normalized_count': self.normalized_count,
            'validation_errors': self.validation_errors,
            'success_rate': success_rate,
            'symbols_tracked': len(self.price_history),
            'config': {
                'normalize_symbols': self.normalize_symbols,
                'normalize_timestamps': self.normalize_timestamps,
                'normalize_prices': self.normalize_prices,
                'normalize_volumes': self.normalize_volumes,
                'price_precision': self.price_precision,
                'volume_precision': self.volume_precision,
                'validate_data': self.validate_data
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset processing metrics"""
        self.processed_count = 0
        self.normalized_count = 0
        self.validation_errors = 0
    
    def clear_history(self) -> None:
        """Clear price history cache"""
        self.price_history.clear()