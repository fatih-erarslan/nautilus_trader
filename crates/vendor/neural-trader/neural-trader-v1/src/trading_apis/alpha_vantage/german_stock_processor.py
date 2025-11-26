"""
German Stock Data Processor for Alpha Vantage
Handles German stock data processing, validation, and transformation
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass 
class GermanStockData:
    """Standardized German stock data structure"""
    symbol: str
    name: Optional[str]
    exchange: str
    currency: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float]
    timestamp: datetime
    trading_session: str  # 'pre_market', 'market', 'after_hours'
    raw_data: Dict[str, Any]


class GermanStockProcessor:
    """
    Processor for German stock data from Alpha Vantage
    Handles data validation, transformation, and enrichment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # German market configuration
        self.german_exchanges = {
            'XETRA': {
                'suffix': '.DE',
                'currency': 'EUR',
                'timezone': 'Europe/Berlin',
                'trading_hours': {'start': '09:00', 'end': '17:30'},
                'pre_market': {'start': '08:00', 'end': '09:00'},
                'after_hours': {'start': '17:30', 'end': '22:00'}
            },
            'STUTTGART': {
                'suffix': '.STU',
                'currency': 'EUR',
                'timezone': 'Europe/Berlin',
                'trading_hours': {'start': '08:00', 'end': '22:00'},
                'pre_market': None,
                'after_hours': None
            }
        }
        
        # DAX 40 components with company info
        self.dax_components = {
            'SAP.DE': {'name': 'SAP SE', 'sector': 'Technology'},
            'ASML.DE': {'name': 'ASML Holding NV', 'sector': 'Technology'},
            'SIE.DE': {'name': 'Siemens AG', 'sector': 'Industrial'},
            'BMW.DE': {'name': 'Bayerische Motoren Werke AG', 'sector': 'Consumer Discretionary'},
            'ALV.DE': {'name': 'Allianz SE', 'sector': 'Financial Services'},
            'BAS.DE': {'name': 'BASF SE', 'sector': 'Materials'},
            'VOW3.DE': {'name': 'Volkswagen AG', 'sector': 'Consumer Discretionary'},
            'DTE.DE': {'name': 'Deutsche Telekom AG', 'sector': 'Communication Services'},
            'MUV2.DE': {'name': 'Munich Re', 'sector': 'Financial Services'},
            'DAI.DE': {'name': 'Mercedes-Benz Group AG', 'sector': 'Consumer Discretionary'},
            'DB1.DE': {'name': 'Deutsche Börse AG', 'sector': 'Financial Services'},
            'HEN3.DE': {'name': 'Henkel AG & Co. KGaA', 'sector': 'Consumer Staples'}
        }
        
        # Validation thresholds
        self.validation_config = {
            'min_price': 0.01,
            'max_price': 10000.0,
            'min_volume': 0,
            'max_volume': 1000000000,
            'max_change_percent': 50.0,  # 50% daily change threshold
            'stale_data_minutes': 60
        }
        
        logger.info("German Stock Processor initialized")
    
    def process_alpha_vantage_quote(self, raw_data: Dict[str, Any], symbol: str) -> Optional[GermanStockData]:
        """Process Alpha Vantage quote data into standardized format"""
        try:
            # Extract data from Alpha Vantage Global Quote format
            if not raw_data:
                logger.warning(f"No data for symbol {symbol}")
                return None
            
            # Alpha Vantage Global Quote fields
            quote_symbol = raw_data.get('01. Symbol', '')
            price = float(raw_data.get('05. Price', 0))
            change = float(raw_data.get('09. Change', 0))
            change_percent = raw_data.get('10. Change Percent', '0%')
            volume = int(raw_data.get('06. Volume', 0))
            latest_trading_day = raw_data.get('07. Latest Trading Day', '')
            
            # Parse change percentage
            if isinstance(change_percent, str):
                change_percent = float(change_percent.replace('%', ''))
            else:
                change_percent = float(change_percent)
            
            # Determine exchange from symbol
            exchange = self._get_exchange_from_symbol(symbol)
            
            # Get company name if available
            company_name = self.dax_components.get(symbol, {}).get('name')
            
            # Parse timestamp
            try:
                if latest_trading_day:
                    # Parse date and set to market close time (17:30 CET)
                    date_part = datetime.fromisoformat(latest_trading_day)
                    timestamp = date_part.replace(hour=17, minute=30, second=0, microsecond=0)
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            # Determine trading session
            trading_session = self._get_trading_session(timestamp, exchange)
            
            # Create standardized data structure
            german_stock_data = GermanStockData(
                symbol=symbol,
                name=company_name,
                exchange=exchange,
                currency=self.german_exchanges[exchange]['currency'],
                price=price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                market_cap=None,  # Not provided in quote
                timestamp=timestamp,
                trading_session=trading_session,
                raw_data=raw_data
            )
            
            # Validate data
            if self.validate_stock_data(german_stock_data):
                return german_stock_data
            else:
                logger.warning(f"Data validation failed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing quote for {symbol}: {e}")
            return None
    
    def process_intraday_data(self, raw_data: Dict[str, Any], symbol: str) -> Optional[pd.DataFrame]:
        """Process intraday data into DataFrame"""
        try:
            if not raw_data or 'time_series' not in raw_data:
                return None
            
            time_series = raw_data['time_series']
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Add German market metadata
            exchange = self._get_exchange_from_symbol(symbol)
            df['exchange'] = exchange
            df['currency'] = self.german_exchanges[exchange]['currency']
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing intraday data for {symbol}: {e}")
            return None
    
    def validate_stock_data(self, data: GermanStockData) -> bool:
        """Validate German stock data"""
        try:
            # Price validation
            if data.price < self.validation_config['min_price'] or data.price > self.validation_config['max_price']:
                logger.warning(f"Price out of range for {data.symbol}: {data.price}")
                return False
            
            # Volume validation
            if data.volume < self.validation_config['min_volume'] or data.volume > self.validation_config['max_volume']:
                logger.warning(f"Volume out of range for {data.symbol}: {data.volume}")
                return False
            
            # Change validation
            if abs(data.change_percent) > self.validation_config['max_change_percent']:
                logger.warning(f"Change percent too large for {data.symbol}: {data.change_percent}%")
                return False
            
            # Data freshness validation
            if data.timestamp:
                age_minutes = (datetime.now() - data.timestamp).total_seconds() / 60
                if age_minutes > self.validation_config['stale_data_minutes']:
                    logger.warning(f"Stale data for {data.symbol}: {age_minutes:.1f} minutes old")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {data.symbol}: {e}")
            return False
    
    def _get_exchange_from_symbol(self, symbol: str) -> str:
        """Determine exchange from symbol suffix"""
        for exchange, config in self.german_exchanges.items():
            if symbol.endswith(config['suffix']):
                return exchange
        return 'XETRA'  # Default to XETRA
    
    def _get_trading_session(self, timestamp: datetime, exchange: str) -> str:
        """Determine trading session based on timestamp"""
        if exchange not in self.german_exchanges:
            return 'market'
        
        config = self.german_exchanges[exchange]
        time_str = timestamp.strftime('%H:%M')
        
        # Check pre-market
        if config.get('pre_market'):
            pre_start = config['pre_market']['start']
            pre_end = config['pre_market']['end']
            if pre_start <= time_str < pre_end:
                return 'pre_market'
        
        # Check main market
        market_start = config['trading_hours']['start']
        market_end = config['trading_hours']['end']
        if market_start <= time_str < market_end:
            return 'market'
        
        # Check after-hours
        if config.get('after_hours'):
            after_start = config['after_hours']['start']
            after_end = config['after_hours']['end']
            if after_start <= time_str < after_end:
                return 'after_hours'
        
        return 'closed'
    
    def get_dax_symbols(self) -> List[str]:
        """Get list of DAX component symbols"""
        return list(self.dax_components.keys())
    
    def is_dax_component(self, symbol: str) -> bool:
        """Check if symbol is a DAX component"""
        return symbol in self.dax_components
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information for a symbol"""
        return self.dax_components.get(symbol)
    
    async def process_batch_quotes(self, raw_quotes: Dict[str, Dict[str, Any]]) -> List[GermanStockData]:
        """Process batch of quotes"""
        results = []
        
        for symbol, quote_data in raw_quotes.items():
            if 'error' in quote_data:
                logger.warning(f"Error in quote data for {symbol}: {quote_data['error']}")
                continue
            
            processed_data = self.process_alpha_vantage_quote(quote_data, symbol)
            if processed_data:
                results.append(processed_data)
        
        return results
    
    def calculate_performance_metrics(self, data: List[GermanStockData]) -> Dict[str, Any]:
        """Calculate performance metrics for German stocks"""
        if not data:
            return {}
        
        # Calculate metrics
        prices = [d.price for d in data]
        changes = [d.change_percent for d in data]
        volumes = [d.volume for d in data]
        
        metrics = {
            'total_symbols': len(data),
            'avg_price': sum(prices) / len(prices),
            'avg_change_percent': sum(changes) / len(changes),
            'total_volume': sum(volumes),
            'positive_movers': len([d for d in data if d.change > 0]),
            'negative_movers': len([d for d in data if d.change < 0]),
            'top_gainer': max(data, key=lambda x: x.change_percent) if data else None,
            'top_loser': min(data, key=lambda x: x.change_percent) if data else None,
            'timestamp': datetime.now()
        }
        
        return metrics
    
    def convert_to_trading_format(self, data: GermanStockData) -> Dict[str, Any]:
        """Convert to format expected by trading strategies"""
        return {
            'ticker': data.symbol,
            'price': data.price,
            'change': data.change,
            'change_percent': data.change_percent / 100,  # Convert to decimal
            'volume': data.volume,
            'currency': data.currency,
            'exchange': data.exchange,
            'timestamp': data.timestamp.isoformat(),
            'trading_session': data.trading_session,
            'is_dax_component': self.is_dax_component(data.symbol),
            'company_info': self.get_company_info(data.symbol)
        }