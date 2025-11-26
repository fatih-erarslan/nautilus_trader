"""
Bond feed handler for fixed income market data
Focuses on Treasury yields and bond pricing
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from collections import defaultdict

from ..realtime_manager import DataFeed, DataPoint, RealtimeManager
from ..alpha_vantage import AlphaVantageSource
from ..yahoo_realtime import YahooRealtimeSource

logger = logging.getLogger(__name__)


class BondFeed(DataFeed):
    """Bond feed handler for fixed income data"""
    
    # Common Treasury and Bond instruments
    TREASURY_SYMBOLS = [
        "^TNX",    # 10-Year Treasury Note Yield
        "^FVX",    # 5-Year Treasury Note Yield  
        "^TYX",    # 30-Year Treasury Bond Yield
        "^IRX",    # 3-Month Treasury Bill Yield
        "^TNS",    # 2-Year Treasury Note Yield
        "^TNF",    # 1-Year Treasury Note Yield
    ]
    
    BOND_ETFS = [
        "TLT",     # 20+ Year Treasury Bond ETF
        "IEF",     # 7-10 Year Treasury Bond ETF
        "SHY",     # 1-3 Year Treasury Bond ETF
        "TBT",     # UltraShort 20+ Year Treasury ProShares
        "TMF",     # 3x Long 20+ Year Treasury ETF
        "TMV",     # 3x Inverse 20+ Year Treasury ETF
        "AGG",     # Core Total US Bond Market ETF
        "BND",     # Total Bond Market ETF
        "HYG",     # High Yield Corporate Bond ETF
        "LQD",     # Investment Grade Corporate Bond ETF
    ]
    
    def __init__(self, symbols: List[str] = None, config: Dict[str, Any] = None):
        # Default to Treasury yields and major bond ETFs
        default_symbols = self.TREASURY_SYMBOLS + self.BOND_ETFS
        self.symbols = set(symbols or default_symbols)
        self.config = config or {}
        
        # Data sources
        self.sources: Dict[str, Any] = {}
        self.manager = RealtimeManager()
        
        # Feed state
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Data tracking
        self.latest_data: Dict[str, DataPoint] = {}
        self.yield_curve_data: Dict[str, float] = {}
        self.data_callbacks: List[callable] = []
        
        # Source priorities
        self.source_priorities = {
            'alpha_vantage': 2,   # Good for bond data
            'yahoo_realtime': 1   # Backup for ETFs
        }
        
        # Configuration
        self.enable_yahoo = self.config.get('enable_yahoo', True)
        self.enable_alpha_vantage = self.config.get('enable_alpha_vantage', False)
        self.alpha_vantage_api_key = self.config.get('alpha_vantage_api_key')
        
        # Bond-specific settings
        self.update_interval = self.config.get('update_interval', 5.0)  # Bonds update less frequently
        self.track_yield_curve = self.config.get('track_yield_curve', True)
    
    async def start(self) -> None:
        """Start the bond feed"""
        if self.is_running:
            return
        
        logger.info(f"Starting bond feed for {len(self.symbols)} instruments")
        
        # Initialize sources
        await self._initialize_sources()
        
        # Connect sources
        await self._connect_sources()
        
        # Subscribe to symbols
        await self._subscribe_to_symbols()
        
        # Start yield curve tracking
        if self.track_yield_curve:
            self.tasks.append(asyncio.create_task(self._yield_curve_updater()))
        
        self.is_running = True
        logger.info("Bond feed started successfully")
    
    async def stop(self) -> None:
        """Stop the bond feed"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown manager
        await self.manager.shutdown()
        
        logger.info("Bond feed stopped")
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to additional bond instruments"""
        new_symbols = set(symbols) - self.symbols
        if not new_symbols:
            return
        
        self.symbols.update(new_symbols)
        
        # Subscribe to new symbols on active sources
        for source_name, source in self.sources.items():
            if hasattr(source, 'subscribe'):
                await source.subscribe(list(new_symbols))
        
        logger.info(f"Subscribed to {len(new_symbols)} new bond instruments")
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from bond instruments"""
        symbols_to_remove = set(symbols) & self.symbols
        if not symbols_to_remove:
            return
        
        self.symbols -= symbols_to_remove
        
        # Unsubscribe from sources
        for source_name, source in self.sources.items():
            if hasattr(source, 'unsubscribe'):
                await source.unsubscribe(list(symbols_to_remove))
        
        # Clean up data
        for symbol in symbols_to_remove:
            self.latest_data.pop(symbol, None)
            self.yield_curve_data.pop(symbol, None)
        
        logger.info(f"Unsubscribed from {len(symbols_to_remove)} bond instruments")
    
    async def _initialize_sources(self) -> None:
        """Initialize bond data sources"""
        # Alpha Vantage (good for Treasury data)
        if self.enable_alpha_vantage and self.alpha_vantage_api_key:
            alpha_source = AlphaVantageSource(
                api_key=self.alpha_vantage_api_key,
                rate_limit=5
            )
            alpha_source.set_data_callback(self._handle_data_point)
            self.sources['alpha_vantage'] = alpha_source
            self.manager.add_source(alpha_source)
        
        # Yahoo Finance (for bond ETFs)
        if self.enable_yahoo:
            yahoo_source = YahooRealtimeSource(
                use_websocket=False,
                update_interval=self.update_interval
            )
            yahoo_source.set_data_callback(self._handle_data_point)
            self.sources['yahoo_realtime'] = yahoo_source
            self.manager.add_source(yahoo_source)
        
        logger.info(f"Initialized {len(self.sources)} bond data sources")
    
    async def _connect_sources(self) -> None:
        """Connect to all bond data sources"""
        results = await self.manager.connect_all()
        
        connected_sources = [name for name, success in results.items() if success]
        failed_sources = [name for name, success in results.items() if not success]
        
        if connected_sources:
            logger.info(f"Connected to bond sources: {connected_sources}")
        
        if failed_sources:
            logger.warning(f"Failed to connect to bond sources: {failed_sources}")
        
        if not connected_sources:
            raise RuntimeError("Failed to connect to any bond data sources")
    
    async def _subscribe_to_symbols(self) -> None:
        """Subscribe to bond instruments on all sources"""
        symbols_list = list(self.symbols)
        
        for source_name, source in self.sources.items():
            try:
                await source.subscribe(symbols_list)
                logger.info(f"Subscribed to {len(symbols_list)} bonds on {source_name}")
            except Exception as e:
                logger.error(f"Failed to subscribe to {source_name}: {e}")
    
    async def _handle_data_point(self, data_point: DataPoint) -> None:
        """Handle incoming bond data point"""
        try:
            symbol = data_point.symbol
            source_name = data_point.source
            
            # Update yield curve data for Treasury instruments
            if symbol in self.TREASURY_SYMBOLS and self.track_yield_curve:
                self.yield_curve_data[symbol] = data_point.price
            
            # Calculate duration and convexity estimates if possible
            enhanced_data_point = await self._enhance_bond_data(data_point)
            
            # Update latest data if this source has higher priority
            if (symbol not in self.latest_data or 
                self._get_source_priority(source_name) > self._get_source_priority(self.latest_data[symbol].source)):
                
                self.latest_data[symbol] = enhanced_data_point
            
            # Notify callbacks
            for callback in self.data_callbacks:
                try:
                    await callback(enhanced_data_point)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error handling bond data point: {e}")
    
    async def _enhance_bond_data(self, data_point: DataPoint) -> DataPoint:
        """Enhance bond data with calculated metrics"""
        # Add bond-specific metadata
        if data_point.metadata is None:
            data_point.metadata = {}
        
        symbol = data_point.symbol
        price = data_point.price
        
        # Add instrument type
        if symbol in self.TREASURY_SYMBOLS:
            data_point.metadata['instrument_type'] = 'treasury_yield'
            data_point.metadata['yield_value'] = price
        elif symbol in self.BOND_ETFS:
            data_point.metadata['instrument_type'] = 'bond_etf'
            data_point.metadata['nav'] = price
        else:
            data_point.metadata['instrument_type'] = 'bond'
        
        # Calculate yield spread to 10-year if available
        if '^TNX' in self.yield_curve_data and symbol != '^TNX':
            ten_year_yield = self.yield_curve_data['^TNX']
            if symbol in self.TREASURY_SYMBOLS:
                spread = price - ten_year_yield
                data_point.metadata['spread_to_10y'] = spread
        
        return data_point
    
    async def _yield_curve_updater(self) -> None:
        """Periodically update yield curve calculations"""
        while self.is_running:
            try:
                # Calculate yield curve metrics
                curve_metrics = self._calculate_yield_curve_metrics()
                
                if curve_metrics:
                    # Create synthetic data point for yield curve
                    curve_data_point = DataPoint(
                        source="bond_feed_calculated",
                        symbol="YIELD_CURVE",
                        timestamp=datetime.now(),
                        price=curve_metrics.get('slope', 0),
                        volume=0,
                        metadata=curve_metrics
                    )
                    
                    # Store and notify callbacks
                    self.latest_data['YIELD_CURVE'] = curve_data_point
                    
                    for callback in self.data_callbacks:
                        try:
                            await callback(curve_data_point)
                        except Exception as e:
                            logger.error(f"Yield curve callback error: {e}")
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Yield curve updater error: {e}")
    
    def _calculate_yield_curve_metrics(self) -> Optional[Dict[str, Any]]:
        """Calculate yield curve slope, steepness, and curvature"""
        try:
            # Get yield data
            yields = {}
            maturities = {
                '^IRX': 0.25,   # 3-month
                '^TNF': 1.0,    # 1-year
                '^TNS': 2.0,    # 2-year
                '^FVX': 5.0,    # 5-year
                '^TNX': 10.0,   # 10-year
                '^TYX': 30.0,   # 30-year
            }
            
            for symbol, maturity in maturities.items():
                if symbol in self.yield_curve_data:
                    yields[maturity] = self.yield_curve_data[symbol]
            
            if len(yields) < 3:
                return None
            
            # Sort by maturity
            sorted_yields = sorted(yields.items())
            
            # Calculate slope (30Y - 3M)
            slope = None
            if 30.0 in yields and 0.25 in yields:
                slope = yields[30.0] - yields[0.25]
            
            # Calculate steepness (10Y - 2Y)
            steepness = None
            if 10.0 in yields and 2.0 in yields:
                steepness = yields[10.0] - yields[2.0]
            
            # Calculate curvature (2*5Y - 2Y - 10Y)
            curvature = None
            if all(m in yields for m in [2.0, 5.0, 10.0]):
                curvature = 2 * yields[5.0] - yields[2.0] - yields[10.0]
            
            return {
                'slope': slope,
                'steepness': steepness,
                'curvature': curvature,
                'yields': yields,
                'curve_shape': self._classify_curve_shape(sorted_yields)
            }
            
        except Exception as e:
            logger.error(f"Error calculating yield curve metrics: {e}")
            return None
    
    def _classify_curve_shape(self, sorted_yields: List[tuple]) -> str:
        """Classify the yield curve shape"""
        if len(sorted_yields) < 3:
            return "insufficient_data"
        
        # Simple classification based on slope
        short_yield = sorted_yields[0][1]
        long_yield = sorted_yields[-1][1]
        
        if long_yield > short_yield + 0.5:
            return "normal"
        elif abs(long_yield - short_yield) <= 0.5:
            return "flat"
        else:
            return "inverted"
    
    def _get_source_priority(self, source_name: str) -> int:
        """Get priority for a bond data source"""
        return self.source_priorities.get(source_name, 0)
    
    def add_callback(self, callback: callable) -> None:
        """Add data callback"""
        self.data_callbacks.append(callback)
    
    def remove_callback(self, callback: callable) -> None:
        """Remove data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def get_latest_yield(self, symbol: str) -> Optional[float]:
        """Get latest yield for a Treasury instrument"""
        data_point = self.latest_data.get(symbol)
        if data_point and symbol in self.TREASURY_SYMBOLS:
            return data_point.price
        return None
    
    def get_yield_curve(self) -> Dict[float, float]:
        """Get current yield curve data"""
        curve_data = {}
        maturities = {
            '^IRX': 0.25,   # 3-month
            '^TNF': 1.0,    # 1-year
            '^TNS': 2.0,    # 2-year
            '^FVX': 5.0,    # 5-year
            '^TNX': 10.0,   # 10-year
            '^TYX': 30.0,   # 30-year
        }
        
        for symbol, maturity in maturities.items():
            yield_value = self.get_latest_yield(symbol)
            if yield_value is not None:
                curve_data[maturity] = yield_value
        
        return curve_data
    
    def get_yield_curve_metrics(self) -> Optional[Dict[str, Any]]:
        """Get yield curve slope, steepness, and curvature"""
        curve_data = self.latest_data.get('YIELD_CURVE')
        if curve_data and curve_data.metadata:
            return curve_data.metadata
        return None
    
    async def get_quote(self, symbol: str) -> Optional[DataPoint]:
        """Get real-time quote for a bond instrument"""
        # Try to get from cache first
        if symbol in self.latest_data:
            data_point = self.latest_data[symbol]
            # Return if recent (within 30 seconds for bonds)
            if (datetime.now() - data_point.timestamp).total_seconds() < 30:
                return data_point
        
        # Fetch fresh data from highest priority source
        for source_name in sorted(self.sources.keys(), 
                                 key=lambda x: self.source_priorities.get(x, 0), 
                                 reverse=True):
            
            source = self.sources[source_name]
            if hasattr(source, 'fetch_quote'):
                try:
                    quote = await source.fetch_quote(symbol)
                    if quote:
                        enhanced_quote = await self._enhance_bond_data(quote)
                        self.latest_data[symbol] = enhanced_quote
                        return enhanced_quote
                except Exception as e:
                    logger.error(f"Error fetching bond quote from {source_name}: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bond feed metrics"""
        metrics = {
            'feed_type': 'bond',
            'is_running': self.is_running,
            'instruments_count': len(self.symbols),
            'instruments': list(self.symbols),
            'sources_count': len(self.sources),
            'data_points_count': len(self.latest_data),
            'callbacks_count': len(self.data_callbacks),
            'yield_curve_points': len(self.yield_curve_data),
            'track_yield_curve': self.track_yield_curve
        }
        
        # Add source metrics
        for source_name, source in self.sources.items():
            if hasattr(source, 'get_metrics'):
                metrics[f'{source_name}_metrics'] = source.get_metrics()
        
        # Add yield curve metrics
        curve_metrics = self.get_yield_curve_metrics()
        if curve_metrics:
            metrics['yield_curve_metrics'] = curve_metrics
        
        return metrics
    
    def get_symbols(self) -> List[str]:
        """Get list of subscribed bond instruments"""
        return list(self.symbols)
    
    def is_symbol_subscribed(self, symbol: str) -> bool:
        """Check if bond instrument is subscribed"""
        return symbol in self.symbols
    
    def is_market_open(self) -> bool:
        """Check if bond market is open"""
        # Bond markets generally follow stock market hours
        from datetime import time
        now = datetime.now()
        
        # Weekend check
        if now.weekday() >= 5:
            return False
        
        # Simple market hours check (9:00 AM - 5:00 PM ET)
        current_hour = now.hour
        return 9 <= current_hour < 17