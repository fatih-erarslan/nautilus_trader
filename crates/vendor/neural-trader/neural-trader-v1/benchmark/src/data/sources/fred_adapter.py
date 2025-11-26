"""FRED (Federal Reserve Economic Data) adapter for economic indicators.

Provides access to economic data series from the Federal Reserve Bank of St. Louis.
Useful for macro-economic indicators like GDP, inflation, interest rates, etc.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import aiohttp

from ..realtime_feed import DataUpdate, DataSource

logger = logging.getLogger(__name__)


@dataclass
class FREDConfig:
    """Configuration for FRED adapter."""
    api_key: str
    series_ids: List[str]  # List of FRED series IDs to track
    update_interval: float = 3600.0  # 1 hour (economic data updates less frequently)
    realtime_start: Optional[str] = None  # YYYY-MM-DD format
    realtime_end: Optional[str] = None
    observation_start: Optional[str] = None
    observation_end: Optional[str] = None
    units: str = "lin"  # lin, chg, ch1, pch, pc1, pca, cch, cca, log
    frequency: str = "d"  # d, w, bw, m, q, sa, a, wef, weth, wew, wetu, wem, wesu, wesa, bwew, bwem


class FREDSeries:
    """Common FRED series IDs for economic indicators."""
    # Interest Rates
    DFF = "DFF"  # Federal Funds Rate
    DGS10 = "DGS10"  # 10-Year Treasury Rate
    DGS2 = "DGS2"  # 2-Year Treasury Rate
    DGS30 = "DGS30"  # 30-Year Treasury Rate
    SOFR = "SOFR"  # Secured Overnight Financing Rate
    
    # Inflation
    CPIAUCSL = "CPIAUCSL"  # Consumer Price Index
    CPILFESL = "CPILFESL"  # Core CPI
    DFEDTARU = "DFEDTARU"  # Fed Target Rate Upper
    DFEDTARL = "DFEDTARL"  # Fed Target Rate Lower
    
    # Economic Indicators
    GDP = "GDP"  # Gross Domestic Product
    GDPC1 = "GDPC1"  # Real GDP
    UNRATE = "UNRATE"  # Unemployment Rate
    PAYEMS = "PAYEMS"  # Nonfarm Payrolls
    
    # Market Indicators
    DEXUSEU = "DEXUSEU"  # USD/EUR Exchange Rate
    DEXJPUS = "DEXJPUS"  # USD/JPY Exchange Rate
    VIXCLS = "VIXCLS"  # VIX Volatility Index
    
    # Money Supply
    M1SL = "M1SL"  # M1 Money Supply
    M2SL = "M2SL"  # M2 Money Supply
    
    @classmethod
    def get_all_series(cls) -> List[str]:
        """Get all available series IDs."""
        return [
            value for name, value in vars(cls).items() 
            if not name.startswith('_') and isinstance(value, str)
        ]


class FREDAdapter:
    """FRED data adapter for economic indicators."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, config: FREDConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._update_task = None
        self._callbacks: List[Callable] = []
        
        # Cache for series metadata
        self._series_info: Dict[str, Dict] = {}
        self._last_values: Dict[str, float] = {}  # Track last values for change detection
        
        # Metrics
        self._total_requests = 0
        self._failed_requests = 0
        self._last_update_time = 0
    
    async def start(self):
        """Start the adapter."""
        if self._running:
            return
        
        self._session = aiohttp.ClientSession()
        self._running = True
        
        # Fetch series info on start
        await self._fetch_series_info()
        
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info(f"FRED adapter started for {len(self.config.series_ids)} series")
    
    async def stop(self):
        """Stop the adapter."""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
        
        logger.info("FRED adapter stopped")
    
    def add_callback(self, callback: Callable[[DataUpdate], None]):
        """Add callback for data updates."""
        self._callbacks.append(callback)
    
    async def _fetch_series_info(self):
        """Fetch metadata for all tracked series."""
        for series_id in self.config.series_ids:
            try:
                info = await self._get_series_metadata(series_id)
                if info:
                    self._series_info[series_id] = info
            except Exception as e:
                logger.error(f"Error fetching info for {series_id}: {e}")
    
    async def _get_series_metadata(self, series_id: str) -> Optional[Dict]:
        """Get metadata for a series."""
        params = {
            "series_id": series_id,
            "api_key": self.config.api_key,
            "file_type": "json"
        }
        
        url = f"{self.BASE_URL}/series"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if "seriess" in data and data["seriess"]:
                        return data["seriess"][0]
        except Exception as e:
            logger.error(f"Error getting metadata for {series_id}: {e}")
        
        return None
    
    async def _update_loop(self):
        """Main update loop for economic data."""
        while self._running:
            try:
                # Fetch updates for all series
                for series_id in self.config.series_ids:
                    await self._fetch_series_observations(series_id)
                
                self._last_update_time = time.time()
                
                # Sleep until next update
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def _fetch_series_observations(self, series_id: str):
        """Fetch latest observations for a series."""
        params = {
            "series_id": series_id,
            "api_key": self.config.api_key,
            "file_type": "json",
            "limit": 10,  # Get last 10 observations
            "sort_order": "desc",  # Most recent first
            "units": self.config.units,
            "frequency": self.config.frequency,
        }
        
        # Add date filters if specified
        if self.config.realtime_start:
            params["realtime_start"] = self.config.realtime_start
        if self.config.realtime_end:
            params["realtime_end"] = self.config.realtime_end
        
        url = f"{self.BASE_URL}/series/observations"
        
        try:
            self._total_requests += 1
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "observations" in data and data["observations"]:
                        # Process latest observation
                        latest = data["observations"][0]
                        await self._process_observation(series_id, latest)
                        
                        # Also check for any newer updates we might have missed
                        for obs in data["observations"][1:]:
                            if self._is_new_observation(series_id, obs):
                                await self._process_observation(series_id, obs)
                else:
                    logger.error(f"HTTP error {response.status} for {series_id}")
                    self._failed_requests += 1
                    
        except Exception as e:
            logger.error(f"Error fetching observations for {series_id}: {e}")
            self._failed_requests += 1
    
    def _is_new_observation(self, series_id: str, observation: Dict) -> bool:
        """Check if observation is new based on date."""
        # This is a simplified check - in production you'd track last observation date
        return False
    
    async def _process_observation(self, series_id: str, observation: Dict):
        """Process a FRED observation into a data update."""
        try:
            value_str = observation.get("value", "")
            
            # Skip missing values (marked as ".")
            if value_str == "." or not value_str:
                return
            
            value = float(value_str)
            date_str = observation.get("date", "")
            
            # Parse date
            obs_date = datetime.strptime(date_str, "%Y-%m-%d")
            timestamp = obs_date.timestamp()
            
            # Get series info for metadata
            series_info = self._series_info.get(series_id, {})
            
            # Calculate change from last value
            change = None
            change_percent = None
            if series_id in self._last_values:
                last_value = self._last_values[series_id]
                change = value - last_value
                if last_value != 0:
                    change_percent = (change / last_value) * 100
            
            self._last_values[series_id] = value
            
            # Create metadata
            metadata = {
                "source": "fred",
                "series_id": series_id,
                "title": series_info.get("title", series_id),
                "units": series_info.get("units", self.config.units),
                "frequency": series_info.get("frequency", self.config.frequency),
                "seasonal_adjustment": series_info.get("seasonal_adjustment", ""),
                "observation_date": date_str,
                "change": change,
                "change_percent": change_percent,
            }
            
            # Create data update
            update = DataUpdate(
                symbol=f"FRED:{series_id}",  # Prefix with FRED to distinguish
                price=value,  # Using price field for the economic value
                timestamp=timestamp,
                source=DataSource.REST,
                metadata=metadata
            )
            
            # Notify callbacks
            for callback in self._callbacks:
                try:
                    await callback(update)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing observation for {series_id}: {e}")
    
    async def get_series_data(self, series_id: str, 
                            observation_start: Optional[str] = None,
                            observation_end: Optional[str] = None) -> Optional[List[Dict]]:
        """Get historical data for a series."""
        params = {
            "series_id": series_id,
            "api_key": self.config.api_key,
            "file_type": "json",
            "sort_order": "asc",
            "units": self.config.units,
        }
        
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        
        url = f"{self.BASE_URL}/series/observations"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("observations", [])
                    
        except Exception as e:
            logger.error(f"Error getting series data for {series_id}: {e}")
        
        return None
    
    async def search_series(self, search_text: str, limit: int = 10) -> Optional[List[Dict]]:
        """Search for FRED series by text."""
        params = {
            "search_text": search_text,
            "api_key": self.config.api_key,
            "file_type": "json",
            "limit": limit,
        }
        
        url = f"{self.BASE_URL}/series/search"
        
        try:
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("seriess", [])
                    
        except Exception as e:
            logger.error(f"Error searching series: {e}")
        
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        success_rate = 1 - (self._failed_requests / self._total_requests) if self._total_requests > 0 else 0
        
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": success_rate,
            "series_tracked": len(self.config.series_ids),
            "series_info_loaded": len(self._series_info),
            "last_update": self._last_update_time,
            "update_interval": self.config.update_interval,
        }
    
    def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """Get upcoming economic data releases."""
        # This would require additional API integration
        # For now, return static schedule for common indicators
        calendar = [
            {
                "series_id": "UNRATE",
                "name": "Unemployment Rate",
                "frequency": "Monthly",
                "release_day": "First Friday of month",
            },
            {
                "series_id": "CPIAUCSL",
                "name": "Consumer Price Index",
                "frequency": "Monthly",
                "release_day": "Around 13th of month",
            },
            {
                "series_id": "GDP",
                "name": "Gross Domestic Product",
                "frequency": "Quarterly",
                "release_day": "End of month after quarter",
            },
            {
                "series_id": "DFF",
                "name": "Federal Funds Rate",
                "frequency": "Daily",
                "release_day": "Daily",
            },
        ]
        
        return calendar