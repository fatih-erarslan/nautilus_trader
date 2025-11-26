"""Treasury yield data collection."""

import requests
import yfinance as yf
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TreasuryYieldCollector:
    """Collects treasury yield data from various sources."""
    
    def __init__(self):
        """Initialize the yield collector."""
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # ETF to yield mapping for approximation
        self.etf_mapping = {
            "SHY": "1-3Y",   # 1-3 year treasuries
            "IEI": "3-7Y",   # 3-7 year treasuries
            "IEF": "7-10Y",  # 7-10 year treasuries
            "TLT": "20Y+",   # 20+ year treasuries
        }
        
        # Mock yields for demonstration (would use real API in production)
        self.mock_yields = {
            "1M": 5.25,
            "3M": 5.45,
            "6M": 5.40,
            "1Y": 5.10,
            "2Y": 4.85,
            "5Y": 4.55,
            "10Y": 4.65,
            "30Y": 4.85,
        }
        
    def get_current_yields(self) -> Dict[str, float]:
        """Get current treasury yields across the curve.
        
        Returns:
            Dictionary of maturity to yield percentage
        """
        # Check cache
        cache_key = "current_yields"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
        
        yields = {}
        
        # Try to get real yields from ETF data
        try:
            # Get ETF yields as proxy
            etf_yields = self._get_etf_yields()
            
            # Combine with mock data for complete curve
            yields.update(self.mock_yields)
            
            # Update with any real ETF data
            if etf_yields:
                yields.update(etf_yields)
                
        except Exception as e:
            logger.error(f"Error fetching yields: {e}")
            # Fall back to mock data
            yields = self.mock_yields.copy()
        
        # Cache the results
        self.cache[cache_key] = (yields, datetime.now())
        
        return yields
    
    def _get_etf_yields(self) -> Dict[str, float]:
        """Get implied yields from bond ETFs.
        
        Returns:
            Dictionary of approximate yields
        """
        yields = {}
        
        for etf, maturity in self.etf_mapping.items():
            try:
                ticker = yf.Ticker(etf)
                info = ticker.info
                
                # Try to get yield from ETF info
                if "yield" in info and info["yield"]:
                    yields[maturity] = info["yield"] * 100
                    
            except Exception as e:
                logger.debug(f"Could not get yield for {etf}: {e}")
                
        return yields
    
    def get_real_yields(self) -> Dict[str, float]:
        """Get real (inflation-adjusted) yields.
        
        Returns:
            Dictionary of real yields
        """
        # Get nominal yields
        nominal_yields = self.get_current_yields()
        
        # For demonstration, subtract estimated inflation
        estimated_inflation = 2.5  # 2.5% inflation assumption
        
        real_yields = {}
        for maturity, nominal_yield in nominal_yields.items():
            if maturity in ["5Y", "10Y", "30Y"]:  # TIPS are available for these
                real_yields[maturity] = nominal_yield - estimated_inflation
                
        return real_yields
    
    def get_yield_curve_data(self) -> Dict[str, any]:
        """Get complete yield curve data with analysis.
        
        Returns:
            Dictionary with yields and curve metrics
        """
        yields = self.get_current_yields()
        
        # Calculate key spreads
        spreads = {}
        if "2Y" in yields and "10Y" in yields:
            spreads["2s10s"] = yields["10Y"] - yields["2Y"]
            
        if "2Y" in yields and "30Y" in yields:
            spreads["2s30s"] = yields["30Y"] - yields["2Y"]
            
        if "5Y" in yields and "30Y" in yields:
            spreads["5s30s"] = yields["30Y"] - yields["5Y"]
        
        # Determine curve shape
        if spreads.get("2s10s", 0) < -0.1:
            shape = "inverted"
        elif spreads.get("2s10s", 0) < 0.3:
            shape = "flat"
        else:
            shape = "normal"
        
        return {
            "yields": yields,
            "spreads": spreads,
            "shape": shape,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_historical_yields(
        self,
        maturity: str,
        days_back: int = 30
    ) -> Dict[str, float]:
        """Get historical yields for a specific maturity.
        
        Args:
            maturity: Treasury maturity (e.g., "10Y")
            days_back: Number of days of history
            
        Returns:
            Dictionary of date to yield
        """
        # This would connect to a real data source
        # For now, return mock data with some variation
        historical = {}
        base_yield = self.mock_yields.get(maturity, 4.5)
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            # Add some random variation
            variation = (i % 7 - 3) * 0.05  # +/- 15 bps variation
            historical[date.strftime("%Y-%m-%d")] = base_yield + variation
            
        return historical