"""Treasury yield and bond market data collection"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class TreasuryYieldCollector:
    """Collects treasury yield and bond market data from free sources"""
    
    def __init__(self):
        """Initialize the yield collector"""
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)  # Longer cache for yields
        
        # ETF mappings for yield approximation
        self.treasury_etf_mapping = {
            "SHY": {"maturity": "1-3Y", "avg_maturity": 2},
            "IEI": {"maturity": "3-7Y", "avg_maturity": 5},
            "IEF": {"maturity": "7-10Y", "avg_maturity": 8.5},
            "TLH": {"maturity": "10-20Y", "avg_maturity": 15},
            "TLT": {"maturity": "20Y+", "avg_maturity": 25}
        }
        
        # Bond ETFs for analysis
        self.bond_etfs = {
            "treasuries": ["SHY", "IEI", "IEF", "TLH", "TLT", "GOVT", "BIL"],
            "corporates": ["LQD", "VCIT", "VCSH", "IGIB"],
            "high_yield": ["HYG", "JNK", "SJNK", "HYLD"],
            "tips": ["TIP", "STIP", "LTPZ"],
            "munis": ["MUB", "SUB", "HYD"],
            "international": ["BNDX", "IAGG", "BWX"]
        }
        
    def get_current_yields(self) -> Dict[str, float]:
        """
        Get current treasury yields across the curve
        
        Returns:
            Dictionary with maturity and yield values
        """
        cache_key = "current_yields"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_data
                
        yields = {}
        
        try:
            # Use treasury ETFs and market data to approximate yields
            # These symbols represent treasury yield indices
            yield_symbols = {
                "^IRX": "3M",   # 13 Week Treasury Bill
                "^FVX": "5Y",   # 5 Year Treasury
                "^TNX": "10Y",  # 10 Year Treasury  
                "^TYX": "30Y"   # 30 Year Treasury
            }
            
            for symbol, maturity in yield_symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.history(period="1d")
                    if not info.empty:
                        # These indices show yield in percentage
                        yields[maturity] = round(info['Close'].iloc[-1], 2)
                except Exception as e:
                    logger.warning(f"Failed to get {maturity} yield: {str(e)}")
                    
            # Fill in missing yields using interpolation and ETF data
            yields.update(self._estimate_missing_yields(yields))
            
            # Cache the results
            self.cache[cache_key] = (yields, datetime.now())
            
        except Exception as e:
            logger.error(f"Error fetching yields: {str(e)}")
            # Return approximate yields as fallback
            yields = self._get_fallback_yields()
            
        return yields
    
    def get_bond_etf_data(self, etfs: List[str]) -> Dict[str, Dict]:
        """
        Get data for bond ETFs including yield and duration
        
        Args:
            etfs: List of ETF symbols
            
        Returns:
            Dictionary with ETF data
        """
        etf_data = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(self._get_single_etf_data, etf): etf 
                      for etf in etfs}
            
            for future in as_completed(futures):
                etf = futures[future]
                try:
                    data = future.result()
                    if data:
                        etf_data[etf] = data
                except Exception as e:
                    logger.error(f"Error fetching {etf} data: {str(e)}")
                    
        return etf_data
    
    def get_yield_curve_history(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical yield curve data
        
        Args:
            days: Number of days of history
            
        Returns:
            DataFrame with yield history
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data for key maturities
        symbols = {
            "^FVX": "5Y",
            "^TNX": "10Y",
            "^TYX": "30Y"
        }
        
        history_data = {}
        
        for symbol, maturity in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date)
                if not hist.empty:
                    history_data[maturity] = hist['Close']
            except Exception as e:
                logger.error(f"Error fetching history for {symbol}: {str(e)}")
                
        # Create DataFrame
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Estimate 2Y yield based on 5Y/10Y
            if '5Y' in df.columns and '10Y' in df.columns:
                # Simple interpolation
                df['2Y'] = df['5Y'] - (df['10Y'] - df['5Y']) * 0.3
                
            # Calculate spreads
            if '2Y' in df.columns and '10Y' in df.columns:
                df['2s10s_spread'] = df['10Y'] - df['2Y']
                
            return df
        else:
            return pd.DataFrame()
    
    def get_real_yields(self) -> Dict[str, float]:
        """
        Get TIPS (inflation-protected) yields
        
        Returns:
            Dictionary with real yield values
        """
        real_yields = {}
        
        # Use TIPS ETFs to estimate real yields
        tips_etfs = {
            "STIP": "5Y_TIPS",   # 0-5 Year TIPS
            "TIP": "10Y_TIPS",   # Broad TIPS
            "LTPZ": "30Y_TIPS"   # Long-term TIPS
        }
        
        for etf, label in tips_etfs.items():
            try:
                data = self._get_single_etf_data(etf)
                if data and 'yield' in data:
                    # Approximate real yield from ETF yield
                    real_yields[label] = data['yield'] - 2.5  # Rough adjustment
            except Exception as e:
                logger.error(f"Error getting real yield for {etf}: {str(e)}")
                
        # Ensure reasonable values
        for key in real_yields:
            real_yields[key] = max(-3, min(3, real_yields[key]))
            
        return real_yields
    
    def get_credit_spreads(self) -> Dict[str, float]:
        """
        Get corporate bond spreads
        
        Returns:
            Dictionary with credit spread values in basis points
        """
        spreads = {}
        
        try:
            # Get treasury yield for comparison
            treasury_10y = self.get_current_yields().get("10Y", 4.5)
            
            # Investment Grade - use historical average spreads with current adjustments
            lqd_data = self._get_single_etf_data("LQD")
            if lqd_data and 'yield' in lqd_data and lqd_data['yield'] > treasury_10y:
                calculated_spread = (lqd_data['yield'] - treasury_10y) * 100
                spreads["IG_spread"] = max(50, min(500, calculated_spread))  # Ensure reasonable range
            else:
                spreads["IG_spread"] = 120  # Default spread
                
            # High Yield
            hyg_data = self._get_single_etf_data("HYG")
            if hyg_data and 'yield' in hyg_data and hyg_data['yield'] > treasury_10y:
                calculated_spread = (hyg_data['yield'] - treasury_10y) * 100
                spreads["HY_spread"] = max(200, min(1000, calculated_spread))  # Ensure reasonable range
            else:
                spreads["HY_spread"] = 400  # Default spread
                
            # Ensure HY spread is higher than IG spread
            if spreads["HY_spread"] <= spreads["IG_spread"]:
                spreads["HY_spread"] = spreads["IG_spread"] + 200
                
            # Approximate rating-based spreads
            spreads["AAA_spread"] = max(25, spreads["IG_spread"] * 0.5)
            spreads["BBB_spread"] = spreads["IG_spread"] * 1.5
            
            # Ensure all spreads are positive and reasonable
            for key in spreads:
                spreads[key] = max(25, spreads[key])  # Minimum 25bp spread
            
        except Exception as e:
            logger.error(f"Error calculating credit spreads: {str(e)}")
            # Return default spreads
            spreads = {
                "IG_spread": 120,
                "HY_spread": 400,
                "AAA_spread": 60,
                "BBB_spread": 180
            }
            
        return spreads
    
    def get_international_yields(self) -> Dict[str, Dict[str, float]]:
        """
        Get international government bond yields
        
        Returns:
            Dictionary with country yields
        """
        intl_yields = {}
        
        # Use international bond ETFs as proxies
        country_etfs = {
            "GSGB": "Germany",  # Actually doesn't exist, using approximation
            "JGBL": "Japan",
            "GUKG": "UK",
            "CAN": "Canada"
        }
        
        # For this implementation, use approximate values
        # In production, would scrape from central bank websites
        intl_yields = {
            "Germany": {"10Y": 2.5, "2Y": 3.0},
            "Japan": {"10Y": 0.8, "2Y": -0.1},
            "UK": {"10Y": 4.3, "2Y": 4.5},
            "Canada": {"10Y": 3.8, "2Y": 4.0}
        }
        
        return intl_yields
    
    def get_fed_data(self) -> Dict[str, any]:
        """
        Get Federal Reserve data and rate expectations
        
        Returns:
            Dictionary with Fed-related data
        """
        fed_data = {}
        
        try:
            # Current Fed Funds Rate (approximate from short-term yields)
            yields = self.get_current_yields()
            if "3M" in yields:
                # Fed funds typically close to 3M treasury
                fed_data["fed_funds_rate"] = yields["3M"]
            else:
                fed_data["fed_funds_rate"] = 5.25  # Default
                
            # Target range (typical 25bp range)
            fed_data["fed_funds_target_upper"] = fed_data["fed_funds_rate"] + 0.125
            fed_data["fed_funds_target_lower"] = fed_data["fed_funds_rate"] - 0.125
            
            # Next meeting (simplified - usually every 6 weeks)
            today = datetime.now()
            # Find next Wednesday (Fed usually meets on Wednesday)
            days_ahead = 2 - today.weekday()  # Wednesday is 2
            if days_ahead <= 0:
                days_ahead += 7
            next_meeting = today + timedelta(days=days_ahead + 35)  # ~6 weeks
            fed_data["next_meeting_date"] = next_meeting.strftime("%Y-%m-%d")
            
            # Rate probabilities (simplified)
            fed_data["rate_probabilities"] = {
                "hike_25bp": 0.2,
                "unchanged": 0.6,
                "cut_25bp": 0.2
            }
            
        except Exception as e:
            logger.error(f"Error getting Fed data: {str(e)}")
            fed_data = {
                "fed_funds_rate": 5.25,
                "fed_funds_target_upper": 5.375,
                "fed_funds_target_lower": 5.125,
                "next_meeting_date": "2024-03-20",
                "rate_probabilities": {
                    "hike_25bp": 0.2,
                    "unchanged": 0.6,
                    "cut_25bp": 0.2
                }
            }
            
        return fed_data
    
    def _get_single_etf_data(self, etf: str) -> Optional[Dict]:
        """Get data for a single ETF"""
        try:
            ticker = yf.Ticker(etf)
            info = ticker.info
            hist = ticker.history(period="1mo")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            
            # Calculate metrics
            data = {
                "price": round(current_price, 2),
                "yield": info.get("yield", self._estimate_etf_yield(etf, current_price)),
                "duration": self._estimate_duration(etf),
                "ytd_return": self._calculate_ytd_return(ticker),
                "expense_ratio": info.get("expenseRatio", 0.001),
                "volume": info.get("averageVolume", 0)
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {etf} data: {str(e)}")
            return None
    
    def _estimate_etf_yield(self, etf: str, price: float) -> float:
        """Estimate ETF yield based on type and current rates"""
        # Use fallback yields to avoid circular dependency
        fallback_yields = self._get_fallback_yields()
        
        # Map ETFs to approximate yields
        if etf == "SHY":
            return fallback_yields.get("2Y", 4.5)
        elif etf == "IEF":
            return fallback_yields.get("10Y", 4.6)
        elif etf == "TLT":
            return fallback_yields.get("30Y", 4.8)
        elif etf in ["HYG", "JNK"]:
            return fallback_yields.get("10Y", 4.6) + 4.0  # Add credit spread
        elif etf in ["LQD", "VCIT"]:
            return fallback_yields.get("10Y", 4.6) + 1.2  # Add IG spread
        else:
            return 4.5  # Default
    
    def _estimate_duration(self, etf: str) -> float:
        """Estimate effective duration for ETF"""
        duration_map = {
            "SHY": 1.9,
            "IEI": 4.5,
            "IEF": 7.5,
            "TLH": 12.0,
            "TLT": 17.0,
            "AGG": 6.5,
            "BND": 6.5,
            "LQD": 8.5,
            "HYG": 4.0,
            "VCSH": 2.7,
            "VCIT": 5.5
        }
        
        return duration_map.get(etf, 5.0)  # Default 5 years
    
    def _calculate_ytd_return(self, ticker) -> float:
        """Calculate year-to-date return"""
        try:
            start_of_year = datetime(datetime.now().year, 1, 1)
            ytd_data = ticker.history(start=start_of_year)
            
            if not ytd_data.empty:
                start_price = ytd_data['Close'].iloc[0]
                end_price = ytd_data['Close'].iloc[-1]
                return round((end_price / start_price - 1) * 100, 2)
        except:
            pass
            
        return 0.0
    
    def _estimate_missing_yields(self, known_yields: Dict[str, float]) -> Dict[str, float]:
        """Estimate missing yields using interpolation"""
        estimated = {}
        
        # Common maturity points in years
        maturity_map = {
            "3M": 0.25,
            "6M": 0.5,
            "1Y": 1.0,
            "2Y": 2.0,
            "5Y": 5.0,
            "10Y": 10.0,
            "30Y": 30.0
        }
        
        # Convert known yields to maturity/yield pairs
        known_points = [(maturity_map[m], y) for m, y in known_yields.items() 
                       if m in maturity_map]
        
        if len(known_points) >= 2:
            known_points.sort()
            
            # Interpolate missing points
            for maturity_label, years in maturity_map.items():
                if maturity_label not in known_yields:
                    # Find surrounding points
                    lower = None
                    upper = None
                    
                    for mat, yld in known_points:
                        if mat <= years:
                            lower = (mat, yld)
                        elif mat > years and upper is None:
                            upper = (mat, yld)
                            
                    if lower and upper:
                        # Linear interpolation
                        slope = (upper[1] - lower[1]) / (upper[0] - lower[0])
                        estimated[maturity_label] = round(
                            lower[1] + slope * (years - lower[0]), 2
                        )
                    elif lower:
                        # Extrapolate from lower
                        estimated[maturity_label] = round(lower[1] + 0.1 * (years - lower[0]), 2)
                    elif upper:
                        # Extrapolate from upper
                        estimated[maturity_label] = round(upper[1] - 0.1 * (upper[0] - years), 2)
                        
        # Add reasonable defaults for any still missing
        defaults = {
            "3M": 5.25,
            "6M": 5.20,
            "1Y": 5.00,
            "2Y": 4.85
        }
        
        for maturity, default in defaults.items():
            if maturity not in known_yields and maturity not in estimated:
                estimated[maturity] = default
                
        return estimated
    
    def _get_fallback_yields(self) -> Dict[str, float]:
        """Return reasonable fallback yields"""
        return {
            "3M": 5.25,
            "6M": 5.20,
            "1Y": 5.00,
            "2Y": 4.85,
            "5Y": 4.55,
            "10Y": 4.65,
            "30Y": 4.85
        }
    
    def clear_cache(self):
        """Clear the data cache"""
        self.cache.clear()
        logger.info("Yield data cache cleared")