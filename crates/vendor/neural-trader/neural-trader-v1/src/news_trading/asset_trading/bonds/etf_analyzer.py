"""Bond ETF analysis for trading."""

import yfinance as yf
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class BondETFAnalyzer:
    """Analyzes bond ETFs for trading opportunities."""
    
    def __init__(self):
        """Initialize the bond ETF analyzer."""
        self.bond_etfs = {
            "TLT": {
                "name": "20+ Year Treasury",
                "duration_risk": "high",
                "avg_duration": 17.5,
            },
            "IEF": {
                "name": "7-10 Year Treasury",
                "duration_risk": "medium",
                "avg_duration": 7.5,
            },
            "IEI": {
                "name": "3-7 Year Treasury",
                "duration_risk": "medium-low",
                "avg_duration": 4.5,
            },
            "SHY": {
                "name": "1-3 Year Treasury",
                "duration_risk": "low",
                "avg_duration": 1.9,
            },
            "HYG": {
                "name": "High Yield Corporate",
                "duration_risk": "medium",
                "credit_risk": "high",
                "avg_duration": 3.8,
            },
            "LQD": {
                "name": "Investment Grade Corporate",
                "duration_risk": "medium",
                "credit_risk": "medium",
                "avg_duration": 8.4,
            },
            "TIP": {
                "name": "Treasury Inflation Protected",
                "duration_risk": "medium",
                "inflation_hedge": True,
                "avg_duration": 7.5,
            },
        }
        
    def analyze_bond_etfs(self, etf_list: List[str]) -> Dict[str, Dict]:
        """Analyze multiple bond ETFs.
        
        Args:
            etf_list: List of ETF tickers
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        for etf in etf_list:
            if etf in self.bond_etfs:
                analysis[etf] = self._analyze_single_etf(etf)
            else:
                logger.warning(f"Unknown bond ETF: {etf}")
                
        return analysis
    
    def _analyze_single_etf(self, etf: str) -> Dict:
        """Analyze a single bond ETF.
        
        Args:
            etf: ETF ticker
            
        Returns:
            Analysis results
        """
        etf_info = self.bond_etfs[etf].copy()
        
        try:
            # Get market data
            ticker = yf.Ticker(etf)
            data = ticker.history(period="3mo")
            
            if not data.empty:
                # Calculate metrics
                current_price = data['Close'].iloc[-1]
                sma_50 = data['Close'].iloc[-50:].mean() if len(data) >= 50 else data['Close'].mean()
                volatility = data['Close'].pct_change().std() * (252 ** 0.5)  # Annualized
                
                # Relative value assessment
                price_to_sma = current_price / sma_50
                
                if price_to_sma > 1.02:
                    relative_value = "expensive"
                elif price_to_sma < 0.98:
                    relative_value = "cheap"
                else:
                    relative_value = "fair"
                
                etf_info.update({
                    "current_price": round(current_price, 2),
                    "sma_50": round(sma_50, 2),
                    "volatility": round(volatility * 100, 1),
                    "relative_value": relative_value,
                    "price_to_sma": round(price_to_sma, 3),
                })
                
                # Add trading recommendation
                etf_info["recommendation"] = self._get_recommendation(etf, etf_info)
                
        except Exception as e:
            logger.error(f"Error analyzing {etf}: {e}")
            etf_info["error"] = str(e)
            
        return etf_info
    
    def _get_recommendation(self, etf: str, analysis: Dict) -> str:
        """Get trading recommendation for ETF.
        
        Args:
            etf: ETF ticker
            analysis: Analysis results
            
        Returns:
            Trading recommendation
        """
        if "error" in analysis:
            return "no_data"
        
        relative_value = analysis.get("relative_value", "fair")
        duration_risk = analysis.get("duration_risk", "medium")
        
        # Simple recommendation logic
        if relative_value == "cheap" and duration_risk in ["low", "medium-low"]:
            return "buy"
        elif relative_value == "expensive" and duration_risk == "high":
            return "sell"
        elif relative_value == "cheap" and duration_risk == "high":
            return "buy_cautious"  # Cheap but risky
        else:
            return "hold"
    
    def compare_etf_yields(self, etf_list: List[str]) -> Dict[str, float]:
        """Compare yields across bond ETFs.
        
        Args:
            etf_list: List of ETF tickers
            
        Returns:
            Dictionary of ETF to yield
        """
        yields = {}
        
        for etf in etf_list:
            try:
                ticker = yf.Ticker(etf)
                info = ticker.info
                
                # Try to get yield information
                if "yield" in info and info["yield"]:
                    yields[etf] = round(info["yield"] * 100, 2)
                else:
                    # Estimate from dividend yield
                    if "dividendYield" in info and info["dividendYield"]:
                        yields[etf] = round(info["dividendYield"] * 100, 2)
                        
            except Exception as e:
                logger.debug(f"Could not get yield for {etf}: {e}")
                
        return yields
    
    def get_duration_ladder(self) -> List[Dict]:
        """Get bond ETFs organized by duration.
        
        Returns:
            List of ETFs sorted by duration
        """
        ladder = []
        
        for etf, info in self.bond_etfs.items():
            if "avg_duration" in info:
                ladder.append({
                    "etf": etf,
                    "name": info["name"],
                    "duration": info["avg_duration"],
                    "duration_risk": info["duration_risk"],
                })
        
        # Sort by duration
        ladder.sort(key=lambda x: x["duration"])
        
        return ladder
    
    def calculate_etf_correlation(self, etf1: str, etf2: str, period: str = "1y") -> float:
        """Calculate correlation between two bond ETFs.
        
        Args:
            etf1: First ETF ticker
            etf2: Second ETF ticker
            period: Time period for correlation
            
        Returns:
            Correlation coefficient
        """
        try:
            ticker1 = yf.Ticker(etf1)
            ticker2 = yf.Ticker(etf2)
            
            data1 = ticker1.history(period=period)['Close']
            data2 = ticker2.history(period=period)['Close']
            
            # Align data
            common_dates = data1.index.intersection(data2.index)
            
            if len(common_dates) < 20:
                return 0.0
            
            returns1 = data1.loc[common_dates].pct_change().dropna()
            returns2 = data2.loc[common_dates].pct_change().dropna()
            
            correlation = returns1.corr(returns2)
            
            return round(correlation, 3)
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0