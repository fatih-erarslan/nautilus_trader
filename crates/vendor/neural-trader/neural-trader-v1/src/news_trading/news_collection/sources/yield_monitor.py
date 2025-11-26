"""Yield curve monitoring for bond trading signals - GREEN phase"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class YieldCurveMonitor:
    """Monitor yield curve changes for trading signals"""
    
    def __init__(self):
        self.maturities = ["3M", "2Y", "5Y", "10Y", "30Y"]
        
    def analyze_yield_changes(self, current_yields: Dict[str, float], 
                            previous_yields: Dict[str, float]) -> Dict[str, Any]:
        """Analyze yield curve changes for trading signals"""
        
        # Calculate spreads
        current_2_10_spread = current_yields["10Y"] - current_yields["2Y"]
        previous_2_10_spread = previous_yields["10Y"] - previous_yields["2Y"]
        
        current_3m_10y_spread = current_yields["10Y"] - current_yields["3M"]
        
        # Determine curve shape
        if current_3m_10y_spread < 0:
            curve_shape = "inverted"
        elif current_2_10_spread < 0.5:
            curve_shape = "flat"
        else:
            curve_shape = "normal"
        
        # Check if steepening or flattening
        spread_change = current_2_10_spread - previous_2_10_spread
        steepening = spread_change > 0
        
        # Identify significant moves (10bp = 0.10%)
        significant_moves = {}
        for maturity in self.maturities:
            if maturity in current_yields and maturity in previous_yields:
                change = current_yields[maturity] - previous_yields[maturity]
                if abs(change) >= 0.09999:  # 10 basis points (with floating point tolerance)
                    significant_moves[maturity] = change
        
        # Determine trading signal
        if steepening and curve_shape != "inverted":
            trading_signal = "steepener"
        elif not steepening and abs(spread_change) > 0.05:
            trading_signal = "flattener"
        elif curve_shape == "inverted":
            trading_signal = "recession_hedge"
        else:
            trading_signal = "neutral"
        
        return {
            "curve_shape": curve_shape,
            "steepening": steepening,
            "2_10_spread": current_2_10_spread,
            "3m_10y_spread": current_3m_10y_spread,
            "spread_change": spread_change,
            "significant_moves": significant_moves,
            "trading_signal": trading_signal
        }