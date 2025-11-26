"""Yield curve analysis for bond trading."""

from typing import Dict, List, Tuple
import numpy as np


class YieldCurveAnalyzer:
    """Analyzes yield curve shape and trading implications."""
    
    def __init__(self):
        """Initialize the yield curve analyzer."""
        self.recession_threshold = -0.1  # 2s10s inversion threshold
        
    def analyze_curve_shape(self, yields: Dict[str, float]) -> Dict[str, any]:
        """Analyze yield curve shape and implications.
        
        Args:
            yields: Dictionary of maturity to yield
            
        Returns:
            Analysis results including shape, spreads, and trading bias
        """
        # Calculate key spreads
        spreads = self._calculate_spreads(yields)
        
        # Determine curve shape
        curve_type = self._determine_curve_type(spreads)
        
        # Calculate recession probability
        recession_prob = self._calculate_recession_probability(spreads)
        
        # Determine trading bias
        trading_bias = self._get_trading_bias(curve_type, spreads)
        
        return {
            "type": curve_type,
            "2s10s_spread": spreads.get("2s10s", 0),
            "2s30s_spread": spreads.get("2s30s", 0),
            "5s30s_spread": spreads.get("5s30s", 0),
            "recession_probability": recession_prob,
            "trading_bias": trading_bias,
            "steepness": self._calculate_steepness(yields),
        }
    
    def _calculate_spreads(self, yields: Dict[str, float]) -> Dict[str, float]:
        """Calculate yield spreads.
        
        Args:
            yields: Yield data
            
        Returns:
            Dictionary of spreads
        """
        spreads = {}
        
        # 2s10s spread (most watched)
        if "2Y" in yields and "10Y" in yields:
            spreads["2s10s"] = round(yields["10Y"] - yields["2Y"], 2)
        
        # 2s30s spread
        if "2Y" in yields and "30Y" in yields:
            spreads["2s30s"] = round(yields["30Y"] - yields["2Y"], 2)
        
        # 5s30s spread
        if "5Y" in yields and "30Y" in yields:
            spreads["5s30s"] = round(yields["30Y"] - yields["5Y"], 2)
        
        # 3m10y spread
        if "3M" in yields and "10Y" in yields:
            spreads["3m10y"] = round(yields["10Y"] - yields["3M"], 2)
        
        return spreads
    
    def _determine_curve_type(self, spreads: Dict[str, float]) -> str:
        """Determine the type of yield curve.
        
        Args:
            spreads: Yield spreads
            
        Returns:
            Curve type (normal, flat, inverted)
        """
        two_ten = spreads.get("2s10s", 0)
        
        if two_ten < self.recession_threshold:
            return "inverted"
        elif two_ten < 0.3:
            return "flat"
        else:
            return "normal"
    
    def _calculate_recession_probability(self, spreads: Dict[str, float]) -> float:
        """Calculate recession probability based on curve shape.
        
        Args:
            spreads: Yield spreads
            
        Returns:
            Recession probability (0-1)
        """
        # Based on historical data, 2s10s inversion has preceded recessions
        two_ten = spreads.get("2s10s", 0)
        
        if two_ten < -0.5:
            return 0.9  # Deep inversion
        elif two_ten < -0.2:
            return 0.7  # Moderate inversion
        elif two_ten < 0:
            return 0.5  # Slight inversion
        elif two_ten < 0.5:
            return 0.3  # Flat curve
        else:
            return 0.1  # Normal curve
    
    def _get_trading_bias(self, curve_type: str, spreads: Dict[str, float]) -> str:
        """Determine trading bias based on curve shape.
        
        Args:
            curve_type: Type of curve
            spreads: Yield spreads
            
        Returns:
            Trading bias recommendation
        """
        if curve_type == "inverted":
            # Inverted curve suggests rate cuts ahead
            return "long_duration"
        elif curve_type == "normal" and spreads.get("2s10s", 0) >= 1.0:
            # Steep curve suggests rising rates
            return "short_duration"
        else:
            return "neutral"
    
    def _calculate_steepness(self, yields: Dict[str, float]) -> float:
        """Calculate overall curve steepness.
        
        Args:
            yields: Yield data
            
        Returns:
            Steepness score
        """
        if "2Y" not in yields or "30Y" not in yields:
            return 0
        
        # Simple steepness: 30Y - 2Y spread
        return yields["30Y"] - yields["2Y"]
    
    def identify_curve_trades(self, current_shape: Dict, historical_avg: Dict) -> List[Dict]:
        """Identify curve trading opportunities.
        
        Args:
            current_shape: Current curve analysis
            historical_avg: Historical average spreads
            
        Returns:
            List of potential trades
        """
        trades = []
        
        current_2s10s = current_shape.get("2s10s_spread", 0)
        avg_2s10s = historical_avg.get("2s10s", 0.5)
        
        # Curve steepener trade
        if current_2s10s < avg_2s10s - 0.3:
            trades.append({
                "type": "curve_steepener",
                "description": "Long 10Y, Short 2Y",
                "rationale": "Curve below historical average",
                "confidence": 0.7,
            })
        
        # Curve flattener trade
        elif current_2s10s > avg_2s10s + 0.5:
            trades.append({
                "type": "curve_flattener",
                "description": "Short 10Y, Long 2Y",
                "rationale": "Curve above historical average",
                "confidence": 0.7,
            })
        
        # Recession hedge
        if current_shape["recession_probability"] > 0.6:
            trades.append({
                "type": "recession_hedge",
                "description": "Long TLT (20+ Year Treasuries)",
                "rationale": "High recession probability",
                "confidence": 0.8,
            })
        
        return trades
    
    def calculate_duration_risk(
        self,
        maturity: str,
        yield_change_bps: float = 100
    ) -> float:
        """Calculate duration risk for a maturity.
        
        Args:
            maturity: Bond maturity
            yield_change_bps: Yield change in basis points
            
        Returns:
            Estimated price change percentage
        """
        # Simplified duration estimates
        duration_map = {
            "2Y": 1.9,
            "5Y": 4.5,
            "10Y": 8.5,
            "30Y": 20.0,
        }
        
        duration = duration_map.get(maturity, 5)
        
        # Price change = -Duration * Yield Change
        price_change = -duration * (yield_change_bps / 100)
        
        return price_change