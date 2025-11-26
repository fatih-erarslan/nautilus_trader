"""Duration-based bond trading strategies."""

from typing import Dict, List, Tuple


class DurationTrader:
    """Trades bonds based on duration and interest rate views."""
    
    def __init__(self):
        """Initialize the duration trader."""
        self.fed_stance_impact = {
            "hawkish": -0.5,          # Negative for bonds
            "neutral": 0.0,
            "dovish": 0.5,            # Positive for bonds
            "pivot_to_dovish": 1.0,   # Very positive
            "pivot_to_hawkish": -1.0, # Very negative
        }
        
    def analyze_duration_trade(self, market_data: Dict) -> Dict:
        """Analyze market conditions for duration trades.
        
        Args:
            market_data: Market indicators
            
        Returns:
            Trading signal
        """
        fed_stance = market_data.get("fed_stance", "neutral")
        ten_year_yield = market_data.get("10y_yield", 4.5)
        ten_year_sma = market_data.get("10y_sma_50", 4.5)
        inflation_trend = market_data.get("inflation_trend", "stable")
        tlt_price = market_data.get("tlt_price", 100)
        tlt_sma = market_data.get("tlt_sma_20", 100)
        
        # Calculate duration score
        score = 0
        
        # Fed policy impact
        score += self.fed_stance_impact.get(fed_stance, 0) * 2
        
        # Yield trend
        if ten_year_yield > ten_year_sma:
            score -= 0.5  # Rising yields bad for bonds
        else:
            score += 0.5  # Falling yields good for bonds
        
        # Inflation trend
        if inflation_trend == "declining":
            score += 0.5
        elif inflation_trend == "rising":
            score -= 0.5
        
        # Technical setup
        if tlt_price < tlt_sma * 0.98:  # Oversold
            score += 0.5
        elif tlt_price > tlt_sma * 1.02:  # Overbought
            score -= 0.5
        
        # Generate signal
        if score >= 1.5:
            return self._create_long_duration_signal(market_data)
        elif score <= -1.5:
            return self._create_short_duration_signal(market_data)
        else:
            return {
                "position": "neutral",
                "rationale": "No clear duration trade",
                "score": score,
            }
    
    def _create_long_duration_signal(self, market_data: Dict) -> Dict:
        """Create long duration trading signal.
        
        Args:
            market_data: Market data
            
        Returns:
            Long duration signal
        """
        ten_year_yield = market_data.get("10y_yield", 4.5)
        fed_stance = market_data.get("fed_stance", "neutral")
        
        # Stop loss based on yield levels
        stop_yield = ten_year_yield + 0.15  # 15 bps higher
        
        # Position sizing based on conviction
        if fed_stance == "pivot_to_dovish":
            position_size = "large"
            confidence = 0.8
        else:
            position_size = "medium"
            confidence = 0.6
        
        return {
            "position": "long_tlt",
            "rationale": f"Fed pivot supports duration",
            "instruments": ["TLT", "IEF"],  # Long-term and intermediate
            "stop_yield": round(stop_yield, 2),
            "target_yield": round(ten_year_yield - 0.30, 2),  # 30 bps lower
            "position_size": position_size,
            "confidence": confidence,
        }
    
    def _create_short_duration_signal(self, market_data: Dict) -> Dict:
        """Create short duration trading signal.
        
        Args:
            market_data: Market data
            
        Returns:
            Short duration signal
        """
        ten_year_yield = market_data.get("10y_yield", 4.5)
        
        return {
            "position": "short_tlt",
            "rationale": "Rising rate environment",
            "instruments": ["TLT"],  # Short long-term bonds
            "alternatives": ["SHY", "SGOV"],  # Move to short duration
            "stop_yield": round(ten_year_yield - 0.15, 2),  # 15 bps lower
            "target_yield": round(ten_year_yield + 0.30, 2),  # 30 bps higher
            "position_size": "medium",
            "confidence": 0.6,
        }
    
    def analyze_flight_to_quality(self, market_data: Dict) -> Dict:
        """Analyze flight-to-quality scenarios.
        
        Args:
            market_data: Market indicators
            
        Returns:
            Flight-to-quality signal
        """
        vix = market_data.get("vix", 20)
        spy_change = market_data.get("spy_change", 0)
        credit_spreads_widening = market_data.get("credit_spreads_widening", False)
        ten_year_yield = market_data.get("10y_yield", 4.5)
        yield_1d_ago = market_data.get("10y_yield_1d_ago", 4.5)
        dollar_index = market_data.get("dollar_index", 100)
        
        # Risk-off indicators
        risk_off_score = 0
        
        if vix > 30:
            risk_off_score += 2
        elif vix > 25:
            risk_off_score += 1
        
        if spy_change < -0.02:  # 2% equity decline
            risk_off_score += 2
        elif spy_change < -0.01:
            risk_off_score += 1
        
        if credit_spreads_widening:
            risk_off_score += 1
        
        if ten_year_yield < yield_1d_ago - 0.05:  # Yields falling
            risk_off_score += 1
        
        if dollar_index > 103:  # Strong dollar
            risk_off_score += 1
        
        # Generate signal
        if risk_off_score >= 4:
            return {
                "position": "long_treasuries",
                "instruments": ["TLT", "IEF"],
                "rationale": "Flight to quality - risk-off environment",
                "confidence": min(0.9, 0.5 + risk_off_score * 0.1),
                "urgency": "high" if risk_off_score >= 5 else "medium",
                "size": "large" if vix > 35 else "medium",
            }
        else:
            return {
                "position": "monitor",
                "rationale": "No clear flight-to-quality signal",
                "risk_off_score": risk_off_score,
            }
    
    def calculate_duration_hedge_ratio(
        self,
        portfolio_duration: float,
        hedge_instrument_duration: float,
        portfolio_value: float
    ) -> float:
        """Calculate hedge ratio for duration hedging.
        
        Args:
            portfolio_duration: Duration of portfolio to hedge
            hedge_instrument_duration: Duration of hedging instrument
            portfolio_value: Value of portfolio
            
        Returns:
            Hedge ratio (notional amount to hedge)
        """
        if hedge_instrument_duration == 0:
            return 0
        
        # Duration matching
        hedge_ratio = portfolio_duration / hedge_instrument_duration
        
        # Notional amount
        hedge_notional = portfolio_value * hedge_ratio
        
        return round(hedge_notional, 2)
    
    def get_curve_positioning(self, yield_curve_shape: str) -> Dict[str, str]:
        """Get recommended curve positioning.
        
        Args:
            yield_curve_shape: Current curve shape
            
        Returns:
            Positioning recommendations
        """
        if yield_curve_shape == "inverted":
            return {
                "2Y": "underweight",
                "5Y": "neutral",
                "10Y": "overweight",
                "30Y": "overweight",
                "rationale": "Position for curve normalization",
            }
        elif yield_curve_shape == "steep":
            return {
                "2Y": "overweight",
                "5Y": "overweight",
                "10Y": "neutral",
                "30Y": "underweight",
                "rationale": "Position for curve flattening",
            }
        else:
            return {
                "2Y": "neutral",
                "5Y": "neutral",
                "10Y": "neutral",
                "30Y": "neutral",
                "rationale": "Neutral curve positioning",
            }