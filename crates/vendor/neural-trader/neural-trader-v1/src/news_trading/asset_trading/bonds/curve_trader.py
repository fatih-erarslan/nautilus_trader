"""Yield curve trading strategies."""

from typing import Dict, List


class YieldCurveTrader:
    """Trades yield curve steepeners and flatteners."""
    
    def __init__(self):
        """Initialize the yield curve trader."""
        self.curve_regimes = {
            "steepening": {
                "2Y": "underperform",
                "10Y": "outperform",
                "trade": "steepener",
            },
            "flattening": {
                "2Y": "outperform",
                "10Y": "underperform",
                "trade": "flattener",
            },
            "bull_steepening": {
                "2Y": "strong_outperform",
                "10Y": "outperform",
                "trade": "bull_steepener",
            },
            "bear_flattening": {
                "2Y": "underperform",
                "10Y": "strong_underperform",
                "trade": "bear_flattener",
            },
        }
        
    def generate_curve_trades(self, curve_data: Dict) -> List[Dict]:
        """Generate yield curve trades based on curve dynamics.
        
        Args:
            curve_data: Curve analysis data
            
        Returns:
            List of curve trades
        """
        two_year_yield = curve_data.get("2y_yield", 5.0)
        ten_year_yield = curve_data.get("10y_yield", 4.5)
        curve_trend = curve_data.get("curve_trend", "stable")
        fed_policy = curve_data.get("fed_policy", "neutral")
        
        current_spread = ten_year_yield - two_year_yield
        trades = []
        
        # Determine regime
        regime = self._determine_regime(curve_data)
        
        if regime == "steepening" and current_spread < 0:
            # Inverted curve likely to steepen
            trades.extend(self._create_steepener_trade())
        elif regime == "flattening" and current_spread > 1.0:
            # Steep curve likely to flatten
            trades.extend(self._create_flattener_trade())
        elif regime == "bull_steepening":
            # Rates falling, curve steepening
            trades.extend(self._create_bull_steepener_trade())
        elif regime == "bear_flattening":
            # Rates rising, curve flattening
            trades.extend(self._create_bear_flattener_trade())
        
        return trades
    
    def _determine_regime(self, curve_data: Dict) -> str:
        """Determine the yield curve regime.
        
        Args:
            curve_data: Curve data
            
        Returns:
            Regime type
        """
        curve_trend = curve_data.get("curve_trend", "stable")
        fed_policy = curve_data.get("fed_policy", "neutral")
        two_year_trend = curve_data.get("2y_trend", "stable")
        ten_year_trend = curve_data.get("10y_trend", "stable")
        
        if curve_trend == "steepening":
            if two_year_trend == "falling" and ten_year_trend in ["stable", "falling"]:
                return "bull_steepening"
            else:
                return "steepening"
        elif curve_trend == "flattening":
            if two_year_trend == "rising" and ten_year_trend in ["stable", "rising"]:
                return "bear_flattening"
            else:
                return "flattening"
        else:
            return "stable"
    
    def _create_steepener_trade(self) -> List[Dict]:
        """Create curve steepener trade.
        
        Returns:
            List of trades for steepener
        """
        return [
            {
                "action": "long",
                "instrument": "IEF",  # 7-10 year Treasury ETF
                "size": 1.0,
                "rationale": "Long intermediate duration",
            },
            {
                "action": "short",
                "instrument": "SHY",  # 1-3 year Treasury ETF
                "size": 1.0,
                "rationale": "Short front-end duration",
            },
        ]
    
    def _create_flattener_trade(self) -> List[Dict]:
        """Create curve flattener trade.
        
        Returns:
            List of trades for flattener
        """
        return [
            {
                "action": "short",
                "instrument": "IEF",  # 7-10 year Treasury ETF
                "size": 1.0,
                "rationale": "Short intermediate duration",
            },
            {
                "action": "long",
                "instrument": "SHY",  # 1-3 year Treasury ETF
                "size": 1.0,
                "rationale": "Long front-end duration",
            },
        ]
    
    def _create_bull_steepener_trade(self) -> List[Dict]:
        """Create bull steepener trade (rates falling, curve steepening).
        
        Returns:
            List of trades
        """
        return [
            {
                "action": "long",
                "instrument": "TLT",  # Long-term treasuries
                "size": 0.7,
                "rationale": "Long duration benefits most from falling rates",
            },
            {
                "action": "long",
                "instrument": "IEF",  # Intermediate treasuries
                "size": 0.3,
                "rationale": "Moderate long exposure",
            },
        ]
    
    def _create_bear_flattener_trade(self) -> List[Dict]:
        """Create bear flattener trade (rates rising, curve flattening).
        
        Returns:
            List of trades
        """
        return [
            {
                "action": "short",
                "instrument": "TLT",  # Long-term treasuries
                "size": 1.0,
                "rationale": "Short long duration in rising rates",
            },
            {
                "action": "long",
                "instrument": "SGOV",  # Ultra-short treasuries
                "size": 1.0,
                "rationale": "Safe haven in short duration",
            },
        ]
    
    def calculate_curve_trade_size(
        self,
        account_size: float,
        risk_budget: float,
        curve_volatility: float
    ) -> Dict[str, float]:
        """Calculate appropriate size for curve trades.
        
        Args:
            account_size: Total account value
            risk_budget: Risk budget as percentage
            curve_volatility: Historical curve volatility
            
        Returns:
            Position sizes for each leg
        """
        # Risk budget in dollars
        dollar_risk = account_size * risk_budget
        
        # Assume curve trade has half the volatility of outright position
        adjusted_volatility = curve_volatility * 0.5
        
        # Position size based on volatility
        total_position = dollar_risk / adjusted_volatility if adjusted_volatility > 0 else 0
        
        # Split between legs (usually equal weight)
        return {
            "long_leg": total_position * 0.5,
            "short_leg": total_position * 0.5,
        }
    
    def monitor_curve_trade(
        self,
        entry_spread: float,
        current_spread: float,
        target_spread: float,
        stop_spread: float
    ) -> str:
        """Monitor an active curve trade.
        
        Args:
            entry_spread: Spread at trade entry
            current_spread: Current spread
            target_spread: Target spread
            stop_spread: Stop loss spread
            
        Returns:
            Action to take (hold/close/stop)
        """
        # For steepener (long spread)
        if target_spread > entry_spread:
            if current_spread >= target_spread:
                return "close_profit"
            elif current_spread <= stop_spread:
                return "close_loss"
            else:
                return "hold"
        
        # For flattener (short spread)
        else:
            if current_spread <= target_spread:
                return "close_profit"
            elif current_spread >= stop_spread:
                return "close_loss"
            else:
                return "hold"
    
    def get_historical_curve_stats(self, period: str = "2Y") -> Dict[str, float]:
        """Get historical curve statistics.
        
        Args:
            period: Historical period
            
        Returns:
            Curve statistics
        """
        # In production, would fetch real historical data
        # Mock data for demonstration
        return {
            "avg_2s10s": 0.75,
            "std_2s10s": 0.60,
            "min_2s10s": -0.50,
            "max_2s10s": 2.50,
            "current_percentile": 0.25,  # Current spread in historical context
        }