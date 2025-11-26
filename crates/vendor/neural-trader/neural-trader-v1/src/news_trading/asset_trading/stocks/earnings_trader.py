"""Earnings gap trading strategies."""

from typing import Dict, Any, Tuple


class EarningsGapTrader:
    """Trades post-earnings gaps and momentum."""
    
    def __init__(self):
        """Initialize the earnings gap trader."""
        self.min_gap_percent = 3.0  # Minimum gap to consider
        self.max_gap_percent = 15.0  # Maximum gap (avoid extremes)
        
    def analyze_earnings_gap(self, earnings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze post-earnings gap for trading opportunity.
        
        Args:
            earnings_data: Dictionary with earnings information
            
        Returns:
            Trading signal with entry zones and targets
        """
        ticker = earnings_data.get("ticker", "")
        eps_actual = earnings_data.get("eps_actual", 0)
        eps_estimate = earnings_data.get("eps_estimate", 0)
        revenue_beat = earnings_data.get("revenue_beat", False)
        guidance = earnings_data.get("guidance", "maintained")
        gap_percent = earnings_data.get("gap_percent", 0)
        pre_earnings_price = earnings_data.get("pre_earnings_price", 0)
        current_price = earnings_data.get("current_price", 0)
        
        # Calculate earnings beat percentage
        if eps_estimate != 0:
            eps_beat_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
        else:
            eps_beat_pct = 0
        
        # Analyze gap quality
        gap_quality = self._assess_gap_quality(
            eps_beat_pct, revenue_beat, guidance, gap_percent
        )
        
        # Determine trading action
        if gap_quality == "high" and self.min_gap_percent <= abs(gap_percent) <= self.max_gap_percent:
            if gap_percent > 0:
                return self._create_bullish_gap_trade(earnings_data)
            else:
                return self._create_bearish_gap_trade(earnings_data)
        
        return {
            "action": "no_trade",
            "reason": f"Gap quality: {gap_quality}, Gap size: {gap_percent}%",
        }
    
    def _assess_gap_quality(
        self,
        eps_beat_pct: float,
        revenue_beat: bool,
        guidance: str,
        gap_percent: float
    ) -> str:
        """Assess the quality of the earnings gap.
        
        Args:
            eps_beat_pct: EPS beat percentage
            revenue_beat: Whether revenue beat estimates
            guidance: Guidance status
            gap_percent: Gap percentage
            
        Returns:
            Gap quality rating (high/medium/low)
        """
        quality_score = 0
        
        # EPS beat quality
        if eps_beat_pct > 10:
            quality_score += 2
        elif eps_beat_pct > 5:
            quality_score += 1
        elif eps_beat_pct < -10:
            quality_score -= 2
        elif eps_beat_pct < -5:
            quality_score -= 1
        
        # Revenue beat
        if revenue_beat:
            quality_score += 1
        
        # Guidance impact
        if guidance == "raised":
            quality_score += 2
        elif guidance == "lowered":
            quality_score -= 2
        
        # Gap direction alignment
        if (gap_percent > 0 and quality_score > 0) or (gap_percent < 0 and quality_score < 0):
            quality_score = abs(quality_score)
        else:
            quality_score = 0  # Gap doesn't align with fundamentals
        
        # Determine quality rating
        if quality_score >= 3:
            return "high"
        elif quality_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _create_bullish_gap_trade(self, earnings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create bullish gap trading setup.
        
        Args:
            earnings_data: Earnings data
            
        Returns:
            Bullish trading signal
        """
        pre_price = earnings_data["pre_earnings_price"]
        current_price = earnings_data["current_price"]
        gap_percent = earnings_data["gap_percent"]
        
        # Calculate entry zones
        if gap_percent > 5:
            # Wait for pullback on large gaps
            entry_zone_high = pre_price * 1.04
            entry_zone_low = pre_price * 1.02
            action = "buy_pullback"
        else:
            # Can enter on smaller gaps
            entry_zone_high = current_price * 1.005
            entry_zone_low = current_price * 0.995
            action = "buy"
        
        # Calculate targets
        target_1 = pre_price * (1 + gap_percent / 100 * 1.5)
        target_2 = pre_price * (1 + gap_percent / 100 * 2.0)
        
        # Stop loss below gap fill
        stop_loss = pre_price * 0.98
        
        return {
            "action": action,
            "entry_zone": (round(entry_zone_low, 2), round(entry_zone_high, 2)),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "holding_period": "3-5 days",
            "risk_reward": round((target_1 - entry_zone_high) / (entry_zone_high - stop_loss), 1),
        }
    
    def _create_bearish_gap_trade(self, earnings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create bearish gap trading setup.
        
        Args:
            earnings_data: Earnings data
            
        Returns:
            Bearish trading signal
        """
        pre_price = earnings_data["pre_earnings_price"]
        current_price = earnings_data["current_price"]
        gap_percent = earnings_data["gap_percent"]
        
        # Calculate entry zones
        if abs(gap_percent) > 5:
            # Wait for bounce on large gaps
            entry_zone_low = pre_price * 0.96
            entry_zone_high = pre_price * 0.98
            action = "short_bounce"
        else:
            # Can enter on smaller gaps
            entry_zone_low = current_price * 0.995
            entry_zone_high = current_price * 1.005
            action = "short"
        
        # Calculate targets
        target_1 = pre_price * (1 + gap_percent / 100 * 1.5)
        target_2 = pre_price * (1 + gap_percent / 100 * 2.0)
        
        # Stop loss above gap fill
        stop_loss = pre_price * 1.02
        
        return {
            "action": action,
            "entry_zone": (round(entry_zone_low, 2), round(entry_zone_high, 2)),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "holding_period": "3-5 days",
            "risk_reward": round((entry_zone_low - target_1) / (stop_loss - entry_zone_low), 1),
        }
    
    def calculate_position_size(
        self,
        account_size: float,
        risk_per_trade: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """Calculate appropriate position size for earnings gap trade.
        
        Args:
            account_size: Total account value
            risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Number of shares to trade
        """
        # Calculate dollar risk
        dollar_risk = account_size * risk_per_trade
        
        # Calculate per-share risk
        per_share_risk = abs(entry_price - stop_loss)
        
        if per_share_risk == 0:
            return 0
        
        # Calculate position size
        shares = int(dollar_risk / per_share_risk)
        
        # Ensure we don't exceed account size
        max_shares = int(account_size * 0.25 / entry_price)  # Max 25% position
        
        return min(shares, max_shares)