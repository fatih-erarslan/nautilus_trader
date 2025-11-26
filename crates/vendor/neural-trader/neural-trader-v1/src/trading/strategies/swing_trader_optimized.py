"""Optimized Swing Trading Strategy with Advanced Pattern Recognition and Risk Management."""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class SwingPattern:
    """Represents a detected swing trading pattern."""
    pattern_type: str
    confidence: float
    entry_zone: Tuple[float, float]
    stop_loss: float
    target_prices: List[float]
    risk_reward: float
    timeframe: str
    supporting_factors: List[str]


class OptimizedSwingTradingEngine:
    """
    Optimized swing trading engine with multi-pattern recognition,
    dynamic position sizing, and advanced risk management.
    """
    
    def __init__(self, account_size: float = 100000, max_risk_per_trade: float = 0.015):
        """
        Initialize optimized swing trading engine.
        
        Args:
            account_size: Total account size for position sizing
            max_risk_per_trade: Maximum risk per trade (1.5% default)
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        
        # Optimized parameters based on backtesting
        self.min_risk_reward = 2.0  # Increased from 1.5
        self.max_position_pct = 0.12  # Reduced from 50% to 12% per position
        self.max_portfolio_heat = 0.06  # Max 6% portfolio risk at any time
        self.max_holding_days = 8  # Reduced from 10
        self.trailing_stop_activation = 0.04  # Activate trailing stop at 4% profit
        
        # Pattern recognition parameters
        self.pattern_weights = {
            'bullish_flag': 1.2,
            'ascending_triangle': 1.15,
            'inverse_head_shoulders': 1.25,
            'double_bottom': 1.1,
            'pullback_to_support': 1.0,
            'breakout_consolidation': 1.05,
            'volume_surge': 0.9,
            'momentum_divergence': 0.95
        }
        
        # Market regime parameters
        self.regime_adjustments = {
            'bull': {'position_multiplier': 1.2, 'stop_tightness': 0.9},
            'bear': {'position_multiplier': 0.5, 'stop_tightness': 1.2},
            'volatile': {'position_multiplier': 0.6, 'stop_tightness': 1.3},
            'sideways': {'position_multiplier': 0.8, 'stop_tightness': 1.0}
        }
        
        # Risk tracking
        self.open_positions = []
        self.daily_trades = 0
        self.last_trade_date = None
        self.consecutive_losses = 0
        self.portfolio_heat = 0.0
        
    def identify_swing_patterns(self, market_data: Dict) -> List[SwingPattern]:
        """
        Identify multiple swing trading patterns with confidence scoring.
        
        Args:
            market_data: Comprehensive market data
            
        Returns:
            List of detected swing patterns sorted by confidence
        """
        patterns = []
        
        # Extract data
        price = market_data["price"]
        high = market_data.get("high", price)
        low = market_data.get("low", price)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        atr = market_data.get("atr", price * 0.02)
        
        # Pattern 1: Bullish Flag Pattern
        if self._detect_bullish_flag(market_data):
            patterns.append(SwingPattern(
                pattern_type="bullish_flag",
                confidence=0.85,
                entry_zone=(price * 0.99, price * 1.01),
                stop_loss=price - (1.5 * atr),
                target_prices=[price + (2 * atr), price + (3 * atr)],
                risk_reward=2.5,
                timeframe="4H-Daily",
                supporting_factors=["Strong trend", "Volume confirmation", "Flag consolidation"]
            ))
        
        # Pattern 2: Ascending Triangle Breakout
        if self._detect_ascending_triangle(market_data):
            patterns.append(SwingPattern(
                pattern_type="ascending_triangle",
                confidence=0.80,
                entry_zone=(price * 1.0, price * 1.02),
                stop_loss=market_data.get("triangle_support", price * 0.97),
                target_prices=[price * 1.05, price * 1.08],
                risk_reward=2.8,
                timeframe="Daily",
                supporting_factors=["Higher lows", "Resistance test", "Volume increase"]
            ))
        
        # Pattern 3: Pullback to Key Moving Average
        ma_50 = market_data.get("ma_50", price)
        if self._detect_ma_pullback(market_data):
            patterns.append(SwingPattern(
                pattern_type="pullback_to_support",
                confidence=0.75,
                entry_zone=(ma_50 * 1.005, ma_50 * 1.015),
                stop_loss=ma_50 * 0.98,
                target_prices=[price * 1.04, price * 1.07],
                risk_reward=2.2,
                timeframe="Daily",
                supporting_factors=["MA support", "Oversold bounce", "Trend continuation"]
            ))
        
        # Pattern 4: Breakout from Consolidation
        if self._detect_consolidation_breakout(market_data):
            patterns.append(SwingPattern(
                pattern_type="breakout_consolidation",
                confidence=0.78,
                entry_zone=(price * 1.0, price * 1.015),
                stop_loss=market_data.get("consolidation_low", price * 0.97),
                target_prices=[price * 1.06, price * 1.10],
                risk_reward=3.0,
                timeframe="Daily-Weekly",
                supporting_factors=["Range breakout", "Volume surge", "Momentum increase"]
            ))
        
        # Sort patterns by confidence
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply market regime filter
        regime = self._identify_market_regime(market_data)
        filtered_patterns = []
        
        for pattern in patterns:
            # Adjust confidence based on market regime
            if regime == 'bear' and 'bullish' in pattern.pattern_type:
                pattern.confidence *= 0.7
            elif regime == 'volatile':
                pattern.confidence *= 0.8
            
            # Only keep high confidence patterns
            if pattern.confidence >= 0.65:
                filtered_patterns.append(pattern)
        
        return filtered_patterns[:3]  # Return top 3 patterns
    
    def calculate_dynamic_position_size(self, pattern: SwingPattern, market_data: Dict) -> Dict:
        """
        Calculate position size with Kelly Criterion and volatility adjustment.
        
        Args:
            pattern: Detected swing pattern
            market_data: Current market data
            
        Returns:
            Position sizing details
        """
        entry_price = np.mean(pattern.entry_zone)
        stop_loss = pattern.stop_loss
        risk_per_share = abs(entry_price - stop_loss)
        
        # Kelly Criterion calculation
        win_rate = 0.65  # Historical win rate
        avg_win = pattern.risk_reward
        kelly_fraction = (win_rate * avg_win - (1 - win_rate)) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Volatility adjustment
        atr = market_data.get("atr", entry_price * 0.02)
        volatility_factor = min(1.0, (entry_price * 0.02) / atr)
        
        # Pattern confidence adjustment
        confidence_factor = pattern.confidence
        
        # Market regime adjustment
        regime = self._identify_market_regime(market_data)
        regime_factor = self.regime_adjustments[regime]['position_multiplier']
        
        # Calculate base position size
        max_risk_amount = self.account_size * self.max_risk_per_trade
        base_shares = int(max_risk_amount / risk_per_share)
        
        # Apply all adjustments
        adjusted_shares = int(base_shares * kelly_fraction * volatility_factor * 
                            confidence_factor * regime_factor)
        
        # Check position limits
        position_value = adjusted_shares * entry_price
        max_position_value = self.account_size * self.max_position_pct
        
        if position_value > max_position_value:
            adjusted_shares = int(max_position_value / entry_price)
            position_value = adjusted_shares * entry_price
        
        # Check portfolio heat
        position_risk = adjusted_shares * risk_per_share
        if self.portfolio_heat + (position_risk / self.account_size) > self.max_portfolio_heat:
            # Reduce position to stay within heat limit
            available_risk = max(0, self.max_portfolio_heat - self.portfolio_heat)
            adjusted_shares = int((available_risk * self.account_size) / risk_per_share)
        
        actual_risk = adjusted_shares * risk_per_share
        
        return {
            "shares": adjusted_shares,
            "position_value": adjusted_shares * entry_price,
            "risk_amount": actual_risk,
            "position_pct": (adjusted_shares * entry_price) / self.account_size,
            "risk_pct": actual_risk / self.account_size,
            "kelly_fraction": kelly_fraction,
            "volatility_adjustment": volatility_factor,
            "confidence_adjustment": confidence_factor,
            "regime_adjustment": regime_factor
        }
    
    def create_trailing_stop_strategy(self, position: Dict, market_data: Dict) -> Dict:
        """
        Create adaptive trailing stop strategy based on volatility and profit.
        
        Args:
            position: Current position details
            market_data: Current market data
            
        Returns:
            Trailing stop configuration
        """
        current_price = market_data["current_price"]
        entry_price = position["entry_price"]
        profit_pct = (current_price - entry_price) / entry_price
        atr = market_data.get("atr", current_price * 0.02)
        
        # Dynamic trailing stop based on profit tiers
        if profit_pct < self.trailing_stop_activation:
            # Fixed stop until activation threshold
            return {
                "type": "fixed",
                "stop_price": position["stop_loss"],
                "trail_amount": 0,
                "status": "waiting_for_profit"
            }
        elif profit_pct < 0.08:
            # Tight trailing stop for small profits
            trail_amount = 1.5 * atr
        elif profit_pct < 0.15:
            # Medium trailing stop for moderate profits
            trail_amount = 2.0 * atr
        else:
            # Wider trailing stop for large profits
            trail_amount = 2.5 * atr
        
        # Calculate trailing stop price
        highest_price = market_data.get("highest_price_since_entry", current_price)
        trailing_stop_price = highest_price - trail_amount
        
        # Never let trailing stop go below breakeven
        breakeven_price = entry_price * 1.002  # Include commission
        trailing_stop_price = max(trailing_stop_price, breakeven_price)
        
        return {
            "type": "trailing",
            "stop_price": trailing_stop_price,
            "trail_amount": trail_amount,
            "trail_pct": trail_amount / highest_price,
            "profit_locked": max(0, (trailing_stop_price - entry_price) / entry_price),
            "status": "active"
        }
    
    def check_advanced_exit_conditions(self, position: Dict, pattern: SwingPattern, market_data: Dict) -> Dict:
        """
        Check multiple exit conditions with priority system.
        
        Args:
            position: Current position
            pattern: Original pattern that triggered entry
            market_data: Current market data
            
        Returns:
            Exit decision with detailed reasoning
        """
        current_price = market_data["current_price"]
        entry_price = position["entry_price"]
        holding_days = (datetime.now() - position["entry_date"]).days
        
        # Priority 1: Stop Loss
        if current_price <= position["stop_loss"]:
            return {
                "exit": True,
                "reason": "stop_loss_hit",
                "priority": 1,
                "exit_price": current_price,
                "profit_pct": (current_price - entry_price) / entry_price
            }
        
        # Priority 2: Target Hit
        for i, target in enumerate(pattern.target_prices):
            if current_price >= target:
                return {
                    "exit": True,
                    "reason": f"target_{i+1}_hit",
                    "priority": 2,
                    "exit_price": current_price,
                    "profit_pct": (current_price - entry_price) / entry_price,
                    "target_level": i + 1
                }
        
        # Priority 3: Trailing Stop
        trailing_stop = self.create_trailing_stop_strategy(position, market_data)
        if trailing_stop["type"] == "trailing" and current_price <= trailing_stop["stop_price"]:
            return {
                "exit": True,
                "reason": "trailing_stop_hit",
                "priority": 3,
                "exit_price": current_price,
                "profit_pct": (current_price - entry_price) / entry_price,
                "profit_locked": trailing_stop["profit_locked"]
            }
        
        # Priority 4: Pattern Breakdown
        if self._detect_pattern_breakdown(pattern, market_data):
            return {
                "exit": True,
                "reason": "pattern_breakdown",
                "priority": 4,
                "exit_price": current_price,
                "profit_pct": (current_price - entry_price) / entry_price,
                "breakdown_factors": market_data.get("breakdown_factors", [])
            }
        
        # Priority 5: Time-based Exit
        if holding_days > self.max_holding_days:
            return {
                "exit": True,
                "reason": "max_holding_period",
                "priority": 5,
                "exit_price": current_price,
                "profit_pct": (current_price - entry_price) / entry_price,
                "holding_days": holding_days
            }
        
        # Priority 6: Adverse Market Regime Change
        if self._detect_adverse_regime_change(position, market_data):
            return {
                "exit": True,
                "reason": "adverse_regime_change",
                "priority": 6,
                "exit_price": current_price,
                "profit_pct": (current_price - entry_price) / entry_price,
                "new_regime": market_data.get("current_regime", "unknown")
            }
        
        # No exit conditions met
        return {
            "exit": False,
            "current_profit_pct": (current_price - entry_price) / entry_price,
            "holding_days": holding_days,
            "trailing_stop": trailing_stop,
            "pattern_health": self._assess_pattern_health(pattern, market_data)
        }
    
    def _detect_bullish_flag(self, market_data: Dict) -> bool:
        """Detect bullish flag pattern."""
        prices = market_data.get("price_history", [])
        if len(prices) < 20:
            return False
        
        # Look for strong uptrend followed by consolidation
        trend_prices = prices[-20:-10]
        flag_prices = prices[-10:]
        
        trend_return = (trend_prices[-1] - trend_prices[0]) / trend_prices[0]
        flag_volatility = np.std(flag_prices) / np.mean(flag_prices)
        
        return (trend_return > 0.05 and  # 5% move in trend
                flag_volatility < 0.02 and  # Low volatility in flag
                min(flag_prices) > trend_prices[-1] * 0.97)  # Shallow pullback
    
    def _detect_ascending_triangle(self, market_data: Dict) -> bool:
        """Detect ascending triangle pattern."""
        highs = market_data.get("recent_highs", [])
        lows = market_data.get("recent_lows", [])
        
        if len(highs) < 3 or len(lows) < 3:
            return False
        
        # Check for flat top (resistance)
        high_variance = np.std(highs[-3:]) / np.mean(highs[-3:])
        
        # Check for rising lows
        low_trend = (lows[-1] - lows[-3]) / lows[-3]
        
        return high_variance < 0.01 and low_trend > 0.02
    
    def _detect_ma_pullback(self, market_data: Dict) -> bool:
        """Detect pullback to moving average support."""
        price = market_data["price"]
        ma_50 = market_data.get("ma_50", price)
        rsi = market_data.get("rsi_14", 50)
        trend = market_data.get("trend", "neutral")
        
        # Price near MA50, oversold RSI, in uptrend
        distance_to_ma = abs(price - ma_50) / ma_50
        
        return (distance_to_ma < 0.02 and  # Within 2% of MA
                rsi < 40 and  # Oversold
                trend == "bullish" and  # In uptrend
                price > ma_50 * 0.99)  # Just above MA
    
    def _detect_consolidation_breakout(self, market_data: Dict) -> bool:
        """Detect breakout from consolidation range."""
        prices = market_data.get("price_history", [])
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        if len(prices) < 20:
            return False
        
        # Check for recent consolidation
        consolidation_prices = prices[-15:-1]
        consolidation_range = (max(consolidation_prices) - min(consolidation_prices)) / np.mean(consolidation_prices)
        
        # Check for breakout
        current_price = market_data["price"]
        breakout = current_price > max(consolidation_prices) * 1.01
        
        return (consolidation_range < 0.05 and  # Tight range
                breakout and  # Price breakout
                volume_ratio > 1.5)  # Volume confirmation
    
    def _identify_market_regime(self, market_data: Dict) -> str:
        """Identify current market regime."""
        vix = market_data.get("vix", 20)
        trend_strength = market_data.get("trend_strength", 0)
        
        if vix > 30:
            return "volatile"
        elif trend_strength > 0.7:
            return "bull"
        elif trend_strength < -0.7:
            return "bear"
        else:
            return "sideways"
    
    def _detect_pattern_breakdown(self, pattern: SwingPattern, market_data: Dict) -> bool:
        """Detect if the original pattern has broken down."""
        price = market_data["current_price"]
        
        # Check if price has fallen below pattern support
        if pattern.pattern_type == "bullish_flag":
            flag_support = market_data.get("flag_support", pattern.stop_loss)
            return price < flag_support
        elif pattern.pattern_type == "ascending_triangle":
            triangle_support = market_data.get("triangle_support", pattern.stop_loss)
            return price < triangle_support
        
        # Generic breakdown: price below entry zone
        return price < pattern.entry_zone[0] * 0.98
    
    def _detect_adverse_regime_change(self, position: Dict, market_data: Dict) -> bool:
        """Detect adverse market regime change."""
        entry_regime = position.get("entry_regime", "unknown")
        current_regime = self._identify_market_regime(market_data)
        
        # Bull to bear or volatile is adverse
        if entry_regime == "bull" and current_regime in ["bear", "volatile"]:
            return True
        
        # Any position in volatile regime (except if entered in volatile)
        if current_regime == "volatile" and entry_regime != "volatile":
            return True
        
        return False
    
    def _assess_pattern_health(self, pattern: SwingPattern, market_data: Dict) -> float:
        """Assess the health of the pattern (0-1 score)."""
        score = 1.0
        price = market_data["current_price"]
        
        # Check if price is still within expected range
        if price < pattern.entry_zone[0]:
            score *= 0.7
        
        # Check if supporting factors still valid
        current_volume = market_data.get("volume_ratio", 1.0)
        if "Volume confirmation" in pattern.supporting_factors and current_volume < 0.8:
            score *= 0.8
        
        # Check trend alignment
        if market_data.get("trend", "neutral") != "bullish":
            score *= 0.6
        
        return max(0, min(score, 1.0))
    
    def generate_swing_signal(self, market_data: Dict) -> Dict:
        """
        Generate comprehensive swing trading signal.
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal with detailed analysis
        """
        # Reset daily trade counter if new day
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
        
        # Check if we can take more trades today
        if self.daily_trades >= 3:
            return {
                "action": "hold",
                "reason": "Daily trade limit reached",
                "patterns": [],
                "risk_assessment": {"portfolio_heat": self.portfolio_heat}
            }
        
        # Identify swing patterns
        patterns = self.identify_swing_patterns(market_data)
        
        if not patterns:
            return {
                "action": "hold",
                "reason": "No high-confidence patterns detected",
                "patterns": [],
                "market_regime": self._identify_market_regime(market_data)
            }
        
        # Use highest confidence pattern
        best_pattern = patterns[0]
        
        # Calculate position size
        position_size = self.calculate_dynamic_position_size(best_pattern, market_data)
        
        # Check if position size is viable
        if position_size["shares"] == 0:
            return {
                "action": "hold",
                "reason": "Position size too small or portfolio heat limit reached",
                "patterns": [best_pattern.pattern_type],
                "portfolio_heat": self.portfolio_heat
            }
        
        # Prepare entry signal
        entry_price = np.mean(best_pattern.entry_zone)
        
        signal = {
            "action": "buy",
            "pattern": best_pattern.pattern_type,
            "confidence": best_pattern.confidence,
            "entry_zone": best_pattern.entry_zone,
            "entry_price": entry_price,
            "stop_loss": best_pattern.stop_loss,
            "target_prices": best_pattern.target_prices,
            "position_size": position_size,
            "risk_reward": best_pattern.risk_reward,
            "market_regime": self._identify_market_regime(market_data),
            "supporting_factors": best_pattern.supporting_factors,
            "risk_assessment": {
                "portfolio_heat": self.portfolio_heat,
                "consecutive_losses": self.consecutive_losses,
                "position_risk_pct": position_size["risk_pct"]
            }
        }
        
        # Update tracking
        self.daily_trades += 1
        self.portfolio_heat += position_size["risk_pct"]
        
        return signal