"""Swing Trading Strategy implementation for 3-10 day holding periods."""

from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import uuid
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class SwingTradingStrategy:
    """Swing trading strategy with 3-10 day holding periods."""
    
    def __init__(self, account_size: float = 100000, 
                 max_risk_per_trade: float = 0.02,
                 min_risk_reward: float = 1.5):
        """
        Initialize swing trading strategy.
        
        Args:
            account_size: Total account size for position sizing
            max_risk_per_trade: Maximum risk per trade (default 2%)
            min_risk_reward: Minimum risk/reward ratio (default 1.5)
        """
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.min_risk_reward = min_risk_reward
        self.max_position_pct = 0.10  # Maximum 10% per position for safety
        self.max_holding_days = 10
        self.min_holding_days = 3
        
    def identify_swing_setup(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify valid swing trading setups.
        
        Args:
            market_data: Dictionary containing price and technical indicators
            
        Returns:
            Dictionary with setup validity and details
        """
        price = market_data.get("price", 0)
        ma_50 = market_data.get("ma_50", 0)
        ma_200 = market_data.get("ma_200", 0)
        rsi = market_data.get("rsi_14", 50)
        volume_ratio = market_data.get("volume_ratio", 1.0)
        
        # Bullish continuation setup
        if price > ma_50 > ma_200 and 40 < rsi < 70 and volume_ratio > 1.0:
            return {
                "valid": True,
                "setup_type": "bullish_continuation",
                "entry_zone": (ma_50 * 1.005, ma_50 * 1.015),
                "confidence": 0.75
            }
            
        # Oversold bounce setup
        if price < ma_50 and rsi < 30 and volume_ratio > 1.2:
            support = market_data.get("support_level", price * 0.98)
            return {
                "valid": True,
                "setup_type": "oversold_bounce",
                "entry_zone": (support * 0.995, support * 1.005),
                "confidence": 0.65
            }
            
        # Overbought or unclear setup
        if rsi > 70 or volume_ratio < 0.9:
            return {"valid": False}
            
        return {"valid": False}
        
    def calculate_position_size(self, trade_setup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate position size based on risk management.
        
        Args:
            trade_setup: Dictionary with entry_price, stop_loss, and atr
            
        Returns:
            Dictionary with position sizing details
        """
        entry_price = trade_setup["entry_price"]
        stop_loss = trade_setup["stop_loss"]
        risk_per_share = abs(entry_price - stop_loss)
        
        # Maximum risk amount
        max_risk_amount = self.account_size * self.max_risk_per_trade
        
        # Calculate shares based on risk
        shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = shares * entry_price
        
        # Check position size limits (max 10% of account)
        max_position_value = self.account_size * self.max_position_pct
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
            
        return {
            "shares": shares,
            "position_value": position_value,
            "risk_amount": shares * risk_per_share,
            "position_pct": position_value / self.account_size if self.account_size > 0 else 0
        }
        
    async def generate_signal(self, market_data: Dict[str, Any], 
                            news_catalyst: Optional[Dict[str, Any]] = None) -> Optional[TradingSignal]:
        """
        Generate swing trading signal.
        
        Args:
            market_data: Market data including price and indicators
            news_catalyst: Optional news data for confirmation
            
        Returns:
            TradingSignal if valid setup, None otherwise
        """
        setup = self.identify_swing_setup(market_data)
        if not setup["valid"]:
            return None
            
        # Calculate entry, stop, and target
        price = market_data["price"]
        atr = market_data.get("atr_14", price * 0.02)
        
        # Swing trades use 2 ATR stops and aim for 3+ ATR targets
        stop_loss = price - (2 * atr)
        take_profit = price + (3.5 * atr)
        
        # Adjust confidence based on news catalyst
        confidence = setup["confidence"]
        if news_catalyst and news_catalyst.get("sentiment_score", news_catalyst.get("sentiment", 0)) > 0.7:
            confidence = min(confidence * 1.2, 0.95)
            
        # Calculate position size
        position_calc = self.calculate_position_size({
            "entry_price": price,
            "stop_loss": stop_loss,
            "atr": atr
        })
        
        # Create trading signal
        signal = TradingSignal(
            id=f"swing-{str(uuid.uuid4())[:8]}",
            timestamp=datetime.now(),
            asset=market_data["ticker"],
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.SWING,
            strength=confidence,
            confidence=confidence,
            risk_level=self._determine_risk_level(setup["setup_type"]),
            position_size=position_calc["position_pct"],
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period="3-7 days",
            source_events=[news_catalyst.get("source_id", "news")] if news_catalyst else [],
            reasoning=self._generate_reasoning(setup, market_data, news_catalyst),
            technical_indicators={
                "setup_type": setup["setup_type"],
                "rsi": market_data.get("rsi_14"),
                "volume_ratio": market_data.get("volume_ratio"),
                "ma_position": "above" if price > market_data.get("ma_50", price) else "below"
            }
        )
        
        return signal
        
    def check_exit_conditions(self, position: Dict[str, Any], 
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if position should be exited.
        
        Args:
            position: Current position details
            market_data: Current market data
            
        Returns:
            Dictionary with exit decision and reason
        """
        current_price = market_data["current_price"]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        entry_date = position["entry_date"]
        
        # Check profit target
        if current_price >= take_profit:
            return {"exit": True, "reason": "profit_target_hit"}
            
        # Check stop loss
        if current_price <= stop_loss:
            return {"exit": True, "reason": "stop_loss_hit"}
            
        # Check max holding period
        holding_days = (datetime.now() - entry_date).days
        if holding_days > self.max_holding_days:
            return {"exit": True, "reason": "max_holding_period"}
            
        # Check trailing stop if in profit
        if current_price > entry_price:
            trailing_stop = position.get("trailing_stop_pct", 0.02)
            highest_price = market_data.get("highest_price_since_entry", current_price)
            trailing_stop_price = highest_price * (1 - trailing_stop)
            
            if current_price <= trailing_stop_price:
                return {"exit": True, "reason": "trailing_stop_hit"}
                
        return {"exit": False, "reason": None}
        
    async def generate_bond_swing_signal(self, bond_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Generate swing trading signal for bonds.
        
        Args:
            bond_data: Bond-specific market data
            
        Returns:
            TradingSignal for bonds if valid setup
        """
        ticker = bond_data["ticker"]
        current_yield = bond_data["yield_current"]
        ma_yield = bond_data["yield_ma_50"]
        price = bond_data["price"]
        atr = bond_data.get("atr", price * 0.005)
        
        # Bonds: Higher yields = lower prices
        # If current yield > MA yield, bonds are oversold
        if current_yield > ma_yield * 1.02 and bond_data.get("volume_ratio", 1) > 1.1:
            # Buy signal for oversold bonds
            stop_loss = price - (1.5 * atr)
            take_profit = price + (2.5 * atr)
            
            signal = TradingSignal(
                id=f"bond-swing-{str(uuid.uuid4())[:8]}",
                timestamp=datetime.now(),
                asset=ticker,
                asset_type=AssetType.BOND,
                signal_type=SignalType.BUY,
                strategy=TradingStrategy.SWING,
                strength=0.70,
                confidence=0.75,
                risk_level=RiskLevel.MEDIUM,
                position_size=0.08,  # Conservative for bonds
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                holding_period="5-15 days",  # Longer for bonds
                source_events=["bond-analysis"],
                reasoning="Yields above MA suggesting oversold bonds, expecting mean reversion"
            )
            
            return signal
            
        return None
        
    def _determine_risk_level(self, setup_type: str) -> RiskLevel:
        """Determine risk level based on setup type."""
        if setup_type == "bullish_continuation":
            return RiskLevel.MEDIUM
        elif setup_type == "oversold_bounce":
            return RiskLevel.HIGH
        else:
            return RiskLevel.MEDIUM
            
    def _generate_reasoning(self, setup: Dict[str, Any], 
                          market_data: Dict[str, Any],
                          news_catalyst: Optional[Dict[str, Any]]) -> str:
        """Generate reasoning for the trade."""
        reasoning = f"{setup['setup_type'].replace('_', ' ').title()} setup detected"
        
        if market_data.get("volume_ratio", 1) > 1.2:
            reasoning += " with strong volume confirmation"
            
        if news_catalyst and news_catalyst.get("sentiment_score", news_catalyst.get("sentiment", 0)) > 0.7:
            reasoning += f". Positive news catalyst: {news_catalyst.get('headline', 'N/A')}"
            
        return reasoning