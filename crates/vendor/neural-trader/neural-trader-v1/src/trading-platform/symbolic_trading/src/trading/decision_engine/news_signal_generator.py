"""
News Signal Generator
Converts sentiment analysis data into actionable trading signals
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import re

from .models import (
    TradingSignal, SignalType, RiskLevel, 
    TradingStrategy, AssetType
)


class NewsSignalGenerator:
    """
    Generates trading signals from news sentiment data
    
    Analyzes sentiment scores, confidence levels, and market impact
    to produce risk-managed trading signals with appropriate sizing
    and risk parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with configuration
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "sentiment_thresholds": {
                "bullish": 0.3,
                "bearish": -0.3,
                "strong_bullish": 0.6,
                "strong_bearish": -0.6
            },
            "position_size_base": 0.05,  # 5% base position
            "position_size_max": 0.10,   # 10% max position
            "confidence_min": 0.6,        # Minimum confidence to trade
            "atr_multipliers": {
                "stop_loss": {
                    "short-term": 1.5,
                    "medium-term": 2.0,
                    "long-term": 2.5
                },
                "take_profit": {
                    "short-term": 2.5,
                    "medium-term": 3.5,
                    "long-term": 5.0
                }
            }
        }
    
    async def generate_signal(self, sentiment_data: Dict[str, Any]) -> TradingSignal:
        """
        Generate trading signal from sentiment data
        
        Args:
            sentiment_data: Dictionary containing sentiment analysis results
            
        Returns:
            TradingSignal object
        """
        # Extract key data
        asset = sentiment_data["asset"]
        sentiment_score = sentiment_data["sentiment_score"]
        confidence = sentiment_data["confidence"]
        
        # Determine signal type
        signal_type = self._determine_signal_type(sentiment_score)
        
        # Calculate signal strength
        strength = abs(sentiment_score)
        
        # Get current market data
        market_data = await self._fetch_market_data(asset)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            strength, confidence, sentiment_data.get("market_impact", {})
        )
        
        # Determine strategy
        strategy = self._determine_strategy(sentiment_data)
        
        # Calculate risk levels
        entry_price = market_data["current_price"]
        stop_loss, take_profit = self._calculate_risk_levels(
            entry_price, signal_type, market_data, sentiment_data
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(sentiment_data, market_data)
        
        # Determine holding period
        holding_period = self._determine_holding_period(strategy, sentiment_data)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(sentiment_data, signal_type, confidence)
        
        # Detect asset type
        asset_type = self._detect_asset_type(asset)
        
        return TradingSignal(
            id=self._generate_signal_id(),
            timestamp=datetime.now(),
            asset=asset,
            asset_type=asset_type,
            signal_type=signal_type,
            strategy=strategy,
            strength=strength,
            confidence=confidence,
            risk_level=risk_level,
            position_size=position_size if signal_type != SignalType.HOLD else 0,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=holding_period,
            source_events=sentiment_data.get("source_events", []),
            reasoning=reasoning
        )
    
    def _determine_signal_type(self, sentiment_score: float) -> SignalType:
        """Determine signal type based on sentiment score"""
        thresholds = self.config["sentiment_thresholds"]
        
        if sentiment_score > thresholds["bullish"]:
            return SignalType.BUY
        elif sentiment_score < thresholds["bearish"]:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    async def _fetch_market_data(self, asset: str) -> Dict[str, Any]:
        """
        Fetch current market data for asset
        
        This is a placeholder that should be replaced with actual
        market data fetching logic in production.
        """
        # This method would normally fetch real market data
        # For now, return mock data
        return {
            "current_price": 100.0,
            "atr": 5.0,
            "volume_24h": 1000000
        }
    
    def _calculate_position_size(self, strength: float, confidence: float, 
                               market_impact: Dict[str, Any]) -> float:
        """Calculate position size based on signal strength and confidence"""
        base_size = self.config["position_size_base"]
        max_size = self.config["position_size_max"]
        
        # Scale by confidence and strength
        size = base_size * confidence * strength
        
        # Adjust for market impact magnitude
        impact_magnitude = market_impact.get("magnitude", 0.5)
        size *= (0.5 + impact_magnitude * 0.5)
        
        # Cap at maximum
        return min(size, max_size)
    
    def _determine_strategy(self, sentiment_data: Dict[str, Any]) -> TradingStrategy:
        """Determine trading strategy based on sentiment data"""
        market_impact = sentiment_data.get("market_impact", {})
        timeframe = market_impact.get("timeframe", "medium-term")
        
        if timeframe == "short-term":
            # High impact short-term news suits swing trading
            if market_impact.get("magnitude", 0) > 0.7:
                return TradingStrategy.DAY_TRADE
            else:
                return TradingStrategy.SWING
        elif timeframe == "long-term":
            # Long-term fundamental changes suit position trading
            return TradingStrategy.POSITION
        else:
            # Medium-term default to swing
            return TradingStrategy.SWING
    
    def _calculate_risk_levels(self, entry_price: float, signal_type: SignalType,
                             market_data: Dict[str, Any], 
                             sentiment_data: Dict[str, Any]) -> tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        atr = market_data["atr"]
        market_impact = sentiment_data.get("market_impact", {})
        timeframe = market_impact.get("timeframe", "short-term")
        
        # Get ATR multipliers based on timeframe
        stop_multiplier = self.config["atr_multipliers"]["stop_loss"][timeframe]
        profit_multiplier = self.config["atr_multipliers"]["take_profit"][timeframe]
        
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - (stop_multiplier * atr)
            take_profit = entry_price + (profit_multiplier * atr)
        elif signal_type == SignalType.SELL:
            stop_loss = entry_price + (stop_multiplier * atr)
            take_profit = entry_price - (profit_multiplier * atr)
        else:  # HOLD
            stop_loss = entry_price
            take_profit = entry_price
        
        return stop_loss, take_profit
    
    def _assess_risk_level(self, sentiment_data: Dict[str, Any], 
                          market_data: Dict[str, Any]) -> RiskLevel:
        """Assess risk level based on various factors"""
        # Calculate volatility ratio (ATR as % of price)
        volatility_ratio = market_data["atr"] / market_data["current_price"]
        
        # Check expected volatility from sentiment
        market_impact = sentiment_data.get("market_impact", {})
        expected_volatility = market_impact.get("volatility_expected", "medium")
        
        # Consider sentiment direction for risk assessment
        sentiment_score = sentiment_data.get("sentiment_score", 0)
        is_bearish = sentiment_score < 0
        
        # Assess based on volatility and sentiment
        if volatility_ratio > 0.10 or expected_volatility == "extreme":
            return RiskLevel.EXTREME
        elif volatility_ratio > 0.08 or expected_volatility == "high":
            return RiskLevel.HIGH
        elif volatility_ratio > 0.05 or is_bearish:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_holding_period(self, strategy: TradingStrategy, 
                                sentiment_data: Dict[str, Any]) -> str:
        """Determine expected holding period based on strategy"""
        timeframe = sentiment_data.get("market_impact", {}).get("timeframe", "medium-term")
        
        if strategy == TradingStrategy.DAY_TRADE:
            return "1 day"
        elif strategy == TradingStrategy.SWING:
            if timeframe == "short-term":
                return "3-7 days"
            else:
                return "5-10 days"
        elif strategy == TradingStrategy.POSITION:
            return "1-3 months"
        elif strategy == TradingStrategy.MOMENTUM:
            return "1-3 days"
        else:
            return "3-7 days"  # Default
    
    def _generate_reasoning(self, sentiment_data: Dict[str, Any], 
                          signal_type: SignalType, confidence: float) -> str:
        """Generate human-readable reasoning for the signal"""
        direction = sentiment_data.get("market_impact", {}).get("direction", "neutral")
        magnitude = sentiment_data.get("market_impact", {}).get("magnitude", 0.5)
        catalysts = sentiment_data.get("market_impact", {}).get("catalysts", [])
        
        # Build reasoning
        confidence_pct = int(confidence * 100)
        sentiment_strength = abs(sentiment_data["sentiment_score"])
        
        reasoning = f"{direction.capitalize()} sentiment detected with {confidence_pct}% confidence. "
        reasoning += f"Sentiment strength: {sentiment_strength:.2f}. "
        
        if catalysts:
            reasoning += f"Key catalysts: {', '.join(catalysts[:2])}. "
        
        if signal_type != SignalType.HOLD:
            reasoning += f"Market impact expected to be {magnitude:.1%}."
        
        return reasoning
    
    def _detect_asset_type(self, asset: str) -> AssetType:
        """Detect asset type from symbol"""
        # Common crypto symbols
        crypto_symbols = {"BTC", "ETH", "ADA", "SOL", "DOT", "MATIC", "AVAX", "LINK"}
        
        # Bond patterns
        bond_patterns = [r"US\d+Y", r"T-\w+", r"BOND", r"YIELD"]
        
        # Forex patterns
        forex_patterns = [r"[A-Z]{3}/[A-Z]{3}", r"[A-Z]{6}"]
        
        # Check crypto
        if asset.upper() in crypto_symbols or asset.endswith("-USD") or asset.endswith("USDT"):
            return AssetType.CRYPTO
        
        # Check bonds
        for pattern in bond_patterns:
            if re.match(pattern, asset.upper()):
                return AssetType.BOND
        
        # Check forex
        for pattern in forex_patterns:
            if re.match(pattern, asset.upper()) and len(asset) == 6:
                return AssetType.FOREX
        
        # Default to equity
        return AssetType.EQUITY
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        return f"signal-{uuid.uuid4().hex[:8]}"