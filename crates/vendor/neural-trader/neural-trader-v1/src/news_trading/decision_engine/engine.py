"""Complete Trading Decision Engine implementation."""

from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from .base import TradingDecisionEngine
from .models import TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
from .risk_manager import RiskManager
from ..strategies import SwingTradingStrategy, MomentumTradingStrategy, MirrorTradingStrategy


class NewsDecisionEngine(TradingDecisionEngine):
    """Main trading decision engine that converts sentiment to signals."""
    
    def __init__(self, account_size: float = 100000, max_portfolio_risk: float = 0.2):
        """
        Initialize decision engine with strategies and risk management.
        
        Args:
            account_size: Total account size
            max_portfolio_risk: Maximum portfolio risk allowed
        """
        self.account_size = account_size
        self.risk_manager = RiskManager(
            max_position_size=0.1,
            max_portfolio_risk=max_portfolio_risk,
            max_correlation=0.7
        )
        
        # Initialize strategies
        self.strategies = {
            "swing": SwingTradingStrategy(account_size),
            "momentum": MomentumTradingStrategy(),
            "mirror": MirrorTradingStrategy(account_size)
        }
        
        # Active signals storage
        self._active_signals: List[TradingSignal] = []
        
    async def process_sentiment(self, sentiment_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Process sentiment data into trading signal.
        
        Args:
            sentiment_data: Sentiment analysis results
            
        Returns:
            Trading signal if actionable
        """
        sentiment_score = sentiment_data.get("sentiment_score", 0)
        confidence = sentiment_data.get("confidence", 0)
        
        # Skip neutral or low confidence sentiments
        if abs(sentiment_score) < 0.3 or confidence < 0.6:
            return None
            
        # Determine strategy based on market conditions
        strategy = self._select_strategy(sentiment_data)
        
        # Generate market data for the asset
        market_data = await self._fetch_market_data(sentiment_data["asset"])
        
        # Generate signal based on strategy
        signal = None
        if strategy == "momentum":
            signal = await self.strategies["momentum"].generate_signal(market_data)
        elif strategy == "swing":
            signal = await self.strategies["swing"].generate_signal(
                market_data, 
                news_catalyst=sentiment_data
            )
            
        if signal:
            # Override some fields from sentiment
            signal.asset = sentiment_data["asset"]
            signal.source_events = sentiment_data.get("source_events", [])
            
            # Determine asset type
            asset_type = sentiment_data.get("asset_type", "equity")
            signal.asset_type = self._map_asset_type(asset_type)
            
            # Adjust signal type based on sentiment
            if sentiment_score < 0:
                if signal.signal_type == SignalType.BUY:
                    signal.signal_type = SignalType.SELL
                    
            # Validate with risk manager
            signal = await self.risk_manager.validate_signal(signal)
            
            # Add to active signals
            if signal.position_size > 0:
                self._active_signals.append(signal)
                
        return signal
        
    async def evaluate_portfolio(self, current_positions: Dict[str, Any]) -> List[TradingSignal]:
        """
        Evaluate portfolio and generate rebalancing signals.
        
        Args:
            current_positions: Current portfolio positions
            
        Returns:
            List of rebalancing signals
        """
        signals = []
        
        for asset, position in current_positions.items():
            pnl = position.get("unrealized_pnl", 0)
            
            # Take profits if up > 20%
            if pnl > 0.20:
                signal = TradingSignal(
                    id=f"rebalance-{str(uuid.uuid4())[:8]}",
                    timestamp=datetime.now(),
                    asset=asset,
                    asset_type=AssetType.EQUITY,  # Default
                    signal_type=SignalType.SELL,
                    strategy=TradingStrategy.POSITION,
                    strength=0.8,
                    confidence=0.9,
                    risk_level=RiskLevel.LOW,
                    position_size=position["size"] * 0.5,  # Sell half
                    entry_price=position["current_price"],
                    stop_loss=position["current_price"] * 0.98,
                    take_profit=position["current_price"] * 1.05,
                    holding_period="immediate",
                    source_events=["portfolio-rebalance"],
                    reasoning=f"Taking profits after {pnl:.1%} gain"
                )
                signals.append(signal)
                
            # Cut losses if down > 15%
            elif pnl < -0.15:
                signal = TradingSignal(
                    id=f"stop-loss-{str(uuid.uuid4())[:8]}",
                    timestamp=datetime.now(),
                    asset=asset,
                    asset_type=AssetType.EQUITY,
                    signal_type=SignalType.SELL,
                    strategy=TradingStrategy.POSITION,
                    strength=0.9,
                    confidence=0.9,
                    risk_level=RiskLevel.HIGH,
                    position_size=position["size"],  # Sell all
                    entry_price=position["current_price"],
                    stop_loss=position["current_price"] * 0.95,
                    take_profit=position["current_price"] * 1.02,
                    holding_period="immediate",
                    source_events=["stop-loss"],
                    reasoning=f"Cutting losses after {pnl:.1%} decline"
                )
                signals.append(signal)
                
        return signals
        
    async def process_market_data(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """
        Process market data for technical signals.
        
        Args:
            market_data: Market data by asset
            
        Returns:
            List of technical trading signals
        """
        signals = []
        
        for asset, data in market_data.items():
            # Add ticker to data
            data["ticker"] = asset
            
            # Check momentum conditions
            if (data.get("price_change_20d", 0) > 0.15 and 
                data.get("volume_ratio", 1) > 1.5):
                
                momentum_signal = await self.strategies["momentum"].generate_signal(data)
                if momentum_signal:
                    momentum_signal.asset = asset
                    signals.append(momentum_signal)
                    
            # Check swing setup
            if (data.get("rsi", 50) < 70 and 
                data.get("price", 0) > data.get("ma_50", 0)):
                
                swing_signal = await self.strategies["swing"].generate_signal(data)
                if swing_signal:
                    swing_signal.asset = asset
                    signals.append(swing_signal)
                    
        return signals
        
    def set_risk_parameters(self, params: Dict[str, Any]) -> None:
        """Update risk management parameters."""
        if "max_position_size" in params:
            self.risk_manager.max_position_size = params["max_position_size"]
        if "max_portfolio_risk" in params:
            self.risk_manager.max_portfolio_risk = params["max_portfolio_risk"]
        if "max_correlation" in params:
            self.risk_manager.max_correlation = params["max_correlation"]
            
    def get_active_signals(self) -> List[TradingSignal]:
        """Get currently active trading signals."""
        return self._active_signals.copy()
        
    def _select_strategy(self, sentiment_data: Dict[str, Any]) -> str:
        """Select appropriate strategy based on conditions."""
        market_impact = sentiment_data.get("market_impact", {})
        magnitude = market_impact.get("magnitude", 0)
        
        # Check for strong trend conditions
        market_conditions = sentiment_data.get("market_conditions", {})
        if market_conditions.get("trend") == "strong_uptrend" and magnitude > 0.7:
            return "momentum"
            
        # Default to swing trading
        return "swing"
        
    async def _fetch_market_data(self, asset: str) -> Dict[str, Any]:
        """Fetch market data for asset (mock implementation)."""
        # In real implementation, this would fetch from market data provider
        return {
            "ticker": asset,
            "price": 100.0,
            "ma_50": 98.0,
            "ma_200": 95.0,
            "rsi_14": 55,
            "rsi": 55,  # Alias for rsi_14
            "volume_ratio": 1.2,
            "atr_14": 2.0,
            "atr": 2.0,  # Alias for atr_14
            "price_change_5d": 0.05,
            "price_change_20d": 0.10,
            "relative_strength": 75,
            "volume_ratio_5d": 1.2
        }
        
    def _map_asset_type(self, asset_type_str: str) -> AssetType:
        """Map string asset type to enum."""
        mapping = {
            "equity": AssetType.EQUITY,
            "stock": AssetType.EQUITY,
            "bond": AssetType.BOND,
            "crypto": AssetType.CRYPTO,
            "commodity": AssetType.COMMODITY,
            "forex": AssetType.FOREX
        }
        return mapping.get(asset_type_str.lower(), AssetType.EQUITY)