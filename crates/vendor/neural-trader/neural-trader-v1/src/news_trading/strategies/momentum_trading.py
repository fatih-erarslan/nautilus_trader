"""Momentum Trading Strategy implementation for trend following."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
from src.news_trading.decision_engine.models import (
    TradingSignal, SignalType, RiskLevel, TradingStrategy, AssetType
)


class MomentumTradingStrategy:
    """Momentum trading strategy for following strong trends."""
    
    def __init__(self, lookback_periods: List[int] = None, momentum_threshold: float = 0.50):
        """
        Initialize momentum trading strategy.
        
        Args:
            lookback_periods: Periods for momentum calculation (default [5, 20, 60])
            momentum_threshold: Minimum momentum score to trade (default 0.50)
        """
        self.lookback_periods = lookback_periods or [5, 20, 60]
        self.momentum_threshold = momentum_threshold
        self.momentum_thresholds = {
            "strong": 0.75,
            "moderate": 0.50,
            "weak": 0.25
        }
        
    def calculate_momentum_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate composite momentum score.
        
        Args:
            data: Market data with price changes and indicators
            
        Returns:
            Momentum score between 0 and 1
        """
        scores = []
        
        # Price momentum (40% weight)
        price_score = self._score_price_momentum(
            data.get("price_change_5d", 0),
            data.get("price_change_20d", 0)
        )
        scores.append(price_score * 0.4)
        
        # Volume momentum (20% weight)
        volume_ratio = data.get("volume_ratio_5d", 1.0)
        volume_score = min(volume_ratio / 2, 1.0)  # Cap at 1.0
        scores.append(volume_score * 0.2)
        
        # Relative strength (25% weight)
        rs = data.get("relative_strength", 50)
        rs_score = rs / 100
        scores.append(rs_score * 0.25)
        
        # Fundamental momentum (15% weight)
        fundamental_score = self._score_fundamentals(data)
        scores.append(fundamental_score * 0.15)
        
        # Cap total score at 1.0
        return min(sum(scores), 1.0)
        
    def get_momentum_tier(self, score: float) -> str:
        """Get momentum tier based on score."""
        if score >= self.momentum_thresholds["strong"]:
            return "strong"
        elif score >= self.momentum_thresholds["moderate"]:
            return "moderate"
        else:
            return "weak"
            
    def _score_price_momentum(self, change_5d: float, change_20d: float) -> float:
        """
        Score price momentum with acceleration bonus.
        
        Args:
            change_5d: 5-day price change
            change_20d: 20-day price change
            
        Returns:
            Price momentum score
        """
        # Acceleration bonus if 5d > 20d annualized
        acceleration = 1.2 if (change_5d * 4) > change_20d else 1.0
        
        # Base score on 20d performance
        if change_20d > 0.20:
            base_score = 1.0
        elif change_20d > 0.10:
            base_score = 0.7
        elif change_20d > 0.05:
            base_score = 0.4
        else:
            base_score = 0.0
            
        # Apply acceleration and cap at 1.0
        return min(base_score * acceleration, 1.0)
            
    def _score_fundamentals(self, data: Dict[str, Any]) -> float:
        """Score fundamental momentum factors."""
        score = 0.5  # Base score
        
        # Earnings revision
        if data.get("earnings_revision") == "positive":
            score += 0.2
        elif data.get("earnings_revision") == "negative":
            score -= 0.2
            
        # Analyst momentum
        analyst_momentum = data.get("analyst_momentum", 0)
        if analyst_momentum > 3:
            score += 0.2
        elif analyst_momentum < -3:
            score -= 0.2
            
        # Sector rank
        sector_rank = data.get("sector_rank", 10)
        if sector_rank <= 3:
            score += 0.1
            
        return max(0, min(1, score))
        
    async def analyze_earnings_momentum(self, earnings_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze earnings-based momentum.
        
        Args:
            earnings_data: Earnings information
            
        Returns:
            Momentum signal if strong earnings momentum detected
        """
        ticker = earnings_data["ticker"]
        eps_surprise = (earnings_data["eps_actual"] - earnings_data["eps_estimate"]) / earnings_data["eps_estimate"]
        revenue_surprise = (earnings_data["revenue_actual"] - earnings_data["revenue_estimate"]) / earnings_data["revenue_estimate"]
        
        # Check for consistent beats
        surprise_history = earnings_data.get("surprise_history", [])
        avg_surprise = sum(surprise_history) / len(surprise_history) if surprise_history else 0
        
        # Strong earnings momentum criteria
        if (eps_surprise > 0.15 and revenue_surprise > 0.05 and 
            earnings_data.get("guidance") == "raised" and avg_surprise > 0.10):
            
            strength = min(0.6 + eps_surprise + avg_surprise, 1.0)
            
            return {
                "ticker": ticker,
                "momentum_type": "earnings_acceleration",
                "strength": strength,
                "confidence": 0.80,
                "suggested_holding": "4-8 weeks",
                "reasoning": f"Strong earnings beat ({eps_surprise:.1%}) with raised guidance"
            }
            
        return None
        
    async def identify_sector_rotation(self, sector_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify sector rotation opportunities.
        
        Args:
            sector_data: Performance data by sector
            
        Returns:
            Sector rotation signals
        """
        # Sort sectors by performance
        sector_performance = []
        for ticker, data in sector_data.items():
            perf = data["performance_1m"]
            volume = data["volume_surge"]
            # Combined score
            score = perf * 0.7 + (volume - 1.0) * 0.3
            sector_performance.append((ticker, perf, volume, score))
            
        # Sort by combined score
        sector_performance.sort(key=lambda x: x[3], reverse=True)
        
        # Identify leaders and laggards
        long_sectors = []
        avoid_sectors = []
        
        for ticker, perf, volume, score in sector_performance:
            if perf > 0.03 and volume > 1.2:  # Positive with volume
                long_sectors.append(ticker)
            elif perf < -0.01:  # Negative performance
                avoid_sectors.append(ticker)
                
        # Determine rotation theme
        theme = ""
        if "XLE" in long_sectors[:2]:
            theme = "energy_leading"
        elif "XLK" in long_sectors[:2]:
            theme = "tech_momentum"
            
        rotation_strength = abs(sector_performance[0][1] - sector_performance[-1][1])
        
        return {
            "long_sectors": long_sectors[:2],  # Top 2
            "avoid_sectors": avoid_sectors,
            "rotation_strength": min(rotation_strength * 10, 1.0),
            "theme": theme
        }
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Generate momentum trading signal.
        
        Args:
            market_data: Market data with momentum indicators
            
        Returns:
            TradingSignal if momentum is strong enough
        """
        # Calculate momentum score
        momentum_score = self.calculate_momentum_score(market_data)
        
        if momentum_score < self.momentum_threshold:
            return None
            
        ticker = market_data["ticker"]
        price = market_data["price"]
        atr = market_data.get("atr", price * 0.02)
        
        # Momentum trades use tighter stops
        stop_loss = price - (1.5 * atr)  # Tighter than swing
        
        # Take profit based on momentum strength
        momentum_tier = self.get_momentum_tier(momentum_score)
        if momentum_tier == "strong":
            take_profit = price + (5 * atr)  # Let winners run
        else:
            take_profit = price + (3 * atr)
            
        # Position sizing based on momentum
        position_calc = self.calculate_position_size(
            momentum_score=momentum_score,
            volatility=atr / price,
            account_size=100000  # Default
        )
        
        signal = TradingSignal(
            id=f"momentum-{str(uuid.uuid4())[:8]}",
            timestamp=datetime.now(),
            asset=ticker,
            asset_type=AssetType.EQUITY,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MOMENTUM,
            strength=momentum_score,
            confidence=min(momentum_score * 1.1, 0.95),
            risk_level=self._determine_risk_level(momentum_score),
            position_size=position_calc["position_pct"],
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period="1-4 weeks",
            source_events=["momentum-analysis"],
            reasoning=f"Strong {momentum_tier} momentum detected, trend following entry",
            momentum_score=momentum_score,
            technical_indicators={
                "momentum_tier": momentum_tier,
                "relative_strength": market_data.get("relative_strength"),
                "volume_surge": market_data.get("volume_ratio_5d")
            }
        )
        
        return signal
        
    def calculate_position_size(self, momentum_score: float, 
                              volatility: float, 
                              account_size: float) -> Dict[str, Any]:
        """
        Calculate position size based on momentum strength.
        
        Args:
            momentum_score: Momentum score (0-1)
            volatility: Asset volatility
            account_size: Total account size
            
        Returns:
            Position sizing details
        """
        # Base position size on momentum strength
        if momentum_score >= 0.75:
            base_position = 0.08  # 8% for strong momentum
        elif momentum_score >= 0.50:
            base_position = 0.05  # 5% for moderate
        else:
            base_position = 0.03  # 3% for weak
            
        # Adjust for volatility
        if volatility > 0.03:  # High volatility
            base_position *= 0.7
            
        position_pct = min(base_position, 0.08)  # Cap at 8%
        
        return {
            "position_pct": position_pct,
            "position_value": account_size * position_pct,
            "momentum_adjusted": True
        }
        
    def check_exit_conditions(self, position: Dict[str, Any], 
                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check momentum exit conditions.
        
        Args:
            position: Current position details
            market_data: Current market data
            
        Returns:
            Exit decision and reason
        """
        current_price = market_data["current_price"]
        current_momentum = market_data.get("current_momentum", 0)
        entry_momentum = position.get("entry_momentum", 0.5)
        
        # Stop loss hit
        if current_price <= position["stop_loss"]:
            return {"exit": True, "reason": "stop_loss_hit"}
            
        # Trend reversal - check this before momentum exhaustion
        if market_data.get("price_change_5d", 0) < -0.05:
            return {"exit": True, "reason": "trend_reversal"}
            
        # Momentum exhaustion
        if current_momentum < 0.4 and current_momentum < entry_momentum * 0.5:
            return {"exit": True, "reason": "momentum_exhaustion"}
            
        # Volume dry up
        if market_data.get("volume_ratio", 1) < 0.7:
            return {"exit": True, "reason": "volume_exhaustion"}
            
        return {"exit": False, "reason": None}
        
    async def generate_crypto_momentum_signal(self, crypto_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Generate momentum signal for cryptocurrency.
        
        Args:
            crypto_data: Crypto-specific market data
            
        Returns:
            TradingSignal for crypto if momentum detected
        """
        # Calculate crypto momentum with adjusted weights
        momentum_data = {
            "price_change_5d": crypto_data["price_change_5d"],
            "price_change_20d": crypto_data["price_change_20d"],
            "volume_ratio_5d": crypto_data["volume_ratio_5d"],
            "relative_strength": 80,  # Default high for crypto leaders
            "sector_rank": 1 if crypto_data.get("dominance_change", 0) > 0 else 5
        }
        
        momentum_score = self.calculate_momentum_score(momentum_data)
        
        if momentum_score < 0.60:  # Higher threshold for crypto
            return None
            
        ticker = crypto_data["ticker"]
        price = crypto_data["price"]
        atr = crypto_data.get("atr", price * 0.04)
        
        signal = TradingSignal(
            id=f"crypto-momentum-{str(uuid.uuid4())[:8]}",
            timestamp=datetime.now(),
            asset=ticker,
            asset_type=AssetType.CRYPTO,
            signal_type=SignalType.BUY,
            strategy=TradingStrategy.MOMENTUM,
            strength=momentum_score,
            confidence=momentum_score * 0.9,  # Slightly lower confidence for crypto
            risk_level=RiskLevel.HIGH,  # Always high for crypto
            position_size=min(0.05, momentum_score * 0.06),  # Conservative sizing
            entry_price=price,
            stop_loss=price - (2 * atr),
            take_profit=price + (6 * atr),  # Higher targets for crypto
            holding_period="1-8 weeks",
            source_events=["crypto-momentum"],
            reasoning=f"Crypto momentum surge detected with {crypto_data['price_change_20d']:.1%} gain",
            momentum_score=momentum_score
        )
        
        return signal
        
    def _determine_risk_level(self, momentum_score: float) -> RiskLevel:
        """Determine risk level based on momentum score."""
        if momentum_score >= 0.8:
            return RiskLevel.MEDIUM  # Strong trends are safer
        elif momentum_score >= 0.6:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH  # Weak momentum is risky