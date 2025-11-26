"""
Analytics and metrics data models for Polymarket

This module defines data structures for trading signals, sentiment analysis,
risk metrics, and performance analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any


class SignalStrength(Enum):
    """Trading signal strength enumeration."""
    
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    NEUTRAL = "neutral"
    STRONG = "strong"
    VERY_STRONG = "very_strong"
    
    @property
    def numeric_value(self) -> float:
        """Convert signal strength to numeric value."""
        mapping = {
            "very_weak": -2.0,
            "weak": -1.0,
            "neutral": 0.0,
            "strong": 1.0,
            "very_strong": 2.0
        }
        return mapping[self.value]


@dataclass
class TradingSignal:
    """Trading signal with direction and confidence."""
    
    id: str
    market_id: str
    outcome_id: str
    signal_type: str  # "buy", "sell", "hold"
    strength: SignalStrength
    confidence: Decimal  # 0.0 to 1.0
    price_target: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    reasoning: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate signal data."""
        if not isinstance(self.confidence, Decimal):
            self.confidence = Decimal(str(self.confidence))
        if self.price_target and not isinstance(self.price_target, Decimal):
            self.price_target = Decimal(str(self.price_target))
        if self.stop_loss and not isinstance(self.stop_loss, Decimal):
            self.stop_loss = Decimal(str(self.stop_loss))
        
        # Validate confidence range
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    @property
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_type.lower() == "buy"
    
    @property
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_type.lower() == "sell"
    
    @property
    def weighted_strength(self) -> float:
        """Calculate confidence-weighted signal strength."""
        return self.strength.numeric_value * float(self.confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "signal_type": self.signal_type,
            "strength": self.strength.value,
            "confidence": float(self.confidence),
            "reasoning": self.reasoning,
            "generated_at": self.generated_at.isoformat(),
            "is_expired": self.is_expired,
            "weighted_strength": self.weighted_strength
        }
        
        if self.price_target:
            result["price_target"] = float(self.price_target)
        if self.stop_loss:
            result["stop_loss"] = float(self.stop_loss)
        if self.expires_at:
            result["expires_at"] = self.expires_at.isoformat()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingSignal":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            signal_type=data["signal_type"],
            strength=SignalStrength(data["strength"]),
            confidence=Decimal(str(data["confidence"])),
            price_target=Decimal(str(data["price_target"])) if data.get("price_target") else None,
            stop_loss=Decimal(str(data["stop_loss"])) if data.get("stop_loss") else None,
            reasoning=data.get("reasoning", ""),
            generated_at=datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00")),
            expires_at=datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00")) if data.get("expires_at") else None
        )


@dataclass
class SentimentScore:
    """Market sentiment analysis score."""
    
    market_id: str
    outcome_id: str
    sentiment: Decimal  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: Decimal  # 0.0 to 1.0
    source: str
    keywords: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate sentiment data."""
        if not isinstance(self.sentiment, Decimal):
            self.sentiment = Decimal(str(self.sentiment))
        if not isinstance(self.confidence, Decimal):
            self.confidence = Decimal(str(self.confidence))
        
        # Validate ranges
        if not (-1 <= self.sentiment <= 1):
            raise ValueError(f"Sentiment must be between -1 and 1, got {self.sentiment}")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment < 0
    
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return self.sentiment == 0
    
    @property
    def weighted_sentiment(self) -> float:
        """Calculate confidence-weighted sentiment."""
        return float(self.sentiment * self.confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "sentiment": float(self.sentiment),
            "confidence": float(self.confidence),
            "source": self.source,
            "keywords": self.keywords,
            "generated_at": self.generated_at.isoformat(),
            "is_bullish": self.is_bullish,
            "is_bearish": self.is_bearish,
            "weighted_sentiment": self.weighted_sentiment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentimentScore":
        """Create instance from dictionary."""
        return cls(
            market_id=data["market_id"],
            outcome_id=data["outcome_id"],
            sentiment=Decimal(str(data["sentiment"])),
            confidence=Decimal(str(data["confidence"])),
            source=data["source"],
            keywords=data.get("keywords", []),
            generated_at=datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        )


@dataclass
class PriceDataPoint:
    """Single price data point with timestamp"""
    timestamp: datetime
    price: Decimal
    volume: Decimal
    
    def __post_init__(self):
        """Convert to Decimal types"""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.volume, Decimal):
            self.volume = Decimal(str(self.volume))
        
        # Validate price range
        if self.price < 0 or self.price > 1:
            raise ValueError("Price must be between 0 and 1")
        if self.volume < 0:
            raise ValueError("Volume must be non-negative")


@dataclass
class PriceHistory:
    """Historical price data for a market outcome"""
    market_id: str
    outcome_id: str
    prices: List[PriceDataPoint]
    start_date: datetime
    end_date: datetime
    interval: str
    
    def __post_init__(self):
        """Validate and sort price history"""
        # Sort prices by timestamp
        self.prices.sort(key=lambda x: x.timestamp)
        
        # Validate date range
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
    
    @property
    def total_volume(self) -> Decimal:
        """Get total volume across all price points"""
        return sum(point.volume for point in self.prices)
    
    @property
    def avg_price(self) -> Decimal:
        """Get volume-weighted average price"""
        if not self.prices:
            return Decimal('0')
        
        total_value = sum(point.price * point.volume for point in self.prices)
        total_volume = self.total_volume
        
        if total_volume == 0:
            return sum(point.price for point in self.prices) / len(self.prices)
        
        return total_value / total_volume
    
    @property
    def price_change(self) -> Optional[Decimal]:
        """Get price change from first to last point"""
        if len(self.prices) < 2:
            return None
        
        first_price = self.prices[0].price
        last_price = self.prices[-1].price
        
        if first_price == 0:
            return None
        
        return (last_price - first_price) / first_price


@dataclass
class VolumeData:
    """Volume analytics for a market"""
    market_id: str
    total_volume: Decimal
    volume_24h: Decimal
    volume_7d: Decimal
    volume_30d: Decimal
    peak_volume_24h: Decimal
    avg_trade_size: Decimal
    trade_count_24h: int
    unique_traders_24h: int
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Convert to Decimal types"""
        decimal_fields = [
            'total_volume', 'volume_24h', 'volume_7d', 'volume_30d',
            'peak_volume_24h', 'avg_trade_size'
        ]
        
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))


@dataclass  
class LiquidityMetrics:
    """Liquidity metrics for a market"""
    market_id: str
    total_liquidity: Decimal
    bid_liquidity: Decimal
    ask_liquidity: Decimal
    spread: Decimal
    depth_1_percent: Decimal  # Liquidity within 1% of mid price
    depth_5_percent: Decimal  # Liquidity within 5% of mid price
    market_impact_score: Decimal  # Higher = less liquid
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Convert to Decimal types"""
        decimal_fields = [
            'total_liquidity', 'bid_liquidity', 'ask_liquidity', 'spread',
            'depth_1_percent', 'depth_5_percent', 'market_impact_score'
        ]
        
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))


@dataclass
class MarketAnalytics:
    """Comprehensive market analytics"""
    market_id: str
    volume_24h: Decimal
    volume_7d: Decimal
    volume_30d: Decimal
    price_change_24h: Decimal
    price_change_7d: Decimal
    liquidity: Decimal
    spread: Decimal
    participants: int
    trades_24h: int
    last_trade_price: Decimal
    last_updated: datetime
    
    # Additional metrics
    momentum_score: Optional[Decimal] = None
    sentiment_score: Optional[Decimal] = None
    prediction_confidence: Optional[Decimal] = None
    
    def __post_init__(self):
        """Convert to Decimal types"""
        decimal_fields = [
            'volume_24h', 'volume_7d', 'volume_30d', 'price_change_24h',
            'price_change_7d', 'liquidity', 'spread', 'last_trade_price'
        ]
        
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))
        
        # Convert optional decimal fields
        for field_name in ['momentum_score', 'sentiment_score', 'prediction_confidence']:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))
    
    @property
    def activity_score(self) -> Decimal:
        """Calculate market activity score (0-100)"""
        # Normalize based on volume and trade count
        volume_score = min(self.volume_24h / 10000, 1) * 50  # Max 50 points for volume
        trade_score = min(self.trades_24h / 100, 1) * 30  # Max 30 points for trades
        participant_score = min(self.participants / 1000, 1) * 20  # Max 20 points for participants
        
        return volume_score + trade_score + participant_score
    
    @property
    def quality_score(self) -> Decimal:
        """Calculate market quality score (0-100)"""
        # Lower spread = higher quality
        spread_score = max(0, (Decimal('0.1') - self.spread) * 500)  # Max 50 points
        
        # Higher liquidity = higher quality  
        liquidity_score = min(self.liquidity / 50000, 1) * 30  # Max 30 points
        
        # More participants = higher quality
        participant_score = min(self.participants / 500, 1) * 20  # Max 20 points
        
        return min(spread_score + liquidity_score + participant_score, 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analytics to dictionary for serialization"""
        result = {
            'market_id': self.market_id,
            'volume_24h': float(self.volume_24h),
            'volume_7d': float(self.volume_7d),
            'volume_30d': float(self.volume_30d),
            'price_change_24h': float(self.price_change_24h),
            'price_change_7d': float(self.price_change_7d),
            'liquidity': float(self.liquidity),
            'spread': float(self.spread),
            'participants': self.participants,
            'trades_24h': self.trades_24h,
            'last_trade_price': float(self.last_trade_price),
            'last_updated': self.last_updated.isoformat(),
            'activity_score': float(self.activity_score),
            'quality_score': float(self.quality_score),
        }
        
        # Add optional fields if they exist
        if self.momentum_score is not None:
            result['momentum_score'] = float(self.momentum_score)
        if self.sentiment_score is not None:
            result['sentiment_score'] = float(self.sentiment_score)
        if self.prediction_confidence is not None:
            result['prediction_confidence'] = float(self.prediction_confidence)
        
        return result


@dataclass
class RiskMetrics:
    """Risk assessment metrics for markets and positions."""
    
    market_id: str
    volatility: Decimal
    liquidity_risk: Decimal  # 0.0 (low) to 1.0 (high)
    market_depth: Decimal
    bid_ask_spread: Decimal
    price_impact: Decimal
    var_95: Decimal  # Value at Risk at 95% confidence
    max_drawdown: Decimal
    beta: Optional[Decimal] = None
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Convert to Decimal types."""
        decimal_fields = [
            'volatility', 'liquidity_risk', 'market_depth', 
            'bid_ask_spread', 'price_impact', 'var_95', 'max_drawdown'
        ]
        
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))
        
        if self.beta and not isinstance(self.beta, Decimal):
            self.beta = Decimal(str(self.beta))
    
    @property
    def risk_level(self) -> str:
        """Calculate overall risk level."""
        # Weighted risk score
        risk_score = (
            float(self.volatility) * 0.3 +
            float(self.liquidity_risk) * 0.3 +
            float(self.bid_ask_spread) * 0.2 +
            float(self.price_impact) * 0.2
        )
        
        if risk_score < 0.2:
            return "low"
        elif risk_score < 0.4:
            return "medium"
        elif risk_score < 0.7:
            return "high"
        else:
            return "very_high"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "market_id": self.market_id,
            "volatility": float(self.volatility),
            "liquidity_risk": float(self.liquidity_risk),
            "market_depth": float(self.market_depth),
            "bid_ask_spread": float(self.bid_ask_spread),
            "price_impact": float(self.price_impact),
            "var_95": float(self.var_95),
            "max_drawdown": float(self.max_drawdown),
            "risk_level": self.risk_level,
            "calculated_at": self.calculated_at.isoformat()
        }
        
        if self.beta:
            result["beta"] = float(self.beta)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskMetrics":
        """Create instance from dictionary."""
        return cls(
            market_id=data["market_id"],
            volatility=Decimal(str(data["volatility"])),
            liquidity_risk=Decimal(str(data["liquidity_risk"])),
            market_depth=Decimal(str(data["market_depth"])),
            bid_ask_spread=Decimal(str(data["bid_ask_spread"])),
            price_impact=Decimal(str(data["price_impact"])),
            var_95=Decimal(str(data["var_95"])),
            max_drawdown=Decimal(str(data["max_drawdown"])),
            beta=Decimal(str(data["beta"])) if data.get("beta") else None,
            calculated_at=datetime.fromisoformat(data["calculated_at"].replace("Z", "+00:00"))
        )


@dataclass
class PerformanceMetrics:
    """Performance analysis metrics."""
    
    strategy_id: str
    time_period: str
    total_return: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: Decimal
    average_loss: Decimal
    profit_factor: Decimal
    calculated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Convert to Decimal types and validate."""
        decimal_fields = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'average_win', 'average_loss', 'profit_factor'
        ]
        
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))
    
    @property
    def loss_rate(self) -> Decimal:
        """Calculate loss rate."""
        return Decimal('1') - self.win_rate
    
    @property
    def risk_reward_ratio(self) -> Decimal:
        """Calculate risk-reward ratio."""
        if self.average_loss != 0:
            return abs(self.average_win / self.average_loss)
        return Decimal('0')
    
    @property
    def performance_rating(self) -> str:
        """Calculate overall performance rating."""
        # Weighted performance score
        score = (
            float(self.sharpe_ratio) * 0.4 +
            float(self.win_rate) * 0.3 +
            float(self.profit_factor) * 0.2 +
            max(0, -float(self.max_drawdown)) * 0.1  # Lower drawdown is better
        )
        
        if score >= 2.0:
            return "excellent"
        elif score >= 1.5:
            return "very_good"
        elif score >= 1.0:
            return "good"
        elif score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy_id": self.strategy_id,
            "time_period": self.time_period,
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "max_drawdown": float(self.max_drawdown),
            "win_rate": float(self.win_rate),
            "loss_rate": float(self.loss_rate),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "average_win": float(self.average_win),
            "average_loss": float(self.average_loss),
            "profit_factor": float(self.profit_factor),
            "risk_reward_ratio": float(self.risk_reward_ratio),
            "performance_rating": self.performance_rating,
            "calculated_at": self.calculated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create instance from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            time_period=data["time_period"],
            total_return=Decimal(str(data["total_return"])),
            annualized_return=Decimal(str(data["annualized_return"])),
            volatility=Decimal(str(data["volatility"])),
            sharpe_ratio=Decimal(str(data["sharpe_ratio"])),
            max_drawdown=Decimal(str(data["max_drawdown"])),
            win_rate=Decimal(str(data["win_rate"])),
            total_trades=data["total_trades"],
            winning_trades=data["winning_trades"],
            losing_trades=data["losing_trades"],
            average_win=Decimal(str(data["average_win"])),
            average_loss=Decimal(str(data["average_loss"])),
            profit_factor=Decimal(str(data["profit_factor"])),
            calculated_at=datetime.fromisoformat(data["calculated_at"].replace("Z", "+00:00"))
        )


@dataclass
class MarketAnalysis:
    """Comprehensive market analysis combining multiple metrics."""
    
    market_id: str
    price_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    sentiment_scores: List[SentimentScore] = field(default_factory=list)
    trading_signals: List[TradingSignal] = field(default_factory=list)
    risk_metrics: Optional[RiskMetrics] = None
    liquidity_metrics: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def overall_sentiment(self) -> Optional[Decimal]:
        """Calculate weighted average sentiment."""
        if not self.sentiment_scores:
            return None
        
        total_weight = sum(score.confidence for score in self.sentiment_scores)
        if total_weight == 0:
            return None
        
        weighted_sentiment = sum(
            score.sentiment * score.confidence 
            for score in self.sentiment_scores
        )
        
        return weighted_sentiment / total_weight
    
    @property
    def signal_consensus(self) -> Optional[str]:
        """Calculate consensus from trading signals."""
        if not self.trading_signals:
            return None
        
        buy_strength = sum(
            signal.weighted_strength for signal in self.trading_signals
            if signal.is_buy_signal
        )
        sell_strength = sum(
            abs(signal.weighted_strength) for signal in self.trading_signals
            if signal.is_sell_signal
        )
        
        if buy_strength > sell_strength * 1.2:
            return "strong_buy"
        elif buy_strength > sell_strength:
            return "buy"
        elif sell_strength > buy_strength * 1.2:
            return "strong_sell"
        elif sell_strength > buy_strength:
            return "sell"
        else:
            return "hold"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "market_id": self.market_id,
            "price_analysis": self.price_analysis,
            "volume_analysis": self.volume_analysis,
            "sentiment_scores": [score.to_dict() for score in self.sentiment_scores],
            "trading_signals": [signal.to_dict() for signal in self.trading_signals],
            "liquidity_metrics": self.liquidity_metrics,
            "generated_at": self.generated_at.isoformat(),
            "overall_sentiment": float(self.overall_sentiment) if self.overall_sentiment else None,
            "signal_consensus": self.signal_consensus
        }
        
        if self.risk_metrics:
            result["risk_metrics"] = self.risk_metrics.to_dict()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketAnalysis":
        """Create instance from dictionary."""
        sentiment_scores = [
            SentimentScore.from_dict(score_data) 
            for score_data in data.get("sentiment_scores", [])
        ]
        
        trading_signals = [
            TradingSignal.from_dict(signal_data)
            for signal_data in data.get("trading_signals", [])
        ]
        
        risk_metrics = None
        if data.get("risk_metrics"):
            risk_metrics = RiskMetrics.from_dict(data["risk_metrics"])
        
        return cls(
            market_id=data["market_id"],
            price_analysis=data.get("price_analysis", {}),
            volume_analysis=data.get("volume_analysis", {}),
            sentiment_scores=sentiment_scores,
            trading_signals=trading_signals,
            risk_metrics=risk_metrics,
            liquidity_metrics=data.get("liquidity_metrics", {}),
            generated_at=datetime.fromisoformat(data["generated_at"].replace("Z", "+00:00"))
        )