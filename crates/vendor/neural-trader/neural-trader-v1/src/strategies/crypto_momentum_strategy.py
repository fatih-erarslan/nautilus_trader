"""
Cryptocurrency Momentum Trading Strategy with Fee Optimization
Integrates with Neural Trader system for enhanced predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market volatility regimes"""
    LOW_VOL = "low_volatility"
    MEDIUM_VOL = "medium_volatility"
    HIGH_VOL = "high_volatility"
    EXTREME_VOL = "extreme_volatility"


class SignalStrength(Enum):
    """Trading signal strength levels"""
    WEAK = 0.5
    MODERATE = 0.7
    STRONG = 0.85
    VERY_STRONG = 0.95


@dataclass
class CryptoSignal:
    """Trading signal for crypto momentum strategy"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    predicted_move: float
    confidence: float
    signal_strength: SignalStrength
    stop_loss: float
    take_profit: float
    position_size: float
    fee_efficiency_ratio: float
    expected_holding_hours: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeeStructure:
    """Exchange fee configuration"""
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    has_fee_token: bool = False
    fee_token_discount: float = 0.25  # 25% discount with token
    vip_tier: int = 0
    
    @property
    def effective_maker_fee(self) -> float:
        fee = self.maker_fee
        if self.has_fee_token:
            fee *= (1 - self.fee_token_discount)
        return fee
    
    @property
    def effective_taker_fee(self) -> float:
        fee = self.taker_fee
        if self.has_fee_token:
            fee *= (1 - self.fee_token_discount)
        return fee
    
    @property
    def round_trip_fee(self) -> float:
        """Calculate total fee for entering and exiting position"""
        return self.effective_taker_fee * 2  # Assume market orders


class CryptoMomentumStrategy:
    """
    Fee-optimized momentum strategy for cryptocurrency trading
    Only trades on moves large enough to overcome fee burden
    """
    
    def __init__(
        self,
        min_move_threshold: float = 0.015,  # 1.5% minimum
        confidence_threshold: float = 0.75,
        fee_structure: Optional[FeeStructure] = None,
        max_position_pct: float = 0.1,  # 10% of portfolio
        use_pyramiding: bool = True,
        neural_weight: float = 0.4  # Weight for neural predictions
    ):
        self.min_move_threshold = min_move_threshold
        self.confidence_threshold = confidence_threshold
        self.fee_structure = fee_structure or FeeStructure()
        self.max_position_pct = max_position_pct
        self.use_pyramiding = use_pyramiding
        self.neural_weight = neural_weight
        
        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_gross_profit = 0.0
        self.trades_executed = 0
        self.winning_trades = 0
        
        # Current positions
        self.positions: Dict[str, CryptoSignal] = {}
        
    def calculate_fee_efficiency(
        self,
        expected_move: float,
        position_size: float,
        entry_price: float
    ) -> float:
        """Calculate profit to fee ratio"""
        expected_profit = position_size * entry_price * expected_move
        round_trip_fees = position_size * entry_price * self.fee_structure.round_trip_fee
        
        if round_trip_fees == 0:
            return float('inf')
        
        return expected_profit / round_trip_fees
    
    def detect_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Determine current market volatility regime"""
        # Calculate ATR (Average True Range)
        high = price_data['high'].iloc[-20:]
        low = price_data['low'].iloc[-20:]
        close = price_data['close'].iloc[-20:]
        
        tr = pd.DataFrame()
        tr['hl'] = high - low
        tr['hc'] = abs(high - close.shift(1))
        tr['lc'] = abs(low - close.shift(1))
        
        atr = tr.max(axis=1).mean()
        atr_pct = atr / close.iloc[-1]
        
        # Classify regime based on ATR percentage
        if atr_pct < 0.01:
            return MarketRegime.LOW_VOL
        elif atr_pct < 0.025:
            return MarketRegime.MEDIUM_VOL
        elif atr_pct < 0.05:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.EXTREME_VOL
    
    def calculate_momentum_score(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate technical momentum indicators"""
        close = price_data['close']
        
        # Price momentum (rate of change)
        roc_5 = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) > 5 else 0
        roc_10 = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) > 10 else 0
        roc_20 = (close.iloc[-1] / close.iloc[-20] - 1) if len(close) > 20 else 0
        
        # RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(close) > 14 else 50
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = (macd - signal).iloc[-1] if len(close) > 26 else 0
        
        # Volume confirmation (if available)
        volume_score = 1.0
        if volume_data is not None and len(volume_data) > 20:
            vol_ma = volume_data['volume'].rolling(20).mean()
            current_vol = volume_data['volume'].iloc[-1]
            volume_score = min(current_vol / vol_ma.iloc[-1], 2.0) if vol_ma.iloc[-1] > 0 else 1.0
        
        # Combine indicators
        momentum_components = {
            'roc_5': roc_5,
            'roc_10': roc_10,
            'roc_20': roc_20,
            'rsi': rsi,
            'macd_histogram': macd_histogram,
            'volume_score': volume_score
        }
        
        # Calculate weighted momentum score
        if rsi > 70:  # Overbought
            rsi_weight = (rsi - 70) / 30
            direction = 1
        elif rsi < 30:  # Oversold
            rsi_weight = (30 - rsi) / 30
            direction = -1
        else:
            rsi_weight = 0
            direction = 1 if macd_histogram > 0 else -1
        
        # Normalize components and calculate score
        momentum_score = (
            min(abs(roc_5), 0.1) * 0.2 +
            min(abs(roc_10), 0.1) * 0.3 +
            min(abs(roc_20), 0.1) * 0.2 +
            rsi_weight * 0.2 +
            min(abs(macd_histogram), 0.01) * 10 * 0.1
        ) * volume_score * direction
        
        # Ensure momentum score is normalized between -0.1 and 0.1
        momentum_score = max(-0.1, min(0.1, momentum_score))
        
        return momentum_score, momentum_components
    
    def integrate_neural_prediction(
        self,
        base_signal: float,
        neural_forecast: Dict[str, float]
    ) -> float:
        """Combine technical momentum with neural predictions"""
        neural_signal = neural_forecast.get('predicted_return', 0)
        neural_confidence = neural_forecast.get('confidence', 0.5)
        
        # Weight neural prediction based on confidence
        weighted_neural = neural_signal * neural_confidence * self.neural_weight
        weighted_technical = base_signal * (1 - self.neural_weight)
        
        return weighted_neural + weighted_technical
    
    def calculate_position_size(
        self,
        signal_strength: float,
        expected_move: float,
        portfolio_value: float,
        current_positions: int
    ) -> float:
        """Calculate position size with Kelly Criterion and fee consideration"""
        # Base Kelly fraction
        win_prob = min(0.5 + signal_strength * 0.4, 0.9)  # Convert signal to probability
        loss_prob = 1 - win_prob
        
        win_amount = expected_move - self.fee_structure.round_trip_fee
        loss_amount = expected_move * 0.5 + self.fee_structure.round_trip_fee  # Assume 50% stop loss
        
        if loss_amount <= 0:
            return 0
        
        # Kelly formula
        kelly_fraction = (win_prob * win_amount - loss_prob * loss_amount) / loss_amount
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% for safety
        
        # Adjust for number of positions
        position_adjustment = 1 / (1 + current_positions * 0.2)  # Reduce size with more positions
        
        # Calculate final position size
        position_size = portfolio_value * self.max_position_pct * kelly_fraction * position_adjustment
        
        return position_size
    
    def generate_signal(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        neural_forecast: Optional[Dict[str, float]] = None,
        portfolio_value: float = 100000
    ) -> Optional[CryptoSignal]:
        """Generate trading signal if conditions are met"""
        
        # Check minimum data requirements
        if len(price_data) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(price_data)} candles")
            return None
        
        # Get current price
        current_price = price_data['close'].iloc[-1]
        
        # Calculate momentum score
        momentum_score, components = self.calculate_momentum_score(price_data, volume_data)
        
        # Integrate neural predictions if available
        if neural_forecast:
            momentum_score = self.integrate_neural_prediction(momentum_score, neural_forecast)
            expected_move = neural_forecast.get('predicted_return', abs(momentum_score))
            neural_confidence = neural_forecast.get('confidence', 0.5)
        else:
            expected_move = abs(momentum_score)
            neural_confidence = 0.5
        
        # Check if move is large enough to overcome fees
        if abs(expected_move) < self.min_move_threshold:
            return None
        
        # Calculate combined confidence
        technical_confidence = min(abs(momentum_score) / 0.05, 1.0)  # Normalize to 0-1
        combined_confidence = (technical_confidence + neural_confidence) / 2
        
        if combined_confidence < self.confidence_threshold:
            return None
        
        # Determine signal strength
        if combined_confidence >= 0.95:
            signal_strength = SignalStrength.VERY_STRONG
        elif combined_confidence >= 0.85:
            signal_strength = SignalStrength.STRONG
        elif combined_confidence >= 0.7:
            signal_strength = SignalStrength.MODERATE
        else:
            signal_strength = SignalStrength.WEAK
        
        # Calculate position size
        position_size = self.calculate_position_size(
            combined_confidence,
            abs(expected_move),
            portfolio_value,
            len(self.positions)
        )
        
        # Check fee efficiency
        fee_efficiency = self.calculate_fee_efficiency(
            abs(expected_move),
            position_size / current_price,
            current_price
        )
        
        if fee_efficiency < 7:  # Require 7x profit to fee ratio
            logger.info(f"Signal rejected for {symbol}: Fee efficiency {fee_efficiency:.2f} < 7")
            return None
        
        # Determine direction
        direction = "long" if momentum_score > 0 else "short"
        
        # Calculate stop loss and take profit
        if direction == "long":
            stop_loss = current_price * (1 - abs(expected_move) * 0.5)
            take_profit = current_price * (1 + abs(expected_move) * 0.8)
        else:
            stop_loss = current_price * (1 + abs(expected_move) * 0.5)
            take_profit = current_price * (1 - abs(expected_move) * 0.8)
        
        # Estimate holding period based on timeframe
        expected_holding_hours = 4 if signal_strength.value >= 0.85 else 8
        
        # Create signal
        signal = CryptoSignal(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            predicted_move=expected_move,
            confidence=combined_confidence,
            signal_strength=signal_strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            fee_efficiency_ratio=fee_efficiency,
            expected_holding_hours=expected_holding_hours,
            metadata={
                'momentum_components': components,
                'neural_confidence': neural_confidence,
                'technical_confidence': technical_confidence,
                'market_regime': self.detect_market_regime(price_data).value
            }
        )
        
        logger.info(
            f"Signal generated for {symbol}: {direction.upper()} "
            f"Expected: {expected_move:.2%} Confidence: {combined_confidence:.2%} "
            f"Fee Efficiency: {fee_efficiency:.1f}x"
        )
        
        return signal
    
    def should_pyramid(
        self,
        symbol: str,
        current_price: float,
        position: CryptoSignal
    ) -> bool:
        """Determine if we should add to winning position"""
        if not self.use_pyramiding:
            return False
        
        if symbol not in self.positions:
            return False
        
        # Calculate current profit
        if position.direction == "long":
            profit_pct = (current_price - position.entry_price) / position.entry_price
        else:
            profit_pct = (position.entry_price - current_price) / position.entry_price
        
        # Pyramid if profit > 1% and we haven't pyramided recently
        if profit_pct > 0.01 and position.metadata.get('pyramid_count', 0) < 3:
            # Check if adding won't create fee burden
            additional_fees = current_price * 0.5 * position.position_size * self.fee_structure.round_trip_fee
            remaining_expected_move = position.predicted_move - profit_pct
            
            if remaining_expected_move > self.fee_structure.round_trip_fee * 3:
                return True
        
        return False
    
    def update_position_tracking(self, signal: CryptoSignal, executed_price: float):
        """Track position and update performance metrics"""
        # Add to positions
        self.positions[signal.symbol] = signal
        
        # Update fee tracking
        entry_fee = signal.position_size * self.fee_structure.effective_taker_fee
        self.total_fees_paid += entry_fee
        self.trades_executed += 1
        
        logger.info(
            f"Position opened for {signal.symbol}: "
            f"Size: ${signal.position_size:.2f} "
            f"Entry: ${executed_price:.2f} "
            f"Fee: ${entry_fee:.2f}"
        )
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "target_reached"
    ) -> Dict[str, float]:
        """Close position and calculate P&L"""
        if symbol not in self.positions:
            return {"error": "Position not found"}
        
        position = self.positions[symbol]
        
        # Calculate P&L
        if position.direction == "long":
            gross_pnl = (exit_price - position.entry_price) * (position.position_size / position.entry_price)
        else:
            gross_pnl = (position.entry_price - exit_price) * (position.position_size / position.entry_price)
        
        # Calculate fees
        exit_fee = position.position_size * self.fee_structure.effective_taker_fee
        total_fees = exit_fee + position.position_size * self.fee_structure.effective_taker_fee
        
        # Net P&L
        net_pnl = gross_pnl - total_fees
        
        # Update tracking
        self.total_gross_profit += gross_pnl
        self.total_fees_paid += exit_fee
        if net_pnl > 0:
            self.winning_trades += 1
        
        # Remove position
        del self.positions[symbol]
        
        result = {
            "symbol": symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "gross_pnl": gross_pnl,
            "fees": total_fees,
            "net_pnl": net_pnl,
            "return_pct": net_pnl / position.position_size,
            "reason": reason
        }
        
        logger.info(
            f"Position closed for {symbol}: "
            f"Net P&L: ${net_pnl:.2f} ({result['return_pct']:.2%}) "
            f"Fees: ${total_fees:.2f} "
            f"Reason: {reason}"
        )
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate strategy performance metrics"""
        if self.trades_executed == 0:
            return {
                "status": "No trades executed",
                "trades": 0
            }
        
        win_rate = self.winning_trades / self.trades_executed if self.trades_executed > 0 else 0
        fee_efficiency = self.total_gross_profit / self.total_fees_paid if self.total_fees_paid > 0 else 0
        net_profit = self.total_gross_profit - self.total_fees_paid
        
        return {
            "total_trades": self.trades_executed,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "gross_profit": self.total_gross_profit,
            "total_fees": self.total_fees_paid,
            "net_profit": net_profit,
            "fee_efficiency_ratio": fee_efficiency,
            "avg_fee_per_trade": self.total_fees_paid / self.trades_executed,
            "fee_drag_pct": self.total_fees_paid / abs(self.total_gross_profit) if self.total_gross_profit != 0 else 0,
            "active_positions": len(self.positions),
            "positions": {k: {
                "direction": v.direction,
                "entry": v.entry_price,
                "size": v.position_size,
                "expected_move": v.predicted_move
            } for k, v in self.positions.items()}
        }