"""Swing trading setup detection."""

from typing import Dict, List, Any, Optional
import numpy as np


class SwingSetupDetector:
    """Detects swing trading setups in stock data."""
    
    def __init__(self):
        """Initialize the swing setup detector."""
        self.min_confidence = 0.6
        
    def detect_setup(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect swing trading setup from market data.
        
        Args:
            market_data: Dictionary of market indicators
            
        Returns:
            Setup details including type, entry, stop, target
        """
        # Check for bullish reversal
        if self._is_bullish_reversal(market_data):
            return self._create_bullish_setup(market_data)
        
        # Check for bearish reversal
        if self._is_bearish_reversal(market_data):
            return self._create_bearish_setup(market_data)
        
        # Default to no setup
        return {
            "type": "none",
            "confidence": 0,
        }
    
    def _is_bullish_reversal(self, data: Dict[str, float]) -> bool:
        """Check if conditions indicate bullish reversal.
        
        Args:
            data: Market data
            
        Returns:
            True if bullish reversal detected
        """
        # Price above short-term moving averages
        price_above_sma20 = data.get("price", 0) > data.get("sma_20", float('inf'))
        
        # Uptrending moving averages
        sma_uptrend = (
            data.get("sma_20", 0) > data.get("sma_50", 0) and
            data.get("sma_50", 0) > data.get("sma_200", 0)
        )
        
        # RSI oversold bounce
        rsi_bounce = 30 < data.get("rsi", 50) < 50
        
        # Volume confirmation
        volume_surge = data.get("volume_avg_ratio", 1) > 1.1
        
        return price_above_sma20 and (sma_uptrend or rsi_bounce) and volume_surge
    
    def _is_bearish_reversal(self, data: Dict[str, float]) -> bool:
        """Check if conditions indicate bearish reversal.
        
        Args:
            data: Market data
            
        Returns:
            True if bearish reversal detected
        """
        # Price below short-term moving averages
        price_below_sma20 = data.get("price", float('inf')) < data.get("sma_20", 0)
        
        # Downtrending moving averages
        sma_downtrend = (
            data.get("sma_20", float('inf')) < data.get("sma_50", float('inf')) and
            data.get("sma_50", float('inf')) < data.get("sma_200", float('inf'))
        )
        
        # RSI overbought reversal
        rsi_reversal = 50 < data.get("rsi", 50) < 70
        
        # Volume confirmation
        volume_surge = data.get("volume_avg_ratio", 1) > 1.1
        
        return price_below_sma20 and (sma_downtrend or rsi_reversal) and volume_surge
    
    def _create_bullish_setup(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Create bullish swing setup.
        
        Args:
            data: Market data
            
        Returns:
            Bullish setup details
        """
        price = data.get("price", 0)
        recent_low = data.get("recent_low", price * 0.97)
        recent_high = data.get("recent_high", price * 1.05)
        
        # Entry slightly above current price
        entry_price = price * 1.003  # 0.3% above
        
        # Stop below recent low
        stop_loss = recent_low * 0.995  # 0.5% below recent low
        
        # Targets based on risk/reward
        risk = entry_price - stop_loss
        target_1 = entry_price + (risk * 1.5)  # 1.5:1 R/R
        target_2 = entry_price + (risk * 2.5)  # 2.5:1 R/R
        
        # Ensure target_1 doesn't exceed recent high
        target_1 = min(target_1, recent_high)
        
        return {
            "type": "bullish_reversal",
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "confidence": self._calculate_confidence(data),
            "risk_reward": round(1.5, 1),
        }
    
    def _create_bearish_setup(self, data: Dict[str, float]) -> Dict[str, Any]:
        """Create bearish swing setup.
        
        Args:
            data: Market data
            
        Returns:
            Bearish setup details
        """
        price = data.get("price", 0)
        recent_high = data.get("recent_high", price * 1.03)
        recent_low = data.get("recent_low", price * 0.95)
        
        # Entry slightly below current price
        entry_price = price * 0.997  # 0.3% below
        
        # Stop above recent high
        stop_loss = recent_high * 1.005  # 0.5% above recent high
        
        # Targets based on risk/reward
        risk = stop_loss - entry_price
        target_1 = entry_price - (risk * 1.5)  # 1.5:1 R/R
        target_2 = entry_price - (risk * 2.5)  # 2.5:1 R/R
        
        # Ensure target_1 doesn't go below recent low
        target_1 = max(target_1, recent_low)
        
        return {
            "type": "bearish_reversal",
            "entry_price": round(entry_price, 2),
            "stop_loss": round(stop_loss, 2),
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "confidence": self._calculate_confidence(data),
            "risk_reward": round(1.5, 1),
        }
    
    def _calculate_confidence(self, data: Dict[str, float]) -> float:
        """Calculate setup confidence score.
        
        Args:
            data: Market data
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence
        
        # Add points for confirming factors
        if data.get("volume_avg_ratio", 1) > 1.2:
            confidence += 0.1
            
        if 40 < data.get("rsi", 50) < 60:
            confidence += 0.1
            
        # Moving average alignment
        if (data.get("sma_20", 0) > data.get("sma_50", 0) and
            data.get("sma_50", 0) > data.get("sma_200", 0)):
            confidence += 0.15
            
        return min(1.0, confidence)
    
    def detect_breakout(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect breakout trading setup.
        
        Args:
            market_data: Dictionary of market indicators
            
        Returns:
            Breakout setup details
        """
        price = market_data.get("price", 0)
        resistance = market_data.get("resistance", price * 1.02)
        support = market_data.get("support", price * 0.98)
        consolidation_days = market_data.get("consolidation_days", 0)
        volume_surge = market_data.get("volume_surge", 1)
        atr = market_data.get("atr", price * 0.02)
        
        # Check for resistance breakout
        if price > resistance * 0.995 and volume_surge > 1.5 and consolidation_days >= 5:
            return {
                "type": "resistance_breakout",
                "entry_price": round(price * 1.002, 2),
                "stop_loss": round(resistance - atr, 2),
                "target_1": round(resistance + (2 * atr), 2),
                "target_2": round(resistance + (4 * atr), 2),
                "confidence": min(0.9, 0.6 + (volume_surge - 1.5) * 0.2),
            }
        
        # Check for support breakdown
        if price < support * 1.005 and volume_surge > 1.5 and consolidation_days >= 5:
            return {
                "type": "support_breakdown",
                "entry_price": round(price * 0.998, 2),
                "stop_loss": round(support + atr, 2),
                "target_1": round(support - (2 * atr), 2),
                "target_2": round(support - (4 * atr), 2),
                "confidence": min(0.9, 0.6 + (volume_surge - 1.5) * 0.2),
            }
        
        return {
            "type": "no_breakout",
            "confidence": 0,
        }
    
    def detect_momentum_setup(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect momentum-based swing setup.
        
        Args:
            market_data: Dictionary of market indicators
            
        Returns:
            Momentum setup details
        """
        price = market_data.get("price", 0)
        sma_10 = market_data.get("sma_10", price)
        sma_20 = market_data.get("sma_20", price)
        sma_50 = market_data.get("sma_50", price)
        rsi = market_data.get("rsi", 50)
        volume_ratio = market_data.get("volume_avg_ratio", 1)
        price_change_5d = market_data.get("price_change_5d", 0)
        sector_strength = market_data.get("sector_strength", 0.5)
        
        # Strong momentum criteria
        strong_momentum = (
            price > sma_10 > sma_20 > sma_50 and
            55 < rsi < 70 and
            volume_ratio > 1.3 and
            price_change_5d > 0.05 and
            sector_strength > 0.6
        )
        
        if strong_momentum:
            # Trailing stop based on ATR or percentage
            trailing_stop_pct = 0.03  # 3% trailing stop
            
            return {
                "type": "momentum_long",
                "entry_price": round(price * 1.001, 2),
                "stop_loss": round(price * (1 - trailing_stop_pct), 2),
                "trailing_stop": trailing_stop_pct,
                "target_1": round(price * 1.05, 2),
                "target_2": round(price * 1.10, 2),
                "confidence": min(0.9, 0.6 + sector_strength * 0.3),
                "hold_days": "5-10",
            }
        
        return {
            "type": "no_momentum",
            "confidence": 0,
        }