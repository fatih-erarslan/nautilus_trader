"""
Risk Manager
Manages position sizing, portfolio risk, and trading limits
"""
from typing import Dict, Any, Optional
from dataclasses import replace
from datetime import datetime

from .models import (
    TradingSignal, SignalType, RiskLevel, 
    AssetType, PortfolioPosition, TradingStrategy
)


class RiskManager:
    """
    Comprehensive risk management system for trading decisions
    
    Handles position sizing, portfolio risk limits, stop loss validation,
    diversification rules, and maximum loss per trade calculations.
    """
    
    def __init__(self, 
                 max_position_size: float = 0.1,
                 max_portfolio_risk: float = 0.2,
                 max_correlation: float = 0.7,
                 max_loss_per_trade: float = 0.02):
        """
        Initialize risk manager with limits
        
        Args:
            max_position_size: Maximum size for any single position (0-1)
            max_portfolio_risk: Maximum total portfolio risk (0-1)
            max_correlation: Maximum correlation between positions (0-1)
            max_loss_per_trade: Maximum loss per trade as fraction of portfolio
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        self.max_loss_per_trade = max_loss_per_trade
        
        # Risk level position limits
        self.risk_level_limits = {
            RiskLevel.LOW: 0.10,
            RiskLevel.MEDIUM: 0.07,
            RiskLevel.HIGH: 0.05,
            RiskLevel.EXTREME: 0.02
        }
        
        # Diversification rules
        self.diversification_rules = {
            "max_per_asset_type": {
                AssetType.CRYPTO: 0.5,
                AssetType.EQUITY: 0.7,
                AssetType.BOND: 0.4,
                AssetType.COMMODITY: 0.3,
                AssetType.FOREX: 0.3
            },
            "min_positions": 1,
            "max_positions": 30
        }
    
    async def validate_signal(self, signal: TradingSignal, 
                            current_positions: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """
        Validate and adjust trading signal based on risk rules
        
        Args:
            signal: Trading signal to validate
            current_positions: Current portfolio positions
            
        Returns:
            Validated and potentially adjusted TradingSignal
        """
        # Start with a copy of the signal
        validated_signal = replace(signal)
        
        # 1. Validate position size against maximum
        validated_signal.position_size = min(
            validated_signal.position_size, 
            self.max_position_size
        )
        
        # 2. Apply risk level limits
        if signal.risk_level in self.risk_level_limits:
            risk_limit = self.risk_level_limits[signal.risk_level]
            validated_signal.position_size = min(
                validated_signal.position_size,
                risk_limit
            )
        
        # 3. Check portfolio risk if positions provided
        if current_positions:
            portfolio_risk = self.calculate_portfolio_risk(current_positions)
            signal_risk = self._signal_risk(validated_signal)
            
            # Adjust position size if portfolio risk would exceed limit
            if portfolio_risk + signal_risk > self.max_portfolio_risk:
                available_risk = max(0, self.max_portfolio_risk - portfolio_risk)
                if signal_risk > 0:
                    adjustment_factor = available_risk / signal_risk
                    validated_signal.position_size *= adjustment_factor
        
        # 4. Validate stop loss
        validated_signal = self._validate_stop_loss(validated_signal)
        
        # 5. Apply max loss per trade
        validated_signal = self.adjust_position_for_max_loss(validated_signal)
        
        return validated_signal
    
    def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate total portfolio risk
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            Total portfolio risk as fraction (0-1)
        """
        total_risk = 0.0
        
        for asset, position in positions.items():
            position_size = position.get("size", 0)
            position_risk = position.get("risk", 0.02)  # Default 2% risk
            total_risk += position_size * position_risk
            
        return total_risk
    
    def _signal_risk(self, signal: TradingSignal) -> float:
        """Calculate risk for a single signal"""
        # Calculate risk based on stop loss distance
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            return signal.position_size * stop_distance
        return 0.0
    
    def _validate_stop_loss(self, signal: TradingSignal) -> TradingSignal:
        """
        Validate and correct stop loss if necessary
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            Signal with validated stop loss
        """
        if signal.signal_type == SignalType.BUY:
            # For buy signals, stop loss must be below entry
            if signal.stop_loss >= signal.entry_price:
                # Set stop loss to 2% below entry as default
                signal = replace(signal, stop_loss=signal.entry_price * 0.98)
        
        elif signal.signal_type == SignalType.SELL:
            # For sell signals, stop loss must be above entry
            if signal.stop_loss <= signal.entry_price:
                # Set stop loss to 2% above entry as default
                signal = replace(signal, stop_loss=signal.entry_price * 1.02)
        
        return signal
    
    def adjust_position_for_max_loss(self, signal: TradingSignal) -> TradingSignal:
        """
        Adjust position size to limit maximum loss per trade
        
        Args:
            signal: Trading signal to adjust
            
        Returns:
            Signal with adjusted position size
        """
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            # Calculate potential loss percentage
            loss_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            
            # Calculate potential portfolio loss
            potential_loss = signal.position_size * loss_pct
            
            # Adjust if exceeds max loss
            if potential_loss > self.max_loss_per_trade:
                adjustment_factor = self.max_loss_per_trade / potential_loss
                signal = replace(
                    signal, 
                    position_size=signal.position_size * adjustment_factor
                )
        
        return signal
    
    def signal_to_position(self, signal: TradingSignal, 
                          current_price: float) -> PortfolioPosition:
        """
        Convert a trading signal to a portfolio position
        
        Args:
            signal: Trading signal
            current_price: Current market price
            
        Returns:
            PortfolioPosition object
        """
        # Calculate unrealized P&L
        if signal.signal_type == SignalType.BUY:
            unrealized_pnl = (current_price - signal.entry_price) / signal.entry_price
        else:  # SELL/SHORT
            unrealized_pnl = (signal.entry_price - current_price) / signal.entry_price
        
        return PortfolioPosition(
            asset=signal.asset,
            asset_type=signal.asset_type,
            size=signal.position_size,
            entry_price=signal.entry_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=signal.timestamp,
            strategy=signal.strategy
        )
    
    def set_risk_level_limits(self, limits: Dict[RiskLevel, float]) -> None:
        """
        Set position size limits based on risk level
        
        Args:
            limits: Dictionary mapping RiskLevel to max position size
        """
        self.risk_level_limits.update(limits)
    
    def set_diversification_rules(self, rules: Dict[str, Any]) -> None:
        """
        Set portfolio diversification rules
        
        Args:
            rules: Dictionary containing diversification parameters
        """
        self.diversification_rules.update(rules)
    
    def check_diversification(self, asset_type: AssetType, 
                            proposed_size: float,
                            current_positions: Dict[str, Any]) -> bool:
        """
        Check if adding position maintains diversification rules
        
        Args:
            asset_type: Type of asset to add
            proposed_size: Proposed position size
            current_positions: Current portfolio positions
            
        Returns:
            True if position can be added within diversification rules
        """
        # Calculate current allocation by asset type
        type_allocation = {t: 0.0 for t in AssetType}
        
        for asset, position in current_positions.items():
            pos_type = position.get("asset_type", AssetType.EQUITY)
            pos_size = position.get("size", 0)
            type_allocation[pos_type] += pos_size
        
        # Check if adding position would exceed limit
        max_allowed = self.diversification_rules["max_per_asset_type"].get(
            asset_type, 1.0
        )
        
        new_allocation = type_allocation[asset_type] + proposed_size
        
        return new_allocation <= max_allowed
    
    def get_risk_metrics(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for portfolio
        
        Args:
            positions: Current portfolio positions
            
        Returns:
            Dictionary of risk metrics
        """
        total_exposure = sum(p.get("size", 0) for p in positions.values())
        total_risk = self.calculate_portfolio_risk(positions)
        
        # Calculate risk by asset type
        risk_by_type = {t: 0.0 for t in AssetType}
        for asset, position in positions.items():
            pos_type = position.get("asset_type", AssetType.EQUITY)
            pos_risk = position.get("size", 0) * position.get("risk", 0.02)
            risk_by_type[pos_type] += pos_risk
        
        return {
            "total_exposure": total_exposure,
            "total_risk": total_risk,
            "risk_utilization": total_risk / self.max_portfolio_risk,
            "position_count": len(positions),
            "risk_by_type": risk_by_type
        }