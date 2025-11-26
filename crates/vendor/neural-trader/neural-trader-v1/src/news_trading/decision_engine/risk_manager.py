"""Risk Management component for Trading Decision Engine."""

from typing import Dict, Any, Optional
from .models import TradingSignal, RiskLevel, AssetType


class RiskManager:
    """Manages risk for trading decisions."""
    
    def __init__(self, max_position_size: float = 0.1, 
                 max_portfolio_risk: float = 0.2,
                 max_correlation: float = 0.7):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum size for any single position
            max_portfolio_risk: Maximum total portfolio risk
            max_correlation: Maximum correlation between positions
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlation = max_correlation
        
        # Risk parameters by asset type
        self.min_stop_loss = {
            AssetType.EQUITY: 0.02,  # 2% minimum
            AssetType.BOND: 0.01,    # 1% minimum
            AssetType.CRYPTO: 0.05,  # 5% minimum
            AssetType.COMMODITY: 0.03,
            AssetType.FOREX: 0.01
        }
        
    async def validate_signal(self, signal: TradingSignal, 
                            current_positions: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """
        Validate and adjust signal based on risk parameters.
        
        Args:
            signal: Trading signal to validate
            current_positions: Current portfolio positions
            
        Returns:
            Adjusted trading signal
        """
        # Validate position size
        signal.position_size = min(signal.position_size, self.max_position_size)
        
        # Check portfolio risk if positions provided
        if current_positions:
            portfolio_risk = self.calculate_portfolio_risk(current_positions)
            signal_risk = self._calculate_signal_risk(signal)
            
            # Adjust position size if portfolio risk too high
            if portfolio_risk + signal_risk >= self.max_portfolio_risk:
                available_risk = max(0, self.max_portfolio_risk - portfolio_risk)
                if signal_risk > 0 and available_risk < signal_risk:
                    adjustment_factor = available_risk / signal_risk
                    signal.position_size *= adjustment_factor
                    
        # Validate stop loss
        validated_stop = self.validate_stop_loss({
            "asset_type": signal.asset_type,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "atr": abs(signal.entry_price - signal.stop_loss) / 2  # Estimate
        })
        signal.stop_loss = validated_stop
        
        return signal
        
    def calculate_portfolio_risk(self, positions: Dict[str, Any]) -> float:
        """
        Calculate total portfolio risk.
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            Total portfolio risk as a fraction
        """
        total_risk = 0.0
        
        for asset, position in positions.items():
            position_risk = position["size"] * position.get("risk", 0.02)
            total_risk += position_risk
            
        return total_risk
        
    def adjust_for_correlation(self, proposed_size: float, 
                             asset: str,
                             correlation_data: Dict[str, float],
                             current_positions: Dict[str, Any]) -> float:
        """
        Adjust position size based on correlations.
        
        Args:
            proposed_size: Proposed position size
            asset: Asset symbol
            correlation_data: Correlation with other assets
            current_positions: Current positions
            
        Returns:
            Adjusted position size
        """
        max_correlation = 0.0
        
        # Find highest correlation with existing positions
        for position_asset, position_data in current_positions.items():
            if position_asset in correlation_data:
                correlation = abs(correlation_data[position_asset])
                max_correlation = max(max_correlation, correlation)
                
        # Reduce position size if correlation too high
        if max_correlation > self.max_correlation:
            reduction_factor = 1 - (max_correlation - self.max_correlation)
            proposed_size *= max(reduction_factor, 0.5)  # At least 50% reduction
            
        return proposed_size
        
    def validate_stop_loss(self, signal_data: Dict[str, Any]) -> float:
        """
        Validate stop loss is appropriate for asset type.
        
        Args:
            signal_data: Signal data with asset type and prices
            
        Returns:
            Validated stop loss price
        """
        asset_type = signal_data["asset_type"]
        entry_price = signal_data["entry_price"]
        stop_loss = signal_data["stop_loss"]
        
        # Calculate stop percentage
        stop_pct = abs(entry_price - stop_loss) / entry_price
        
        # Get minimum stop for asset type
        min_stop_pct = self.min_stop_loss.get(asset_type, 0.02)
        
        # Adjust if stop is too tight
        if stop_pct < min_stop_pct:
            if stop_loss < entry_price:  # Long position
                stop_loss = entry_price * (1 - min_stop_pct)
            else:  # Short position
                stop_loss = entry_price * (1 + min_stop_pct)
                
        return stop_loss
        
    def score_signal_risk(self, risk_level: RiskLevel, 
                         volatility: float,
                         position_size: float) -> float:
        """
        Score the risk of a signal.
        
        Args:
            risk_level: Signal risk level
            volatility: Asset volatility
            position_size: Position size
            
        Returns:
            Risk score between 0 and 1
        """
        # Base score from risk level
        risk_scores = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.EXTREME: 1.0
        }
        
        base_score = risk_scores.get(risk_level, 0.5)
        
        # Adjust for volatility
        volatility_factor = min(volatility * 10, 1.0)  # Cap at 1.0
        
        # Adjust for position size
        size_factor = position_size * 5  # 20% position = 1.0
        
        # Combined score
        risk_score = (base_score + volatility_factor + size_factor) / 3
        
        return min(risk_score, 1.0)
        
    def _calculate_signal_risk(self, signal: TradingSignal) -> float:
        """
        Calculate risk for a single signal.
        
        Args:
            signal: Trading signal
            
        Returns:
            Risk as a fraction of portfolio
        """
        # Risk is position size * stop loss percentage
        stop_loss_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        return signal.position_size * stop_loss_pct