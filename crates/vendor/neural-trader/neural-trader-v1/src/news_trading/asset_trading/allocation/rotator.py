"""Asset rotation strategies for multi-asset portfolios."""

from typing import Dict, List, Optional


class AssetRotator:
    """Manages dynamic asset allocation and rotation."""
    
    def __init__(self):
        """Initialize the asset rotator."""
        self.risk_regimes = {
            "risk_on": {"stocks": 0.7, "bonds": 0.2, "cash": 0.1},
            "neutral": {"stocks": 0.6, "bonds": 0.3, "cash": 0.1},
            "risk_off": {"stocks": 0.3, "bonds": 0.5, "cash": 0.2},
            "crisis": {"stocks": 0.1, "bonds": 0.6, "cash": 0.3},
        }
        
    def calculate_allocation(self, market_conditions: Dict) -> Dict[str, float]:
        """Calculate optimal asset allocation based on market conditions.
        
        Args:
            market_conditions: Current market indicators
            
        Returns:
            Asset allocation percentages
        """
        # Determine market regime
        regime = self._determine_regime(market_conditions)
        
        # Get base allocation
        base_allocation = self.risk_regimes[regime].copy()
        
        # Apply tactical adjustments
        allocation = self._apply_tactical_overlay(base_allocation, market_conditions)
        
        # Ensure allocations sum to 1
        total = sum(allocation.values())
        if total > 0:
            for asset in allocation:
                allocation[asset] /= total
        
        return allocation
    
    def _determine_regime(self, conditions: Dict) -> str:
        """Determine current market regime.
        
        Args:
            conditions: Market conditions
            
        Returns:
            Regime type
        """
        vix = conditions.get("vix", 20)
        spy_trend = conditions.get("spy_trend", "neutral")
        yield_curve = conditions.get("yield_curve", "normal")
        economic_data = conditions.get("economic_data", "stable")
        credit_spreads = conditions.get("credit_spreads", "normal")
        
        # Score risk level
        risk_score = 0
        
        # VIX level
        if vix > 35:
            risk_score += 3
        elif vix > 25:
            risk_score += 2
        elif vix > 20:
            risk_score += 1
        
        # Market trend
        if spy_trend == "declining":
            risk_score += 2
        elif spy_trend == "weak":
            risk_score += 1
        
        # Yield curve
        if yield_curve == "inverting":
            risk_score += 2
        elif yield_curve == "flat":
            risk_score += 1
        
        # Economic data
        if economic_data == "weakening":
            risk_score += 2
        elif economic_data == "mixed":
            risk_score += 1
        
        # Credit spreads
        if credit_spreads == "widening":
            risk_score += 2
        elif credit_spreads == "elevated":
            risk_score += 1
        
        # Map score to regime
        if risk_score >= 8:
            return "crisis"
        elif risk_score >= 5:
            return "risk_off"
        elif risk_score >= 2:
            return "neutral"
        else:
            return "risk_on"
    
    def _apply_tactical_overlay(
        self,
        base_allocation: Dict[str, float],
        conditions: Dict
    ) -> Dict[str, float]:
        """Apply tactical adjustments to base allocation.
        
        Args:
            base_allocation: Base allocation
            conditions: Market conditions
            
        Returns:
            Adjusted allocation
        """
        allocation = base_allocation.copy()
        
        # Momentum adjustment
        if conditions.get("stock_momentum", 0) > 0.5:
            # Increase stocks if momentum strong
            allocation["stocks"] = min(0.8, allocation["stocks"] * 1.1)
            allocation["bonds"] *= 0.9
        
        # Valuation adjustment
        if conditions.get("stock_valuation", 0) < -0.5:
            # Reduce stocks if expensive
            allocation["stocks"] *= 0.9
            allocation["bonds"] *= 1.1
        
        # Dollar strength adjustment
        if conditions.get("dollar_strength", "neutral") == "strong":
            # Strong dollar favors domestic assets
            allocation["cash"] *= 1.1
        
        return allocation
    
    def correlation_adjustment(self, correlation_data: Dict) -> Dict[str, float]:
        """Adjust allocation based on asset correlations.
        
        Args:
            correlation_data: Correlation information
            
        Returns:
            Allocation adjustments
        """
        stock_bond_corr = correlation_data.get("stock_bond_corr", -0.2)
        rolling_window = correlation_data.get("rolling_window", 60)
        historical_avg = correlation_data.get("historical_avg", -0.2)
        
        adjustments = {
            "diversification_boost": 0,
            "alternative_allocation": 0,
        }
        
        # When correlation is high, need more diversification
        if stock_bond_corr > 0.5:
            adjustments["diversification_boost"] = 0.2
            adjustments["alternative_allocation"] = 0.1  # Add alternatives
        elif stock_bond_corr > historical_avg + 0.3:
            adjustments["diversification_boost"] = 0.1
            adjustments["alternative_allocation"] = 0.05
        
        return adjustments
    
    def tactical_allocation(
        self,
        signals: Dict[str, float],
        base_stock: float = 0.6,
        base_bond: float = 0.4
    ) -> Dict[str, float]:
        """Calculate tactical asset allocation based on signals.
        
        Args:
            signals: Market signals
            base_stock: Base stock allocation
            base_bond: Base bond allocation
            
        Returns:
            Tactical allocation
        """
        stock_score = 0
        bond_score = 0
        
        # Momentum signals
        stock_score += signals.get("stock_momentum", 0) * 0.3
        bond_score += signals.get("bond_momentum", 0) * 0.3
        
        # Valuation signals (inverse)
        stock_score -= signals.get("stock_valuation", 0) * 0.2
        bond_score -= signals.get("bond_valuation", 0) * 0.2
        
        # Macro signals
        macro = signals.get("macro_score", 0)
        stock_score += macro * 0.2
        bond_score -= macro * 0.1  # Bonds inverse to macro
        
        # Apply tilts (max 10% each way)
        stock_tilt = max(-0.1, min(0.1, stock_score * 0.1))
        bond_tilt = max(-0.1, min(0.1, bond_score * 0.1))
        
        # Ensure they offset
        if stock_tilt + bond_tilt > 0:
            bond_tilt = -stock_tilt
        else:
            stock_tilt = -bond_tilt
        
        return {
            "stocks": base_stock + stock_tilt,
            "bonds": base_bond + bond_tilt,
        }
    
    def detect_regime(self, data: Dict) -> str:
        """Detect market regime for rotation.
        
        Args:
            data: Market data
            
        Returns:
            Detected regime
        """
        volatility = data.get("volatility", 15)
        trend = data.get("trend", "neutral")
        correlation = data.get("correlation", 0)
        
        if volatility < 15 and trend == "up":
            return "risk_on"
        elif volatility > 25 and trend == "down":
            return "risk_off"
        else:
            return "neutral"
    
    def get_rotation_signals(
        self,
        asset_performance: Dict[str, float],
        lookback_days: int = 60
    ) -> List[Dict]:
        """Generate rotation signals based on relative performance.
        
        Args:
            asset_performance: Recent performance by asset
            lookback_days: Performance lookback period
            
        Returns:
            List of rotation signals
        """
        signals = []
        
        # Find best and worst performers
        sorted_assets = sorted(
            asset_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if len(sorted_assets) >= 2:
            best = sorted_assets[0]
            worst = sorted_assets[-1]
            
            # Momentum rotation signal
            if best[1] - worst[1] > 0.1:  # 10% performance gap
                signals.append({
                    "type": "momentum_rotation",
                    "from": worst[0],
                    "to": best[0],
                    "strength": min(1.0, (best[1] - worst[1]) * 5),
                    "timeframe": f"{lookback_days} days",
                })
        
        return signals
    
    def calculate_rebalance_threshold(
        self,
        volatility: float,
        correlation: float
    ) -> float:
        """Calculate dynamic rebalancing threshold.
        
        Args:
            volatility: Portfolio volatility
            correlation: Asset correlation
            
        Returns:
            Rebalancing threshold percentage
        """
        # Base threshold
        threshold = 0.05  # 5%
        
        # Adjust for volatility
        if volatility > 20:
            threshold = 0.10  # Wider bands in high vol
        elif volatility < 10:
            threshold = 0.03  # Tighter bands in low vol
        
        # Adjust for correlation
        if abs(correlation) > 0.7:
            threshold *= 1.2  # Wider bands when highly correlated
        
        return threshold