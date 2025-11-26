"""Market Regime Adaptive Momentum Parameters - Disaster Recovery System"""

from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np
from .momentum_trader import MomentumEngine


class MomentumRegimeAdapter:
    """
    Adapts momentum parameters based on market regime detection
    Uses optimized parameters from genetic algorithm disaster recovery
    """
    
    def __init__(self):
        """Initialize with optimized regime-specific parameters"""
        
        # OPTIMIZED REGIME-SPECIFIC PARAMETERS
        # Genetic algorithm optimization results for each market condition
        self.regime_parameters = {
            'bull_market': {
                'momentum_thresholds': {
                    'strong': 0.484,
                    'moderate': 0.267, 
                    'weak': 0.173
                },
                'max_position_pct': 0.261,  # Aggressive in bull markets
                'min_position_pct': 0.024,
                'lookback_periods': [6, 15, 81],
                'stop_loss_pct': 0.022,
                'expected_sharpe': 0.967,
                'expected_return': 1.0
            },
            
            'bear_market': {
                'momentum_thresholds': {
                    'strong': 0.563,
                    'moderate': 0.308,
                    'weak': 0.175  
                },
                'max_position_pct': 0.145,  # Conservative in bear markets
                'min_position_pct': 0.028,
                'lookback_periods': [4, 16, 81],
                'stop_loss_pct': 0.058,     # Wider stops in volatile conditions
                'expected_sharpe': 0.264,
                'expected_return': 0.690
            },
            
            'sideways_market': {
                'momentum_thresholds': {
                    'strong': 0.459,
                    'moderate': 0.250,
                    'weak': 0.05    # Very low threshold for sideways
                },
                'max_position_pct': 0.15,   # Moderate sizing
                'min_position_pct': 0.026,
                'lookback_periods': [4, 14, 76],
                'stop_loss_pct': 0.051,
                'expected_sharpe': 0.557,
                'expected_return': 0.885
            },
            
            'high_volatility': {
                'momentum_thresholds': {
                    'strong': 0.40,
                    'moderate': 0.35,
                    'weak': 0.12
                },
                'max_position_pct': 0.20,
                'min_position_pct': 0.023,
                'lookback_periods': [3, 11, 33],
                'stop_loss_pct': 0.132,     # Much wider stops
                'expected_sharpe': 0.492,
                'expected_return': 0.842
            }
        }
        
        # Default to high volatility parameters (most conservative)
        self.current_regime = 'high_volatility'
        self.regime_confidence = 0.5
        
    def detect_market_regime(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Detect current market regime based on market indicators
        
        Args:
            market_data: Dictionary with market indicators
            
        Returns:
            Tuple of (regime_name, confidence_score)
        """
        
        # Market indicators for regime detection
        spy_return_1m = market_data.get('spy_return_1m', 0)
        spy_return_3m = market_data.get('spy_return_3m', 0)
        spy_return_6m = market_data.get('spy_return_6m', 0)
        vix_level = market_data.get('vix_level', 20)
        vix_change = market_data.get('vix_change_5d', 0)
        market_breadth = market_data.get('market_breadth', 0.5)  # % stocks above 50MA
        bond_yield_change = market_data.get('bond_yield_change_1m', 0)
        
        regime_scores = {
            'bull_market': 0,
            'bear_market': 0, 
            'sideways_market': 0,
            'high_volatility': 0
        }
        
        # Bull market indicators
        if spy_return_1m > 0.03 and spy_return_3m > 0.08:  # Strong positive returns
            regime_scores['bull_market'] += 3
        if spy_return_6m > 0.15:  # Strong 6-month performance
            regime_scores['bull_market'] += 2
        if vix_level < 18:  # Low volatility
            regime_scores['bull_market'] += 2
        if market_breadth > 0.65:  # Strong breadth
            regime_scores['bull_market'] += 2
        if bond_yield_change > 0.5:  # Rising yields (risk-on)
            regime_scores['bull_market'] += 1
            
        # Bear market indicators  
        if spy_return_1m < -0.05 and spy_return_3m < -0.10:  # Strong negative returns
            regime_scores['bear_market'] += 3
        if spy_return_6m < -0.15:  # Poor 6-month performance
            regime_scores['bear_market'] += 2
        if vix_level > 30:  # High fear
            regime_scores['bear_market'] += 2
        if market_breadth < 0.35:  # Weak breadth
            regime_scores['bear_market'] += 2
        if bond_yield_change < -0.5:  # Falling yields (risk-off)
            regime_scores['bear_market'] += 1
            
        # Sideways market indicators
        if abs(spy_return_3m) < 0.05:  # Low 3-month return
            regime_scores['sideways_market'] += 2
        if abs(spy_return_6m) < 0.10:  # Low 6-month return  
            regime_scores['sideways_market'] += 2
        if 18 <= vix_level <= 25:  # Moderate volatility
            regime_scores['sideways_market'] += 2
        if 0.45 <= market_breadth <= 0.60:  # Mixed breadth
            regime_scores['sideways_market'] += 1
        if abs(bond_yield_change) < 0.3:  # Stable yields
            regime_scores['sideways_market'] += 1
            
        # High volatility indicators
        if vix_level > 25:  # Elevated volatility
            regime_scores['high_volatility'] += 3
        if abs(vix_change) > 3:  # Large VIX moves
            regime_scores['high_volatility'] += 2
        if abs(spy_return_1m) > 0.08:  # Large monthly moves
            regime_scores['high_volatility'] += 2
        if market_breadth < 0.3 or market_breadth > 0.8:  # Extreme breadth
            regime_scores['high_volatility'] += 1
            
        # Determine regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[best_regime]
        
        # Calculate confidence (0-1 scale)
        total_possible_score = 10  # Maximum possible score
        confidence = min(max_score / total_possible_score, 1.0)
        
        # Require minimum confidence to change regime
        if confidence < 0.6 and best_regime != self.current_regime:
            # Stay with current regime if confidence is low
            return self.current_regime, self.regime_confidence
        
        self.current_regime = best_regime
        self.regime_confidence = confidence
        
        return best_regime, confidence
    
    def get_adaptive_parameters(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get adaptive parameters based on current market regime
        
        Args:
            market_data: Current market data for regime detection
            
        Returns:
            Optimized parameters for current market regime
        """
        
        # Detect current market regime
        regime, confidence = self.detect_market_regime(market_data)
        
        # Get base parameters for detected regime
        base_params = self.regime_parameters[regime].copy()
        
        # If confidence is low, blend with default parameters
        if confidence < 0.8:
            default_params = self.regime_parameters['high_volatility']
            blend_factor = confidence  # Use confidence as blend factor
            
            # Blend momentum thresholds
            for threshold_type in ['strong', 'moderate', 'weak']:
                base_value = base_params['momentum_thresholds'][threshold_type]
                default_value = default_params['momentum_thresholds'][threshold_type]
                blended_value = (base_value * blend_factor + 
                               default_value * (1 - blend_factor))
                base_params['momentum_thresholds'][threshold_type] = blended_value
            
            # Blend position sizing
            base_params['max_position_pct'] = (
                base_params['max_position_pct'] * blend_factor +
                default_params['max_position_pct'] * (1 - blend_factor)
            )
            
            base_params['stop_loss_pct'] = (
                base_params['stop_loss_pct'] * blend_factor +
                default_params['stop_loss_pct'] * (1 - blend_factor)
            )
        
        # Add regime metadata
        base_params['detected_regime'] = regime
        base_params['regime_confidence'] = confidence
        base_params['adaptation_timestamp'] = datetime.now().isoformat()
        
        return base_params
    
    def create_adaptive_momentum_engine(self, market_data: Dict[str, Any], 
                                      portfolio_size: float = 100000) -> MomentumEngine:
        """
        Create a momentum engine with adaptive parameters
        
        Args:
            market_data: Current market data
            portfolio_size: Portfolio size for position sizing
            
        Returns:
            MomentumEngine configured with optimal parameters for current regime
        """
        
        # Get adaptive parameters
        adaptive_params = self.get_adaptive_parameters(market_data)
        
        # Create momentum engine
        lookback_periods = adaptive_params['lookback_periods']
        engine = MomentumEngine(lookback_periods, portfolio_size)
        
        # Update with adaptive parameters
        engine.momentum_thresholds = adaptive_params['momentum_thresholds']
        engine.max_position_pct = adaptive_params['max_position_pct']
        engine.min_position_pct = adaptive_params['min_position_pct']
        
        return engine
    
    def get_regime_performance_metrics(self, regime: str = None) -> Dict[str, float]:
        """
        Get expected performance metrics for a specific regime
        
        Args:
            regime: Market regime name (defaults to current regime)
            
        Returns:
            Expected performance metrics
        """
        
        target_regime = regime or self.current_regime
        
        if target_regime not in self.regime_parameters:
            target_regime = 'high_volatility'  # Default fallback
            
        params = self.regime_parameters[target_regime]
        
        return {
            'expected_sharpe_ratio': params['expected_sharpe'],
            'expected_annual_return': params['expected_return'],
            'regime': target_regime,
            'confidence': self.regime_confidence if target_regime == self.current_regime else 0.0
        }
    
    def log_regime_change(self, old_regime: str, new_regime: str, confidence: float):
        """Log regime change for monitoring"""
        
        print(f"MARKET REGIME CHANGE DETECTED:")
        print(f"  From: {old_regime}")
        print(f"  To: {new_regime}")
        print(f"  Confidence: {confidence:.1%}")
        
        old_params = self.regime_parameters[old_regime]
        new_params = self.regime_parameters[new_regime]
        
        print(f"  Parameter Changes:")
        print(f"    Max Position: {old_params['max_position_pct']:.1%} → {new_params['max_position_pct']:.1%}")
        print(f"    Strong Threshold: {old_params['momentum_thresholds']['strong']:.3f} → {new_params['momentum_thresholds']['strong']:.3f}")
        print(f"    Expected Sharpe: {old_params['expected_sharpe']:.3f} → {new_params['expected_sharpe']:.3f}")


def create_disaster_recovery_momentum_engine(market_data: Dict[str, Any], 
                                           portfolio_size: float = 100000) -> Tuple[MomentumEngine, Dict[str, Any]]:
    """
    Create a momentum engine with disaster recovery parameters
    
    Args:
        market_data: Current market conditions
        portfolio_size: Portfolio size
        
    Returns:
        Tuple of (MomentumEngine, regime_info)
    """
    
    adapter = MomentumRegimeAdapter()
    engine = adapter.create_adaptive_momentum_engine(market_data, portfolio_size)
    regime_info = adapter.get_regime_performance_metrics()
    
    return engine, regime_info


if __name__ == "__main__":
    # Example usage
    mock_market_data = {
        'spy_return_1m': 0.05,      # Strong monthly return
        'spy_return_3m': 0.12,      # Strong quarterly return
        'spy_return_6m': 0.18,      # Strong 6-month return
        'vix_level': 16,            # Low volatility
        'vix_change_5d': -2,        # Declining fear
        'market_breadth': 0.72,     # Strong breadth
        'bond_yield_change_1m': 0.8 # Rising yields
    }
    
    adapter = MomentumRegimeAdapter()
    regime, confidence = adapter.detect_market_regime(mock_market_data)
    
    print(f"Detected Market Regime: {regime}")
    print(f"Confidence: {confidence:.1%}")
    
    params = adapter.get_adaptive_parameters(mock_market_data)
    print(f"Optimized Parameters for {regime}:")
    print(f"  Strong Threshold: {params['momentum_thresholds']['strong']:.3f}")
    print(f"  Max Position: {params['max_position_pct']:.1%}")
    print(f"  Expected Sharpe: {params['expected_sharpe']:.3f}")
    print(f"  Expected Return: {params['expected_return']:.1%}")