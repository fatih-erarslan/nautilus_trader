"""
Adaptive Risk Management System
Dynamic risk controls with regime-based adjustments and correlation-aware portfolio limits
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Risk metrics container"""
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    correlation_risk: float

@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    position_size: float
    var_contribution: float
    marginal_var: float
    correlation_exposure: float
    liquidity_risk: float
    concentration_risk: float

class AdaptiveRiskManager:
    """
    Comprehensive risk management system with dynamic adjustments
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Risk parameters
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.02)  # 2% daily VaR
        self.max_position_size = config.get('max_position_size', 0.05)    # 5% per position
        self.max_sector_exposure = config.get('max_sector_exposure', 0.3)  # 30% per sector
        self.max_correlation = config.get('max_correlation', 0.7)          # 70% max correlation
        self.volatility_lookback = config.get('volatility_lookback', 20)   # 20 days
        self.confidence_level = config.get('confidence_level', 0.95)       # 95% confidence
        
        # Risk state
        self.current_positions = {}
        self.sector_exposures = {}
        self.correlation_matrix = {}
        self.risk_metrics = None
        self.volatility_regime = 'medium'
        
        # Monte Carlo simulation parameters
        self.mc_simulations = config.get('mc_simulations', 10000)
        self.time_horizon = config.get('time_horizon', 1)  # 1 day
        
        logger.info("Adaptive Risk Manager initialized")
    
    async def calculate_position_size(self, symbol: str, entry_price: float, 
                                    stop_loss: float, confidence: float) -> float:
        """Calculate optimal position size using Kelly Criterion with risk adjustments"""
        try:
            # Base position size using Kelly Criterion
            win_rate = min(0.6 + confidence * 0.3, 0.9)  # Scale with confidence
            avg_win = abs(entry_price - stop_loss) * 2   # 2:1 reward to risk
            avg_loss = abs(entry_price - stop_loss)
            
            if avg_loss == 0:
                return 0
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            
            # Apply volatility adjustment
            volatility_adjustment = await self._get_volatility_adjustment(symbol)
            adjusted_size = kelly_fraction * volatility_adjustment
            
            # Apply correlation adjustment
            correlation_adjustment = await self._get_correlation_adjustment(symbol)
            adjusted_size *= correlation_adjustment
            
            # Apply regime-based adjustment
            regime_adjustment = self._get_regime_adjustment()
            adjusted_size *= regime_adjustment
            
            # Apply portfolio heat adjustment
            heat_adjustment = await self._get_portfolio_heat_adjustment()
            adjusted_size *= heat_adjustment
            
            # Final position size
            final_size = min(adjusted_size, self.max_position_size)
            
            logger.debug(f"Position size for {symbol}: {final_size:.4f} "
                        f"(Kelly: {kelly_fraction:.4f}, Vol: {volatility_adjustment:.4f}, "
                        f"Corr: {correlation_adjustment:.4f}, Regime: {regime_adjustment:.4f})")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return self.max_position_size * 0.5  # Conservative fallback
    
    async def assess_portfolio_risk(self, positions: Dict[str, Any]) -> RiskMetrics:
        """Assess overall portfolio risk using Monte Carlo simulation"""
        try:
            if not positions:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            # Get historical data for positions
            position_data = await self._get_position_data(positions)
            
            # Calculate portfolio statistics
            portfolio_returns = self._calculate_portfolio_returns(positions, position_data)
            portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            
            # Monte Carlo simulation for VaR calculation
            simulated_returns = await self._monte_carlo_simulation(positions, position_data)
            
            # Calculate risk metrics
            var_95 = np.percentile(simulated_returns, 5)
            var_99 = np.percentile(simulated_returns, 1)
            expected_shortfall = np.mean(simulated_returns[simulated_returns <= var_95])
            
            # Drawdown calculation
            cumulative_returns = np.cumsum(portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            # Performance ratios
            mean_return = np.mean(portfolio_returns) * 252  # Annualized
            sharpe_ratio = mean_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = mean_return / downside_volatility if downside_volatility > 0 else 0
            
            # Correlation risk
            correlation_risk = await self._calculate_correlation_risk(positions)
            
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                correlation_risk=correlation_risk
            )
            
            self.risk_metrics = risk_metrics
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    async def check_risk_limits(self, new_position: Dict[str, Any]) -> Dict[str, bool]:
        """Check if new position violates risk limits"""
        try:
            checks = {
                'position_size_ok': True,
                'sector_exposure_ok': True,
                'correlation_ok': True,
                'portfolio_var_ok': True,
                'concentration_ok': True
            }
            
            symbol = new_position['symbol']
            position_size = new_position['position_size']
            
            # Position size check
            if position_size > self.max_position_size:
                checks['position_size_ok'] = False
            
            # Sector exposure check
            sector = await self._get_symbol_sector(symbol)
            current_sector_exposure = self.sector_exposures.get(sector, 0)
            if current_sector_exposure + position_size > self.max_sector_exposure:
                checks['sector_exposure_ok'] = False
            
            # Correlation check
            correlation_risk = await self._calculate_position_correlation_risk(new_position)
            if correlation_risk > self.max_correlation:
                checks['correlation_ok'] = False
            
            # Portfolio VaR check
            temp_positions = self.current_positions.copy()
            temp_positions[symbol] = new_position
            risk_metrics = await self.assess_portfolio_risk(temp_positions)
            if abs(risk_metrics.var_95) > self.max_portfolio_risk:
                checks['portfolio_var_ok'] = False
            
            # Concentration risk check
            total_portfolio_value = sum(p.get('value', 0) for p in self.current_positions.values())
            if total_portfolio_value > 0:
                concentration = position_size / (total_portfolio_value + new_position.get('value', 0))
                if concentration > self.max_position_size:
                    checks['concentration_ok'] = False
            
            return checks
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return {k: False for k in ['position_size_ok', 'sector_exposure_ok', 
                                     'correlation_ok', 'portfolio_var_ok', 'concentration_ok']}
    
    async def calculate_dynamic_stops(self, symbol: str, position: Dict[str, Any], 
                                    current_price: float) -> Dict[str, float]:
        """Calculate dynamic stop loss levels based on volatility and momentum"""
        try:
            # Get volatility data
            volatility = await self._get_symbol_volatility(symbol)
            
            # Base stop distance (ATR-based)
            atr = volatility * current_price  # Simplified ATR
            base_stop_distance = atr * 2
            
            # Volatility adjustment
            if self.volatility_regime == 'high':
                base_stop_distance *= 1.5
            elif self.volatility_regime == 'low':
                base_stop_distance *= 0.7
            
            # Position direction
            direction = position.get('direction', 'long')
            entry_price = position.get('entry_price', current_price)
            
            if direction == 'long':
                # Trailing stop for long positions
                initial_stop = entry_price - base_stop_distance
                breakeven_stop = entry_price
                profit_stop = current_price - base_stop_distance * 0.5
                
                # Use highest of trailing stops
                dynamic_stop = max(initial_stop, breakeven_stop, profit_stop)
                
            else:
                # Trailing stop for short positions
                initial_stop = entry_price + base_stop_distance
                breakeven_stop = entry_price
                profit_stop = current_price + base_stop_distance * 0.5
                
                # Use lowest of trailing stops
                dynamic_stop = min(initial_stop, breakeven_stop, profit_stop)
            
            return {
                'dynamic_stop': dynamic_stop,
                'initial_stop': initial_stop,
                'breakeven_stop': breakeven_stop,
                'profit_stop': profit_stop,
                'stop_distance': base_stop_distance
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stops for {symbol}: {e}")
            return {'dynamic_stop': current_price * 0.98}  # 2% fallback
    
    async def update_volatility_regime(self, market_data: Dict[str, Any]):
        """Update volatility regime classification"""
        try:
            # Calculate market volatility
            returns = market_data.get('returns', [])
            if len(returns) < 10:
                return
            
            current_vol = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else np.std(returns) * np.sqrt(252)
            historical_vol = np.std(returns[-60:]) * np.sqrt(252) if len(returns) >= 60 else current_vol
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            if vol_ratio > 1.5:
                self.volatility_regime = 'high'
            elif vol_ratio < 0.7:
                self.volatility_regime = 'low'
            else:
                self.volatility_regime = 'medium'
            
            logger.debug(f"Volatility regime updated to: {self.volatility_regime} (ratio: {vol_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Error updating volatility regime: {e}")
    
    def get_risk_adjusted_targets(self, position: Dict[str, Any], confidence: float) -> Dict[str, float]:
        """Calculate risk-adjusted profit targets"""
        try:
            entry_price = position.get('entry_price', 100)
            stop_loss = position.get('stop_loss', entry_price * 0.98)
            direction = position.get('direction', 'long')
            
            risk_amount = abs(entry_price - stop_loss)
            base_target_multiple = 2.0  # Base 2:1 reward to risk
            
            # Adjust target multiple based on confidence
            confidence_multiplier = 1 + confidence  # 1.0 to 2.0
            target_multiple = base_target_multiple * confidence_multiplier
            
            # Regime adjustments
            if self.volatility_regime == 'high':
                target_multiple *= 1.2  # Higher targets in volatile markets
            elif self.volatility_regime == 'low':
                target_multiple *= 0.8  # Lower targets in calm markets
            
            if direction == 'long':
                primary_target = entry_price + risk_amount * target_multiple
                secondary_target = entry_price + risk_amount * target_multiple * 1.5
            else:
                primary_target = entry_price - risk_amount * target_multiple
                secondary_target = entry_price - risk_amount * target_multiple * 1.5
            
            return {
                'primary_target': primary_target,
                'secondary_target': secondary_target,
                'target_multiple': target_multiple,
                'risk_amount': risk_amount
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted targets: {e}")
            return {'primary_target': position.get('entry_price', 100) * 1.02}
    
    # Helper methods
    async def _get_volatility_adjustment(self, symbol: str) -> float:
        """Get volatility-based position size adjustment"""
        try:
            volatility = await self._get_symbol_volatility(symbol)
            
            # Inverse relationship: higher volatility = smaller positions
            base_volatility = 0.2  # 20% base volatility
            adjustment = base_volatility / volatility if volatility > 0 else 1.0
            
            return min(max(adjustment, 0.3), 2.0)  # Clamp between 30% and 200%
            
        except Exception as e:
            logger.error(f"Error getting volatility adjustment for {symbol}: {e}")
            return 1.0
    
    async def _get_correlation_adjustment(self, symbol: str) -> float:
        """Get correlation-based position size adjustment"""
        try:
            if not self.current_positions:
                return 1.0
            
            # Calculate average correlation with existing positions
            correlations = []
            for existing_symbol in self.current_positions.keys():
                if existing_symbol != symbol:
                    corr = await self._get_correlation(symbol, existing_symbol)
                    correlations.append(abs(corr))
            
            if not correlations:
                return 1.0
            
            avg_correlation = np.mean(correlations)
            
            # Reduce position size for highly correlated assets
            if avg_correlation > 0.7:
                return 0.5
            elif avg_correlation > 0.5:
                return 0.7
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error getting correlation adjustment for {symbol}: {e}")
            return 1.0
    
    def _get_regime_adjustment(self) -> float:
        """Get market regime-based adjustment"""
        regime_adjustments = {
            'low': 1.2,    # Larger positions in low vol
            'medium': 1.0,  # Normal positions
            'high': 0.8     # Smaller positions in high vol
        }
        
        return regime_adjustments.get(self.volatility_regime, 1.0)
    
    async def _get_portfolio_heat_adjustment(self) -> float:
        """Adjust for current portfolio heat/risk"""
        try:
            if not self.risk_metrics:
                return 1.0
            
            # Reduce position sizes if portfolio VaR is high
            current_var = abs(self.risk_metrics.var_95)
            target_var = self.max_portfolio_risk
            
            if current_var > target_var * 0.8:
                return 0.6  # Reduce new positions significantly
            elif current_var > target_var * 0.6:
                return 0.8  # Moderate reduction
            else:
                return 1.0  # No adjustment needed
                
        except Exception as e:
            logger.error(f"Error getting portfolio heat adjustment: {e}")
            return 1.0
    
    # Mock data access methods (replace with real implementations)
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get symbol volatility"""
        return 0.2  # Mock 20% volatility
    
    async def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        return 0.3  # Mock correlation
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Get symbol sector"""
        return 'Technology'  # Mock sector
    
    async def _get_position_data(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Get historical data for positions"""
        return {}  # Mock implementation
    
    def _calculate_portfolio_returns(self, positions: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
        """Calculate portfolio returns"""
        return np.random.normal(0.001, 0.02, 100)  # Mock returns
    
    async def _monte_carlo_simulation(self, positions: Dict[str, Any], data: Dict[str, Any]) -> np.ndarray:
        """Run Monte Carlo simulation"""
        return np.random.normal(0, 0.02, self.mc_simulations)  # Mock simulation
    
    async def _calculate_correlation_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate portfolio correlation risk"""
        return 0.5  # Mock correlation risk
    
    async def _calculate_position_correlation_risk(self, position: Dict[str, Any]) -> float:
        """Calculate correlation risk for new position"""
        return 0.3  # Mock correlation risk