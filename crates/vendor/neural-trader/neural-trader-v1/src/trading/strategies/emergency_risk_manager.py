"""Emergency Risk Management System - Sophisticated risk controls to prevent 29.3% drawdowns."""

from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass
import statistics
import math
from collections import defaultdict


class RiskState(Enum):
    """Portfolio risk states."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DrawdownState(Enum):
    """Drawdown monitoring states."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DANGER = "danger"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class PositionRisk:
    """Individual position risk metrics."""
    ticker: str
    position_size: float
    current_price: float
    entry_price: float
    stop_loss: float
    trailing_stop: Optional[float]
    time_stop: Optional[datetime]
    atr: float
    volatility: float
    beta: float
    correlation_score: float
    var_contribution: float
    position_heat: float
    days_held: int


@dataclass
class PortfolioRisk:
    """Portfolio-wide risk metrics."""
    total_exposure: float
    portfolio_heat: float
    max_correlation: float
    avg_correlation: float
    var_95: float
    cvar_95: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    beta: float
    sector_concentration: Dict[str, float]
    strategy_allocation: Dict[str, float]
    risk_state: RiskState
    drawdown_state: DrawdownState


class EmergencyRiskManager:
    """
    Sophisticated risk management system to prevent catastrophic drawdowns.
    
    This system implements:
    1. Dynamic position sizing based on multiple factors
    2. Multi-level stop loss system
    3. Portfolio-wide risk controls
    4. Drawdown circuit breakers
    5. Correlation-based position limits
    6. Volatility-adjusted sizing
    7. Time-based stops
    8. Partial profit taking
    """
    
    def __init__(self, portfolio_size: float = 100000):
        """Initialize emergency risk manager with conservative defaults."""
        self.portfolio_size = portfolio_size
        
        # CRITICAL RISK PARAMETERS - Designed to prevent 29.3% drawdowns
        self.risk_limits = {
            # Position sizing limits
            "max_single_position": 0.10,     # Max 10% in any single position (was 50%!)
            "min_position_size": 0.005,      # Min 0.5% position
            "max_correlated_positions": 3,    # Max positions with >0.7 correlation
            "max_sector_exposure": 0.30,      # Max 30% in any sector
            
            # Portfolio heat (total risk)
            "max_portfolio_heat": 0.06,       # Max 6% total portfolio risk
            "elevated_heat_threshold": 0.04,  # Elevated risk at 4%
            "critical_heat_threshold": 0.05,  # Critical risk at 5%
            
            # Drawdown limits
            "warning_drawdown": 0.05,         # Warning at 5% drawdown
            "danger_drawdown": 0.08,          # Danger at 8% drawdown
            "circuit_breaker_drawdown": 0.10, # Emergency stop at 10% drawdown
            "max_acceptable_drawdown": 0.15,  # Never exceed 15% drawdown
            
            # Correlation limits
            "max_correlation": 0.70,          # Max correlation between positions
            "correlation_penalty_threshold": 0.50,  # Start reducing at 0.5 correlation
            
            # Volatility limits
            "max_position_volatility": 0.30,  # Max 30% annualized volatility
            "volatility_scale_factor": 0.15,  # Target 15% portfolio volatility
            
            # Time limits
            "max_holding_days_profitable": 90,     # Max 90 days for profitable positions
            "max_holding_days_losing": 30,        # Max 30 days for losing positions
            "partial_profit_days": 45,             # Take partial profits after 45 days
        }
        
        # Stop loss parameters
        self.stop_loss_params = {
            "atr_multiplier": 2.0,            # 2x ATR for initial stop
            "min_stop_distance": 0.02,        # Min 2% stop distance
            "max_stop_distance": 0.08,        # Max 8% stop distance
            "breakeven_profit_threshold": 0.02,  # Move to breakeven at 2% profit
            "trailing_activation": 0.04,      # Activate trailing at 4% profit
            "trailing_distance": 0.02,         # 2% trailing stop distance
            "time_stop_reduction": 0.001,      # Daily stop reduction for time stops
        }
        
        # Dynamic sizing parameters
        self.sizing_params = {
            "base_kelly_fraction": 0.25,      # Conservative Kelly fraction
            "win_rate_threshold": 0.55,       # Minimum win rate for full sizing
            "streak_bonus_max": 0.20,          # Max 20% bonus for win streaks
            "streak_penalty_max": 0.40,       # Max 40% penalty for loss streaks
            "volatility_scale_power": 1.5,     # Volatility scaling exponent
        }
        
        # Portfolio state tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.daily_pnl_history: List[float] = []
        self.trade_history: List[Dict] = []
        self.current_drawdown = 0.0
        self.peak_equity = portfolio_size
        self.risk_state = RiskState.NORMAL
        self.drawdown_state = DrawdownState.HEALTHY
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        
    def calculate_position_size(self, signal: Dict, portfolio_state: Dict) -> Dict:
        """
        Calculate sophisticated position size with multiple risk adjustments.
        
        Args:
            signal: Trading signal with confidence, volatility, etc.
            portfolio_state: Current portfolio state
            
        Returns:
            Position sizing recommendation with detailed breakdown
        """
        # Extract signal parameters
        base_confidence = signal.get("confidence", 0.5)
        asset_volatility = signal.get("volatility", 0.20)
        expected_return = signal.get("expected_return", 0.08)
        win_probability = signal.get("win_probability", base_confidence)
        strategy_type = signal.get("strategy", "unknown")
        
        # 1. Kelly Criterion base sizing
        kelly_size = self._calculate_kelly_size(
            win_probability, expected_return, asset_volatility
        )
        
        # 2. Volatility adjustment
        vol_adjustment = self._calculate_volatility_adjustment(
            asset_volatility, portfolio_state.get("portfolio_volatility", 0.15)
        )
        
        # 3. Win/Loss streak adjustment
        streak_adjustment = self._calculate_streak_adjustment(
            self.trade_history[-10:] if self.trade_history else []
        )
        
        # 4. Correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(
            signal["ticker"], portfolio_state.get("current_positions", {})
        )
        
        # 5. Drawdown adjustment
        drawdown_adjustment = self._calculate_drawdown_adjustment(
            self.current_drawdown
        )
        
        # 6. Portfolio heat adjustment
        heat_adjustment = self._calculate_heat_adjustment(
            portfolio_state.get("portfolio_heat", 0.0)
        )
        
        # 7. Market regime adjustment
        regime_adjustment = self._calculate_regime_adjustment(
            portfolio_state.get("market_regime", "normal"),
            portfolio_state.get("vix_level", 20)
        )
        
        # Calculate final position size
        base_size = kelly_size * self.risk_limits["max_single_position"]
        
        adjusted_size = (
            base_size *
            vol_adjustment *
            streak_adjustment *
            correlation_adjustment *
            drawdown_adjustment *
            heat_adjustment *
            regime_adjustment
        )
        
        # Apply absolute limits
        final_size = max(
            self.risk_limits["min_position_size"],
            min(adjusted_size, self.risk_limits["max_single_position"])
        )
        
        # Additional reduction if portfolio is in critical state
        if self.risk_state == RiskState.CRITICAL:
            final_size *= 0.5
        elif self.risk_state == RiskState.EMERGENCY:
            final_size = 0  # No new positions in emergency
        
        return {
            "position_size_pct": final_size,
            "position_size_dollars": final_size * self.portfolio_size,
            "kelly_base": kelly_size,
            "adjustments": {
                "volatility": vol_adjustment,
                "streak": streak_adjustment,
                "correlation": correlation_adjustment,
                "drawdown": drawdown_adjustment,
                "portfolio_heat": heat_adjustment,
                "market_regime": regime_adjustment
            },
            "risk_state": self.risk_state.value,
            "approved": final_size > 0,
            "reasoning": self._generate_sizing_reasoning(
                final_size, base_size, signal
            )
        }
    
    def _calculate_kelly_size(self, win_prob: float, expected_return: float, 
                             volatility: float) -> float:
        """Calculate Kelly Criterion position size."""
        if volatility <= 0 or expected_return <= 0:
            return 0.0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        loss_prob = 1 - win_prob
        
        # Approximate win/loss ratio from expected return and volatility
        sharpe = expected_return / volatility
        win_loss_ratio = 1 + sharpe * np.sqrt(252/252)  # Daily Sharpe approximation
        
        kelly_fraction = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio
        
        # Apply conservative Kelly fraction
        conservative_kelly = max(0, kelly_fraction * self.sizing_params["base_kelly_fraction"])
        
        return min(conservative_kelly, 1.0)
    
    def _calculate_volatility_adjustment(self, asset_vol: float, 
                                       portfolio_vol: float) -> float:
        """Adjust position size based on volatility."""
        target_vol = self.risk_limits["volatility_scale_factor"]
        
        # Scale position to achieve target portfolio volatility
        if asset_vol > 0:
            vol_ratio = target_vol / asset_vol
            # Apply scaling with power factor for more aggressive reduction
            adjustment = vol_ratio ** self.sizing_params["volatility_scale_power"]
        else:
            adjustment = 1.0
        
        # Additional penalty for extreme volatility
        if asset_vol > self.risk_limits["max_position_volatility"]:
            adjustment *= 0.5
        
        return max(0.2, min(adjustment, 1.2))
    
    def _calculate_streak_adjustment(self, recent_trades: List[Dict]) -> float:
        """Adjust size based on recent win/loss streaks."""
        if not recent_trades:
            return 1.0
        
        # Count consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        
        for trade in reversed(recent_trades):
            if trade.get("pnl", 0) > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1
        
        # Calculate adjustment
        if consecutive_wins >= 3:
            # Bonus for win streak, but capped
            bonus = min(consecutive_wins * 0.05, 
                       self.sizing_params["streak_bonus_max"])
            return 1.0 + bonus
        elif consecutive_losses >= 2:
            # Penalty for loss streak, more aggressive
            penalty = min(consecutive_losses * 0.15, 
                         self.sizing_params["streak_penalty_max"])
            return 1.0 - penalty
        
        return 1.0
    
    def _calculate_correlation_adjustment(self, ticker: str, 
                                        current_positions: Dict) -> float:
        """Reduce position size for correlated assets."""
        if not current_positions:
            return 1.0
        
        max_correlation = 0.0
        correlated_count = 0
        
        for existing_ticker in current_positions:
            correlation = self._get_correlation(ticker, existing_ticker)
            max_correlation = max(max_correlation, correlation)
            
            if correlation > self.risk_limits["correlation_penalty_threshold"]:
                correlated_count += 1
        
        # Reduce size based on correlation
        if max_correlation > self.risk_limits["max_correlation"]:
            return 0.3  # Severe reduction for highly correlated
        elif max_correlation > self.risk_limits["correlation_penalty_threshold"]:
            # Linear reduction from 1.0 to 0.5
            reduction = (max_correlation - 0.5) / 0.2
            return 1.0 - (reduction * 0.5)
        
        # Additional reduction for multiple correlated positions
        if correlated_count >= self.risk_limits["max_correlated_positions"]:
            return 0.0  # Block new correlated positions
        
        return 1.0
    
    def _calculate_drawdown_adjustment(self, current_drawdown: float) -> float:
        """Reduce position size during drawdowns."""
        drawdown = abs(current_drawdown)
        
        if drawdown >= self.risk_limits["circuit_breaker_drawdown"]:
            return 0.0  # No new positions during circuit breaker
        elif drawdown >= self.risk_limits["danger_drawdown"]:
            return 0.3  # Severe reduction in danger zone
        elif drawdown >= self.risk_limits["warning_drawdown"]:
            # Linear reduction from 1.0 to 0.5
            reduction = (drawdown - 0.05) / 0.03
            return 1.0 - (reduction * 0.5)
        
        return 1.0
    
    def _calculate_heat_adjustment(self, portfolio_heat: float) -> float:
        """Reduce position size when portfolio heat is high."""
        if portfolio_heat >= self.risk_limits["critical_heat_threshold"]:
            return 0.2  # Minimal new positions
        elif portfolio_heat >= self.risk_limits["elevated_heat_threshold"]:
            # Linear reduction
            reduction = (portfolio_heat - 0.04) / 0.01
            return 1.0 - (reduction * 0.6)
        
        return 1.0
    
    def _calculate_regime_adjustment(self, regime: str, vix: float) -> float:
        """Adjust position size based on market regime."""
        adjustments = {
            "bull": 1.0,
            "bear": 0.6,
            "high_volatility": 0.4,
            "crisis": 0.2,
            "normal": 0.8
        }
        
        base_adjustment = adjustments.get(regime, 0.8)
        
        # Additional VIX adjustment
        if vix > 35:
            base_adjustment *= 0.5
        elif vix > 25:
            base_adjustment *= 0.7
        elif vix < 15:
            base_adjustment *= 1.1
        
        return base_adjustment
    
    def calculate_stop_losses(self, position: Dict, market_data: Dict) -> Dict:
        """
        Calculate multi-level stop loss system.
        
        Args:
            position: Position details
            market_data: Current market data including ATR
            
        Returns:
            Multi-level stop loss configuration
        """
        entry_price = position["entry_price"]
        current_price = position["current_price"]
        atr = market_data.get("atr", current_price * 0.02)
        days_held = position.get("days_held", 0)
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. Initial stop loss (ATR-based)
        atr_stop_distance = atr * self.stop_loss_params["atr_multiplier"]
        atr_stop_pct = atr_stop_distance / entry_price
        
        # Apply min/max limits
        stop_distance_pct = max(
            self.stop_loss_params["min_stop_distance"],
            min(atr_stop_pct, self.stop_loss_params["max_stop_distance"])
        )
        
        initial_stop = entry_price * (1 - stop_distance_pct)
        
        # 2. Breakeven stop
        breakeven_stop = None
        if pnl_pct >= self.stop_loss_params["breakeven_profit_threshold"]:
            breakeven_stop = entry_price * 1.001  # Slightly above entry
        
        # 3. Trailing stop
        trailing_stop = None
        if pnl_pct >= self.stop_loss_params["trailing_activation"]:
            # Trail from highest price
            highest_price = position.get("highest_price", current_price)
            trailing_distance = self.stop_loss_params["trailing_distance"]
            
            # Tighten trailing stop as profits grow
            if pnl_pct > 0.10:
                trailing_distance *= 0.8  # Tighter stop for larger profits
            elif pnl_pct > 0.20:
                trailing_distance *= 0.6  # Even tighter for huge profits
            
            trailing_stop = highest_price * (1 - trailing_distance)
        
        # 4. Time-based stop
        time_stop = None
        if days_held > 10:
            # Gradually tighten stop over time
            time_reduction = min(
                days_held * self.stop_loss_params["time_stop_reduction"],
                stop_distance_pct * 0.5  # Max 50% reduction
            )
            time_stop = entry_price * (1 - (stop_distance_pct - time_reduction))
        
        # 5. Volatility-adjusted stop
        volatility = market_data.get("volatility", 0.20)
        if volatility > 0.30:  # High volatility
            # Widen stops in volatile markets
            vol_adjustment = 1.2
        elif volatility < 0.10:  # Low volatility
            # Tighten stops in quiet markets
            vol_adjustment = 0.8
        else:
            vol_adjustment = 1.0
        
        # Select the highest (most conservative) stop
        active_stops = [
            initial_stop * vol_adjustment,
            breakeven_stop,
            trailing_stop,
            time_stop
        ]
        active_stops = [s for s in active_stops if s is not None]
        
        final_stop = max(active_stops) if active_stops else initial_stop
        
        return {
            "stop_loss_price": final_stop,
            "initial_stop": initial_stop,
            "breakeven_stop": breakeven_stop,
            "trailing_stop": trailing_stop,
            "time_stop": time_stop,
            "stop_type": self._determine_active_stop_type(
                final_stop, initial_stop, breakeven_stop, trailing_stop, time_stop
            ),
            "stop_distance_pct": (current_price - final_stop) / current_price,
            "volatility_adjustment": vol_adjustment
        }
    
    def _determine_active_stop_type(self, final: float, initial: float, 
                                   breakeven: Optional[float], trailing: Optional[float],
                                   time: Optional[float]) -> str:
        """Determine which stop type is active."""
        if trailing and abs(final - trailing) < 0.001:
            return "trailing"
        elif breakeven and abs(final - breakeven) < 0.001:
            return "breakeven"
        elif time and abs(final - time) < 0.001:
            return "time"
        else:
            return "initial"
    
    def calculate_portfolio_risk(self, positions: Dict[str, PositionRisk]) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            positions: Current positions with risk metrics
            
        Returns:
            Portfolio-wide risk assessment
        """
        if not positions:
            return PortfolioRisk(
                total_exposure=0.0,
                portfolio_heat=0.0,
                max_correlation=0.0,
                avg_correlation=0.0,
                var_95=0.0,
                cvar_95=0.0,
                current_drawdown=self.current_drawdown,
                max_drawdown=0.0,
                volatility=0.0,
                beta=0.0,
                sector_concentration={},
                strategy_allocation={},
                risk_state=RiskState.NORMAL,
                drawdown_state=DrawdownState.HEALTHY
            )
        
        # Calculate total exposure
        total_exposure = sum(p.position_size for p in positions.values())
        
        # Calculate portfolio heat (total risk)
        portfolio_heat = sum(p.position_heat for p in positions.values())
        
        # Calculate correlation metrics
        correlations = []
        for t1 in positions:
            for t2 in positions:
                if t1 < t2:
                    corr = self._get_correlation(t1, t2)
                    correlations.append(corr)
        
        max_correlation = max(correlations) if correlations else 0.0
        avg_correlation = statistics.mean(correlations) if correlations else 0.0
        
        # Calculate VaR and CVaR (simplified)
        position_vars = [p.var_contribution for p in positions.values()]
        var_95 = sum(position_vars)  # Simple sum for now
        cvar_95 = var_95 * 1.2  # CVaR approximation
        
        # Calculate portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(positions)
        
        # Calculate portfolio beta
        portfolio_beta = self._calculate_portfolio_beta(positions)
        
        # Calculate sector concentration
        sector_exposure = defaultdict(float)
        for pos in positions.values():
            sector = self._get_sector(pos.ticker)
            sector_exposure[sector] += pos.position_size
        
        # Determine risk state
        risk_state = self._determine_risk_state(
            portfolio_heat, total_exposure, self.current_drawdown
        )
        
        # Determine drawdown state
        drawdown_state = self._determine_drawdown_state(self.current_drawdown)
        
        return PortfolioRisk(
            total_exposure=total_exposure,
            portfolio_heat=portfolio_heat,
            max_correlation=max_correlation,
            avg_correlation=avg_correlation,
            var_95=var_95,
            cvar_95=cvar_95,
            current_drawdown=self.current_drawdown,
            max_drawdown=self.current_drawdown,  # Track separately in practice
            volatility=portfolio_vol,
            beta=portfolio_beta,
            sector_concentration=dict(sector_exposure),
            strategy_allocation={},  # Would track by strategy
            risk_state=risk_state,
            drawdown_state=drawdown_state
        )
    
    def _determine_risk_state(self, heat: float, exposure: float, 
                             drawdown: float) -> RiskState:
        """Determine overall portfolio risk state."""
        # Emergency state triggers
        if (drawdown >= self.risk_limits["circuit_breaker_drawdown"] or
            heat >= self.risk_limits["max_portfolio_heat"]):
            return RiskState.EMERGENCY
        
        # Critical state
        if (heat >= self.risk_limits["critical_heat_threshold"] or
            drawdown >= self.risk_limits["danger_drawdown"]):
            return RiskState.CRITICAL
        
        # High state
        if (heat >= self.risk_limits["elevated_heat_threshold"] or
            drawdown >= self.risk_limits["warning_drawdown"]):
            return RiskState.HIGH
        
        # Elevated state
        if exposure > 0.7 or heat > 0.03:
            return RiskState.ELEVATED
        
        return RiskState.NORMAL
    
    def _determine_drawdown_state(self, drawdown: float) -> DrawdownState:
        """Determine drawdown state."""
        dd = abs(drawdown)
        
        if dd >= self.risk_limits["circuit_breaker_drawdown"]:
            return DrawdownState.CIRCUIT_BREAKER
        elif dd >= self.risk_limits["danger_drawdown"]:
            return DrawdownState.DANGER
        elif dd >= self.risk_limits["warning_drawdown"]:
            return DrawdownState.WARNING
        
        return DrawdownState.HEALTHY
    
    def should_exit_position(self, position: PositionRisk, 
                           market_data: Dict) -> Tuple[bool, str]:
        """
        Determine if position should be exited based on risk rules.
        
        Args:
            position: Position risk metrics
            market_data: Current market data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        current_price = market_data["current_price"]
        pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # 1. Stop loss hit
        stop_config = self.calculate_stop_losses(
            {"entry_price": position.entry_price, 
             "current_price": current_price,
             "days_held": position.days_held},
            market_data
        )
        
        if current_price <= stop_config["stop_loss_price"]:
            return True, f"{stop_config['stop_type']}_stop_hit"
        
        # 2. Time-based exits
        if position.days_held > self.risk_limits["max_holding_days_profitable"] and pnl_pct > 0:
            return True, "max_holding_time_profitable"
        
        if position.days_held > self.risk_limits["max_holding_days_losing"] and pnl_pct < 0:
            return True, "max_holding_time_losing"
        
        # 3. Portfolio risk limits
        if self.risk_state == RiskState.EMERGENCY:
            return True, "portfolio_emergency_state"
        
        # 4. Correlation limits exceeded
        correlated_count = sum(
            1 for other in self.positions.values()
            if other.ticker != position.ticker and
            self._get_correlation(position.ticker, other.ticker) > 0.7
        )
        
        if correlated_count >= self.risk_limits["max_correlated_positions"]:
            return True, "correlation_limit_exceeded"
        
        # 5. Volatility spike
        if position.volatility > self.risk_limits["max_position_volatility"] * 1.5:
            return True, "extreme_volatility"
        
        return False, "hold"
    
    def calculate_partial_exit(self, position: PositionRisk) -> Dict:
        """
        Calculate partial profit taking strategy.
        
        Args:
            position: Current position
            
        Returns:
            Partial exit recommendation
        """
        pnl_pct = (position.current_price - position.entry_price) / position.entry_price
        days_held = position.days_held
        
        # No partial exits for losing positions
        if pnl_pct <= 0:
            return {"action": "hold", "exit_pct": 0.0}
        
        # Partial exit triggers
        exit_pct = 0.0
        reasons = []
        
        # 1. Time-based partial exits
        if days_held > self.risk_limits["partial_profit_days"] and pnl_pct > 0.05:
            exit_pct = 0.25
            reasons.append("time_based_profit_taking")
        
        # 2. Profit-based partial exits
        if pnl_pct > 0.20:
            exit_pct = max(exit_pct, 0.50)
            reasons.append("large_profit_taking")
        elif pnl_pct > 0.10:
            exit_pct = max(exit_pct, 0.33)
            reasons.append("moderate_profit_taking")
        
        # 3. Risk-based partial exits
        if self.risk_state in [RiskState.HIGH, RiskState.CRITICAL]:
            exit_pct = max(exit_pct, 0.50)
            reasons.append("portfolio_risk_reduction")
        
        # 4. Volatility-based partial exits
        if position.volatility > 0.40 and pnl_pct > 0.05:
            exit_pct = max(exit_pct, 0.33)
            reasons.append("high_volatility_reduction")
        
        return {
            "action": "partial_exit" if exit_pct > 0 else "hold",
            "exit_pct": exit_pct,
            "exit_shares": int(position.position_size * exit_pct),
            "reasons": reasons,
            "expected_impact": {
                "risk_reduction": exit_pct * position.position_heat,
                "exposure_reduction": exit_pct * position.position_size
            }
        }
    
    def update_portfolio_state(self, current_equity: float, positions: Dict):
        """Update portfolio state and risk metrics."""
        # Update drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_drawdown = (current_equity - self.peak_equity) / self.peak_equity
        
        # Update positions
        self.positions = positions
        
        # Update risk state
        portfolio_risk = self.calculate_portfolio_risk(positions)
        self.risk_state = portfolio_risk.risk_state
        self.drawdown_state = portfolio_risk.drawdown_state
    
    def _get_correlation(self, ticker1: str, ticker2: str) -> float:
        """Get correlation between two assets."""
        key = tuple(sorted([ticker1, ticker2]))
        # In practice, would calculate from price history
        # For now, return mock correlation
        if ticker1 == ticker2:
            return 1.0
        return self.correlation_matrix.get(key, 0.3)
    
    def _get_sector(self, ticker: str) -> str:
        """Get sector for a ticker."""
        # In practice, would use real sector mapping
        sector_map = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "XOM": "Energy",
            "JNJ": "Healthcare",
            "AMZN": "Consumer Discretionary"
        }
        return sector_map.get(ticker, "Other")
    
    def _calculate_portfolio_volatility(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate portfolio volatility."""
        if not positions:
            return 0.0
        
        # Simplified calculation
        weighted_vols = []
        for pos in positions.values():
            weight = pos.position_size / self.portfolio_size
            weighted_vols.append(weight * pos.volatility)
        
        # Simple sum for now (ignores correlations)
        return sum(weighted_vols)
    
    def _calculate_portfolio_beta(self, positions: Dict[str, PositionRisk]) -> float:
        """Calculate portfolio beta."""
        if not positions:
            return 0.0
        
        total_value = sum(p.position_size * p.current_price for p in positions.values())
        if total_value == 0:
            return 0.0
        
        weighted_beta = sum(
            (p.position_size * p.current_price / total_value) * p.beta
            for p in positions.values()
        )
        
        return weighted_beta
    
    def _generate_sizing_reasoning(self, final_size: float, base_size: float, 
                                  signal: Dict) -> str:
        """Generate human-readable sizing explanation."""
        if final_size == 0:
            return "Position blocked due to emergency risk state or correlation limits"
        
        reduction = 1 - (final_size / base_size) if base_size > 0 else 0
        
        if reduction > 0.5:
            return f"Position heavily reduced ({reduction:.0%}) due to risk controls"
        elif reduction > 0.2:
            return f"Position moderately reduced ({reduction:.0%}) for risk management"
        else:
            return f"Position approved at {final_size:.1%} of portfolio"
    
    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        portfolio_risk = self.calculate_portfolio_risk(self.positions)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_metrics": {
                "total_exposure": f"{portfolio_risk.total_exposure:.1%}",
                "portfolio_heat": f"{portfolio_risk.portfolio_heat:.1%}",
                "current_drawdown": f"{portfolio_risk.current_drawdown:.1%}",
                "portfolio_volatility": f"{portfolio_risk.volatility:.1%}",
                "portfolio_beta": f"{portfolio_risk.beta:.2f}"
            },
            "risk_states": {
                "portfolio_risk_state": portfolio_risk.risk_state.value,
                "drawdown_state": portfolio_risk.drawdown_state.value
            },
            "position_count": len(self.positions),
            "correlation_metrics": {
                "max_correlation": f"{portfolio_risk.max_correlation:.2f}",
                "avg_correlation": f"{portfolio_risk.avg_correlation:.2f}"
            },
            "sector_concentration": portfolio_risk.sector_concentration,
            "risk_limits_usage": {
                "heat_usage": f"{portfolio_risk.portfolio_heat / self.risk_limits['max_portfolio_heat']:.1%}",
                "drawdown_usage": f"{abs(self.current_drawdown) / self.risk_limits['max_acceptable_drawdown']:.1%}"
            },
            "recommendations": self._generate_risk_recommendations(portfolio_risk)
        }
    
    def _generate_risk_recommendations(self, portfolio_risk: PortfolioRisk) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        if portfolio_risk.risk_state == RiskState.EMERGENCY:
            recommendations.append("EMERGENCY: Close all positions immediately")
        elif portfolio_risk.risk_state == RiskState.CRITICAL:
            recommendations.append("CRITICAL: Reduce exposure by 50% immediately")
        elif portfolio_risk.risk_state == RiskState.HIGH:
            recommendations.append("HIGH RISK: Consider reducing positions")
        
        if portfolio_risk.portfolio_heat > self.risk_limits["elevated_heat_threshold"]:
            recommendations.append("Portfolio heat elevated - tighten stops")
        
        if portfolio_risk.max_correlation > 0.8:
            recommendations.append("High correlation detected - diversify holdings")
        
        if abs(self.current_drawdown) > self.risk_limits["warning_drawdown"]:
            recommendations.append("In drawdown - reduce position sizes")
        
        return recommendations