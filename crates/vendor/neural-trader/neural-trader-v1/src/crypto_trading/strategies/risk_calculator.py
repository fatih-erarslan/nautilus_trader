"""
Risk Calculator

Comprehensive risk metrics and calculations for crypto trading strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum


class RiskMetricType(Enum):
    """Types of risk metrics"""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAXIMUM_DRAWDOWN = "max_dd"
    SHARPE_RATIO = "sharpe"
    SORTINO_RATIO = "sortino"
    CALMAR_RATIO = "calmar"
    INFORMATION_RATIO = "info_ratio"
    BETA = "beta"
    ALPHA = "alpha"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_dev"
    UPSIDE_POTENTIAL = "upside_pot"


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a portfolio or position"""
    # Basic risk metrics
    volatility: float
    downside_deviation: float
    maximum_drawdown: float
    maximum_drawdown_duration: int  # Days
    
    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Market risk
    beta: float
    alpha: float
    correlation: float
    tracking_error: float
    
    # Tail risk
    skewness: float
    kurtosis: float
    tail_ratio: float
    
    # Other metrics
    win_rate: float
    profit_factor: float
    recovery_factor: float
    ulcer_index: float


@dataclass
class StressTestResult:
    """Result of stress testing"""
    scenario: str
    portfolio_loss: float
    worst_position_loss: float
    var_breach: bool
    recovery_time_estimate: int  # Days
    affected_positions: List[str]


class RiskCalculator:
    """Calculate comprehensive risk metrics for crypto portfolios"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk calculator
        
        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 365
        
    def calculate_risk_metrics(self,
                             returns: np.ndarray,
                             benchmark_returns: Optional[np.ndarray] = None,
                             prices: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            returns: Array of returns
            benchmark_returns: Optional benchmark returns for relative metrics
            prices: Optional price series for drawdown calculations
            
        Returns:
            Comprehensive risk metrics
        """
        # Basic statistics
        volatility = self.calculate_volatility(returns)
        downside_dev = self.calculate_downside_deviation(returns, self.daily_rf)
        
        # Drawdown metrics
        if prices is not None:
            max_dd, max_dd_duration = self.calculate_maximum_drawdown(prices)
        else:
            # Estimate from returns
            cumulative = np.cumprod(1 + returns)
            max_dd, max_dd_duration = self.calculate_maximum_drawdown(cumulative)
        
        # VaR and CVaR
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        cvar_95 = self.calculate_cvar(returns, 0.95)
        cvar_99 = self.calculate_cvar(returns, 0.99)
        
        # Risk-adjusted returns
        sharpe = self.calculate_sharpe_ratio(returns, self.daily_rf)
        sortino = self.calculate_sortino_ratio(returns, self.daily_rf)
        calmar = self.calculate_calmar_ratio(returns, max_dd)
        
        # Market risk (if benchmark provided)
        if benchmark_returns is not None:
            beta, alpha = self.calculate_beta_alpha(returns, benchmark_returns)
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
            tracking_error = np.std(returns - benchmark_returns) * np.sqrt(252)
            info_ratio = self.calculate_information_ratio(returns, benchmark_returns)
        else:
            beta, alpha, correlation, tracking_error, info_ratio = 0, 0, 0, 0, 0
        
        # Distribution metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        tail_ratio = self.calculate_tail_ratio(returns)
        
        # Performance metrics
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        recovery_factor = self.calculate_recovery_factor(returns, max_dd)
        ulcer_index = self.calculate_ulcer_index(returns)
        
        return RiskMetrics(
            volatility=volatility,
            downside_deviation=downside_dev,
            maximum_drawdown=max_dd,
            maximum_drawdown_duration=max_dd_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            tracking_error=tracking_error,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index
        )
    
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        vol = np.std(returns)
        return vol * np.sqrt(252) if annualize else vol
    
    def calculate_downside_deviation(self, 
                                   returns: np.ndarray,
                                   threshold: float = 0,
                                   annualize: bool = True) -> float:
        """Calculate downside deviation (semi-standard deviation)"""
        downside_returns = returns[returns < threshold]
        if len(downside_returns) == 0:
            return 0.0
        
        downside_dev = np.std(downside_returns)
        return downside_dev * np.sqrt(252) if annualize else downside_dev
    
    def calculate_maximum_drawdown(self, 
                                 prices: np.ndarray) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Returns:
            Tuple of (max_drawdown, max_duration_days)
        """
        cumulative = prices / prices[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        max_dd = abs(np.min(drawdowns))
        
        # Calculate duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and drawdown_start is None:
                drawdown_start = i
            elif dd >= 0 and drawdown_start is not None:
                current_duration = i - drawdown_start
                max_duration = max(max_duration, current_duration)
                drawdown_start = None
                
        return max_dd, max_duration
    
    def calculate_var(self, 
                     returns: np.ndarray,
                     confidence: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk
        
        Args:
            returns: Array of returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            method: 'historical', 'parametric', or 'cornish-fisher'
            
        Returns:
            VaR value (negative number representing loss)
        """
        if method == 'historical':
            return np.percentile(returns, (1 - confidence) * 100)
        
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence)
            return mean + z_score * std
        
        elif method == 'cornish-fisher':
            # Cornish-Fisher expansion for non-normal distributions
            mean = np.mean(returns)
            std = np.std(returns)
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            z = stats.norm.ppf(1 - confidence)
            cf_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3*z) * kurt / 24
            
            return mean + cf_z * std
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_cvar(self,
                      returns: np.ndarray,
                      confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        Args:
            returns: Array of returns
            confidence: Confidence level
            
        Returns:
            CVaR value
        """
        var = self.calculate_var(returns, confidence)
        conditional_returns = returns[returns <= var]
        
        if len(conditional_returns) == 0:
            return var
            
        return np.mean(conditional_returns)
    
    def calculate_sharpe_ratio(self,
                             returns: np.ndarray,
                             risk_free_rate: float = 0) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def calculate_sortino_ratio(self,
                              returns: np.ndarray,
                              risk_free_rate: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate
        downside_dev = self.calculate_downside_deviation(returns, risk_free_rate, annualize=False)
        
        if downside_dev == 0:
            return 0
            
        return np.mean(excess_returns) / downside_dev * np.sqrt(252)
    
    def calculate_calmar_ratio(self,
                             returns: np.ndarray,
                             max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        if max_drawdown == 0:
            return 0
            
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    def calculate_information_ratio(self,
                                  returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """Calculate Information ratio"""
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0
            
        return np.mean(active_returns) / tracking_error * np.sqrt(252)
    
    def calculate_beta_alpha(self,
                           returns: np.ndarray,
                           benchmark_returns: np.ndarray) -> Tuple[float, float]:
        """Calculate beta and alpha relative to benchmark"""
        # Ensure equal length
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        # Calculate beta
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            beta = 1.0
        else:
            beta = covariance / benchmark_variance
        
        # Calculate alpha (annualized)
        portfolio_return = np.mean(returns) * 252
        benchmark_return = np.mean(benchmark_returns) * 252
        alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return beta, alpha
    
    def calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (ratio of 95th percentile to 5th percentile gains)"""
        gains = returns[returns > 0]
        losses = abs(returns[returns < 0])
        
        if len(gains) == 0 or len(losses) == 0:
            return 0
            
        gain_95 = np.percentile(gains, 95)
        loss_95 = np.percentile(losses, 95)
        
        if loss_95 == 0:
            return float('inf')
            
        return gain_95 / loss_95
    
    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate (percentage of positive returns)"""
        if len(returns) == 0:
            return 0
        return np.sum(returns > 0) / len(returns)
    
    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = returns[returns > 0]
        losses = abs(returns[returns < 0])
        
        if len(losses) == 0 or np.sum(losses) == 0:
            return float('inf') if len(profits) > 0 else 0
            
        return np.sum(profits) / np.sum(losses)
    
    def calculate_recovery_factor(self,
                                returns: np.ndarray,
                                max_drawdown: float) -> float:
        """Calculate recovery factor (total return / max drawdown)"""
        if max_drawdown == 0:
            return float('inf')
            
        total_return = np.prod(1 + returns) - 1
        return total_return / max_drawdown
    
    def calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index (measures downside volatility)"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        # Sum of squared drawdowns
        squared_dd = drawdowns ** 2
        ulcer = np.sqrt(np.mean(squared_dd)) * 100
        
        return ulcer
    
    def stress_test_portfolio(self,
                            positions: List[Dict[str, float]],
                            scenarios: Dict[str, Dict[str, float]]) -> List[StressTestResult]:
        """
        Perform stress testing on portfolio
        
        Args:
            positions: List of positions with exposures
            scenarios: Dictionary of scenarios with shock factors
            
        Returns:
            List of stress test results
        """
        results = []
        
        for scenario_name, shocks in scenarios.items():
            total_loss = 0
            worst_loss = 0
            affected = []
            
            for position in positions:
                position_loss = 0
                
                # Apply shocks based on position characteristics
                for shock_type, shock_value in shocks.items():
                    if shock_type in position:
                        position_loss += position[shock_type] * shock_value
                        
                total_loss += position_loss * position.get('value', 0)
                
                if position_loss < worst_loss:
                    worst_loss = position_loss
                    
                if position_loss < -0.05:  # More than 5% loss
                    affected.append(position.get('id', 'unknown'))
                    
            # Estimate recovery time based on severity
            recovery_days = int(abs(total_loss) * 100)  # Rough estimate
            
            results.append(StressTestResult(
                scenario=scenario_name,
                portfolio_loss=total_loss,
                worst_position_loss=worst_loss,
                var_breach=total_loss < -0.05,  # 5% threshold
                recovery_time_estimate=recovery_days,
                affected_positions=affected
            ))
            
        return results
    
    def calculate_risk_contribution(self,
                                  weights: np.ndarray,
                                  covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Covariance matrix of returns
            
        Returns:
            Array of risk contributions
        """
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Marginal contributions to risk
        marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
        
        # Component contributions
        contrib = weights * marginal_contrib
        
        return contrib
    
    def calculate_liquidity_risk(self,
                               volumes: np.ndarray,
                               position_size: float,
                               avg_spread: float = 0.001) -> Dict[str, float]:
        """
        Calculate liquidity risk metrics
        
        Args:
            volumes: Historical volumes
            position_size: Size of position
            avg_spread: Average bid-ask spread
            
        Returns:
            Dictionary of liquidity risk metrics
        """
        avg_volume = np.mean(volumes)
        volume_volatility = np.std(volumes) / avg_volume
        
        # Days to liquidate (assuming max 10% of daily volume)
        days_to_liquidate = position_size / (avg_volume * 0.1)
        
        # Estimated liquidation cost
        immediate_impact = avg_spread * position_size
        market_impact = position_size / avg_volume * 0.01  # 1% per 100% of volume
        total_cost = immediate_impact + market_impact * position_size
        
        return {
            'days_to_liquidate': days_to_liquidate,
            'immediate_impact_cost': immediate_impact,
            'market_impact_cost': market_impact * position_size,
            'total_liquidation_cost': total_cost,
            'volume_volatility': volume_volatility,
            'liquidity_score': 1 / (1 + days_to_liquidate)  # 0-1 score
        }
    
    def monte_carlo_var(self,
                       returns: np.ndarray,
                       confidence: float = 0.95,
                       n_simulations: int = 10000,
                       time_horizon: int = 1) -> Dict[str, float]:
        """
        Monte Carlo simulation for VaR calculation
        
        Args:
            returns: Historical returns
            confidence: Confidence level
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR estimates
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate simulations
        simulated_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            n_simulations
        )
        
        # Calculate VaR and CVaR
        var = np.percentile(simulated_returns, (1 - confidence) * 100)
        cvar = np.mean(simulated_returns[simulated_returns <= var])
        
        # Calculate probability of loss
        prob_loss = np.sum(simulated_returns < 0) / n_simulations
        
        # Expected return
        expected_return = np.mean(simulated_returns)
        
        return {
            'var': var,
            'cvar': cvar,
            'probability_of_loss': prob_loss,
            'expected_return': expected_return,
            'best_case': np.percentile(simulated_returns, 95),
            'worst_case': np.percentile(simulated_returns, 5)
        }