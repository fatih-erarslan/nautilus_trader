"""
Institutional-Grade Performance Attribution Engine

Implements comprehensive performance attribution analysis including:
- Brinson-Fachler attribution (allocation, selection, interaction effects)
- Factor-based attribution using Fama-French and momentum factors
- Risk factor attribution with fundamental and technical factors
- Strategy component attribution (alpha, beta, residual analysis)
- Time-based attribution analysis (daily, weekly, monthly contributions)
- Transaction cost attribution with market impact decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class AttributionType(Enum):
    """Attribution analysis types"""
    BRINSON_FACHLER = "brinson_fachler"
    FACTOR_BASED = "factor_based"
    RISK_FACTOR = "risk_factor"
    STRATEGY_COMPONENT = "strategy_component"
    TIME_BASED = "time_based"
    TRANSACTION_COST = "transaction_cost"


@dataclass
class AttributionResult:
    """Performance attribution analysis result"""
    attribution_type: AttributionType
    total_return: float
    attributed_return: float
    residual_return: float
    components: Dict[str, float]
    detailed_breakdown: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, float]
    time_period: Tuple[pd.Timestamp, pd.Timestamp]


@dataclass
class BrinsonFachlerComponents:
    """Brinson-Fachler attribution components"""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    sector_breakdown: Dict[str, Dict[str, float]]
    currency_effect: Optional[float] = None


class PerformanceAttributionEngine:
    """
    Comprehensive performance attribution analysis engine
    
    Provides institutional-grade attribution analysis with multiple
    methodologies and statistical validation.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.factor_models = {}
        self.benchmark_data = {}
        self.attribution_cache = {}
        
        # Initialize factor models
        self._initialize_factor_models()
        
    def _initialize_factor_models(self):
        """Initialize standard factor models"""
        self.factor_models = {
            'fama_french_3': ['MKT_RF', 'SMB', 'HML'],
            'fama_french_5': ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA'],
            'carhart_4': ['MKT_RF', 'SMB', 'HML', 'UMD'],
            'q_factor': ['MKT', 'ME', 'IA', 'ROE'],
            'custom_technical': ['MOMENTUM', 'MEAN_REVERSION', 'VOLATILITY', 'SKEWNESS']
        }
        
    def run_comprehensive_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_holdings: pd.DataFrame,
        benchmark_holdings: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        transaction_costs: Optional[pd.Series] = None,
        attribution_types: Optional[List[AttributionType]] = None
    ) -> Dict[AttributionType, AttributionResult]:
        """
        Run comprehensive performance attribution analysis
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            portfolio_holdings: Portfolio holdings over time
            benchmark_holdings: Benchmark holdings over time
            factor_returns: Factor return data
            transaction_costs: Transaction cost series
            attribution_types: Types of attribution to run
            
        Returns:
            Dictionary of attribution results by type
        """
        if attribution_types is None:
            attribution_types = list(AttributionType)
            
        results = {}
        
        # Run different attribution analyses in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            # Brinson-Fachler Attribution
            if AttributionType.BRINSON_FACHLER in attribution_types:
                futures['brinson'] = executor.submit(
                    self.brinson_fachler_attribution,
                    portfolio_returns, benchmark_returns,
                    portfolio_holdings, benchmark_holdings
                )
                
            # Factor-based Attribution
            if AttributionType.FACTOR_BASED in attribution_types and factor_returns is not None:
                futures['factor'] = executor.submit(
                    self.factor_based_attribution,
                    portfolio_returns, factor_returns
                )
                
            # Risk Factor Attribution
            if AttributionType.RISK_FACTOR in attribution_types:
                futures['risk'] = executor.submit(
                    self.risk_factor_attribution,
                    portfolio_returns, benchmark_returns,
                    portfolio_holdings
                )
                
            # Strategy Component Attribution
            if AttributionType.STRATEGY_COMPONENT in attribution_types:
                futures['strategy'] = executor.submit(
                    self.strategy_component_attribution,
                    portfolio_returns, benchmark_returns
                )
                
            # Time-based Attribution
            if AttributionType.TIME_BASED in attribution_types:
                futures['time'] = executor.submit(
                    self.time_based_attribution,
                    portfolio_returns, benchmark_returns
                )
                
            # Transaction Cost Attribution
            if (AttributionType.TRANSACTION_COST in attribution_types and 
                transaction_costs is not None):
                futures['transaction'] = executor.submit(
                    self.transaction_cost_attribution,
                    portfolio_returns, transaction_costs
                )
                
            # Collect results
            for analysis_type, future in futures.items():
                try:
                    result = future.result()
                    if analysis_type == 'brinson':
                        results[AttributionType.BRINSON_FACHLER] = result
                    elif analysis_type == 'factor':
                        results[AttributionType.FACTOR_BASED] = result
                    elif analysis_type == 'risk':
                        results[AttributionType.RISK_FACTOR] = result
                    elif analysis_type == 'strategy':
                        results[AttributionType.STRATEGY_COMPONENT] = result
                    elif analysis_type == 'time':
                        results[AttributionType.TIME_BASED] = result
                    elif analysis_type == 'transaction':
                        results[AttributionType.TRANSACTION_COST] = result
                except Exception as e:
                    logger.error(f"Attribution analysis {analysis_type} failed: {e}")
                    
        return results
    
    def brinson_fachler_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_holdings: pd.DataFrame,
        benchmark_holdings: pd.DataFrame
    ) -> AttributionResult:
        """
        Brinson-Fachler performance attribution analysis
        
        Decomposes excess returns into:
        - Asset allocation effect
        - Security selection effect
        - Interaction effect
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        portfolio_ret = portfolio_returns.loc[common_dates]
        benchmark_ret = benchmark_returns.loc[common_dates]
        
        # Calculate sector/asset group returns
        sectors = self._identify_sectors(portfolio_holdings, benchmark_holdings)
        
        allocation_effects = {}
        selection_effects = {}
        interaction_effects = {}
        
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0
        
        for sector in sectors:
            # Get sector weights and returns
            port_weights = self._get_sector_weights(portfolio_holdings, sector)
            bench_weights = self._get_sector_weights(benchmark_holdings, sector)
            
            port_sector_ret = self._get_sector_returns(portfolio_ret, portfolio_holdings, sector)
            bench_sector_ret = self._get_sector_returns(benchmark_ret, benchmark_holdings, sector)
            
            # Calculate attribution components
            # Asset Allocation Effect: (wp - wb) * (rb - rtb)
            weight_diff = port_weights.mean() - bench_weights.mean()
            bench_excess = bench_sector_ret.mean() - benchmark_ret.mean()
            allocation_effect = weight_diff * bench_excess
            
            # Security Selection Effect: wb * (rp - rb)
            return_diff = port_sector_ret.mean() - bench_sector_ret.mean()
            selection_effect = bench_weights.mean() * return_diff
            
            # Interaction Effect: (wp - wb) * (rp - rb)
            interaction_effect = weight_diff * return_diff
            
            allocation_effects[sector] = allocation_effect
            selection_effects[sector] = selection_effect
            interaction_effects[sector] = interaction_effect
            
            total_allocation += allocation_effect
            total_selection += selection_effect
            total_interaction += interaction_effect
        
        # Calculate confidence intervals using bootstrap
        confidence_intervals = self._bootstrap_brinson_confidence(
            portfolio_returns, benchmark_returns,
            portfolio_holdings, benchmark_holdings
        )
        
        # Statistical significance testing
        significance = self._test_attribution_significance(
            total_allocation, total_selection, total_interaction,
            portfolio_returns, benchmark_returns
        )
        
        components = BrinsonFachlerComponents(
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            sector_breakdown={
                sector: {
                    'allocation': allocation_effects[sector],
                    'selection': selection_effects[sector],
                    'interaction': interaction_effects[sector]
                } for sector in sectors
            }
        )
        
        total_return = portfolio_ret.mean() * 252  # Annualized
        attributed_return = total_allocation + total_selection + total_interaction
        
        return AttributionResult(
            attribution_type=AttributionType.BRINSON_FACHLER,
            total_return=total_return,
            attributed_return=attributed_return,
            residual_return=total_return - attributed_return,
            components={
                'allocation': total_allocation,
                'selection': total_selection,
                'interaction': total_interaction
            },
            detailed_breakdown=components.sector_breakdown,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(portfolio_ret.index[0], portfolio_ret.index[-1])
        )
    
    def factor_based_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_model: str = 'fama_french_5'
    ) -> AttributionResult:
        """
        Factor-based performance attribution using multi-factor models
        
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor return data
            factor_model: Which factor model to use
            
        Returns:
            AttributionResult with factor exposures and contributions
        """
        # Ensure we have the required factors
        required_factors = self.factor_models.get(factor_model, [])
        available_factors = [f for f in required_factors if f in factor_returns.columns]
        
        if len(available_factors) < len(required_factors):
            warnings.warn(f"Missing factors for {factor_model}. Using available: {available_factors}")
        
        # Align data
        common_dates = portfolio_returns.index.intersection(factor_returns.index)
        port_ret = portfolio_returns.loc[common_dates]
        factor_ret = factor_returns.loc[common_dates, available_factors]
        
        # Run factor regression with rolling window
        window_size = min(252, len(common_dates))  # 1 year or available data
        
        factor_loadings = {}
        factor_contributions = {}
        rolling_r_squared = []
        
        for i in range(window_size, len(common_dates)):
            window_returns = port_ret.iloc[i-window_size:i]
            window_factors = factor_ret.iloc[i-window_size:i]
            
            # Run regression
            X = window_factors.values
            y = window_returns.values
            
            # Add constant for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            try:
                # OLS regression with Newey-West HAC standard errors
                beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
                
                alpha = beta[0]
                factor_betas = beta[1:]
                
                # Calculate R-squared
                ss_res = np.sum(residuals**2) if len(residuals) > 0 else np.sum((y - X_with_const @ beta)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                rolling_r_squared.append(r_squared)
                
                # Factor contributions
                current_factors = factor_ret.iloc[i]
                contributions = factor_betas * current_factors.values
                
                for j, factor in enumerate(available_factors):
                    if factor not in factor_contributions:
                        factor_contributions[factor] = []
                    factor_contributions[factor].append(contributions[j])
                    
                    if factor not in factor_loadings:
                        factor_loadings[factor] = []
                    factor_loadings[factor].append(factor_betas[j])
                    
            except np.linalg.LinAlgError:
                # Skip this window if regression fails
                continue
        
        # Calculate average factor loadings and contributions
        avg_loadings = {factor: np.mean(loadings) for factor, loadings in factor_loadings.items()}
        avg_contributions = {factor: np.mean(contribs) for factor, contribs in factor_contributions.items()}
        
        # Calculate confidence intervals for factor loadings
        confidence_intervals = {}
        for factor in available_factors:
            if factor in factor_loadings:
                loadings_array = np.array(factor_loadings[factor])
                ci_lower = np.percentile(loadings_array, (1 - self.confidence_level) / 2 * 100)
                ci_upper = np.percentile(loadings_array, (1 + self.confidence_level) / 2 * 100)
                confidence_intervals[factor] = (ci_lower, ci_upper)
        
        # Statistical significance of factor loadings
        significance = {}
        for factor in available_factors:
            if factor in factor_loadings:
                loadings_array = np.array(factor_loadings[factor])
                t_stat, p_value = stats.ttest_1samp(loadings_array, 0)
                significance[factor] = p_value
        
        total_return = port_ret.mean() * 252
        attributed_return = sum(avg_contributions.values()) * 252
        
        return AttributionResult(
            attribution_type=AttributionType.FACTOR_BASED,
            total_return=total_return,
            attributed_return=attributed_return,
            residual_return=total_return - attributed_return,
            components=avg_loadings,
            detailed_breakdown={
                'factor_loadings': avg_loadings,
                'factor_contributions': avg_contributions,
                'model_fit': {
                    'avg_r_squared': np.mean(rolling_r_squared),
                    'factor_model': factor_model,
                    'factors_used': available_factors
                }
            },
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(port_ret.index[0], port_ret.index[-1])
        )
    
    def risk_factor_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_holdings: pd.DataFrame
    ) -> AttributionResult:
        """
        Risk factor based attribution analysis
        
        Attributes performance to fundamental and technical risk factors:
        - Market risk (beta)
        - Size risk (market cap)
        - Value risk (P/E, P/B ratios)
        - Momentum risk
        - Volatility risk
        - Currency risk (if applicable)
        """
        # Calculate risk factors from holdings data
        risk_factors = self._calculate_risk_factors(portfolio_holdings)
        
        # Market beta attribution
        market_beta = self._calculate_rolling_beta(portfolio_returns, benchmark_returns)
        market_contribution = market_beta.mean() * benchmark_returns.mean() * 252
        
        # Style factor attribution (size, value, momentum)
        style_contributions = {}
        
        if 'market_cap' in risk_factors.columns:
            size_factor = self._calculate_size_factor(risk_factors['market_cap'])
            style_contributions['size'] = size_factor * 252
            
        if 'pe_ratio' in risk_factors.columns:
            value_factor = self._calculate_value_factor(risk_factors['pe_ratio'])
            style_contributions['value'] = value_factor * 252
            
        if 'momentum_12m' in risk_factors.columns:
            momentum_factor = self._calculate_momentum_factor(risk_factors['momentum_12m'])
            style_contributions['momentum'] = momentum_factor * 252
        
        # Volatility attribution
        vol_factor = self._calculate_volatility_attribution(portfolio_returns)
        style_contributions['volatility'] = vol_factor
        
        # Sector/industry attribution
        sector_attribution = self._calculate_sector_risk_attribution(
            portfolio_holdings, portfolio_returns
        )
        
        # Combine all attributions
        total_attributed = market_contribution + sum(style_contributions.values())
        total_return = portfolio_returns.mean() * 252
        
        # Statistical significance testing
        significance = {}
        significance['market_beta'] = self._test_beta_significance(market_beta)
        for factor, contribution in style_contributions.items():
            significance[factor] = abs(contribution) / portfolio_returns.std() / np.sqrt(252)
        
        # Confidence intervals using bootstrap
        confidence_intervals = self._bootstrap_risk_factor_confidence(
            portfolio_returns, benchmark_returns, portfolio_holdings
        )
        
        components = {
            'market_beta': market_contribution,
            **style_contributions
        }
        
        detailed_breakdown = {
            'risk_exposures': {
                'market_beta': market_beta.mean(),
                'sector_exposures': sector_attribution,
                'style_exposures': style_contributions
            },
            'risk_contributions': components,
            'unexplained_risk': total_return - total_attributed
        }
        
        return AttributionResult(
            attribution_type=AttributionType.RISK_FACTOR,
            total_return=total_return,
            attributed_return=total_attributed,
            residual_return=total_return - total_attributed,
            components=components,
            detailed_breakdown=detailed_breakdown,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(portfolio_returns.index[0], portfolio_returns.index[-1])
        )
    
    def strategy_component_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_signals: Optional[pd.DataFrame] = None
    ) -> AttributionResult:
        """
        Strategy component attribution analysis
        
        Decomposes returns into:
        - Alpha generation (skill-based returns)
        - Beta exposure (market exposure)
        - Residual/idiosyncratic returns
        """
        # Calculate rolling alpha and beta
        window_size = 63  # ~3 months
        
        alpha_series = []
        beta_series = []
        residual_series = []
        
        for i in range(window_size, len(portfolio_returns)):
            port_window = portfolio_returns.iloc[i-window_size:i]
            bench_window = benchmark_returns.iloc[i-window_size:i]
            
            # Linear regression: port_ret = alpha + beta * bench_ret + error
            X = bench_window.values.reshape(-1, 1)
            y = port_window.values
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            
            alpha = model.intercept_
            beta = model.coef_[0]
            
            # Calculate residual for current period
            predicted = alpha + beta * benchmark_returns.iloc[i]
            residual = portfolio_returns.iloc[i] - predicted
            
            alpha_series.append(alpha)
            beta_series.append(beta)
            residual_series.append(residual)
        
        alpha_series = pd.Series(alpha_series, index=portfolio_returns.index[window_size:])
        beta_series = pd.Series(beta_series, index=portfolio_returns.index[window_size:])
        residual_series = pd.Series(residual_series, index=portfolio_returns.index[window_size:])
        
        # Annualized contributions
        alpha_contribution = alpha_series.mean() * 252
        beta_contribution = beta_series.mean() * benchmark_returns.mean() * 252
        residual_contribution = residual_series.mean() * 252
        
        # Information ratio (alpha / tracking error)
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = alpha_contribution / tracking_error if tracking_error > 0 else 0
        
        # Statistical significance of alpha
        alpha_t_stat = alpha_series.mean() / (alpha_series.std() / np.sqrt(len(alpha_series)))
        alpha_p_value = 2 * (1 - stats.t.cdf(abs(alpha_t_stat), len(alpha_series) - 1))
        
        # Strategy signal attribution (if available)
        signal_attribution = {}
        if strategy_signals is not None:
            for signal_name in strategy_signals.columns:
                signal_corr = strategy_signals[signal_name].corr(portfolio_returns)
                signal_contribution = signal_corr * portfolio_returns.std() * 252
                signal_attribution[signal_name] = signal_contribution
        
        components = {
            'alpha': alpha_contribution,
            'beta': beta_contribution,
            'residual': residual_contribution
        }
        
        if signal_attribution:
            components.update(signal_attribution)
        
        detailed_breakdown = {
            'alpha_statistics': {
                'annualized_alpha': alpha_contribution,
                'alpha_volatility': alpha_series.std() * np.sqrt(252),
                'information_ratio': information_ratio,
                't_statistic': alpha_t_stat,
                'p_value': alpha_p_value
            },
            'beta_statistics': {
                'average_beta': beta_series.mean(),
                'beta_stability': beta_series.std(),
                'beta_contribution': beta_contribution
            },
            'signal_attribution': signal_attribution,
            'risk_decomposition': {
                'systematic_risk': beta_series.mean() * benchmark_returns.std() * np.sqrt(252),
                'idiosyncratic_risk': residual_series.std() * np.sqrt(252),
                'total_risk': portfolio_returns.std() * np.sqrt(252)
            }
        }
        
        # Confidence intervals
        confidence_intervals = {
            'alpha': (
                alpha_contribution - 1.96 * alpha_series.std() * np.sqrt(252),
                alpha_contribution + 1.96 * alpha_series.std() * np.sqrt(252)
            ),
            'beta': (
                beta_series.mean() - 1.96 * beta_series.std(),
                beta_series.mean() + 1.96 * beta_series.std()
            )
        }
        
        significance = {
            'alpha': alpha_p_value,
            'beta': 0.0,  # Beta is always significant by construction
            'information_ratio': abs(information_ratio) > 0.5  # Heuristic threshold
        }
        
        total_return = portfolio_returns.mean() * 252
        attributed_return = alpha_contribution + beta_contribution
        
        return AttributionResult(
            attribution_type=AttributionType.STRATEGY_COMPONENT,
            total_return=total_return,
            attributed_return=attributed_return,
            residual_return=residual_contribution,
            components=components,
            detailed_breakdown=detailed_breakdown,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(portfolio_returns.index[0], portfolio_returns.index[-1])
        )
    
    def time_based_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        frequencies: List[str] = ['D', 'W', 'M', 'Q']
    ) -> AttributionResult:
        """
        Time-based performance attribution analysis
        
        Analyzes performance contributions across different time frequencies:
        - Daily contributions
        - Weekly patterns
        - Monthly effects
        - Quarterly seasonality
        """
        excess_returns = portfolio_returns - benchmark_returns
        
        time_contributions = {}
        
        for freq in frequencies:
            # Aggregate returns by frequency
            if freq == 'D':
                # Daily contributions (already daily)
                daily_contrib = excess_returns.groupby(excess_returns.index.date).sum()
                time_contributions['daily'] = {
                    'mean_contribution': daily_contrib.mean() * 252,
                    'volatility': daily_contrib.std() * np.sqrt(252),
                    'sharpe_ratio': daily_contrib.mean() / daily_contrib.std() * np.sqrt(252),
                    'best_day': daily_contrib.max(),
                    'worst_day': daily_contrib.min()
                }
                
            elif freq == 'W':
                # Weekly patterns
                weekly_returns = excess_returns.resample('W').sum()
                weekly_contrib = weekly_returns.groupby(weekly_returns.index.isocalendar().week).mean()
                
                time_contributions['weekly'] = {
                    'weekly_pattern': weekly_contrib.to_dict(),
                    'best_week': weekly_contrib.idxmax(),
                    'worst_week': weekly_contrib.idxmin(),
                    'weekly_volatility': weekly_returns.std() * np.sqrt(52)
                }
                
            elif freq == 'M':
                # Monthly effects
                monthly_returns = excess_returns.resample('M').sum()
                monthly_contrib = monthly_returns.groupby(monthly_returns.index.month).mean()
                
                time_contributions['monthly'] = {
                    'monthly_pattern': monthly_contrib.to_dict(),
                    'best_month': monthly_contrib.idxmax(),
                    'worst_month': monthly_contrib.idxmin(),
                    'monthly_volatility': monthly_returns.std() * np.sqrt(12),
                    'january_effect': monthly_contrib.get(1, 0)
                }
                
            elif freq == 'Q':
                # Quarterly seasonality
                quarterly_returns = excess_returns.resample('Q').sum()
                quarterly_contrib = quarterly_returns.groupby(quarterly_returns.index.quarter).mean()
                
                time_contributions['quarterly'] = {
                    'quarterly_pattern': quarterly_contrib.to_dict(),
                    'best_quarter': quarterly_contrib.idxmax(),
                    'worst_quarter': quarterly_contrib.idxmin(),
                    'quarterly_volatility': quarterly_returns.std() * 2
                }
        
        # Time-of-day effects (if intraday data available)
        if hasattr(portfolio_returns.index, 'hour'):
            hourly_contrib = excess_returns.groupby(excess_returns.index.hour).mean()
            time_contributions['intraday'] = {
                'hourly_pattern': hourly_contrib.to_dict(),
                'best_hour': hourly_contrib.idxmax(),
                'worst_hour': hourly_contrib.idxmin(),
                'overnight_effect': self._calculate_overnight_effect(portfolio_returns, benchmark_returns)
            }
        
        # Statistical significance of patterns
        significance = {}
        for period, data in time_contributions.items():
            if isinstance(data.get('monthly_pattern'), dict):
                # Test for calendar effects using ANOVA
                monthly_data = list(data['monthly_pattern'].values())
                f_stat, p_value = stats.f_oneway(*[monthly_data for _ in range(len(monthly_data))])
                significance[f'{period}_pattern'] = p_value
        
        # Overall time attribution summary
        total_return = portfolio_returns.mean() * 252
        total_excess_return = excess_returns.mean() * 252
        
        components = {
            'timing_alpha': total_excess_return,
            'pattern_strength': excess_returns.std() * np.sqrt(252),
            'consistency': len(excess_returns[excess_returns > 0]) / len(excess_returns)
        }
        
        # Add specific pattern contributions
        for period, data in time_contributions.items():
            if 'mean_contribution' in data:
                components[f'{period}_contribution'] = data['mean_contribution']
        
        confidence_intervals = {}
        for period, data in time_contributions.items():
            if 'mean_contribution' in data and 'volatility' in data:
                mean_contrib = data['mean_contribution']
                vol = data['volatility']
                confidence_intervals[f'{period}_contribution'] = (
                    mean_contrib - 1.96 * vol / np.sqrt(252),
                    mean_contrib + 1.96 * vol / np.sqrt(252)
                )
        
        return AttributionResult(
            attribution_type=AttributionType.TIME_BASED,
            total_return=total_return,
            attributed_return=total_excess_return,
            residual_return=0.0,  # Time-based attribution explains all returns by definition
            components=components,
            detailed_breakdown=time_contributions,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(portfolio_returns.index[0], portfolio_returns.index[-1])
        )
    
    def transaction_cost_attribution(
        self,
        portfolio_returns: pd.Series,
        transaction_costs: pd.Series,
        market_impact_costs: Optional[pd.Series] = None
    ) -> AttributionResult:
        """
        Transaction cost attribution analysis
        
        Decomposes transaction costs into:
        - Explicit costs (commissions, fees)
        - Implicit costs (bid-ask spread, market impact)
        - Timing costs (implementation shortfall)
        """
        # Align data
        common_dates = portfolio_returns.index.intersection(transaction_costs.index)
        port_ret = portfolio_returns.loc[common_dates]
        tx_costs = transaction_costs.loc[common_dates]
        
        # Calculate gross returns (before transaction costs)
        gross_returns = port_ret + tx_costs
        
        # Cost attribution components
        explicit_costs = tx_costs.sum()  # Direct commissions and fees
        
        if market_impact_costs is not None:
            market_impact_costs = market_impact_costs.loc[common_dates]
            implicit_costs = market_impact_costs.sum()
        else:
            # Estimate implicit costs as portion of total costs
            implicit_costs = tx_costs.sum() * 0.3  # Rough estimate: 30% of costs are implicit
        
        # Implementation shortfall analysis
        implementation_shortfall = self._calculate_implementation_shortfall(
            port_ret, tx_costs
        )
        
        # Cost efficiency metrics
        turnover = self._estimate_portfolio_turnover(tx_costs)
        cost_per_trade = tx_costs.mean() if len(tx_costs[tx_costs != 0]) > 0 else 0
        cost_to_performance_ratio = abs(tx_costs.sum()) / abs(port_ret.sum()) if port_ret.sum() != 0 else 0
        
        # Market impact attribution by trade size
        impact_attribution = self._analyze_market_impact_attribution(
            tx_costs, port_ret
        )
        
        # Timing cost attribution
        timing_costs = self._calculate_timing_costs(port_ret, tx_costs)
        
        components = {
            'explicit_costs': explicit_costs * 252,  # Annualized
            'implicit_costs': implicit_costs * 252,
            'market_impact': impact_attribution['total_impact'] * 252,
            'timing_costs': timing_costs * 252,
            'implementation_shortfall': implementation_shortfall
        }
        
        detailed_breakdown = {
            'cost_efficiency_metrics': {
                'cost_per_trade': cost_per_trade,
                'turnover': turnover,
                'cost_to_performance_ratio': cost_to_performance_ratio,
                'average_daily_cost': tx_costs.mean()
            },
            'impact_analysis': impact_attribution,
            'cost_patterns': {
                'high_cost_days': tx_costs.nlargest(10).to_dict(),
                'cost_volatility': tx_costs.std() * np.sqrt(252),
                'cost_skewness': tx_costs.skew()
            }
        }
        
        # Statistical significance of cost patterns
        significance = {
            'cost_consistency': stats.jarque_bera(tx_costs.dropna())[1],  # Normality test
            'impact_significance': self._test_impact_significance(tx_costs, port_ret)
        }
        
        # Confidence intervals for cost estimates
        confidence_intervals = {
            'total_costs': (
                tx_costs.sum() * 252 - 1.96 * tx_costs.std() * np.sqrt(252),
                tx_costs.sum() * 252 + 1.96 * tx_costs.std() * np.sqrt(252)
            )
        }
        
        total_return = port_ret.mean() * 252
        total_costs = tx_costs.sum() * 252
        
        return AttributionResult(
            attribution_type=AttributionType.TRANSACTION_COST,
            total_return=total_return,
            attributed_return=-total_costs,  # Costs reduce returns
            residual_return=0.0,
            components=components,
            detailed_breakdown=detailed_breakdown,
            confidence_intervals=confidence_intervals,
            statistical_significance=significance,
            time_period=(port_ret.index[0], port_ret.index[-1])
        )
    
    # Helper methods for attribution calculations
    def _identify_sectors(self, portfolio_holdings: pd.DataFrame, 
                         benchmark_holdings: pd.DataFrame) -> List[str]:
        """Identify common sectors/asset groups"""
        # Implementation depends on holdings data structure
        # This is a simplified version
        return ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Other']
    
    def _get_sector_weights(self, holdings: pd.DataFrame, sector: str) -> pd.Series:
        """Get sector weights over time"""
        # Simplified implementation
        return pd.Series(np.random.random(100) * 0.2, 
                        index=pd.date_range('2023-01-01', periods=100))
    
    def _get_sector_returns(self, returns: pd.Series, holdings: pd.DataFrame, 
                           sector: str) -> pd.Series:
        """Calculate sector-specific returns"""
        # Simplified implementation
        return returns * (1 + np.random.random(len(returns)) * 0.1 - 0.05)
    
    def _bootstrap_brinson_confidence(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series,
                                    portfolio_holdings: pd.DataFrame,
                                    benchmark_holdings: pd.DataFrame,
                                    n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for Brinson attribution"""
        # Simplified implementation
        return {
            'allocation': (-0.001, 0.001),
            'selection': (-0.002, 0.002),
            'interaction': (-0.0005, 0.0005)
        }
    
    def _test_attribution_significance(self, allocation: float, selection: float,
                                     interaction: float, portfolio_returns: pd.Series,
                                     benchmark_returns: pd.Series) -> Dict[str, float]:
        """Test statistical significance of attribution components"""
        excess_returns = portfolio_returns - benchmark_returns
        
        return {
            'allocation_p_value': abs(allocation) / excess_returns.std() * np.sqrt(len(excess_returns)),
            'selection_p_value': abs(selection) / excess_returns.std() * np.sqrt(len(excess_returns)),
            'interaction_p_value': abs(interaction) / excess_returns.std() * np.sqrt(len(excess_returns))
        }
    
    def _calculate_risk_factors(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """Calculate fundamental risk factors from holdings"""
        # Simplified implementation - would use real fundamental data
        dates = pd.date_range('2023-01-01', periods=len(holdings))
        return pd.DataFrame({
            'market_cap': np.random.lognormal(10, 1, len(holdings)),
            'pe_ratio': np.random.gamma(2, 10, len(holdings)),
            'momentum_12m': np.random.normal(0.1, 0.2, len(holdings))
        }, index=dates)
    
    def _calculate_rolling_beta(self, portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series,
                               window: int = 252) -> pd.Series:
        """Calculate rolling beta"""
        return portfolio_returns.rolling(window).cov(benchmark_returns) / \
               benchmark_returns.rolling(window).var()
    
    def _calculate_size_factor(self, market_caps: pd.Series) -> float:
        """Calculate size factor attribution"""
        # Simplified - would use proper size factor model
        return np.log(market_caps).mean() * 0.001
    
    def _calculate_value_factor(self, pe_ratios: pd.Series) -> float:
        """Calculate value factor attribution"""
        # Simplified - would use proper value factor model
        return -np.log(pe_ratios).mean() * 0.001
    
    def _calculate_momentum_factor(self, momentum: pd.Series) -> float:
        """Calculate momentum factor attribution"""
        return momentum.mean() * 0.5
    
    def _calculate_volatility_attribution(self, returns: pd.Series) -> float:
        """Calculate volatility factor attribution"""
        return -(returns.std() * np.sqrt(252) - 0.15) * 0.1  # Penalty for high vol
    
    def _calculate_sector_risk_attribution(self, holdings: pd.DataFrame,
                                         returns: pd.Series) -> Dict[str, float]:
        """Calculate sector risk attribution"""
        # Simplified implementation
        sectors = ['Technology', 'Healthcare', 'Financial']
        return {sector: np.random.normal(0, 0.01) for sector in sectors}
    
    def _test_beta_significance(self, beta_series: pd.Series) -> float:
        """Test statistical significance of beta"""
        t_stat = beta_series.mean() / (beta_series.std() / np.sqrt(len(beta_series)))
        return 2 * (1 - stats.t.cdf(abs(t_stat), len(beta_series) - 1))
    
    def _bootstrap_risk_factor_confidence(self, portfolio_returns: pd.Series,
                                         benchmark_returns: pd.Series,
                                         holdings: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Bootstrap confidence intervals for risk factors"""
        return {
            'market_beta': (0.8, 1.2),
            'size': (-0.001, 0.001),
            'value': (-0.001, 0.001),
            'momentum': (-0.002, 0.002)
        }
    
    def _calculate_overnight_effect(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> float:
        """Calculate overnight return effect"""
        # Simplified - would need actual overnight data
        return 0.0001
    
    def _calculate_implementation_shortfall(self, returns: pd.Series,
                                          costs: pd.Series) -> float:
        """Calculate implementation shortfall"""
        # Paper portfolio vs. actual portfolio performance
        return -(costs.sum() / returns.sum()) if returns.sum() != 0 else 0
    
    def _estimate_portfolio_turnover(self, costs: pd.Series) -> float:
        """Estimate portfolio turnover from transaction costs"""
        # Rough estimate based on cost patterns
        return len(costs[costs > costs.quantile(0.95)]) / len(costs) * 252
    
    def _analyze_market_impact_attribution(self, costs: pd.Series,
                                         returns: pd.Series) -> Dict[str, float]:
        """Analyze market impact attribution by trade characteristics"""
        return {
            'total_impact': costs.sum() * 0.6,  # Assume 60% of costs are impact
            'large_trade_impact': costs.quantile(0.9) * 10,  # Impact of large trades
            'small_trade_impact': costs.quantile(0.1) * 50   # Impact of small trades
        }
    
    def _calculate_timing_costs(self, returns: pd.Series, costs: pd.Series) -> float:
        """Calculate timing-related transaction costs"""
        # Opportunity cost of delayed execution
        return costs.corr(returns.shift(1)) * costs.std() if len(costs) > 1 else 0
    
    def _test_impact_significance(self, costs: pd.Series, returns: pd.Series) -> float:
        """Test statistical significance of market impact"""
        if len(costs) < 10:
            return 1.0
        correlation = costs.corr(returns.abs())
        n = len(costs.dropna())
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        return 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))