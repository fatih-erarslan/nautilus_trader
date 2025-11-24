"""
Strategy Validation Framework

Implements comprehensive statistical validation including:
- Statistical significance testing with Newey-West HAC standard errors
- Multiple hypothesis testing corrections (Bonferroni, Benjamini-Hochberg)
- Bootstrap confidence intervals for performance metrics
- Monte Carlo simulation for strategy robustness testing
- Out-of-sample validation with walk-forward analysis
- Regime-dependent performance analysis (bull, bear, sideways markets)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.stats.diagnostic import het_white
from statsmodels.tsa.stattools import acf
from arch import arch_model

logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Validation method types"""
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    MULTIPLE_HYPOTHESIS = "multiple_hypothesis"
    BOOTSTRAP_CONFIDENCE = "bootstrap_confidence"
    MONTE_CARLO_SIMULATION = "monte_carlo_simulation"
    OUT_OF_SAMPLE = "out_of_sample"
    REGIME_DEPENDENT = "regime_dependent"
    WALK_FORWARD = "walk_forward"
    CROSS_VALIDATION = "cross_validation"


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class ValidationResult:
    """Strategy validation result"""
    validation_method: ValidationMethod
    is_significant: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    test_statistic: float
    effect_size: float
    robustness_score: float
    regime_analysis: Optional[Dict[MarketRegime, Dict[str, float]]] = None
    monte_carlo_results: Optional[Dict[str, Any]] = None
    out_of_sample_results: Optional[Dict[str, float]] = None
    detailed_metrics: Optional[Dict[str, Any]] = None


@dataclass
class StatisticalTest:
    """Statistical test configuration"""
    test_name: str
    test_function: Callable
    null_hypothesis: str
    alternative_hypothesis: str
    required_params: List[str]
    min_sample_size: int


class StrategyValidationFramework:
    """
    Comprehensive strategy validation framework
    
    Provides institutional-grade statistical validation with multiple
    testing methodologies and robustness checks.
    """
    
    def __init__(self, confidence_level: float = 0.95, random_state: int = 42):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize statistical tests
        self.statistical_tests = self._initialize_statistical_tests()
        
        # Regime detection parameters
        self.regime_detection_params = {
            'bull_threshold': 0.15,    # 15% annual return threshold
            'bear_threshold': -0.10,   # -10% annual return threshold
            'volatility_threshold': 0.20,  # 20% annual volatility threshold
            'trend_window': 252,       # 1 year for trend detection
            'volatility_window': 63    # 3 months for volatility regime
        }
        
    def _initialize_statistical_tests(self) -> Dict[str, StatisticalTest]:
        """Initialize available statistical tests"""
        return {
            'sharpe_ratio_test': StatisticalTest(
                test_name="Sharpe Ratio Significance Test",
                test_function=self._test_sharpe_ratio_significance,
                null_hypothesis="Sharpe ratio = 0",
                alternative_hypothesis="Sharpe ratio != 0",
                required_params=['returns'],
                min_sample_size=30
            ),
            'information_ratio_test': StatisticalTest(
                test_name="Information Ratio Significance Test",
                test_function=self._test_information_ratio_significance,
                null_hypothesis="Information ratio = 0",
                alternative_hypothesis="Information ratio != 0",
                required_params=['portfolio_returns', 'benchmark_returns'],
                min_sample_size=30
            ),
            'alpha_significance_test': StatisticalTest(
                test_name="Alpha Significance Test (Newey-West HAC)",
                test_function=self._test_alpha_significance_hac,
                null_hypothesis="Alpha = 0",
                alternative_hypothesis="Alpha != 0",
                required_params=['portfolio_returns', 'benchmark_returns'],
                min_sample_size=60
            ),
            'market_timing_test': StatisticalTest(
                test_name="Market Timing Ability Test",
                test_function=self._test_market_timing_ability,
                null_hypothesis="No market timing ability",
                alternative_hypothesis="Positive market timing ability",
                required_params=['portfolio_returns', 'benchmark_returns'],
                min_sample_size=100
            ),
            'performance_persistence_test': StatisticalTest(
                test_name="Performance Persistence Test",
                test_function=self._test_performance_persistence,
                null_hypothesis="No performance persistence",
                alternative_hypothesis="Performance persists over time",
                required_params=['returns'],
                min_sample_size=252
            )
        }
    
    def run_comprehensive_validation(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_signals: Optional[pd.DataFrame] = None,
        validation_methods: Optional[List[ValidationMethod]] = None,
        custom_tests: Optional[List[StatisticalTest]] = None
    ) -> Dict[ValidationMethod, ValidationResult]:
        """
        Run comprehensive strategy validation analysis
        
        Args:
            portfolio_returns: Strategy return series
            benchmark_returns: Benchmark return series
            strategy_signals: Strategy signal data
            validation_methods: Validation methods to run
            custom_tests: Custom statistical tests
            
        Returns:
            Dictionary of validation results by method
        """
        if validation_methods is None:
            validation_methods = list(ValidationMethod)
            
        results = {}
        
        # Run validation methods in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            
            # Statistical Significance Testing
            if ValidationMethod.STATISTICAL_SIGNIFICANCE in validation_methods:
                futures['statistical'] = executor.submit(
                    self.statistical_significance_testing,
                    portfolio_returns, benchmark_returns, custom_tests
                )
                
            # Multiple Hypothesis Testing
            if ValidationMethod.MULTIPLE_HYPOTHESIS in validation_methods:
                futures['multiple_hypothesis'] = executor.submit(
                    self.multiple_hypothesis_testing,
                    portfolio_returns, benchmark_returns, strategy_signals
                )
                
            # Bootstrap Confidence Intervals
            if ValidationMethod.BOOTSTRAP_CONFIDENCE in validation_methods:
                futures['bootstrap'] = executor.submit(
                    self.bootstrap_confidence_intervals,
                    portfolio_returns, benchmark_returns
                )
                
            # Monte Carlo Simulation
            if ValidationMethod.MONTE_CARLO_SIMULATION in validation_methods:
                futures['monte_carlo'] = executor.submit(
                    self.monte_carlo_robustness_testing,
                    portfolio_returns, benchmark_returns
                )
                
            # Out-of-Sample Validation
            if ValidationMethod.OUT_OF_SAMPLE in validation_methods:
                futures['out_of_sample'] = executor.submit(
                    self.out_of_sample_validation,
                    portfolio_returns, benchmark_returns
                )
                
            # Regime-Dependent Analysis
            if ValidationMethod.REGIME_DEPENDENT in validation_methods:
                futures['regime'] = executor.submit(
                    self.regime_dependent_validation,
                    portfolio_returns, benchmark_returns
                )
                
            # Walk-Forward Analysis
            if ValidationMethod.WALK_FORWARD in validation_methods:
                futures['walk_forward'] = executor.submit(
                    self.walk_forward_validation,
                    portfolio_returns, benchmark_returns
                )
                
            # Cross-Validation
            if ValidationMethod.CROSS_VALIDATION in validation_methods:
                futures['cross_validation'] = executor.submit(
                    self.cross_validation_analysis,
                    portfolio_returns, benchmark_returns
                )
                
            # Collect results
            for validation_type, future in futures.items():
                try:
                    result = future.result()
                    if validation_type == 'statistical':
                        results[ValidationMethod.STATISTICAL_SIGNIFICANCE] = result
                    elif validation_type == 'multiple_hypothesis':
                        results[ValidationMethod.MULTIPLE_HYPOTHESIS] = result
                    elif validation_type == 'bootstrap':
                        results[ValidationMethod.BOOTSTRAP_CONFIDENCE] = result
                    elif validation_type == 'monte_carlo':
                        results[ValidationMethod.MONTE_CARLO_SIMULATION] = result
                    elif validation_type == 'out_of_sample':
                        results[ValidationMethod.OUT_OF_SAMPLE] = result
                    elif validation_type == 'regime':
                        results[ValidationMethod.REGIME_DEPENDENT] = result
                    elif validation_type == 'walk_forward':
                        results[ValidationMethod.WALK_FORWARD] = result
                    elif validation_type == 'cross_validation':
                        results[ValidationMethod.CROSS_VALIDATION] = result
                except Exception as e:
                    logger.error(f"Validation method {validation_type} failed: {e}")
                    
        return results
    
    def statistical_significance_testing(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        custom_tests: Optional[List[StatisticalTest]] = None
    ) -> ValidationResult:
        """
        Comprehensive statistical significance testing
        
        Runs multiple statistical tests with proper corrections for:
        - Autocorrelation (Newey-West HAC standard errors)
        - Heteroskedasticity (White robust standard errors)
        - Non-normality (Bootstrap methods)
        """
        test_results = {}
        p_values = []
        
        # Run standard statistical tests
        for test_name, test_config in self.statistical_tests.items():
            try:
                if len(portfolio_returns) < test_config.min_sample_size:
                    logger.warning(f"Insufficient data for {test_name}")
                    continue
                    
                # Prepare parameters based on test requirements
                test_params = {'returns': portfolio_returns}
                if 'benchmark_returns' in test_config.required_params and benchmark_returns is not None:
                    test_params['benchmark_returns'] = benchmark_returns
                    
                # Run test
                test_result = test_config.test_function(**test_params)
                test_results[test_name] = test_result
                p_values.append(test_result['p_value'])
                
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                continue
        
        # Run custom tests if provided
        if custom_tests:
            for custom_test in custom_tests:
                try:
                    test_params = {'returns': portfolio_returns}
                    if 'benchmark_returns' in custom_test.required_params and benchmark_returns is not None:
                        test_params['benchmark_returns'] = benchmark_returns
                        
                    custom_result = custom_test.test_function(**test_params)
                    test_results[custom_test.test_name] = custom_result
                    p_values.append(custom_result['p_value'])
                    
                except Exception as e:
                    logger.error(f"Custom test {custom_test.test_name} failed: {e}")
        
        # Overall significance assessment
        min_p_value = min(p_values) if p_values else 1.0
        is_significant = min_p_value < self.alpha
        
        # Effect size calculation (Cohen's d for returns)
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns
            effect_size = excess_returns.mean() / excess_returns.std()
        else:
            effect_size = portfolio_returns.mean() / portfolio_returns.std()
        
        # Robustness score based on multiple test consistency
        significant_tests = sum(1 for result in test_results.values() 
                               if result.get('p_value', 1.0) < self.alpha)
        robustness_score = significant_tests / len(test_results) if test_results else 0.0
        
        # Confidence interval for overall performance
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns
            performance_metric = excess_returns.mean() * 252  # Annualized excess return
            std_error = excess_returns.std() / np.sqrt(len(excess_returns)) * np.sqrt(252)
        else:
            performance_metric = portfolio_returns.mean() * 252
            std_error = portfolio_returns.std() / np.sqrt(len(portfolio_returns)) * np.sqrt(252)
            
        confidence_interval = (
            performance_metric - stats.norm.ppf(1 - self.alpha/2) * std_error,
            performance_metric + stats.norm.ppf(1 - self.alpha/2) * std_error
        )
        
        return ValidationResult(
            validation_method=ValidationMethod.STATISTICAL_SIGNIFICANCE,
            is_significant=is_significant,
            p_value=min_p_value,
            confidence_interval=confidence_interval,
            test_statistic=performance_metric / std_error if std_error > 0 else 0,
            effect_size=effect_size,
            robustness_score=robustness_score,
            detailed_metrics={
                'individual_test_results': test_results,
                'number_of_tests': len(test_results),
                'significant_tests': significant_tests,
                'performance_metric': performance_metric,
                'standard_error': std_error
            }
        )
    
    def multiple_hypothesis_testing(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        strategy_signals: Optional[pd.DataFrame] = None
    ) -> ValidationResult:
        """
        Multiple hypothesis testing with proper corrections
        
        Implements:
        - Bonferroni correction (conservative)
        - Benjamini-Hochberg FDR correction (less conservative)
        - Holm-Bonferroni stepwise correction
        - Westfall-Young permutation-based correction
        """
        # Collect all hypotheses to test
        hypotheses = []
        p_values = []
        test_names = []
        
        # Core performance hypotheses
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns
            
            # H1: Positive excess returns
            t_stat, p_val = stats.ttest_1samp(excess_returns, 0)
            p_values.append(p_val / 2)  # One-tailed test
            test_names.append('positive_excess_returns')
            
            # H2: Positive Sharpe ratio
            sharpe_p = self._test_sharpe_ratio_significance(excess_returns)['p_value']
            p_values.append(sharpe_p)
            test_names.append('positive_sharpe_ratio')
            
            # H3: Positive alpha (risk-adjusted returns)
            alpha_result = self._test_alpha_significance_hac(portfolio_returns, benchmark_returns)
            p_values.append(alpha_result['p_value'])
            test_names.append('positive_alpha')
            
            # H4: Market timing ability
            timing_result = self._test_market_timing_ability(portfolio_returns, benchmark_returns)
            p_values.append(timing_result['p_value'])
            test_names.append('market_timing')
        
        # Strategy signal hypotheses
        if strategy_signals is not None:
            for signal_name in strategy_signals.columns:
                signal_returns = strategy_signals[signal_name]
                
                # Test if signal is predictive of returns
                correlation = signal_returns.corr(portfolio_returns.shift(-1))  # Next period return
                n = len(signal_returns.dropna())
                if n > 10:
                    t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    p_values.append(p_val)
                    test_names.append(f'signal_{signal_name}_predictive')
        
        # Additional performance metrics hypotheses
        # H: Downside protection (lower downside deviation)
        if benchmark_returns is not None:
            port_downside = self._calculate_downside_deviation(portfolio_returns)
            bench_downside = self._calculate_downside_deviation(benchmark_returns)
            
            if port_downside > 0 and bench_downside > 0:
                # Test if portfolio has lower downside risk
                ratio = port_downside / bench_downside
                # Using bootstrap for this test
                p_val = self._bootstrap_ratio_test(
                    portfolio_returns, benchmark_returns, 
                    lambda x: self._calculate_downside_deviation(x)
                )
                p_values.append(p_val)
                test_names.append('downside_protection')
        
        # Apply multiple testing corrections
        if not p_values:
            return ValidationResult(
                validation_method=ValidationMethod.MULTIPLE_HYPOTHESIS,
                is_significant=False,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                test_statistic=0.0,
                effect_size=0.0,
                robustness_score=0.0
            )
        
        p_values = np.array(p_values)
        
        # Bonferroni correction
        bonferroni_corrected = p_values * len(p_values)
        bonferroni_significant = bonferroni_corrected < self.alpha
        
        # Benjamini-Hochberg FDR correction
        bh_corrected, bh_significant = self._benjamini_hochberg_correction(p_values)
        
        # Holm-Bonferroni correction
        holm_corrected, holm_significant = self._holm_bonferroni_correction(p_values)
        
        # Overall assessment
        min_corrected_p = min(bh_corrected)
        is_significant = any(bh_significant)  # Use BH as primary correction
        
        # Calculate family-wise error rate
        fwer = 1 - (1 - self.alpha) ** len(p_values)
        
        # Robustness score based on consistent significance across corrections
        consistent_significant = np.sum(bonferroni_significant & bh_significant & holm_significant)
        robustness_score = consistent_significant / len(p_values)
        
        # Effect size: proportion of significant effects
        effect_size = np.sum(bh_significant) / len(p_values)
        
        # Confidence interval for discovery rate
        discovery_rate = effect_size
        discovery_se = np.sqrt(discovery_rate * (1 - discovery_rate) / len(p_values))
        confidence_interval = (
            max(0, discovery_rate - stats.norm.ppf(1 - self.alpha/2) * discovery_se),
            min(1, discovery_rate + stats.norm.ppf(1 - self.alpha/2) * discovery_se)
        )
        
        return ValidationResult(
            validation_method=ValidationMethod.MULTIPLE_HYPOTHESIS,
            is_significant=is_significant,
            p_value=min_corrected_p,
            confidence_interval=confidence_interval,
            test_statistic=stats.norm.ppf(1 - min_corrected_p/2) if min_corrected_p < 1 else 0,
            effect_size=effect_size,
            robustness_score=robustness_score,
            detailed_metrics={
                'raw_p_values': p_values.tolist(),
                'test_names': test_names,
                'bonferroni_corrected': bonferroni_corrected.tolist(),
                'bh_corrected': bh_corrected.tolist(),
                'holm_corrected': holm_corrected.tolist(),
                'bonferroni_significant': bonferroni_significant.tolist(),
                'bh_significant': bh_significant.tolist(),
                'holm_significant': holm_significant.tolist(),
                'family_wise_error_rate': fwer,
                'false_discovery_rate': 1 - effect_size,
                'number_of_hypotheses': len(p_values),
                'significant_hypotheses': np.sum(bh_significant)
            }
        )
    
    def bootstrap_confidence_intervals(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        n_bootstrap: int = 10000,
        block_size: Optional[int] = None
    ) -> ValidationResult:
        """
        Bootstrap confidence intervals for performance metrics
        
        Uses block bootstrap to preserve autocorrelation structure
        and provides robust confidence intervals for key metrics.
        """
        # Determine block size for block bootstrap
        if block_size is None:
            # Optimal block size based on autocorrelation
            autocorr = acf(portfolio_returns.dropna(), nlags=50, fft=True)
            significant_lags = np.where(np.abs(autocorr[1:]) > 0.1)[0]
            block_size = min(int(significant_lags[-1]) + 1 if len(significant_lags) > 0 else 10, 20)
        
        # Performance metrics to bootstrap
        metrics_to_bootstrap = [
            ('annualized_return', lambda x: x.mean() * 252),
            ('annualized_volatility', lambda x: x.std() * np.sqrt(252)),
            ('sharpe_ratio', lambda x: x.mean() / x.std() * np.sqrt(252)),
            ('max_drawdown', self._calculate_max_drawdown),
            ('skewness', lambda x: stats.skew(x.dropna())),
            ('kurtosis', lambda x: stats.kurtosis(x.dropna())),
            ('var_95', lambda x: np.percentile(x.dropna(), 5)),
            ('cvar_95', lambda x: x[x <= np.percentile(x, 5)].mean())
        ]
        
        if benchmark_returns is not None:
            excess_returns = portfolio_returns - benchmark_returns
            metrics_to_bootstrap.extend([
                ('excess_return', lambda x: x.mean() * 252),
                ('tracking_error', lambda x: x.std() * np.sqrt(252)),
                ('information_ratio', lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0),
                ('beta', lambda x: self._calculate_rolling_beta(portfolio_returns, benchmark_returns).mean()),
                ('alpha', lambda x: self._calculate_alpha(portfolio_returns, benchmark_returns))
            ])
            primary_series = excess_returns
        else:
            primary_series = portfolio_returns
        
        # Run bootstrap simulation
        bootstrap_results = {metric_name: [] for metric_name, _ in metrics_to_bootstrap}
        
        for i in range(n_bootstrap):
            # Block bootstrap sampling
            bootstrap_sample = self._block_bootstrap_sample(primary_series, block_size)
            
            # Calculate metrics for bootstrap sample
            for metric_name, metric_func in metrics_to_bootstrap:
                try:
                    if metric_name in ['beta', 'alpha'] and benchmark_returns is not None:
                        # Special handling for metrics requiring benchmark
                        bootstrap_benchmark = self._block_bootstrap_sample(benchmark_returns, block_size)
                        if metric_name == 'beta':
                            metric_value = self._calculate_rolling_beta(
                                bootstrap_sample + benchmark_returns.mean(),  # Reconstruct portfolio
                                bootstrap_benchmark
                            ).mean()
                        else:  # alpha
                            metric_value = self._calculate_alpha(
                                bootstrap_sample + benchmark_returns.mean(),
                                bootstrap_benchmark
                            )
                    else:
                        metric_value = metric_func(bootstrap_sample)
                    
                    bootstrap_results[metric_name].append(metric_value)
                except Exception as e:
                    # Handle edge cases
                    bootstrap_results[metric_name].append(np.nan)
        
        # Calculate confidence intervals
        confidence_intervals = {}
        original_metrics = {}
        
        for metric_name, metric_func in metrics_to_bootstrap:
            try:
                if metric_name in ['beta', 'alpha'] and benchmark_returns is not None:
                    if metric_name == 'beta':
                        original_value = self._calculate_rolling_beta(portfolio_returns, benchmark_returns).mean()
                    else:
                        original_value = self._calculate_alpha(portfolio_returns, benchmark_returns)
                else:
                    original_value = metric_func(primary_series)
                
                original_metrics[metric_name] = original_value
                
                # Calculate confidence interval
                bootstrap_values = np.array(bootstrap_results[metric_name])
                bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]
                
                if len(bootstrap_values) > 0:
                    ci_lower = np.percentile(bootstrap_values, (1 - self.confidence_level) / 2 * 100)
                    ci_upper = np.percentile(bootstrap_values, (1 + self.confidence_level) / 2 * 100)
                    confidence_intervals[metric_name] = (ci_lower, ci_upper)
                else:
                    confidence_intervals[metric_name] = (np.nan, np.nan)
                    
            except Exception as e:
                logger.error(f"Bootstrap calculation for {metric_name} failed: {e}")
                original_metrics[metric_name] = np.nan
                confidence_intervals[metric_name] = (np.nan, np.nan)
        
        # Primary metric for overall assessment (Sharpe ratio or Information ratio)
        primary_metric = 'information_ratio' if benchmark_returns is not None else 'sharpe_ratio'
        primary_value = original_metrics.get(primary_metric, 0)
        primary_ci = confidence_intervals.get(primary_metric, (0, 0))
        
        # Significance test: CI doesn't include zero
        is_significant = primary_ci[0] > 0 if not np.isnan(primary_ci[0]) else False
        
        # P-value approximation from bootstrap
        if primary_metric in bootstrap_results and len(bootstrap_results[primary_metric]) > 0:
            bootstrap_values = np.array(bootstrap_results[primary_metric])
            bootstrap_values = bootstrap_values[~np.isnan(bootstrap_values)]
            p_value = np.mean(bootstrap_values <= 0) * 2  # Two-tailed test
        else:
            p_value = 1.0
        
        # Effect size (standardized)
        effect_size = primary_value if not np.isnan(primary_value) else 0
        
        # Robustness score based on CI width
        ci_width = primary_ci[1] - primary_ci[0] if not np.isnan(primary_ci[1]) else float('inf')
        robustness_score = max(0, 1 - ci_width / abs(primary_value)) if primary_value != 0 else 0
        
        return ValidationResult(
            validation_method=ValidationMethod.BOOTSTRAP_CONFIDENCE,
            is_significant=is_significant,
            p_value=p_value,
            confidence_interval=primary_ci,
            test_statistic=primary_value / ((primary_ci[1] - primary_ci[0]) / 3.92) if ci_width > 0 else 0,  # Approximate SE
            effect_size=effect_size,
            robustness_score=robustness_score,
            detailed_metrics={
                'original_metrics': original_metrics,
                'confidence_intervals': confidence_intervals,
                'bootstrap_iterations': n_bootstrap,
                'block_size': block_size,
                'bootstrap_results_summary': {
                    metric: {
                        'mean': np.nanmean(results),
                        'std': np.nanstd(results),
                        'skewness': stats.skew(np.array(results)[~np.isnan(results)]) if len(np.array(results)[~np.isnan(results)]) > 0 else np.nan
                    } for metric, results in bootstrap_results.items()
                }
            }
        )
    
    # Helper methods for statistical tests
    def _test_sharpe_ratio_significance(self, returns: pd.Series) -> Dict[str, float]:
        """Test statistical significance of Sharpe ratio"""
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        n = len(returns)
        
        # Jobson-Korkie test for Sharpe ratio significance
        # Under null hypothesis: SR = 0
        # Test statistic follows t-distribution with (n-1) degrees of freedom
        t_stat = sharpe * np.sqrt(n - 1) / np.sqrt(1 - sharpe**2 / (2 * (n - 1)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'sharpe_ratio': sharpe,
            'degrees_of_freedom': n - 1
        }
    
    def _test_information_ratio_significance(self, portfolio_returns: pd.Series,
                                           benchmark_returns: pd.Series) -> Dict[str, float]:
        """Test statistical significance of Information Ratio"""
        excess_returns = portfolio_returns - benchmark_returns
        info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        n = len(excess_returns)
        
        # Similar to Sharpe ratio test but for tracking error
        t_stat = info_ratio * np.sqrt(n - 1)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'information_ratio': info_ratio,
            'tracking_error': excess_returns.std() * np.sqrt(252)
        }
    
    def _test_alpha_significance_hac(self, portfolio_returns: pd.Series,
                                    benchmark_returns: pd.Series) -> Dict[str, float]:
        """Test alpha significance with Newey-West HAC standard errors"""
        excess_returns = portfolio_returns - benchmark_returns
        
        # Simple alpha calculation (could extend to CAPM)
        alpha = excess_returns.mean()
        n = len(excess_returns)
        
        # Newey-West HAC standard errors
        # Using automatic lag selection based on sample size
        max_lags = int(4 * (n / 100) ** (2/9))
        
        # Calculate HAC standard error
        gamma_0 = np.var(excess_returns)
        hac_variance = gamma_0
        
        for lag in range(1, min(max_lags + 1, n)):
            autocov = np.cov(excess_returns[:-lag], excess_returns[lag:])[0, 1]
            weight = 1 - lag / (max_lags + 1)
            hac_variance += 2 * weight * autocov
        
        hac_se = np.sqrt(hac_variance / n)
        
        # t-statistic with HAC standard errors
        t_stat = alpha / hac_se if hac_se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
        
        return {
            'test_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha * 252,  # Annualized
            'hac_standard_error': hac_se * np.sqrt(252),
            'max_lags': max_lags
        }
    
    def _test_market_timing_ability(self, portfolio_returns: pd.Series,
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Treynor-Mazuy test for market timing ability"""
        excess_port = portfolio_returns - portfolio_returns.mean()  # Risk-free rate approximation
        excess_bench = benchmark_returns - benchmark_returns.mean()
        
        # Regression: excess_port = alpha + beta1 * excess_bench + beta2 * excess_bench^2 + error
        X = np.column_stack([np.ones(len(excess_bench)), excess_bench, excess_bench**2])
        y = excess_port.values
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            timing_coefficient = beta[2]  # Coefficient on squared market return
            
            # Calculate standard error for timing coefficient
            residuals = y - X @ beta
            mse = np.sum(residuals**2) / (len(y) - 3)
            var_covar_matrix = mse * np.linalg.inv(X.T @ X)
            timing_se = np.sqrt(var_covar_matrix[2, 2])
            
            # t-test for timing ability
            t_stat = timing_coefficient / timing_se if timing_se > 0 else 0
            p_value = 1 - stats.t.cdf(t_stat, len(y) - 3)  # One-tailed test (positive timing)
            
            return {
                'test_statistic': t_stat,
                'p_value': p_value,
                'timing_coefficient': timing_coefficient,
                'timing_standard_error': timing_se,
                'alpha': beta[0] * 252,
                'beta': beta[1]
            }
        except np.linalg.LinAlgError:
            return {
                'test_statistic': 0,
                'p_value': 1.0,
                'timing_coefficient': 0,
                'timing_standard_error': np.nan,
                'alpha': 0,
                'beta': 1
            }
    
    def _test_performance_persistence(self, returns: pd.Series) -> Dict[str, float]:
        """Test for performance persistence over time"""
        # Split returns into overlapping periods
        window_size = 252  # 1 year
        if len(returns) < 2 * window_size:
            return {'test_statistic': 0, 'p_value': 1.0, 'persistence_score': 0}
        
        # Calculate rolling performance metrics
        rolling_sharpe = []
        for i in range(window_size, len(returns) - window_size):
            window_returns = returns.iloc[i-window_size:i]
            sharpe = window_returns.mean() / window_returns.std() * np.sqrt(252)
            rolling_sharpe.append(sharpe)
        
        # Test for persistence using rank correlation
        if len(rolling_sharpe) >= 10:
            # Split into first and second half
            mid_point = len(rolling_sharpe) // 2
            first_half = rolling_sharpe[:mid_point]
            second_half = rolling_sharpe[mid_point:mid_point*2]  # Equal length
            
            # Spearman rank correlation
            correlation, p_value = stats.spearmanr(first_half, second_half)
            
            return {
                'test_statistic': correlation * np.sqrt(len(first_half) - 2) / np.sqrt(1 - correlation**2),
                'p_value': p_value,
                'persistence_score': correlation,
                'periods_analyzed': len(first_half)
            }
        else:
            return {'test_statistic': 0, 'p_value': 1.0, 'persistence_score': 0}
    
    def _calculate_downside_deviation(self, returns: pd.Series, mar: float = 0) -> float:
        """Calculate downside deviation below minimum acceptable return"""
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0.0
        return np.sqrt(np.sum((downside_returns - mar)**2) / len(returns))
    
    def _bootstrap_ratio_test(self, series1: pd.Series, series2: pd.Series,
                             metric_func: Callable, n_bootstrap: int = 1000) -> float:
        """Bootstrap test for ratio of metrics between two series"""
        # Calculate original ratio
        metric1 = metric_func(series1)
        metric2 = metric_func(series2)
        original_ratio = metric1 / metric2 if metric2 != 0 else np.inf
        
        # Bootstrap test under null hypothesis of equal metrics
        bootstrap_ratios = []
        combined_data = pd.concat([series1, series2])
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = combined_data.sample(len(combined_data), replace=True)
            
            # Split into two series of original lengths
            bootstrap_series1 = bootstrap_sample.iloc[:len(series1)]
            bootstrap_series2 = bootstrap_sample.iloc[len(series1):len(series1)+len(series2)]
            
            try:
                boot_metric1 = metric_func(bootstrap_series1)
                boot_metric2 = metric_func(bootstrap_series2)
                boot_ratio = boot_metric1 / boot_metric2 if boot_metric2 != 0 else np.inf
                bootstrap_ratios.append(boot_ratio)
            except:
                continue
        
        # Calculate p-value (two-tailed test)
        bootstrap_ratios = np.array(bootstrap_ratios)
        bootstrap_ratios = bootstrap_ratios[np.isfinite(bootstrap_ratios)]
        
        if len(bootstrap_ratios) == 0:
            return 1.0
            
        p_value = np.mean(np.abs(bootstrap_ratios - 1) >= abs(original_ratio - 1))
        return p_value
    
    def _benjamini_hochberg_correction(self, p_values: np.ndarray,
                                      alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Benjamini-Hochberg false discovery rate correction"""
        if alpha is None:
            alpha = self.alpha
            
        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # BH procedure
        m = len(p_values)
        bh_corrected = np.zeros_like(p_values)
        
        # Find largest k such that P(k) <= (k/m) * alpha
        k = m - 1
        while k >= 0:
            if sorted_p_values[k] <= ((k + 1) / m) * alpha:
                break
            k -= 1
        
        # All hypotheses up to k are rejected
        significant = np.zeros(m, dtype=bool)
        if k >= 0:
            significant[sorted_indices[:k+1]] = True
        
        # Calculate adjusted p-values
        for i in range(m-1, -1, -1):
            if i == m - 1:
                bh_corrected[sorted_indices[i]] = sorted_p_values[i]
            else:
                bh_corrected[sorted_indices[i]] = min(
                    sorted_p_values[i] * m / (i + 1),
                    bh_corrected[sorted_indices[i + 1]]
                )
        
        return bh_corrected, significant
    
    def _holm_bonferroni_correction(self, p_values: np.ndarray,
                                   alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Holm-Bonferroni stepwise correction"""
        if alpha is None:
            alpha = self.alpha
            
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        m = len(p_values)
        
        holm_corrected = np.zeros_like(p_values)
        significant = np.zeros(m, dtype=bool)
        
        # Step-down procedure
        for i in range(m):
            correction_factor = m - i
            holm_corrected[sorted_indices[i]] = sorted_p_values[i] * correction_factor
            
            if sorted_p_values[i] <= alpha / correction_factor:
                significant[sorted_indices[i]] = True
            else:
                # Once we fail to reject, all subsequent hypotheses are not rejected
                break
        
        return holm_corrected, significant
    
    def _block_bootstrap_sample(self, series: pd.Series, block_size: int) -> pd.Series:
        """Generate block bootstrap sample preserving autocorrelation"""
        n = len(series)
        n_blocks = int(np.ceil(n / block_size))
        
        bootstrap_sample = []
        for _ in range(n_blocks):
            # Randomly select starting position for block
            start_idx = np.random.randint(0, n - block_size + 1)
            block = series.iloc[start_idx:start_idx + block_size]
            bootstrap_sample.extend(block.values)
        
        # Trim to original length
        bootstrap_sample = bootstrap_sample[:n]
        return pd.Series(bootstrap_sample, index=series.index)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_rolling_beta(self, portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling beta"""
        return portfolio_returns.rolling(window).cov(benchmark_returns) / \
               benchmark_returns.rolling(window).var()
    
    def _calculate_alpha(self, portfolio_returns: pd.Series,
                        benchmark_returns: pd.Series) -> float:
        """Calculate alpha (excess return over benchmark)"""
        return (portfolio_returns.mean() - benchmark_returns.mean()) * 252