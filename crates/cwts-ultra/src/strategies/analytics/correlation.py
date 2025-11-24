"""
Correlation Analysis Engine

Implements advanced correlation analysis including:
- Dynamic correlation estimation using DCC-GARCH models
- Rolling correlation analysis with breakpoint detection
- Copula-based dependence modeling for tail risk analysis
- Principal component analysis for dimensionality reduction
- Hierarchical clustering for strategy similarity analysis
- Network analysis for strategy interconnectedness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, OAS
import networkx as nx
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from arch import arch_model
from arch.univariate import GARCH, ConstantMean
from statsmodels.tsa.stattools import coint
from statsmodels.stats.diagnostic import het_breuschpagan

logger = logging.getLogger(__name__)


class CorrelationType(Enum):
    """Correlation analysis types"""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    DISTANCE = "distance"
    MUTUAL_INFORMATION = "mutual_information"
    COPULA = "copula"
    DYNAMIC = "dynamic"


class CopulaType(Enum):
    """Copula types for dependence modeling"""
    GAUSSIAN = "gaussian"
    T_COPULA = "t_copula"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"
    ARCHIMEDEAN = "archimedean"


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    correlation_type: CorrelationType
    correlation_matrix: pd.DataFrame
    significance_matrix: Optional[pd.DataFrame]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]]
    dynamic_correlations: Optional[pd.DataFrame]
    regime_correlations: Optional[Dict[str, pd.DataFrame]]
    network_metrics: Optional[Dict[str, Any]]
    clustering_results: Optional[Dict[str, Any]]
    pca_results: Optional[Dict[str, Any]]
    copula_parameters: Optional[Dict[str, Any]]
    tail_dependencies: Optional[pd.DataFrame]


@dataclass
class NetworkMetrics:
    """Network analysis metrics"""
    centrality_measures: Dict[str, Dict[str, float]]
    clustering_coefficient: Dict[str, float]
    degree_distribution: Dict[str, int]
    community_detection: Dict[str, List[str]]
    network_density: float
    average_path_length: float
    small_world_coefficient: float


class CorrelationAnalysisEngine:
    """
    Comprehensive correlation analysis engine
    
    Provides advanced correlation analysis with multiple methodologies
    including dynamic correlations, copula modeling, and network analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.correlation_cache = {}
        self.garch_models = {}
        
    def run_comprehensive_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        correlation_types: Optional[List[CorrelationType]] = None,
        regime_indicators: Optional[pd.Series] = None,
        rolling_window: int = 252,
        min_periods: int = 60
    ) -> Dict[CorrelationType, CorrelationResult]:
        """
        Run comprehensive correlation analysis
        
        Args:
            returns_data: DataFrame of return series
            correlation_types: Types of correlation analysis to run
            regime_indicators: Market regime indicators
            rolling_window: Window size for rolling correlations
            min_periods: Minimum periods for correlation calculation
            
        Returns:
            Dictionary of correlation results by type
        """
        if correlation_types is None:
            correlation_types = [
                CorrelationType.PEARSON,
                CorrelationType.DYNAMIC,
                CorrelationType.COPULA
            ]
        
        results = {}
        
        # Run different correlation analyses in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {}
            
            # Static correlation analysis
            if CorrelationType.PEARSON in correlation_types:
                futures['pearson'] = executor.submit(
                    self.static_correlation_analysis,
                    returns_data, CorrelationType.PEARSON
                )
            
            if CorrelationType.SPEARMAN in correlation_types:
                futures['spearman'] = executor.submit(
                    self.static_correlation_analysis,
                    returns_data, CorrelationType.SPEARMAN
                )
                
            if CorrelationType.KENDALL in correlation_types:
                futures['kendall'] = executor.submit(
                    self.static_correlation_analysis,
                    returns_data, CorrelationType.KENDALL
                )
            
            # Dynamic correlation analysis
            if CorrelationType.DYNAMIC in correlation_types:
                futures['dynamic'] = executor.submit(
                    self.dynamic_correlation_analysis,
                    returns_data, rolling_window, min_periods
                )
            
            # Copula-based dependence analysis
            if CorrelationType.COPULA in correlation_types:
                futures['copula'] = executor.submit(
                    self.copula_dependence_analysis,
                    returns_data
                )
            
            # Distance correlation
            if CorrelationType.DISTANCE in correlation_types:
                futures['distance'] = executor.submit(
                    self.distance_correlation_analysis,
                    returns_data
                )
            
            # Collect results
            for analysis_type, future in futures.items():
                try:
                    result = future.result()
                    if analysis_type == 'pearson':
                        results[CorrelationType.PEARSON] = result
                    elif analysis_type == 'spearman':
                        results[CorrelationType.SPEARMAN] = result
                    elif analysis_type == 'kendall':
                        results[CorrelationType.KENDALL] = result
                    elif analysis_type == 'dynamic':
                        results[CorrelationType.DYNAMIC] = result
                    elif analysis_type == 'copula':
                        results[CorrelationType.COPULA] = result
                    elif analysis_type == 'distance':
                        results[CorrelationType.DISTANCE] = result
                except Exception as e:
                    logger.error(f"Correlation analysis {analysis_type} failed: {e}")
        
        return results
    
    def static_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        correlation_type: CorrelationType = CorrelationType.PEARSON
    ) -> CorrelationResult:
        """
        Static correlation analysis with significance testing
        
        Args:
            returns_data: DataFrame of return series
            correlation_type: Type of correlation to calculate
            
        Returns:
            CorrelationResult with static correlations and significance tests
        """
        # Calculate correlation matrix
        if correlation_type == CorrelationType.PEARSON:
            corr_matrix = returns_data.corr(method='pearson')
        elif correlation_type == CorrelationType.SPEARMAN:
            corr_matrix = returns_data.corr(method='spearman')
        elif correlation_type == CorrelationType.KENDALL:
            corr_matrix = returns_data.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation type: {correlation_type}")
        
        # Statistical significance testing
        significance_matrix = self._calculate_correlation_significance(
            returns_data, correlation_type
        )
        
        # Confidence intervals
        confidence_intervals = self._calculate_correlation_confidence_intervals(
            returns_data, correlation_type
        )
        
        # Network analysis
        network_metrics = self._network_analysis(corr_matrix)
        
        # Hierarchical clustering
        clustering_results = self._hierarchical_clustering(corr_matrix)
        
        # Principal Component Analysis
        pca_results = self._principal_component_analysis(returns_data)
        
        return CorrelationResult(
            correlation_type=correlation_type,
            correlation_matrix=corr_matrix,
            significance_matrix=significance_matrix,
            confidence_intervals=confidence_intervals,
            dynamic_correlations=None,
            regime_correlations=None,
            network_metrics=network_metrics,
            clustering_results=clustering_results,
            pca_results=pca_results,
            copula_parameters=None,
            tail_dependencies=None
        )
    
    def dynamic_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        rolling_window: int = 252,
        min_periods: int = 60
    ) -> CorrelationResult:
        """
        Dynamic correlation analysis using DCC-GARCH models
        
        Args:
            returns_data: DataFrame of return series
            rolling_window: Window size for rolling correlations
            min_periods: Minimum periods for calculation
            
        Returns:
            CorrelationResult with dynamic correlations
        """
        # Rolling correlation analysis
        rolling_correlations = self._rolling_correlation_analysis(
            returns_data, rolling_window, min_periods
        )
        
        # DCC-GARCH model estimation
        try:
            dcc_correlations = self._dcc_garch_analysis(returns_data)
        except Exception as e:
            logger.warning(f"DCC-GARCH analysis failed: {e}")
            dcc_correlations = rolling_correlations
        
        # Regime-dependent correlations
        regime_correlations = self._regime_dependent_correlations(returns_data)
        
        # Breakpoint detection in correlations
        breakpoints = self._correlation_breakpoint_detection(rolling_correlations)
        
        # Average correlation matrix for network analysis
        avg_correlation = rolling_correlations.mean()
        network_metrics = self._network_analysis(
            self._construct_correlation_matrix(avg_correlation, returns_data.columns)
        )
        
        return CorrelationResult(
            correlation_type=CorrelationType.DYNAMIC,
            correlation_matrix=self._construct_correlation_matrix(avg_correlation, returns_data.columns),
            significance_matrix=None,
            confidence_intervals=None,
            dynamic_correlations=dcc_correlations,
            regime_correlations=regime_correlations,
            network_metrics=network_metrics,
            clustering_results=None,
            pca_results=None,
            copula_parameters=None,
            tail_dependencies=None
        )
    
    def copula_dependence_analysis(
        self,
        returns_data: pd.DataFrame,
        copula_types: Optional[List[CopulaType]] = None
    ) -> CorrelationResult:
        """
        Copula-based dependence modeling for tail risk analysis
        
        Args:
            returns_data: DataFrame of return series
            copula_types: Types of copulas to fit
            
        Returns:
            CorrelationResult with copula parameters and tail dependencies
        """
        if copula_types is None:
            copula_types = [CopulaType.GAUSSIAN, CopulaType.T_COPULA]
        
        # Transform to uniform margins using empirical CDF
        uniform_data = self._transform_to_uniform_margins(returns_data)
        
        # Fit different copula models
        copula_parameters = {}
        tail_dependencies = pd.DataFrame(
            index=returns_data.columns,
            columns=returns_data.columns
        )
        
        for i, asset1 in enumerate(returns_data.columns):
            for j, asset2 in enumerate(returns_data.columns):
                if i >= j:
                    continue
                
                # Bivariate data
                u = uniform_data[asset1].values
                v = uniform_data[asset2].values
                
                # Remove NaN values
                mask = ~(np.isnan(u) | np.isnan(v))
                u_clean = u[mask]
                v_clean = v[mask]
                
                if len(u_clean) < 50:  # Minimum sample size
                    continue
                
                # Fit copula models
                best_copula = self._fit_best_copula(u_clean, v_clean, copula_types)
                copula_parameters[f"{asset1}_{asset2}"] = best_copula
                
                # Calculate tail dependencies
                lower_tail, upper_tail = self._calculate_tail_dependencies(
                    u_clean, v_clean, best_copula
                )
                tail_dependencies.loc[asset1, asset2] = upper_tail
                tail_dependencies.loc[asset2, asset1] = lower_tail
        
        # Linear correlation from Gaussian copula (for network analysis)
        gaussian_correlations = self._extract_gaussian_correlations(copula_parameters)
        network_metrics = self._network_analysis(gaussian_correlations)
        
        return CorrelationResult(
            correlation_type=CorrelationType.COPULA,
            correlation_matrix=gaussian_correlations,
            significance_matrix=None,
            confidence_intervals=None,
            dynamic_correlations=None,
            regime_correlations=None,
            network_metrics=network_metrics,
            clustering_results=None,
            pca_results=None,
            copula_parameters=copula_parameters,
            tail_dependencies=tail_dependencies.astype(float)
        )
    
    def distance_correlation_analysis(
        self,
        returns_data: pd.DataFrame
    ) -> CorrelationResult:
        """
        Distance correlation analysis for non-linear dependencies
        
        Args:
            returns_data: DataFrame of return series
            
        Returns:
            CorrelationResult with distance correlations
        """
        assets = returns_data.columns
        n_assets = len(assets)
        
        distance_corr_matrix = pd.DataFrame(
            np.eye(n_assets),
            index=assets,
            columns=assets
        )
        
        # Calculate pairwise distance correlations
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i >= j:
                    continue
                
                x = returns_data[asset1].dropna().values
                y = returns_data[asset2].dropna().values
                
                # Align series
                min_len = min(len(x), len(y))
                x = x[-min_len:]
                y = y[-min_len:]
                
                if len(x) < 30:
                    continue
                
                # Calculate distance correlation
                dcorr = self._calculate_distance_correlation(x, y)
                distance_corr_matrix.loc[asset1, asset2] = dcorr
                distance_corr_matrix.loc[asset2, asset1] = dcorr
        
        # Network analysis
        network_metrics = self._network_analysis(distance_corr_matrix)
        
        # Clustering based on distance correlations
        clustering_results = self._hierarchical_clustering(distance_corr_matrix)
        
        return CorrelationResult(
            correlation_type=CorrelationType.DISTANCE,
            correlation_matrix=distance_corr_matrix,
            significance_matrix=None,
            confidence_intervals=None,
            dynamic_correlations=None,
            regime_correlations=None,
            network_metrics=network_metrics,
            clustering_results=clustering_results,
            pca_results=None,
            copula_parameters=None,
            tail_dependencies=None
        )
    
    # Helper methods for correlation analysis
    def _calculate_correlation_significance(
        self,
        returns_data: pd.DataFrame,
        correlation_type: CorrelationType
    ) -> pd.DataFrame:
        """Calculate statistical significance of correlations"""
        n_assets = len(returns_data.columns)
        significance_matrix = pd.DataFrame(
            np.ones((n_assets, n_assets)),
            index=returns_data.columns,
            columns=returns_data.columns
        )
        
        for i, asset1 in enumerate(returns_data.columns):
            for j, asset2 in enumerate(returns_data.columns):
                if i >= j:
                    continue
                
                x = returns_data[asset1].dropna()
                y = returns_data[asset2].dropna()
                
                # Align series
                common_index = x.index.intersection(y.index)
                if len(common_index) < 10:
                    continue
                
                x_aligned = x.loc[common_index]
                y_aligned = y.loc[common_index]
                
                # Calculate correlation and p-value
                if correlation_type == CorrelationType.PEARSON:
                    corr_coef, p_value = stats.pearsonr(x_aligned, y_aligned)
                elif correlation_type == CorrelationType.SPEARMAN:
                    corr_coef, p_value = stats.spearmanr(x_aligned, y_aligned)
                elif correlation_type == CorrelationType.KENDALL:
                    corr_coef, p_value = stats.kendalltau(x_aligned, y_aligned)
                else:
                    p_value = 1.0
                
                significance_matrix.iloc[i, j] = p_value
                significance_matrix.iloc[j, i] = p_value
        
        return significance_matrix
    
    def _calculate_correlation_confidence_intervals(
        self,
        returns_data: pd.DataFrame,
        correlation_type: CorrelationType
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for correlations"""
        confidence_intervals = {}
        
        for i, asset1 in enumerate(returns_data.columns):
            for j, asset2 in enumerate(returns_data.columns):
                if i >= j:
                    continue
                
                x = returns_data[asset1].dropna()
                y = returns_data[asset2].dropna()
                
                # Align series
                common_index = x.index.intersection(y.index)
                if len(common_index) < 10:
                    continue
                
                x_aligned = x.loc[common_index]
                y_aligned = y.loc[common_index]
                n = len(x_aligned)
                
                # Calculate correlation
                if correlation_type == CorrelationType.PEARSON:
                    corr_coef, _ = stats.pearsonr(x_aligned, y_aligned)
                elif correlation_type == CorrelationType.SPEARMAN:
                    corr_coef, _ = stats.spearmanr(x_aligned, y_aligned)
                elif correlation_type == CorrelationType.KENDALL:
                    corr_coef, _ = stats.kendalltau(x_aligned, y_aligned)
                else:
                    continue
                
                # Fisher transformation for confidence interval
                if correlation_type == CorrelationType.PEARSON and abs(corr_coef) < 0.99:
                    z_score = 0.5 * np.log((1 + corr_coef) / (1 - corr_coef))
                    se = 1 / np.sqrt(n - 3)
                    z_alpha = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                    
                    z_lower = z_score - z_alpha * se
                    z_upper = z_score + z_alpha * se
                    
                    # Transform back
                    corr_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                    corr_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                    
                    confidence_intervals[f"{asset1}_{asset2}"] = (corr_lower, corr_upper)
        
        return confidence_intervals
    
    def _rolling_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        window: int,
        min_periods: int
    ) -> pd.DataFrame:
        """Calculate rolling correlations between all pairs"""
        rolling_correlations = pd.DataFrame(index=returns_data.index)
        
        for i, asset1 in enumerate(returns_data.columns):
            for j, asset2 in enumerate(returns_data.columns):
                if i >= j:
                    continue
                
                pair_name = f"{asset1}_{asset2}"
                rolling_corr = returns_data[asset1].rolling(
                    window=window, 
                    min_periods=min_periods
                ).corr(returns_data[asset2])
                
                rolling_correlations[pair_name] = rolling_corr
        
        return rolling_correlations.dropna()
    
    def _dcc_garch_analysis(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamic Conditional Correlation GARCH analysis
        
        Note: This is a simplified implementation. Full DCC-GARCH would require
        more sophisticated multivariate GARCH modeling.
        """
        # For each asset pair, fit DCC model
        dcc_correlations = pd.DataFrame(index=returns_data.index)
        
        for i, asset1 in enumerate(returns_data.columns):
            for j, asset2 in enumerate(returns_data.columns):
                if i >= j:
                    continue
                
                pair_name = f"{asset1}_{asset2}"
                
                try:
                    # Simplified DCC: EWMA of rolling correlations
                    rolling_corr = returns_data[asset1].rolling(63).corr(returns_data[asset2])
                    dcc_corr = rolling_corr.ewm(span=20).mean()
                    dcc_correlations[pair_name] = dcc_corr
                except Exception as e:
                    logger.warning(f"DCC calculation failed for {pair_name}: {e}")
                    continue
        
        return dcc_correlations.dropna()
    
    def _regime_dependent_correlations(
        self,
        returns_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate correlations in different market regimes"""
        # Simple regime detection based on volatility and returns
        market_return = returns_data.mean(axis=1)
        market_vol = returns_data.std(axis=1).rolling(20).mean()
        
        # Define regimes
        vol_threshold = market_vol.quantile(0.7)
        return_threshold = market_return.rolling(60).mean()
        
        bull_regime = (market_return > return_threshold) & (market_vol < vol_threshold)
        bear_regime = (market_return < return_threshold) & (market_vol > vol_threshold)
        sideways_regime = ~(bull_regime | bear_regime)
        
        regimes = {
            'bull': returns_data[bull_regime],
            'bear': returns_data[bear_regime],
            'sideways': returns_data[sideways_regime]
        }
        
        regime_correlations = {}
        for regime_name, regime_data in regimes.items():
            if len(regime_data) > 30:  # Minimum sample size
                regime_correlations[regime_name] = regime_data.corr()
        
        return regime_correlations
    
    def _correlation_breakpoint_detection(self, rolling_correlations: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect structural breaks in correlations"""
        breakpoints = {}
        
        for column in rolling_correlations.columns:
            series = rolling_correlations[column].dropna()
            if len(series) < 100:
                continue
            
            # Simple breakpoint detection using variance changes
            window_size = 60
            variance_changes = []
            
            for i in range(window_size, len(series) - window_size):
                before_var = series.iloc[i-window_size:i].var()
                after_var = series.iloc[i:i+window_size].var()
                
                if before_var > 0 and after_var > 0:
                    variance_ratio = max(before_var, after_var) / min(before_var, after_var)
                    variance_changes.append((i, variance_ratio))
            
            # Find significant breakpoints
            if variance_changes:
                breakpoint_indices = [
                    idx for idx, ratio in variance_changes 
                    if ratio > 2.0  # Threshold for significant change
                ]
                breakpoints[column] = breakpoint_indices
        
        return breakpoints
    
    def _transform_to_uniform_margins(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Transform data to uniform margins using empirical CDF"""
        uniform_data = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for column in returns_data.columns:
            series = returns_data[column].dropna()
            ranks = series.rank(method='average')
            uniform_data[column] = ranks / (len(ranks) + 1)
        
        return uniform_data
    
    def _fit_best_copula(
        self,
        u: np.ndarray,
        v: np.ndarray,
        copula_types: List[CopulaType]
    ) -> Dict[str, Any]:
        """Fit different copula models and select best one"""
        best_copula = {'type': CopulaType.GAUSSIAN, 'parameters': {}, 'aic': np.inf}
        
        for copula_type in copula_types:
            try:
                if copula_type == CopulaType.GAUSSIAN:
                    # Gaussian copula parameter estimation
                    # Transform to normal using inverse normal CDF
                    x_norm = stats.norm.ppf(u)
                    y_norm = stats.norm.ppf(v)
                    
                    # Remove infinite values
                    mask = np.isfinite(x_norm) & np.isfinite(y_norm)
                    if np.sum(mask) < 10:
                        continue
                    
                    x_clean = x_norm[mask]
                    y_clean = y_norm[mask]
                    
                    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
                    
                    # Calculate AIC (simplified)
                    log_likelihood = self._gaussian_copula_loglikelihood(u, v, correlation)
                    aic = -2 * log_likelihood + 2 * 1  # 1 parameter
                    
                    if aic < best_copula['aic']:
                        best_copula = {
                            'type': copula_type,
                            'parameters': {'correlation': correlation},
                            'aic': aic
                        }
                
                elif copula_type == CopulaType.T_COPULA:
                    # Simplified t-copula (assume degrees of freedom = 4)
                    x_t = stats.t.ppf(u, df=4)
                    y_t = stats.t.ppf(v, df=4)
                    
                    mask = np.isfinite(x_t) & np.isfinite(y_t)
                    if np.sum(mask) < 10:
                        continue
                    
                    correlation = np.corrcoef(x_t[mask], y_t[mask])[0, 1]
                    
                    log_likelihood = self._t_copula_loglikelihood(u, v, correlation, 4)
                    aic = -2 * log_likelihood + 2 * 2  # 2 parameters (correlation, df)
                    
                    if aic < best_copula['aic']:
                        best_copula = {
                            'type': copula_type,
                            'parameters': {'correlation': correlation, 'df': 4},
                            'aic': aic
                        }
            
            except Exception as e:
                logger.warning(f"Copula fitting failed for {copula_type}: {e}")
                continue
        
        return best_copula
    
    def _calculate_tail_dependencies(
        self,
        u: np.ndarray,
        v: np.ndarray,
        copula_info: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate upper and lower tail dependencies"""
        # Empirical tail dependence
        threshold = 0.05  # 5% threshold
        
        # Lower tail dependence
        lower_mask = (u <= threshold) & (v <= threshold)
        lower_tail = np.sum(lower_mask) / np.sum(u <= threshold) if np.sum(u <= threshold) > 0 else 0
        
        # Upper tail dependence
        upper_threshold = 1 - threshold
        upper_mask = (u >= upper_threshold) & (v >= upper_threshold)
        upper_tail = np.sum(upper_mask) / np.sum(u >= upper_threshold) if np.sum(u >= upper_threshold) > 0 else 0
        
        return lower_tail, upper_tail
    
    def _extract_gaussian_correlations(self, copula_parameters: Dict[str, Any]) -> pd.DataFrame:
        """Extract correlation matrix from Gaussian copula parameters"""
        # This is a simplified implementation
        # Would need more sophisticated matrix reconstruction in practice
        
        assets = set()
        for key in copula_parameters.keys():
            asset1, asset2 = key.split('_')
            assets.add(asset1)
            assets.add(asset2)
        
        assets = sorted(list(assets))
        n_assets = len(assets)
        
        corr_matrix = pd.DataFrame(
            np.eye(n_assets),
            index=assets,
            columns=assets
        )
        
        for key, copula_info in copula_parameters.items():
            if copula_info['type'] == CopulaType.GAUSSIAN:
                asset1, asset2 = key.split('_')
                correlation = copula_info['parameters'].get('correlation', 0)
                corr_matrix.loc[asset1, asset2] = correlation
                corr_matrix.loc[asset2, asset1] = correlation
        
        return corr_matrix
    
    def _calculate_distance_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance correlation between two series"""
        # Simplified distance correlation calculation
        n = len(x)
        if n < 4:
            return 0.0
        
        # Calculate pairwise distances
        a_ij = np.abs(x[:, np.newaxis] - x[np.newaxis, :])
        b_ij = np.abs(y[:, np.newaxis] - y[np.newaxis, :])
        
        # Double centering
        a_mean = np.mean(a_ij)
        b_mean = np.mean(b_ij)
        a_row_means = np.mean(a_ij, axis=1)
        b_row_means = np.mean(b_ij, axis=1)
        a_col_means = np.mean(a_ij, axis=0)
        b_col_means = np.mean(b_ij, axis=0)
        
        A = a_ij - a_row_means[:, np.newaxis] - a_col_means[np.newaxis, :] + a_mean
        B = b_ij - b_row_means[:, np.newaxis] - b_col_means[np.newaxis, :] + b_mean
        
        # Calculate distance covariance and variances
        dcov_xy = np.sqrt(np.mean(A * B))
        dcov_xx = np.sqrt(np.mean(A * A))
        dcov_yy = np.sqrt(np.mean(B * B))
        
        # Distance correlation
        if dcov_xx > 0 and dcov_yy > 0:
            dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            dcorr = 0.0
        
        return dcorr
    
    def _network_analysis(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform network analysis on correlation matrix"""
        # Convert correlation to adjacency matrix (absolute values, threshold)
        threshold = 0.3
        adjacency = correlation_matrix.abs() > threshold
        
        # Create NetworkX graph
        G = nx.from_pandas_adjacency(adjacency)
        
        if len(G.nodes()) == 0:
            return {'network_density': 0, 'average_clustering': 0}
        
        # Calculate network metrics
        try:
            # Centrality measures
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            
            # Clustering coefficient
            clustering_coeff = nx.clustering(G)
            
            # Community detection
            try:
                import community  # python-louvain package
                communities = community.best_partition(G)
            except ImportError:
                communities = {}
            
            # Network-level metrics
            density = nx.density(G)
            average_clustering = nx.average_clustering(G)
            
            # Average path length (only for connected components)
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return {'error': str(e)}
        
        return {
            'centrality_measures': {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality,
                'closeness': closeness_centrality
            },
            'clustering_coefficient': clustering_coeff,
            'communities': communities,
            'network_density': density,
            'average_clustering': average_clustering,
            'average_path_length': avg_path_length,
            'number_of_nodes': G.number_of_nodes(),
            'number_of_edges': G.number_of_edges()
        }
    
    def _hierarchical_clustering(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Perform hierarchical clustering on correlation matrix"""
        # Convert correlation to distance
        distance_matrix = 1 - correlation_matrix.abs()
        
        # Perform hierarchical clustering
        try:
            condensed_distances = squareform(distance_matrix.values)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Extract clusters at different levels
            cluster_results = {}
            for n_clusters in [2, 3, 4, 5]:
                if n_clusters <= len(correlation_matrix):
                    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                    
                    # Create cluster mapping
                    cluster_mapping = {}
                    for i, asset in enumerate(correlation_matrix.index):
                        cluster_mapping[asset] = clusters[i]
                    
                    cluster_results[f'{n_clusters}_clusters'] = cluster_mapping
            
            return {
                'linkage_matrix': linkage_matrix,
                'cluster_assignments': cluster_results,
                'distance_matrix': distance_matrix
            }
        
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            return {'error': str(e)}
    
    def _principal_component_analysis(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        try:
            # Standardize data
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_data.dropna())
            
            # Fit PCA
            pca = PCA()
            pca_transformed = pca.fit_transform(returns_scaled)
            
            # Create loadings DataFrame
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(len(pca.components_))],
                index=returns_data.columns
            )
            
            # Calculate explained variance ratios
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Find number of components for 80% and 95% variance
            n_components_80 = np.argmax(cumulative_variance >= 0.8) + 1
            n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            return {
                'loadings': loadings,
                'explained_variance_ratio': explained_variance,
                'cumulative_variance': cumulative_variance,
                'n_components_80pct': n_components_80,
                'n_components_95pct': n_components_95,
                'principal_components': pd.DataFrame(
                    pca_transformed,
                    columns=[f'PC{i+1}' for i in range(pca_transformed.shape[1])],
                    index=returns_data.dropna().index
                )
            }
        
        except Exception as e:
            logger.error(f"PCA analysis failed: {e}")
            return {'error': str(e)}
    
    def _construct_correlation_matrix(self, avg_correlations: pd.Series, columns: pd.Index) -> pd.DataFrame:
        """Construct full correlation matrix from average pairwise correlations"""
        n = len(columns)
        matrix = pd.DataFrame(np.eye(n), index=columns, columns=columns)
        
        for pair_name, corr_value in avg_correlations.items():
            if '_' in pair_name:
                assets = pair_name.split('_')
                if len(assets) == 2 and assets[0] in columns and assets[1] in columns:
                    matrix.loc[assets[0], assets[1]] = corr_value
                    matrix.loc[assets[1], assets[0]] = corr_value
        
        return matrix
    
    def _gaussian_copula_loglikelihood(self, u: np.ndarray, v: np.ndarray, rho: float) -> float:
        """Calculate log-likelihood of Gaussian copula"""
        # Transform to standard normal
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        
        # Remove infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) == 0:
            return -np.inf
        
        # Log-likelihood
        log_likelihood = np.sum(
            -0.5 * np.log(1 - rho**2) - 
            (rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2))
        )
        
        return log_likelihood
    
    def _t_copula_loglikelihood(self, u: np.ndarray, v: np.ndarray, rho: float, df: float) -> float:
        """Calculate log-likelihood of t-copula"""
        # Transform to t-distribution
        x = stats.t.ppf(u, df=df)
        y = stats.t.ppf(v, df=df)
        
        # Remove infinite values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        if len(x) == 0:
            return -np.inf
        
        # Simplified log-likelihood calculation
        log_likelihood = np.sum(
            -0.5 * np.log(1 - rho**2) - 
            ((df + 2) / 2) * np.log(1 + (x**2 + y**2 - 2*rho*x*y) / (df * (1 - rho**2)))
        )
        
        return log_likelihood