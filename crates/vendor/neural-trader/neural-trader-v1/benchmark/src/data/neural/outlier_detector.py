"""
Outlier Detector for Time Series Data

This module provides sophisticated outlier detection and handling techniques
specifically designed for financial time series data used in neural forecasting.

Key Features:
- Multiple detection algorithms
- Context-aware outlier identification
- Adaptive thresholds
- Market regime consideration
- Real-time detection capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)


class OutlierMethod(Enum):
    """Available outlier detection methods."""
    STATISTICAL = "statistical"  # Z-score and IQR
    ISOLATION_FOREST = "isolation_forest"
    MODIFIED_Z_SCORE = "modified_z_score"  # Using MAD
    GRUBBS_TEST = "grubbs_test"
    CONTEXTUAL = "contextual"  # Consider market context
    ENSEMBLE = "ensemble"  # Combine multiple methods
    SEASONAL_DECOMPOSE = "seasonal_decompose"
    PERCENTAGE_CHANGE = "percentage_change"  # Based on returns
    AUTO_SELECT = "auto_select"


class OutlierAction(Enum):
    """Actions to take with detected outliers."""
    REMOVE = "remove"
    CAP = "cap"  # Cap at threshold
    REPLACE_MEDIAN = "replace_median"
    REPLACE_INTERPOLATE = "replace_interpolate"
    FLAG_ONLY = "flag_only"  # Just mark, don't modify


@dataclass
class OutlierConfig:
    """Configuration for outlier detection."""
    # Primary method
    method: OutlierMethod = OutlierMethod.AUTO_SELECT
    
    # Statistical thresholds
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    modified_z_threshold: float = 3.5
    
    # Percentage change thresholds
    max_daily_change: float = 0.2  # 20% max daily change
    max_intraday_change: float = 0.1  # 10% max intraday change
    
    # Isolation Forest parameters
    contamination: float = 0.01  # Expected outlier ratio
    
    # Contextual parameters
    rolling_window: int = 50  # Window for rolling statistics
    volatility_adjustment: bool = True  # Adjust for volatility regimes
    
    # Quality control
    max_outlier_ratio: float = 0.05  # Maximum 5% outliers
    min_data_points: int = 30  # Minimum data for reliable detection
    
    # Action to take
    outlier_action: OutlierAction = OutlierAction.CAP
    
    # Advanced options
    consider_volume: bool = True  # Consider volume in detection
    market_hours_only: bool = False  # Only detect during market hours
    seasonal_adjustment: bool = True


@dataclass
class OutlierInfo:
    """Information about detected outliers."""
    index: int
    value: float
    method: str
    score: float  # Outlier score
    severity: str  # 'mild', 'moderate', 'severe'
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutlierResult:
    """Result of outlier detection and handling."""
    success: bool
    data: Optional[pd.DataFrame] = None
    outliers_detected: List[OutlierInfo] = field(default_factory=list)
    outliers_handled: int = 0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class OutlierDetector:
    """
    Advanced outlier detector for time series financial data.
    
    This class provides sophisticated outlier detection capabilities
    that account for the unique characteristics of financial time series,
    including volatility clustering, market regimes, and seasonal patterns.
    """
    
    def __init__(self, config: OutlierConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize method-specific components
        self.isolation_forest = None
        self.scaler = StandardScaler()
        
        # Detection statistics
        self.detection_stats = {
            'total_processed': 0,
            'total_outliers_detected': 0,
            'outliers_by_method': {method.value: 0 for method in OutlierMethod},
            'false_positive_rate': 0.0,
            'detection_accuracy': []
        }
        
        self.logger.info(f"OutlierDetector initialized with method: {config.method.value}")
    
    async def detect_and_handle(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> OutlierResult:
        """
        Detect and handle outliers in time series data.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Primary column to analyze for outliers
            feature_columns: Additional columns to consider
        
        Returns:
            OutlierResult with processed data and outlier information
        """
        try:
            self.logger.info(f"Starting outlier detection for {len(data)} samples")
            
            # Validate inputs
            validation_result = self._validate_input(data, target_column, feature_columns)
            if not validation_result['valid']:
                return OutlierResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            # Prepare data for analysis
            analysis_data = self._prepare_analysis_data(data, target_column, feature_columns)
            
            # Select detection method
            selected_method = self._select_method(analysis_data)
            
            # Detect outliers
            outliers = await self._detect_outliers(analysis_data, selected_method)
            
            if not outliers:
                self.logger.info("No outliers detected")
                return OutlierResult(
                    success=True,
                    data=data.copy(),
                    outliers_detected=[],
                    quality_metrics={'outlier_ratio': 0.0, 'data_quality': 1.0}
                )
            
            # Validate outlier detection results
            validation_result = self._validate_outliers(outliers, analysis_data)
            if not validation_result['valid']:
                return OutlierResult(
                    success=False,
                    errors=validation_result['errors'],
                    warnings=validation_result.get('warnings', [])
                )
            
            # Handle outliers based on configuration
            handled_data, handling_info = await self._handle_outliers(
                data.copy(), outliers, target_column, feature_columns
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                data, handled_data, outliers, analysis_data
            )
            
            # Update statistics
            self._update_statistics(selected_method, outliers, quality_metrics)
            
            return OutlierResult(
                success=True,
                data=handled_data,
                outliers_detected=outliers,
                outliers_handled=handling_info['handled_count'],
                quality_metrics=quality_metrics,
                warnings=handling_info.get('warnings', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
            return OutlierResult(
                success=False,
                errors=[str(e)]
            )
    
    def _validate_input(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate input data."""
        errors = []
        
        # Check if data is empty
        if data.empty:
            errors.append("Input data is empty")
        
        # Check minimum data requirement
        if len(data) < self.config.min_data_points:
            errors.append(f"Insufficient data: {len(data)} < {self.config.min_data_points}")
        
        # Check target column
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found")
        
        # Check feature columns
        if feature_columns:
            missing_features = set(feature_columns) - set(data.columns)
            if missing_features:
                errors.append(f"Missing feature columns: {missing_features}")
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data must have DatetimeIndex")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _prepare_analysis_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Prepare data for outlier analysis."""
        # Extract target series
        target_series = data[target_column].copy()
        
        # Calculate returns
        returns = target_series.pct_change().dropna()
        log_returns = np.log(target_series / target_series.shift(1)).dropna()
        
        # Calculate rolling statistics
        rolling_mean = target_series.rolling(window=self.config.rolling_window).mean()
        rolling_std = target_series.rolling(window=self.config.rolling_window).std()
        rolling_median = target_series.rolling(window=self.config.rolling_window).median()
        
        # Volume data if available
        volume_series = None
        if self.config.consider_volume and 'volume' in data.columns:
            volume_series = data['volume'].copy()
        
        # Market hours filtering
        market_hours_mask = None
        if self.config.market_hours_only:
            market_hours_mask = self._get_market_hours_mask(data.index)
        
        return {
            'original_data': data,
            'target_series': target_series,
            'returns': returns,
            'log_returns': log_returns,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_median': rolling_median,
            'volume_series': volume_series,
            'market_hours_mask': market_hours_mask,
            'feature_columns': feature_columns or []
        }
    
    def _select_method(self, analysis_data: Dict[str, Any]) -> OutlierMethod:
        """Select the best outlier detection method based on data characteristics."""
        if self.config.method != OutlierMethod.AUTO_SELECT:
            return self.config.method
        
        # Auto-select based on data characteristics
        target_series = analysis_data['target_series']
        returns = analysis_data['returns']
        
        # Data characteristics
        data_length = len(target_series)
        return_volatility = returns.std() if len(returns) > 0 else 0
        has_volume = analysis_data['volume_series'] is not None
        
        # Selection logic
        if data_length < 100:
            return OutlierMethod.STATISTICAL
        elif return_volatility > 0.05:  # High volatility
            return OutlierMethod.CONTEXTUAL
        elif has_volume and data_length > 500:
            return OutlierMethod.ISOLATION_FOREST
        elif data_length > 1000:
            return OutlierMethod.ENSEMBLE
        else:
            return OutlierMethod.MODIFIED_Z_SCORE
    
    async def _detect_outliers(
        self,
        analysis_data: Dict[str, Any],
        method: OutlierMethod
    ) -> List[OutlierInfo]:
        """Detect outliers using the specified method."""
        if method == OutlierMethod.STATISTICAL:
            return self._statistical_detection(analysis_data)
        elif method == OutlierMethod.ISOLATION_FOREST:
            return await self._isolation_forest_detection(analysis_data)
        elif method == OutlierMethod.MODIFIED_Z_SCORE:
            return self._modified_z_score_detection(analysis_data)
        elif method == OutlierMethod.GRUBBS_TEST:
            return self._grubbs_test_detection(analysis_data)
        elif method == OutlierMethod.CONTEXTUAL:
            return self._contextual_detection(analysis_data)
        elif method == OutlierMethod.ENSEMBLE:
            return await self._ensemble_detection(analysis_data)
        elif method == OutlierMethod.SEASONAL_DECOMPOSE:
            return self._seasonal_decompose_detection(analysis_data)
        elif method == OutlierMethod.PERCENTAGE_CHANGE:
            return self._percentage_change_detection(analysis_data)
        else:
            raise ValueError(f"Unknown outlier detection method: {method.value}")
    
    def _statistical_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Statistical outlier detection using Z-score and IQR."""
        outliers = []
        target_series = analysis_data['target_series']
        
        # Z-score method
        mean_val = target_series.mean()
        std_val = target_series.std()
        
        if std_val > 0:
            z_scores = np.abs((target_series - mean_val) / std_val)
            z_outliers = target_series[z_scores > self.config.z_score_threshold]
            
            for idx, value in z_outliers.items():
                outliers.append(OutlierInfo(
                    index=target_series.index.get_loc(idx),
                    value=value,
                    method='z_score',
                    score=z_scores[idx],
                    severity=self._classify_severity(z_scores[idx], self.config.z_score_threshold)
                ))
        
        # IQR method
        Q1 = target_series.quantile(0.25)
        Q3 = target_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.config.iqr_multiplier * IQR
        upper_bound = Q3 + self.config.iqr_multiplier * IQR
        
        iqr_outliers = target_series[(target_series < lower_bound) | (target_series > upper_bound)]
        
        for idx, value in iqr_outliers.items():
            # Avoid duplicates
            if not any(o.index == target_series.index.get_loc(idx) for o in outliers):
                distance = min(abs(value - lower_bound), abs(value - upper_bound))
                score = distance / IQR if IQR > 0 else 0
                
                outliers.append(OutlierInfo(
                    index=target_series.index.get_loc(idx),
                    value=value,
                    method='iqr',
                    score=score,
                    severity=self._classify_severity(score, 1.0)
                ))
        
        return outliers
    
    async def _isolation_forest_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Isolation Forest outlier detection."""
        outliers = []
        target_series = analysis_data['target_series']
        
        # Prepare features
        features = []
        feature_names = []
        
        # Add target values
        features.append(target_series.values.reshape(-1, 1))
        feature_names.append('target')
        
        # Add returns if available
        if len(analysis_data['returns']) > 0:
            returns_aligned = analysis_data['returns'].reindex(target_series.index, fill_value=0)
            features.append(returns_aligned.values.reshape(-1, 1))
            feature_names.append('returns')
        
        # Add volume if available
        if analysis_data['volume_series'] is not None:
            volume_aligned = analysis_data['volume_series'].reindex(target_series.index, fill_value=0)
            features.append(volume_aligned.values.reshape(-1, 1))
            feature_names.append('volume')
        
        # Add rolling statistics
        if analysis_data['rolling_std'] is not None:
            rolling_std_aligned = analysis_data['rolling_std'].reindex(target_series.index, fill_value=0)
            features.append(rolling_std_aligned.values.reshape(-1, 1))
            feature_names.append('rolling_std')
        
        # Combine features
        if len(features) > 1:
            X = np.hstack(features)
        else:
            X = features[0]
        
        # Handle NaN values
        valid_mask = ~np.any(np.isnan(X), axis=1)
        X_clean = X[valid_mask]
        valid_indices = target_series.index[valid_mask]
        
        if len(X_clean) < self.config.min_data_points:
            return []
        
        # Initialize and fit Isolation Forest
        if self.isolation_forest is None:
            self.isolation_forest = IsolationForest(
                contamination=self.config.contamination,
                random_state=42
            )
        
        # Fit and predict
        outlier_labels = self.isolation_forest.fit_predict(X_clean)
        outlier_scores = self.isolation_forest.score_samples(X_clean)
        
        # Extract outliers
        outlier_indices = np.where(outlier_labels == -1)[0]
        
        for idx in outlier_indices:
            original_idx = valid_indices[idx]
            outliers.append(OutlierInfo(
                index=target_series.index.get_loc(original_idx),
                value=target_series[original_idx],
                method='isolation_forest',
                score=abs(outlier_scores[idx]),
                severity=self._classify_severity(abs(outlier_scores[idx]), 0.5)
            ))
        
        return outliers
    
    def _modified_z_score_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Modified Z-score detection using Median Absolute Deviation (MAD)."""
        outliers = []
        target_series = analysis_data['target_series']
        
        # Calculate MAD
        median_val = target_series.median()
        mad = np.median(np.abs(target_series - median_val))
        
        if mad > 0:
            # Modified Z-score
            modified_z_scores = 0.6745 * (target_series - median_val) / mad
            abs_modified_z = np.abs(modified_z_scores)
            
            outlier_mask = abs_modified_z > self.config.modified_z_threshold
            outlier_series = target_series[outlier_mask]
            outlier_scores = abs_modified_z[outlier_mask]
            
            for idx, value in outlier_series.items():
                outliers.append(OutlierInfo(
                    index=target_series.index.get_loc(idx),
                    value=value,
                    method='modified_z_score',
                    score=outlier_scores[idx],
                    severity=self._classify_severity(outlier_scores[idx], self.config.modified_z_threshold)
                ))
        
        return outliers
    
    def _grubbs_test_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Grubbs test for outlier detection."""
        outliers = []
        target_series = analysis_data['target_series'].dropna()
        
        if len(target_series) < 3:
            return outliers
        
        # Grubbs test statistic
        mean_val = target_series.mean()
        std_val = target_series.std()
        
        if std_val > 0:
            # Test for maximum value
            max_deviation = np.abs(target_series - mean_val).max()
            grubbs_stat = max_deviation / std_val
            
            # Critical value (simplified, should use proper statistical tables)
            n = len(target_series)
            critical_value = (n - 1) / np.sqrt(n) * np.sqrt(stats.t.ppf(1 - 0.05/(2*n), n-2)**2 / (n - 2 + stats.t.ppf(1 - 0.05/(2*n), n-2)**2))
            
            if grubbs_stat > critical_value:
                # Find the most extreme value
                extreme_idx = np.abs(target_series - mean_val).idxmax()
                outliers.append(OutlierInfo(
                    index=target_series.index.get_loc(extreme_idx),
                    value=target_series[extreme_idx],
                    method='grubbs_test',
                    score=grubbs_stat,
                    severity=self._classify_severity(grubbs_stat, critical_value)
                ))
        
        return outliers
    
    def _contextual_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Contextual outlier detection considering market conditions."""
        outliers = []
        target_series = analysis_data['target_series']
        returns = analysis_data['returns']
        rolling_std = analysis_data['rolling_std']
        
        # Volatility-adjusted detection
        if rolling_std is not None and len(returns) > 0:
            # Calculate dynamic thresholds based on volatility
            volatility_multiplier = rolling_std / rolling_std.mean()
            adjusted_thresholds = self.config.z_score_threshold * volatility_multiplier
            
            # Detect outliers in returns space
            returns_aligned = returns.reindex(target_series.index, fill_value=0)
            
            for idx in target_series.index:
                if idx in returns_aligned.index and idx in adjusted_thresholds.index:
                    return_val = abs(returns_aligned[idx])
                    threshold = adjusted_thresholds[idx] * returns.std()
                    
                    if return_val > threshold:
                        outliers.append(OutlierInfo(
                            index=target_series.index.get_loc(idx),
                            value=target_series[idx],
                            method='contextual',
                            score=return_val / threshold,
                            severity=self._classify_severity(return_val / threshold, 1.0),
                            context={
                                'volatility_regime': 'high' if volatility_multiplier[idx] > 1.5 else 'normal',
                                'return_value': return_val,
                                'adjusted_threshold': threshold
                            }
                        ))
        
        return outliers
    
    async def _ensemble_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Ensemble outlier detection combining multiple methods."""
        all_outliers = []
        
        # Run multiple detection methods
        methods_to_run = [
            OutlierMethod.STATISTICAL,
            OutlierMethod.MODIFIED_Z_SCORE,
            OutlierMethod.CONTEXTUAL
        ]
        
        for method in methods_to_run:
            method_outliers = await self._detect_outliers(analysis_data, method)
            all_outliers.extend(method_outliers)
        
        # Aggregate results - consensus approach
        outlier_votes = {}
        for outlier in all_outliers:
            key = outlier.index
            if key not in outlier_votes:
                outlier_votes[key] = {'outliers': [], 'votes': 0}
            outlier_votes[key]['outliers'].append(outlier)
            outlier_votes[key]['votes'] += 1
        
        # Keep outliers with multiple votes
        consensus_outliers = []
        min_votes = max(1, len(methods_to_run) // 2)  # Majority consensus
        
        for key, vote_info in outlier_votes.items():
            if vote_info['votes'] >= min_votes:
                # Take the outlier with highest score
                best_outlier = max(vote_info['outliers'], key=lambda x: x.score)
                best_outlier.method = 'ensemble'
                best_outlier.context = {'votes': vote_info['votes'], 'methods': [o.method for o in vote_info['outliers']]}
                consensus_outliers.append(best_outlier)
        
        return consensus_outliers
    
    def _seasonal_decompose_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Seasonal decomposition-based outlier detection."""
        outliers = []
        target_series = analysis_data['target_series']
        
        # Simple seasonal decomposition
        if len(target_series) >= 48:  # Need enough data for patterns
            # Use rolling statistics to approximate trend
            trend = target_series.rolling(window=24, center=True).mean()
            detrended = target_series - trend
            
            # Detect outliers in detrended series
            if len(detrended.dropna()) > 0:
                residuals = detrended.dropna()
                threshold = residuals.std() * self.config.z_score_threshold
                
                outlier_mask = np.abs(residuals) > threshold
                outlier_series = residuals[outlier_mask]
                
                for idx, value in outlier_series.items():
                    outliers.append(OutlierInfo(
                        index=target_series.index.get_loc(idx),
                        value=target_series[idx],
                        method='seasonal_decompose',
                        score=abs(value) / threshold,
                        severity=self._classify_severity(abs(value) / threshold, 1.0),
                        context={'residual_value': value, 'threshold': threshold}
                    ))
        
        return outliers
    
    def _percentage_change_detection(self, analysis_data: Dict[str, Any]) -> List[OutlierInfo]:
        """Outlier detection based on percentage changes."""
        outliers = []
        returns = analysis_data['returns']
        target_series = analysis_data['target_series']
        
        if len(returns) > 0:
            # Daily change threshold
            daily_outliers = returns[np.abs(returns) > self.config.max_daily_change]
            
            for idx, return_val in daily_outliers.items():
                if idx in target_series.index:
                    outliers.append(OutlierInfo(
                        index=target_series.index.get_loc(idx),
                        value=target_series[idx],
                        method='percentage_change',
                        score=abs(return_val) / self.config.max_daily_change,
                        severity=self._classify_severity(abs(return_val) / self.config.max_daily_change, 1.0),
                        context={'return_value': return_val, 'threshold': self.config.max_daily_change}
                    ))
        
        return outliers
    
    def _classify_severity(self, score: float, threshold: float) -> str:
        """Classify outlier severity based on score."""
        ratio = score / threshold if threshold > 0 else score
        
        if ratio <= 1.2:
            return 'mild'
        elif ratio <= 2.0:
            return 'moderate'
        else:
            return 'severe'
    
    def _validate_outliers(
        self,
        outliers: List[OutlierInfo],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate detected outliers."""
        errors = []
        warnings = []
        
        # Check outlier ratio
        total_points = len(analysis_data['target_series'])
        outlier_ratio = len(outliers) / total_points
        
        if outlier_ratio > self.config.max_outlier_ratio:
            errors.append(
                f"Too many outliers detected: {outlier_ratio:.1%} > {self.config.max_outlier_ratio:.1%}"
            )
        
        # Check for suspicious patterns
        if len(outliers) > 0:
            outlier_indices = [o.index for o in outliers]
            consecutive_outliers = self._count_consecutive_outliers(outlier_indices)
            
            if consecutive_outliers > 5:
                warnings.append(f"Many consecutive outliers detected: {consecutive_outliers}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _count_consecutive_outliers(self, indices: List[int]) -> int:
        """Count maximum consecutive outliers."""
        if not indices:
            return 0
        
        sorted_indices = sorted(indices)
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive
    
    async def _handle_outliers(
        self,
        data: pd.DataFrame,
        outliers: List[OutlierInfo],
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handle detected outliers according to configuration."""
        handling_info = {
            'handled_count': 0,
            'warnings': []
        }
        
        if not outliers:
            return data, handling_info
        
        target_series = data[target_column].copy()
        
        for outlier in outliers:
            idx = target_series.index[outlier.index]
            original_value = outlier.value
            
            if self.config.outlier_action == OutlierAction.REMOVE:
                # Remove the outlier (set to NaN)
                data.loc[idx, target_column] = np.nan
                if feature_columns:
                    for col in feature_columns:
                        if col in data.columns:
                            data.loc[idx, col] = np.nan
                handling_info['handled_count'] += 1
            
            elif self.config.outlier_action == OutlierAction.CAP:
                # Cap at reasonable bounds
                bounds = self._calculate_capping_bounds(target_series, outlier)
                capped_value = np.clip(original_value, bounds['lower'], bounds['upper'])
                data.loc[idx, target_column] = capped_value
                handling_info['handled_count'] += 1
            
            elif self.config.outlier_action == OutlierAction.REPLACE_MEDIAN:
                # Replace with rolling median
                rolling_median = target_series.rolling(window=self.config.rolling_window, center=True).median()
                if not pd.isna(rolling_median.loc[idx]):
                    data.loc[idx, target_column] = rolling_median.loc[idx]
                    handling_info['handled_count'] += 1
            
            elif self.config.outlier_action == OutlierAction.REPLACE_INTERPOLATE:
                # Mark for interpolation
                data.loc[idx, target_column] = np.nan
                handling_info['handled_count'] += 1
                # Interpolate
                data[target_column] = data[target_column].interpolate(method='linear', limit=1)
            
            elif self.config.outlier_action == OutlierAction.FLAG_ONLY:
                # Just add a flag column
                if 'outlier_flag' not in data.columns:
                    data['outlier_flag'] = False
                data.loc[idx, 'outlier_flag'] = True
        
        # Apply interpolation if requested
        if self.config.outlier_action == OutlierAction.REPLACE_INTERPOLATE:
            data[target_column] = data[target_column].interpolate(method='linear')
        
        return data, handling_info
    
    def _calculate_capping_bounds(
        self,
        series: pd.Series,
        outlier: OutlierInfo
    ) -> Dict[str, float]:
        """Calculate bounds for capping outliers."""
        # Use rolling statistics around the outlier
        window_start = max(0, outlier.index - self.config.rolling_window // 2)
        window_end = min(len(series), outlier.index + self.config.rolling_window // 2)
        
        window_data = series.iloc[window_start:window_end]
        
        # Calculate bounds based on percentiles
        lower_bound = window_data.quantile(0.05)
        upper_bound = window_data.quantile(0.95)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound
        }
    
    def _get_market_hours_mask(self, datetime_index: pd.DatetimeIndex) -> pd.Series:
        """Get mask for market hours (9:30 AM - 4:00 PM EST)."""
        # Convert to Eastern Time
        et_index = datetime_index.tz_convert('US/Eastern') if datetime_index.tz is not None else datetime_index
        
        # Market hours mask
        mask = (
            (et_index.time >= pd.Timestamp('09:30:00').time()) &
            (et_index.time <= pd.Timestamp('16:00:00').time()) &
            (et_index.weekday < 5)  # Monday=0, Friday=4
        )
        
        return pd.Series(mask, index=datetime_index)
    
    def _calculate_quality_metrics(
        self,
        original_data: pd.DataFrame,
        processed_data: pd.DataFrame,
        outliers: List[OutlierInfo],
        analysis_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate quality metrics after outlier handling."""
        target_series = analysis_data['target_series']
        
        metrics = {
            'outlier_ratio': len(outliers) / len(target_series),
            'severe_outlier_ratio': sum(1 for o in outliers if o.severity == 'severe') / len(target_series),
            'data_completeness': 1 - processed_data.isna().sum().sum() / processed_data.size,
            'outliers_handled': len(outliers)
        }
        
        # Calculate stability metrics
        if len(outliers) > 0:
            severity_scores = [o.score for o in outliers]
            metrics['avg_outlier_score'] = np.mean(severity_scores)
            metrics['max_outlier_score'] = np.max(severity_scores)
        else:
            metrics['avg_outlier_score'] = 0
            metrics['max_outlier_score'] = 0
        
        # Overall quality score
        metrics['data_quality'] = (
            0.4 * (1 - metrics['outlier_ratio']) +
            0.3 * metrics['data_completeness'] +
            0.3 * (1 - metrics['severe_outlier_ratio'])
        )
        
        return metrics
    
    def _update_statistics(
        self,
        method: OutlierMethod,
        outliers: List[OutlierInfo],
        quality_metrics: Dict[str, float]
    ):
        """Update detection statistics."""
        self.detection_stats['total_processed'] += 1
        self.detection_stats['total_outliers_detected'] += len(outliers)
        self.detection_stats['outliers_by_method'][method.value] += len(outliers)
        
        # Track detection accuracy (simplified)
        if quality_metrics['data_quality'] > 0.8:
            self.detection_stats['detection_accuracy'].append(1.0)
        else:
            self.detection_stats['detection_accuracy'].append(0.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        stats = self.detection_stats.copy()
        
        if stats['detection_accuracy']:
            stats['avg_accuracy'] = np.mean(stats['detection_accuracy'])
        else:
            stats['avg_accuracy'] = 0
        
        return stats
    
    def get_config(self) -> OutlierConfig:
        """Get current configuration."""
        return self.config