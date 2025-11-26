"""
Missing Value Imputer for Time Series Data

This module provides advanced missing value imputation techniques specifically
designed for time series data used in neural forecasting models.

Key Features:
- Multiple imputation strategies
- Time-aware interpolation methods
- Seasonal pattern preservation
- Quality-driven method selection
- Performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import interpolate
from sklearn.impute import KNNImputer
import warnings

logger = logging.getLogger(__name__)


class ImputationStrategy(Enum):
    """Available imputation strategies."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATION = "linear_interpolation"
    SPLINE_INTERPOLATION = "spline_interpolation"
    SEASONAL_DECOMPOSE = "seasonal_decompose"
    KNN_IMPUTATION = "knn_imputation"
    KALMAN_FILTER = "kalman_filter"
    MEAN_IMPUTATION = "mean_imputation"
    MEDIAN_IMPUTATION = "median_imputation"
    AUTO_SELECT = "auto_select"


@dataclass
class ImputationConfig:
    """Configuration for missing value imputation."""
    # Primary strategy
    strategy: ImputationStrategy = ImputationStrategy.AUTO_SELECT
    
    # Fallback strategies (in order of preference)
    fallback_strategies: List[ImputationStrategy] = field(
        default_factory=lambda: [
            ImputationStrategy.LINEAR_INTERPOLATION,
            ImputationStrategy.FORWARD_FILL,
            ImputationStrategy.BACKWARD_FILL
        ]
    )
    
    # Gap handling
    max_gap_size: int = 12  # Maximum gap size to impute (in time steps)
    min_valid_ratio: float = 0.8  # Minimum ratio of valid data required
    
    # Method-specific parameters
    knn_neighbors: int = 5
    spline_order: int = 3
    seasonal_periods: List[int] = field(default_factory=lambda: [24, 168])  # Daily, weekly
    
    # Quality thresholds
    max_imputation_ratio: float = 0.2  # Maximum 20% imputation per column
    quality_threshold: float = 0.85  # Minimum quality score for imputation
    
    # Performance settings
    use_parallel: bool = True
    chunk_size: int = 10000


@dataclass
class ImputationResult:
    """Result of missing value imputation."""
    success: bool
    data: Optional[pd.DataFrame] = None
    imputation_info: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class GapInfo:
    """Information about missing data gaps."""
    start_idx: int
    end_idx: int
    size: int
    column: str
    severity: str  # 'small', 'medium', 'large', 'extreme'


class MissingValueImputer:
    """
    Advanced missing value imputer for time series data.
    
    This class provides multiple sophisticated imputation strategies
    specifically designed for financial time series data, with automatic
    strategy selection based on data characteristics and gap patterns.
    """
    
    def __init__(self, config: ImputationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize method-specific components
        self.knn_imputer = None
        
        # Statistics tracking
        self.imputation_stats = {
            'total_processed': 0,
            'total_gaps_filled': 0,
            'strategy_usage': {strategy.value: 0 for strategy in ImputationStrategy},
            'quality_scores': []
        }
        
        self.logger.info(f"MissingValueImputer initialized with strategy: {config.strategy.value}")
    
    async def impute(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> ImputationResult:
        """
        Impute missing values in time series data.
        
        Args:
            data: Input DataFrame with missing values
            target_column: Primary target column for forecasting
            feature_columns: Additional feature columns to impute
        
        Returns:
            ImputationResult with imputed data and metadata
        """
        try:
            self.logger.info(f"Starting imputation for {len(data)} samples")
            
            # Analyze missing data patterns
            missing_analysis = self._analyze_missing_patterns(data, target_column, feature_columns)
            
            if missing_analysis['total_missing'] == 0:
                self.logger.info("No missing values found")
                return ImputationResult(
                    success=True,
                    data=data.copy(),
                    imputation_info={'strategy_used': 'none', 'gaps_filled': 0},
                    quality_metrics={'completeness': 1.0}
                )
            
            # Check if imputation is feasible
            feasibility_check = self._check_imputation_feasibility(missing_analysis)
            if not feasibility_check['feasible']:
                return ImputationResult(
                    success=False,
                    errors=feasibility_check['errors']
                )
            
            # Select imputation strategy
            selected_strategy = self._select_strategy(missing_analysis)
            
            # Perform imputation
            imputation_result = await self._perform_imputation(
                data, target_column, feature_columns, selected_strategy, missing_analysis
            )
            
            if not imputation_result['success']:
                # Try fallback strategies
                for fallback_strategy in self.config.fallback_strategies:
                    if fallback_strategy == selected_strategy:
                        continue
                    
                    self.logger.warning(f"Trying fallback strategy: {fallback_strategy.value}")
                    imputation_result = await self._perform_imputation(
                        data, target_column, feature_columns, fallback_strategy, missing_analysis
                    )
                    
                    if imputation_result['success']:
                        break
            
            if not imputation_result['success']:
                return ImputationResult(
                    success=False,
                    errors=imputation_result['errors']
                )
            
            # Validate imputation quality
            quality_metrics = self._assess_imputation_quality(
                data, imputation_result['data'], missing_analysis
            )
            
            # Update statistics
            self._update_statistics(selected_strategy, missing_analysis, quality_metrics)
            
            return ImputationResult(
                success=True,
                data=imputation_result['data'],
                imputation_info=imputation_result['info'],
                quality_metrics=quality_metrics,
                warnings=imputation_result.get('warnings', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error during imputation: {e}")
            return ImputationResult(
                success=False,
                errors=[str(e)]
            )
    
    def _analyze_missing_patterns(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analyze patterns in missing data."""
        columns_to_analyze = [target_column]
        if feature_columns:
            columns_to_analyze.extend(feature_columns)
        
        missing_info = {
            'total_samples': len(data),
            'total_missing': 0,
            'missing_by_column': {},
            'gaps': [],
            'largest_gap': 0,
            'missing_ratio': 0,
            'consecutive_patterns': {},
            'has_seasonal_gaps': False
        }
        
        for column in columns_to_analyze:
            if column not in data.columns:
                continue
            
            col_missing = data[column].isna()
            missing_count = col_missing.sum()
            missing_info['total_missing'] += missing_count
            missing_info['missing_by_column'][column] = {
                'count': missing_count,
                'ratio': missing_count / len(data),
                'gaps': []
            }
            
            # Find consecutive gaps
            gaps = self._find_consecutive_gaps(col_missing, column)
            missing_info['missing_by_column'][column]['gaps'] = gaps
            missing_info['gaps'].extend(gaps)
            
            if gaps:
                largest_gap_size = max(gap.size for gap in gaps)
                missing_info['largest_gap'] = max(missing_info['largest_gap'], largest_gap_size)
        
        missing_info['missing_ratio'] = missing_info['total_missing'] / (len(data) * len(columns_to_analyze))
        
        # Detect seasonal patterns in missing data
        missing_info['has_seasonal_gaps'] = self._detect_seasonal_missing_patterns(missing_info['gaps'])
        
        return missing_info
    
    def _find_consecutive_gaps(self, missing_mask: pd.Series, column: str) -> List[GapInfo]:
        """Find consecutive missing value gaps."""
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, is_missing in enumerate(missing_mask):
            if is_missing and not in_gap:
                # Start of new gap
                in_gap = True
                gap_start = i
            elif not is_missing and in_gap:
                # End of gap
                gap_size = i - gap_start
                severity = self._classify_gap_severity(gap_size)
                gaps.append(GapInfo(gap_start, i - 1, gap_size, column, severity))
                in_gap = False
        
        # Handle gap that extends to end of data
        if in_gap:
            gap_size = len(missing_mask) - gap_start
            severity = self._classify_gap_severity(gap_size)
            gaps.append(GapInfo(gap_start, len(missing_mask) - 1, gap_size, column, severity))
        
        return gaps
    
    def _classify_gap_severity(self, gap_size: int) -> str:
        """Classify gap severity based on size."""
        if gap_size <= 3:
            return 'small'
        elif gap_size <= self.config.max_gap_size:
            return 'medium'
        elif gap_size <= self.config.max_gap_size * 2:
            return 'large'
        else:
            return 'extreme'
    
    def _detect_seasonal_missing_patterns(self, gaps: List[GapInfo]) -> bool:
        """Detect if missing data follows seasonal patterns."""
        if len(gaps) < 3:
            return False
        
        # Check for regular intervals between gaps
        gap_intervals = []
        for i in range(1, len(gaps)):
            interval = gaps[i].start_idx - gaps[i-1].end_idx
            gap_intervals.append(interval)
        
        # Look for patterns in intervals
        if len(gap_intervals) >= 2:
            interval_std = np.std(gap_intervals)
            interval_mean = np.mean(gap_intervals)
            # If intervals are relatively consistent, might be seasonal
            return interval_std / interval_mean < 0.3 if interval_mean > 0 else False
        
        return False
    
    def _check_imputation_feasibility(self, missing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check if imputation is feasible given the missing data patterns."""
        errors = []
        
        # Check overall missing ratio
        if missing_analysis['missing_ratio'] > self.config.max_imputation_ratio:
            errors.append(
                f"Too much missing data: {missing_analysis['missing_ratio']:.1%} > "
                f"{self.config.max_imputation_ratio:.1%}"
            )
        
        # Check for extremely large gaps
        if missing_analysis['largest_gap'] > self.config.max_gap_size * 3:
            errors.append(
                f"Gap too large for reliable imputation: {missing_analysis['largest_gap']} samples"
            )
        
        # Check minimum valid data requirement
        valid_ratio = 1 - missing_analysis['missing_ratio']
        if valid_ratio < self.config.min_valid_ratio:
            errors.append(
                f"Insufficient valid data: {valid_ratio:.1%} < {self.config.min_valid_ratio:.1%}"
            )
        
        return {
            'feasible': len(errors) == 0,
            'errors': errors
        }
    
    def _select_strategy(self, missing_analysis: Dict[str, Any]) -> ImputationStrategy:
        """Select the best imputation strategy based on data characteristics."""
        if self.config.strategy != ImputationStrategy.AUTO_SELECT:
            return self.config.strategy
        
        # Auto-select based on missing patterns
        total_gaps = len(missing_analysis['gaps'])
        largest_gap = missing_analysis['largest_gap']
        missing_ratio = missing_analysis['missing_ratio']
        has_seasonal = missing_analysis['has_seasonal_gaps']
        
        # Strategy selection logic
        if missing_ratio < 0.01:  # Very little missing data
            return ImputationStrategy.LINEAR_INTERPOLATION
        
        elif largest_gap <= 3:  # Small gaps
            return ImputationStrategy.LINEAR_INTERPOLATION
        
        elif has_seasonal and total_gaps > 5:  # Seasonal patterns
            return ImputationStrategy.SEASONAL_DECOMPOSE
        
        elif largest_gap <= self.config.max_gap_size and missing_ratio < 0.1:
            return ImputationStrategy.SPLINE_INTERPOLATION
        
        elif missing_ratio < 0.05:  # Low missing ratio
            return ImputationStrategy.KNN_IMPUTATION
        
        else:  # Default fallback
            return ImputationStrategy.FORWARD_FILL
    
    async def _perform_imputation(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]],
        strategy: ImputationStrategy,
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual imputation using the selected strategy."""
        try:
            data_copy = data.copy()
            columns_to_impute = [target_column]
            if feature_columns:
                columns_to_impute.extend([col for col in feature_columns if col in data.columns])
            
            imputation_info = {
                'strategy_used': strategy.value,
                'gaps_filled': 0,
                'columns_imputed': [],
                'imputation_details': {}
            }
            
            if strategy == ImputationStrategy.FORWARD_FILL:
                result = self._forward_fill_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.BACKWARD_FILL:
                result = self._backward_fill_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.LINEAR_INTERPOLATION:
                result = self._linear_interpolation_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.SPLINE_INTERPOLATION:
                result = self._spline_interpolation_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.SEASONAL_DECOMPOSE:
                result = self._seasonal_decompose_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.KNN_IMPUTATION:
                result = await self._knn_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.MEAN_IMPUTATION:
                result = self._mean_imputation(data_copy, columns_to_impute, missing_analysis)
            
            elif strategy == ImputationStrategy.MEDIAN_IMPUTATION:
                result = self._median_imputation(data_copy, columns_to_impute, missing_analysis)
            
            else:
                return {
                    'success': False,
                    'errors': [f"Unknown imputation strategy: {strategy.value}"]
                }
            
            if result['success']:
                imputation_info.update(result['info'])
                return {
                    'success': True,
                    'data': result['data'],
                    'info': imputation_info,
                    'warnings': result.get('warnings', [])
                }
            else:
                return {
                    'success': False,
                    'errors': result['errors']
                }
                
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Imputation failed: {str(e)}"]
            }
    
    def _forward_fill_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward fill imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                # Apply forward fill with limit
                data[column] = data[column].fillna(method='ffill', limit=self.config.max_gap_size)
                
                final_missing = data[column].isna().sum()
                filled_count = initial_missing - final_missing
                
                if filled_count > 0:
                    gaps_filled += filled_count
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Forward fill failed: {str(e)}"]
            }
    
    def _backward_fill_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Backward fill imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                # Apply backward fill with limit
                data[column] = data[column].fillna(method='bfill', limit=self.config.max_gap_size)
                
                final_missing = data[column].isna().sum()
                filled_count = initial_missing - final_missing
                
                if filled_count > 0:
                    gaps_filled += filled_count
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Backward fill failed: {str(e)}"]
            }
    
    def _linear_interpolation_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Linear interpolation imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            warnings = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                # Check for large gaps
                column_gaps = missing_analysis['missing_by_column'].get(column, {}).get('gaps', [])
                large_gaps = [gap for gap in column_gaps if gap.size > self.config.max_gap_size]
                
                if large_gaps:
                    warnings.append(f"Large gaps in {column} may reduce interpolation quality")
                
                # Apply linear interpolation with limit
                data[column] = data[column].interpolate(
                    method='linear',
                    limit=self.config.max_gap_size,
                    limit_direction='both'
                )
                
                final_missing = data[column].isna().sum()
                filled_count = initial_missing - final_missing
                
                if filled_count > 0:
                    gaps_filled += filled_count
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                },
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Linear interpolation failed: {str(e)}"]
            }
    
    def _spline_interpolation_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Spline interpolation imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            warnings = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                # Spline interpolation requires sufficient data points
                valid_data = data[column].dropna()
                if len(valid_data) < self.config.spline_order + 1:
                    warnings.append(f"Insufficient data for spline interpolation in {column}")
                    # Fallback to linear
                    data[column] = data[column].interpolate(method='linear', limit=self.config.max_gap_size)
                else:
                    data[column] = data[column].interpolate(
                        method='spline',
                        order=self.config.spline_order,
                        limit=self.config.max_gap_size
                    )
                
                final_missing = data[column].isna().sum()
                filled_count = initial_missing - final_missing
                
                if filled_count > 0:
                    gaps_filled += filled_count
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                },
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Spline interpolation failed: {str(e)}"]
            }
    
    def _seasonal_decompose_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Seasonal decomposition-based imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            warnings = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                # Seasonal imputation requires sufficient data
                if len(data) < max(self.config.seasonal_periods) * 2:
                    warnings.append(f"Insufficient data for seasonal imputation in {column}")
                    # Fallback to linear interpolation
                    data[column] = data[column].interpolate(method='linear', limit=self.config.max_gap_size)
                else:
                    # Use seasonal patterns to inform imputation
                    # This is a simplified implementation
                    for period in self.config.seasonal_periods:
                        if len(data) >= period * 2:
                            # Fill using same-season values
                            seasonal_data = data[column].copy()
                            for i in range(len(seasonal_data)):
                                if pd.isna(seasonal_data.iloc[i]):
                                    # Look for similar seasonal positions
                                    seasonal_indices = list(range(i % period, len(seasonal_data), period))
                                    seasonal_values = [seasonal_data.iloc[idx] for idx in seasonal_indices 
                                                     if not pd.isna(seasonal_data.iloc[idx])]
                                    if seasonal_values:
                                        seasonal_data.iloc[i] = np.mean(seasonal_values)
                            
                            data[column] = seasonal_data
                            break
                
                final_missing = data[column].isna().sum()
                filled_count = initial_missing - final_missing
                
                if filled_count > 0:
                    gaps_filled += filled_count
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                },
                'warnings': warnings
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Seasonal imputation failed: {str(e)}"]
            }
    
    async def _knn_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """KNN-based imputation."""
        try:
            # Initialize KNN imputer if not already done
            if self.knn_imputer is None:
                self.knn_imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            
            gaps_filled = 0
            columns_imputed = []
            
            # Only use numeric columns for KNN
            numeric_columns = [col for col in columns if col in data.columns and 
                             pd.api.types.is_numeric_dtype(data[col])]
            
            if not numeric_columns:
                return {
                    'success': False,
                    'errors': ['No numeric columns available for KNN imputation']
                }
            
            initial_missing = data[numeric_columns].isna().sum().sum()
            
            # Apply KNN imputation
            imputed_data = self.knn_imputer.fit_transform(data[numeric_columns])
            data[numeric_columns] = imputed_data
            
            final_missing = data[numeric_columns].isna().sum().sum()
            gaps_filled = initial_missing - final_missing
            columns_imputed = numeric_columns
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"KNN imputation failed: {str(e)}"]
            }
    
    def _mean_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mean imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                if initial_missing > 0:
                    column_mean = data[column].mean()
                    data[column] = data[column].fillna(column_mean)
                    
                    gaps_filled += initial_missing
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Mean imputation failed: {str(e)}"]
            }
    
    def _median_imputation(
        self,
        data: pd.DataFrame,
        columns: List[str],
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Median imputation."""
        try:
            gaps_filled = 0
            columns_imputed = []
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                initial_missing = data[column].isna().sum()
                
                if initial_missing > 0:
                    column_median = data[column].median()
                    data[column] = data[column].fillna(column_median)
                    
                    gaps_filled += initial_missing
                    columns_imputed.append(column)
            
            return {
                'success': True,
                'data': data,
                'info': {
                    'gaps_filled': gaps_filled,
                    'columns_imputed': columns_imputed
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Median imputation failed: {str(e)}"]
            }
    
    def _assess_imputation_quality(
        self,
        original_data: pd.DataFrame,
        imputed_data: pd.DataFrame,
        missing_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Assess the quality of imputation."""
        quality_metrics = {}
        
        # Overall completeness
        total_elements = imputed_data.size
        missing_elements = imputed_data.isna().sum().sum()
        quality_metrics['completeness'] = 1 - (missing_elements / total_elements)
        
        # Imputation ratio
        original_missing = original_data.isna().sum().sum()
        imputed_missing = imputed_data.isna().sum().sum()
        filled_count = original_missing - imputed_missing
        quality_metrics['imputation_ratio'] = filled_count / original_missing if original_missing > 0 else 0
        
        # Continuity assessment (check for abrupt changes at imputation points)
        continuity_scores = []
        for column in imputed_data.select_dtypes(include=[np.number]).columns:
            if column in original_data.columns:
                continuity_score = self._assess_continuity(
                    original_data[column], imputed_data[column]
                )
                continuity_scores.append(continuity_score)
        
        quality_metrics['continuity'] = np.mean(continuity_scores) if continuity_scores else 0
        
        # Overall quality score
        quality_metrics['overall_quality'] = (
            0.4 * quality_metrics['completeness'] +
            0.3 * quality_metrics['imputation_ratio'] +
            0.3 * quality_metrics['continuity']
        )
        
        return quality_metrics
    
    def _assess_continuity(self, original: pd.Series, imputed: pd.Series) -> float:
        """Assess continuity of imputed values."""
        try:
            # Find imputed positions
            imputed_mask = original.isna() & ~imputed.isna()
            imputed_positions = imputed_mask[imputed_mask].index
            
            if len(imputed_positions) == 0:
                return 1.0  # No imputation, perfect continuity
            
            continuity_scores = []
            
            for pos in imputed_positions:
                pos_idx = imputed.index.get_loc(pos)
                
                # Check neighbors
                neighbors = []
                if pos_idx > 0:
                    neighbors.append(imputed.iloc[pos_idx - 1])
                if pos_idx < len(imputed) - 1:
                    neighbors.append(imputed.iloc[pos_idx + 1])
                
                if neighbors:
                    imputed_value = imputed.iloc[pos_idx]
                    neighbor_mean = np.mean(neighbors)
                    
                    # Calculate relative difference
                    if neighbor_mean != 0:
                        relative_diff = abs(imputed_value - neighbor_mean) / abs(neighbor_mean)
                        continuity_score = max(0, 1 - relative_diff)
                    else:
                        continuity_score = 1.0 if imputed_value == 0 else 0.0
                    
                    continuity_scores.append(continuity_score)
            
            return np.mean(continuity_scores) if continuity_scores else 1.0
            
        except Exception:
            return 0.5  # Neutral score if assessment fails
    
    def _update_statistics(
        self,
        strategy: ImputationStrategy,
        missing_analysis: Dict[str, Any],
        quality_metrics: Dict[str, float]
    ):
        """Update imputation statistics."""
        self.imputation_stats['total_processed'] += 1
        self.imputation_stats['total_gaps_filled'] += len(missing_analysis['gaps'])
        self.imputation_stats['strategy_usage'][strategy.value] += 1
        self.imputation_stats['quality_scores'].append(quality_metrics['overall_quality'])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get imputation statistics."""
        stats = self.imputation_stats.copy()
        
        if stats['quality_scores']:
            stats['avg_quality'] = np.mean(stats['quality_scores'])
            stats['min_quality'] = np.min(stats['quality_scores'])
            stats['max_quality'] = np.max(stats['quality_scores'])
        else:
            stats['avg_quality'] = 0
            stats['min_quality'] = 0
            stats['max_quality'] = 0
        
        return stats