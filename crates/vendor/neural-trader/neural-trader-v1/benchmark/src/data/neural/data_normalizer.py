"""
Data Normalizer for Neural Networks

This module provides advanced data normalization and scaling techniques
specifically optimized for neural time series forecasting models.

Key Features:
- Multiple normalization methods
- Robust scaling techniques
- Time-aware normalization
- Invertible transformations
- Financial data specific scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    QuantileTransformer, PowerTransformer
)
import pickle
import warnings

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Available normalization methods."""
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"      # Min-max scaling
    ROBUST = "robust"      # Robust scaling using median and IQR
    QUANTILE = "quantile"  # Quantile transformation
    POWER = "power"        # Power transformation (Yeo-Johnson)
    UNIT_VECTOR = "unit_vector"  # Unit vector scaling
    FINANCIAL = "financial"      # Financial data specific scaling
    ROLLING = "rolling"          # Rolling window normalization
    AUTO_SELECT = "auto_select"   # Automatic method selection


@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    # Primary method
    method: NormalizationMethod = NormalizationMethod.ROBUST
    
    # Method-specific parameters
    feature_range: Tuple[float, float] = (-1, 1)  # For MinMax scaling
    quantile_range: Tuple[float, float] = (25.0, 75.0)  # For Robust scaling
    n_quantiles: int = 1000  # For Quantile transformation
    output_distribution: str = 'normal'  # For Quantile transformation
    
    # Rolling normalization parameters
    rolling_window: int = 250  # Rolling window size
    min_periods: int = 30     # Minimum periods for rolling
    
    # Feature-specific settings
    per_feature_scaling: bool = True
    scale_target_separately: bool = True
    normalize_features: bool = True
    normalize_target: bool = True
    
    # Robust outlier handling
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    cap_outliers: bool = True
    
    # Financial data specific
    use_log_returns: bool = True
    volatility_adjustment: bool = True
    market_regime_adjustment: bool = False
    
    # Performance settings
    store_scalers: bool = True
    enable_inverse: bool = True


@dataclass
class NormalizationResult:
    """Result of data normalization."""
    success: bool
    data: Optional[pd.DataFrame] = None
    scaler_params: Dict[str, Any] = field(default_factory=dict)
    normalization_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DataNormalizer:
    """
    Advanced data normalizer for neural time series forecasting.
    
    This class provides sophisticated normalization techniques specifically
    designed for financial time series data, ensuring optimal performance
    for neural forecasting models while maintaining interpretability.
    """
    
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store fitted scalers for inverse transformation
        self.scalers = {}
        self.scaler_metadata = {}
        
        # Normalization statistics
        self.normalization_stats = {
            'total_normalized': 0,
            'methods_used': {method.value: 0 for method in NormalizationMethod},
            'feature_stats': {},
            'inverse_transforms': 0
        }
        
        self.logger.info(f"DataNormalizer initialized with method: {config.method.value}")
    
    async def normalize(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> NormalizationResult:
        """
        Normalize time series data for neural networks.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Primary target column for forecasting
            feature_columns: Feature columns to normalize
        
        Returns:
            NormalizationResult with normalized data and scaler parameters
        """
        try:
            self.logger.info(f"Normalizing data with {len(data)} samples")
            
            # Validate input
            validation_result = self._validate_input(data, target_column, feature_columns)
            if not validation_result['valid']:
                return NormalizationResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            # Prepare data for normalization
            normalized_data = data.copy()
            warnings = []
            scaler_params = {}
            
            # Determine columns to normalize
            columns_to_normalize = []
            if self.config.normalize_target:
                columns_to_normalize.append(target_column)
            
            if self.config.normalize_features and feature_columns:
                columns_to_normalize.extend([col for col in feature_columns if col in data.columns])
            
            if not columns_to_normalize:
                return NormalizationResult(
                    success=True,
                    data=normalized_data,
                    scaler_params={},
                    normalization_info={'method': 'none', 'columns_normalized': []}
                )
            
            # Select normalization method
            selected_method = self._select_method(data, target_column, feature_columns)
            
            # Apply preprocessing if needed
            preprocessing_result = await self._preprocess_for_normalization(
                normalized_data, target_column, feature_columns
            )
            if preprocessing_result['warnings']:
                warnings.extend(preprocessing_result['warnings'])
            
            # Normalize target column separately if requested
            if self.config.scale_target_separately and target_column in columns_to_normalize:
                target_result = await self._normalize_column(
                    normalized_data, target_column, selected_method, 'target'
                )
                if target_result['success']:
                    normalized_data = target_result['data']
                    scaler_params[target_column] = target_result['scaler_params']
                else:
                    warnings.extend(target_result['warnings'])
            
            # Normalize feature columns
            if feature_columns and self.config.normalize_features:
                if self.config.per_feature_scaling:
                    # Normalize each feature separately
                    for feature in feature_columns:
                        if feature in data.columns and feature != target_column:
                            feature_result = await self._normalize_column(
                                normalized_data, feature, selected_method, 'feature'
                            )
                            if feature_result['success']:
                                normalized_data = feature_result['data']
                                scaler_params[feature] = feature_result['scaler_params']
                            else:
                                warnings.extend(feature_result['warnings'])
                else:
                    # Normalize all features together
                    feature_columns_clean = [col for col in feature_columns 
                                           if col in data.columns and col != target_column]
                    if feature_columns_clean:
                        joint_result = await self._normalize_joint(
                            normalized_data, feature_columns_clean, selected_method
                        )
                        if joint_result['success']:
                            normalized_data = joint_result['data']
                            scaler_params['features_joint'] = joint_result['scaler_params']
                        else:
                            warnings.extend(joint_result['warnings'])
            
            # Post-processing validation
            quality_metrics = self._assess_normalization_quality(data, normalized_data, columns_to_normalize)
            
            # Update statistics
            self._update_statistics(selected_method, columns_to_normalize, quality_metrics)
            
            # Prepare result
            normalization_info = {
                'method': selected_method.value,
                'columns_normalized': columns_to_normalize,
                'quality_metrics': quality_metrics,
                'preprocessing_applied': preprocessing_result.get('applied', False)
            }
            
            return NormalizationResult(
                success=True,
                data=normalized_data,
                scaler_params=scaler_params,
                normalization_info=normalization_info,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error in normalization: {e}")
            return NormalizationResult(
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
        
        if data.empty:
            errors.append("Input data is empty")
        
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found")
        
        if feature_columns:
            missing_features = set(feature_columns) - set(data.columns)
            if missing_features:
                errors.append(f"Missing feature columns: {missing_features}")
        
        # Check for sufficient numeric data
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if target_column not in numeric_columns:
            errors.append(f"Target column '{target_column}' must be numeric")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _select_method(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> NormalizationMethod:
        """Select the best normalization method based on data characteristics."""
        if self.config.method != NormalizationMethod.AUTO_SELECT:
            return self.config.method
        
        # Analyze data characteristics
        target_data = data[target_column].dropna()
        
        # Calculate basic statistics
        skewness = abs(target_data.skew()) if len(target_data) > 3 else 0
        kurtosis = abs(target_data.kurtosis()) if len(target_data) > 4 else 0
        outlier_ratio = self._calculate_outlier_ratio(target_data)
        
        # Method selection logic
        if outlier_ratio > 0.05:  # High outlier ratio
            return NormalizationMethod.ROBUST
        elif skewness > 2 or kurtosis > 7:  # Highly skewed or heavy-tailed
            return NormalizationMethod.QUANTILE
        elif len(target_data) > 1000:  # Large dataset
            return NormalizationMethod.STANDARD
        else:  # Default
            return NormalizationMethod.ROBUST
    
    def _calculate_outlier_ratio(self, data: pd.Series) -> float:
        """Calculate the ratio of outliers in the data."""
        if len(data) < 10:
            return 0.0
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) / len(data)
    
    async def _preprocess_for_normalization(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Preprocess data before normalization."""
        warnings = []
        applied = False
        
        # Log transformation for financial data
        if self.config.use_log_returns and target_column in data.columns:
            target_data = data[target_column]
            if (target_data > 0).all():  # Only if all values are positive
                log_returns = np.log(target_data / target_data.shift(1)).dropna()
                if not log_returns.empty:
                    # Store original prices for reference
                    data[f'{target_column}_original'] = target_data
                    # Replace with log returns (aligned with original index)
                    aligned_returns = log_returns.reindex(data.index, fill_value=0)
                    data[target_column] = aligned_returns
                    applied = True
                    warnings.append("Applied log returns transformation to target")
        
        # Volatility adjustment
        if self.config.volatility_adjustment and len(data) > 50:
            returns = data[target_column].pct_change().dropna()
            if len(returns) > 30:
                rolling_vol = returns.rolling(window=30).std()
                vol_regime = rolling_vol > rolling_vol.median()
                data['volatility_regime'] = vol_regime.reindex(data.index, fill_value=False)
                applied = True
        
        return {
            'warnings': warnings,
            'applied': applied
        }
    
    async def _normalize_column(
        self,
        data: pd.DataFrame,
        column: str,
        method: NormalizationMethod,
        column_type: str
    ) -> Dict[str, Any]:
        """Normalize a single column."""
        try:
            column_data = data[column].dropna()
            
            if len(column_data) < 10:
                return {
                    'success': False,
                    'warnings': [f"Insufficient data for normalizing {column}"]
                }
            
            # Convert to numpy array
            values = column_data.values.reshape(-1, 1)
            
            # Create scaler based on method
            scaler = self._create_scaler(method)
            
            # Fit and transform
            normalized_values = scaler.fit_transform(values)
            
            # Update data
            data.loc[column_data.index, column] = normalized_values.flatten()
            
            # Store scaler
            scaler_key = f"{column}_{method.value}"
            if self.config.store_scalers:
                self.scalers[scaler_key] = scaler
                self.scaler_metadata[scaler_key] = {
                    'column': column,
                    'method': method.value,
                    'column_type': column_type,
                    'fitted_samples': len(column_data),
                    'original_range': (column_data.min(), column_data.max()),
                    'normalized_range': (normalized_values.min(), normalized_values.max())
                }
            
            # Create scaler parameters for return
            scaler_params = {
                'method': method.value,
                'scaler_object': scaler if self.config.enable_inverse else None,
                'fitted_samples': len(column_data),
                'original_stats': {
                    'mean': column_data.mean(),
                    'std': column_data.std(),
                    'min': column_data.min(),
                    'max': column_data.max()
                }
            }
            
            # Add method-specific parameters
            if hasattr(scaler, 'scale_'):
                scaler_params['scale_'] = scaler.scale_
            if hasattr(scaler, 'min_'):
                scaler_params['min_'] = scaler.min_
            if hasattr(scaler, 'data_min_'):
                scaler_params['data_min_'] = scaler.data_min_
            if hasattr(scaler, 'data_max_'):
                scaler_params['data_max_'] = scaler.data_max_
            
            return {
                'success': True,
                'data': data,
                'scaler_params': scaler_params
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Failed to normalize {column}: {str(e)}"]
            }
    
    async def _normalize_joint(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: NormalizationMethod
    ) -> Dict[str, Any]:
        """Normalize multiple columns jointly."""
        try:
            # Extract data for all columns
            joint_data = data[columns].dropna()
            
            if len(joint_data) < 10:
                return {
                    'success': False,
                    'warnings': ["Insufficient data for joint normalization"]
                }
            
            # Create scaler
            scaler = self._create_scaler(method)
            
            # Fit and transform
            normalized_values = scaler.fit_transform(joint_data.values)
            
            # Update data
            data.loc[joint_data.index, columns] = normalized_values
            
            # Store scaler
            scaler_key = f"joint_{method.value}"
            if self.config.store_scalers:
                self.scalers[scaler_key] = scaler
                self.scaler_metadata[scaler_key] = {
                    'columns': columns,
                    'method': method.value,
                    'column_type': 'joint_features',
                    'fitted_samples': len(joint_data)
                }
            
            # Create scaler parameters
            scaler_params = {
                'method': method.value,
                'scaler_object': scaler if self.config.enable_inverse else None,
                'columns': columns,
                'fitted_samples': len(joint_data)
            }
            
            return {
                'success': True,
                'data': data,
                'scaler_params': scaler_params
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Failed joint normalization: {str(e)}"]
            }
    
    def _create_scaler(self, method: NormalizationMethod):
        """Create scaler object based on method."""
        if method == NormalizationMethod.STANDARD:
            return StandardScaler()
        
        elif method == NormalizationMethod.MINMAX:
            return MinMaxScaler(feature_range=self.config.feature_range)
        
        elif method == NormalizationMethod.ROBUST:
            return RobustScaler(quantile_range=self.config.quantile_range)
        
        elif method == NormalizationMethod.QUANTILE:
            return QuantileTransformer(
                n_quantiles=self.config.n_quantiles,
                output_distribution=self.config.output_distribution,
                random_state=42
            )
        
        elif method == NormalizationMethod.POWER:
            return PowerTransformer(method='yeo-johnson', standardize=True)
        
        elif method == NormalizationMethod.UNIT_VECTOR:
            # Custom unit vector scaler
            return UnitVectorScaler()
        
        elif method == NormalizationMethod.FINANCIAL:
            # Custom financial scaler
            return FinancialScaler()
        
        elif method == NormalizationMethod.ROLLING:
            # Custom rolling scaler
            return RollingScaler(window=self.config.rolling_window)
        
        else:
            # Default to robust scaler
            return RobustScaler()
    
    def _assess_normalization_quality(
        self,
        original_data: pd.DataFrame,
        normalized_data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, float]:
        """Assess the quality of normalization."""
        quality_metrics = {}
        
        for column in columns:
            if column in original_data.columns and column in normalized_data.columns:
                orig_col = original_data[column].dropna()
                norm_col = normalized_data[column].dropna()
                
                if len(norm_col) > 0:
                    # Basic statistics
                    quality_metrics[f'{column}_mean'] = norm_col.mean()
                    quality_metrics[f'{column}_std'] = norm_col.std()
                    quality_metrics[f'{column}_min'] = norm_col.min()
                    quality_metrics[f'{column}_max'] = norm_col.max()
                    quality_metrics[f'{column}_skew'] = norm_col.skew() if len(norm_col) > 3 else 0
                    
                    # Normalization effectiveness
                    if len(orig_col) > 0:
                        # Variance reduction ratio
                        orig_var = orig_col.var()
                        norm_var = norm_col.var()
                        quality_metrics[f'{column}_var_ratio'] = norm_var / orig_var if orig_var > 0 else 1
                        
                        # Range compression
                        orig_range = orig_col.max() - orig_col.min()
                        norm_range = norm_col.max() - norm_col.min()
                        quality_metrics[f'{column}_range_ratio'] = norm_range / orig_range if orig_range > 0 else 1
        
        # Overall quality score
        mean_values = [abs(v) for k, v in quality_metrics.items() if k.endswith('_mean')]
        std_values = [v for k, v in quality_metrics.items() if k.endswith('_std')]
        
        if mean_values and std_values:
            avg_mean = np.mean(mean_values)
            avg_std = np.mean(std_values)
            quality_metrics['overall_quality'] = 1 / (1 + avg_mean + abs(avg_std - 1))
        else:
            quality_metrics['overall_quality'] = 0.5
        
        return quality_metrics
    
    def inverse_transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        column: str,
        scaler_params: Dict[str, Any]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Normalized data to transform back
            column: Column name to transform
            scaler_params: Scaler parameters from normalization
        
        Returns:
            Data in original scale
        """
        try:
            if not self.config.enable_inverse:
                self.logger.warning("Inverse transformation not enabled")
                return data
            
            # Get scaler object
            scaler = scaler_params.get('scaler_object')
            if scaler is None:
                # Try to reconstruct from stored scalers
                method = scaler_params.get('method', 'robust')
                scaler_key = f"{column}_{method}"
                scaler = self.scalers.get(scaler_key)
            
            if scaler is None:
                self.logger.warning(f"No scaler found for column {column}")
                return data
            
            # Perform inverse transformation
            if isinstance(data, pd.DataFrame):
                if column in data.columns:
                    col_data = data[column].dropna()
                    if len(col_data) > 0:
                        values = col_data.values.reshape(-1, 1)
                        original_values = scaler.inverse_transform(values)
                        data.loc[col_data.index, column] = original_values.flatten()
                        
                        self.normalization_stats['inverse_transforms'] += 1
                
                return data
            
            elif isinstance(data, np.ndarray):
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                
                original_values = scaler.inverse_transform(data)
                self.normalization_stats['inverse_transforms'] += 1
                
                return original_values.flatten() if original_values.shape[1] == 1 else original_values
            
            else:
                self.logger.error(f"Unsupported data type for inverse transform: {type(data)}")
                return data
                
        except Exception as e:
            self.logger.error(f"Error in inverse transformation: {e}")
            return data
    
    def _update_statistics(
        self,
        method: NormalizationMethod,
        columns: List[str],
        quality_metrics: Dict[str, float]
    ):
        """Update normalization statistics."""
        self.normalization_stats['total_normalized'] += 1
        self.normalization_stats['methods_used'][method.value] += 1
        
        for column in columns:
            if column not in self.normalization_stats['feature_stats']:
                self.normalization_stats['feature_stats'][column] = {
                    'normalizations': 0,
                    'avg_quality': 0
                }
            
            self.normalization_stats['feature_stats'][column]['normalizations'] += 1
            
            # Update average quality
            column_quality = quality_metrics.get(f'{column}_std', 1.0)
            current_avg = self.normalization_stats['feature_stats'][column]['avg_quality']
            count = self.normalization_stats['feature_stats'][column]['normalizations']
            new_avg = (current_avg * (count - 1) + column_quality) / count
            self.normalization_stats['feature_stats'][column]['avg_quality'] = new_avg
    
    def save_scalers(self, filepath: str):
        """Save all fitted scalers to file."""
        try:
            scaler_data = {
                'scalers': self.scalers,
                'metadata': self.scaler_metadata,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            self.logger.info(f"Scalers saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving scalers: {e}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        try:
            with open(filepath, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.scalers = scaler_data.get('scalers', {})
            self.scaler_metadata = scaler_data.get('metadata', {})
            
            self.logger.info(f"Scalers loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get normalization statistics."""
        return self.normalization_stats.copy()
    
    def get_config(self) -> NormalizationConfig:
        """Get current configuration."""
        return self.config


class UnitVectorScaler:
    """Custom unit vector scaler."""
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-8)  # Add small value to avoid division by zero
    
    def fit_transform(self, X):
        return self.transform(X)
    
    def inverse_transform(self, X):
        # Cannot perfectly inverse unit vector scaling
        return X


class FinancialScaler:
    """Custom scaler for financial data."""
    
    def __init__(self):
        self.returns_std = None
        self.price_median = None
    
    def fit(self, X):
        # Assume X contains price data
        prices = X.flatten()
        returns = np.diff(prices) / prices[:-1]
        
        self.returns_std = np.std(returns)
        self.price_median = np.median(prices)
        
        return self
    
    def transform(self, X):
        # Normalize by volatility and center around median
        prices = X.flatten()
        returns = np.diff(np.concatenate([[self.price_median], prices])) / np.concatenate([[self.price_median], prices[:-1]])
        
        if self.returns_std > 0:
            normalized = returns / self.returns_std
        else:
            normalized = returns
        
        return normalized.reshape(-1, 1)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        # Approximate inverse (not exact)
        normalized = X.flatten()
        returns = normalized * self.returns_std
        
        # Reconstruct prices (approximate)
        prices = [self.price_median]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices[1:]).reshape(-1, 1)


class RollingScaler:
    """Rolling window scaler."""
    
    def __init__(self, window=250):
        self.window = window
        self.rolling_stats = None
    
    def fit(self, X):
        # Store rolling statistics
        if len(X) >= self.window:
            self.rolling_stats = {
                'window': self.window,
                'last_values': X[-self.window:].copy()
            }
        return self
    
    def transform(self, X):
        if self.rolling_stats is None:
            return StandardScaler().fit_transform(X)
        
        # Apply rolling normalization
        result = np.zeros_like(X)
        for i in range(len(X)):
            if i >= self.window:
                window_data = X[i-self.window:i]
            else:
                # Use available data
                window_data = X[:i+1]
            
            if len(window_data) > 1:
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std > 0:
                    result[i] = (X[i] - mean) / std
                else:
                    result[i] = 0
            else:
                result[i] = 0
        
        return result
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        # Simplified inverse (not exact for rolling)
        return X