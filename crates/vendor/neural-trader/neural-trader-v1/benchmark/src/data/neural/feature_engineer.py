"""
Feature Engineer for Neural Time Series Forecasting

This module provides advanced feature engineering capabilities specifically
designed for financial time series data used in neural forecasting models.

Key Features:
- Technical indicators
- Temporal features
- Lag features
- Statistical features
- Market microstructure features
- Volatility features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Types of features to engineer."""
    TECHNICAL_INDICATORS = "technical_indicators"
    TEMPORAL_FEATURES = "temporal_features"
    LAG_FEATURES = "lag_features"
    STATISTICAL_FEATURES = "statistical_features"
    VOLATILITY_FEATURES = "volatility_features"
    VOLUME_FEATURES = "volume_features"
    PRICE_TRANSFORM = "price_transform"
    MARKET_MICROSTRUCTURE = "market_microstructure"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Feature types to include
    add_technical_indicators: bool = True
    add_temporal_features: bool = True
    add_lag_features: bool = True
    add_statistical_features: bool = True
    add_volatility_features: bool = True
    add_volume_features: bool = True
    add_price_transforms: bool = True
    add_market_microstructure: bool = False
    
    # Technical indicator parameters
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Lag feature parameters
    lookback_periods: List[int] = field(default_factory=lambda: [1, 5, 12, 24])
    
    # Statistical feature parameters
    rolling_windows: List[int] = field(default_factory=lambda: [12, 24, 48])
    
    # Volatility parameters
    volatility_windows: List[int] = field(default_factory=lambda: [12, 24])
    garch_enabled: bool = False
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.001
    
    # Performance options
    parallel_processing: bool = True
    memory_efficient: bool = True


@dataclass
class FeatureResult:
    """Result of feature engineering."""
    success: bool
    data: Optional[pd.DataFrame] = None
    feature_names: List[str] = field(default_factory=list)
    feature_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class FeatureEngineer:
    """
    Advanced feature engineer for neural time series forecasting.
    
    This class provides comprehensive feature engineering capabilities
    specifically designed for financial time series data, creating
    features that are informative for neural forecasting models.
    """
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature metadata tracking
        self.feature_catalog = {}
        self.feature_stats = {
            'features_created': 0,
            'feature_types': {ft.value: 0 for ft in FeatureType},
            'processing_times': []
        }
        
        self.logger.info("FeatureEngineer initialized")
    
    async def engineer_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        existing_features: Optional[List[str]] = None
    ) -> FeatureResult:
        """
        Engineer features from time series data.
        
        Args:
            data: Input DataFrame with time series data
            target_column: Primary target column
            existing_features: Existing feature columns to preserve
        
        Returns:
            FeatureResult with engineered features
        """
        try:
            self.logger.info(f"Engineering features from {len(data)} samples")
            
            # Validate input
            validation_result = self._validate_input(data, target_column)
            if not validation_result['valid']:
                return FeatureResult(
                    success=False,
                    errors=validation_result['errors']
                )
            
            # Initialize result DataFrame
            result_data = data.copy()
            new_features = []
            warnings = []
            
            # Track original columns
            original_columns = set(data.columns)
            if existing_features:
                existing_features = [col for col in existing_features if col in original_columns]
            else:
                existing_features = []
            
            # Engineer different types of features
            if self.config.add_technical_indicators:
                tech_result = await self._add_technical_indicators(result_data, target_column)
                if tech_result['success']:
                    result_data = tech_result['data']
                    new_features.extend(tech_result['features'])
                else:
                    warnings.extend(tech_result['warnings'])
            
            if self.config.add_temporal_features:
                temporal_result = await self._add_temporal_features(result_data, target_column)
                if temporal_result['success']:
                    result_data = temporal_result['data']
                    new_features.extend(temporal_result['features'])
                else:
                    warnings.extend(temporal_result['warnings'])
            
            if self.config.add_lag_features:
                lag_result = await self._add_lag_features(result_data, target_column)
                if lag_result['success']:
                    result_data = lag_result['data']
                    new_features.extend(lag_result['features'])
                else:
                    warnings.extend(lag_result['warnings'])
            
            if self.config.add_statistical_features:
                stat_result = await self._add_statistical_features(result_data, target_column)
                if stat_result['success']:
                    result_data = stat_result['data']
                    new_features.extend(stat_result['features'])
                else:
                    warnings.extend(stat_result['warnings'])
            
            if self.config.add_volatility_features:
                vol_result = await self._add_volatility_features(result_data, target_column)
                if vol_result['success']:
                    result_data = vol_result['data']
                    new_features.extend(vol_result['features'])
                else:
                    warnings.extend(vol_result['warnings'])
            
            if self.config.add_volume_features and 'volume' in data.columns:
                volume_result = await self._add_volume_features(result_data, target_column)
                if volume_result['success']:
                    result_data = volume_result['data']
                    new_features.extend(volume_result['features'])
                else:
                    warnings.extend(volume_result['warnings'])
            
            if self.config.add_price_transforms:
                transform_result = await self._add_price_transforms(result_data, target_column)
                if transform_result['success']:
                    result_data = transform_result['data']
                    new_features.extend(transform_result['features'])
                else:
                    warnings.extend(transform_result['warnings'])
            
            if self.config.add_market_microstructure:
                micro_result = await self._add_market_microstructure(result_data, target_column)
                if micro_result['success']:
                    result_data = micro_result['data']
                    new_features.extend(micro_result['features'])
                else:
                    warnings.extend(micro_result['warnings'])
            
            # Feature selection if requested
            if self.config.max_features and len(new_features) > self.config.max_features:
                selection_result = self._select_features(result_data, target_column, new_features)
                if selection_result['success']:
                    result_data = selection_result['data']
                    new_features = selection_result['features']
                else:
                    warnings.extend(selection_result['warnings'])
            
            # Clean up features (remove those with too many NaNs)
            cleanup_result = self._cleanup_features(result_data, new_features)
            result_data = cleanup_result['data']
            final_features = cleanup_result['features']
            
            # Combine all feature names
            all_features = existing_features + final_features
            
            # Create metadata
            feature_metadata = self._create_feature_metadata(final_features, result_data)
            
            # Update statistics
            self.feature_stats['features_created'] += len(final_features)
            
            return FeatureResult(
                success=True,
                data=result_data,
                feature_names=all_features,
                feature_metadata=feature_metadata,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return FeatureResult(
                success=False,
                errors=[str(e)]
            )
    
    def _validate_input(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate input data."""
        errors = []
        
        if data.empty:
            errors.append("Input data is empty")
        
        if target_column not in data.columns:
            errors.append(f"Target column '{target_column}' not found")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            errors.append("Data must have DatetimeIndex")
        
        if len(data) < max(self.config.rolling_windows, default=1):
            errors.append("Insufficient data for feature engineering")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _add_technical_indicators(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add technical indicator features."""
        try:
            new_features = []
            price = data[target_column]
            
            # Simple Moving Averages
            for period in self.config.sma_periods:
                if len(data) >= period:
                    feature_name = f'sma_{period}'
                    data[feature_name] = price.rolling(window=period).mean()
                    new_features.append(feature_name)
                    
                    # SMA ratio
                    ratio_name = f'sma_ratio_{period}'
                    data[ratio_name] = price / data[feature_name]
                    new_features.append(ratio_name)
            
            # Exponential Moving Averages
            for period in self.config.ema_periods:
                feature_name = f'ema_{period}'
                data[feature_name] = price.ewm(span=period).mean()
                new_features.append(feature_name)
            
            # RSI
            if len(data) >= self.config.rsi_period + 1:
                rsi = self._calculate_rsi(price, self.config.rsi_period)
                data['rsi'] = rsi
                new_features.append('rsi')
            
            # Bollinger Bands
            if len(data) >= self.config.bollinger_period:
                bb_middle = price.rolling(window=self.config.bollinger_period).mean()
                bb_std = price.rolling(window=self.config.bollinger_period).std()
                
                data['bb_upper'] = bb_middle + (bb_std * self.config.bollinger_std)
                data['bb_lower'] = bb_middle - (bb_std * self.config.bollinger_std)
                data['bb_width'] = data['bb_upper'] - data['bb_lower']
                data['bb_position'] = (price - data['bb_lower']) / data['bb_width']
                
                new_features.extend(['bb_upper', 'bb_lower', 'bb_width', 'bb_position'])
            
            # MACD
            if len(data) >= max(self.config.macd_fast, self.config.macd_slow):
                ema_fast = price.ewm(span=self.config.macd_fast).mean()
                ema_slow = price.ewm(span=self.config.macd_slow).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=self.config.macd_signal).mean()
                
                data['macd'] = macd_line
                data['macd_signal'] = signal_line
                data['macd_histogram'] = macd_line - signal_line
                
                new_features.extend(['macd', 'macd_signal', 'macd_histogram'])
            
            # Rate of Change
            for period in [1, 5, 10]:
                if len(data) >= period:
                    feature_name = f'roc_{period}'
                    data[feature_name] = price.pct_change(periods=period)
                    new_features.append(feature_name)
            
            self.feature_stats['feature_types'][FeatureType.TECHNICAL_INDICATORS.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Technical indicators failed: {str(e)}"]
            }
    
    async def _add_temporal_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add temporal features."""
        try:
            new_features = []
            
            # Basic time features
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            
            new_features.extend(['hour', 'day_of_week', 'day_of_month', 'month', 'quarter'])
            
            # Cyclical encoding for temporal features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            new_features.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'])
            
            # Market session indicators
            data['is_market_open'] = (
                (data.index.hour >= 9) & (data.index.hour < 16) & 
                (data.index.dayofweek < 5)
            ).astype(int)
            
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            
            new_features.extend(['is_market_open', 'is_weekend'])
            
            # Time since market open/close
            market_open_hour = 9.5  # 9:30 AM
            market_close_hour = 16   # 4:00 PM
            
            data['time_since_open'] = np.where(
                data['is_market_open'],
                (data.index.hour + data.index.minute/60) - market_open_hour,
                np.nan
            )
            
            data['time_to_close'] = np.where(
                data['is_market_open'],
                market_close_hour - (data.index.hour + data.index.minute/60),
                np.nan
            )
            
            new_features.extend(['time_since_open', 'time_to_close'])
            
            self.feature_stats['feature_types'][FeatureType.TEMPORAL_FEATURES.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Temporal features failed: {str(e)}"]
            }
    
    async def _add_lag_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add lag features."""
        try:
            new_features = []
            price = data[target_column]
            
            # Price lags
            for lag in self.config.lookback_periods:
                if len(data) > lag:
                    feature_name = f'{target_column}_lag_{lag}'
                    data[feature_name] = price.shift(lag)
                    new_features.append(feature_name)
            
            # Return lags
            returns = price.pct_change()
            for lag in self.config.lookback_periods:
                if len(data) > lag:
                    feature_name = f'return_lag_{lag}'
                    data[feature_name] = returns.shift(lag)
                    new_features.append(feature_name)
            
            # Volatility lags
            volatility = returns.rolling(window=12).std()
            for lag in [1, 5, 12]:
                if len(data) > lag:
                    feature_name = f'volatility_lag_{lag}'
                    data[feature_name] = volatility.shift(lag)
                    new_features.append(feature_name)
            
            self.feature_stats['feature_types'][FeatureType.LAG_FEATURES.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Lag features failed: {str(e)}"]
            }
    
    async def _add_statistical_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add statistical features."""
        try:
            new_features = []
            price = data[target_column]
            returns = price.pct_change()
            
            # Rolling statistics
            for window in self.config.rolling_windows:
                if len(data) >= window:
                    # Price statistics
                    data[f'price_mean_{window}'] = price.rolling(window=window).mean()
                    data[f'price_std_{window}'] = price.rolling(window=window).std()
                    data[f'price_min_{window}'] = price.rolling(window=window).min()
                    data[f'price_max_{window}'] = price.rolling(window=window).max()
                    
                    # Price position in range
                    price_range = data[f'price_max_{window}'] - data[f'price_min_{window}']
                    data[f'price_position_{window}'] = (price - data[f'price_min_{window}']) / price_range
                    
                    new_features.extend([
                        f'price_mean_{window}', f'price_std_{window}',
                        f'price_min_{window}', f'price_max_{window}',
                        f'price_position_{window}'
                    ])
                    
                    # Return statistics
                    data[f'return_mean_{window}'] = returns.rolling(window=window).mean()
                    data[f'return_std_{window}'] = returns.rolling(window=window).std()
                    data[f'return_skew_{window}'] = returns.rolling(window=window).skew()
                    data[f'return_kurt_{window}'] = returns.rolling(window=window).kurt()
                    
                    new_features.extend([
                        f'return_mean_{window}', f'return_std_{window}',
                        f'return_skew_{window}', f'return_kurt_{window}'
                    ])
            
            # Percentile features
            for window in [24, 48]:
                if len(data) >= window:
                    data[f'price_percentile_25_{window}'] = price.rolling(window=window).quantile(0.25)
                    data[f'price_percentile_75_{window}'] = price.rolling(window=window).quantile(0.75)
                    
                    new_features.extend([
                        f'price_percentile_25_{window}',
                        f'price_percentile_75_{window}'
                    ])
            
            self.feature_stats['feature_types'][FeatureType.STATISTICAL_FEATURES.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Statistical features failed: {str(e)}"]
            }
    
    async def _add_volatility_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add volatility features."""
        try:
            new_features = []
            price = data[target_column]
            returns = price.pct_change()
            
            # Simple volatility measures
            for window in self.config.volatility_windows:
                if len(data) >= window:
                    # Standard deviation of returns
                    data[f'volatility_{window}'] = returns.rolling(window=window).std()
                    
                    # Parkinson volatility (if we have OHLC data)
                    if all(col in data.columns for col in ['high', 'low']):
                        high = data['high']
                        low = data['low']
                        parkinson_vol = np.sqrt(
                            (np.log(high / low) ** 2).rolling(window=window).mean() / (4 * np.log(2))
                        )
                        data[f'parkinson_volatility_{window}'] = parkinson_vol
                        new_features.append(f'parkinson_volatility_{window}')
                    
                    new_features.append(f'volatility_{window}')
            
            # Realized volatility
            squared_returns = returns ** 2
            data['realized_volatility_daily'] = squared_returns.rolling(window=288).sum()  # 288 5-min periods in a day
            new_features.append('realized_volatility_daily')
            
            # Volatility regimes
            long_vol = returns.rolling(window=48).std()
            short_vol = returns.rolling(window=12).std()
            data['volatility_regime'] = (short_vol > long_vol).astype(int)
            new_features.append('volatility_regime')
            
            # VIX-like features (volatility of volatility)
            if len(data) >= 48:
                vol_of_vol = long_vol.rolling(window=24).std()
                data['volatility_of_volatility'] = vol_of_vol
                new_features.append('volatility_of_volatility')
            
            self.feature_stats['feature_types'][FeatureType.VOLATILITY_FEATURES.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Volatility features failed: {str(e)}"]
            }
    
    async def _add_volume_features(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add volume-based features."""
        try:
            new_features = []
            
            if 'volume' not in data.columns:
                return {'success': True, 'data': data, 'features': []}
            
            volume = data['volume']
            price = data[target_column]
            
            # Volume moving averages
            for period in [10, 20, 50]:
                if len(data) >= period:
                    feature_name = f'volume_sma_{period}'
                    data[feature_name] = volume.rolling(window=period).mean()
                    new_features.append(feature_name)
                    
                    # Volume ratio
                    ratio_name = f'volume_ratio_{period}'
                    data[ratio_name] = volume / data[feature_name]
                    new_features.append(ratio_name)
            
            # Volume-price features
            data['vwap'] = (price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
            data['price_volume_correlation'] = price.rolling(window=20).corr(volume)
            
            new_features.extend(['vwap', 'price_volume_correlation'])
            
            # On-Balance Volume
            price_change = price.diff()
            obv = (volume * np.sign(price_change)).cumsum()
            data['obv'] = obv
            new_features.append('obv')
            
            # Volume rate of change
            data['volume_roc'] = volume.pct_change()
            new_features.append('volume_roc')
            
            self.feature_stats['feature_types'][FeatureType.VOLUME_FEATURES.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Volume features failed: {str(e)}"]
            }
    
    async def _add_price_transforms(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add price transformation features."""
        try:
            new_features = []
            price = data[target_column]
            
            # Log price
            data['log_price'] = np.log(price)
            new_features.append('log_price')
            
            # Log returns
            data['log_return'] = np.log(price / price.shift(1))
            new_features.append('log_return')
            
            # Differenced price
            data['price_diff'] = price.diff()
            new_features.append('price_diff')
            
            # Percentage returns
            data['return'] = price.pct_change()
            new_features.append('return')
            
            # Cumulative returns
            data['cumulative_return'] = (1 + data['return']).cumprod() - 1
            new_features.append('cumulative_return')
            
            # Price relative to recent high/low
            if len(data) >= 20:
                rolling_high = price.rolling(window=20).max()
                rolling_low = price.rolling(window=20).min()
                
                data['price_from_high'] = (price - rolling_high) / rolling_high
                data['price_from_low'] = (price - rolling_low) / rolling_low
                
                new_features.extend(['price_from_high', 'price_from_low'])
            
            self.feature_stats['feature_types'][FeatureType.PRICE_TRANSFORM.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Price transforms failed: {str(e)}"]
            }
    
    async def _add_market_microstructure(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, Any]:
        """Add market microstructure features."""
        try:
            new_features = []
            
            # Bid-ask spread features (if available)
            if all(col in data.columns for col in ['bid', 'ask']):
                data['bid_ask_spread'] = data['ask'] - data['bid']
                data['bid_ask_spread_pct'] = data['bid_ask_spread'] / data[target_column]
                new_features.extend(['bid_ask_spread', 'bid_ask_spread_pct'])
            
            # Trade size features (if available)
            if 'trade_size' in data.columns:
                data['avg_trade_size'] = data['trade_size'].rolling(window=20).mean()
                data['trade_size_ratio'] = data['trade_size'] / data['avg_trade_size']
                new_features.extend(['avg_trade_size', 'trade_size_ratio'])
            
            # Price impact features
            price = data[target_column]
            if 'volume' in data.columns:
                volume = data['volume']
                returns = price.pct_change()
                
                # Kyle's lambda (price impact measure)
                if len(data) >= 20:
                    covariance = returns.rolling(window=20).cov(volume)
                    variance = volume.rolling(window=20).var()
                    kyle_lambda = covariance / variance
                    data['kyle_lambda'] = kyle_lambda
                    new_features.append('kyle_lambda')
            
            self.feature_stats['feature_types'][FeatureType.MARKET_MICROSTRUCTURE.value] += len(new_features)
            
            return {
                'success': True,
                'data': data,
                'features': new_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Market microstructure features failed: {str(e)}"]
            }
    
    def _calculate_rsi(self, price: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = price.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Select most important features."""
        try:
            # Simple correlation-based feature selection
            target = data[target_column]
            feature_data = data[feature_names]
            
            # Calculate correlations
            correlations = {}
            for feature in feature_names:
                if feature in feature_data.columns:
                    valid_data = pd.concat([target, feature_data[feature]], axis=1).dropna()
                    if len(valid_data) > 10:
                        corr = valid_data.corr().iloc[0, 1]
                        correlations[feature] = abs(corr) if not np.isnan(corr) else 0
            
            # Sort by correlation and select top features
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat for feat, corr in sorted_features[:self.config.max_features] 
                               if corr >= self.config.feature_importance_threshold]
            
            # Keep only selected features
            columns_to_keep = [col for col in data.columns if col not in feature_names or col in selected_features]
            selected_data = data[columns_to_keep]
            
            return {
                'success': True,
                'data': selected_data,
                'features': selected_features
            }
            
        except Exception as e:
            return {
                'success': False,
                'warnings': [f"Feature selection failed: {str(e)}"]
            }
    
    def _cleanup_features(
        self,
        data: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Clean up features by removing those with too many NaNs."""
        cleaned_features = []
        
        for feature in feature_names:
            if feature in data.columns:
                nan_ratio = data[feature].isna().sum() / len(data)
                if nan_ratio < 0.5:  # Keep features with less than 50% NaNs
                    cleaned_features.append(feature)
                else:
                    # Remove feature with too many NaNs
                    data = data.drop(columns=[feature])
        
        return {
            'data': data,
            'features': cleaned_features
        }
    
    def _create_feature_metadata(
        self,
        feature_names: List[str],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create metadata for engineered features."""
        metadata = {
            'total_features': len(feature_names),
            'feature_types': {},
            'feature_stats': {}
        }
        
        # Categorize features by type
        for feature in feature_names:
            if feature in data.columns:
                feature_type = self._classify_feature_type(feature)
                if feature_type not in metadata['feature_types']:
                    metadata['feature_types'][feature_type] = []
                metadata['feature_types'][feature_type].append(feature)
                
                # Basic statistics
                metadata['feature_stats'][feature] = {
                    'mean': data[feature].mean() if pd.api.types.is_numeric_dtype(data[feature]) else None,
                    'std': data[feature].std() if pd.api.types.is_numeric_dtype(data[feature]) else None,
                    'missing_ratio': data[feature].isna().sum() / len(data)
                }
        
        return metadata
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on name."""
        if any(indicator in feature_name.lower() for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb_']):
            return 'technical_indicator'
        elif any(temporal in feature_name.lower() for temporal in ['hour', 'day', 'month', 'time_']):
            return 'temporal'
        elif 'lag' in feature_name.lower():
            return 'lag'
        elif any(stat in feature_name.lower() for stat in ['mean', 'std', 'min', 'max', 'percentile']):
            return 'statistical'
        elif 'volatility' in feature_name.lower() or 'vol' in feature_name.lower():
            return 'volatility'
        elif 'volume' in feature_name.lower():
            return 'volume'
        elif any(transform in feature_name.lower() for transform in ['log', 'return', 'diff']):
            return 'price_transform'
        else:
            return 'other'
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature engineering statistics."""
        return self.feature_stats.copy()
    
    def get_config(self) -> FeatureConfig:
        """Get current configuration."""
        return self.config