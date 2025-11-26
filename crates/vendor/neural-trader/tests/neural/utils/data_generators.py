"""
Neural Forecasting Test Data Generators

Utilities for generating various types of test data for neural forecasting tests.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import random
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TimeSeriesParams:
    """Parameters for time series generation."""
    n_points: int = 1000
    freq: str = 'H'  # Hourly frequency
    trend: float = 0.01
    seasonality_periods: List[int] = None
    seasonality_amplitudes: List[float] = None
    noise_level: float = 0.1
    volatility_clustering: bool = True
    outlier_probability: float = 0.02
    missing_data_probability: float = 0.01


class SyntheticTimeSeriesGenerator:
    """Generate synthetic time series data for testing."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_single_series(self, params: TimeSeriesParams) -> pd.DataFrame:
        """Generate a single time series with specified characteristics."""
        # Create timestamp index
        start_time = datetime.now() - timedelta(hours=params.n_points)
        timestamps = pd.date_range(
            start=start_time,
            periods=params.n_points,
            freq=params.freq
        )
        
        # Generate base signal
        t = np.arange(params.n_points)
        
        # Trend component
        trend = params.trend * t
        
        # Seasonal components
        seasonal = np.zeros(params.n_points)
        if params.seasonality_periods:
            periods = params.seasonality_periods or [24, 168]  # Daily and weekly
            amplitudes = params.seasonality_amplitudes or [1.0, 0.5]
            
            for period, amplitude in zip(periods, amplitudes):
                seasonal += amplitude * np.sin(2 * np.pi * t / period)
        
        # Noise component with optional volatility clustering
        if params.volatility_clustering:
            # GARCH-like volatility
            volatility = np.ones(params.n_points) * params.noise_level
            for i in range(1, params.n_points):
                volatility[i] = (0.1 + 0.85 * volatility[i-1] + 
                               0.05 * (volatility[i-1] * np.random.randn()) ** 2)
            noise = np.random.randn(params.n_points) * volatility
        else:
            noise = np.random.normal(0, params.noise_level, params.n_points)
        
        # Combine components
        base_series = trend + seasonal + noise
        
        # Add outliers
        if params.outlier_probability > 0:
            outlier_mask = np.random.random(params.n_points) < params.outlier_probability
            outlier_multiplier = np.random.choice([-1, 1], size=params.n_points) * 3
            base_series[outlier_mask] += outlier_multiplier[outlier_mask] * params.noise_level
        
        # Convert to price series (assuming log returns)
        initial_price = 100.0
        log_returns = base_series / 100  # Scale returns
        prices = initial_price * np.exp(np.cumsum(log_returns))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'returns': log_returns,
            'volume': self._generate_volume(params.n_points, prices),
            'volatility': np.abs(log_returns)
        })
        
        # Add missing data
        if params.missing_data_probability > 0:
            missing_mask = np.random.random(params.n_points) < params.missing_data_probability
            df.loc[missing_mask, 'price'] = np.nan
        
        return df
    
    def _generate_volume(self, n_points: int, prices: np.ndarray) -> np.ndarray:
        """Generate volume data correlated with price movements."""
        # Higher volume during price movements
        price_changes = np.abs(np.diff(prices, prepend=prices[0]))
        volume_base = np.random.lognormal(10, 0.5, n_points)
        volume_multiplier = 1 + 2 * (price_changes / np.max(price_changes))
        return volume_base * volume_multiplier
    
    def generate_multi_asset_series(self, 
                                  asset_names: List[str],
                                  params: TimeSeriesParams,
                                  correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, pd.DataFrame]:
        """Generate correlated multi-asset time series."""
        n_assets = len(asset_names)
        
        # Generate correlated noise if correlation matrix provided
        if correlation_matrix is not None:
            assert correlation_matrix.shape == (n_assets, n_assets)
            # Cholesky decomposition for correlated noise generation
            L = np.linalg.cholesky(correlation_matrix)
            correlated_noise = L @ np.random.randn(n_assets, params.n_points)
        else:
            correlated_noise = np.random.randn(n_assets, params.n_points)
        
        assets_data = {}
        
        for i, asset in enumerate(asset_names):
            # Modify params for each asset
            asset_params = TimeSeriesParams(
                n_points=params.n_points,
                freq=params.freq,
                trend=np.random.uniform(-0.01, 0.02),  # Random trend per asset
                seasonality_periods=params.seasonality_periods,
                seasonality_amplitudes=[
                    amp * np.random.uniform(0.5, 1.5) 
                    for amp in (params.seasonality_amplitudes or [1.0, 0.5])
                ],
                noise_level=params.noise_level * np.random.uniform(0.8, 1.2),
                volatility_clustering=params.volatility_clustering,
                outlier_probability=params.outlier_probability,
                missing_data_probability=params.missing_data_probability
            )
            
            # Generate series with correlated noise
            series = self.generate_single_series(asset_params)
            
            # Apply correlation to returns
            if correlation_matrix is not None:
                series['returns'] = correlated_noise[i] * asset_params.noise_level
                # Recalculate prices
                initial_price = 100.0 + np.random.uniform(-50, 50)  # Different base prices
                series['price'] = initial_price * np.exp(np.cumsum(series['returns']))
            
            assets_data[asset] = series
        
        return assets_data


class MarketScenarioGenerator:
    """Generate specific market scenario data."""
    
    SCENARIOS = {
        'bull_market': {
            'trend': 0.02,
            'volatility': 0.15,
            'duration_days': 90
        },
        'bear_market': {
            'trend': -0.015,
            'volatility': 0.25,
            'duration_days': 60
        },
        'market_crash': {
            'trend': -0.1,
            'volatility': 0.5,
            'duration_days': 10
        },
        'sideways_market': {
            'trend': 0.0,
            'volatility': 0.1,
            'duration_days': 120
        },
        'high_volatility': {
            'trend': 0.005,
            'volatility': 0.4,
            'duration_days': 30
        }
    }
    
    def generate_scenario(self, scenario_name: str, n_points: int = 1000) -> pd.DataFrame:
        """Generate data for a specific market scenario."""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.SCENARIOS.keys())}")
        
        config = self.SCENARIOS[scenario_name]
        
        # Generate scenario-specific time series
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(
            n_points=n_points,
            trend=config['trend'],
            noise_level=config['volatility'],
            volatility_clustering=True,
            outlier_probability=0.05 if 'crash' in scenario_name else 0.01
        )
        
        return generator.generate_single_series(params)
    
    def generate_regime_switching_series(self, 
                                       regime_configs: List[Dict],
                                       transition_probabilities: np.ndarray,
                                       n_points: int = 1000) -> pd.DataFrame:
        """Generate series with regime switching behavior."""
        n_regimes = len(regime_configs)
        current_regime = 0
        regime_history = []
        
        # Generate regime sequence
        for i in range(n_points):
            regime_history.append(current_regime)
            # Transition to next regime based on probabilities
            if np.random.random() < transition_probabilities[current_regime]:
                current_regime = np.random.choice(n_regimes)
        
        # Generate data for each regime
        all_series = []
        generator = SyntheticTimeSeriesGenerator()
        
        for regime_config in regime_configs:
            params = TimeSeriesParams(**regime_config)
            series = generator.generate_single_series(params)
            all_series.append(series)
        
        # Combine series based on regime history
        combined_data = []
        for i, regime in enumerate(regime_history):
            row_data = all_series[regime].iloc[i % len(all_series[regime])]
            combined_data.append(row_data)
        
        result_df = pd.DataFrame(combined_data)
        result_df['regime'] = regime_history
        
        return result_df


class NewsEventGenerator:
    """Generate synthetic news events data."""
    
    EVENT_TYPES = [
        'earnings_announcement',
        'product_launch',
        'merger_acquisition',
        'regulatory_change',
        'executive_change',
        'market_news',
        'economic_data',
        'geopolitical_event'
    ]
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_events(self, 
                       start_time: datetime,
                       end_time: datetime,
                       assets: List[str],
                       event_frequency: str = 'daily') -> pd.DataFrame:
        """Generate news events within a time period."""
        
        # Calculate number of events based on frequency
        time_diff = end_time - start_time
        if event_frequency == 'hourly':
            n_events = int(time_diff.total_seconds() / 3600 * 0.1)  # 10% of hours
        elif event_frequency == 'daily':
            n_events = int(time_diff.days * 0.5)  # 0.5 events per day
        else:
            n_events = 20  # Default
        
        events = []
        
        for i in range(n_events):
            # Random timestamp within range
            random_seconds = random.randint(0, int(time_diff.total_seconds()))
            event_time = start_time + timedelta(seconds=random_seconds)
            
            # Random event characteristics
            event_type = random.choice(self.EVENT_TYPES)
            asset = random.choice(assets)
            
            # Sentiment score (-1 to 1)
            sentiment = np.random.normal(0, 0.3)
            sentiment = np.clip(sentiment, -1, 1)
            
            # Magnitude (0 to 1)
            magnitude = np.random.beta(2, 5)  # Skewed towards lower values
            
            # Impact duration (minutes)
            impact_duration = np.random.lognormal(3, 1)  # Log-normal distribution
            
            events.append({
                'timestamp': event_time,
                'event_id': f'event_{i+1:04d}',
                'asset': asset,
                'event_type': event_type,
                'sentiment_score': sentiment,
                'magnitude': magnitude,
                'impact_duration_minutes': impact_duration,
                'title': f'{event_type.replace("_", " ").title()} for {asset}',
                'description': f'Generated test event: {event_type} affecting {asset}'
            })
        
        return pd.DataFrame(events).sort_values('timestamp')
    
    def generate_event_features(self, events_df: pd.DataFrame, 
                              embedding_dim: int = 128) -> torch.Tensor:
        """Generate neural network features from events."""
        n_events = len(events_df)
        
        # Create feature vectors
        features = torch.zeros(n_events, embedding_dim)
        
        for i, event in events_df.iterrows():
            # Encode event type (one-hot-like)
            type_idx = self.EVENT_TYPES.index(event['event_type'])
            features[i, type_idx] = 1.0
            
            # Sentiment and magnitude
            features[i, len(self.EVENT_TYPES)] = event['sentiment_score']
            features[i, len(self.EVENT_TYPES) + 1] = event['magnitude']
            
            # Random embeddings for other features
            features[i, len(self.EVENT_TYPES) + 2:] = torch.randn(
                embedding_dim - len(self.EVENT_TYPES) - 2
            ) * 0.1
        
        return features


class ModelTestDataGenerator:
    """Generate data specifically for model testing."""
    
    @staticmethod
    def generate_training_validation_data(n_train: int = 8000,
                                        n_val: int = 2000,
                                        input_size: int = 168,
                                        horizon: int = 24) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate training and validation tensors."""
        
        # Generate synthetic time series
        total_points = n_train + n_val + input_size + horizon
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(
            n_points=total_points,
            trend=0.01,
            seasonality_periods=[24, 168],  # Daily and weekly
            seasonality_amplitudes=[1.0, 0.5],
            noise_level=0.1,
            volatility_clustering=True
        )
        
        series_df = generator.generate_single_series(params)
        series = series_df['price'].values
        
        # Create sliding windows
        X, y = [], []
        for i in range(len(series) - input_size - horizon + 1):
            X.append(series[i:i + input_size])
            y.append(series[i + input_size:i + input_size + horizon])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Split into train/validation
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        return X_train, y_train, X_val, y_val
    
    @staticmethod
    def generate_batch_data(batch_size: int,
                          input_size: int,
                          device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Generate batch data for inference testing."""
        return torch.randn(batch_size, input_size, device=device)
    
    @staticmethod
    def generate_streaming_data(n_timesteps: int = 1000,
                              input_size: int = 168) -> List[np.ndarray]:
        """Generate streaming data for real-time testing."""
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(
            n_points=n_timesteps + input_size,
            trend=0.005,
            noise_level=0.05
        )
        
        series_df = generator.generate_single_series(params)
        series = series_df['price'].values
        
        # Create streaming windows
        streaming_data = []
        for i in range(n_timesteps):
            window = series[i:i + input_size]
            streaming_data.append(window)
        
        return streaming_data


# Utility functions for test data preparation
def prepare_nhits_format(df: pd.DataFrame, 
                        asset_id: str = 'asset_1',
                        value_col: str = 'price') -> pd.DataFrame:
    """Convert DataFrame to NHITS expected format."""
    nhits_df = pd.DataFrame({
        'unique_id': asset_id,
        'ds': df['timestamp'],
        'y': df[value_col]
    })
    return nhits_df


def create_multi_asset_nhits_format(multi_asset_data: Dict[str, pd.DataFrame],
                                   value_col: str = 'price') -> pd.DataFrame:
    """Convert multi-asset data to NHITS format."""
    all_data = []
    
    for asset_id, df in multi_asset_data.items():
        asset_df = prepare_nhits_format(df, asset_id, value_col)
        all_data.append(asset_df)
    
    return pd.concat(all_data, ignore_index=True)


def add_external_regressors(df: pd.DataFrame,
                          n_regressors: int = 5) -> pd.DataFrame:
    """Add external regressor columns to DataFrame."""
    n_points = len(df)
    
    for i in range(n_regressors):
        regressor_name = f'regressor_{i+1}'
        # Generate correlated regressor
        df[regressor_name] = np.random.randn(n_points) * 0.1
    
    return df


# Export main classes and functions
__all__ = [
    'TimeSeriesParams',
    'SyntheticTimeSeriesGenerator',
    'MarketScenarioGenerator', 
    'NewsEventGenerator',
    'ModelTestDataGenerator',
    'prepare_nhits_format',
    'create_multi_asset_nhits_format',
    'add_external_regressors'
]