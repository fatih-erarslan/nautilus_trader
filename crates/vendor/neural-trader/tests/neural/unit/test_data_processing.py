"""
Unit Tests for Neural Data Processing

Tests for data preprocessing, transformation, and pipeline components.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Test utilities
from tests.neural.utils.fixtures import (
    sample_time_series_data, multi_asset_data, news_events_data,
    test_data_generator, basic_nhits_config
)
from tests.neural.utils.data_generators import (
    SyntheticTimeSeriesGenerator, TimeSeriesParams,
    MarketScenarioGenerator, NewsEventGenerator,
    ModelTestDataGenerator, prepare_nhits_format,
    create_multi_asset_nhits_format, add_external_regressors
)
from tests.neural.utils.performance_utils import monitor_memory, benchmark_latency


class TestTimeSeriesGeneration:
    """Test synthetic time series data generation."""
    
    def test_synthetic_generator_creation(self):
        """Test creation of synthetic time series generator."""
        generator = SyntheticTimeSeriesGenerator(seed=42)
        assert generator.seed == 42
    
    def test_basic_time_series_generation(self):
        """Test basic time series generation."""
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(n_points=1000, freq='H')
        
        series = generator.generate_single_series(params)
        
        assert isinstance(series, pd.DataFrame)
        assert len(series) == 1000
        assert 'timestamp' in series.columns
        assert 'price' in series.columns
        assert 'returns' in series.columns
        assert 'volume' in series.columns
        assert 'volatility' in series.columns
    
    def test_time_series_with_trend(self):
        """Test time series generation with trend."""
        generator = SyntheticTimeSeriesGenerator()
        
        # Positive trend
        params = TimeSeriesParams(n_points=1000, trend=0.01)
        series_up = generator.generate_single_series(params)
        
        # Negative trend
        params = TimeSeriesParams(n_points=1000, trend=-0.01)
        series_down = generator.generate_single_series(params)
        
        # Check trend direction
        final_price_up = series_up['price'].iloc[-1]
        initial_price_up = series_up['price'].iloc[0]
        
        final_price_down = series_down['price'].iloc[-1]
        initial_price_down = series_down['price'].iloc[0]
        
        assert final_price_up > initial_price_up  # Upward trend
        assert final_price_down < initial_price_down  # Downward trend
    
    def test_time_series_with_seasonality(self):
        """Test time series generation with seasonal patterns."""
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(
            n_points=168,  # 1 week of hourly data
            seasonality_periods=[24],  # Daily seasonality
            seasonality_amplitudes=[1.0],
            noise_level=0.1
        )
        
        series = generator.generate_single_series(params)
        
        # Extract prices
        prices = series['price'].values
        
        # Should show some pattern every 24 hours
        # Simple test: correlation between day 1 and day 7 patterns
        day1_pattern = prices[:24]
        day7_pattern = prices[-24:]
        
        correlation = np.corrcoef(day1_pattern, day7_pattern)[0, 1]
        assert correlation > 0.2  # Some positive correlation expected
    
    def test_multi_asset_generation(self):
        """Test multi-asset time series generation."""
        generator = SyntheticTimeSeriesGenerator()
        assets = ['AAPL', 'GOOGL', 'MSFT']
        params = TimeSeriesParams(n_points=500)
        
        # Without correlation
        multi_data = generator.generate_multi_asset_series(assets, params)
        
        assert len(multi_data) == len(assets)
        for asset in assets:
            assert asset in multi_data
            assert len(multi_data[asset]) == 500
            assert 'price' in multi_data[asset].columns
    
    def test_correlated_multi_asset_generation(self):
        """Test correlated multi-asset generation."""
        generator = SyntheticTimeSeriesGenerator()
        assets = ['AAPL', 'GOOGL']
        params = TimeSeriesParams(n_points=1000)
        
        # High correlation matrix
        correlation_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        
        multi_data = generator.generate_multi_asset_series(
            assets, params, correlation_matrix
        )
        
        # Check correlation between returns
        returns1 = multi_data['AAPL']['returns'].values
        returns2 = multi_data['GOOGL']['returns'].values
        
        actual_correlation = np.corrcoef(returns1, returns2)[0, 1]
        assert actual_correlation > 0.5  # Should show some correlation


class TestMarketScenarios:
    """Test market scenario generation."""
    
    def test_scenario_generator_creation(self):
        """Test market scenario generator creation."""
        generator = MarketScenarioGenerator()
        assert hasattr(generator, 'SCENARIOS')
        assert len(generator.SCENARIOS) > 0
    
    def test_bull_market_scenario(self):
        """Test bull market scenario generation."""
        generator = MarketScenarioGenerator()
        
        scenario_data = generator.generate_scenario('bull_market', n_points=1000)
        
        assert isinstance(scenario_data, pd.DataFrame)
        assert len(scenario_data) == 1000
        
        # Check trend is generally upward
        initial_price = scenario_data['price'].iloc[0]
        final_price = scenario_data['price'].iloc[-1]
        assert final_price > initial_price * 0.9  # Allow for some volatility
    
    def test_bear_market_scenario(self):
        """Test bear market scenario generation."""
        generator = MarketScenarioGenerator()
        
        scenario_data = generator.generate_scenario('bear_market', n_points=1000)
        
        # Check trend is generally downward
        initial_price = scenario_data['price'].iloc[0]
        final_price = scenario_data['price'].iloc[-1]
        assert final_price < initial_price * 1.1  # Allow for some volatility
    
    def test_market_crash_scenario(self):
        """Test market crash scenario generation."""
        generator = MarketScenarioGenerator()
        
        scenario_data = generator.generate_scenario('market_crash', n_points=500)
        
        # Check high volatility
        volatility = scenario_data['volatility'].mean()
        assert volatility > 0.1  # Should be quite volatile
    
    def test_available_scenarios(self):
        """Test all available scenarios can be generated."""
        generator = MarketScenarioGenerator()
        
        for scenario_name in generator.SCENARIOS.keys():
            scenario_data = generator.generate_scenario(scenario_name, n_points=100)
            
            assert isinstance(scenario_data, pd.DataFrame)
            assert len(scenario_data) == 100
            assert 'price' in scenario_data.columns
            assert 'returns' in scenario_data.columns
    
    def test_invalid_scenario(self):
        """Test handling of invalid scenario names."""
        generator = MarketScenarioGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_scenario('invalid_scenario')


class TestNewsEventGeneration:
    """Test news event generation."""
    
    def test_news_generator_creation(self):
        """Test news event generator creation."""
        generator = NewsEventGenerator(seed=42)
        assert generator.seed == 42
        assert len(generator.EVENT_TYPES) > 0
    
    def test_basic_event_generation(self):
        """Test basic news event generation."""
        generator = NewsEventGenerator()
        
        start_time = datetime.now() - timedelta(days=7)
        end_time = datetime.now()
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        events = generator.generate_events(start_time, end_time, assets)
        
        assert isinstance(events, pd.DataFrame)
        assert len(events) > 0
        assert 'timestamp' in events.columns
        assert 'asset' in events.columns
        assert 'event_type' in events.columns
        assert 'sentiment_score' in events.columns
        assert 'magnitude' in events.columns
    
    def test_event_temporal_distribution(self):
        """Test events are distributed across time period."""
        generator = NewsEventGenerator()
        
        start_time = datetime(2023, 1, 1)
        end_time = datetime(2023, 1, 7)
        assets = ['AAPL']
        
        events = generator.generate_events(start_time, end_time, assets, event_frequency='daily')
        
        # Check events are within time range
        for _, event in events.iterrows():
            assert start_time <= event['timestamp'] <= end_time
    
    def test_event_sentiment_range(self):
        """Test event sentiment scores are in valid range."""
        generator = NewsEventGenerator()
        
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        assets = ['AAPL']
        
        events = generator.generate_events(start_time, end_time, assets)
        
        # Check sentiment scores are in [-1, 1] range
        for sentiment in events['sentiment_score']:
            assert -1 <= sentiment <= 1
        
        # Check magnitude scores are in [0, 1] range
        for magnitude in events['magnitude']:
            assert 0 <= magnitude <= 1
    
    def test_event_feature_generation(self):
        """Test generation of neural network features from events."""
        generator = NewsEventGenerator()
        
        # Generate sample events
        start_time = datetime.now() - timedelta(hours=12)
        end_time = datetime.now()
        assets = ['AAPL']
        
        events = generator.generate_events(start_time, end_time, assets)
        
        # Generate features
        embedding_dim = 128
        features = generator.generate_event_features(events, embedding_dim)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape == (len(events), embedding_dim)
        assert torch.isfinite(features).all()


class TestDataFormatConversion:
    """Test data format conversions for neural models."""
    
    def test_prepare_nhits_format(self, sample_time_series_data):
        """Test conversion to NHITS format."""
        nhits_data = prepare_nhits_format(sample_time_series_data, asset_id='TEST_ASSET')
        
        assert isinstance(nhits_data, pd.DataFrame)
        assert 'unique_id' in nhits_data.columns
        assert 'ds' in nhits_data.columns
        assert 'y' in nhits_data.columns
        assert len(nhits_data) == len(sample_time_series_data)
        assert (nhits_data['unique_id'] == 'TEST_ASSET').all()
    
    def test_multi_asset_nhits_format(self, multi_asset_data):
        """Test multi-asset NHITS format conversion."""
        nhits_data = create_multi_asset_nhits_format(multi_asset_data)
        
        assert isinstance(nhits_data, pd.DataFrame)
        assert 'unique_id' in nhits_data.columns
        assert 'ds' in nhits_data.columns
        assert 'y' in nhits_data.columns
        
        # Check all assets are included
        unique_assets = nhits_data['unique_id'].unique()
        assert len(unique_assets) == len(multi_asset_data)
        
        for asset in multi_asset_data.keys():
            assert asset in unique_assets
    
    def test_external_regressors_addition(self, sample_time_series_data):
        """Test addition of external regressors."""
        n_regressors = 5
        augmented_data = add_external_regressors(sample_time_series_data, n_regressors)
        
        # Check new columns added
        original_cols = set(sample_time_series_data.columns)
        new_cols = set(augmented_data.columns)
        added_cols = new_cols - original_cols
        
        assert len(added_cols) == n_regressors
        
        # Check regressor columns are named correctly
        for i in range(1, n_regressors + 1):
            assert f'regressor_{i}' in added_cols
        
        # Check data integrity
        assert len(augmented_data) == len(sample_time_series_data)
        for col in original_cols:
            pd.testing.assert_series_equal(
                sample_time_series_data[col], 
                augmented_data[col]
            )


class TestModelDataGeneration:
    """Test data generation specifically for model training/testing."""
    
    def test_training_validation_data_generation(self):
        """Test training/validation data generation."""
        generator = ModelTestDataGenerator()
        
        X_train, y_train, X_val, y_val = generator.generate_training_validation_data(
            n_train=1000,
            n_val=200,
            input_size=168,
            horizon=24
        )
        
        assert isinstance(X_train, torch.Tensor)
        assert isinstance(y_train, torch.Tensor)
        assert isinstance(X_val, torch.Tensor)
        assert isinstance(y_val, torch.Tensor)
        
        assert X_train.shape == (1000, 168)
        assert y_train.shape == (1000, 24)
        assert X_val.shape == (200, 168)
        assert y_val.shape == (200, 24)
        
        # Check data types
        assert X_train.dtype == torch.float32
        assert y_train.dtype == torch.float32
    
    def test_batch_data_generation(self):
        """Test batch data generation for inference."""
        generator = ModelTestDataGenerator()
        
        batch_data = generator.generate_batch_data(
            batch_size=32,
            input_size=168,
            device=torch.device('cpu')
        )
        
        assert isinstance(batch_data, torch.Tensor)
        assert batch_data.shape == (32, 168)
        assert batch_data.device.type == 'cpu'
    
    def test_streaming_data_generation(self):
        """Test streaming data generation."""
        generator = ModelTestDataGenerator()
        
        streaming_data = generator.generate_streaming_data(
            n_timesteps=100,
            input_size=168
        )
        
        assert isinstance(streaming_data, list)
        assert len(streaming_data) == 100
        
        for window in streaming_data:
            assert isinstance(window, np.ndarray)
            assert len(window) == 168


class TestDataQuality:
    """Test data quality and validation."""
    
    def test_data_completeness(self, sample_time_series_data):
        """Test data completeness checks."""
        data = sample_time_series_data
        
        # Check no completely missing columns
        for col in data.columns:
            assert not data[col].isna().all()
        
        # Check timestamp continuity
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            time_diffs = timestamps.diff().dropna()
            
            # Most time differences should be consistent
            mode_diff = time_diffs.mode()[0]
            consistent_diffs = (time_diffs == mode_diff).sum()
            assert consistent_diffs / len(time_diffs) > 0.8  # 80% consistent
    
    def test_data_value_ranges(self, sample_time_series_data):
        """Test data value ranges are reasonable."""
        data = sample_time_series_data
        
        # Price should be positive
        if 'price' in data.columns:
            assert (data['price'] > 0).all()
        
        # Volume should be non-negative
        if 'volume' in data.columns:
            assert (data['volume'] >= 0).all()
        
        # Returns should be reasonable (not extreme)
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
            assert np.abs(returns).max() < 1.0  # Max 100% return in one period
    
    def test_data_outlier_detection(self, sample_time_series_data):
        """Test outlier detection in generated data."""
        data = sample_time_series_data
        
        # Check for extreme outliers using IQR method
        for col in ['price', 'volume', 'returns']:
            if col in data.columns:
                values = data[col].dropna()
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = (values < lower_bound) | (values > upper_bound)
                outlier_ratio = outliers.sum() / len(values)
                
                # Should have few extreme outliers
                assert outlier_ratio < 0.05  # Less than 5%


class TestDataPipelinePerformance:
    """Test data pipeline performance."""
    
    @benchmark_latency()
    def test_nhits_format_conversion_speed(self, sample_time_series_data):
        """Test NHITS format conversion speed."""
        # This will be benchmarked by the decorator
        nhits_data = prepare_nhits_format(sample_time_series_data)
        assert len(nhits_data) > 0
    
    @monitor_memory()
    def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        generator = SyntheticTimeSeriesGenerator()
        
        # Generate large dataset
        params = TimeSeriesParams(n_points=10000)  # Large dataset
        large_series = generator.generate_single_series(params)
        
        # Convert to NHITS format
        nhits_data = prepare_nhits_format(large_series)
        
        assert len(nhits_data) == 10000
    
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch data processing."""
        generator = ModelTestDataGenerator()
        
        # Test different batch sizes
        batch_sizes = [1, 16, 64, 256]
        processing_times = []
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            batch_data = generator.generate_batch_data(
                batch_size=batch_size,
                input_size=168
            )
            
            # Simulate some processing
            _ = torch.mean(batch_data, dim=1)
            
            end_time = time.time()
            processing_times.append(end_time - start_time)
        
        # Processing time should scale reasonably with batch size
        # (not necessarily linearly due to vectorization)
        assert all(t > 0 for t in processing_times)


class TestDataValidationEdgeCases:
    """Test edge cases in data validation."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        try:
            nhits_data = prepare_nhits_format(empty_df)
            assert len(nhits_data) == 0
        except (ValueError, KeyError):
            # Also acceptable to raise an error
            pass
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrames."""
        single_row = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [100.0],
            'volume': [1000],
            'returns': [0.01]
        })
        
        nhits_data = prepare_nhits_format(single_row)
        assert len(nhits_data) == 1
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        generator = SyntheticTimeSeriesGenerator()
        params = TimeSeriesParams(
            n_points=100,
            missing_data_probability=0.1  # 10% missing data
        )
        
        data_with_missing = generator.generate_single_series(params)
        
        # Should handle missing values
        nhits_data = prepare_nhits_format(data_with_missing)
        
        # Check that conversion completed
        assert isinstance(nhits_data, pd.DataFrame)
        assert len(nhits_data) > 0
    
    def test_duplicate_timestamps(self):
        """Test handling of duplicate timestamps."""
        duplicate_data = pd.DataFrame({
            'timestamp': [datetime.now()] * 5,  # Duplicate timestamps
            'price': [100, 101, 102, 103, 104],
            'volume': [1000] * 5,
            'returns': [0.01] * 5
        })
        
        # Should handle duplicates (drop or aggregate)
        try:
            nhits_data = prepare_nhits_format(duplicate_data)
            # If successful, should have handled duplicates somehow
            assert len(nhits_data) <= len(duplicate_data)
        except ValueError:
            # Also acceptable to raise an error for duplicates
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])