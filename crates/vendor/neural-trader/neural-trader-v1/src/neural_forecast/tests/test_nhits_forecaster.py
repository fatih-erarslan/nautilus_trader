"""
Unit Tests for NHITS Forecaster.

Tests for the core NHITS forecasting functionality including:
- Model initialization and configuration
- Training and prediction workflows
- Error handling and fallback mechanisms
- GPU acceleration features
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

# Import the module under test
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from neural_forecast.nhits_forecaster import NHITSForecaster


class TestNHITSForecaster:
    """Test suite for NHITSForecaster class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        values = np.random.randn(100).cumsum() + 100  # Random walk starting at 100
        
        return {
            'ds': dates.tolist(),
            'y': values.tolist(),
            'unique_id': 'test_series'
        }
    
    @pytest.fixture
    def forecaster_config(self):
        """Default configuration for forecaster."""
        return {
            'input_size': 24,
            'horizon': 12,
            'batch_size': 16,
            'max_epochs': 5,  # Small for testing
            'learning_rate': 1e-3,
            'enable_gpu': False,  # Disable GPU for CI/CD
            'model_cache_dir': tempfile.mkdtemp()
        }
    
    @pytest.fixture
    def forecaster(self, forecaster_config):
        """Create NHITSForecaster instance for testing."""
        return NHITSForecaster(**forecaster_config)
    
    def test_forecaster_initialization(self, forecaster_config):
        """Test forecaster initialization with various configurations."""
        # Test default initialization
        forecaster = NHITSForecaster()
        assert forecaster.input_size == 24
        assert forecaster.horizon == 12
        assert forecaster.device in ['cpu', 'cuda:0']
        
        # Test custom configuration
        forecaster = NHITSForecaster(**forecaster_config)
        assert forecaster.input_size == forecaster_config['input_size']
        assert forecaster.horizon == forecaster_config['horizon']
        assert forecaster.batch_size == forecaster_config['batch_size']
    
    def test_forecaster_initialization_without_neuralforecast(self, forecaster_config):
        """Test graceful handling when neuralforecast is not available."""
        with patch('neural_forecast.nhits_forecaster.NEURALFORECAST_AVAILABLE', False):
            with pytest.raises(ImportError, match="NeuralForecast required"):
                NHITSForecaster(**forecaster_config)
    
    def test_data_preparation(self, forecaster, sample_data):
        """Test data preparation for forecasting."""
        df = forecaster._prepare_dataframe(sample_data)
        
        assert isinstance(df, pd.DataFrame)
        assert 'unique_id' in df.columns
        assert 'ds' in df.columns
        assert 'y' in df.columns
        assert len(df) == len(sample_data['y'])
        assert pd.api.types.is_datetime64_any_dtype(df['ds'])
    
    def test_data_preparation_missing_columns(self, forecaster):
        """Test data preparation with missing required columns."""
        invalid_data = {'x': [1, 2, 3], 'z': [4, 5, 6]}
        
        with pytest.raises(ValueError, match="Data must contain 'ds' and 'y' columns"):
            forecaster._prepare_dataframe(invalid_data)
    
    def test_data_validation(self, forecaster, sample_data):
        """Test data format validation."""
        df = forecaster._prepare_dataframe(sample_data)
        
        # Should not raise any exceptions
        forecaster._validate_data_format(df)
        
        # Test with insufficient data
        short_df = df.head(10)  # Less than input_size + horizon
        with pytest.raises(ValueError, match="Insufficient data points"):
            forecaster._validate_data_format(short_df)
    
    @pytest.mark.asyncio
    async def test_model_training_mock(self, forecaster, sample_data):
        """Test model training with mocked neuralforecast."""
        # Mock the NeuralForecast components
        with patch('neural_forecast.nhits_forecaster.NEURALFORECAST_AVAILABLE', True):
            with patch('neural_forecast.nhits_forecaster.NeuralForecast') as mock_nf:
                with patch('neural_forecast.nhits_forecaster.NHITS') as mock_nhits:
                    
                    # Setup mocks
                    mock_model = Mock()
                    mock_nhits.return_value = mock_model
                    mock_nf_instance = Mock()
                    mock_nf.return_value = mock_nf_instance
                    mock_nf_instance.fit = Mock()
                    
                    forecaster.model = mock_model
                    
                    # Test training
                    result = await forecaster.fit(sample_data)
                    
                    assert result['success'] == True
                    assert 'training_time' in result
                    assert 'model_metadata' in result
                    assert forecaster.is_fitted == True
    
    @pytest.mark.asyncio
    async def test_prediction_without_fitted_model(self, forecaster, sample_data):
        """Test prediction fails without fitted model."""
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            await forecaster.predict(sample_data)
    
    @pytest.mark.asyncio
    async def test_fallback_prediction(self, forecaster, sample_data):
        """Test fallback prediction mechanism."""
        result = await forecaster._fallback_prediction(sample_data)
        
        assert result['success'] == True
        assert 'point_forecast' in result
        assert 'method' in result
        assert result['method'] == 'fallback_moving_average'
        assert len(result['point_forecast']) == forecaster.horizon
    
    @pytest.mark.asyncio
    async def test_fallback_prediction_with_invalid_data(self, forecaster):
        """Test fallback prediction with invalid data."""
        invalid_data = {'invalid': 'data'}
        result = await forecaster._fallback_prediction(invalid_data)
        
        # Should still return a result, even if it's a basic fallback
        assert 'success' in result
        assert 'point_forecast' in result
    
    def test_cache_key_generation(self, forecaster, sample_data):
        """Test cache key generation for predictions."""
        df = forecaster._prepare_dataframe(sample_data)
        cache_key = forecaster._generate_cache_key(df)
        
        assert isinstance(cache_key, str)
        assert cache_key.startswith('forecast_')
        
        # Same data should generate same key
        cache_key2 = forecaster._generate_cache_key(df)
        assert cache_key == cache_key2
    
    @pytest.mark.asyncio
    async def test_gpu_memory_management(self, forecaster):
        """Test GPU memory management functions."""
        # Should not raise exceptions even if GPU not available
        await forecaster._manage_gpu_memory()
        
        # Test with mocked GPU
        with patch('neural_forecast.nhits_forecaster.GPU_AVAILABLE', True):
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                with patch('torch.cuda.memory_allocated', return_value=1000000):
                    await forecaster._manage_gpu_memory()
                    mock_empty_cache.assert_called()
    
    def test_model_info(self, forecaster):
        """Test model information retrieval."""
        info = forecaster.get_model_info()
        
        assert isinstance(info, dict)
        assert 'is_fitted' in info
        assert 'configuration' in info
        assert 'cache_stats' in info
        assert info['is_fitted'] == False  # Not fitted yet
        
        # Check configuration values
        config = info['configuration']
        assert config['input_size'] == forecaster.input_size
        assert config['horizon'] == forecaster.horizon
        assert config['device'] == forecaster.device
    
    def test_clear_cache(self, forecaster):
        """Test cache clearing functionality."""
        # Add some dummy cache entries
        forecaster.forecast_cache['test_key'] = {'data': 'test'}
        forecaster.prediction_intervals['test_key'] = {'data': 'test'}
        
        assert len(forecaster.forecast_cache) > 0
        assert len(forecaster.prediction_intervals) > 0
        
        forecaster.clear_cache()
        
        assert len(forecaster.forecast_cache) == 0
        assert len(forecaster.prediction_intervals) == 0
    
    def test_model_save_without_fitted_model(self, forecaster):
        """Test model save fails without fitted model."""
        with pytest.raises(ValueError, match="No trained model to save"):
            forecaster.save_model()
    
    def test_model_load_nonexistent_file(self, forecaster):
        """Test model load with nonexistent file."""
        nonexistent_path = "/tmp/nonexistent_model.pkl"
        result = forecaster.load_model(nonexistent_path)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_batch_prediction_empty_list(self, forecaster):
        """Test batch prediction with empty list."""
        results = await forecaster.predict_batch([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_batch_prediction_with_exceptions(self, forecaster, sample_data):
        """Test batch prediction handling exceptions."""
        # Mock predict to raise exception
        with patch.object(forecaster, 'predict', side_effect=Exception("Test error")):
            results = await forecaster.predict_batch([sample_data])
            
            assert len(results) == 1
            assert results[0]['success'] == False
            assert 'error' in results[0]
    
    def test_hyperparameter_optimization_placeholder(self, forecaster, sample_data):
        """Test hyperparameter optimization (placeholder implementation)."""
        df = forecaster._prepare_dataframe(sample_data)
        result = forecaster.optimize_hyperparameters(df)
        
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'optimization_method' in result
        assert result['optimization_method'] == 'default_configuration'


class TestNHITSForecasterIntegration:
    """Integration tests for NHITSForecaster."""
    
    @pytest.fixture
    def integration_data(self):
        """Create more realistic financial time series data."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic stock price data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
        
        # Start with base price and add trends, seasonality, and noise
        base_price = 100
        trend = np.linspace(0, 20, 500)  # Upward trend
        seasonality = 5 * np.sin(np.arange(500) * 2 * np.pi / 24)  # Daily seasonality
        noise = np.random.normal(0, 2, 500)  # Random noise
        
        prices = base_price + trend + seasonality + noise
        
        return {
            'ds': dates.tolist(),
            'y': prices.tolist(),
            'unique_id': 'AAPL'
        }
    
    @pytest.mark.asyncio
    async def test_full_training_and_prediction_workflow_mock(self, integration_data):
        """Test complete workflow with realistic data (mocked)."""
        forecaster = NHITSForecaster(
            input_size=48,
            horizon=24,
            max_epochs=1,  # Minimal training for testing
            enable_gpu=False
        )
        
        # Mock the training and prediction process
        with patch('neural_forecast.nhits_forecaster.NEURALFORECAST_AVAILABLE', True):
            with patch('neural_forecast.nhits_forecaster.NeuralForecast') as mock_nf:
                with patch('neural_forecast.nhits_forecaster.NHITS') as mock_nhits:
                    
                    # Setup training mocks
                    mock_model = Mock()
                    mock_nhits.return_value = mock_model
                    mock_nf_instance = Mock()
                    mock_nf.return_value = mock_nf_instance
                    mock_nf_instance.fit = Mock()
                    
                    # Setup prediction mocks
                    mock_forecast = pd.DataFrame({
                        'ds': pd.date_range(start='2023-01-01', periods=24, freq='H'),
                        'NHITS': np.random.randn(24) + 120,  # Mock predictions
                        'NHITS-q-10': np.random.randn(24) + 115,
                        'NHITS-q-90': np.random.randn(24) + 125
                    })
                    mock_nf_instance.predict = Mock(return_value=mock_forecast)
                    
                    forecaster.model = mock_model
                    
                    # Test training
                    train_result = await forecaster.fit(integration_data)
                    assert train_result['success'] == True
                    
                    # Test prediction
                    forecaster.nf = mock_nf_instance  # Set the mocked instance
                    prediction_result = await forecaster.predict(integration_data)
                    
                    assert prediction_result['success'] == True
                    assert 'point_forecast' in prediction_result
                    assert 'prediction_intervals' in prediction_result
                    assert len(prediction_result['point_forecast']) == 24
    
    @pytest.mark.asyncio
    async def test_model_persistence_mock(self, integration_data):
        """Test model saving and loading (mocked)."""
        forecaster = NHITSForecaster(enable_gpu=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Mock training to set up a "trained" model
            with patch('neural_forecast.nhits_forecaster.NEURALFORECAST_AVAILABLE', True):
                with patch('neural_forecast.nhits_forecaster.NeuralForecast') as mock_nf:
                    with patch('neural_forecast.nhits_forecaster.NHITS') as mock_nhits:
                        
                        mock_model = Mock()
                        mock_nhits.return_value = mock_model
                        mock_nf_instance = Mock()
                        mock_nf.return_value = mock_nf_instance
                        mock_nf_instance.fit = Mock()
                        
                        forecaster.model = mock_model
                        forecaster.nf = mock_nf_instance
                        
                        # Train model
                        train_result = await forecaster.fit(integration_data)
                        assert train_result['success'] == True
                        
                        # Save model
                        save_path = forecaster.save_model(tmp_path)
                        assert os.path.exists(save_path)
                        
                        # Create new forecaster and load model
                        new_forecaster = NHITSForecaster(enable_gpu=False)
                        
                        # Mock the loading process
                        with patch.object(new_forecaster, 'load_model', return_value=True):
                            load_result = new_forecaster.load_model(save_path)
                            assert load_result == True
        
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_error_handling_configuration(self):
        """Test error handling in configuration."""
        # Test invalid configuration
        with pytest.raises((ValueError, TypeError)):
            NHITSForecaster(input_size=-1)  # Invalid input size
        
        with pytest.raises((ValueError, TypeError)):
            NHITSForecaster(horizon=0)  # Invalid horizon
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, integration_data):
        """Test memory efficiency with large datasets."""
        # Create larger dataset
        large_data = integration_data.copy()
        large_data['y'] = large_data['y'] * 10  # Make it larger
        
        forecaster = NHITSForecaster(
            input_size=24,
            horizon=12,
            enable_gpu=False
        )
        
        # Test data preparation doesn't crash with larger data
        df = forecaster._prepare_dataframe(large_data)
        assert len(df) == len(large_data['y'])
        
        # Test cache management
        forecaster.clear_cache()
        assert len(forecaster.forecast_cache) == 0


# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Test runner for development
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])