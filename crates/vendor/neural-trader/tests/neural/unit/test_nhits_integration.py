"""
Unit Tests for NHITS Integration

Comprehensive unit tests for NHITS model integration in the AI News Trading Platform.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import warnings

# Test utilities
from tests.neural.utils.fixtures import (
    basic_nhits_config, high_performance_nhits_config, 
    sample_time_series_data, gpu_available, device
)
from tests.neural.utils.data_generators import (
    SyntheticTimeSeriesGenerator, TimeSeriesParams,
    ModelTestDataGenerator, prepare_nhits_format
)
from tests.neural.utils.gpu_utils import (
    GPUDetector, skip_if_no_gpu, require_gpu_memory,
    gpu_memory_context
)
from tests.neural.utils.mock_objects import (
    MockNHITSConfig, MockNHITSModel, MockRealTimeEngine,
    create_mock_nhits_model
)
from tests.neural.utils.performance_utils import (
    benchmark_latency, monitor_memory, BenchmarkConfig
)

# Neural forecasting components (mock if not available)
try:
    from plans.neuralforecast.NHITS_Implementation_Guide import (
        NHITSConfig, OptimizedNHITS, RealTimeNHITSEngine,
        TradingDataLoader, NHITSStack
    )
    NEURAL_COMPONENTS_AVAILABLE = True
except ImportError:
    NEURAL_COMPONENTS_AVAILABLE = False
    # Use mock implementations
    NHITSConfig = MockNHITSConfig
    OptimizedNHITS = MockNHITSModel
    RealTimeNHITSEngine = MockRealTimeEngine


class TestNHITSConfig:
    """Test NHITS configuration management."""
    
    def test_default_config_creation(self):
        """Test creation of default NHITS configuration."""
        config = NHITSConfig()
        
        assert config.h > 0
        assert config.input_size > 0
        assert len(config.n_freq_downsample) > 0
        assert len(config.n_pool_kernel_size) > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
    
    def test_custom_config_creation(self):
        """Test creation of custom NHITS configuration."""
        custom_config = NHITSConfig(
            h=48,
            input_size=336,
            batch_size=64,
            learning_rate=0.001,
            use_gpu=True
        )
        
        assert custom_config.h == 48
        assert custom_config.input_size == 336
        assert custom_config.batch_size == 64
        assert custom_config.learning_rate == 0.001
        assert custom_config.use_gpu == True
    
    def test_config_validation(self, basic_nhits_config):
        """Test configuration validation."""
        config = basic_nhits_config
        
        # Test positive values
        assert config.h > 0
        assert config.input_size > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0
        
        # Test list lengths match
        assert len(config.n_freq_downsample) == len(config.n_pool_kernel_size)
        
        # Test confidence levels are between 0 and 1
        for level in config.confidence_levels:
            assert 0 < level < 1
    
    def test_config_gpu_detection(self):
        """Test GPU configuration detection."""
        config = NHITSConfig(use_gpu=True)
        
        if torch.cuda.is_available():
            assert config.use_gpu == True
        else:
            # Should handle gracefully even if GPU not available
            assert isinstance(config.use_gpu, bool)


class TestNHITSModel:
    """Test NHITS model implementation."""
    
    def test_model_creation(self, basic_nhits_config):
        """Test NHITS model creation."""
        model = OptimizedNHITS(basic_nhits_config)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'config')
        assert model.config == basic_nhits_config
    
    def test_model_forward_pass(self, basic_nhits_config, device):
        """Test model forward pass."""
        model = OptimizedNHITS(basic_nhits_config)
        if device.type == 'cuda':
            model = model.to(device)
        
        # Create sample input
        batch_size = 2
        input_tensor = torch.randn(batch_size, basic_nhits_config.input_size, device=device)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        assert isinstance(output, dict)
        assert 'point_forecast' in output
        assert output['point_forecast'].shape == (batch_size, basic_nhits_config.h)
    
    def test_model_output_shapes(self, basic_nhits_config):
        """Test model output shapes with different batch sizes."""
        model = OptimizedNHITS(basic_nhits_config)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, basic_nhits_config.input_size)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            assert output['point_forecast'].shape == (batch_size, basic_nhits_config.h)
            if 'backcast_residual' in output:
                assert output['backcast_residual'].shape == (batch_size, basic_nhits_config.input_size)
    
    @skip_if_no_gpu
    def test_model_gpu_compatibility(self, basic_nhits_config):
        """Test model GPU compatibility."""
        model = OptimizedNHITS(basic_nhits_config)
        model = model.cuda()
        
        input_tensor = torch.randn(4, basic_nhits_config.input_size, device='cuda')
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output['point_forecast'].device.type == 'cuda'
    
    def test_model_parameter_count(self, basic_nhits_config):
        """Test model parameter count is reasonable."""
        model = OptimizedNHITS(basic_nhits_config)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should have parameters but not excessive
        assert param_count > 1000  # Minimum complexity
        assert param_count < 50_000_000  # Maximum reasonable size
    
    def test_model_gradient_flow(self, basic_nhits_config):
        """Test gradient flow through model."""
        model = OptimizedNHITS(basic_nhits_config)
        model.train()
        
        input_tensor = torch.randn(2, basic_nhits_config.input_size, requires_grad=True)
        target = torch.randn(2, basic_nhits_config.h)
        
        # Forward pass
        output = model(input_tensor)
        loss = nn.MSELoss()(output['point_forecast'], target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()


class TestNHITSStack:
    """Test individual NHITS stack components."""
    
    @pytest.mark.skipif(not NEURAL_COMPONENTS_AVAILABLE, reason="Neural components not available")
    def test_stack_creation(self):
        """Test NHITS stack creation."""
        stack = NHITSStack(
            input_size=168,
            h=24,
            pool_size=4,
            freq_downsample=2,
            stack_id=0
        )
        
        assert isinstance(stack, nn.Module)
        assert hasattr(stack, 'pooling')
        assert hasattr(stack, 'backcast_fc')
        assert hasattr(stack, 'forecast_fc')
    
    @pytest.mark.skipif(not NEURAL_COMPONENTS_AVAILABLE, reason="Neural components not available")
    def test_stack_forward_pass(self):
        """Test stack forward pass."""
        input_size = 168
        h = 24
        batch_size = 4
        
        stack = NHITSStack(
            input_size=input_size,
            h=h,
            pool_size=4,
            freq_downsample=2,
            stack_id=0
        )
        
        input_tensor = torch.randn(batch_size, input_size)
        
        with torch.no_grad():
            backcast, forecast = stack(input_tensor)
        
        assert backcast.shape == (batch_size, input_size)
        assert forecast.shape == (batch_size, h)


class TestRealTimeEngine:
    """Test real-time inference engine."""
    
    def test_engine_creation(self, basic_nhits_config):
        """Test real-time engine creation."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            # Create and save a mock model
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            
            assert engine.config == basic_nhits_config
            assert hasattr(engine, 'model')
    
    def test_engine_prediction(self, basic_nhits_config):
        """Test engine prediction functionality."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            # Create and save a mock model
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            
            # Test prediction
            input_data = np.random.randn(basic_nhits_config.input_size)
            result = engine.predict(input_data)
            
            assert isinstance(result, dict)
            assert 'predictions' in result
            assert 'inference_time_ms' in result
            assert 'timestamp' in result
            assert len(result['predictions']) == basic_nhits_config.h
    
    @pytest.mark.asyncio
    async def test_engine_streaming(self, basic_nhits_config):
        """Test engine streaming predictions."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            
            # Create mock data stream
            async def mock_data_stream():
                for i in range(5):
                    yield np.random.randn(basic_nhits_config.input_size + i)
                    await asyncio.sleep(0.01)
            
            # Test streaming
            predictions = []
            async for prediction in engine.stream_predict(mock_data_stream()):
                predictions.append(prediction)
            
            assert len(predictions) > 0
            for pred in predictions:
                assert 'predictions' in pred
                assert 'timestamp' in pred
    
    @benchmark_latency()
    def test_engine_latency(self, basic_nhits_config):
        """Test engine inference latency."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            mock_model = MockNHITSModel(basic_nhits_config)
            torch.save(mock_model.state_dict(), tmp_file.name)
            
            engine = RealTimeNHITSEngine(tmp_file.name, basic_nhits_config)
            input_data = np.random.randn(basic_nhits_config.input_size)
            
            # This will be benchmarked by the decorator
            result = engine.predict(input_data)
            
            assert result is not None


class TestTradingDataLoader:
    """Test trading data loader functionality."""
    
    @pytest.mark.skipif(not NEURAL_COMPONENTS_AVAILABLE, reason="Neural components not available")
    def test_data_loader_creation(self, basic_nhits_config):
        """Test data loader creation."""
        loader = TradingDataLoader(basic_nhits_config)
        
        assert loader.config == basic_nhits_config
        assert hasattr(loader, 'device')
        assert hasattr(loader, 'input_buffer')
        assert hasattr(loader, 'target_buffer')
    
    @pytest.mark.skipif(not NEURAL_COMPONENTS_AVAILABLE, reason="Neural components not available")
    def test_data_loader_batch_preparation(self, basic_nhits_config):
        """Test batch preparation."""
        loader = TradingDataLoader(basic_nhits_config)
        
        # Create sample data
        total_size = basic_nhits_config.input_size + basic_nhits_config.h
        sample_data = np.random.randn(basic_nhits_config.batch_size, total_size)
        
        input_batch, target_batch = loader.prepare_batch(sample_data)
        
        assert input_batch.shape == (basic_nhits_config.batch_size, basic_nhits_config.input_size)
        assert target_batch.shape == (basic_nhits_config.batch_size, basic_nhits_config.h)
    
    @skip_if_no_gpu
    @pytest.mark.skipif(not NEURAL_COMPONENTS_AVAILABLE, reason="Neural components not available")
    def test_data_loader_gpu_transfer(self, basic_nhits_config):
        """Test GPU data transfer."""
        gpu_config = basic_nhits_config
        gpu_config.use_gpu = True
        
        loader = TradingDataLoader(gpu_config)
        
        total_size = gpu_config.input_size + gpu_config.h
        sample_data = np.random.randn(gpu_config.batch_size, total_size)
        
        input_batch, target_batch = loader.prepare_batch(sample_data)
        
        if torch.cuda.is_available():
            assert input_batch.device.type == 'cuda'
            assert target_batch.device.type == 'cuda'


class TestNHITSIntegration:
    """Test NHITS integration with trading platform."""
    
    def test_time_series_data_compatibility(self, sample_time_series_data):
        """Test compatibility with trading time series data."""
        # Convert to NHITS format
        nhits_data = prepare_nhits_format(sample_time_series_data)
        
        assert 'unique_id' in nhits_data.columns
        assert 'ds' in nhits_data.columns
        assert 'y' in nhits_data.columns
        assert len(nhits_data) > 0
    
    def test_model_training_simulation(self, basic_nhits_config):
        """Test model training simulation."""
        # Generate training data
        generator = ModelTestDataGenerator()
        X_train, y_train, X_val, y_val = generator.generate_training_validation_data(
            n_train=100,  # Small for testing
            n_val=20,
            input_size=basic_nhits_config.input_size,
            horizon=basic_nhits_config.h
        )
        
        model = OptimizedNHITS(basic_nhits_config)
        
        # Test training (mock)
        if hasattr(model, 'fit'):
            metrics = model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            assert isinstance(metrics, dict)
            assert 'train_loss' in metrics or 'training_time' in metrics
    
    def test_prediction_pipeline(self, basic_nhits_config, sample_time_series_data):
        """Test end-to-end prediction pipeline."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Prepare input data
        recent_data = sample_time_series_data['price'].values[-basic_nhits_config.input_size:]
        input_tensor = torch.tensor(recent_data, dtype=torch.float32).unsqueeze(0)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        predictions = output['point_forecast']
        
        assert predictions.shape == (1, basic_nhits_config.h)
        assert not torch.isnan(predictions).any()
        assert torch.isfinite(predictions).all()
    
    @monitor_memory()
    def test_memory_efficiency(self, basic_nhits_config):
        """Test memory efficiency of NHITS implementation."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, basic_nhits_config.input_size)
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Memory should be released
            del input_tensor, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def test_model_serialization(self, basic_nhits_config):
        """Test model serialization and loading."""
        model = OptimizedNHITS(basic_nhits_config)
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            # Save model
            torch.save(model.state_dict(), tmp_file.name)
            
            # Load model
            loaded_model = OptimizedNHITS(basic_nhits_config)
            loaded_model.load_state_dict(torch.load(tmp_file.name, map_location='cpu'))
            
            # Test equivalence
            input_tensor = torch.randn(2, basic_nhits_config.input_size)
            
            with torch.no_grad():
                original_output = model(input_tensor)
                loaded_output = loaded_model(input_tensor)
            
            torch.testing.assert_close(
                original_output['point_forecast'], 
                loaded_output['point_forecast'],
                rtol=1e-5, atol=1e-7
            )


class TestNHITSErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_shapes(self, basic_nhits_config):
        """Test handling of invalid input shapes."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Wrong input size
        wrong_input = torch.randn(2, basic_nhits_config.input_size + 10)
        
        with pytest.raises((RuntimeError, ValueError)):
            model(wrong_input)
    
    def test_empty_input_handling(self, basic_nhits_config):
        """Test handling of empty inputs."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Empty batch
        empty_input = torch.empty(0, basic_nhits_config.input_size)
        
        # Should handle gracefully or raise appropriate error
        try:
            with torch.no_grad():
                output = model(empty_input)
            assert output['point_forecast'].shape[0] == 0
        except (RuntimeError, ValueError):
            # Also acceptable to raise an error
            pass
    
    def test_nan_input_handling(self, basic_nhits_config):
        """Test handling of NaN inputs."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Input with NaN values
        nan_input = torch.randn(2, basic_nhits_config.input_size)
        nan_input[0, :5] = float('nan')
        
        with torch.no_grad():
            output = model(nan_input)
        
        # Output should either be NaN or model should handle it
        predictions = output['point_forecast']
        
        # At minimum, should not crash and produce output of correct shape
        assert predictions.shape == (2, basic_nhits_config.h)
    
    def test_extreme_values_handling(self, basic_nhits_config):
        """Test handling of extreme input values."""
        model = OptimizedNHITS(basic_nhits_config)
        
        # Very large values
        large_input = torch.ones(2, basic_nhits_config.input_size) * 1e6
        
        # Very small values
        small_input = torch.ones(2, basic_nhits_config.input_size) * 1e-6
        
        for test_input in [large_input, small_input]:
            with torch.no_grad():
                output = model(test_input)
            
            predictions = output['point_forecast']
            assert torch.isfinite(predictions).all()
            assert predictions.shape == (2, basic_nhits_config.h)


class TestNHITSPerformanceRegression:
    """Test for performance regressions."""
    
    def test_inference_speed_regression(self, basic_nhits_config):
        """Test inference speed doesn't regress."""
        model = OptimizedNHITS(basic_nhits_config)
        model.eval()
        
        input_tensor = torch.randn(16, basic_nhits_config.input_size)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        for _ in range(50):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(input_tensor)
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        mean_time = np.mean(times)
        
        # Should be reasonably fast (adjust threshold as needed)
        assert mean_time < 100.0, f"Inference too slow: {mean_time:.2f}ms"
    
    @skip_if_no_gpu
    def test_gpu_memory_efficiency(self, basic_nhits_config):
        """Test GPU memory efficiency."""
        gpu_config = basic_nhits_config
        gpu_config.use_gpu = True
        
        with gpu_memory_context("nhits_memory_test"):
            model = OptimizedNHITS(gpu_config).cuda()
            
            # Test with increasing batch sizes
            batch_sizes = [1, 8, 16, 32]
            
            for batch_size in batch_sizes:
                input_tensor = torch.randn(batch_size, gpu_config.input_size, device='cuda')
                
                with torch.no_grad():
                    output = model(input_tensor)
                
                # Clean up
                del input_tensor, output
                torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])