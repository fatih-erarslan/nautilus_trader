"""
Unit Tests for Neural Model Manager.

Tests for model lifecycle management including:
- Model training and versioning
- Performance monitoring and evaluation
- Model registry operations
- Automated retraining workflows
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import json
import os
from pathlib import Path

# Import the module under test
import sys
sys.path.insert(0, '/workspaces/ai-news-trader/src')

from neural_forecast.neural_model_manager import NeuralModelManager, ModelMetrics, ModelVersion
from neural_forecast.nhits_forecaster import NHITSForecaster


class TestNeuralModelManager:
    """Test suite for NeuralModelManager class."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup is handled by tempfile
    
    @pytest.fixture
    def sample_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        values = np.random.randn(100).cumsum() + 100
        
        return {
            'ds': dates.tolist(),
            'y': values.tolist(),
            'unique_id': 'test_series'
        }
    
    @pytest.fixture
    def model_manager(self, temp_registry_path):
        """Create NeuralModelManager instance for testing."""
        return NeuralModelManager(
            model_registry_path=temp_registry_path,
            auto_retrain_enabled=False,  # Disable for testing
            max_model_versions=3
        )
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample model metrics."""
        return ModelMetrics(
            mae=0.5,
            mse=0.25,
            rmse=0.5,
            mape=5.0,
            smape=4.5,
            r2_score=0.85,
            training_time=120.0,
            prediction_time=0.1,
            timestamp=datetime.now().isoformat()
        )
    
    def test_model_manager_initialization(self, temp_registry_path):
        """Test model manager initialization."""
        manager = NeuralModelManager(model_registry_path=temp_registry_path)
        
        assert manager.registry_path.exists()
        assert manager.auto_retrain_enabled == True  # Default
        assert manager.max_model_versions == 10  # Default
        assert isinstance(manager.model_versions, dict)
        assert isinstance(manager.active_models, dict)
    
    def test_registry_path_creation(self, temp_registry_path):
        """Test registry path creation."""
        registry_path = Path(temp_registry_path) / "new_registry"
        manager = NeuralModelManager(model_registry_path=str(registry_path))
        
        assert manager.registry_path.exists()
        assert manager.registry_db_path.parent.exists()
    
    def test_model_id_generation(self, model_manager):
        """Test model ID generation from configuration."""
        config1 = {'input_size': 24, 'horizon': 12, 'learning_rate': 0.001}
        config2 = {'input_size': 24, 'horizon': 12, 'learning_rate': 0.001}
        config3 = {'input_size': 48, 'horizon': 12, 'learning_rate': 0.001}
        
        id1 = model_manager._generate_model_id(config1)
        id2 = model_manager._generate_model_id(config2)
        id3 = model_manager._generate_model_id(config3)
        
        assert id1 == id2  # Same config should generate same ID
        assert id1 != id3  # Different config should generate different ID
        assert len(id1) == 12  # Should be 12 characters
    
    def test_version_generation(self, model_manager):
        """Test version number generation."""
        model_id = "test_model_123"
        
        # First version
        version1 = model_manager._generate_version(model_id)
        assert version1 == "v1.0.0"
        
        # Add a mock version to test increment
        mock_version = ModelVersion(
            version="v1.0.5",
            model_id=model_id,
            created_at=datetime.now().isoformat(),
            model_path="/tmp/test",
            metrics=ModelMetrics(0, 0, 0, 0, 0, 0, 0, 0, datetime.now().isoformat()),
            config={},
            status="active"
        )
        
        model_manager.model_versions[model_id] = [mock_version]
        version2 = model_manager._generate_version(model_id)
        assert version2 == "v1.0.6"
    
    @pytest.mark.asyncio
    async def test_model_training_mock(self, model_manager, sample_data):
        """Test model training with mocked components."""
        config = {
            'input_size': 24,
            'horizon': 12,
            'max_epochs': 1,
            'enable_gpu': False
        }
        
        # Mock the NHITSForecaster
        with patch('neural_forecast.neural_model_manager.NHITSForecaster') as mock_forecaster_class:
            mock_forecaster = Mock()
            mock_forecaster_class.return_value = mock_forecaster
            
            # Mock the fit method
            mock_forecaster.fit = AsyncMock(return_value={'success': True})
            mock_forecaster.save_model = Mock(return_value="/tmp/test_model.pkl")
            
            # Mock the evaluation
            with patch.object(model_manager, '_evaluate_model') as mock_evaluate:
                mock_metrics = ModelMetrics(
                    mae=0.5, mse=0.25, rmse=0.5, mape=5.0, smape=4.5,
                    r2_score=0.85, training_time=120.0, prediction_time=0.1,
                    timestamp=datetime.now().isoformat()
                )
                mock_evaluate.return_value = mock_metrics
                
                # Test training
                result = await model_manager.train_model(
                    data=sample_data,
                    config=config,
                    model_name="test_model"
                )
                
                assert result['success'] == True
                assert 'model_id' in result
                assert 'version' in result
                assert result['model_name'] == "test_model"
                
                # Check that model was added to registry
                assert "test_model" in model_manager.active_models
    
    @pytest.mark.asyncio
    async def test_model_training_failure(self, model_manager, sample_data):
        """Test handling of model training failure."""
        config = {'invalid': 'config'}
        
        with patch('neural_forecast.neural_model_manager.NHITSForecaster') as mock_forecaster_class:
            mock_forecaster = Mock()
            mock_forecaster_class.return_value = mock_forecaster
            mock_forecaster.fit = AsyncMock(return_value={'success': False, 'error': 'Training failed'})
            
            result = await model_manager.train_model(
                data=sample_data,
                config=config,
                model_name="failing_model"
            )
            
            assert result['success'] == False
            assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, model_manager, sample_data):
        """Test model performance evaluation."""
        # Create mock forecaster
        mock_forecaster = Mock()
        mock_forecaster.predict = AsyncMock(return_value={
            'success': True,
            'point_forecast': [100.5, 101.2, 99.8, 102.1, 100.9]
        })
        
        model_manager.forecaster = mock_forecaster
        
        # Create validation data
        val_data = pd.DataFrame({
            'y': [100.0, 101.0, 100.0, 102.0, 101.0],
            'ds': pd.date_range(start='2023-01-01', periods=5, freq='H'),
            'unique_id': 'test'
        })
        
        metrics = await model_manager._evaluate_model(sample_data, val_data)
        
        assert isinstance(metrics, ModelMetrics)
        assert metrics.mae >= 0
        assert metrics.mse >= 0
        assert metrics.rmse >= 0
        assert metrics.prediction_time >= 0
    
    @pytest.mark.asyncio
    async def test_model_evaluation_prediction_failure(self, model_manager, sample_data):
        """Test model evaluation with prediction failure."""
        mock_forecaster = Mock()
        mock_forecaster.predict = AsyncMock(return_value={'success': False, 'error': 'Prediction failed'})
        
        model_manager.forecaster = mock_forecaster
        
        metrics = await model_manager._evaluate_model(sample_data)
        
        # Should return default metrics on failure
        assert metrics.mae == float('inf')
        assert metrics.r2_score == -1.0
    
    @pytest.mark.asyncio
    async def test_model_loading_mock(self, model_manager):
        """Test model loading functionality."""
        # Setup mock model in registry
        model_manager.active_models["test_model"] = "v1.0.0"
        
        mock_version = ModelVersion(
            version="v1.0.0",
            model_id="test_123",
            created_at=datetime.now().isoformat(),
            model_path="/tmp/test_model.pkl",
            metrics=ModelMetrics(0.5, 0.25, 0.5, 5.0, 4.5, 0.85, 120.0, 0.1, datetime.now().isoformat()),
            config={'input_size': 24, 'horizon': 12},
            status="active"
        )
        
        model_manager.model_versions["test_123"] = [mock_version]
        
        # Mock the forecaster loading
        with patch('neural_forecast.neural_model_manager.NHITSForecaster') as mock_forecaster_class:
            mock_forecaster = Mock()
            mock_forecaster_class.return_value = mock_forecaster
            mock_forecaster.load_model = Mock(return_value=True)
            
            result = await model_manager.load_model("test_model")
            
            assert result['success'] == True
            assert result['model_name'] == "test_model"
            assert result['version'] == "v1.0.0"
    
    @pytest.mark.asyncio
    async def test_model_loading_not_found(self, model_manager):
        """Test model loading with non-existent model."""
        result = await model_manager.load_model("non_existent_model")
        
        assert result['success'] == False
        assert 'error' in result
    
    @pytest.mark.asyncio
    async def test_prediction_with_loaded_model(self, model_manager, sample_data):
        """Test prediction with loaded model."""
        # Mock forecaster
        mock_forecaster = Mock()
        mock_forecaster.predict = AsyncMock(return_value={
            'success': True,
            'point_forecast': [100.5, 101.2],
            'prediction_intervals': {}
        })
        
        model_manager.forecaster = mock_forecaster
        
        result = await model_manager.predict(sample_data, model_name="test_model")
        
        assert result['success'] == True
        assert 'point_forecast' in result
    
    @pytest.mark.asyncio
    async def test_prediction_without_loaded_model(self, model_manager, sample_data):
        """Test prediction fails without loaded model."""
        with pytest.raises(ValueError, match="No model loaded"):
            await model_manager.predict(sample_data)
    
    def test_model_listing(self, model_manager):
        """Test model listing functionality."""
        # Add mock models to registry
        model_manager.active_models["model1"] = "v1.0.0"
        model_manager.active_models["model2"] = "v1.0.1"
        
        mock_version1 = ModelVersion(
            version="v1.0.0",
            model_id="model1_123",
            created_at=datetime.now().isoformat(),
            model_path="/tmp/model1.pkl",
            metrics=ModelMetrics(0.5, 0.25, 0.5, 5.0, 4.5, 0.85, 120.0, 0.1, datetime.now().isoformat()),
            config={'input_size': 24},
            status="active"
        )
        
        model_manager.model_versions["model1_123"] = [mock_version1]
        
        result = model_manager.list_models()
        
        assert 'active_models' in result
        assert 'total_model_families' in result
        assert len(result['active_models']) >= 1
    
    def test_performance_history_tracking(self, model_manager):
        """Test performance history tracking."""
        model_name = "test_model"
        
        # Simulate some prediction history
        model_manager.performance_history[model_name] = [
            {
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'prediction_time': 0.1
            },
            {
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'prediction_time': 0.0
            }
        ]
        
        history = model_manager.get_model_performance_history(model_name)
        
        assert history['model_name'] == model_name
        assert len(history['history']) == 2
        assert 'summary' in history
        assert history['summary']['total_predictions'] == 2
        assert history['summary']['successful_predictions'] == 1
        assert history['summary']['success_rate'] == 0.5
    
    def test_performance_history_empty(self, model_manager):
        """Test performance history for non-existent model."""
        history = model_manager.get_model_performance_history("non_existent")
        
        assert history['model_name'] == "non_existent"
        assert history['history'] == []
        assert history['summary']['total_predictions'] == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_versions(self, model_manager):
        """Test cleanup of old model versions."""
        model_id = "test_model_123"
        
        # Create more versions than max_model_versions
        old_versions = []
        for i in range(5):  # More than max_model_versions (3)
            version = ModelVersion(
                version=f"v1.0.{i}",
                model_id=model_id,
                created_at=(datetime.now() - timedelta(days=i)).isoformat(),
                model_path=f"/tmp/model_{i}.pkl",
                metrics=ModelMetrics(0.5, 0.25, 0.5, 5.0, 4.5, 0.85, 120.0, 0.1, datetime.now().isoformat()),
                config={'input_size': 24},
                status="active"
            )
            old_versions.append(version)
        
        model_manager.model_versions[model_id] = old_versions
        
        # Mock file system operations
        with patch('shutil.rmtree') as mock_rmtree:
            await model_manager._cleanup_old_versions(model_id)
            
            # Should keep only max_model_versions (3) versions
            assert len(model_manager.model_versions[model_id]) == model_manager.max_model_versions
            
            # Should have called rmtree for removed versions
            assert mock_rmtree.call_count == 2  # 5 - 3 = 2 removed
    
    @pytest.mark.asyncio
    async def test_retraining_needed(self, model_manager, sample_data):
        """Test automatic retraining logic."""
        model_name = "test_model"
        
        # Setup model with poor performance history
        model_manager.performance_history[model_name] = [
            {'success': False, 'timestamp': datetime.now().isoformat()},
            {'success': False, 'timestamp': datetime.now().isoformat()},
            {'success': False, 'timestamp': datetime.now().isoformat()}
        ]
        
        # Enable auto-retrain
        model_manager.auto_retrain_enabled = True
        model_manager.performance_threshold = 0.5  # 50% success rate required
        
        # Mock model info and training
        model_manager.active_models[model_name] = "v1.0.0"
        mock_version = ModelVersion(
            version="v1.0.0",
            model_id="test_123",
            created_at=datetime.now().isoformat(),
            model_path="/tmp/test.pkl",
            metrics=ModelMetrics(0.5, 0.25, 0.5, 5.0, 4.5, 0.85, 120.0, 0.1, datetime.now().isoformat()),
            config={'input_size': 24, 'horizon': 12},
            status="active"
        )
        model_manager.model_versions["test_123"] = [mock_version]
        
        with patch.object(model_manager, 'train_model') as mock_train:
            mock_train.return_value = {'success': True, 'version': 'v1.0.1'}
            
            result = await model_manager.retrain_if_needed(model_name, sample_data)
            
            assert result['retrain_needed'] == True
            assert 'retrain_result' in result
    
    @pytest.mark.asyncio
    async def test_retraining_not_needed(self, model_manager, sample_data):
        """Test retraining not needed with good performance."""
        model_name = "test_model"
        
        # Setup model with good performance history
        model_manager.performance_history[model_name] = [
            {'success': True, 'timestamp': datetime.now().isoformat()},
            {'success': True, 'timestamp': datetime.now().isoformat()},
            {'success': True, 'timestamp': datetime.now().isoformat()}
        ]
        
        model_manager.auto_retrain_enabled = True
        model_manager.performance_threshold = 0.5
        
        result = await model_manager.retrain_if_needed(model_name, sample_data)
        
        assert result['retrain_needed'] == False
        assert 'Performance acceptable' in result['reason']
    
    def test_model_config_export(self, model_manager):
        """Test model configuration export."""
        model_name = "test_model"
        
        # Setup model
        model_manager.active_models[model_name] = "v1.0.0"
        mock_version = ModelVersion(
            version="v1.0.0",
            model_id="test_123",
            created_at=datetime.now().isoformat(),
            model_path="/tmp/test.pkl",
            metrics=ModelMetrics(0.5, 0.25, 0.5, 5.0, 4.5, 0.85, 120.0, 0.1, datetime.now().isoformat()),
            config={'input_size': 24, 'horizon': 12, 'learning_rate': 0.001},
            status="active",
            notes="Test model"
        )
        model_manager.model_versions["test_123"] = [mock_version]
        
        config = model_manager.export_model_config(model_name)
        
        assert config['model_name'] == model_name
        assert config['version'] == "v1.0.0"
        assert 'config' in config
        assert config['config']['input_size'] == 24
        assert 'metrics' in config
    
    def test_model_config_export_not_found(self, model_manager):
        """Test model configuration export for non-existent model."""
        config = model_manager.export_model_config("non_existent")
        
        assert 'error' in config
    
    def test_registry_save_and_load(self, model_manager, temp_registry_path):
        """Test registry persistence."""
        # Add test data
        model_manager.active_models["test_model"] = "v1.0.0"
        model_manager.performance_history["test_model"] = [
            {'success': True, 'timestamp': datetime.now().isoformat()}
        ]
        
        # Save registry
        model_manager._save_registry()
        
        # Verify file exists
        assert model_manager.registry_db_path.exists()
        
        # Create new manager and load
        new_manager = NeuralModelManager(model_registry_path=temp_registry_path)
        
        assert "test_model" in new_manager.active_models
        assert "test_model" in new_manager.performance_history
    
    def test_health_check(self, model_manager):
        """Test health check functionality."""
        health = model_manager.health_check()
        
        assert 'status' in health
        assert 'timestamp' in health
        assert 'registry_path' in health
        assert 'active_models' in health
        assert 'forecaster_loaded' in health
        
        # Should be healthy by default
        assert health['status'] in ['healthy', 'warning']


# Test runner for development
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])