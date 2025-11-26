"""
Neural Models Client Tests
========================

Test suite for the neural models client functionality.
"""

import pytest
import asyncio
from uuid import uuid4, UUID
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from supabase_client.clients.neural_models import (
    NeuralModelsClient,
    CreateModelRequest,
    StartTrainingRequest,
    PredictionRequest,
    ModelStatus,
    TrainingStatus
)
from supabase_client.client import AsyncSupabaseClient
from supabase_client.models.database_models import NeuralModel, TrainingRun

class TestNeuralModelsClient:
    """Test suite for NeuralModelsClient."""
    
    @pytest.fixture
    def mock_supabase(self):
        """Create mock Supabase client."""
        return AsyncMock(spec=AsyncSupabaseClient)
    
    @pytest.fixture
    def neural_client(self, mock_supabase):
        """Create neural models client with mock."""
        return NeuralModelsClient(mock_supabase)
    
    @pytest.fixture
    def sample_user_id(self):
        """Sample user ID for testing."""
        return uuid4()
    
    @pytest.fixture
    def sample_model_data(self, sample_user_id):
        """Sample model data for testing."""
        return {
            "id": str(uuid4()),
            "user_id": str(sample_user_id),
            "name": "Test Model",
            "model_type": "lstm",
            "status": ModelStatus.TRAINING.value,
            "symbols": ["AAPL", "GOOGL"],
            "configuration": {"layers": 3, "units": 128},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
    
    @pytest.mark.asyncio
    async def test_create_model_success(self, neural_client, mock_supabase, sample_user_id):
        """Test successful model creation."""
        # Arrange
        request = CreateModelRequest(
            name="Test LSTM Model",
            model_type="lstm",
            symbols=["AAPL", "GOOGL"],
            configuration={"layers": 3, "units": 128}
        )
        
        mock_supabase.select.return_value = [{"id": str(sample_user_id)}]  # User exists
        mock_supabase.count.return_value = 2  # Under limit
        mock_supabase.insert.return_value = [{"id": "model-123", "name": "Test LSTM Model"}]
        
        # Act
        result, error = await neural_client.create_model(sample_user_id, request)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["name"] == "Test LSTM Model"
        mock_supabase.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_model_user_not_found(self, neural_client, mock_supabase, sample_user_id):
        """Test model creation with non-existent user."""
        # Arrange
        request = CreateModelRequest(
            name="Test Model",
            model_type="lstm",
            symbols=["AAPL"]
        )
        
        mock_supabase.select.return_value = []  # User not found
        
        # Act
        result, error = await neural_client.create_model(sample_user_id, request)
        
        # Assert
        assert result is None
        assert error == "User not found"
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_model_limit_exceeded(self, neural_client, mock_supabase, sample_user_id):
        """Test model creation when limit is exceeded."""
        # Arrange
        request = CreateModelRequest(
            name="Test Model",
            model_type="lstm",
            symbols=["AAPL"]
        )
        
        mock_supabase.select.return_value = [{"id": str(sample_user_id)}]
        mock_supabase.count.return_value = 10  # At limit
        
        # Act
        result, error = await neural_client.create_model(sample_user_id, request)
        
        # Assert
        assert result is None
        assert "Maximum number of models reached" in error
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_model_success(self, neural_client, mock_supabase, sample_model_data):
        """Test successful model retrieval."""
        # Arrange
        model_id = sample_model_data["id"]
        mock_supabase.select.return_value = [sample_model_data]
        
        # Act
        result, error = await neural_client.get_model(model_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["id"] == model_id
        assert result["name"] == "Test Model"
    
    @pytest.mark.asyncio
    async def test_get_model_not_found(self, neural_client, mock_supabase):
        """Test model retrieval when model doesn't exist."""
        # Arrange
        model_id = str(uuid4())
        mock_supabase.select.return_value = []
        
        # Act
        result, error = await neural_client.get_model(model_id)
        
        # Assert
        assert result is None
        assert error == "Model not found"
    
    @pytest.mark.asyncio
    async def test_start_training_success(self, neural_client, mock_supabase, sample_model_data):
        """Test successful training start."""
        # Arrange
        request = StartTrainingRequest(
            model_id=sample_model_data["id"],
            training_data={"data": "sample"},
            epochs=100,
            batch_size=32
        )
        
        mock_supabase.select.return_value = [sample_model_data]  # Model exists
        mock_supabase.insert.return_value = [{"id": "training-123", "status": "running"}]
        mock_supabase.update.return_value = [{}]
        
        # Act
        result, error = await neural_client.start_training(request)
        
        # Assert
        assert error is None
        assert result is not None
        mock_supabase.insert.assert_called_once()
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_training_model_not_found(self, neural_client, mock_supabase):
        """Test training start with non-existent model."""
        # Arrange
        request = StartTrainingRequest(
            model_id=str(uuid4()),
            training_data={"data": "sample"}
        )
        
        mock_supabase.select.return_value = []  # Model not found
        
        # Act
        result, error = await neural_client.start_training(request)
        
        # Assert
        assert result is None
        assert error == "Model not found"
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_training_status_success(self, neural_client, mock_supabase):
        """Test successful training status retrieval."""
        # Arrange
        training_id = str(uuid4())
        training_data = {
            "id": training_id,
            "status": TrainingStatus.RUNNING.value,
            "progress": 0.5,
            "current_epoch": 50,
            "total_epochs": 100
        }
        
        mock_supabase.select.return_value = [training_data]
        
        # Act
        result, error = await neural_client.get_training_status(training_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["status"] == TrainingStatus.RUNNING.value
        assert result["progress"] == 0.5
    
    @pytest.mark.asyncio
    async def test_make_prediction_success(self, neural_client, mock_supabase, sample_model_data):
        """Test successful prediction."""
        # Arrange
        request = PredictionRequest(
            model_id=sample_model_data["id"],
            input_data={"features": [1, 2, 3]},
            symbols=["AAPL"]
        )
        
        # Mock model exists and is trained
        trained_model = sample_model_data.copy()
        trained_model["status"] = ModelStatus.TRAINED.value
        mock_supabase.select.return_value = [trained_model]
        mock_supabase.insert.return_value = [{"id": "pred-123", "prediction": 0.75}]
        
        # Act
        result, error = await neural_client.make_prediction(request)
        
        # Assert
        assert error is None
        assert result is not None
        mock_supabase.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_prediction_model_not_trained(self, neural_client, mock_supabase, sample_model_data):
        """Test prediction with untrained model."""
        # Arrange
        request = PredictionRequest(
            model_id=sample_model_data["id"],
            input_data={"features": [1, 2, 3]},
            symbols=["AAPL"]
        )
        
        # Model exists but not trained
        mock_supabase.select.return_value = [sample_model_data]  # Status is TRAINING
        
        # Act
        result, error = await neural_client.make_prediction(request)
        
        # Assert
        assert result is None
        assert "Model is not trained" in error
        mock_supabase.insert.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_list_user_models(self, neural_client, mock_supabase, sample_user_id, sample_model_data):
        """Test listing user models."""
        # Arrange
        models_data = [sample_model_data, {**sample_model_data, "id": str(uuid4()), "name": "Model 2"}]
        mock_supabase.select.return_value = models_data
        
        # Act
        result, error = await neural_client.list_user_models(sample_user_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert len(result) == 2
        assert all(model["user_id"] == str(sample_user_id) for model in result)
    
    @pytest.mark.asyncio
    async def test_update_model_success(self, neural_client, mock_supabase, sample_model_data):
        """Test successful model update."""
        # Arrange
        model_id = sample_model_data["id"]
        updates = {"name": "Updated Model Name", "configuration": {"layers": 4}}
        
        mock_supabase.select.return_value = [sample_model_data]  # Model exists
        updated_model = {**sample_model_data, **updates}
        mock_supabase.update.return_value = [updated_model]
        
        # Act
        result, error = await neural_client.update_model(model_id, updates)
        
        # Assert
        assert error is None
        assert result is not None
        assert result["name"] == "Updated Model Name"
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_model_success(self, neural_client, mock_supabase, sample_model_data):
        """Test successful model deletion."""
        # Arrange
        model_id = sample_model_data["id"]
        mock_supabase.select.return_value = [sample_model_data]  # Model exists
        mock_supabase.update.return_value = [{}]  # Soft delete
        
        # Act
        success, error = await neural_client.delete_model(model_id)
        
        # Assert
        assert error is None
        assert success is True
        mock_supabase.update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_model_performance(self, neural_client, mock_supabase, sample_model_data):
        """Test model performance retrieval."""
        # Arrange
        model_id = sample_model_data["id"]
        mock_supabase.select.return_value = [sample_model_data]  # Model exists
        
        predictions_data = [
            {"id": "pred-1", "created_at": datetime.utcnow().isoformat(), "accuracy": 0.8},
            {"id": "pred-2", "created_at": datetime.utcnow().isoformat(), "accuracy": 0.9}
        ]
        
        # Mock multiple select calls for predictions
        mock_supabase.select.side_effect = [
            [sample_model_data],  # Model exists check
            predictions_data  # Recent predictions
        ]
        
        # Act
        result, error = await neural_client.get_model_performance(model_id)
        
        # Assert
        assert error is None
        assert result is not None
        assert "average_accuracy" in result
        assert "prediction_count" in result
        assert result["prediction_count"] == 2
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, neural_client, mock_supabase, sample_user_id):
        """Test exception handling in client methods."""
        # Arrange
        request = CreateModelRequest(
            name="Test Model",
            model_type="lstm",
            symbols=["AAPL"]
        )
        
        mock_supabase.select.side_effect = Exception("Database error")
        
        # Act
        result, error = await neural_client.create_model(sample_user_id, request)
        
        # Assert
        assert result is None
        assert "Model creation failed" in error
        assert "Database error" in error

@pytest.mark.asyncio
async def test_neural_models_integration():
    """Integration test for neural models workflow."""
    # This would be a higher-level test that exercises multiple methods
    # in a realistic workflow scenario
    pass

if __name__ == "__main__":
    pytest.main([__file__])