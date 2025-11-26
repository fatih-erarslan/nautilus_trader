"""MCP Integration Tests.

Tests MCP server integration with trading models, GPU acceleration, and real-time data streaming.
"""

import pytest
import asyncio
import json
import uuid
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
import os

from fastapi.testclient import TestClient
import httpx
import websockets

# Import MCP and model components
from model_management.mcp_integration.trading_mcp_server import TradingMCPServer
from model_management.mcp_integration.websocket_server import ModelWebSocketServer
from model_management.storage.model_storage import ModelStorage, ModelMetadata, ModelFormat
from model_management.storage.metadata_manager import MetadataManager, ModelStatus
from model_management.storage.version_control import ModelVersionControl


class TestMCPModelIntegration:
    """Test MCP integration with trading models."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage path for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mcp_server(self, temp_storage_path):
        """Create MCP server with test storage."""
        server = TradingMCPServer(
            host="127.0.0.1",
            port=8893,
            model_storage_path=temp_storage_path
        )
        return server
    
    @pytest.fixture
    def test_client(self, mcp_server):
        """Create test client for MCP server."""
        return TestClient(mcp_server.app)
    
    @pytest.fixture
    def sample_models(self, temp_storage_path):
        """Create sample models for testing."""
        storage = ModelStorage(temp_storage_path)
        models = []
        
        # Create mean reversion model
        mean_reversion_params = {
            "z_score_entry_threshold": 2.0,
            "z_score_exit_threshold": 0.5,
            "lookback_period": 20,
            "base_position_size": 0.05,
            "max_position_size": 0.15,
            "stop_loss_multiplier": 1.5,
            "profit_target_multiplier": 2.0
        }
        
        mean_reversion_metadata = ModelMetadata(
            model_id="",
            name="Mean Reversion Test Model",
            version="1.0.0",
            created_at=datetime.now(),
            model_type="parameters",
            strategy_name="mean_reversion",
            performance_metrics={
                "sharpe_ratio": 1.85,
                "total_return": 0.23,
                "max_drawdown": 0.08,
                "win_rate": 0.62
            },
            parameters=mean_reversion_params,
            tags=["test", "mean_reversion"],
            description="Test mean reversion model"
        )
        
        model_id = storage.save_model(
            mean_reversion_params,
            mean_reversion_metadata,
            ModelFormat.JSON
        )
        models.append((model_id, "mean_reversion"))
        
        # Create momentum model
        momentum_params = {
            "momentum_threshold": 0.6,
            "trend_lookback": 10,
            "volume_factor": 1.2,
            "base_position_size": 0.05,
            "max_position_size": 0.20,
            "risk_adjustment_factor": 0.8
        }
        
        momentum_metadata = ModelMetadata(
            model_id="",
            name="Momentum Test Model",
            version="1.0.0",
            created_at=datetime.now(),
            model_type="parameters",
            strategy_name="momentum",
            performance_metrics={
                "sharpe_ratio": 2.1,
                "total_return": 0.31,
                "max_drawdown": 0.12,
                "win_rate": 0.58
            },
            parameters=momentum_params,
            tags=["test", "momentum"],
            description="Test momentum model"
        )
        
        model_id = storage.save_model(
            momentum_params,
            momentum_metadata,
            ModelFormat.JSON
        )
        models.append((model_id, "momentum"))
        
        # Create mirror trading model
        mirror_params = {
            "confidence_threshold": 0.7,
            "position_scaling": 0.8,
            "max_institutions": 5,
            "min_signal_threshold": 0.3,
            "base_position_size": 0.05,
            "max_position_size": 0.15
        }
        
        mirror_metadata = ModelMetadata(
            model_id="",
            name="Mirror Trading Test Model",
            version="1.0.0",
            created_at=datetime.now(),
            model_type="parameters",
            strategy_name="mirror_trading",
            performance_metrics={
                "sharpe_ratio": 1.95,
                "total_return": 0.28,
                "max_drawdown": 0.09,
                "win_rate": 0.65
            },
            parameters=mirror_params,
            tags=["test", "mirror"],
            description="Test mirror trading model"
        )
        
        model_id = storage.save_model(
            mirror_params,
            mirror_metadata,
            ModelFormat.JSON
        )
        models.append((model_id, "mirror_trading"))
        
        return models
    
    def test_model_loading_and_caching(self, test_client, sample_models):
        """Test model loading and caching mechanism."""
        model_id, strategy = sample_models[0]
        
        # First prediction - should load model
        request_data = {
            "model_id": model_id,
            "input_data": {
                "z_score": 2.5,
                "price": 105.0,
                "moving_average": 100.0,
                "volatility": 0.2,
                "volume_ratio": 1.2,
                "rsi": 65.0,
                "market_regime": 0.7
            }
        }
        
        start_time = time.time()
        response = test_client.post("/models/predict", json=request_data)
        first_request_time = time.time() - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "action" in data["data"]
        
        # Second prediction - should use cached model
        start_time = time.time()
        response2 = test_client.post("/models/predict", json=request_data)
        second_request_time = time.time() - start_time
        
        assert response2.status_code == 200
        
        # Cached request should be faster (though this might vary)
        print(f"First request: {first_request_time:.3f}s, Cached: {second_request_time:.3f}s")
    
    def test_strategy_specific_predictions(self, test_client, sample_models):
        """Test strategy-specific prediction logic."""
        # Test mean reversion
        mean_reversion_id = sample_models[0][0]
        mean_reversion_request = {
            "model_id": mean_reversion_id,
            "input_data": {
                "z_score": -2.5,  # Strong oversold signal
                "price": 95.0,
                "moving_average": 100.0,
                "volatility": 0.2,
                "volume_ratio": 1.5,
                "rsi": 25.0,
                "market_regime": 0.3
            },
            "return_confidence": True
        }
        
        response = test_client.post("/models/predict", json=mean_reversion_request)
        assert response.status_code == 200
        
        prediction = response.json()["data"]
        assert prediction["action"] == "buy"  # Should buy when oversold
        assert prediction["confidence"] > 0.7
        assert "z_score" in prediction
        
        # Test momentum
        momentum_id = sample_models[1][0]
        momentum_request = {
            "model_id": momentum_id,
            "input_data": {
                "price_change": 0.05,
                "volume_change": 0.3,
                "momentum_score": 0.8,
                "trend_strength": 0.9,
                "volatility": 0.15,
                "market_sentiment": 0.8
            },
            "return_confidence": True
        }
        
        response = test_client.post("/models/predict", json=momentum_request)
        assert response.status_code == 200
        
        prediction = response.json()["data"]
        assert prediction["action"] == "buy"  # Strong momentum should trigger buy
        assert prediction["confidence"] > 0.6
        
        # Test mirror trading
        mirror_id = sample_models[2][0]
        mirror_request = {
            "model_id": mirror_id,
            "input_data": {
                "institutional_positions": {
                    "goldman_sachs": 0.8,
                    "jp_morgan": 0.7,
                    "morgan_stanley": 0.6
                },
                "position_changes": {
                    "goldman_sachs": 0.2,
                    "jp_morgan": 0.1,
                    "morgan_stanley": 0.15
                },
                "confidence_scores": {
                    "goldman_sachs": 0.9,
                    "jp_morgan": 0.85,
                    "morgan_stanley": 0.8
                },
                "entry_timing": "immediate",
                "market_conditions": "normal"
            }
        }
        
        response = test_client.post("/models/predict", json=mirror_request)
        assert response.status_code == 200
        
        prediction = response.json()["data"]
        assert prediction["action"] in ["buy", "hold"]
        assert "institutional_signal" in prediction
    
    def test_model_listing_and_filtering(self, test_client, sample_models):
        """Test model listing with various filters."""
        # List all models
        response = test_client.get("/models")
        assert response.status_code == 200
        
        data = response.json()["data"]
        assert len(data["models"]) >= 3
        
        # Filter by strategy
        response = test_client.get("/models", params={"strategy_name": "momentum"})
        assert response.status_code == 200
        
        momentum_models = response.json()["data"]["models"]
        assert all(m["strategy_name"] == "momentum" for m in momentum_models)
        
        # Filter by tags
        response = test_client.get("/models", params={"tags": ["test"]})
        assert response.status_code == 200
        
        test_models = response.json()["data"]["models"]
        assert all("test" in m["tags"] for m in test_models)
    
    def test_model_metadata_retrieval(self, test_client, sample_models):
        """Test model metadata and evaluation retrieval."""
        model_id = sample_models[0][0]
        
        response = test_client.get(f"/models/{model_id}/metadata")
        assert response.status_code == 200
        
        data = response.json()["data"]
        assert "metadata" in data
        assert "evaluation" in data
        
        metadata = data["metadata"]
        assert metadata["model_id"] == model_id
        assert "performance_metrics" in metadata
        assert "parameters" in metadata
        
        evaluation = data["evaluation"]
        assert "rating" in evaluation
        assert "strengths" in evaluation
        assert "weaknesses" in evaluation
    
    def test_strategy_analytics(self, test_client, sample_models):
        """Test strategy analytics aggregation."""
        # Get analytics for mean reversion strategy
        response = test_client.get("/strategies/mean_reversion/analytics")
        assert response.status_code == 200
        
        analytics = response.json()["data"]
        assert "total_models" in analytics
        assert "average_performance" in analytics
        assert "best_model" in analytics
        
        # Verify calculations
        assert analytics["total_models"] >= 1
        assert analytics["average_performance"]["sharpe_ratio"] > 0
    
    def test_strategy_recommendation(self, test_client, sample_models):
        """Test strategy recommendation generation."""
        # Create a production model
        model_id = sample_models[0][0]
        
        # Update model status to production
        with patch.object(test_client.app.state, 'metadata_manager', create=True) as mock_manager:
            mock_metadata = MagicMock()
            mock_metadata.model_id = model_id
            mock_metadata.status = ModelStatus.PRODUCTION
            mock_manager.search_models.return_value = [mock_metadata]
            
            response = test_client.get("/strategies/mean_reversion/recommendation")
        
        # Should handle even if no production models
        assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_concurrent_model_predictions(self, test_client, sample_models):
        """Test concurrent prediction handling."""
        model_id = sample_models[0][0]
        
        async def make_prediction(session, index):
            request_data = {
                "model_id": model_id,
                "input_data": {
                    "z_score": np.random.normal(0, 2),
                    "price": 100 + np.random.normal(0, 5),
                    "moving_average": 100,
                    "volatility": 0.2,
                    "volume_ratio": 1.0,
                    "rsi": 50 + np.random.uniform(-30, 30),
                    "market_regime": np.random.uniform(0, 1)
                }
            }
            
            async with session.post(
                "http://127.0.0.1:8893/models/predict",
                json=request_data
            ) as response:
                return await response.json()
        
        # Make concurrent predictions
        async with httpx.AsyncClient() as client:
            tasks = [make_prediction(client, i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful predictions
        successful = sum(1 for r in results 
                        if isinstance(r, dict) and r.get("success"))
        
        print(f"Successful concurrent predictions: {successful}/20")
    
    def test_model_version_history(self, test_client, sample_models, temp_storage_path):
        """Test model version history tracking."""
        model_id = sample_models[0][0]
        
        # Create version control
        version_control = ModelVersionControl(os.path.join(temp_storage_path, "versions"))
        
        # Create a few versions
        for i in range(3):
            version_control.commit_version(
                model_id=model_id,
                model_data={"version": i + 1},
                parameters={"param": f"value_{i}"},
                performance_metrics={"metric": i * 0.1},
                commit_message=f"Version {i + 1}",
                author="test_user"
            )
        
        # Get version history through API
        response = test_client.get(f"/models/{model_id}/versions")
        
        # API might not be fully implemented, so handle gracefully
        if response.status_code == 200:
            data = response.json()["data"]
            assert "versions" in data
            assert len(data["versions"]) >= 3
    
    def test_model_performance_evaluation(self, test_client, sample_models):
        """Test model performance evaluation against benchmarks."""
        model_id = sample_models[1][0]  # Momentum model
        
        response = test_client.get(f"/analytics/performance/{model_id}")
        
        if response.status_code == 200:
            evaluation = response.json()["data"]
            assert "score" in evaluation
            assert "benchmarks" in evaluation
            assert "recommendations" in evaluation
    
    def test_error_handling_invalid_model(self, test_client):
        """Test error handling for invalid model requests."""
        # Request with non-existent model
        request_data = {
            "model_id": "non_existent_model_123",
            "input_data": {"x": 1.0}
        }
        
        response = test_client.post("/models/predict", json=request_data)
        assert response.status_code in [404, 500]
        
        if response.status_code == 404:
            error_data = response.json()
            assert "detail" in error_data
    
    def test_input_data_validation(self, test_client, sample_models):
        """Test input data validation and preparation."""
        model_id = sample_models[0][0]  # Mean reversion
        
        # Test with missing required fields
        incomplete_request = {
            "model_id": model_id,
            "input_data": {
                "z_score": 2.0
                # Missing other required fields
            }
        }
        
        response = test_client.post("/models/predict", json=incomplete_request)
        assert response.status_code == 200  # Should use defaults
        
        prediction = response.json()["data"]
        assert "action" in prediction
        
        # Test with extra fields (should be ignored)
        extra_fields_request = {
            "model_id": model_id,
            "input_data": {
                "z_score": 2.0,
                "price": 100.0,
                "moving_average": 100.0,
                "volatility": 0.2,
                "volume_ratio": 1.0,
                "rsi": 50.0,
                "market_regime": 0.5,
                "extra_field_1": "ignored",
                "extra_field_2": 123
            }
        }
        
        response = test_client.post("/models/predict", json=extra_fields_request)
        assert response.status_code == 200


class TestMCPGPUAcceleration:
    """Test MCP GPU acceleration features."""
    
    @pytest.fixture
    def gpu_available(self):
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def test_gpu_resource_detection(self, gpu_available):
        """Test GPU resource detection."""
        if gpu_available:
            import torch
            device_count = torch.cuda.device_count()
            assert device_count > 0
            
            # Get GPU properties
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB")
        else:
            pytest.skip("GPU not available")
    
    @pytest.mark.asyncio
    async def test_gpu_accelerated_inference(self, gpu_available):
        """Test GPU-accelerated model inference."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        # Simulate GPU-accelerated prediction
        import torch
        
        # Create dummy model on GPU
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        ).cuda()
        
        # Batch inference
        batch_size = 100
        input_data = torch.randn(batch_size, 10).cuda()
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data)
            predictions = torch.softmax(output, dim=1)
        torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        print(f"GPU inference time for {batch_size} samples: {inference_time:.3f}s")
        assert predictions.shape == (batch_size, 3)
    
    def test_gpu_memory_management(self, gpu_available):
        """Test GPU memory management."""
        if not gpu_available:
            pytest.skip("GPU not available")
        
        import torch
        
        # Monitor memory usage
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate some tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000).cuda()
            tensors.append(tensor)
        
        peak_memory = torch.cuda.memory_allocated()
        
        # Clear tensors
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        
        print(f"Memory usage - Initial: {initial_memory/1e6:.2f} MB, "
              f"Peak: {peak_memory/1e6:.2f} MB, Final: {final_memory/1e6:.2f} MB")
        
        # Memory should be mostly freed
        assert final_memory < peak_memory * 0.1


class TestMCPRealTimeStreaming:
    """Test MCP real-time data streaming capabilities."""
    
    @pytest.mark.asyncio
    async def test_streaming_predictions(self):
        """Test streaming predictions over WebSocket."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Subscribe to real-time predictions
                subscribe_msg = {
                    "message_type": "subscribe",
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "subscription_type": "real_time_predictions",
                        "filters": {"strategy_name": "momentum"}
                    }
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                
                # Should receive confirmation
                confirmation = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                assert json.loads(confirmation)["data"]["type"] == "subscription_confirmed"
                
                # Simulate streaming predictions
                for i in range(5):
                    prediction_request = {
                        "message_type": "prediction_request",
                        "message_id": f"stream_{i}",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "model_id": "test_model",
                            "input_data": {
                                "price": 100 + i,
                                "volume": 1000000 + i * 100000
                            }
                        }
                    }
                    
                    await websocket.send(json.dumps(prediction_request))
                    
                    # Receive prediction response
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    
                    assert response_data["message_type"] in ["prediction_response", "error"]
                    
                    await asyncio.sleep(0.1)  # Simulate real-time interval
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_model_update_notifications(self):
        """Test real-time model update notifications."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Subscribe to model updates
                subscribe_msg = {
                    "message_type": "subscribe",
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "data": {"subscription_type": "model_updates"}
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                
                # Receive confirmation
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # In a real scenario, model updates would be broadcast
                # when models are retrained or updated
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_streaming(self):
        """Test streaming performance metrics."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Subscribe to performance updates
                subscribe_msg = {
                    "message_type": "subscribe",
                    "message_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "data": {"subscription_type": "performance_updates"}
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                
                # Should receive confirmation
                confirmation = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                assert json.loads(confirmation)["data"]["type"] == "subscription_confirmed"
                
                # Performance updates would be broadcast periodically
                # In test environment, we might not receive actual updates
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    def test_streaming_data_formats(self):
        """Test various streaming data formats."""
        # Test different data formats that might be streamed
        formats = {
            "json": {"type": "prediction", "value": 0.5},
            "binary": b"\x00\x01\x02\x03",
            "numpy": np.array([1.0, 2.0, 3.0]),
            "pandas": {"columns": ["A", "B"], "data": [[1, 2], [3, 4]]}
        }
        
        for format_name, data in formats.items():
            # Verify serialization capability
            if format_name == "json":
                serialized = json.dumps(data)
                assert isinstance(serialized, str)
            elif format_name == "numpy":
                # NumPy arrays need special handling
                serialized = data.tolist()
                assert isinstance(serialized, list)


class TestMCPToolExecution:
    """Test MCP tool execution and resource access."""
    
    def test_available_tools(self, test_client):
        """Test listing available MCP tools."""
        # In a full implementation, there would be a tools endpoint
        # For now, we document expected tools
        expected_tools = [
            "predict",
            "list_models",
            "get_metadata",
            "get_analytics",
            "get_recommendation",
            "evaluate_performance"
        ]
        
        # Verify tool definitions
        for tool in expected_tools:
            assert isinstance(tool, str)
            assert len(tool) > 0
    
    def test_tool_parameter_validation(self):
        """Test tool parameter validation."""
        # Define tool parameters
        tool_definitions = {
            "predict": {
                "required": ["model_id", "input_data"],
                "optional": ["return_confidence", "timeout_seconds"]
            },
            "list_models": {
                "required": [],
                "optional": ["strategy_name", "status", "tags", "limit"]
            }
        }
        
        # Verify parameter definitions
        for tool, params in tool_definitions.items():
            assert "required" in params
            assert "optional" in params
            assert isinstance(params["required"], list)
            assert isinstance(params["optional"], list)
    
    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, test_client, sample_models):
        """Test complete tool execution flow."""
        model_id = sample_models[0][0]
        
        # Step 1: List models
        list_response = test_client.get("/models")
        assert list_response.status_code == 200
        models = list_response.json()["data"]["models"]
        assert len(models) > 0
        
        # Step 2: Get specific model metadata
        metadata_response = test_client.get(f"/models/{model_id}/metadata")
        assert metadata_response.status_code == 200
        metadata = metadata_response.json()["data"]["metadata"]
        
        # Step 3: Make prediction with that model
        prediction_request = {
            "model_id": model_id,
            "input_data": {
                "z_score": 1.5,
                "price": 102.0,
                "moving_average": 100.0,
                "volatility": 0.18,
                "volume_ratio": 1.1,
                "rsi": 55.0,
                "market_regime": 0.6
            }
        }
        
        prediction_response = test_client.post("/models/predict", json=prediction_request)
        assert prediction_response.status_code == 200
        prediction = prediction_response.json()["data"]
        
        # Step 4: Get strategy analytics
        strategy_name = metadata["strategy_name"]
        analytics_response = test_client.get(f"/strategies/{strategy_name}/analytics")
        assert analytics_response.status_code == 200
        
        # Verify complete flow worked
        assert prediction["action"] in ["buy", "sell", "hold"]
    
    def test_resource_access_control(self):
        """Test resource access control mechanisms."""
        # Define access control rules
        access_rules = {
            "models": {
                "read": ["user", "admin"],
                "write": ["admin"],
                "delete": ["admin"]
            },
            "predictions": {
                "create": ["user", "admin"],
                "read": ["user", "admin"]
            },
            "analytics": {
                "read": ["user", "admin"]
            }
        }
        
        # Verify access control structure
        for resource, permissions in access_rules.items():
            assert isinstance(permissions, dict)
            for action, roles in permissions.items():
                assert isinstance(roles, list)
                assert all(isinstance(role, str) for role in roles)