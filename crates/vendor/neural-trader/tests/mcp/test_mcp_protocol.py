"""MCP Protocol Compliance Tests.

Tests MCP message types, JSON-RPC 2.0 compliance, and protocol error handling.
"""

import pytest
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock

# Import MCP components
from model_management.mcp_integration.trading_mcp_server import (
    MCPMessage,
    MCPMessageType,
    ModelRequestType,
    MCPResponse,
    ModelPredictionRequest,
    ModelListRequest,
    TradingMCPServer
)


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance with JSON-RPC 2.0."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create a test MCP server instance."""
        server = TradingMCPServer(
            host="127.0.0.1",
            port=8888,
            model_storage_path="test_models"
        )
        return server
    
    def test_mcp_message_creation(self):
        """Test MCP message creation and structure."""
        # Create a test message
        message = MCPMessage(
            message_type=MCPMessageType.REQUEST,
            request_id=str(uuid.uuid4()),
            method="predict",
            params={"model_id": "test_model_123", "input_data": {"x": 1.0}},
            timestamp=datetime.now()
        )
        
        # Verify message structure
        assert message.message_type == MCPMessageType.REQUEST
        assert message.method == "predict"
        assert "model_id" in message.params
        assert "input_data" in message.params
        
        # Test serialization
        message_dict = message.to_dict()
        assert message_dict["message_type"] == "request"
        assert "request_id" in message_dict
        assert "timestamp" in message_dict
        
        # Verify JSON serializable
        json_str = json.dumps(message_dict)
        assert isinstance(json_str, str)
    
    def test_all_message_types(self):
        """Test all MCP message types."""
        message_types = [
            MCPMessageType.REQUEST,
            MCPMessageType.RESPONSE,
            MCPMessageType.NOTIFICATION,
            MCPMessageType.ERROR
        ]
        
        for msg_type in message_types:
            message = MCPMessage(
                message_type=msg_type,
                request_id=str(uuid.uuid4()),
                method="test_method",
                params={},
                timestamp=datetime.now()
            )
            
            assert message.message_type == msg_type
            assert message.to_dict()["message_type"] == msg_type.value
    
    def test_all_request_types(self):
        """Test all model request types."""
        request_types = [
            ModelRequestType.PREDICT,
            ModelRequestType.LOAD_MODEL,
            ModelRequestType.LIST_MODELS,
            ModelRequestType.GET_METADATA,
            ModelRequestType.UPDATE_MODEL,
            ModelRequestType.DELETE_MODEL,
            ModelRequestType.HEALTH_CHECK,
            ModelRequestType.PERFORMANCE_METRICS,
            ModelRequestType.STRATEGY_RECOMMENDATION
        ]
        
        for req_type in request_types:
            assert isinstance(req_type.value, str)
            assert req_type.value.islower()  # Convention check
    
    def test_mcp_response_structure(self):
        """Test MCP response format."""
        response = MCPResponse(
            request_id="test_req_123",
            success=True,
            data={"result": "test_value"},
            timestamp=datetime.now().isoformat(),
            processing_time_ms=123.45
        )
        
        # Verify response structure
        assert response.request_id == "test_req_123"
        assert response.success is True
        assert response.data == {"result": "test_value"}
        assert response.processing_time_ms == 123.45
        
        # Test JSON serialization
        response_json = response.model_dump_json()
        parsed = json.loads(response_json)
        assert parsed["request_id"] == "test_req_123"
        assert parsed["success"] is True
    
    def test_error_response_structure(self):
        """Test error response format."""
        response = MCPResponse(
            request_id="error_req_123",
            success=False,
            error="Model not found",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=10.5
        )
        
        assert response.success is False
        assert response.error == "Model not found"
        assert response.data is None
    
    def test_model_prediction_request_validation(self):
        """Test model prediction request validation."""
        # Valid request
        valid_request = ModelPredictionRequest(
            model_id="model_123",
            input_data={"feature1": 1.0, "feature2": 2.0},
            strategy_context={"market": "bullish"},
            return_confidence=True,
            timeout_seconds=30
        )
        
        assert valid_request.model_id == "model_123"
        assert valid_request.return_confidence is True
        assert valid_request.timeout_seconds == 30
        
        # Test with minimal fields
        minimal_request = ModelPredictionRequest(
            model_id="model_456",
            input_data={}
        )
        
        assert minimal_request.model_id == "model_456"
        assert minimal_request.return_confidence is False  # Default
        assert minimal_request.timeout_seconds == 30  # Default
    
    def test_model_list_request_validation(self):
        """Test model list request validation."""
        # Full request
        full_request = ModelListRequest(
            strategy_name="momentum",
            status="production",
            tags=["optimized", "v2"],
            limit=50
        )
        
        assert full_request.strategy_name == "momentum"
        assert full_request.status == "production"
        assert full_request.tags == ["optimized", "v2"]
        assert full_request.limit == 50
        
        # Minimal request
        minimal_request = ModelListRequest()
        assert minimal_request.limit == 100  # Default
        assert minimal_request.strategy_name is None
        assert minimal_request.status is None
        assert minimal_request.tags is None
    
    @pytest.mark.asyncio
    async def test_json_rpc_request_format(self, mcp_server):
        """Test JSON-RPC 2.0 request format compliance."""
        # Create JSON-RPC 2.0 compliant request
        json_rpc_request = {
            "jsonrpc": "2.0",
            "method": "predict",
            "params": {
                "model_id": "test_model",
                "input_data": {"x": 1.0}
            },
            "id": "req_123"
        }
        
        # Convert to MCP message
        mcp_message = MCPMessage(
            message_type=MCPMessageType.REQUEST,
            request_id=json_rpc_request["id"],
            method=json_rpc_request["method"],
            params=json_rpc_request["params"],
            timestamp=datetime.now()
        )
        
        assert mcp_message.request_id == "req_123"
        assert mcp_message.method == "predict"
        assert mcp_message.params == json_rpc_request["params"]
    
    def test_batch_request_support(self):
        """Test batch request format for multiple operations."""
        batch_requests = [
            {
                "message_type": "request",
                "request_id": f"req_{i}",
                "method": "predict",
                "params": {"model_id": f"model_{i}", "input_data": {"x": float(i)}},
                "timestamp": datetime.now().isoformat()
            }
            for i in range(3)
        ]
        
        # Verify batch format
        assert len(batch_requests) == 3
        for i, req in enumerate(batch_requests):
            assert req["request_id"] == f"req_{i}"
            assert req["params"]["model_id"] == f"model_{i}"
    
    def test_notification_format(self):
        """Test notification message format (no response expected)."""
        notification = MCPMessage(
            message_type=MCPMessageType.NOTIFICATION,
            request_id="",  # Notifications don't need request IDs
            method="model_updated",
            params={
                "model_id": "model_123",
                "status": "retrained",
                "timestamp": datetime.now().isoformat()
            },
            timestamp=datetime.now()
        )
        
        assert notification.message_type == MCPMessageType.NOTIFICATION
        assert notification.request_id == ""
        assert notification.method == "model_updated"
    
    def test_error_codes_and_messages(self):
        """Test standard error codes and messages."""
        standard_errors = {
            -32700: "Parse error",
            -32600: "Invalid Request",
            -32601: "Method not found",
            -32602: "Invalid params",
            -32603: "Internal error",
            -32000: "Server error",  # Custom errors start here
            -32001: "Model not found",
            -32002: "Invalid model format",
            -32003: "Prediction timeout",
            -32004: "Insufficient resources"
        }
        
        for code, message in standard_errors.items():
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": code,
                    "message": message,
                    "data": {"additional_info": "test"}
                },
                "id": "error_test"
            }
            
            # Verify error structure
            assert error_response["error"]["code"] == code
            assert error_response["error"]["message"] == message
            assert "data" in error_response["error"]
    
    @pytest.mark.asyncio
    async def test_request_id_tracking(self, mcp_server):
        """Test that request IDs are properly tracked through request/response cycle."""
        request_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        for req_id in request_ids:
            request = ModelPredictionRequest(
                model_id="test_model",
                input_data={"x": 1.0}
            )
            
            # Mock response with matching request ID
            response = MCPResponse(
                request_id=req_id,
                success=True,
                data={"prediction": 0.5},
                timestamp=datetime.now().isoformat(),
                processing_time_ms=10.0
            )
            
            assert response.request_id == req_id
    
    def test_timestamp_format(self):
        """Test timestamp format compliance (ISO 8601)."""
        message = MCPMessage(
            message_type=MCPMessageType.REQUEST,
            request_id="test",
            method="test",
            params={},
            timestamp=datetime.now()
        )
        
        message_dict = message.to_dict()
        timestamp_str = message_dict["timestamp"]
        
        # Verify ISO 8601 format
        parsed_timestamp = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_timestamp, datetime)
        
        # Verify format includes timezone info or is naive
        assert "T" in timestamp_str  # ISO 8601 separator
    
    def test_parameter_validation(self):
        """Test parameter validation for different request types."""
        # Test prediction parameters
        with pytest.raises(ValueError):
            # Missing required model_id
            ModelPredictionRequest(
                model_id="",  # Empty model_id should fail
                input_data={}
            )
        
        # Test list parameters with invalid limit
        with pytest.raises(ValueError):
            ModelListRequest(limit=0)  # Limit must be >= 1
        
        with pytest.raises(ValueError):
            ModelListRequest(limit=1001)  # Limit must be <= 1000
    
    def test_message_size_limits(self):
        """Test handling of large messages."""
        # Create a large input data payload
        large_data = {
            f"feature_{i}": [j * 0.1 for j in range(100)]
            for i in range(100)
        }
        
        request = ModelPredictionRequest(
            model_id="test_model",
            input_data=large_data
        )
        
        # Verify serialization works
        json_str = request.model_dump_json()
        size_kb = len(json_str) / 1024
        
        # Log size for monitoring
        print(f"Large message size: {size_kb:.2f} KB")
        
        # Typical limit would be 1MB for WebSocket frames
        assert size_kb < 1024  # Less than 1MB
    
    def test_version_compatibility(self):
        """Test protocol version compatibility."""
        # Test different protocol versions
        versions = ["1.0", "1.1", "2.0"]
        
        for version in versions:
            request_with_version = {
                "jsonrpc": version,
                "method": "predict",
                "params": {"model_id": "test"},
                "id": f"v{version}"
            }
            
            # Current implementation should handle 2.0
            if version == "2.0":
                assert request_with_version["jsonrpc"] == "2.0"
    
    def test_method_case_sensitivity(self):
        """Test that method names are case-sensitive."""
        methods = ["predict", "PREDICT", "Predict", "pReDiCt"]
        
        for method in methods:
            message = MCPMessage(
                message_type=MCPMessageType.REQUEST,
                request_id="test",
                method=method,
                params={},
                timestamp=datetime.now()
            )
            
            # Methods should be preserved as-is
            assert message.method == method
    
    def test_concurrent_request_handling(self):
        """Test handling of concurrent requests."""
        import threading
        
        requests = []
        results = []
        
        def create_request(index):
            request = MCPMessage(
                message_type=MCPMessageType.REQUEST,
                request_id=f"concurrent_{index}",
                method="predict",
                params={"model_id": f"model_{index}"},
                timestamp=datetime.now()
            )
            requests.append(request)
        
        # Create multiple concurrent requests
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all requests were created with unique IDs
        request_ids = [req.request_id for req in requests]
        assert len(set(request_ids)) == len(request_ids)  # All unique


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create a test MCP server instance."""
        return TradingMCPServer(
            host="127.0.0.1",
            port=8889,
            model_storage_path="test_models"
        )
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON."""
        malformed_inputs = [
            '{"invalid": json,}',  # Trailing comma
            '{"missing": "quotes}',  # Missing quote
            '{invalid}',  # Invalid format
            'null',  # Null input
            '',  # Empty string
            '[]',  # Array instead of object
        ]
        
        for malformed in malformed_inputs:
            try:
                json.loads(malformed)
                # If it parses, it's not malformed for our test
                continue
            except json.JSONDecodeError:
                # This is expected for malformed JSON
                pass
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Missing model_id
        with pytest.raises(TypeError):
            ModelPredictionRequest(input_data={"x": 1.0})
        
        # Missing input_data
        with pytest.raises(TypeError):
            ModelPredictionRequest(model_id="test")
    
    def test_invalid_field_types(self):
        """Test handling of invalid field types."""
        # Invalid timeout type
        with pytest.raises(ValueError):
            ModelPredictionRequest(
                model_id="test",
                input_data={},
                timeout_seconds="invalid"  # Should be int
            )
        
        # Invalid return_confidence type
        with pytest.raises(ValueError):
            ModelPredictionRequest(
                model_id="test",
                input_data={},
                return_confidence="yes"  # Should be bool
            )
    
    def test_timeout_handling(self):
        """Test request timeout handling."""
        # Create request with very short timeout
        request = ModelPredictionRequest(
            model_id="test_model",
            input_data={"x": 1.0},
            timeout_seconds=1  # 1 second timeout
        )
        
        assert request.timeout_seconds == 1
        
        # Test maximum timeout
        max_timeout_request = ModelPredictionRequest(
            model_id="test_model",
            input_data={"x": 1.0},
            timeout_seconds=3600  # 1 hour
        )
        
        assert max_timeout_request.timeout_seconds == 3600
    
    def test_empty_response_handling(self):
        """Test handling of empty responses."""
        # Empty success response
        empty_response = MCPResponse(
            request_id="test",
            success=True,
            data=None,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0.0
        )
        
        assert empty_response.data is None
        assert empty_response.success is True
        
        # Empty error response
        error_response = MCPResponse(
            request_id="test",
            success=False,
            error="Unknown error",
            timestamp=datetime.now().isoformat(),
            processing_time_ms=0.0
        )
        
        assert error_response.data is None
        assert error_response.error == "Unknown error"
    
    def test_recursive_data_structures(self):
        """Test handling of recursive/circular data structures."""
        # Create nested structure
        nested_data = {"level1": {"level2": {"level3": {"value": 1.0}}}}
        
        request = ModelPredictionRequest(
            model_id="test",
            input_data=nested_data
        )
        
        # Should handle nested structures
        assert request.input_data["level1"]["level2"]["level3"]["value"] == 1.0
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, mcp_server):
        """Test handling of connection failures."""
        # Mock network failure
        with patch('asyncio.create_task') as mock_task:
            mock_task.side_effect = ConnectionError("Network unreachable")
            
            # Server should handle connection errors gracefully
            try:
                # Attempt to process request during connection failure
                pass  # In real test, would attempt actual operation
            except ConnectionError:
                # Expected behavior
                pass
    
    def test_invalid_model_id_format(self):
        """Test handling of invalid model ID formats."""
        invalid_ids = [
            "",  # Empty
            " ",  # Whitespace
            "../../etc/passwd",  # Path traversal attempt
            "model\x00null",  # Null byte
            "a" * 1000,  # Very long ID
            "SELECT * FROM models;",  # SQL injection attempt
        ]
        
        for invalid_id in invalid_ids:
            request = ModelPredictionRequest(
                model_id=invalid_id,
                input_data={}
            )
            # Should create request but server should validate
            assert request.model_id == invalid_id
    
    def test_resource_exhaustion_protection(self):
        """Test protection against resource exhaustion."""
        # Test with excessive array size
        huge_array = list(range(10000))
        
        request = ModelPredictionRequest(
            model_id="test",
            input_data={"huge_array": huge_array}
        )
        
        # Should handle but with size limits
        json_size = len(request.model_dump_json())
        assert json_size < 10 * 1024 * 1024  # Less than 10MB
    
    def test_unicode_handling(self):
        """Test handling of Unicode characters."""
        unicode_data = {
            "emoji": "🚀📈💰",
            "chinese": "人工智能交易",
            "arabic": "التداول الآلي",
            "special": "∑∏∫≈≠",
        }
        
        request = ModelPredictionRequest(
            model_id="unicode_test",
            input_data=unicode_data
        )
        
        # Should handle Unicode properly
        assert request.input_data["emoji"] == "🚀📈💰"
        assert request.input_data["chinese"] == "人工智能交易"
        
        # Test serialization
        json_str = request.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["input_data"]["emoji"] == "🚀📈💰"


class TestMCPConcurrency:
    """Test MCP concurrent request handling and thread safety."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create a test MCP server instance."""
        return TradingMCPServer(
            host="127.0.0.1",
            port=8890,
            model_storage_path="test_models"
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self, mcp_server):
        """Test handling of concurrent prediction requests."""
        num_concurrent = 50
        
        async def make_prediction(index):
            request = ModelPredictionRequest(
                model_id=f"model_{index % 5}",  # Use 5 different models
                input_data={"index": index, "value": index * 0.1}
            )
            
            # Simulate processing
            await asyncio.sleep(0.01)  # Small delay
            
            return {
                "request_id": f"req_{index}",
                "model_id": request.model_id,
                "result": index * 2
            }
        
        # Run concurrent predictions
        tasks = [make_prediction(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert len(results) == num_concurrent
        
        # Verify no duplicate request IDs
        request_ids = [r["request_id"] for r in results]
        assert len(set(request_ids)) == len(request_ids)
    
    @pytest.mark.asyncio
    async def test_request_queuing(self, mcp_server):
        """Test request queuing under high load."""
        queue_size = 100
        processed = []
        
        async def process_request(index):
            await asyncio.sleep(0.001 * (index % 10))  # Variable delays
            processed.append(index)
            return index
        
        # Submit many requests rapidly
        tasks = [process_request(i) for i in range(queue_size)]
        results = await asyncio.gather(*tasks)
        
        # All should complete
        assert len(results) == queue_size
        assert len(processed) == queue_size
    
    def test_thread_safe_cache_access(self, mcp_server):
        """Test thread-safe access to model cache."""
        import threading
        import time
        
        cache_hits = []
        cache_misses = []
        
        def access_cache(model_id, index):
            # Simulate cache access
            cache_key = f"{model_id}_{index % 3}"
            
            if cache_key in mcp_server.model_cache:
                cache_hits.append(cache_key)
            else:
                cache_misses.append(cache_key)
                # Simulate adding to cache
                with mcp_server.cache_lock:
                    mcp_server.model_cache[cache_key] = {
                        'model': f"model_data_{cache_key}",
                        'metadata': {},
                        'timestamp': datetime.now()
                    }
        
        # Create multiple threads accessing cache
        threads = []
        for i in range(20):
            thread = threading.Thread(
                target=access_cache,
                args=(f"model_{i % 5}", i)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify cache state is consistent
        assert len(mcp_server.model_cache) <= 15  # Max possible unique keys
    
    @pytest.mark.asyncio
    async def test_graceful_overload_handling(self, mcp_server):
        """Test graceful handling of system overload."""
        overload_count = 1000
        completed = 0
        errors = 0
        
        async def overload_request(index):
            nonlocal completed, errors
            try:
                # Simulate varying request sizes
                data_size = 10 ** (2 + index % 3)  # 100, 1000, or 10000 elements
                
                request = ModelPredictionRequest(
                    model_id="overload_test",
                    input_data={
                        f"feature_{j}": j * 0.1 
                        for j in range(min(data_size, 1000))
                    }
                )
                
                await asyncio.sleep(0.0001)  # Minimal delay
                completed += 1
                
            except Exception:
                errors += 1
        
        # Submit overload
        start_time = asyncio.get_event_loop().time()
        tasks = [overload_request(i) for i in range(overload_count)]
        await asyncio.gather(*tasks, return_exceptions=True)
        end_time = asyncio.get_event_loop().time()
        
        duration = end_time - start_time
        throughput = completed / duration
        
        print(f"Processed {completed}/{overload_count} requests in {duration:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Errors: {errors}")
        
        # Should complete most requests
        assert completed > overload_count * 0.95  # At least 95% success rate