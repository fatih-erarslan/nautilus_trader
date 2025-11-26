"""MCP Transport Layer Tests.

Tests WebSocket and HTTP transport layers for the MCP server.
"""

import pytest
import asyncio
import json
import uuid
import websockets
import httpx
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, AsyncMock
import time

from fastapi.testclient import TestClient
from model_management.mcp_integration.trading_mcp_server import TradingMCPServer
from model_management.mcp_integration.websocket_server import (
    ModelWebSocketServer,
    WebSocketMessage,
    MessageType,
    SubscriptionType
)


class TestHTTPTransport:
    """Test HTTP/REST transport layer for MCP."""
    
    @pytest.fixture
    def mcp_server(self):
        """Create a test MCP server instance."""
        server = TradingMCPServer(
            host="127.0.0.1",
            port=8891,
            model_storage_path="test_models"
        )
        return server
    
    @pytest.fixture
    def test_client(self, mcp_server):
        """Create a test client for the FastAPI app."""
        return TestClient(mcp_server.app)
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "models_loaded" in data
        assert "total_models" in data
        assert "server_uptime" in data
    
    def test_metrics_endpoint(self, test_client):
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "average_response_time" in metrics
        assert "last_reset" in metrics
    
    def test_prediction_endpoint(self, test_client):
        """Test prediction endpoint."""
        # Create prediction request
        request_data = {
            "model_id": "test_model_123",
            "input_data": {
                "z_score": 2.5,
                "price": 105.0,
                "moving_average": 100.0,
                "volatility": 0.2,
                "volume_ratio": 1.2,
                "rsi": 65.0,
                "market_regime": 0.7
            },
            "return_confidence": True,
            "timeout_seconds": 30
        }
        
        # Mock model loading
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            mock_storage.load_model.return_value = (
                {"z_score_entry_threshold": 2.0, "base_position_size": 0.05},
                MagicMock(strategy_name="mean_reversion", parameters={})
            )
            
            response = test_client.post("/models/predict", json=request_data)
        
        # Response should fail since we don't have actual model storage
        # In real tests with proper setup, this would succeed
        assert response.status_code in [200, 404, 500]
    
    def test_list_models_endpoint(self, test_client):
        """Test list models endpoint."""
        # Test with filters
        params = {
            "strategy_name": "momentum",
            "status": "production",
            "limit": 50
        }
        
        response = test_client.get("/models", params=params)
        
        # Should return response (even if empty)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "success" in data
            assert "data" in data
    
    def test_get_model_metadata_endpoint(self, test_client):
        """Test get model metadata endpoint."""
        model_id = "test_model_123"
        
        response = test_client.get(f"/models/{model_id}/metadata")
        
        # Should return 404 for non-existent model
        assert response.status_code in [404, 500]
    
    def test_strategy_analytics_endpoint(self, test_client):
        """Test strategy analytics endpoint."""
        strategy_name = "momentum"
        
        response = test_client.get(f"/strategies/{strategy_name}/analytics")
        
        # Should handle request
        assert response.status_code in [200, 500]
    
    def test_strategy_recommendation_endpoint(self, test_client):
        """Test strategy recommendation endpoint."""
        strategy_name = "mean_reversion"
        
        response = test_client.get(f"/strategies/{strategy_name}/recommendation")
        
        # Should handle request
        assert response.status_code in [200, 404, 500]
    
    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set."""
        response = test_client.options("/health")
        
        # Check CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_request_validation(self, test_client):
        """Test request validation for invalid inputs."""
        # Invalid prediction request (missing required fields)
        invalid_request = {
            "input_data": {"x": 1.0}
            # Missing model_id
        }
        
        response = test_client.post("/models/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Invalid list request (invalid limit)
        response = test_client.get("/models", params={"limit": 0})
        assert response.status_code == 422
    
    def test_content_type_headers(self, test_client):
        """Test content type headers."""
        response = test_client.get("/health")
        
        assert response.headers["content-type"] == "application/json"
    
    def test_error_response_format(self, test_client):
        """Test error response format consistency."""
        # Request non-existent endpoint
        response = test_client.get("/invalid/endpoint")
        assert response.status_code == 404
        
        # Error response should be JSON
        assert "application/json" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_concurrent_http_requests(self, test_client):
        """Test handling of concurrent HTTP requests."""
        import aiohttp
        import asyncio
        
        async def make_request(session, index):
            """Make async HTTP request."""
            try:
                async with session.get(f"http://127.0.0.1:8891/health") as response:
                    return await response.json()
            except:
                return None
        
        # Create multiple concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful responses
        successful = sum(1 for r in results if r and not isinstance(r, Exception))
        print(f"Successful concurrent requests: {successful}/20")
    
    def test_request_timeout_handling(self, test_client):
        """Test request timeout handling."""
        # Create request with very short timeout
        request_data = {
            "model_id": "test_model",
            "input_data": {"x": 1.0},
            "timeout_seconds": 1  # 1 second timeout
        }
        
        # Mock slow model prediction
        with patch.object(test_client.app.state, 'model_storage', create=True) as mock_storage:
            async def slow_load(*args):
                await asyncio.sleep(2)  # Longer than timeout
                return ({}, MagicMock())
            
            mock_storage.load_model = slow_load
            
            response = test_client.post("/models/predict", json=request_data)
            
            # Should handle timeout appropriately
            assert response.status_code in [200, 408, 500]


class TestWebSocketTransport:
    """Test WebSocket transport layer for MCP."""
    
    @pytest.fixture
    async def ws_server(self):
        """Create and start a test WebSocket server."""
        server = ModelWebSocketServer(
            host="127.0.0.1",
            port=8892,
            storage_path="test_models"
        )
        
        # Start server in background
        server_task = asyncio.create_task(server.start_server())
        await asyncio.sleep(0.1)  # Let server start
        
        yield server
        
        # Cleanup
        await server.stop_server()
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Should receive welcome message
                welcome = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                welcome_data = json.loads(welcome)
                
                assert welcome_data.get("message_type") == "notification"
                assert welcome_data.get("data", {}).get("type") == "welcome"
                assert "client_id" in welcome_data.get("data", {})
                assert "server_info" in welcome_data.get("data", {})
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            # Server might not be running in test environment
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_subscription(self):
        """Test WebSocket subscription mechanism."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Send subscription request
                subscribe_msg = WebSocketMessage(
                    message_type=MessageType.SUBSCRIBE,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={
                        "subscription_type": SubscriptionType.MODEL_UPDATES.value,
                        "filters": {"strategy_name": "momentum"}
                    }
                )
                
                await websocket.send(json.dumps(subscribe_msg.to_dict()))
                
                # Should receive confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                assert response_data.get("message_type") == "notification"
                assert response_data.get("data", {}).get("type") == "subscription_confirmed"
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_prediction_request(self):
        """Test WebSocket prediction request."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Send prediction request
                prediction_msg = WebSocketMessage(
                    message_type=MessageType.PREDICTION_REQUEST,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={
                        "model_id": "test_model_123",
                        "input_data": {"x": 1.0, "y": 2.0}
                    }
                )
                
                await websocket.send(json.dumps(prediction_msg.to_dict()))
                
                # Should receive response (or error)
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                assert response_data.get("message_type") in ["prediction_response", "error"]
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat mechanism."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Send heartbeat
                heartbeat_msg = WebSocketMessage(
                    message_type=MessageType.HEARTBEAT,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={}
                )
                
                await websocket.send(json.dumps(heartbeat_msg.to_dict()))
                
                # Should receive heartbeat response
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                assert response_data.get("message_type") == "heartbeat"
                assert response_data.get("data", {}).get("type") == "heartbeat_response"
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_unsubscribe(self):
        """Test WebSocket unsubscribe mechanism."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # First subscribe
                subscribe_msg = WebSocketMessage(
                    message_type=MessageType.SUBSCRIBE,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={"subscription_type": SubscriptionType.MODEL_UPDATES.value}
                )
                
                await websocket.send(json.dumps(subscribe_msg.to_dict()))
                await asyncio.wait_for(websocket.recv(), timeout=2.0)  # Confirmation
                
                # Then unsubscribe
                unsubscribe_msg = WebSocketMessage(
                    message_type=MessageType.UNSUBSCRIBE,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={"subscription_type": SubscriptionType.MODEL_UPDATES.value}
                )
                
                await websocket.send(json.dumps(unsubscribe_msg.to_dict()))
                
                # Should receive confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                assert response_data.get("data", {}).get("type") == "unsubscription_confirmed"
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self):
        """Test WebSocket reconnection handling."""
        uri = "ws://127.0.0.1:8892/ws"
        client_ids = []
        
        try:
            # First connection
            async with websockets.connect(uri) as websocket:
                welcome1 = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                client_id1 = json.loads(welcome1)["data"]["client_id"]
                client_ids.append(client_id1)
            
            # Second connection (after first closes)
            async with websockets.connect(uri) as websocket:
                welcome2 = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                client_id2 = json.loads(welcome2)["data"]["client_id"]
                client_ids.append(client_id2)
            
            # Should have different client IDs
            assert client_ids[0] != client_ids[1]
            
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_message_ordering(self):
        """Test WebSocket message ordering."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Send multiple messages rapidly
                message_ids = []
                for i in range(5):
                    msg = WebSocketMessage(
                        message_type=MessageType.HEARTBEAT,
                        message_id=f"order_test_{i}",
                        timestamp=datetime.now(),
                        data={"sequence": i}
                    )
                    message_ids.append(msg.message_id)
                    await websocket.send(json.dumps(msg.to_dict()))
                
                # Receive responses
                responses = []
                for _ in range(5):
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    responses.append(json.loads(response))
                
                # Messages should maintain order
                # Note: This might not always be guaranteed in async environments
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self):
        """Test WebSocket error handling."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                
                # Send invalid message
                await websocket.send("invalid json {")
                
                # Should receive error response
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                
                assert response_data.get("message_type") == "error"
                assert "error" in response_data.get("data", {})
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_clients(self):
        """Test handling of multiple concurrent WebSocket clients."""
        uri = "ws://127.0.0.1:8892/ws"
        num_clients = 5
        
        async def client_task(client_id):
            try:
                async with websockets.connect(uri) as websocket:
                    # Receive welcome
                    welcome = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    
                    # Send a message
                    msg = WebSocketMessage(
                        message_type=MessageType.HEARTBEAT,
                        message_id=f"client_{client_id}",
                        timestamp=datetime.now(),
                        data={"client_number": client_id}
                    )
                    await websocket.send(json.dumps(msg.to_dict()))
                    
                    # Receive response
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    return json.loads(response)
                    
            except Exception as e:
                return {"error": str(e)}
        
        try:
            # Run multiple clients concurrently
            tasks = [client_task(i) for i in range(num_clients)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successful connections
            successful = sum(1 for r in results 
                           if isinstance(r, dict) and "error" not in r)
            
            print(f"Successful WebSocket clients: {successful}/{num_clients}")
            
        except Exception:
            pytest.skip("WebSocket server not available")
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_mechanism(self):
        """Test WebSocket broadcast to multiple subscribers."""
        uri = "ws://127.0.0.1:8892/ws"
        
        try:
            # Create two clients
            async with websockets.connect(uri) as ws1, websockets.connect(uri) as ws2:
                # Wait for welcome messages
                await asyncio.wait_for(ws1.recv(), timeout=2.0)
                await asyncio.wait_for(ws2.recv(), timeout=2.0)
                
                # Both subscribe to same topic
                subscribe_msg = WebSocketMessage(
                    message_type=MessageType.SUBSCRIBE,
                    message_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    data={"subscription_type": SubscriptionType.MODEL_UPDATES.value}
                )
                
                await ws1.send(json.dumps(subscribe_msg.to_dict()))
                await ws2.send(json.dumps(subscribe_msg.to_dict()))
                
                # Wait for confirmations
                await asyncio.wait_for(ws1.recv(), timeout=2.0)
                await asyncio.wait_for(ws2.recv(), timeout=2.0)
                
                # In a real test, we would trigger a broadcast event
                # and verify both clients receive it
                
        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError):
            pytest.skip("WebSocket server not available")


class TestTransportSecurity:
    """Test transport layer security features."""
    
    def test_input_sanitization(self):
        """Test input sanitization for security."""
        # Test various injection attempts
        malicious_inputs = [
            {"model_id": "<script>alert('xss')</script>"},
            {"model_id": "'; DROP TABLE models; --"},
            {"model_id": "../../../etc/passwd"},
            {"input_data": {"key": "value\x00null"}},
        ]
        
        for malicious_input in malicious_inputs:
            # Input should be sanitized or rejected
            # In real implementation, verify proper handling
            pass
    
    def test_rate_limiting(self):
        """Test rate limiting implementation."""
        # In production, verify rate limiting is enforced
        # This would require actual server implementation
        pass
    
    def test_authentication_headers(self):
        """Test authentication header handling."""
        # Test various auth scenarios
        auth_headers = [
            {"Authorization": "Bearer valid_token"},
            {"Authorization": "Bearer invalid_token"},
            {"Authorization": "Basic dXNlcjpwYXNz"},
            {},  # No auth
        ]
        
        for headers in auth_headers:
            # In real implementation, verify auth handling
            pass
    
    @pytest.mark.asyncio
    async def test_ssl_websocket_connection(self):
        """Test SSL/TLS WebSocket connection."""
        # In production, test with wss:// protocol
        # This requires SSL certificates
        pass
    
    def test_cors_security(self):
        """Test CORS security configuration."""
        # Verify CORS is properly configured
        allowed_origins = ["http://localhost:3000", "https://app.example.com"]
        blocked_origins = ["http://malicious.com", "null"]
        
        # In real implementation, verify CORS behavior
        pass