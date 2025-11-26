"""MCP Test Client Utilities.

Provides easy-to-use client utilities for testing MCP server functionality.
"""

import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import httpx
import websockets
from contextlib import asynccontextmanager, contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCPTestConfig:
    """Configuration for MCP test client."""
    http_base_url: str = "http://127.0.0.1:8000"
    ws_url: str = "ws://127.0.0.1:8002/ws"
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    log_requests: bool = True
    log_responses: bool = True


@dataclass
class MCPResponse:
    """Wrapper for MCP responses."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    processing_time_ms: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


class MCPHttpClient:
    """HTTP client for MCP server testing."""
    
    def __init__(self, config: MCPTestConfig = None):
        """Initialize HTTP client."""
        self.config = config or MCPTestConfig()
        self.session = httpx.Client(
            base_url=self.config.http_base_url,
            timeout=self.config.timeout
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()
    
    def _log_request(self, method: str, url: str, data: Any = None):
        """Log request details."""
        if self.config.log_requests:
            logger.info(f"MCP Request: {method} {url}")
            if data:
                logger.debug(f"Request data: {json.dumps(data, indent=2)}")
    
    def _log_response(self, response: httpx.Response):
        """Log response details."""
        if self.config.log_responses:
            logger.info(f"MCP Response: {response.status_code}")
            try:
                logger.debug(f"Response data: {json.dumps(response.json(), indent=2)}")
            except:
                logger.debug(f"Response text: {response.text}")
    
    def health_check(self) -> MCPResponse:
        """Check server health."""
        try:
            response = self.session.get("/health")
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=True,
                    data=data,
                    raw_response=data
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"Health check failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return MCPResponse(success=False, error=str(e))
    
    def predict(self, model_id: str, input_data: Dict[str, Any],
                return_confidence: bool = False,
                timeout_seconds: int = 30) -> MCPResponse:
        """Make a prediction request."""
        request_data = {
            "model_id": model_id,
            "input_data": input_data,
            "return_confidence": return_confidence,
            "timeout_seconds": timeout_seconds
        }
        
        self._log_request("POST", "/models/predict", request_data)
        
        try:
            start_time = time.time()
            response = self.session.post("/models/predict", json=request_data)
            processing_time = (time.time() - start_time) * 1000
            
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=data.get("success", True),
                    data=data.get("data"),
                    request_id=data.get("request_id"),
                    timestamp=data.get("timestamp"),
                    processing_time_ms=data.get("processing_time_ms", processing_time),
                    raw_response=data
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"Prediction failed with status {response.status_code}: {response.text}",
                    processing_time_ms=processing_time
                )
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return MCPResponse(success=False, error=str(e))
    
    def list_models(self, strategy_name: Optional[str] = None,
                    status: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    limit: int = 100) -> MCPResponse:
        """List available models."""
        params = {
            "limit": limit
        }
        if strategy_name:
            params["strategy_name"] = strategy_name
        if status:
            params["status"] = status
        if tags:
            params["tags"] = tags
        
        self._log_request("GET", "/models", params)
        
        try:
            response = self.session.get("/models", params=params)
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=data.get("success", True),
                    data=data.get("data"),
                    raw_response=data
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"List models failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"List models error: {e}")
            return MCPResponse(success=False, error=str(e))
    
    def get_model_metadata(self, model_id: str) -> MCPResponse:
        """Get model metadata."""
        self._log_request("GET", f"/models/{model_id}/metadata")
        
        try:
            response = self.session.get(f"/models/{model_id}/metadata")
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=data.get("success", True),
                    data=data.get("data"),
                    raw_response=data
                )
            elif response.status_code == 404:
                return MCPResponse(
                    success=False,
                    error=f"Model {model_id} not found"
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"Get metadata failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Get metadata error: {e}")
            return MCPResponse(success=False, error=str(e))
    
    def get_strategy_analytics(self, strategy_name: str) -> MCPResponse:
        """Get strategy analytics."""
        self._log_request("GET", f"/strategies/{strategy_name}/analytics")
        
        try:
            response = self.session.get(f"/strategies/{strategy_name}/analytics")
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=data.get("success", True),
                    data=data.get("data"),
                    raw_response=data
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"Get analytics failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Get analytics error: {e}")
            return MCPResponse(success=False, error=str(e))
    
    def get_metrics(self) -> MCPResponse:
        """Get server metrics."""
        try:
            response = self.session.get("/metrics")
            self._log_response(response)
            
            if response.status_code == 200:
                data = response.json()
                return MCPResponse(
                    success=True,
                    data=data,
                    raw_response=data
                )
            else:
                return MCPResponse(
                    success=False,
                    error=f"Get metrics failed with status {response.status_code}"
                )
        except Exception as e:
            logger.error(f"Get metrics error: {e}")
            return MCPResponse(success=False, error=str(e))


class MCPWebSocketClient:
    """WebSocket client for MCP server testing."""
    
    def __init__(self, config: MCPTestConfig = None):
        """Initialize WebSocket client."""
        self.config = config or MCPTestConfig()
        self.websocket = None
        self.client_id = None
        self.subscriptions = set()
        self.message_handlers = {}
        self.running = False
    
    @asynccontextmanager
    async def connect(self):
        """Async context manager for WebSocket connection."""
        try:
            self.websocket = await websockets.connect(self.config.ws_url)
            self.running = True
            
            # Wait for welcome message
            welcome = await self.receive_message()
            if welcome and welcome.get("data", {}).get("type") == "welcome":
                self.client_id = welcome["data"].get("client_id")
                logger.info(f"Connected to MCP WebSocket. Client ID: {self.client_id}")
            
            # Start message handler
            handler_task = asyncio.create_task(self._message_handler())
            
            yield self
            
            # Cleanup
            self.running = False
            handler_task.cancel()
            try:
                await handler_task
            except asyncio.CancelledError:
                pass
            
            await self.websocket.close()
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise
    
    async def _message_handler(self):
        """Handle incoming messages."""
        while self.running:
            try:
                message = await self.receive_message()
                if message:
                    message_type = message.get("message_type")
                    if message_type in self.message_handlers:
                        handler = self.message_handlers[message_type]
                        await handler(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message handler error: {e}")
    
    def on_message(self, message_type: str):
        """Decorator for registering message handlers."""
        def decorator(func: Callable):
            self.message_handlers[message_type] = func
            return func
        return decorator
    
    async def send_message(self, message_type: str, data: Dict[str, Any],
                          message_id: Optional[str] = None) -> bool:
        """Send a message to the server."""
        if not self.websocket:
            logger.error("WebSocket not connected")
            return False
        
        message = {
            "message_type": message_type,
            "message_id": message_id or str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "client_id": self.client_id
        }
        
        if self.config.log_requests:
            logger.info(f"Sending WebSocket message: {message_type}")
            logger.debug(f"Message data: {json.dumps(message, indent=2)}")
        
        try:
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return False
    
    async def receive_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Receive a message from the server."""
        if not self.websocket:
            return None
        
        try:
            if timeout:
                message_str = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=timeout
                )
            else:
                message_str = await self.websocket.recv()
            
            message = json.loads(message_str)
            
            if self.config.log_responses:
                logger.info(f"Received WebSocket message: {message.get('message_type')}")
                logger.debug(f"Message data: {json.dumps(message, indent=2)}")
            
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Receive message error: {e}")
            return None
    
    async def subscribe(self, subscription_type: str,
                       filters: Optional[Dict[str, Any]] = None) -> bool:
        """Subscribe to a topic."""
        data = {
            "subscription_type": subscription_type
        }
        if filters:
            data["filters"] = filters
        
        success = await self.send_message("subscribe", data)
        if success:
            self.subscriptions.add(subscription_type)
        
        # Wait for confirmation
        confirmation = await self.receive_message(timeout=5.0)
        return confirmation and confirmation.get("data", {}).get("type") == "subscription_confirmed"
    
    async def unsubscribe(self, subscription_type: str) -> bool:
        """Unsubscribe from a topic."""
        success = await self.send_message("unsubscribe", {
            "subscription_type": subscription_type
        })
        
        if success:
            self.subscriptions.discard(subscription_type)
        
        # Wait for confirmation
        confirmation = await self.receive_message(timeout=5.0)
        return confirmation and confirmation.get("data", {}).get("type") == "unsubscription_confirmed"
    
    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a prediction request via WebSocket."""
        message_id = str(uuid.uuid4())
        
        success = await self.send_message("prediction_request", {
            "model_id": model_id,
            "input_data": input_data
        }, message_id=message_id)
        
        if not success:
            return None
        
        # Wait for prediction response
        start_time = time.time()
        while time.time() - start_time < self.config.timeout:
            message = await self.receive_message(timeout=1.0)
            if message and message.get("message_id") == message_id:
                if message.get("message_type") == "prediction_response":
                    return message.get("data")
                elif message.get("message_type") == "error":
                    logger.error(f"Prediction error: {message.get('data')}")
                    return None
        
        logger.error("Prediction timeout")
        return None
    
    async def heartbeat(self) -> bool:
        """Send heartbeat and wait for response."""
        message_id = str(uuid.uuid4())
        
        success = await self.send_message("heartbeat", {}, message_id=message_id)
        if not success:
            return False
        
        # Wait for heartbeat response
        response = await self.receive_message(timeout=5.0)
        return response and response.get("message_type") == "heartbeat"


class MCPTestScenarios:
    """Pre-built test scenarios for common MCP testing needs."""
    
    @staticmethod
    def create_mean_reversion_input() -> Dict[str, Any]:
        """Create sample input for mean reversion strategy."""
        return {
            "z_score": 2.5,
            "price": 105.0,
            "moving_average": 100.0,
            "volatility": 0.2,
            "volume_ratio": 1.2,
            "rsi": 65.0,
            "market_regime": 0.7
        }
    
    @staticmethod
    def create_momentum_input() -> Dict[str, Any]:
        """Create sample input for momentum strategy."""
        return {
            "price_change": 0.05,
            "volume_change": 0.3,
            "momentum_score": 0.8,
            "trend_strength": 0.9,
            "volatility": 0.15,
            "market_sentiment": 0.8
        }
    
    @staticmethod
    def create_mirror_trading_input() -> Dict[str, Any]:
        """Create sample input for mirror trading strategy."""
        return {
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
    
    @staticmethod
    def create_swing_trading_input() -> Dict[str, Any]:
        """Create sample input for swing trading strategy."""
        return {
            "price": 100.0,
            "support_levels": [95.0, 90.0, 85.0],
            "resistance_levels": [105.0, 110.0, 115.0],
            "trend_direction": "bullish",
            "volume_profile": "increasing",
            "time_in_cycle": 0.3,
            "risk_metrics": {
                "atr": 2.5,
                "volatility": 0.18
            }
        }
    
    @staticmethod
    async def test_full_prediction_flow(client: MCPHttpClient, model_id: str,
                                      strategy: str = "mean_reversion") -> Dict[str, Any]:
        """Test complete prediction flow."""
        results = {
            "health_check": None,
            "model_metadata": None,
            "prediction": None,
            "strategy_analytics": None,
            "metrics": None
        }
        
        # Health check
        results["health_check"] = client.health_check()
        
        # Get model metadata
        results["model_metadata"] = client.get_model_metadata(model_id)
        
        # Create appropriate input based on strategy
        input_creators = {
            "mean_reversion": MCPTestScenarios.create_mean_reversion_input,
            "momentum": MCPTestScenarios.create_momentum_input,
            "mirror_trading": MCPTestScenarios.create_mirror_trading_input,
            "swing_trading": MCPTestScenarios.create_swing_trading_input
        }
        
        input_data = input_creators.get(strategy, MCPTestScenarios.create_mean_reversion_input)()
        
        # Make prediction
        results["prediction"] = client.predict(model_id, input_data, return_confidence=True)
        
        # Get strategy analytics
        results["strategy_analytics"] = client.get_strategy_analytics(strategy)
        
        # Get server metrics
        results["metrics"] = client.get_metrics()
        
        return results


# Example usage
async def example_usage():
    """Example of using MCP test clients."""
    # HTTP client example
    with MCPHttpClient() as client:
        # Health check
        health = client.health_check()
        print(f"Server healthy: {health.success}")
        
        # List models
        models = client.list_models(strategy_name="momentum")
        if models.success and models.data:
            print(f"Found {len(models.data.get('models', []))} momentum models")
        
        # Make prediction
        if models.data and models.data.get('models'):
            model_id = models.data['models'][0]['model_id']
            input_data = MCPTestScenarios.create_momentum_input()
            
            prediction = client.predict(model_id, input_data)
            if prediction.success:
                print(f"Prediction: {prediction.data}")
                print(f"Processing time: {prediction.processing_time_ms:.2f}ms")
    
    # WebSocket client example
    async with MCPWebSocketClient().connect() as ws_client:
        # Subscribe to model updates
        subscribed = await ws_client.subscribe("model_updates")
        print(f"Subscribed to model updates: {subscribed}")
        
        # Make prediction via WebSocket
        input_data = MCPTestScenarios.create_mean_reversion_input()
        prediction = await ws_client.predict("test_model", input_data)
        if prediction:
            print(f"WebSocket prediction: {prediction}")
        
        # Send heartbeat
        alive = await ws_client.heartbeat()
        print(f"WebSocket connection alive: {alive}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())