# MCP Testing Guide

## Overview

This guide provides comprehensive documentation for testing the Model Context Protocol (MCP) implementation in the AI News Trading Platform. The test suite ensures MCP compliance, validates functionality, and benchmarks performance.

## Test Suite Structure

```
tests/mcp/
├── __init__.py
├── test_mcp_protocol.py      # Protocol compliance tests
├── test_mcp_transport.py     # HTTP/WebSocket transport tests
├── test_mcp_integration.py   # Model integration tests
├── test_mcp_performance.py   # Performance benchmarks
├── mcp_test_client.py        # Testing utilities
├── test_runner.py            # Test suite runner
└── MCP_TESTING_GUIDE.md      # This documentation
```

## Quick Start

### Running All Tests

```bash
# Run complete test suite
python tests/mcp/test_runner.py --all

# Run with HTML report
python tests/mcp/test_runner.py --all --html

# Run specific module
python tests/mcp/test_runner.py --module test_mcp_protocol

# Run tests matching pattern
python tests/mcp/test_runner.py --pattern "test_websocket"
```

### Running Individual Test Files

```bash
# Protocol compliance tests
pytest tests/mcp/test_mcp_protocol.py -v

# Transport layer tests
pytest tests/mcp/test_mcp_transport.py -v

# Integration tests
pytest tests/mcp/test_mcp_integration.py -v

# Performance tests
pytest tests/mcp/test_mcp_performance.py -v
```

## Test Categories

### 1. Protocol Compliance Tests (`test_mcp_protocol.py`)

Tests MCP message formats, JSON-RPC 2.0 compliance, and error handling.

**Key Test Classes:**
- `TestMCPProtocolCompliance`: Validates message structures and protocol adherence
- `TestMCPErrorHandling`: Tests error scenarios and edge cases
- `TestMCPConcurrency`: Validates thread-safe operations

**Example Test:**
```python
def test_mcp_message_creation(self):
    """Test MCP message creation and structure."""
    message = MCPMessage(
        message_type=MCPMessageType.REQUEST,
        request_id=str(uuid.uuid4()),
        method="predict",
        params={"model_id": "test_model_123", "input_data": {"x": 1.0}},
        timestamp=datetime.now()
    )
    
    assert message.message_type == MCPMessageType.REQUEST
    assert message.method == "predict"
```

### 2. Transport Layer Tests (`test_mcp_transport.py`)

Tests HTTP REST endpoints and WebSocket connections.

**Key Test Classes:**
- `TestHTTPTransport`: Validates REST API endpoints
- `TestWebSocketTransport`: Tests real-time WebSocket functionality
- `TestTransportSecurity`: Security and authentication tests

**Example HTTP Test:**
```python
def test_prediction_endpoint(self, test_client):
    """Test prediction endpoint."""
    request_data = {
        "model_id": "test_model_123",
        "input_data": {
            "z_score": 2.5,
            "price": 105.0,
            "moving_average": 100.0,
            "volatility": 0.2
        },
        "return_confidence": True
    }
    
    response = test_client.post("/models/predict", json=request_data)
    assert response.status_code == 200
```

**Example WebSocket Test:**
```python
@pytest.mark.asyncio
async def test_websocket_subscription(self):
    """Test WebSocket subscription mechanism."""
    async with websockets.connect("ws://127.0.0.1:8892/ws") as websocket:
        # Subscribe to model updates
        subscribe_msg = {
            "message_type": "subscribe",
            "data": {"subscription_type": "model_updates"}
        }
        await websocket.send(json.dumps(subscribe_msg))
```

### 3. Integration Tests (`test_mcp_integration.py`)

Tests MCP server integration with trading models and GPU acceleration.

**Key Test Classes:**
- `TestMCPModelIntegration`: Model loading and prediction tests
- `TestMCPGPUAcceleration`: GPU resource management tests
- `TestMCPRealTimeStreaming`: Streaming data tests
- `TestMCPToolExecution`: Tool execution flow tests

**Example Integration Test:**
```python
def test_strategy_specific_predictions(self, test_client, sample_models):
    """Test strategy-specific prediction logic."""
    mean_reversion_request = {
        "model_id": mean_reversion_id,
        "input_data": {
            "z_score": -2.5,  # Strong oversold signal
            "price": 95.0,
            "moving_average": 100.0,
            "volatility": 0.2
        }
    }
    
    response = test_client.post("/models/predict", json=mean_reversion_request)
    prediction = response.json()["data"]
    assert prediction["action"] == "buy"  # Should buy when oversold
```

### 4. Performance Tests (`test_mcp_performance.py`)

Benchmarks server performance, scalability, and resource usage.

**Key Test Classes:**
- `TestMCPPerformance`: Standard performance tests
- `TestMCPStressTests`: High-load stress tests (manually triggered)

**Performance Metrics Tracked:**
- Request latency (min, avg, median, p95, p99, max)
- Throughput (requests/second)
- CPU and memory usage
- Cache performance
- Concurrent request handling

**Example Performance Test:**
```python
def test_concurrent_request_handling(self, test_client, mock_model):
    """Test concurrent request handling performance."""
    concurrent_levels = [10, 50, 100]
    
    for num_concurrent in concurrent_levels:
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            
        metrics = self.calculate_metrics(latencies, total_time, successful, failed)
        print(f"Concurrent Requests: {num_concurrent}")
        print(metrics)
```

## Using MCP Test Client Utilities

The `mcp_test_client.py` module provides easy-to-use utilities for testing MCP functionality.

### HTTP Client Example

```python
from tests.mcp.mcp_test_client import MCPHttpClient, MCPTestScenarios

# Create client
with MCPHttpClient() as client:
    # Health check
    health = client.health_check()
    print(f"Server healthy: {health.success}")
    
    # List models
    models = client.list_models(strategy_name="momentum")
    
    # Make prediction
    if models.success and models.data['models']:
        model_id = models.data['models'][0]['model_id']
        input_data = MCPTestScenarios.create_momentum_input()
        
        prediction = client.predict(model_id, input_data)
        print(f"Prediction: {prediction.data}")
        print(f"Processing time: {prediction.processing_time_ms:.2f}ms")
```

### WebSocket Client Example

```python
from tests.mcp.mcp_test_client import MCPWebSocketClient

async with MCPWebSocketClient().connect() as ws_client:
    # Subscribe to updates
    subscribed = await ws_client.subscribe("model_updates")
    
    # Make prediction
    input_data = MCPTestScenarios.create_mean_reversion_input()
    prediction = await ws_client.predict("test_model", input_data)
    
    # Send heartbeat
    alive = await ws_client.heartbeat()
```

### Pre-built Test Scenarios

The `MCPTestScenarios` class provides sample inputs for different strategies:

```python
# Mean Reversion input
mean_reversion_input = MCPTestScenarios.create_mean_reversion_input()

# Momentum input  
momentum_input = MCPTestScenarios.create_momentum_input()

# Mirror Trading input
mirror_input = MCPTestScenarios.create_mirror_trading_input()

# Swing Trading input
swing_input = MCPTestScenarios.create_swing_trading_input()
```

## Performance Benchmarking

### Running Performance Tests

```bash
# Standard performance tests
python tests/mcp/test_runner.py --performance

# Include stress tests (high load)
python tests/mcp/test_runner.py --performance --stress
```

### Performance Targets

The MCP server should meet these performance targets:

- **Single Request Latency**: < 100ms
- **Concurrent Requests (100)**: 
  - Average latency < 500ms
  - P95 latency < 2s
  - Success rate > 95%
- **Sustained Load (50 req/s)**:
  - Average latency < 1s
  - Success rate > 90%
- **WebSocket Throughput**: > 100 messages/second
- **Memory Usage**: < 50MB growth per 1000 requests

## Test Configuration

### MCP Server Configuration

The test suite uses the configuration defined in `.root/mcp.json`:

```json
{
  "mcp": {
    "transport": {
      "http": {
        "port": 8000,
        "rate_limiting": {
          "requests_per_minute": 100
        }
      },
      "websocket": {
        "port": 8002,
        "max_connections": 1000
      }
    }
  }
}
```

### Test Environment Setup

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx websockets psutil

# Set environment variables
export MCP_TEST_HTTP_URL="http://localhost:8000"
export MCP_TEST_WS_URL="ws://localhost:8002/ws"
```

## Debugging Failed Tests

### Common Issues and Solutions

1. **Connection Refused Errors**
   - Ensure MCP server is running: `python model_management/mcp_integration/trading_mcp_server.py`
   - Check port availability: `lsof -i :8000` and `lsof -i :8002`

2. **Model Not Found Errors**
   - Create test models using the sample_models fixture
   - Check model storage path configuration

3. **WebSocket Timeout Errors**
   - Increase timeout in test configuration
   - Check WebSocket server logs for errors

4. **Performance Test Failures**
   - Reduce concurrent request levels for slower systems
   - Check system resource availability
   - Consider running stress tests separately

### Verbose Test Output

```bash
# Run with detailed output
pytest tests/mcp/test_mcp_protocol.py -vvs

# Run with logging
pytest tests/mcp/test_mcp_transport.py -v --log-cli-level=DEBUG
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: MCP Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio httpx websockets
    
    - name: Run MCP Tests
      run: |
        python tests/mcp/test_runner.py --all --html
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: mcp-test-results
        path: test_results/
```

## Extending the Test Suite

### Adding New Protocol Tests

```python
class TestNewMCPFeature:
    """Test new MCP feature."""
    
    def test_new_message_type(self):
        """Test new message type handling."""
        message = MCPMessage(
            message_type=MCPMessageType.NEW_TYPE,
            request_id=str(uuid.uuid4()),
            method="new_method",
            params={"param": "value"},
            timestamp=datetime.now()
        )
        
        # Add assertions
        assert message.message_type == MCPMessageType.NEW_TYPE
```

### Adding New Integration Tests

```python
def test_new_strategy_integration(self, test_client, sample_models):
    """Test new trading strategy integration."""
    # Create strategy-specific input
    input_data = {
        "strategy_param_1": 1.0,
        "strategy_param_2": "value"
    }
    
    # Make prediction
    response = test_client.post("/models/predict", json={
        "model_id": "new_strategy_model",
        "input_data": input_data
    })
    
    # Validate response
    assert response.status_code == 200
    prediction = response.json()["data"]
    assert prediction["action"] in ["buy", "sell", "hold"]
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on others
2. **Fixtures**: Use pytest fixtures for common setup/teardown
3. **Mocking**: Mock external dependencies for unit tests
4. **Async Testing**: Use `pytest.mark.asyncio` for async tests
5. **Performance**: Keep individual tests fast (< 1 second)
6. **Documentation**: Document complex test scenarios
7. **Error Messages**: Provide clear assertion messages

## Troubleshooting

### Check MCP Server Status

```python
# Quick server health check
import httpx

response = httpx.get("http://localhost:8000/health")
print(f"Server status: {response.status_code}")
print(f"Response: {response.json()}")
```

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("mcp_test")
```

### Monitor Resource Usage

```python
import psutil

process = psutil.Process()
print(f"CPU: {process.cpu_percent()}%")
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## Summary

The MCP test suite provides comprehensive validation of:

- Protocol compliance with JSON-RPC 2.0
- HTTP and WebSocket transport layers
- Model integration and predictions
- GPU acceleration capabilities
- Performance and scalability
- Real-time streaming functionality

Regular testing ensures the MCP server remains reliable, performant, and compliant with the Model Context Protocol specification.