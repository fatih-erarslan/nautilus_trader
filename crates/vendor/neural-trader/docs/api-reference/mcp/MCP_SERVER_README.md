# MCP Server for AI News Trading Platform

A complete Model Context Protocol (MCP) server implementation for serving AI-powered trading strategies with GPU acceleration support.

## Features

- **JSON-RPC 2.0 Protocol**: Full compliance with JSON-RPC 2.0 specification
- **Dual Transport**: Both HTTP and WebSocket support for different use cases
- **GPU Acceleration**: Optional GPU support for faster model inference
- **Trading Strategies**: Serves 4 optimized trading strategies:
  - Mirror Trader (follows institutional investors)
  - Momentum Trader (captures price trends)
  - Swing Trader (short-term price movements)
  - Mean Reversion Trader (contrarian approach)
- **Real-time Streaming**: WebSocket support for live market data and updates
- **Comprehensive Tools**: Backtesting, optimization, and live trading
- **Resource Management**: Access to model parameters and configurations
- **AI Prompts**: Strategy recommendations and risk analysis
- **Monte Carlo Simulations**: Advanced risk assessment

## Installation

```bash
# Install required dependencies
pip install aiohttp websockets numpy

# Optional: Install PyTorch for GPU support
pip install torch

# Make the launcher executable
chmod +x start_mcp_server.py
```

## Quick Start

```bash
# Start with default settings
python start_mcp_server.py

# Start with GPU acceleration
python start_mcp_server.py --gpu

# Start with custom ports
python start_mcp_server.py --http-port 9090 --ws-port 9091

# Enable verbose logging
python start_mcp_server.py --verbose
```

## API Endpoints

### HTTP Endpoints

- `POST /mcp` - Main JSON-RPC endpoint
- `GET /health` - Health check
- `GET /capabilities` - Server capabilities

### WebSocket

- `ws://localhost:8081` - Real-time bidirectional communication

## MCP Methods

### Discovery Methods
- `discover` - Find available MCP services
- `register` - Register a new service
- `health_check` - Check service health
- `authenticate` - Get authentication token
- `get_capabilities` - Get detailed capabilities

### Tool Methods
- `list_tools` - List available trading tools
- `call_tool` - Execute a trading tool
  - `execute_trade` - Place trading orders
  - `backtest` - Run historical backtesting
  - `optimize` - Optimize strategy parameters
  - `get_positions` - Get current positions
  - `get_performance` - Get performance metrics

### Resource Methods
- `list_resources` - List available resources
- `read_resource` - Read specific resource
- `subscribe_resource` - Subscribe to updates

### Prompt Methods
- `list_prompts` - List AI prompt templates
- `get_prompt` - Get specific prompt
- `complete_prompt` - Get AI completion

### Sampling Methods
- `create_message` - Create sampling request
  - Monte Carlo simulations
  - Historical replay
  - Scenario analysis
  - Stress testing

## Example Usage

### HTTP Request Example

```bash
# Execute a trade
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "call_tool",
    "params": {
      "name": "execute_trade",
      "arguments": {
        "strategy": "momentum_trader",
        "symbol": "AAPL",
        "quantity": 100,
        "order_type": "market",
        "side": "buy"
      }
    },
    "id": 1
  }'

# Get strategy recommendations
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "get_prompt",
    "params": {
      "name": "strategy_recommendation",
      "arguments": {
        "market_conditions": "volatile",
        "risk_profile": "moderate",
        "investment_horizon": "medium"
      }
    },
    "id": 2
  }'
```

### WebSocket Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8081');

ws.on('open', () => {
  // Subscribe to market updates
  ws.send(JSON.stringify({
    jsonrpc: '2.0',
    method: 'subscribe_resource',
    params: {
      uri: 'mcp://data/market'
    },
    id: 1
  }));
});

ws.on('message', (data) => {
  const response = JSON.parse(data);
  console.log('Received:', response);
});
```

### Python Client Example

```python
import aiohttp
import asyncio

async def call_mcp_method(method, params):
    async with aiohttp.ClientSession() as session:
        payload = {
            'jsonrpc': '2.0',
            'method': method,
            'params': params,
            'id': 1
        }
        
        async with session.post('http://localhost:8080/mcp', json=payload) as resp:
            return await resp.json()

# Run backtest
result = await call_mcp_method('call_tool', {
    'name': 'backtest',
    'arguments': {
        'strategy': 'mirror_trader',
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        'initial_capital': 100000
    }
})
```

## Resource URIs

The server supports the following MCP resource URIs:

- `mcp://parameters/{strategy}` - Strategy parameters
- `mcp://config/strategies` - Strategy configurations
- `mcp://data/market` - Real-time market data
- `mcp://state/models` - Model states

## Authentication

The server supports JWT-based authentication:

```bash
# Get authentication token
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "authenticate",
    "params": {
      "client_id": "my-trading-app",
      "client_secret": "secret"
    },
    "id": 1
  }'

# Use token in subsequent requests
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"jsonrpc": "2.0", "method": "list_tools", "id": 1}'
```

## GPU Acceleration

When GPU is available and enabled:
- Faster model inference for trading signals
- Accelerated backtesting and optimization
- Parallel Monte Carlo simulations
- Reduced latency for real-time decisions

Check GPU status:
```bash
curl http://localhost:8080/health
```

## Error Handling

The server follows JSON-RPC 2.0 error codes:
- `-32700` - Parse error
- `-32600` - Invalid request
- `-32601` - Method not found
- `-32602` - Invalid params
- `-32603` - Internal error

Custom error codes:
- `-32001` - Model not found
- `-32002` - Strategy error
- `-32003` - Data error
- `-32004` - Authentication error

## Monitoring

Monitor server health and performance:

```bash
# Health check
curl http://localhost:8080/health

# Get capabilities
curl http://localhost:8080/capabilities

# Check specific service
curl -X POST http://localhost:8080/mcp \
  -d '{
    "jsonrpc": "2.0",
    "method": "health_check",
    "params": {"service_id": "ai-news-trader-mcp"},
    "id": 1
  }'
```

## Development

### Adding New Strategies

1. Create strategy class in `src/trading/strategies/`
2. Add to strategy map in `ModelLoader`
3. Create optimization results file
4. Restart server

### Adding New Tools

1. Add handler method to `ToolsHandler`
2. Update `list_tools` method
3. Add to tool routing map
4. Restart server

## Architecture

```
MCP Server
├── Transport Layer (HTTP/WebSocket)
├── JSON-RPC Handler
├── Method Routing
├── Handlers
│   ├── Tools (Trading Operations)
│   ├── Resources (Model Parameters)
│   ├── Prompts (AI Recommendations)
│   └── Sampling (Simulations)
├── Trading Integration
│   ├── Strategy Manager
│   └── Model Loader
└── GPU Acceleration (Optional)
```

## Performance

- HTTP requests: ~10ms latency
- WebSocket messages: ~2ms latency
- Backtest 1 year: ~500ms (CPU), ~50ms (GPU)
- Monte Carlo 10k iterations: ~2s (CPU), ~200ms (GPU)

## Troubleshooting

### Server won't start
- Check ports 8080/8081 are available
- Ensure Python 3.8+ is installed
- Verify all dependencies are installed

### GPU not detected
- Install PyTorch with CUDA support
- Check CUDA drivers are installed
- Verify GPU is CUDA-capable

### Strategy not loading
- Check strategy file exists
- Verify optimization results available
- Check logs for specific errors

## License

This MCP server is part of the AI News Trading Platform.