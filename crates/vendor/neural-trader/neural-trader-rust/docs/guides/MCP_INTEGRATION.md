# MCP Integration Guide

Complete guide for integrating Neural Trader MCP server with Claude Desktop and other AI assistants.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Claude Desktop Setup](#claude-desktop-setup)
- [Configuration](#configuration)
- [Transport Modes](#transport-modes)
- [Troubleshooting](#troubleshooting)
- [Performance Tuning](#performance-tuning)
- [Advanced Topics](#advanced-topics)

---

## Overview

The Model Context Protocol (MCP) by Anthropic enables AI assistants to interact with external tools and data sources through a standardized JSON-RPC 2.0 protocol. Neural Trader provides 107+ trading tools accessible via MCP.

### Key Features

- **107+ AI-Accessible Tools** - Complete trading toolkit
- **Multiple Transport Layers** - stdio, HTTP, WebSocket
- **Zero Configuration** - Works out of the box with Claude Desktop
- **High Performance** - Rust-powered NAPI-RS implementation
- **Type-Safe** - Full TypeScript definitions
- **GPU Acceleration** - Optional CUDA/Metal support

### Architecture

```
┌─────────────────────┐
│   Claude Desktop    │
│    (AI Assistant)   │
└──────────┬──────────┘
           │ JSON-RPC 2.0
           │ (stdio/HTTP/WS)
┌──────────▼──────────┐
│   Neural Trader     │
│    MCP Server       │
│  (NAPI-RS/Rust)     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Trading Systems    │
│  • Strategies       │
│  • Neural Networks  │
│  • Risk Management  │
│  • Brokers          │
└─────────────────────┘
```

---

## Quick Start

### Installation

Install globally for CLI access:

```bash
npm install -g @neural-trader/mcp
```

Or use with npx (recommended):

```bash
npx @neural-trader/mcp
```

### Verify Installation

Test the server:

```bash
# Start server in stdio mode
npx @neural-trader/mcp

# Server will output:
# Neural Trader MCP Server v1.0.0
# Transport: stdio
# Tools: 107
# Ready for connections
```

### First MCP Request

Test with a simple ping:

```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping","arguments":{}},"id":1}' | \
  npx @neural-trader/mcp
```

Expected response:

```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "ok",
    "timestamp": "2025-01-14T10:30:00Z"
  },
  "id": 1
}
```

---

## Claude Desktop Setup

### macOS

**Step 1: Locate Configuration File**

```bash
# Configuration file location
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Step 2: Edit Configuration**

Create or edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Step 3: Restart Claude Desktop**

1. Quit Claude Desktop completely (Cmd+Q)
2. Relaunch Claude Desktop
3. Wait for MCP server to initialize (~5 seconds)

**Step 4: Verify Integration**

Ask Claude:

> "What Neural Trader tools do you have access to?"

Claude should respond with a list of 107+ tools.

### Windows

**Step 1: Locate Configuration File**

```powershell
# Configuration file location
%APPDATA%\Claude\claude_desktop_config.json
```

**Step 2: Edit Configuration**

Create or edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Step 3: Restart Claude Desktop**

1. Right-click system tray icon → Exit
2. Relaunch Claude Desktop
3. Wait for initialization

**Step 4: Verify Integration**

Test with a simple query:

> "List available trading strategies"

### Linux

**Step 1: Locate Configuration File**

```bash
# Configuration file location
~/.config/claude/claude_desktop_config.json
```

**Step 2: Edit Configuration**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Step 3: Restart and Verify**

Same as macOS instructions.

---

## Configuration

### Basic Configuration

Minimal configuration for stdio transport:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

### Advanced Configuration

With custom options and environment variables:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": [
        "@neural-trader/mcp",
        "--transport", "stdio",
        "--max-connections", "50"
      ],
      "env": {
        "NEURAL_TRADER_API_KEY": "your_api_key_here",
        "ENABLE_GPU": "true",
        "LOG_LEVEL": "info",
        "ALPACA_API_KEY": "your_alpaca_key",
        "ALPACA_API_SECRET": "your_alpaca_secret"
      }
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEURAL_TRADER_API_KEY` | API key for Neural Trader | None |
| `ENABLE_GPU` | Enable GPU acceleration | `false` |
| `MCP_TRANSPORT` | Transport mode | `stdio` |
| `MCP_PORT` | Port for HTTP/WebSocket | `3000` |
| `MCP_HOST` | Host address | `localhost` |
| `MAX_CONNECTIONS` | Max concurrent connections | `100` |
| `LOG_LEVEL` | Logging level | `info` |
| `LOG_FILE` | Log file path | stdout |
| `ALPACA_API_KEY` | Alpaca broker API key | None |
| `ALPACA_API_SECRET` | Alpaca broker API secret | None |

### Configuration Validation

Test your configuration:

```bash
# Validate configuration
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | jq .

# Test server startup
npx @neural-trader/mcp

# Check logs
tail -f ~/Library/Logs/Claude/mcp*.log
```

---

## Transport Modes

### stdio (Default)

Best for Claude Desktop and local AI assistants.

**Configuration:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Pros:**
- Zero configuration
- Most secure (no network exposure)
- Lowest latency
- Automatic lifecycle management

**Cons:**
- Local only
- Single connection

**Use Cases:**
- Claude Desktop integration
- Local AI assistants
- Development and testing

### HTTP

For web-based integrations and REST APIs.

**Start Server:**

```bash
npx @neural-trader/mcp --transport http --port 8080
```

**Configuration:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": [
        "@neural-trader/mcp",
        "--transport", "http",
        "--port", "8080",
        "--host", "localhost"
      ]
    }
  }
}
```

**Example Request:**

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "list_strategies",
      "arguments": {}
    },
    "id": 1
  }'
```

**Pros:**
- Multiple connections
- Network accessible
- Standard REST API
- CORS support

**Cons:**
- Higher latency than stdio
- Requires port management
- Network security considerations

**Use Cases:**
- Web applications
- Remote AI assistants
- API integrations
- Multiple clients

### WebSocket

For real-time bidirectional communication.

**Start Server:**

```bash
npx @neural-trader/mcp --transport websocket --port 3000
```

**Configuration:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": [
        "@neural-trader/mcp",
        "--transport", "websocket",
        "--port", "3000"
      ]
    }
  }
}
```

**Example Client:**

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3000');

ws.on('open', () => {
  ws.send(JSON.stringify({
    jsonrpc: '2.0',
    method: 'tools/call',
    params: {
      name: 'get_portfolio_status',
      arguments: { include_analytics: true }
    },
    id: 1
  }));
});

ws.on('message', (data) => {
  const response = JSON.parse(data);
  console.log('Response:', response);
});
```

**Pros:**
- Real-time updates
- Bidirectional communication
- Low latency
- Persistent connection

**Cons:**
- More complex setup
- Connection management
- Firewall considerations

**Use Cases:**
- Real-time dashboards
- Live trading systems
- Streaming data
- Event-driven applications

---

## Troubleshooting

### Server Won't Start

**Problem:** Server fails to start with port error.

**Solution:**

```bash
# Check port usage
lsof -i :3000

# Kill existing process
kill -9 <PID>

# Or use different port
npx @neural-trader/mcp --port 8080
```

### Claude Desktop Can't Find Server

**Problem:** Claude doesn't show Neural Trader tools.

**Solution:**

1. **Verify Configuration:**

```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. **Check Logs:**

```bash
# macOS
tail -f ~/Library/Logs/Claude/mcp*.log

# Windows
type %APPDATA%\Claude\Logs\mcp*.log

# Linux
tail -f ~/.config/claude/logs/mcp*.log
```

3. **Test Server Manually:**

```bash
npx @neural-trader/mcp
# Should output "Ready for connections"
```

4. **Restart Claude Desktop:**

Completely quit and relaunch (not just close window).

### Tools Return "Method Not Found"

**Problem:** MCP calls fail with method not found.

**Solution:**

1. **List Available Tools:**

```bash
echo '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":1}' | \
  npx @neural-trader/mcp
```

2. **Verify Tool Name:**

Tool names are case-sensitive. Use exact names from the list.

3. **Check Server Version:**

```bash
npx @neural-trader/mcp --version
```

Update if needed:

```bash
npm update -g @neural-trader/mcp
```

### Connection Timeouts

**Problem:** Requests timeout or disconnect.

**Solution:**

1. **Increase Timeout:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"],
      "env": {
        "REQUEST_TIMEOUT": "60000"
      }
    }
  }
}
```

2. **Enable Keep-Alive (WebSocket):**

```javascript
const ws = new WebSocket('ws://localhost:3000');
setInterval(() => {
  ws.ping();
}, 30000);
```

3. **Check Network:**

```bash
# Test connectivity
curl http://localhost:8080/health

# Check firewall
sudo ufw status
```

### GPU Not Working

**Problem:** GPU acceleration not working.

**Solution:**

1. **Verify GPU Support:**

```bash
# NVIDIA
nvidia-smi

# Apple Silicon
system_profiler SPDisplaysDataType
```

2. **Enable GPU:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "env": {
        "ENABLE_GPU": "true",
        "CUDA_VISIBLE_DEVICES": "0"
      }
    }
  }
}
```

3. **Check CUDA Installation:**

```bash
nvcc --version
```

### Permission Denied Errors

**Problem:** Permission errors when starting server.

**Solution:**

```bash
# Fix npm permissions
sudo chown -R $USER ~/.npm

# Or use npx (recommended)
npx @neural-trader/mcp

# Check file permissions
ls -la ~/Library/Application\ Support/Claude/
```

---

## Performance Tuning

### Optimize for Low Latency

**Use stdio transport:**

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp"]
    }
  }
}
```

**Enable GPU acceleration:**

```json
{
  "env": {
    "ENABLE_GPU": "true"
  }
}
```

### Optimize for High Throughput

**Use HTTP/WebSocket:**

```bash
npx @neural-trader/mcp --transport http --max-connections 100
```

**Enable connection pooling:**

```javascript
const server = new McpServer({
  transport: 'http',
  maxConnections: 100,
  keepAlive: true
});
```

### Memory Optimization

**Limit concurrent operations:**

```json
{
  "env": {
    "MAX_CONNECTIONS": "50",
    "MAX_MEMORY_MB": "2048"
  }
}
```

**Use Rust implementation:**

For production, use the optimized Rust binary:

```bash
cd neural-trader-rust
cargo build --release
./target/release/mcp-server
```

Benefits:
- 10-100x faster
- Lower memory usage
- Native SIMD support

### Caching

**Enable response caching:**

```json
{
  "env": {
    "ENABLE_CACHE": "true",
    "CACHE_TTL_SECONDS": "300"
  }
}
```

### Monitoring

**Enable metrics:**

```json
{
  "env": {
    "ENABLE_METRICS": "true",
    "METRICS_PORT": "9090"
  }
}
```

**View metrics:**

```bash
curl http://localhost:9090/metrics
```

---

## Advanced Topics

### Custom Tool Registration

Register custom tools with the MCP server:

```javascript
const { McpServer } = require('@neural-trader/mcp');

const server = new McpServer();

server.registerTool('custom_indicator', async (params) => {
  const { symbol, period } = params;

  // Custom logic
  const value = await calculateCustomIndicator(symbol, period);

  return {
    symbol,
    indicator_value: value,
    timestamp: new Date().toISOString()
  };
});

await server.start();
```

### Multi-Server Setup

Run multiple MCP servers:

```json
{
  "mcpServers": {
    "neural-trader-primary": {
      "command": "npx",
      "args": ["@neural-trader/mcp", "--port", "3000"]
    },
    "neural-trader-backup": {
      "command": "npx",
      "args": ["@neural-trader/mcp", "--port", "3001"]
    }
  }
}
```

### Load Balancing

Use nginx for load balancing:

```nginx
upstream neural_trader {
    server localhost:3000;
    server localhost:3001;
    server localhost:3002;
}

server {
    listen 8080;

    location /mcp {
        proxy_pass http://neural_trader;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Authentication

Add API key authentication:

```json
{
  "mcpServers": {
    "neural-trader": {
      "env": {
        "REQUIRE_AUTH": "true",
        "API_KEY": "your-secret-key"
      }
    }
  }
}
```

**Client request:**

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### SSL/TLS

Enable HTTPS for production:

```bash
npx @neural-trader/mcp \
  --transport https \
  --port 443 \
  --ssl-cert /path/to/cert.pem \
  --ssl-key /path/to/key.pem
```

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM node:18-alpine

WORKDIR /app

RUN npm install -g @neural-trader/mcp

EXPOSE 3000

CMD ["neural-trader-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "3000"]
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  neural-trader-mcp:
    build: .
    ports:
      - "3000:3000"
    environment:
      - ENABLE_GPU=true
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
```

**Run:**

```bash
docker-compose up -d
```

### Kubernetes Deployment

**Deployment:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-trader-mcp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-trader-mcp
  template:
    metadata:
      labels:
        app: neural-trader-mcp
    spec:
      containers:
      - name: mcp-server
        image: neural-trader/mcp:latest
        ports:
        - containerPort: 3000
        env:
        - name: ENABLE_GPU
          value: "true"
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Service:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-trader-mcp
spec:
  selector:
    app: neural-trader-mcp
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

---

## Best Practices

### Security

1. **Never commit API keys:**
   - Use environment variables
   - Use secret management services

2. **Enable authentication:**
   - Require API keys for HTTP/WebSocket
   - Use TLS for production

3. **Limit network exposure:**
   - Use stdio for local-only access
   - Firewall rules for HTTP/WebSocket
   - VPN for remote access

### Reliability

1. **Health checks:**
   ```bash
   curl http://localhost:8080/health
   ```

2. **Graceful shutdown:**
   ```javascript
   process.on('SIGTERM', async () => {
     await server.stop();
     process.exit(0);
   });
   ```

3. **Error handling:**
   ```javascript
   try {
     await server.callTool('neural_forecast', params);
   } catch (error) {
     console.error('Tool error:', error);
     // Handle error
   }
   ```

### Monitoring

1. **Enable logging:**
   ```json
   {
     "env": {
       "LOG_LEVEL": "info",
       "LOG_FILE": "/var/log/neural-trader-mcp.log"
     }
   }
   ```

2. **Track metrics:**
   - Request latency
   - Error rates
   - GPU utilization
   - Memory usage

3. **Set up alerts:**
   - High error rates
   - Resource exhaustion
   - Connection failures

---

## Next Steps

- [API Reference](/workspaces/neural-trader/neural-trader-rust/docs/api/NEURAL_TRADER_MCP_API.md)
- [NAPI Development Guide](/workspaces/neural-trader/neural-trader-rust/docs/development/NAPI_DEVELOPMENT.md)
- [Examples](/workspaces/neural-trader/neural-trader-rust/docs/examples/)

---

**Last Updated**: 2025-01-14
**Maintained By**: Neural Trader Team
