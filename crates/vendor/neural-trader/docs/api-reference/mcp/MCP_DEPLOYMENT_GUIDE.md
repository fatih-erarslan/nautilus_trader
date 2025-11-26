# MCP Server Deployment Guide

## Quick Start

```bash
# Install dependencies
pip install -r requirements-mcp.txt

# Start the server
python start_mcp_server.py

# Test the server
python test_mcp_server.py

# Run example client
python mcp_client_example.py
```

## Production Deployment

### 1. Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements-mcp.txt .
RUN pip install --no-cache-dir -r requirements-mcp.txt

# Copy application
COPY src/ ./src/
COPY start_mcp_server.py .

# Expose ports
EXPOSE 8080 8081

# Run server
CMD ["python", "start_mcp_server.py", "--host", "0.0.0.0"]
```

Build and run:
```bash
docker build -t ai-news-trader-mcp .
docker run -p 8080:8080 -p 8081:8081 ai-news-trader-mcp
```

### 2. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8080:8080"  # HTTP
      - "8081:8081"  # WebSocket
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./optimization_results.json:/app/optimization_results.json:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Kubernetes Deployment

Create `mcp-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: ai-news-trader-mcp:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 8081
          name: websocket
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: websocket
    port: 8081
    targetPort: 8081
  type: LoadBalancer
```

### 4. GPU-Enabled Deployment

For GPU support, modify the Docker image:

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Rest of Dockerfile...
```

And update Kubernetes deployment:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

## Environment Variables

Configure the server using environment variables:

```bash
# Server configuration
export MCP_HOST=0.0.0.0
export MCP_HTTP_PORT=8080
export MCP_WS_PORT=8081

# GPU configuration
export MCP_GPU_ENABLED=true
export CUDA_VISIBLE_DEVICES=0

# Logging
export MCP_LOG_LEVEL=INFO
```

## SSL/TLS Configuration

For production, use HTTPS and WSS:

1. Generate certificates:
```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

2. Update server to use SSL:
```python
# In start_mcp_server.py
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_cert_chain('cert.pem', 'key.pem')

# Apply to HTTP server
await http_site.start(ssl=ssl_context)

# Apply to WebSocket server
await websockets.serve(handler, host, port, ssl=ssl_context)
```

## Load Balancing

Use nginx for load balancing multiple MCP servers:

```nginx
upstream mcp_http {
    server mcp1:8080;
    server mcp2:8080;
    server mcp3:8080;
}

upstream mcp_ws {
    server mcp1:8081;
    server mcp2:8081;
    server mcp3:8081;
}

server {
    listen 443 ssl;
    
    location /mcp {
        proxy_pass http://mcp_http;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws {
        proxy_pass http://mcp_ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring

### Prometheus Metrics

Add metrics endpoint:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
request_count = Counter('mcp_requests_total', 'Total requests')
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')

# Add metrics endpoint
@app.route('/metrics')
async def metrics(request):
    return web.Response(text=generate_latest())
```

### Grafana Dashboard

Import dashboard for monitoring:
- Request rate
- Response times
- Error rate
- Active connections
- GPU utilization

## Security

### API Key Authentication

```python
API_KEYS = {
    "prod-key-1": "production-client",
    "dev-key-1": "development-client"
}

async def check_api_key(request):
    api_key = request.headers.get('X-API-Key')
    if api_key not in API_KEYS:
        raise web.HTTPUnauthorized()
```

### Rate Limiting

```python
from aiohttp_remotes import setup, XForwardedRelaxed
from aiohttp_rate_limiter import rate_limiter, RateLimiter

limiter = RateLimiter()

@limiter.limit("100/minute")
async def handle_request(request):
    # Handle request
```

## Backup and Recovery

### Strategy Parameters Backup

```bash
# Backup optimization results
aws s3 cp optimization_results.json s3://backup-bucket/mcp/
aws s3 cp momentum_optimization_results.json s3://backup-bucket/mcp/
# ... other strategies

# Restore from backup
aws s3 sync s3://backup-bucket/mcp/ ./
```

### Database Integration

For persistent storage:

```python
import asyncpg

class DatabaseStorage:
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            'postgresql://user:pass@localhost/trading'
        )
    
    async def save_trade(self, trade):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO trades (strategy, symbol, quantity, price, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """, trade.strategy, trade.symbol, trade.quantity, trade.price, trade.timestamp)
```

## Performance Tuning

### Connection Pooling

```python
connector = aiohttp.TCPConnector(
    limit=100,
    limit_per_host=30,
    ttl_dns_cache=300
)
```

### Caching

```python
from aiocache import Cache

cache = Cache(Cache.MEMORY)

@cached(ttl=60)
async def get_market_data(symbol):
    # Fetch market data
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   lsof -i :8080
   kill -9 <PID>
   ```

2. **Memory leaks**
   ```python
   import tracemalloc
   tracemalloc.start()
   ```

3. **Slow responses**
   - Enable query logging
   - Check database indexes
   - Monitor CPU/memory usage

### Debug Mode

```bash
python start_mcp_server.py --verbose --debug
```

## Health Checks

Implement comprehensive health checks:

```python
async def detailed_health_check():
    return {
        "status": "healthy",
        "checks": {
            "database": await check_database(),
            "strategies": await check_strategies(),
            "gpu": await check_gpu(),
            "memory": check_memory_usage(),
            "disk": check_disk_space()
        }
    }
```

## Scaling Guidelines

- **Horizontal Scaling**: Add more server instances behind load balancer
- **Vertical Scaling**: Increase CPU/memory for compute-intensive operations
- **GPU Scaling**: Use multiple GPUs for parallel model inference
- **Caching**: Implement Redis for shared cache across instances

## Deployment Checklist

- [ ] Install all dependencies
- [ ] Configure environment variables
- [ ] Set up SSL certificates
- [ ] Configure firewall rules
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Test health endpoints
- [ ] Load test the deployment
- [ ] Set up log aggregation
- [ ] Configure alerts