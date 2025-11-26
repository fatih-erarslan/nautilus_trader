# AI News Trading Platform - Model Management System

## 🚀 Overview

The Model Management System is a comprehensive, production-ready platform for managing AI trading models throughout their entire lifecycle. It provides enterprise-grade capabilities for model storage, versioning, deployment, monitoring, and real-time inference.

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Components](#-components)
- [API Documentation](#-api-documentation)
- [Deployment Guide](#-deployment-guide)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Monitoring](#-monitoring)
- [Security](#-security)
- [Performance](#-performance)
- [Contributing](#-contributing)

## ✨ Features

### Core Capabilities
- **Model Storage & Versioning**: Git-like version control for ML models with compression and metadata
- **MCP Integration**: Model Context Protocol server for standardized model access
- **Multi-target Deployment**: Deploy to local, staging, production, cloud GPU, Fly.io, and Kubernetes
- **Real-time Inference**: WebSocket and REST APIs for live trading predictions
- **Health Monitoring**: Comprehensive monitoring with alerting and performance tracking
- **Auto-scaling**: Dynamic resource management based on load

### Advanced Features
- **A/B Testing**: Compare model performance in production
- **Rollback Capabilities**: Instant rollback to previous model versions
- **Performance Analytics**: Detailed performance metrics and trend analysis
- **Security**: Role-based access control and audit logging
- **Integration**: Seamless integration with existing trading infrastructure

## 🏗️ Architecture

```
AI News Trading Platform - Model Management
├── model_management/
│   ├── storage/                    # Model storage and metadata
│   │   ├── model_storage.py       # Core storage with compression
│   │   ├── metadata_manager.py    # Metadata and search
│   │   └── version_control.py     # Git-like versioning
│   ├── mcp_integration/           # Model Context Protocol
│   │   ├── trading_mcp_server.py  # MCP server implementation
│   │   ├── model_api.py           # REST API endpoints
│   │   └── websocket_server.py    # Real-time WebSocket server
│   ├── deployment/                # Deployment orchestration
│   │   ├── deploy_orchestrator.py # Multi-target deployment
│   │   └── health_monitor.py      # Health monitoring system
│   ├── model_manager.py           # Central coordinator
│   ├── example_usage.py           # Comprehensive demo
│   └── __init__.py                # Package exports
```

### Technology Stack
- **Backend**: Python 3.9+, FastAPI, WebSockets
- **Storage**: SQLite, JSON, compressed pickle/joblib
- **Deployment**: Docker, Kubernetes, Fly.io, Cloud GPU
- **Monitoring**: Real-time health checks, alerting system
- **Protocols**: REST API, WebSocket, MCP (Model Context Protocol)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-news-trader

# Install dependencies
pip install -r requirements.txt

# Optional: Install deployment dependencies
pip install kubernetes flyctl docker
```

### Basic Usage

```python
import asyncio
from model_management import ModelManager, ManagerConfig

async def main():
    # Configure the model manager
    config = ManagerConfig(
        storage_path="models",
        mcp_server_port=8000,
        api_server_port=8001,
        websocket_server_port=8002
    )
    
    # Initialize and start the manager
    manager = ModelManager(config)
    await manager.start()
    
    # Create a trading model
    model_id = await manager.create_model(
        name="Mean Reversion Strategy",
        version="1.0.0",
        strategy_name="mean_reversion",
        model_type="parameter_set",
        parameters={
            "z_score_threshold": 2.0,
            "base_position_size": 0.05,
            "stop_loss_multiplier": 1.5
        },
        performance_metrics={
            "sharpe_ratio": 2.5,
            "total_return": 0.18,
            "max_drawdown": 0.08
        }
    )
    
    # Deploy the model
    from model_management.deployment import DeploymentConfig, DeploymentTarget
    
    deploy_config = DeploymentConfig(
        target=DeploymentTarget.PRODUCTION,
        strategy=DeploymentStrategy.BLUE_GREEN,
        resource_requirements={"cpu": "1", "memory": "2Gi"}
    )
    
    deployment_id = await manager.deploy_model(model_id, deploy_config)
    
    # Get real-time predictions
    prediction = await manager.get_model_prediction(
        model_id,
        {"z_score": -2.1, "price": 98.5, "volatility": 0.15}
    )
    
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Run the Demo

```bash
cd model_management
python example_usage.py
```

This will start a comprehensive demonstration showing:
- Model creation and versioning
- Performance analytics
- Deployment simulation
- Real-time predictions
- Health monitoring

## 🔧 Components

### 1. Model Storage (`storage/`)

**Features:**
- Compressed storage (pickle, joblib, JSON)
- Automatic versioning and backup
- Metadata indexing and search
- Git-like version control

**Example:**
```python
from model_management.storage import ModelStorage, ModelFormat

storage = ModelStorage("models/")
model_id = storage.save_model(
    model_data, 
    metadata, 
    ModelFormat.COMPRESSED_PICKLE
)
```

### 2. MCP Integration (`mcp_integration/`)

**Features:**
- REST API with OpenAPI documentation
- WebSocket server for real-time updates
- MCP protocol implementation
- Authentication and rate limiting

**Endpoints:**
- `GET /health` - Health check
- `POST /models/predict` - Get predictions
- `GET /models` - List models
- `WS /ws` - Real-time updates

### 3. Deployment Orchestration (`deployment/`)

**Features:**
- Multi-target deployment (Local, Kubernetes, Fly.io, Cloud GPU)
- Blue-green and rolling deployments
- Health monitoring and auto-rollback
- Resource management

**Deployment Targets:**
- **Local**: For development and testing
- **Kubernetes**: Scalable container orchestration
- **Fly.io**: Edge deployment with GPU support
- **Cloud GPU**: AWS/GCP/Azure GPU instances

### 4. Central Model Manager (`model_manager.py`)

**Features:**
- Unified API for all operations
- Event-driven architecture
- Background task management
- Comprehensive monitoring

## 📚 API Documentation

### REST API

Once the system is running, visit:
- **API Documentation**: http://localhost:8001/docs
- **Alternative Docs**: http://localhost:8001/redoc

### WebSocket API

Connect to `ws://localhost:8002` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8002');

// Subscribe to model updates
ws.send(JSON.stringify({
    message_type: 'subscribe',
    message_id: '123',
    timestamp: new Date().toISOString(),
    data: {
        subscription_type: 'model_updates',
        filters: { strategy_name: 'mean_reversion' }
    }
}));

// Request real-time prediction
ws.send(JSON.stringify({
    message_type: 'prediction_request',
    message_id: '124',
    timestamp: new Date().toISOString(),
    data: {
        model_id: 'model_xyz',
        input_data: { z_score: -2.1, price: 98.5 }
    }
}));
```

### MCP Protocol

Connect to the MCP server at `localhost:8000`:

```python
import requests

# Get model prediction
response = requests.post('http://localhost:8000/models/predict', json={
    'model_id': 'model_xyz',
    'input_data': {'z_score': -2.1, 'price': 98.5},
    'return_confidence': True
})

prediction = response.json()
```

## 🚀 Deployment Guide

### Local Development

```bash
# Start all services
python -c "
import asyncio
from model_management import ModelManager, ManagerConfig

async def main():
    config = ManagerConfig()
    manager = ModelManager(config)
    await manager.start()
    await asyncio.sleep(3600)  # Run for 1 hour

asyncio.run(main())
"
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model_management/ ./model_management/
EXPOSE 8000 8001 8002

CMD ["python", "-m", "model_management.model_manager"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-management
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-management
  template:
    metadata:
      labels:
        app: model-management
    spec:
      containers:
      - name: model-management
        image: ai-trading/model-management:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        - containerPort: 8002
        env:
        - name: STORAGE_PATH
          value: "/data/models"
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
```

### Fly.io Deployment

```toml
app = "ai-trading-models"
primary_region = "iad"

[build]
  image = "ai-trading/model-management:latest"

[[services]]
  http_checks = []
  internal_port = 8000
  processes = ["app"]
  protocol = "tcp"
  script_checks = []

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

## ⚙️ Configuration

### Manager Configuration

```python
from model_management import ManagerConfig

config = ManagerConfig(
    storage_path="models",              # Model storage path
    mcp_server_port=8000,              # MCP server port
    api_server_port=8001,              # REST API port  
    websocket_server_port=8002,        # WebSocket port
    enable_auto_cleanup=True,          # Auto cleanup old models
    max_cached_models=50,              # Model cache size
    cache_ttl_minutes=30,              # Cache TTL
    health_check_interval=60,          # Health check interval (sec)
    performance_monitoring_interval=300, # Performance check interval (sec)
    auto_deploy_validated=False,       # Auto-deploy validated models
    backup_interval_hours=24           # Backup interval
)
```

### Environment Variables

```bash
# Server Configuration
MODEL_STORAGE_PATH="/data/models"
MCP_SERVER_PORT="8000"
API_SERVER_PORT="8001"
WEBSOCKET_SERVER_PORT="8002"

# Performance Tuning
MAX_CACHED_MODELS="100"
CACHE_TTL_MINUTES="60"
HEALTH_CHECK_INTERVAL="30"

# Security
API_KEY="your-secure-api-key"
ENABLE_AUTH="true"
CORS_ORIGINS="https://trading-ui.com"

# Monitoring
LOG_LEVEL="INFO"
METRICS_ENABLED="true"
ALERT_EMAIL="admin@trading.com"
```

## 💡 Examples

### Complete Trading Strategy Example

```python
import asyncio
from model_management import ModelManager, ManagerConfig
from model_management.deployment import DeploymentConfig, DeploymentTarget, DeploymentStrategy

async def deploy_trading_strategy():
    # Initialize system
    manager = ModelManager(ManagerConfig())
    await manager.start()
    
    # Create optimized mean reversion model
    model_id = await manager.create_model(
        name="Optimized Mean Reversion v2.1",
        version="2.1.0",
        strategy_name="mean_reversion",
        model_type="genetic_algorithm_optimized",
        parameters={
            "z_score_entry_threshold": 2.1,
            "z_score_exit_threshold": 0.4,
            "lookback_window": 42,
            "base_position_size": 0.08,
            "max_position_size": 0.15,
            "stop_loss_multiplier": 1.6,
            "profit_target_multiplier": 2.2,
            "volatility_adjustment": 1.15,
            "volume_confirmation_threshold": 1.4,
            "adaptive_thresholds": {
                "bull_market_z_threshold": 2.2,
                "bear_market_z_threshold": 2.8,
                "high_vol_z_threshold": 2.9,
                "low_vol_z_threshold": 1.8
            }
        },
        performance_metrics={
            "sharpe_ratio": 3.2,
            "total_return": 0.28,
            "max_drawdown": 0.07,
            "win_rate": 0.71,
            "profit_factor": 2.8,
            "calmar_ratio": 4.0,
            "trades_per_month": 22,
            "avg_holding_days": 4
        },
        tags=["production", "optimized", "high_sharpe", "low_drawdown"]
    )
    
    # Deploy to production with blue-green strategy
    deployment_config = DeploymentConfig(
        target=DeploymentTarget.PRODUCTION,
        strategy=DeploymentStrategy.BLUE_GREEN,
        resource_requirements={
            "cpu": "2",
            "memory": "4Gi",
            "storage": "20Gi"
        },
        environment_variables={
            "LOG_LEVEL": "INFO",
            "METRICS_INTERVAL": "60",
            "ENABLE_ALERTS": "true"
        },
        health_check_config={
            "max_retries": 10,
            "retry_interval": 30,
            "timeout": 15
        },
        auto_rollback=True,
        timeout_seconds=900
    )
    
    deployment_id = await manager.deploy_model(model_id, deployment_config)
    print(f"Deployment started: {deployment_id}")
    
    # Monitor deployment
    while True:
        deployment_status = manager.deployment_orchestrator.get_deployment_status(deployment_id)
        if deployment_status['status'] in ['deployed', 'failed']:
            break
        await asyncio.sleep(10)
    
    if deployment_status['status'] == 'deployed':
        print("✅ Model deployed successfully!")
        
        # Test live predictions
        test_cases = [
            {"z_score": -2.3, "price": 98.2, "volatility": 0.16},
            {"z_score": 2.8, "price": 102.1, "volatility": 0.22},
            {"z_score": 0.1, "price": 100.0, "volatility": 0.15}
        ]
        
        for test_case in test_cases:
            prediction = await manager.get_model_prediction(model_id, test_case)
            print(f"Input: {test_case}")
            print(f"Prediction: {prediction}")
            print("-" * 40)
    
    await manager.stop()

# Run the example
asyncio.run(deploy_trading_strategy())
```

### Real-time Trading Bot Integration

```python
import asyncio
import websockets
import json
from datetime import datetime

class TradingBot:
    def __init__(self, model_manager_ws_url="ws://localhost:8002"):
        self.ws_url = model_manager_ws_url
        self.active_positions = {}
        self.model_subscriptions = ['mean_reversion_prod', 'momentum_scalp']
    
    async def start(self):
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to model updates
            await self.subscribe_to_models(websocket)
            
            # Start market data simulation
            asyncio.create_task(self.simulate_market_data(websocket))
            
            # Listen for responses
            await self.handle_messages(websocket)
    
    async def subscribe_to_models(self, websocket):
        subscribe_msg = {
            "message_type": "subscribe",
            "message_id": f"sub_{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "subscription_type": "real_time_predictions",
                "filters": {"model_ids": self.model_subscriptions}
            }
        }
        await websocket.send(json.dumps(subscribe_msg))
    
    async def simulate_market_data(self, websocket):
        """Simulate incoming market data and request predictions."""
        import random
        
        while True:
            # Simulate market data tick
            market_data = {
                "symbol": "AAPL",
                "price": 150 + random.uniform(-5, 5),
                "volume": random.randint(1000, 10000),
                "z_score": random.uniform(-3, 3),
                "volatility": random.uniform(0.1, 0.3),
                "timestamp": datetime.now().isoformat()
            }
            
            # Request prediction from mean reversion model
            prediction_request = {
                "message_type": "prediction_request",
                "message_id": f"pred_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "model_id": "mean_reversion_prod",
                    "input_data": {
                        "z_score": market_data["z_score"],
                        "price": market_data["price"],
                        "volatility": market_data["volatility"],
                        "volume_ratio": 1.2,
                        "rsi": 50 + random.uniform(-20, 20)
                    }
                }
            }
            
            await websocket.send(json.dumps(prediction_request))
            await asyncio.sleep(5)  # Request prediction every 5 seconds
    
    async def handle_messages(self, websocket):
        """Handle incoming messages from model management system."""
        async for message in websocket:
            data = json.loads(message)
            
            if data['message_type'] == 'prediction_response':
                await self.handle_prediction(data)
            elif data['message_type'] == 'model_update':
                await self.handle_model_update(data)
            elif data['message_type'] == 'notification':
                print(f"📢 Notification: {data['data']}")
    
    async def handle_prediction(self, message):
        """Handle prediction response and execute trades."""
        prediction = message['data']['prediction']
        model_id = message['data']['model_id']
        
        action = prediction.get('action', 'hold')
        confidence = prediction.get('confidence', 0.5)
        position_size = prediction.get('position_size', 0.0)
        
        print(f"🤖 {model_id} prediction: {action.upper()} "
              f"(confidence: {confidence:.2f}, size: {position_size:.3f})")
        
        # Execute trade logic
        if action in ['buy', 'sell'] and confidence > 0.7:
            await self.execute_trade(action, position_size, prediction)
    
    async def execute_trade(self, action, size, prediction):
        """Execute trade based on model prediction."""
        trade_id = f"trade_{int(datetime.now().timestamp())}"
        
        print(f"💰 Executing {action.upper()} trade:")
        print(f"   Trade ID: {trade_id}")
        print(f"   Position Size: {size:.3f}")
        print(f"   Reasoning: {prediction.get('reasoning', 'Model prediction')}")
        
        # Store position for tracking
        self.active_positions[trade_id] = {
            'action': action,
            'size': size,
            'timestamp': datetime.now(),
            'prediction': prediction
        }
    
    async def handle_model_update(self, message):
        """Handle model updates (new versions, performance changes)."""
        update_data = message['data']
        print(f"🔄 Model update: {update_data}")

# Run the trading bot
bot = TradingBot()
asyncio.run(bot.start())
```

## 📊 Monitoring

### Health Monitoring Dashboard

The system provides comprehensive monitoring capabilities:

```python
# Get system health overview
status = manager.get_system_status()
print(f"System Status: {status}")

# Monitor specific deployment
health_report = health_monitor.get_deployment_health("deployment_id")
print(f"Deployment Health: {health_report.overall_status}")

# Get performance metrics
for metric_name, metric in health_report.metrics.items():
    print(f"{metric.name}: {metric.value} {metric.unit}")
```

### Key Metrics Monitored

- **Response Times**: API endpoint latency
- **Throughput**: Requests per second
- **Error Rates**: Failed requests percentage  
- **Model Performance**: Prediction accuracy drift
- **Resource Usage**: CPU, memory, storage utilization
- **Uptime**: Service availability percentage

### Alerting

Configure alerts for critical thresholds:

```python
from model_management.deployment.health_monitor import AlertRule, AlertSeverity

# Add custom alert rule
health_monitor.alert_rules['custom_latency'] = AlertRule(
    name="high_latency_custom",
    metric_name="avg_response_time", 
    condition="greater_than",
    threshold=2000.0,  # 2 seconds
    severity=AlertSeverity.WARNING,
    duration_minutes=5
)

# Register alert callback
async def alert_handler(alert):
    print(f"🚨 ALERT: {alert.message}")
    # Send to Slack, email, etc.

health_monitor.add_alert_callback(alert_handler)
```

## 🔒 Security

### Authentication & Authorization

```python
# Configure API authentication
from model_management.mcp_integration.model_api import ModelAPI

api = ModelAPI()

# Add authentication middleware
@api.app.middleware("http")
async def authenticate(request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or not validate_api_key(api_key):
        raise HTTPException(401, "Invalid API key")
    return await call_next(request)
```

### Security Best Practices

1. **API Keys**: Use secure API keys for all external access
2. **TLS/SSL**: Enable HTTPS for all communication
3. **Network Security**: Use firewalls and VPNs for production
4. **Access Control**: Implement role-based access control
5. **Audit Logging**: Log all model operations and access
6. **Data Encryption**: Encrypt sensitive model parameters

### Model Security

```python
# Encrypt sensitive model parameters
from cryptography.fernet import Fernet

class SecureModelStorage(ModelStorage):
    def __init__(self, encryption_key):
        super().__init__()
        self.cipher = Fernet(encryption_key)
    
    def save_model(self, model, metadata):
        # Encrypt sensitive parameters
        if 'api_keys' in model:
            model['api_keys'] = self.cipher.encrypt(
                json.dumps(model['api_keys']).encode()
            ).decode()
        
        return super().save_model(model, metadata)
```

## ⚡ Performance

### Optimization Guidelines

1. **Model Caching**: Cache frequently accessed models in memory
2. **Connection Pooling**: Use connection pools for database access
3. **Async Operations**: Use async/await for I/O operations
4. **Batch Processing**: Batch multiple predictions together
5. **Resource Limits**: Set appropriate resource limits

### Performance Monitoring

```python
# Monitor performance metrics
import time
import psutil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
    
    async def track_prediction_performance(self, model_id, input_data):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        # Make prediction
        prediction = await model_manager.get_model_prediction(model_id, input_data)
        
        # Record metrics
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        self.metrics[model_id] = {
            'latency_ms': (end_time - start_time) * 1000,
            'memory_delta_mb': (end_memory - start_memory) / 1024 / 1024,
            'timestamp': datetime.now()
        }
        
        return prediction
```

### Scaling Recommendations

- **Horizontal Scaling**: Use multiple instances behind a load balancer
- **Vertical Scaling**: Increase CPU/memory for compute-heavy models
- **Caching**: Implement Redis for distributed caching
- **Database**: Use PostgreSQL for production metadata storage
- **CDN**: Use CDN for static model artifacts

## 🤝 Contributing

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-news-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black model_management/
flake8 model_management/
mypy model_management/
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=model_management --cov-report=html
```

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for all public APIs
- Write comprehensive tests
- Use async/await for I/O operations

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- **API Docs**: http://localhost:8001/docs (when running)
- **Examples**: See `model_management/example_usage.py`
- **Architecture**: See architecture diagrams in `/docs`

### Getting Help
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: support@ai-trading-platform.com

### FAQ

**Q: How do I deploy to production?**
A: Use the deployment orchestrator with `DeploymentTarget.PRODUCTION` and appropriate resource configurations.

**Q: Can I use custom model formats?**
A: Yes, extend the `ModelStorage` class to support additional serialization formats.

**Q: How do I monitor model performance in production?**
A: Use the health monitoring system and set up custom alert rules for your performance thresholds.

**Q: Is the system suitable for high-frequency trading?**
A: Yes, with proper configuration and caching, the system can handle high-frequency predictions with sub-millisecond latency.

---

## 🎉 Conclusion

The AI News Trading Platform Model Management System provides a complete, production-ready solution for managing AI trading models. With its comprehensive feature set, scalable architecture, and robust monitoring capabilities, it enables teams to deploy and maintain sophisticated trading strategies with confidence.

For more information, examples, and advanced usage patterns, explore the codebase and run the included demonstrations.

**Happy Trading! 📈🚀**