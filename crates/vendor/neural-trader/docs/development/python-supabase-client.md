# Python Supabase Client Documentation

## Overview

The Python Supabase Client provides comprehensive wrappers for integrating Python-based trading systems and neural networks with the Supabase backend. This client offers type-safe, async-first APIs for database operations, real-time data streaming, neural model management, trading bot orchestration, and performance monitoring.

## Key Features

- **Type-Safe Operations**: Pydantic models for data validation and serialization
- **Async/Await Support**: Full asynchronous operations for high-performance applications
- **Real-Time Streaming**: WebSocket-based live data feeds
- **Neural Model Lifecycle**: Complete ML model training and inference workflows
- **Trading Bot Management**: Automated trading system orchestration
- **Performance Monitoring**: Comprehensive metrics collection and alerting
- **E2B Sandbox Integration**: Isolated agent execution environments
- **Error Handling**: Robust error handling with detailed error messages

## Installation

### Core Dependencies (Recommended)
```bash
cd src/python
# Install core dependencies (tested and working)
pip install supabase>=2.3.0 pydantic>=2.5.0 asyncio aiohttp websockets
```

### Full Dependencies (Optional)
```bash
# For full ML/trading functionality
pip install -r requirements.txt
```

### Verification
```bash
# Test installation
python -c "import supabase_client; print('✅ Import successful')"
python -c "from supabase_client import NeuralTradingClient; print('✅ Client available')"
```

**Note**: See `/docs/PYTHON_SETUP_ISSUES.md` for detailed setup information and known issues.

## Quick Start

```python
import asyncio
from supabase_client import NeuralTradingClient

async def main():
    # Initialize client
    client = NeuralTradingClient(
        url="https://your-project.supabase.co",
        key="your-anon-key",
        service_key="your-service-key"  # Optional, for admin operations
    )
    
    # Connect to Supabase
    async with client:
        # Create a neural model
        model_result, error = await client.neural_models.create_model(
            user_id="user-uuid",
            CreateModelRequest(
                name="LSTM Price Predictor",
                model_type="lstm",
                symbols=["AAPL", "GOOGL"],
                configuration={
                    "layers": 3,
                    "units": 128,
                    "dropout": 0.2
                }
            )
        )
        
        if error:
            print(f"Error creating model: {error}")
        else:
            print(f"Model created: {model_result['id']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### 1. Configuration

```python
from supabase_client.config import SupabaseConfig

config = SupabaseConfig(
    url="https://your-project.supabase.co",
    anon_key="your-anon-key",
    service_key="your-service-key",
    auto_refresh_token=True,
    persist_session=True,
    headers={"Custom-Header": "value"}
)
```

### 2. Database Client

```python
from supabase_client.client import AsyncSupabaseClient

async def database_operations():
    client = AsyncSupabaseClient(config)
    await client.connect()
    
    # Select with filters
    users = await client.select(
        "profiles",
        columns="id,email,full_name",
        filter_dict={"is_active": True},
        order_by="created_at",
        limit=10
    )
    
    # Insert data
    new_user = await client.insert(
        "profiles",
        {
            "email": "user@example.com",
            "full_name": "John Doe",
            "is_active": True
        }
    )
    
    # Bulk operations
    bulk_data = [{"name": f"Item {i}"} for i in range(100)]
    results = await client.bulk_insert("items", bulk_data, batch_size=50)
    
    await client.disconnect()
```

### 3. Neural Models Management

```python
from supabase_client.clients.neural_models import (
    NeuralModelsClient,
    CreateModelRequest,
    StartTrainingRequest,
    PredictionRequest
)

async def neural_workflow():
    neural_client = NeuralModelsClient(supabase_client)
    
    # Create model
    model_request = CreateModelRequest(
        name="LSTM Momentum Strategy",
        model_type="lstm",
        symbols=["AAPL", "MSFT", "GOOGL"],
        configuration={
            "sequence_length": 60,
            "layers": [128, 64, 32],
            "dropout": 0.3,
            "learning_rate": 0.001
        }
    )
    
    model, error = await neural_client.create_model(user_id, model_request)
    
    # Start training
    training_request = StartTrainingRequest(
        model_id=model["id"],
        training_data={"source": "historical_prices", "symbols": ["AAPL"]},
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    training_run, error = await neural_client.start_training(training_request)
    
    # Monitor training progress
    while True:
        status, error = await neural_client.get_training_status(training_run["id"])
        if status["status"] in ["completed", "failed"]:
            break
        print(f"Training progress: {status['progress']:.2%}")
        await asyncio.sleep(10)
    
    # Make predictions
    prediction_request = PredictionRequest(
        model_id=model["id"],
        input_data={"features": current_market_data},
        symbols=["AAPL"]
    )
    
    prediction, error = await neural_client.make_prediction(prediction_request)
```

### 4. Trading Bots

```python
from supabase_client.clients.trading_bots import (
    TradingBotsClient,
    CreateBotRequest,
    PlaceOrderRequest,
    OrderSide,
    OrderType
)

async def trading_workflow():
    trading_client = TradingBotsClient(supabase_client)
    
    # Create trading bot
    bot_request = CreateBotRequest(
        name="Neural Momentum Bot",
        strategy="neural_momentum",
        account_id=account_id,
        neural_model_id=model_id,
        symbols=["AAPL", "GOOGL"],
        risk_params={
            "max_position_size": 0.05,  # 5% max position
            "stop_loss": 0.02,          # 2% stop loss
            "take_profit": 0.06,        # 6% take profit
            "max_daily_trades": 10
        },
        strategy_params={
            "signal_threshold": 0.7,
            "momentum_window": 20,
            "volume_filter": True
        }
    )
    
    bot, error = await trading_client.create_bot(user_id, bot_request)
    
    # Start bot
    success, error = await trading_client.start_bot(bot["id"])
    
    # Monitor bot performance
    performance, error = await trading_client.calculate_bot_performance(
        bot["id"],
        time_range_hours=24
    )
    
    print(f"Bot Performance:")
    print(f"  Total P&L: ${performance['total_pnl']:.2f}")
    print(f"  Win Rate: {performance['win_rate']:.1%}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    
    # Place manual order
    order_request = PlaceOrderRequest(
        bot_id=bot["id"],
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10
    )
    
    order, error = await trading_client.place_order(order_request)
```

### 5. Real-Time Data Streaming

```python
from supabase_client.real_time.channel_manager import RealtimeChannelManager

async def realtime_workflow():
    realtime = RealtimeChannelManager(supabase_client)
    
    # Market data handler
    async def handle_market_data(data):
        print(f"Price update: {data['symbol']} = ${data['price']}")
        # Update trading algorithms with new price
    
    # Trading signal handler
    async def handle_trading_signal(data):
        print(f"Trading signal: {data['signal']} for {data['symbol']}")
        # Execute trades based on signal
    
    # Bot status handler
    async def handle_bot_status(data):
        if data['status'] == 'error':
            print(f"Bot {data['bot_id']} encountered error: {data['message']}")
    
    # Subscribe to channels
    await realtime.subscribe_to_market_data(symbols=["AAPL", "GOOGL"])
    await realtime.subscribe_to_trading_signals()
    await realtime.subscribe_to_bot_status()
    
    # Set event handlers
    realtime.set_market_data_handler(handle_market_data)
    realtime.set_trading_signal_handler(handle_trading_signal)
    realtime.set_bot_status_handler(handle_bot_status)
    
    # Keep connection alive
    await realtime.keep_alive()
```

### 6. Performance Monitoring

```python
from supabase_client.monitoring.performance_monitor import (
    PerformanceMonitor,
    MetricData,
    MetricType,
    PerformanceThreshold,
    AlertSeverity
)

async def monitoring_workflow():
    monitor = PerformanceMonitor(supabase_client)
    await monitor.start_monitoring()
    
    # Record custom metrics
    latency_metric = MetricData(
        name="api_response_time",
        value=150.5,
        metric_type=MetricType.TIMER,
        tags={"endpoint": "/orders", "method": "POST"}
    )
    
    await monitor.record_metric(latency_metric)
    
    # Set performance thresholds
    threshold = PerformanceThreshold(
        metric_name="api_response_time",
        warning_threshold=200.0,
        error_threshold=500.0,
        critical_threshold=1000.0,
        operator=">"
    )
    
    await monitor.set_threshold(threshold)
    
    # Get system health
    health, error = await monitor.get_system_health()
    print(f"System Status: {health.overall_status}")
    
    # Calculate aggregates
    aggregates, error = await monitor.calculate_aggregates(
        metric_name="api_response_time",
        aggregation="avg",
        window_minutes=60
    )
    
    await monitor.stop_monitoring()
```

### 7. Sandbox Integration

```python
from supabase_client.clients.sandbox_integration import (
    SandboxIntegrationClient,
    CreateSandboxRequest,
    DeployAgentRequest,
    AgentType
)

async def sandbox_workflow():
    sandbox_client = SandboxIntegrationClient(supabase_client)
    
    # Create sandbox
    sandbox_request = CreateSandboxRequest(
        name="Neural Trading Sandbox",
        template="trading-gpu",
        cpu_count=2,
        memory_mb=2048,
        timeout_minutes=120,
        environment_vars={
            "TRADING_MODE": "paper",
            "LOG_LEVEL": "INFO"
        }
    )
    
    sandbox, error = await sandbox_client.create_sandbox(user_id, sandbox_request)
    
    # Deploy trading agent
    agent_request = DeployAgentRequest(
        agent_type=AgentType.NEURAL_SENTIMENT,
        agent_config={
            "model_path": "/models/sentiment_lstm.pkl",
            "confidence_threshold": 0.8
        },
        symbols=["AAPL", "GOOGL"],
        strategy_params={
            "position_size": 0.02,
            "holding_period": 3600
        }
    )
    
    agent, error = await sandbox_client.deploy_agent(sandbox["id"], agent_request)
    
    # Monitor sandbox
    status, error = await sandbox_client.get_sandbox_status(sandbox["id"])
    print(f"Sandbox Status: {status['status']}")
    print(f"Active Agents: {len(status['active_agents'])}")
```

## Error Handling

The client provides comprehensive error handling with detailed error messages:

```python
async def handle_errors():
    try:
        result, error = await client.neural_models.create_model(user_id, request)
        
        if error:
            # Handle application-level errors
            print(f"Application error: {error}")
            return
        
        # Process successful result
        process_model(result)
        
    except SupabaseError as e:
        # Handle Supabase-specific errors
        print(f"Supabase error: {e}")
        if e.original_error:
            print(f"Original error: {e.original_error}")
    
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
```

## Data Models

The client uses Pydantic models for type safety and validation:

```python
from supabase_client.models.database_models import (
    NeuralModel,
    TradingBot,
    Order,
    Position,
    ModelStatus,
    BotStatus
)

# Type-safe model creation
model = NeuralModel(
    id=uuid4(),
    user_id=user_id,
    name="Price Predictor",
    model_type="lstm",
    status=ModelStatus.TRAINING,
    symbols=["AAPL"],
    configuration={"layers": 3}
)

# Validation happens automatically
try:
    bot = TradingBot(
        user_id=user_id,
        name="",  # This will raise validation error
        strategy="invalid_strategy"  # This will also raise validation error
    )
except ValidationError as e:
    print(f"Validation errors: {e.errors()}")
```

## Utilities

### Async Utilities

```python
from supabase_client.utils.async_utils import (
    async_retry,
    timeout_after,
    gather_with_limit,
    AsyncCache,
    RateLimiter
)

# Retry decorator
@async_retry(max_attempts=3, delay=1.0, backoff=2.0)
async def unreliable_operation():
    # This will be retried up to 3 times
    await api_call()

# Timeout operations
result = await timeout_after(5.0, slow_operation(), default_value)

# Limit concurrency
results = await gather_with_limit(
    *[process_item(item) for item in items],
    limit=5  # Process max 5 items concurrently
)

# Caching
cache = AsyncCache(default_ttl=300)  # 5-minute TTL
await cache.set("key", "value")
value = await cache.get("key")

# Rate limiting
limiter = RateLimiter(rate=10.0)  # 10 requests per second
if await limiter.acquire():
    await api_call()
```

### Validation Utilities

```python
from supabase_client.utils.validation_utils import (
    validate_uuid,
    validate_email,
    validate_symbol,
    validate_create_model_request,
    ValidationError
)

# Validate inputs
if not validate_uuid(user_id):
    raise ValueError("Invalid user ID format")

if not validate_email(email):
    raise ValueError("Invalid email format")

# Validate request data
errors = validate_create_model_request(model_data)
if errors:
    raise ValidationError("Invalid model data", errors)
```

## Testing

The client includes comprehensive test coverage:

```bash
# Run all tests
cd src/python
python -m pytest

# Run specific test file
python -m pytest tests/test_neural_models.py

# Run with coverage
python -m pytest --cov=supabase_client --cov-report=html

# Run async tests only
python -m pytest -m asyncio
```

### Test Fixtures

```python
import pytest
from tests.conftest import (
    mock_supabase_client,
    sample_user_id,
    sample_neural_model,
    configured_mock_client
)

@pytest.mark.asyncio
async def test_neural_model_creation(configured_mock_client, sample_user_id):
    neural_client = NeuralModelsClient(configured_mock_client)
    
    request = CreateModelRequest(
        name="Test Model",
        model_type="lstm",
        symbols=["AAPL"]
    )
    
    result, error = await neural_client.create_model(sample_user_id, request)
    
    assert error is None
    assert result["name"] == "Test Model"
```

## Best Practices

### 1. Connection Management

```python
# Use context managers for automatic cleanup
async with NeuralTradingClient(url, key) as client:
    # Client is automatically connected and disconnected
    await client.neural_models.create_model(...)

# Or manage connections manually
client = NeuralTradingClient(url, key)
try:
    await client.connect()
    # Do work
finally:
    await client.disconnect()
```

### 2. Error Handling

```python
# Always check for errors
result, error = await client.neural_models.create_model(...)
if error:
    logger.error(f"Model creation failed: {error}")
    return

# Use try-catch for exceptions
try:
    await client.connect()
except SupabaseError as e:
    logger.error(f"Connection failed: {e}")
    raise
```

### 3. Performance Optimization

```python
# Use bulk operations for large datasets
data_list = [generate_data(i) for i in range(1000)]
results = await client.supabase.bulk_insert("table", data_list, batch_size=100)

# Use connection pooling for high-throughput applications
config = SupabaseConfig(
    url=url,
    anon_key=key,
    pool_size=10,
    max_overflow=20
)

# Cache frequently accessed data
cache = AsyncCache(default_ttl=300)
cached_data = await cache.get("models")
if not cached_data:
    cached_data = await client.neural_models.list_user_models(user_id)
    await cache.set("models", cached_data)
```

### 4. Monitoring and Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("supabase_client")

# Monitor performance
monitor = PerformanceMonitor(client.supabase)
await monitor.start_monitoring()

# Record custom metrics
await monitor.record_metric(MetricData(
    name="model_prediction_time",
    value=processing_time,
    metric_type=MetricType.TIMER,
    tags={"model_id": model_id}
))
```

## Migration from JavaScript/TypeScript

For users migrating from the JavaScript/TypeScript Supabase client:

### JavaScript/TypeScript
```javascript
const { data, error } = await supabase
  .from('neural_models')
  .select('*')
  .eq('user_id', userId)
  .limit(10);
```

### Python
```python
data = await client.supabase.select(
    'neural_models',
    filter_dict={'user_id': user_id},
    limit=10
)
```

### Real-time Subscriptions

#### JavaScript/TypeScript
```javascript
supabase
  .channel('market-data')
  .on('postgres_changes', { event: 'INSERT', schema: 'public', table: 'market_data' }, 
      payload => console.log(payload))
  .subscribe();
```

#### Python
```python
async def handle_market_data(data):
    print(data)

await realtime.subscribe_to_market_data()
realtime.set_market_data_handler(handle_market_data)
```

## Configuration Options

Complete configuration reference:

```python
config = SupabaseConfig(
    # Required
    url="https://your-project.supabase.co",
    anon_key="your-anon-key",
    
    # Optional
    service_key="your-service-key",
    auto_refresh_token=True,
    persist_session=True,
    detect_session_in_url=False,
    headers={"Custom-Header": "value"},
    schema="public",
    
    # Connection settings
    timeout=30,
    retry_attempts=3,
    retry_delay=1.0,
    
    # Real-time settings
    realtime_url="wss://your-project.supabase.co/realtime/v1/websocket",
    realtime_timeout=10,
    
    # E2B settings
    e2b_api_key="your-e2b-key",
    e2b_template="base",
    
    # Performance settings
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)
```

## API Reference

### Core Classes

- **`NeuralTradingClient`**: Main client class providing access to all functionality
- **`AsyncSupabaseClient`**: Core async database client
- **`NeuralModelsClient`**: Neural model management
- **`TradingBotsClient`**: Trading bot orchestration  
- **`SandboxIntegrationClient`**: E2B sandbox management
- **`RealtimeChannelManager`**: Real-time data streaming
- **`PerformanceMonitor`**: Metrics and monitoring

### Data Models

- **`NeuralModel`**: Neural model entity
- **`TradingBot`**: Trading bot entity
- **`Order`**: Trading order entity
- **`Position`**: Portfolio position entity
- **`PerformanceMetric`**: Performance metric entity

### Request/Response Models

- **`CreateModelRequest`**: Neural model creation request
- **`StartTrainingRequest`**: Training initiation request
- **`CreateBotRequest`**: Trading bot creation request
- **`PlaceOrderRequest`**: Order placement request

## Support and Contributing

For support, issues, and feature requests, please refer to the main project repository. Contributions are welcome through pull requests with comprehensive tests and documentation.

## License

This Python Supabase client is part of the Neural Trading Platform and follows the same licensing terms as the main project.