# Python Supabase Client Examples

This directory contains comprehensive examples demonstrating the capabilities of the Python Supabase client for the Neural Trading Platform.

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)
**Demonstrates**: Core client functionality and basic operations
- Client initialization and connection
- User profile creation
- Trading account setup
- Neural model creation
- Trading bot setup
- Basic database operations

**Run:**
```bash
cd src/python
python examples/basic_usage.py
```

### 2. Neural Model Training (`neural_model_training.py`)
**Demonstrates**: Complete neural model lifecycle
- Advanced model configuration
- Training data preparation
- Training process monitoring
- Model evaluation and performance metrics
- Making predictions
- Model updates and versioning

**Run:**
```bash
cd src/python
python examples/neural_model_training.py
```

### 3. Real-Time Trading (`realtime_trading.py`)
**Demonstrates**: Real-time trading system with WebSocket integration
- Real-time market data streaming
- Event-driven trading signals
- Automated order execution
- Multi-bot orchestration
- Performance monitoring
- Live trading simulation

**Run:**
```bash
cd src/python
python examples/realtime_trading.py
```

### 4. Performance Monitoring (`performance_monitoring.py`)
**Demonstrates**: Comprehensive system monitoring and alerting
- Metrics collection and aggregation
- Performance threshold management
- Real-time alerting system
- System health monitoring
- Custom metrics and dashboards

**Run:**
```bash
cd src/python
python examples/performance_monitoring.py
```

## Setup and Configuration

### Environment Variables

Set the following environment variables before running the examples:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key"
export SUPABASE_SERVICE_KEY="your-service-key"  # Optional, for admin operations
```

### Install Dependencies

```bash
cd src/python
pip install -r requirements.txt
```

### Database Setup

Ensure your Supabase database has the required tables. Refer to the main project documentation for the complete database schema.

## Example Features

### Common Patterns

All examples demonstrate these common patterns:

- **Async/Await Usage**: Proper async programming patterns
- **Error Handling**: Comprehensive error handling and validation
- **Resource Management**: Proper connection lifecycle management
- **Type Safety**: Using Pydantic models for data validation
- **Logging**: Informative console output and progress tracking

### Advanced Features

- **Bulk Operations**: Efficient batch processing for large datasets
- **Real-Time Subscriptions**: WebSocket-based live data streaming
- **Performance Monitoring**: Metrics collection and alerting
- **Concurrent Processing**: Multiple async operations running in parallel
- **Event-Driven Architecture**: Reactive programming patterns

## Running Examples

### Basic Example
```bash
# Simple demonstration of core functionality
python examples/basic_usage.py
```

### Neural Model Training
```bash
# Train and evaluate ML models
python examples/neural_model_training.py
```

### Real-Time Trading
```bash
# Live trading simulation with real-time data
python examples/realtime_trading.py
```

### Performance Monitoring
```bash
# System monitoring and alerting
python examples/performance_monitoring.py
```

## Example Output

### Basic Usage Output
```
🚀 Starting basic usage example...
✅ Connected to Supabase
✅ Created user profile: demo@example.com
✅ Created trading account: Demo Trading Account
✅ Created neural model: LSTM Price Predictor
📊 User has 1 neural models
✅ Created trading bot: Demo Momentum Bot
🤖 User has 1 trading bots
📈 Bot status: stopped
🏥 System health: healthy (response time: 125.4ms)
🎉 Basic usage example completed successfully!
👋 Disconnected from Supabase
```

### Neural Model Training Output
```
🧠 Starting neural model training example...
✅ Connected to Supabase
✅ Created user profile
🧠 Creating neural model...
✅ Created model: LSTM Momentum Predictor (ID: ...)
📊 Preparing training data...
✅ Generated 1000 training samples
🚀 Starting model training...
✅ Training started (ID: ...)
📊 Monitoring training progress...
Status: running
Progress: 45.0%
Epoch: 45/100
Loss: 0.0234
Accuracy: 0.867
---
🎉 Training completed successfully!
🔮 Making sample predictions...
Prediction for AAPL: 0.034
Confidence: 0.82
🎉 Neural model training example completed successfully!
```

### Real-Time Trading Output
```
📡 Starting real-time trading example...
✅ Connected to Supabase
🚀 Starting real-time trading system...
✅ Started bot: Momentum Scalper
✅ Started bot: Mean Reversion Bot
✅ Event handlers configured
✅ Subscribed to real-time data for symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
🎯 Real-time trading system is now active!
📈 AAPL: $152.45 (Volume: 2,345,678)
🔍 Generated buy signal for AAPL (strength: 0.67)
🚨 Trading Signal: buy AAPL (Strength: 0.67)
💼 Order order-123: buy 10 AAPL @ $152.45 - filled
⚠️ Alert (profit): Bot Momentum Scalper P&L: $134.56
```

### Performance Monitoring Output
```
📊 Starting performance monitoring example...
✅ Connected to Supabase
🔧 Setting up performance monitoring...
✅ Set threshold for api_response_time
✅ Set threshold for order_execution_time
✅ Performance monitoring started
📊 Starting metric simulation...
📈 Recorded 42 metrics
🏥 System Health Report:
Overall Status: healthy
Active Alerts: 2
Component Health:
  ✅ api: healthy
  ⚠️ trading_engine: warning
  ✅ ml_engine: healthy
💡 ALERT [INFO] api_response_time: Response time within normal range
⚠️ ALERT [WARNING] order_execution_time: Execution time above warning threshold: 234.5ms
```

## Customization

### Adding Custom Metrics
```python
# In performance_monitoring.py
custom_metric = MetricData(
    name="custom_trading_metric",
    value=your_value,
    metric_type=MetricType.GAUGE,
    tags={"strategy": "custom", "symbol": "AAPL"}
)
await monitor.record_metric(custom_metric)
```

### Custom Event Handlers
```python
# In realtime_trading.py
async def custom_signal_handler(data):
    # Your custom signal processing logic
    print(f"Custom signal: {data}")

realtime.set_trading_signal_handler(custom_signal_handler)
```

### Custom Bot Strategies
```python
# In any trading example
bot_config = {
    "name": "Custom Strategy Bot",
    "strategy": "custom_strategy",
    "risk_params": {
        "max_position_size": 0.03,
        "stop_loss": 0.015,
        "take_profit": 0.045
    },
    "strategy_params": {
        "custom_param_1": "value1",
        "custom_param_2": 42
    }
}
```

## Testing Examples

Run examples in test mode:

```bash
# Set to use test/demo environment
export SUPABASE_URL="https://test-project.supabase.co"
export SUPABASE_ANON_KEY="test-key"

# Run examples
python examples/basic_usage.py
```

## Production Considerations

When adapting these examples for production:

1. **Environment Variables**: Use proper secret management
2. **Error Handling**: Implement comprehensive error recovery
3. **Logging**: Use structured logging with appropriate levels
4. **Monitoring**: Set up proper observability and alerting
5. **Rate Limiting**: Implement appropriate rate limiting
6. **Security**: Follow security best practices for API keys
7. **Performance**: Optimize for your specific use case

## Troubleshooting

### Common Issues

1. **Connection Errors**: Check your Supabase URL and API keys
2. **Permission Errors**: Ensure RLS policies allow your operations
3. **Timeout Errors**: Increase timeout settings for slow operations
4. **Memory Issues**: Use batch processing for large datasets

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your example code here
```

### Test Database Connection

```python
async def test_connection():
    client = NeuralTradingClient(url, key)
    await client.connect()
    health = await client.supabase.health_check()
    print(f"Health: {health}")
    await client.disconnect()

asyncio.run(test_connection())
```

## Contributing

To add new examples:

1. Create a new Python file in the `examples/` directory
2. Follow the existing naming convention
3. Include comprehensive documentation and comments
4. Add error handling and progress reporting
5. Update this README with the new example

## Support

For questions about these examples:
- Check the main documentation in `docs/python-supabase-client.md`
- Review the test files in `tests/` for more usage patterns
- Refer to the main project repository for issues and support