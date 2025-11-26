# mcp__ai-news-trader__get_strategy_info

## Description
Get detailed information about a specific trading strategy. This tool provides comprehensive details including configuration parameters, performance history, risk metrics, and implementation specifics for any available trading strategy.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| strategy | string | Yes | N/A | The name/identifier of the trading strategy to retrieve information for |

```typescript
{
  strategy: string  // Required: Strategy identifier (e.g., "momentum_crossover", "neural_swing")
}
```

## Return Value Structure
```json
{
  "strategy_name": "string",
  "full_name": "string",
  "description": "string",
  "version": "string",
  "type": "string",              // "momentum", "mean_reversion", "neural", etc.
  "gpu_supported": boolean,
  "gpu_optimized": boolean,
  "parameters": {
    "default": {
      "param1": "value",
      "param2": "value"
    },
    "ranges": {
      "param1": {"min": 0, "max": 100, "step": 1},
      "param2": {"min": 0.0, "max": 1.0, "step": 0.01}
    },
    "descriptions": {
      "param1": "Description of parameter 1",
      "param2": "Description of parameter 2"
    }
  },
  "performance": {
    "backtest_results": {
      "total_return": number,
      "annualized_return": number,
      "sharpe_ratio": number,
      "sortino_ratio": number,
      "max_drawdown": number,
      "win_rate": number,
      "profit_factor": number,
      "avg_win": number,
      "avg_loss": number,
      "total_trades": number
    },
    "live_results": {         // May be null if no live trading
      "total_return": number,
      "current_position": string,
      "open_trades": number,
      "realized_pnl": number
    },
    "benchmark_comparison": {
      "vs_sp500": number,
      "vs_nasdaq": number,
      "alpha": number,
      "beta": number
    }
  },
  "risk_metrics": {
    "value_at_risk": number,
    "conditional_var": number,
    "downside_deviation": number,
    "max_position_size": number,
    "stop_loss": number,
    "risk_level": "string"    // "low", "medium", "high"
  },
  "requirements": {
    "min_capital": number,
    "data_requirements": ["price", "volume", "sentiment"],
    "update_frequency": "string",  // "tick", "minute", "hour", "daily"
    "computational_requirements": {
      "cpu_cores": number,
      "ram_gb": number,
      "gpu_required": boolean,
      "gpu_memory_gb": number
    }
  },
  "implementation": {
    "language": "string",      // "python", "rust", "cpp"
    "dependencies": ["numpy", "pandas", "torch"],
    "model_type": "string",    // For neural strategies
    "training_required": boolean,
    "last_updated": "string",  // ISO date
    "author": "string",
    "license": "string"
  },
  "status": {
    "current": "string",       // "active", "beta", "deprecated", "maintenance"
    "health": "string",        // "healthy", "degraded", "offline"
    "last_signal": "string",   // ISO date of last trading signal
    "error_rate": number,      // Recent error percentage
    "warnings": ["string"]     // Current warnings if any
  }
}
```

## Examples

### Example 1: Get Basic Strategy Information
```python
# Retrieve information for momentum crossover strategy
strategy_info = mcp__ai_news_trader__get_strategy_info(strategy="momentum_crossover")

print(f"Strategy: {strategy_info['full_name']}")
print(f"Description: {strategy_info['description']}")
print(f"Risk Level: {strategy_info['risk_metrics']['risk_level']}")
print(f"Minimum Capital: ${strategy_info['requirements']['min_capital']:,}")
```

### Example 2: Analyze Strategy Performance
```python
# Deep dive into strategy performance metrics
info = mcp__ai_news_trader__get_strategy_info(strategy="neural_swing")

backtest = info['performance']['backtest_results']
print(f"=== {info['strategy_name']} Performance ===")
print(f"Annual Return: {backtest['annualized_return']*100:.2f}%")
print(f"Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {backtest['max_drawdown']*100:.2f}%")
print(f"Win Rate: {backtest['win_rate']*100:.1f}%")
print(f"Profit Factor: {backtest['profit_factor']:.2f}")

# Compare to benchmark
bench = info['performance']['benchmark_comparison']
print(f"\nVs S&P 500: {bench['vs_sp500']*100:+.2f}%")
print(f"Alpha: {bench['alpha']:.3f}")
print(f"Beta: {bench['beta']:.2f}")
```

### Example 3: Check Strategy Requirements
```python
# Verify system meets strategy requirements
strategy_name = "high_frequency_arbitrage"
info = mcp__ai_news_trader__get_strategy_info(strategy=strategy_name)

requirements = info['requirements']
comp_req = requirements['computational_requirements']

print(f"=== Requirements for {strategy_name} ===")
print(f"Min Capital: ${requirements['min_capital']:,}")
print(f"Data Needs: {', '.join(requirements['data_requirements'])}")
print(f"Update Frequency: {requirements['update_frequency']}")
print(f"\nComputational Requirements:")
print(f"- CPU Cores: {comp_req['cpu_cores']}")
print(f"- RAM: {comp_req['ram_gb']} GB")
print(f"- GPU Required: {comp_req['gpu_required']}")
if comp_req['gpu_required']:
    print(f"- GPU Memory: {comp_req['gpu_memory_gb']} GB")

# Check if system meets requirements
import psutil
if psutil.cpu_count() >= comp_req['cpu_cores']:
    print("✓ CPU requirement met")
else:
    print("✗ Insufficient CPU cores")
```

### Example 4: Parameter Configuration Guide
```python
# Understand strategy parameters for optimization
info = mcp__ai_news_trader__get_strategy_info(strategy="mean_reversion_bands")

params = info['parameters']
print(f"=== {info['strategy_name']} Parameters ===")

for param_name, default_value in params['default'].items():
    param_range = params['ranges'][param_name]
    description = params['descriptions'][param_name]
    
    print(f"\n{param_name}:")
    print(f"  Description: {description}")
    print(f"  Default: {default_value}")
    print(f"  Range: {param_range['min']} - {param_range['max']}")
    print(f"  Step: {param_range['step']}")

# Generate optimization grid
print("\nSuggested optimization grid:")
for param_name, param_range in params['ranges'].items():
    values = []
    current = param_range['min']
    while current <= param_range['max']:
        values.append(current)
        current += param_range['step'] * 5  # Sample every 5 steps
    print(f"{param_name}: {values[:5]}...")  # Show first 5 values
```

### Example 5: Strategy Health Monitoring
```python
# Monitor strategy health and status
import time
from datetime import datetime

def monitor_strategy_health(strategy_name, duration_minutes=5):
    """Monitor strategy health over time"""
    end_time = time.time() + (duration_minutes * 60)
    issues_detected = []
    
    while time.time() < end_time:
        info = mcp__ai_news_trader__get_strategy_info(strategy=strategy_name)
        status = info['status']
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {strategy_name}")
        print(f"Status: {status['current']} | Health: {status['health']}")
        print(f"Error Rate: {status['error_rate']*100:.2f}%")
        
        # Check for warnings
        if status['warnings']:
            print(f"⚠️  Warnings: {', '.join(status['warnings'])}")
            issues_detected.extend(status['warnings'])
        
        # Alert on degraded health
        if status['health'] != 'healthy':
            alert_msg = f"Strategy health degraded: {status['health']}"
            print(f"🚨 {alert_msg}")
            issues_detected.append(alert_msg)
        
        # Check last signal recency
        last_signal = datetime.fromisoformat(status['last_signal'])
        signal_age = (datetime.now() - last_signal).total_seconds() / 3600
        if signal_age > 24:  # No signal in 24 hours
            print(f"⚠️  No signals for {signal_age:.1f} hours")
        
        time.sleep(60)  # Check every minute
    
    return issues_detected

# Usage
issues = monitor_strategy_health("momentum_crossover", duration_minutes=5)
if issues:
    print(f"\nIssues detected during monitoring: {set(issues)}")
```

## Common Use Cases

1. **Strategy Selection**: Detailed comparison before choosing a strategy
2. **Parameter Tuning**: Understanding parameter ranges and defaults
3. **Risk Assessment**: Evaluating risk metrics before deployment
4. **Performance Analysis**: Deep dive into historical performance
5. **System Requirements**: Verifying computational resources
6. **Health Monitoring**: Real-time strategy status tracking

## Error Handling Notes

- **Strategy Not Found**: Invalid strategy name provided
- **Incomplete Data**: Some metrics may be null for new strategies
- **Status Issues**: Strategy may be offline or deprecated
- **Permission Errors**: Some strategies may require special access

### Error Handling Example:
```python
def safe_get_strategy_info(strategy_name):
    """Safely retrieve strategy information with comprehensive error handling"""
    try:
        info = mcp__ai_news_trader__get_strategy_info(strategy=strategy_name)
        
        # Validate critical fields
        required_fields = ['strategy_name', 'parameters', 'performance', 'status']
        for field in required_fields:
            if field not in info:
                print(f"Warning: Missing {field} in strategy info")
        
        # Check strategy status
        if info.get('status', {}).get('current') == 'deprecated':
            print(f"Warning: Strategy '{strategy_name}' is deprecated")
        
        # Validate performance data
        if not info.get('performance', {}).get('backtest_results'):
            print("Warning: No backtest results available")
        
        return info
        
    except ValueError as e:
        print(f"Strategy '{strategy_name}' not found")
        # List available strategies
        strategies = mcp__ai_news_trader__list_strategies()
        print("Available strategies:")
        for s in strategies['strategies'][:5]:
            print(f"  - {s['name']}")
        return None
        
    except Exception as e:
        print(f"Error retrieving strategy info: {e}")
        return None
```

## Performance Tips

1. **Cache Strategy Info**: Strategy details change infrequently
2. **Batch Requests**: Get multiple strategies in parallel
3. **Selective Fields**: Request only needed fields if API supports it
4. **Monitor Changes**: Track version field for strategy updates
5. **Preload Common**: Cache frequently used strategies on startup

### Performance Example:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

class StrategyInfoCache:
    def __init__(self, cache_duration=300):  # 5 minutes
        self.cache = {}
        self.cache_duration = cache_duration
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def get_strategy_info_batch(self, strategy_names):
        """Fetch multiple strategy infos concurrently"""
        tasks = []
        
        for name in strategy_names:
            if name in self.cache and self._is_cache_valid(name):
                tasks.append(asyncio.create_task(
                    self._get_from_cache(name)
                ))
            else:
                tasks.append(asyncio.create_task(
                    self._fetch_and_cache(name)
                ))
        
        return await asyncio.gather(*tasks)
    
    async def _fetch_and_cache(self, strategy_name):
        """Fetch from API and cache result"""
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            self.executor,
            mcp__ai_news_trader__get_strategy_info,
            strategy_name
        )
        
        self.cache[strategy_name] = {
            'data': info,
            'timestamp': time.time()
        }
        return info
    
    def _is_cache_valid(self, strategy_name):
        """Check if cached data is still valid"""
        if strategy_name not in self.cache:
            return False
        
        age = time.time() - self.cache[strategy_name]['timestamp']
        return age < self.cache_duration
```

## Related Tools
- `list_strategies`: Get list of all available strategies
- `simulate_trade`: Test strategy with simulated trades
- `optimize_strategy`: Optimize strategy parameters
- `run_backtest`: Detailed historical testing
- `execute_trade`: Execute trades using the strategy