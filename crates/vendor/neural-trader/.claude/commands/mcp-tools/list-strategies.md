# mcp__ai-news-trader__list_strategies

## Description
List all available trading strategies with GPU capabilities. This tool provides a comprehensive overview of implemented trading strategies, their configurations, performance characteristics, and GPU acceleration support.

## Parameters
This tool accepts no parameters - it returns all available strategies in the system.

```typescript
// No parameters required
{}
```

## Return Value Structure
```json
{
  "strategies": [
    {
      "name": "string",           // Strategy identifier
      "description": "string",    // Strategy description
      "type": "string",          // "momentum", "mean_reversion", "neural", etc.
      "gpu_supported": boolean,   // GPU acceleration available
      "parameters": {            // Default parameters
        "param1": "value",
        "param2": "value"
      },
      "performance_metrics": {   // Historical performance
        "sharpe_ratio": number,
        "max_drawdown": number,
        "win_rate": number,
        "avg_return": number
      },
      "required_data": [        // Data requirements
        "price", "volume", "news_sentiment"
      ],
      "risk_level": "string",   // "low", "medium", "high"
      "min_capital": number,    // Minimum capital required
      "status": "string"        // "active", "beta", "deprecated"
    }
  ],
  "total_count": number,
  "gpu_enabled_count": number,
  "categories": {
    "momentum": number,
    "mean_reversion": number,
    "neural": number,
    "arbitrage": number,
    "sentiment": number
  }
}
```

## Examples

### Example 1: List All Available Strategies
```python
# Get all trading strategies
strategies = mcp__ai_news_trader__list_strategies()
print(f"Total strategies available: {strategies['total_count']}")
print(f"GPU-enabled strategies: {strategies['gpu_enabled_count']}")

for strategy in strategies['strategies']:
    print(f"- {strategy['name']}: {strategy['description']}")
```

### Example 2: Filter GPU-Enabled Strategies
```python
# Find strategies with GPU support
result = mcp__ai_news_trader__list_strategies()
gpu_strategies = [s for s in result['strategies'] if s['gpu_supported']]

print("GPU-Accelerated Strategies:")
for strategy in gpu_strategies:
    print(f"- {strategy['name']}")
    print(f"  Type: {strategy['type']}")
    print(f"  Sharpe Ratio: {strategy['performance_metrics']['sharpe_ratio']}")
```

### Example 3: Strategy Selection by Risk Level
```python
# Select strategies based on risk tolerance
strategies_data = mcp__ai_news_trader__list_strategies()
risk_tolerance = "medium"

suitable_strategies = [
    s for s in strategies_data['strategies'] 
    if s['risk_level'] == risk_tolerance and s['status'] == 'active'
]

print(f"Strategies for {risk_tolerance} risk tolerance:")
for strategy in suitable_strategies:
    print(f"- {strategy['name']}")
    print(f"  Min Capital: ${strategy['min_capital']:,}")
    print(f"  Win Rate: {strategy['performance_metrics']['win_rate']*100:.1f}%")
```

### Example 4: Strategy Category Analysis
```python
# Analyze strategy distribution by category
result = mcp__ai_news_trader__list_strategies()
categories = result['categories']

print("Strategy Distribution:")
for category, count in categories.items():
    percentage = (count / result['total_count']) * 100
    print(f"- {category.capitalize()}: {count} ({percentage:.1f}%)")

# Find best performing category
best_category = None
best_sharpe = -float('inf')

for strategy in result['strategies']:
    sharpe = strategy['performance_metrics']['sharpe_ratio']
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_category = strategy['type']

print(f"\nBest performing category: {best_category} (Sharpe: {best_sharpe:.2f})")
```

### Example 5: Strategy Requirements Check
```python
# Check data requirements for strategies
strategies = mcp__ai_news_trader__list_strategies()
available_data = ["price", "volume", "news_sentiment"]

compatible_strategies = []
for strategy in strategies['strategies']:
    if all(req in available_data for req in strategy['required_data']):
        compatible_strategies.append(strategy)

print(f"Strategies compatible with your data: {len(compatible_strategies)}")
for s in compatible_strategies[:5]:  # Show top 5
    print(f"- {s['name']}: {', '.join(s['required_data'])}")
```

## Common Use Cases

1. **Strategy Discovery**: Browse available trading algorithms
2. **Performance Comparison**: Compare historical metrics across strategies
3. **GPU Planning**: Identify strategies that benefit from GPU acceleration
4. **Risk Assessment**: Filter strategies by risk level
5. **Capital Planning**: Find strategies matching available capital
6. **Data Requirements**: Verify data availability for strategies

## Error Handling Notes

- **Empty Response**: No strategies configured in the system
- **Missing Metrics**: Some strategies may lack performance history
- **Invalid Status**: Deprecated strategies should be filtered out
- **Connection Error**: Ensure MCP server is running (use ping first)

### Error Handling Example:
```python
try:
    strategies = mcp__ai_news_trader__list_strategies()
    
    if not strategies['strategies']:
        print("No strategies available - check server configuration")
        return
    
    # Validate strategy data
    for strategy in strategies['strategies']:
        if 'performance_metrics' not in strategy:
            print(f"Warning: {strategy['name']} missing performance data")
        
        if strategy['status'] == 'deprecated':
            print(f"Note: {strategy['name']} is deprecated")
            
except KeyError as e:
    print(f"Invalid response format: missing {e}")
except Exception as e:
    print(f"Error listing strategies: {e}")
```

## Performance Tips

1. **Cache Results**: Strategy list changes infrequently, cache for 5-10 minutes
2. **Lazy Loading**: Request detailed info only for selected strategies
3. **Parallel Processing**: Use async to fetch strategy details concurrently
4. **Filter Early**: Apply filters at the server level when possible
5. **Batch Operations**: Combine with get_strategy_info for selected strategies

### Performance Example:
```python
import asyncio
from functools import lru_cache
from datetime import datetime, timedelta

class StrategyManager:
    def __init__(self):
        self.cache_ttl = timedelta(minutes=5)
        self.last_fetch = None
        self.cached_strategies = None
    
    @lru_cache(maxsize=1)
    def get_strategies_cached(self):
        """Cache strategy list for 5 minutes"""
        now = datetime.now()
        
        if (self.cached_strategies is None or 
            self.last_fetch is None or
            now - self.last_fetch > self.cache_ttl):
            
            self.cached_strategies = mcp__ai_news_trader__list_strategies()
            self.last_fetch = now
            
        return self.cached_strategies
    
    async def get_strategy_details_batch(self, strategy_names):
        """Fetch details for multiple strategies concurrently"""
        tasks = []
        for name in strategy_names:
            task = asyncio.create_task(
                mcp__ai_news_trader__get_strategy_info(strategy=name)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
```

## Related Tools
- `get_strategy_info`: Get detailed information about a specific strategy
- `simulate_trade`: Test a strategy with simulated trades
- `run_backtest`: Historical performance testing for strategies
- `optimize_strategy`: Optimize strategy parameters
- `run_benchmark`: Compare strategy performance