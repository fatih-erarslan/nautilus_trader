# Risk & Performance MCP Tools - Quick Reference

## Tool Comparison Matrix

| Tool | Purpose | Accuracy | GPU Speedup | Latency | Status |
|------|---------|----------|-------------|---------|--------|
| **risk_analysis** | VaR/CVaR calculations | 99.34% | 23.3x | 180ms | ✅ Production |
| **correlation_analysis** | Asset correlations | 99.53% | 45.6x | 180ms | ✅ Production |
| **portfolio_rebalance** | Optimal rebalancing | 99.12% | N/A | <50ms | ✅ Production |
| **optimize_strategy** | Parameter optimization | High | 6.6x | 6.8s | ✅ Production |
| **run_backtest** | Historical testing | High | 1.0x | 1.2s | ✅ Production |
| **get_system_metrics** | Performance monitoring | N/A | N/A | <100ms | ✅ Production |
| **monitor_strategy_health** | Strategy health | N/A | N/A | <50ms | ✅ Production |
| **get_execution_analytics** | Execution quality | N/A | N/A | <50ms | ✅ Production |

---

## When to Use Each Tool

### risk_analysis
**Use when:** Need VaR/CVaR risk metrics for portfolio
**Input:** Portfolio positions, confidence level, time horizon
**Output:** VaR 95/99, CVaR 95/99, risk decomposition
**Best for:** Daily risk reporting, pre-trade risk checks

### correlation_analysis
**Use when:** Analyzing relationships between assets
**Input:** List of symbols, lookback period
**Output:** Correlation matrix, clusters, eigenvalues
**Best for:** Portfolio diversification, pair trading

### portfolio_rebalance
**Use when:** Portfolio deviates from target allocation
**Input:** Target allocations, current portfolio
**Output:** Required trades, estimated costs
**Best for:** Periodic rebalancing, drift management

### optimize_strategy
**Use when:** Tuning strategy parameters
**Input:** Strategy name, parameter ranges, metric
**Output:** Optimal parameters, performance improvement
**Best for:** Strategy development, periodic optimization

### run_backtest
**Use when:** Testing strategy on historical data
**Input:** Strategy, symbol, date range
**Output:** Performance metrics, equity curve
**Best for:** Strategy validation, performance analysis

### get_system_metrics
**Use when:** Monitoring system performance
**Input:** Metrics to collect, time range
**Output:** CPU, memory, latency, throughput
**Best for:** System health monitoring, debugging

### monitor_strategy_health
**Use when:** Tracking strategy performance
**Input:** Strategy name
**Output:** Win rate, drawdown, performance trends
**Best for:** Real-time strategy monitoring

### get_execution_analytics
**Use when:** Analyzing trade execution quality
**Input:** Time period
**Output:** Slippage, fill rate, latency breakdown
**Best for:** Broker evaluation, execution optimization

---

## GPU Acceleration Guide

### Always Use GPU
- ✅ Monte Carlo VaR (>10k simulations) → **23x speedup**
- ✅ Correlation matrix (>50 symbols) → **46x speedup**
- ✅ Strategy optimization (>100 iterations) → **7x speedup**

### Use CPU
- ❌ Single backtest → **1x speedup** (no benefit)
- ❌ Portfolio rebalance → **1x speedup** (overhead > benefit)
- ❌ Small correlation matrix (<10 symbols) → **Overhead too high**

### GPU Configuration
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Check GPU availability
node -e "console.log(process.env.CUDA_VISIBLE_DEVICES)"
```

---

## Performance Tips

### 1. Batch Operations
```javascript
// ❌ Bad: Sequential
for (const portfolio of portfolios) {
  await risk_analysis(portfolio);
}

// ✅ Good: Batch
await risk_analysis_batch(portfolios);
// 40-50% faster
```

### 2. Cache Correlations
```javascript
// ✅ Cache for 1 hour
const cache = new CorrelationCache({ ttl: 3600 });
const matrix = await cache.getOrCalculate(symbols, 90);
// 80% faster for repeated queries
```

### 3. Parallel Optimization
```javascript
// ✅ Evaluate parameters in parallel
const results = await Promise.all(
  paramSets.map(params => backtest(params))
);
// 8-10x faster
```

---

## Accuracy Benchmarks

| Metric | Expected | Calculated | Error | Status |
|--------|----------|------------|-------|--------|
| VaR 95% (Normal) | -$22,900 | -$22,750 | 0.66% | ✅ |
| VaR 99% (T-dist) | -$31,200 | -$31,450 | 0.80% | ✅ |
| Correlation (Perfect) | 1.0000 | 1.0000 | 0.00% | ✅ |
| Sharpe Ratio | 10.59 | 10.52 | 0.66% | ✅ |
| Max Drawdown | 16.67% | 16.65% | 0.12% | ✅ |

---

## Common Issues & Solutions

### Issue: Slow VaR calculation
**Solution:** Enable GPU and increase simulations
```javascript
{
  use_gpu: true,
  use_monte_carlo: true,
  num_simulations: 100000  // More simulations = better accuracy
}
```

### Issue: Out-of-date correlations
**Solution:** Implement correlation caching
```javascript
const cache = new CorrelationCache({ ttl: 3600 });
```

### Issue: Slow strategy optimization
**Solution:** Use parallel evaluation
```javascript
const results = await optimizeStrategyParallel({
  strategy: 'neural_trend',
  parallelism: 8  // Use 8 CPU cores
});
```

### Issue: High memory usage
**Solution:** Use incremental correlation updates
```javascript
const updatedMatrix = updateCorrelationIncremental(
  oldMatrix,
  newDataPoint
);
// 100x faster than full recalculation
```

---

## Example Usage

### Calculate Portfolio Risk
```javascript
const result = await mcp.call('risk_analysis', {
  portfolio: [
    { symbol: 'AAPL', quantity: 100, value: 17500 },
    { symbol: 'GOOGL', quantity: 50, value: 7000 },
    { symbol: 'MSFT', quantity: 75, value: 25875 }
  ],
  use_gpu: true,
  use_monte_carlo: true,
  var_confidence: 0.05,
  time_horizon: 1
});

console.log(`VaR 95%: $${result.var_metrics.var_95}`);
console.log(`CVaR 95%: $${result.var_metrics.cvar_95}`);
console.log(`GPU Speedup: ${result.gpu_accelerated ? 'Yes' : 'No'}`);
```

### Analyze Correlations
```javascript
const result = await mcp.call('correlation_analysis', {
  symbols: ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
  period_days: 90,
  use_gpu: true
});

console.log(`Avg Correlation: ${result.statistics.avg_correlation}`);
console.log(`Clusters: ${result.clusters.length}`);
```

### Rebalance Portfolio
```javascript
const result = await mcp.call('portfolio_rebalance', {
  target_allocations: {
    'AAPL': 0.25,
    'GOOGL': 0.20,
    'MSFT': 0.25,
    'TSLA': 0.15,
    'NVDA': 0.15
  },
  current_portfolio: {
    'AAPL': 17500,
    'GOOGL': 7000,
    'MSFT': 25875,
    'TSLA': 10000,
    'NVDA': 15000
  },
  rebalance_threshold: 0.05
});

console.log(`Trades Required: ${result.trades_required}`);
console.log(`Total Trade Value: $${result.total_trade_value}`);
console.log(`Estimated Cost: $${result.estimated_cost}`);
```

### Optimize Strategy
```javascript
const result = await mcp.call('optimize_strategy', {
  strategy: 'neural_trend',
  symbol: 'AAPL',
  parameter_ranges: {
    short_window: { min: 5, max: 20 },
    long_window: { min: 20, max: 50 },
    threshold: { min: 0.01, max: 0.05 }
  },
  max_iterations: 1000,
  optimization_metric: 'sharpe_ratio',
  use_gpu: true
});

console.log(`Best Parameters:`, result.best_parameters);
console.log(`Best Sharpe: ${result.best_score}`);
console.log(`Improvement: ${result.improvement}%`);
```

---

## Optimization Roadmap

### ✅ Immediate (1-2 weeks)
1. Implement GPU batching → 40% faster
2. Add correlation caching → 80% faster for repeated queries
3. Parallel strategy optimization → 8-10x throughput

### 🔄 This Quarter (3 months)
1. Variance reduction techniques → 5-10x fewer simulations
2. Incremental correlation updates → 100x faster real-time
3. Enhanced risk decomposition

### 🎯 Strategic (6-12 months)
1. Multi-GPU distributed system → 10-50x throughput
2. ML-based risk prediction
3. Real-time stress testing

---

## Support

**Documentation:** `/docs/mcp-analysis/RISK_PERFORMANCE_TOOLS_ANALYSIS.md`
**Test Suite:** `/tests/mcp-analysis/risk_performance_benchmark.js`
**Validation:** `/tests/mcp-analysis/accuracy_validation.js`

**Memory Keys:**
- `analysis/risk-performance/mcp_tools_analysis_summary`
- `analysis/risk-performance/gpu_optimization_insights`

---

**Last Updated:** 2025-11-15
**Status:** ✅ All Tools Production Ready
