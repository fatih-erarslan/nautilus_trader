# Correlation Analysis MCP Tool

## Overview
The `mcp__ai-news-trader__correlation_analysis` tool performs advanced correlation analysis between multiple assets using GPU acceleration. It identifies relationships, diversification opportunities, and hidden dependencies to optimize portfolio construction and risk management.

## Tool Specifications

### Tool Name
`mcp__ai-news-trader__correlation_analysis`

### Purpose
- Calculate correlation matrices between multiple assets
- Identify lead-lag relationships and causality
- Detect regime changes in correlations
- Find diversification opportunities
- Analyze rolling correlations and stability
- Perform principal component analysis (PCA)
- GPU-accelerated for large asset universes

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbols` | array | List of stock symbols or assets to analyze correlations |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period_days` | integer | 90 | Historical period for correlation calculation |
| `use_gpu` | boolean | true | Enable GPU acceleration for faster computation |

## Return Value Structure

```json
{
  "correlation_summary": {
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "period": {
      "start_date": "2024-09-01",
      "end_date": "2024-12-01",
      "trading_days": 63,
      "data_points": 315
    },
    "quality_score": 0.98
  },
  "correlation_matrix": [
    [1.00, 0.82, 0.75, 0.78, 0.71],
    [0.82, 1.00, 0.79, 0.81, 0.77],
    [0.75, 0.79, 1.00, 0.77, 0.83],
    [0.78, 0.81, 0.77, 1.00, 0.74],
    [0.71, 0.77, 0.83, 0.74, 1.00]
  ],
  "statistical_significance": {
    "p_values": [[...]],
    "significant_correlations": 20,
    "confidence_level": 0.95
  },
  "correlation_metrics": {
    "average_correlation": 0.78,
    "max_correlation": {
      "value": 0.83,
      "pair": ["GOOGL", "META"]
    },
    "min_correlation": {
      "value": 0.71,
      "pair": ["AAPL", "META"]
    },
    "correlation_range": 0.12,
    "effective_assets": 2.34
  },
  "rolling_analysis": {
    "30_day": {
      "mean": 0.76,
      "std": 0.08,
      "trend": "increasing"
    },
    "60_day": {
      "mean": 0.78,
      "std": 0.06,
      "trend": "stable"
    },
    "90_day": {
      "mean": 0.78,
      "std": 0.05,
      "trend": "stable"
    }
  },
  "principal_components": {
    "explained_variance": [0.823, 0.087, 0.054, 0.023, 0.013],
    "cumulative_variance": [0.823, 0.910, 0.964, 0.987, 1.000],
    "loadings": {
      "PC1": [0.447, 0.456, 0.443, 0.448, 0.435],
      "PC2": [-0.632, -0.316, 0.316, 0.632, 0.000]
    },
    "interpretation": {
      "PC1": "Market factor (82.3% variance)",
      "PC2": "Value vs Growth factor (8.7% variance)"
    }
  },
  "clustering_analysis": {
    "clusters": [
      {
        "id": 1,
        "members": ["AAPL", "MSFT"],
        "average_internal_correlation": 0.82
      },
      {
        "id": 2,
        "members": ["GOOGL", "META"],
        "average_internal_correlation": 0.83
      },
      {
        "id": 3,
        "members": ["AMZN"],
        "average_internal_correlation": 1.00
      }
    ],
    "optimal_clusters": 3,
    "silhouette_score": 0.67
  },
  "lead_lag_relationships": {
    "significant_relationships": [
      {
        "leader": "AAPL",
        "follower": "MSFT",
        "lag_days": 1,
        "correlation": 0.34,
        "p_value": 0.012
      },
      {
        "leader": "GOOGL",
        "follower": "META",
        "lag_days": 0,
        "correlation": 0.83,
        "p_value": 0.001
      }
    ]
  },
  "regime_analysis": {
    "current_regime": "high_correlation",
    "regime_history": [
      {
        "regime": "normal",
        "start": "2024-09-01",
        "end": "2024-10-15",
        "avg_correlation": 0.65
      },
      {
        "regime": "high_correlation",
        "start": "2024-10-16",
        "end": "2024-12-01",
        "avg_correlation": 0.85
      }
    ],
    "regime_change_probability": 0.23
  },
  "diversification_analysis": {
    "diversification_ratio": 1.23,
    "effective_number_of_assets": 2.34,
    "concentration_risk": 0.43,
    "recommended_additions": [
      {"symbol": "TLT", "expected_correlation": 0.12},
      {"symbol": "GLD", "expected_correlation": 0.08},
      {"symbol": "VIX", "expected_correlation": -0.45}
    ]
  },
  "network_analysis": {
    "centrality_measures": {
      "AAPL": 0.89,
      "MSFT": 0.87,
      "GOOGL": 0.85,
      "AMZN": 0.83,
      "META": 0.81
    },
    "community_detection": {
      "communities": 2,
      "modularity": 0.34
    }
  },
  "execution_time": {
    "total_ms": 127,
    "gpu_speedup": "15.3x",
    "calculations_per_second": 24750
  }
}
```

## Advanced Usage Examples

### Basic Correlation Analysis
```python
# Simple correlation between tech stocks
result = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
    }
)

print(f"Average correlation: {result['correlation_metrics']['average_correlation']:.3f}")
```

### Portfolio Diversification Analysis
```python
# Analyze correlations for portfolio construction
# Start with core holdings
core_holdings = ["SPY", "QQQ", "IWM"]

# Test potential additions
test_assets = ["TLT", "GLD", "REIT", "UUP", "VIX", "BTC-USD"]
all_symbols = core_holdings + test_assets

correlation_result = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": all_symbols,
        "period_days": 180,
        "use_gpu": true
    }
)

# Find best diversifiers
correlation_matrix = correlation_result['correlation_matrix']
best_diversifiers = []

for i, test_asset in enumerate(test_assets):
    test_idx = len(core_holdings) + i
    # Average correlation with core holdings
    avg_correlation = np.mean([correlation_matrix[test_idx][j] for j in range(len(core_holdings))])
    best_diversifiers.append((test_asset, avg_correlation))

# Sort by lowest correlation
best_diversifiers.sort(key=lambda x: x[1])
print("Best diversifiers:")
for asset, corr in best_diversifiers[:3]:
    print(f"  {asset}: {corr:.3f}")
```

### Sector Correlation Analysis
```python
# Analyze correlations within and across sectors
sectors = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "finance": ["JPM", "BAC", "GS", "MS", "WFC"],
    "healthcare": ["JNJ", "PFE", "UNH", "CVS", "ABT"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"]
}

# Flatten all symbols
all_symbols = [symbol for symbols in sectors.values() for symbol in symbols]

correlation_result = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": all_symbols,
        "period_days": 90,
        "use_gpu": true
    }
)

# Calculate sector correlations
sector_correlations = {}
for sector1, symbols1 in sectors.items():
    for sector2, symbols2 in sectors.items():
        if sector1 <= sector2:  # Avoid duplicates
            correlations = []
            for sym1 in symbols1:
                for sym2 in symbols2:
                    if sym1 != sym2:
                        idx1 = all_symbols.index(sym1)
                        idx2 = all_symbols.index(sym2)
                        correlations.append(correlation_result['correlation_matrix'][idx1][idx2])
            
            avg_correlation = np.mean(correlations) if correlations else 1.0
            sector_correlations[f"{sector1}-{sector2}"] = avg_correlation

print("Sector Correlations:")
for pair, corr in sorted(sector_correlations.items(), key=lambda x: x[1]):
    print(f"  {pair}: {corr:.3f}")
```

### Dynamic Correlation Monitoring
```python
# Monitor changing correlations over time
async def monitor_correlation_changes(symbols, baseline_period=90, check_period=30):
    # Get baseline correlations
    baseline = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": baseline_period
        }
    )
    
    baseline_matrix = np.array(baseline['correlation_matrix'])
    
    # Check recent correlations
    recent = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": check_period
        }
    )
    
    recent_matrix = np.array(recent['correlation_matrix'])
    
    # Find significant changes
    changes = []
    threshold = 0.2  # 20% change threshold
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            baseline_corr = baseline_matrix[i][j]
            recent_corr = recent_matrix[i][j]
            change = recent_corr - baseline_corr
            
            if abs(change) > threshold:
                changes.append({
                    "pair": [symbols[i], symbols[j]],
                    "baseline": baseline_corr,
                    "recent": recent_corr,
                    "change": change
                })
    
    return {
        "significant_changes": changes,
        "regime_shift": len(changes) > len(symbols) / 2,
        "avg_correlation_change": np.mean(recent_matrix) - np.mean(baseline_matrix)
    }

# Example usage
symbols = ["SPY", "TLT", "GLD", "DXY", "VIX"]
changes = await monitor_correlation_changes(symbols)

if changes["regime_shift"]:
    print("WARNING: Correlation regime shift detected!")
    for change in changes["significant_changes"]:
        print(f"  {change['pair'][0]}-{change['pair'][1]}: {change['baseline']:.2f} → {change['recent']:.2f} ({change['change']:+.2f})")
```

## Integration with Other Tools

### 1. Correlation-Based Portfolio Optimization
```python
# Use correlations to optimize portfolio
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TLT", "GLD"]

# Get correlation matrix
correlation = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": symbols,
        "period_days": 180
    }
)

# Use correlations in optimization
optimization = await mcp.call_tool(
    "mcp__ai-news-trader__optimize_strategy",
    {
        "strategy": "minimum_variance",
        "symbol": "PORTFOLIO",
        "parameter_ranges": {
            f"{sym}_weight": [0.05, 0.40] for sym in symbols
        },
        "optimization_metric": "sharpe_ratio"
        # Correlation matrix used internally for optimization
    }
)

# Verify portfolio correlations
final_portfolio = [
    {"symbol": sym, "weight": optimization["best_parameters"][f"{sym}_weight"]}
    for sym in symbols
]

risk_analysis = await mcp.call_tool(
    "mcp__ai-news-trader__risk_analysis",
    {
        "portfolio": final_portfolio
    }
)
```

### 2. Pair Trading Strategy Development
```python
# Find cointegrated pairs for pair trading
symbols = ["XLF", "BAC", "JPM", "WFC", "C", "GS", "MS"]

# Analyze correlations
correlation = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": symbols,
        "period_days": 252  # 1 year for stability
    }
)

# Find highly correlated pairs
high_correlation_pairs = []
correlation_threshold = 0.85

matrix = correlation['correlation_matrix']
for i in range(len(symbols)):
    for j in range(i+1, len(symbols)):
        if matrix[i][j] > correlation_threshold:
            high_correlation_pairs.append({
                "pair": [symbols[i], symbols[j]],
                "correlation": matrix[i][j],
                "lead_lag": next(
                    (rel for rel in correlation['lead_lag_relationships']['significant_relationships']
                     if rel['leader'] in [symbols[i], symbols[j]] and rel['follower'] in [symbols[i], symbols[j]]),
                    None
                )
            })

# Test pair trading strategies
for pair_info in high_correlation_pairs[:3]:  # Top 3 pairs
    pair = pair_info['pair']
    print(f"\nTesting pair: {pair[0]}-{pair[1]} (correlation: {pair_info['correlation']:.3f})")
    
    backtest = await mcp.call_tool(
        "mcp__ai-news-trader__run_backtest",
        {
            "strategy": "pairs_trading",
            "symbol": f"{pair[0]}-{pair[1]}",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
    )
    
    print(f"  Sharpe Ratio: {backtest['performance']['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {backtest['performance']['win_rate']:.1%}")
```

### 3. Risk Factor Analysis
```python
# Analyze factor exposures using PCA
factor_proxies = {
    "market": "SPY",
    "size": "IWM",
    "value": "IWD",
    "growth": "IWF",
    "momentum": "MTUM",
    "quality": "QUAL",
    "low_vol": "USMV",
    "dividend": "DVY"
}

# Add portfolio holdings
portfolio_symbols = ["AAPL", "MSFT", "JPM", "JNJ", "XOM"]
all_symbols = list(factor_proxies.values()) + portfolio_symbols

# Run correlation analysis
correlation = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": all_symbols,
        "period_days": 252,
        "use_gpu": true
    }
)

# Extract factor loadings from PCA
pca = correlation['principal_components']
print("Factor Interpretations:")
for i, (variance, interpretation) in enumerate(zip(pca['explained_variance'][:3], 
                                                   pca['interpretation'].values())):
    print(f"  PC{i+1}: {interpretation} ({variance:.1%} of variance)")

# Calculate portfolio factor exposures
portfolio_loadings = {}
for symbol in portfolio_symbols:
    idx = all_symbols.index(symbol)
    portfolio_loadings[symbol] = {
        f"PC{i+1}": pca['loadings'][f"PC{i+1}"][idx]
        for i in range(3)
    }

print("\nPortfolio Factor Exposures:")
for symbol, loadings in portfolio_loadings.items():
    print(f"  {symbol}: {loadings}")
```

## Performance Optimization Tips

### 1. GPU Acceleration Best Practices
```python
# Optimize GPU usage for large correlation matrices
large_universe = ["SPY", "QQQ", "IWM"] + [f"STOCK_{i}" for i in range(100)]

# Pre-warm GPU cache
warmup = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": large_universe[:10],
        "period_days": 30,
        "use_gpu": true
    }
)

# Run full analysis with warmed GPU
result = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": large_universe,
        "period_days": 90,
        "use_gpu": true
    }
)

print(f"Processed {len(large_universe)} symbols in {result['execution_time']['total_ms']}ms")
print(f"GPU speedup: {result['execution_time']['gpu_speedup']}")
```

### 2. Incremental Correlation Updates
```python
# Efficiently update correlations for new data
async def update_correlations_incrementally(symbols, base_correlations, new_data_days=1):
    # Get recent correlations with overlap
    recent = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": 30,  # Shorter period for recent changes
            "use_gpu": true
        }
    )
    
    # Weight old and new correlations
    decay_factor = 0.98  # Daily decay
    weight_new = 1 - (decay_factor ** new_data_days)
    weight_old = decay_factor ** new_data_days
    
    # Update correlation matrix
    old_matrix = np.array(base_correlations['correlation_matrix'])
    new_matrix = np.array(recent['correlation_matrix'])
    updated_matrix = weight_old * old_matrix + weight_new * new_matrix
    
    return {
        "updated_correlations": updated_matrix.tolist(),
        "change_magnitude": np.mean(np.abs(new_matrix - old_matrix))
    }
```

### 3. Sparse Correlation Computation
```python
# Compute only necessary correlations for efficiency
async def compute_sparse_correlations(focus_symbols, universe_symbols, min_correlation=0.3):
    # First pass: find relevant symbols
    initial_analysis = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": focus_symbols + universe_symbols[:50],  # Sample
            "period_days": 30,  # Quick scan
            "use_gpu": true
        }
    )
    
    # Identify highly correlated symbols
    relevant_symbols = set(focus_symbols)
    correlation_matrix = initial_analysis['correlation_matrix']
    
    for i, symbol in enumerate(focus_symbols):
        for j, other in enumerate(universe_symbols[:50]):
            if abs(correlation_matrix[i][len(focus_symbols) + j]) > min_correlation:
                relevant_symbols.add(other)
    
    # Full analysis on relevant symbols only
    final_analysis = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": list(relevant_symbols),
            "period_days": 90,
            "use_gpu": true
        }
    )
    
    return final_analysis
```

## Risk Management Best Practices

### 1. Correlation Breakdown Detection
```python
# Detect when correlations break down
async def detect_correlation_breakdown(symbols, lookback_days=252, threshold=2.0):
    # Get long-term correlations
    long_term = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": lookback_days
        }
    )
    
    # Get recent correlations
    recent = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": 20  # Last month
        }
    )
    
    # Calculate z-scores for correlation changes
    breakdowns = []
    lt_matrix = np.array(long_term['correlation_matrix'])
    r_matrix = np.array(recent['correlation_matrix'])
    
    # Historical correlation stability
    rolling = long_term['rolling_analysis']['30_day']
    correlation_std = rolling['std']
    
    for i in range(len(symbols)):
        for j in range(i+1, len(symbols)):
            lt_corr = lt_matrix[i][j]
            r_corr = r_matrix[i][j]
            z_score = (r_corr - lt_corr) / (correlation_std + 0.001)
            
            if abs(z_score) > threshold:
                breakdowns.append({
                    "pair": [symbols[i], symbols[j]],
                    "long_term_correlation": lt_corr,
                    "recent_correlation": r_corr,
                    "z_score": z_score,
                    "breakdown_type": "decorrelation" if z_score < 0 else "correlation_spike"
                })
    
    return {
        "breakdowns_detected": len(breakdowns) > 0,
        "breakdown_pairs": breakdowns,
        "market_stress_indicator": len(breakdowns) / (len(symbols) * (len(symbols)-1) / 2)
    }

# Monitor for breakdowns
result = await detect_correlation_breakdown(["SPY", "TLT", "GLD", "DXY", "VIX"])
if result["breakdowns_detected"]:
    print("WARNING: Correlation breakdowns detected!")
    for breakdown in result["breakdown_pairs"]:
        print(f"  {breakdown['pair']}: {breakdown['breakdown_type']} (z-score: {breakdown['z_score']:.2f})")
```

### 2. Concentration Risk Analysis
```python
# Analyze concentration risk from correlations
async def analyze_concentration_risk(portfolio):
    symbols = [holding["symbol"] for holding in portfolio]
    weights = [holding["weight"] for holding in portfolio]
    
    # Get correlations
    correlation = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": 90
        }
    )
    
    # Calculate effective number of assets
    correlation_matrix = np.array(correlation['correlation_matrix'])
    weights_array = np.array(weights)
    
    # Portfolio variance considering correlations
    portfolio_variance = np.dot(weights_array, np.dot(correlation_matrix, weights_array))
    
    # Equal weight portfolio variance
    equal_weights = np.ones(len(symbols)) / len(symbols)
    equal_variance = np.dot(equal_weights, np.dot(correlation_matrix, equal_weights))
    
    # Effective number of assets
    effective_n = equal_variance / portfolio_variance * len(symbols)
    
    # Concentration metrics
    herfindahl_index = np.sum(weights_array ** 2)
    
    # Principal component concentration
    pca = correlation['principal_components']
    pc1_variance = pca['explained_variance'][0]
    
    return {
        "effective_assets": effective_n,
        "concentration_ratio": 1 / effective_n,
        "herfindahl_index": herfindahl_index,
        "systematic_risk_concentration": pc1_variance,
        "diversification_ratio": correlation['diversification_analysis']['diversification_ratio'],
        "recommendations": generate_diversification_recommendations(effective_n, pc1_variance)
    }

def generate_diversification_recommendations(effective_n, pc1_variance):
    recommendations = []
    
    if effective_n < 3:
        recommendations.append("Portfolio highly concentrated - add uncorrelated assets")
    
    if pc1_variance > 0.8:
        recommendations.append("High systematic risk - consider alternative asset classes")
    
    return recommendations
```

### 3. Regime-Aware Correlation Adjustments
```python
# Adjust strategies based on correlation regimes
async def get_regime_adjusted_weights(symbols, base_weights):
    # Analyze current correlation regime
    correlation = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": 60
        }
    )
    
    regime = correlation['regime_analysis']['current_regime']
    avg_correlation = correlation['correlation_metrics']['average_correlation']
    
    # Adjust weights based on regime
    adjusted_weights = base_weights.copy()
    
    if regime == "high_correlation":
        # Reduce risk in high correlation environment
        print(f"High correlation regime detected (avg: {avg_correlation:.2f})")
        
        # Increase cash/defensive allocation
        risk_reduction = 0.2
        for i in range(len(adjusted_weights)):
            adjusted_weights[i] *= (1 - risk_reduction)
        
        # Add defensive allocation
        adjusted_weights.append(risk_reduction)  # Cash position
        symbols.append("CASH")
        
    elif regime == "decorrelated":
        # Increase risk in decorrelated environment
        print(f"Decorrelated regime detected (avg: {avg_correlation:.2f})")
        
        # Can increase concentration in best opportunities
        # Weights remain unchanged or slightly increased
        
    return {
        "regime": regime,
        "original_weights": base_weights,
        "adjusted_weights": adjusted_weights,
        "symbols": symbols
    }
```

## Common Issues and Solutions

### Issue: Non-Positive Definite Correlation Matrix
**Solution**: Use correlation matrix repair techniques
```python
# Handle correlation matrix issues
try:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": large_symbol_list,
            "period_days": 30  # Short period may cause issues
        }
    )
except Exception as e:
    if "positive definite" in str(e).lower():
        # Retry with longer period
        result = await mcp.call_tool(
            "mcp__ai-news-trader__correlation_analysis",
            {
                "symbols": large_symbol_list,
                "period_days": 90  # Longer period for stability
            }
        )
```

### Issue: Spurious Correlations in Small Samples
**Solution**: Use statistical significance testing
```python
# Filter for significant correlations only
result = await mcp.call_tool(
    "mcp__ai-news-trader__correlation_analysis",
    {
        "symbols": symbols,
        "period_days": 60
    }
)

# Use p-values to filter spurious correlations
significant_pairs = []
p_values = result['statistical_significance']['p_values']
correlation_matrix = result['correlation_matrix']

for i in range(len(symbols)):
    for j in range(i+1, len(symbols)):
        if p_values[i][j] < 0.05:  # 95% confidence
            significant_pairs.append({
                "pair": [symbols[i], symbols[j]],
                "correlation": correlation_matrix[i][j],
                "p_value": p_values[i][j]
            })

print(f"Significant correlations: {len(significant_pairs)} out of {len(symbols)*(len(symbols)-1)/2}")
```

### Issue: Unstable Correlations Over Time
**Solution**: Use exponentially weighted correlations
```python
# Analyze correlation stability
symbols = ["SPY", "TLT", "GLD"]
time_windows = [30, 60, 90, 180, 365]
correlation_evolution = {}

for window in time_windows:
    result = await mcp.call_tool(
        "mcp__ai-news-trader__correlation_analysis",
        {
            "symbols": symbols,
            "period_days": window
        }
    )
    correlation_evolution[window] = result['correlation_metrics']['average_correlation']

# Check stability
correlation_std = np.std(list(correlation_evolution.values()))
if correlation_std > 0.15:
    print(f"WARNING: Unstable correlations (std: {correlation_std:.3f})")
    print("Consider using adaptive correlation estimates")
```

## See Also
- [Risk Analysis Tool](risk-analysis.md) - Portfolio risk with correlations
- [Optimize Strategy Tool](optimize-strategy.md) - Correlation-aware optimization
- [Performance Report Tool](performance-report.md) - Performance attribution
- [Run Backtest Tool](run-backtest.md) - Test correlation-based strategies