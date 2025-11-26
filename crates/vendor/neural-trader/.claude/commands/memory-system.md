# Memory System Commands

## Overview
The Claude-Flow memory system provides persistent storage for trading configurations, strategy parameters, market analysis, and results across development sessions.

## Core Memory Commands

### claude-flow memory store
Store trading data and configurations for later retrieval.

**Usage:**
```bash
./claude-flow memory store <key> <data>
```

**Trading-Specific Examples:**
```bash
# Store neural model configurations
./claude-flow memory store "nhits_config" "24h horizon, GPU-optimized, MAE: 0.018, trained on AAPL"

# Store optimized strategy parameters
./claude-flow memory store "momentum_params" "window=20, threshold=0.015, stop_loss=0.05, neural_enhanced=true"

# Store backtest results
./claude-flow memory store "momentum_backtest" "Sharpe: 2.84, Max DD: -12%, Total Return: 33.9%, Period: 2024-Q1"

# Store market analysis
./claude-flow memory store "market_sentiment" "Tech sector bullish, Fed policy supportive, VIX below 20"

# Store risk management settings
./claude-flow memory store "risk_params" "max_position_size=0.05, var_limit=0.02, correlation_threshold=0.7"

# Store trading insights
./claude-flow memory store "trading_insights" "AAPL shows strong momentum after earnings, neural forecast indicates 12% upside"

# Store portfolio allocations
./claude-flow memory store "portfolio_allocation" "60% stocks, 30% bonds, 10% crypto, rebalance monthly"

# Store neural model performance
./claude-flow memory store "model_performance" "Transformer model outperforms LSTM by 15% on tech stocks"
```

### claude-flow memory get
Retrieve specific trading data from memory.

**Usage:**
```bash
./claude-flow memory get <key>
```

**Examples:**
```bash
# Get strategy parameters
./claude-flow memory get "momentum_params"

# Get neural model configuration
./claude-flow memory get "nhits_config"

# Get market analysis
./claude-flow memory get "market_sentiment"
```

### claude-flow memory list
List all stored memory entries with trading data.

**Usage:**
```bash
./claude-flow memory list
```

**Features:**
- Shows 32+ stored entries with metadata
- Displays entry size, update time, and tags
- Organized by namespace and type
- Shows trading-specific memory categories

### claude-flow memory export
Export trading data and configurations to files.

**Usage:**
```bash
./claude-flow memory export <filename>
```

**Examples:**
```bash
# Export all trading configurations
./claude-flow memory export trading_config.json

# Export strategy parameters
./claude-flow memory export strategy_params.json

# Export backtest results
./claude-flow memory export backtest_results.json

# Export neural model configurations
./claude-flow memory export neural_models.json
```

### claude-flow memory import
Import trading configurations from files.

**Usage:**
```bash
./claude-flow memory import <filename>
```

**Examples:**
```bash
# Import previous trading configurations
./claude-flow memory import backup_trading_config.json

# Import optimized parameters
./claude-flow memory import optimized_strategies.json
```

### claude-flow memory stats
Show memory usage statistics for trading data.

**Usage:**
```bash
./claude-flow memory stats
```

### claude-flow memory cleanup
Clean unused memory entries to optimize performance.

**Usage:**
```bash
./claude-flow memory cleanup
```

## Memory Organization for Trading

### Strategy Development Memory
```bash
# Store research findings
./claude-flow memory store "momentum_research" "Momentum works best in trending markets, 20-day window optimal"

# Store architecture decisions
./claude-flow memory store "system_architecture" "Neural forecasting + technical indicators + sentiment analysis"

# Store implementation notes
./claude-flow memory store "implementation_notes" "Use asyncio for real-time data, GPU batch processing for forecasts"

# Store testing results
./claude-flow memory store "test_results" "Unit tests: 98% coverage, Integration tests: all passing"
```

### Trading Performance Memory
```bash
# Store daily trading results
./claude-flow memory store "daily_pnl_2024_06_27" "PnL: +2.3%, Trades: 15, Win Rate: 73%, Best: AAPL +5.2%"

# Store weekly performance
./claude-flow memory store "weekly_performance" "Week ending 2024-06-30: +8.7%, Sharpe: 3.2, Max DD: -1.8%"

# Store monthly analytics
./claude-flow memory store "monthly_report" "June 2024: +22.4%, outperformed S&P by 8.1%, 89% accuracy"

# Store annual performance
./claude-flow memory store "annual_performance" "2024 YTD: +67.3%, Sharpe: 2.84, Calmar: 4.2"
```

### Market Analysis Memory
```bash
# Store sector analysis
./claude-flow memory store "tech_sector_analysis" "FAANG stocks showing divergence, AI sector overbought"

# Store economic indicators
./claude-flow memory store "macro_indicators" "Fed dovish, inflation cooling, GDP growth 2.1%, yield curve normalizing"

# Store volatility analysis
./claude-flow memory store "volatility_analysis" "VIX: 18.5, realized vol below implied, good for momentum strategies"

# Store correlation analysis
./claude-flow memory store "correlation_matrix" "Tech stocks highly correlated (0.85), bonds negative correlation (-0.3)"
```

### Neural Model Memory
```bash
# Store model training results
./claude-flow memory store "nhits_training_log" "200 epochs, final loss: 0.0143, validation MAPE: 1.8%"

# Store model comparisons
./claude-flow memory store "model_comparison" "NHITS > Transformer > LSTM for 24h forecasts, GPU 45x faster"

# Store model deployment info
./claude-flow memory store "prod_model_v1.2" "Deployed 2024-06-27, 94.2% accuracy, 8ms inference time"

# Store feature importance
./claude-flow memory store "feature_importance" "Price: 0.45, Volume: 0.23, Sentiment: 0.18, Technical: 0.14"
```

## Advanced Memory Patterns

### Cross-Agent Coordination
```bash
# Store for agent collaboration
./claude-flow memory store "team_coordination" "Researcher: market bullish, Analyst: momentum strong, Coder: ready to deploy"

# Store workflow states
./claude-flow memory store "workflow_state" "Phase 2 complete: backtesting done, Phase 3: parameter optimization in progress"

# Store shared decisions
./claude-flow memory store "team_decisions" "Agreed: focus on tech stocks, use NHITS model, deploy gradually"
```

### Multi-Stage Development
```bash
# Stage 1: Research
./claude-flow memory store "stage1_research" "Momentum strategies work best in trending markets, neural forecasting improves signals"

# Stage 2: Architecture
./claude-flow memory store "stage2_architecture" "Microservices design: data ingestion, neural engine, strategy engine, risk manager"

# Stage 3: Implementation
./claude-flow memory store "stage3_implementation" "Core modules complete, GPU acceleration working, 95% test coverage"

# Stage 4: Testing
./claude-flow memory store "stage4_testing" "Backtests complete, live simulation successful, ready for production"
```

### Integration with SPARC Modes
```bash
# Memory for coder mode
./claude-flow sparc run coder "Implement momentum strategy using momentum_params from memory"

# Memory for analyzer mode
./claude-flow sparc run analyzer "Analyze performance using backtest_results from memory"

# Memory for optimizer mode
./claude-flow sparc run optimizer "Optimize strategy using risk_params from memory as constraints"

# Memory for tester mode
./claude-flow sparc run tester "Test strategy using test_scenarios from memory"
```

## Memory Backup and Recovery

### Automatic Backups
The memory system automatically creates backups:
- Location: `/workspaces/ai-news-trader/memory/backups/`
- Format: `backup-YYYY-MM-DDTHH-MM-SS-sssZ.json`
- Created on: memory shutdown, export operations

### Manual Backup
```bash
# Create manual backup
./claude-flow memory export complete_backup_$(date +%Y%m%d_%H%M%S).json

# Restore from backup
./claude-flow memory import complete_backup_20240627_120000.json
```

## Memory Search and Organization

### Memory Namespaces
- `default`: General trading data
- `swarm-*`: Swarm operation results
- `neural`: Neural model configurations
- `strategies`: Trading strategy parameters
- `backtests`: Historical testing results
- `analysis`: Market and performance analysis

### Memory Tags
Common tags for trading data:
- `trading`, `neural`, `optimization`
- `momentum`, `mean_reversion`, `swing`
- `backtest`, `performance`, `risk`
- `research`, `analysis`, `deployment`

### Finding Memory Entries
```bash
# List entries by pattern
./claude-flow memory list | grep "momentum"

# List recent entries
./claude-flow memory list | head -10

# List by size
./claude-flow memory list | sort -k4 -h
```

## Best Practices

### Memory Organization
1. Use descriptive keys with context: `"momentum_params_optimized_2024Q2"`
2. Include timestamps for time-sensitive data
3. Tag entries with relevant categories
4. Store metadata with configurations
5. Regular cleanup of outdated entries

### Trading-Specific Practices
1. Store strategy parameters before optimization
2. Save backtest results for comparison
3. Keep neural model training logs
4. Store market analysis for reference
5. Backup before major changes

### Performance Optimization
1. Use compression for large datasets
2. Regular memory cleanup
3. Export old data for archival
4. Monitor memory usage statistics
5. Use batch operations when possible

### Security Considerations
1. Don't store API keys or secrets
2. Encrypt sensitive trading data
3. Regular backup rotation
4. Access control for production data
5. Audit memory access patterns