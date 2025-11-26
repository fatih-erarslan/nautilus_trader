# Neural Trading Workflows - MCP Tool Integration Guide

This guide provides complete, production-ready workflows for AI-powered trading using the MCP tools. Each workflow shows exact tool sequences, data flow, and error handling.

## Table of Contents
1. [Complete Neural Trading Pipeline](#1-complete-neural-trading-pipeline)
2. [Real-time Trading Decision Workflow](#2-real-time-trading-decision-workflow)
3. [Portfolio Optimization Workflow](#3-portfolio-optimization-workflow)
4. [Strategy Development Workflow](#4-strategy-development-workflow)
5. [Risk Management Workflow](#5-risk-management-workflow)

---

## 1. Complete Neural Trading Pipeline

**Research → Train → Backtest → Deploy**

This workflow takes you from initial research to production deployment of a neural trading strategy.

### Step 1: Research Phase
```bash
# Analyze historical market behavior for target symbols
mcp__ai-news-trader__analyze_news(
  symbol="AAPL",
  lookback_hours=168,  # 1 week
  sentiment_model="enhanced",
  use_gpu=true
)

# Store research findings
./claude-flow memory store "aapl_research_2025" "Strong positive sentiment correlation with earnings reports, neural forecasting shows 72% directional accuracy"

# Analyze correlations for portfolio context
mcp__ai-news-trader__correlation_analysis(
  symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
  period_days=90,
  use_gpu=true
)

# Store correlation matrix
./claude-flow memory store "tech_correlations_q1_2025" "AAPL-MSFT: 0.82, AAPL-GOOGL: 0.75, diversification needed"
```

### Step 2: Neural Model Training
```bash
# Prepare training data (assume data collection completed)
./claude-flow-neural prepare-data AAPL --start "2020-01-01" --end "2024-12-31" --features price,volume,sentiment

# Train NHITS model for AAPL
mcp__ai-news-trader__neural_train(
  data_path="/data/aapl_training_2020_2024.csv",
  model_type="nhits",
  epochs=200,
  batch_size=64,
  learning_rate=0.001,
  validation_split=0.2,
  use_gpu=true
)

# Store model ID and parameters
./claude-flow memory store "nhits_aapl_model_v1" "model_id: nhits_aapl_20250627_001, val_loss: 0.0023, horizon: 24h"

# Evaluate model performance
mcp__ai-news-trader__neural_evaluate(
  model_id="nhits_aapl_20250627_001",
  test_data="/data/aapl_test_2025.csv",
  metrics=["mae", "rmse", "mape", "r2_score", "directional_accuracy"],
  use_gpu=true
)
```

### Step 3: Comprehensive Backtesting
```bash
# Run neural-enhanced backtest
mcp__ai-news-trader__neural_backtest(
  model_id="nhits_aapl_20250627_001",
  start_date="2024-01-01",
  end_date="2024-12-31",
  benchmark="SPY",
  rebalance_frequency="daily",
  use_gpu=true
)

# Run traditional strategy backtest for comparison
mcp__ai-news-trader__run_backtest(
  strategy="momentum_neural_enhanced",
  symbol="AAPL",
  start_date="2024-01-01",
  end_date="2024-12-31",
  benchmark="SPY",
  include_costs=true,
  use_gpu=true
)

# Store backtest results
./claude-flow memory store "backtest_results_aapl_2024" "Neural: Sharpe 2.3, Max DD -12%, Win rate 58% | Traditional: Sharpe 1.7, Max DD -18%, Win rate 52%"
```

### Step 4: Strategy Optimization
```bash
# Optimize neural strategy parameters
mcp__ai-news-trader__neural_optimize(
  model_id="nhits_aapl_20250627_001",
  parameter_ranges={
    "confidence_threshold": [0.6, 0.9],
    "position_size": [0.1, 0.3],
    "stop_loss": [0.02, 0.05],
    "take_profit": [0.03, 0.08]
  },
  trials=200,
  optimization_metric="sharpe_ratio",
  use_gpu=true
)

# Store optimal parameters
./claude-flow memory store "optimal_params_aapl_neural" "confidence: 0.75, position: 0.2, stop_loss: 0.03, take_profit: 0.05"
```

### Step 5: Production Deployment
```bash
# Deploy model with monitoring
./claude-flow-neural neural deploy nhits_aapl_20250627_001 --env production --health-check --alert-threshold 0.1

# Start live monitoring
./claude-flow monitor --dashboard --metrics neural_performance,trade_execution,risk_metrics

# Error handling and failover
if [ $? -ne 0 ]; then
  echo "Deployment failed, rolling back..."
  ./claude-flow-neural neural rollback nhits_aapl_20250627_001
  ./claude-flow memory store "deployment_failure_$(date +%s)" "Deployment failed: $(cat deploy.log | tail -n 20)"
fi
```

---

## 2. Real-time Trading Decision Workflow

**News → Analysis → Forecast → Trade**

This workflow processes real-time market events to make trading decisions.

### Step 1: Real-time News Monitoring
```bash
# Monitor news sentiment for watchlist
WATCHLIST=("AAPL" "TSLA" "NVDA" "SPY")

for symbol in "${WATCHLIST[@]}"; do
  # Get real-time sentiment
  mcp__ai-news-trader__get_news_sentiment(
    symbol="$symbol",
    sources=["reuters", "bloomberg", "wsj", "cnbc"]
  )
  
  # Analyze if significant news detected
  mcp__ai-news-trader__analyze_news(
    symbol="$symbol",
    lookback_hours=4,
    sentiment_model="enhanced",
    use_gpu=true
  )
done

# Store significant events
./claude-flow memory store "market_event_$(date +%s)" "TSLA: Major earnings beat detected, sentiment score 0.85"
```

### Step 2: Quick Market Analysis
```bash
# Rapid analysis for high-sentiment symbols
mcp__ai-news-trader__quick_analysis(
  symbol="TSLA",
  use_gpu=true
)

# Get current portfolio exposure
mcp__ai-news-trader__get_portfolio_status(
  include_analytics=true
)

# Store analysis results
./claude-flow memory store "tsla_quick_analysis" "RSI: 68, MACD: bullish, Volume: +150%, Support: 175, Resistance: 185"
```

### Step 3: Neural Forecasting
```bash
# Generate multi-horizon forecasts
HORIZONS=(1 4 8 24)  # 1h, 4h, 8h, 24h

for horizon in "${HORIZONS[@]}"; do
  mcp__ai-news-trader__neural_forecast(
    symbol="TSLA",
    horizon=$horizon,
    confidence_level=0.95,
    model_id="nhits_tsla_production",
    use_gpu=true
  )
done

# Store forecast consensus
./claude-flow memory store "tsla_forecast_consensus" "1h: +1.2%, 4h: +2.1%, 8h: +1.8%, 24h: +2.5%, confidence: high"
```

### Step 4: Risk Assessment
```bash
# Analyze portfolio impact
mcp__ai-news-trader__risk_analysis(
  portfolio=[
    {"symbol": "AAPL", "shares": 100, "entry_price": 175.50},
    {"symbol": "TSLA", "shares": 50, "entry_price": 180.00},
    {"symbol": "SPY", "shares": 200, "entry_price": 470.00}
  ],
  time_horizon=1,
  var_confidence=0.05,
  use_monte_carlo=true,
  use_gpu=true
)

# Store risk metrics
./claude-flow memory store "portfolio_risk_pre_trade" "VaR(95%): $2,450, CVaR: $3,200, Max correlated loss: $4,100"
```

### Step 5: Trade Execution
```bash
# Simulate trade first
mcp__ai-news-trader__simulate_trade(
  strategy="neural_momentum",
  symbol="TSLA",
  action="buy",
  use_gpu=true
)

# If simulation passes risk checks, execute
if [ "$SIMULATION_PASSED" == "true" ]; then
  mcp__ai-news-trader__execute_trade(
    strategy="neural_momentum",
    symbol="TSLA",
    action="buy",
    quantity=25,
    order_type="limit",
    limit_price=181.50
  )
  
  # Store trade record
  ./claude-flow memory store "trade_$(date +%s)" "TSLA BUY 25@181.50, strategy: neural_momentum, forecast: +2.1%"
else
  # Log rejection reason
  ./claude-flow memory store "trade_rejected_$(date +%s)" "TSLA trade rejected: Risk limit exceeded, VaR impact: +$850"
fi
```

### Step 6: Post-Trade Monitoring
```bash
# Monitor position performance
while true; do
  # Check position P&L
  mcp__ai-news-trader__get_portfolio_status(include_analytics=true)
  
  # Update neural forecast
  mcp__ai-news-trader__neural_forecast(
    symbol="TSLA",
    horizon=1,
    model_id="nhits_tsla_production",
    use_gpu=true
  )
  
  # Check stop-loss/take-profit
  if [ "$CURRENT_PL" -le "$STOP_LOSS" ] || [ "$CURRENT_PL" -ge "$TAKE_PROFIT" ]; then
    mcp__ai-news-trader__execute_trade(
      strategy="neural_momentum",
      symbol="TSLA",
      action="sell",
      quantity=25,
      order_type="market"
    )
    break
  fi
  
  sleep 60  # Check every minute
done
```

---

## 3. Portfolio Optimization Workflow

**Correlation → Risk → Optimization → Rebalancing**

This workflow optimizes multi-asset portfolios using advanced analytics.

### Step 1: Correlation Analysis
```bash
# Define portfolio universe
PORTFOLIO_SYMBOLS=("AAPL" "MSFT" "GOOGL" "JPM" "XOM" "GLD" "TLT" "BTC-USD")

# Analyze correlations across multiple timeframes
for period in 30 60 90 180; do
  mcp__ai-news-trader__correlation_analysis(
    symbols=$PORTFOLIO_SYMBOLS,
    period_days=$period,
    use_gpu=true
  )
  
  # Store correlation matrix
  ./claude-flow memory store "correlations_${period}d" "Output stored in correlation_matrix_${period}d.json"
done

# Identify uncorrelated pairs for diversification
./claude-flow sparc run analyzer "Find optimal uncorrelated asset pairs from correlation matrices"
```

### Step 2: Comprehensive Risk Analysis
```bash
# Current portfolio composition
CURRENT_PORTFOLIO=[
  {"symbol": "AAPL", "shares": 100, "entry_price": 175.00},
  {"symbol": "MSFT", "shares": 75, "entry_price": 420.00},
  {"symbol": "JPM", "shares": 200, "entry_price": 180.00},
  {"symbol": "GLD", "shares": 150, "entry_price": 185.00},
  {"symbol": "TLT", "shares": 300, "entry_price": 92.00}
]

# Multi-horizon risk analysis
for horizon in 1 5 20; do
  mcp__ai-news-trader__risk_analysis(
    portfolio=$CURRENT_PORTFOLIO,
    time_horizon=$horizon,
    var_confidence=0.05,
    use_monte_carlo=true,
    use_gpu=true
  )
done

# Store risk profile
./claude-flow memory store "portfolio_risk_profile" "1d VaR: $3,200, 5d VaR: $7,100, 20d VaR: $14,300, Sharpe: 1.85"
```

### Step 3: Strategy-Level Optimization
```bash
# Optimize each strategy in portfolio
STRATEGIES=("momentum_neural" "mean_reversion_neural" "pairs_trading" "trend_following")

for strategy in "${STRATEGIES[@]}"; do
  mcp__ai-news-trader__optimize_strategy(
    strategy=$strategy,
    symbol="PORTFOLIO",  # Optimize for entire portfolio
    parameter_ranges={
      "position_sizing": [0.05, 0.25],
      "rebalance_threshold": [0.02, 0.10],
      "risk_per_trade": [0.01, 0.03],
      "correlation_limit": [0.5, 0.8]
    },
    optimization_metric="risk_adjusted_return",
    max_iterations=500,
    use_gpu=true
  )
  
  # Store optimal parameters
  ./claude-flow memory store "${strategy}_optimal_params" "Results stored in optimization_${strategy}.json"
done
```

### Step 4: Portfolio Rebalancing
```bash
# Calculate optimal weights using neural forecasts
for symbol in "${PORTFOLIO_SYMBOLS[@]}"; do
  # Get neural forecasts
  mcp__ai-news-trader__neural_forecast(
    symbol=$symbol,
    horizon=20,  # 20-day forecast for rebalancing
    confidence_level=0.95,
    use_gpu=true
  )
done

# Run portfolio optimization
mcp__ai-news-trader__optimize_strategy(
  strategy="portfolio_optimization",
  symbol="MULTI",
  parameter_ranges={
    "target_return": [0.08, 0.15],
    "max_volatility": [0.12, 0.20],
    "max_drawdown": [0.15, 0.25],
    "min_sharpe": [1.5, 2.5]
  },
  optimization_metric="sharpe_ratio",
  use_gpu=true
)

# Store optimal allocation
./claude-flow memory store "optimal_allocation_$(date +%Y%m)" "AAPL: 20%, MSFT: 18%, JPM: 15%, GLD: 12%, TLT: 15%, Cash: 20%"
```

### Step 5: Execute Rebalancing
```bash
# Calculate rebalancing trades
REBALANCE_TRADES=[
  {"symbol": "AAPL", "action": "sell", "quantity": 20},
  {"symbol": "GLD", "action": "buy", "quantity": 50},
  {"symbol": "TLT", "action": "buy", "quantity": 100}
]

# Execute rebalancing with VWAP
for trade in "${REBALANCE_TRADES[@]}"; do
  # Simulate first
  mcp__ai-news-trader__simulate_trade(
    strategy="portfolio_rebalance",
    symbol=$trade.symbol,
    action=$trade.action,
    use_gpu=true
  )
  
  # Execute if simulation passes
  mcp__ai-news-trader__execute_trade(
    strategy="portfolio_rebalance",
    symbol=$trade.symbol,
    action=$trade.action,
    quantity=$trade.quantity,
    order_type="limit",
    limit_price=$VWAP_PRICE
  )
  
  # Log rebalancing action
  ./claude-flow memory store "rebalance_$(date +%s)" "$trade executed at $EXECUTION_PRICE"
done

# Verify new portfolio metrics
mcp__ai-news-trader__get_portfolio_status(include_analytics=true)
```

---

## 4. Strategy Development Workflow

**Ideation → Implementation → Testing → Optimization**

This workflow guides you through developing a new neural-enhanced trading strategy.

### Step 1: Strategy Research & Ideation
```bash
# Research market inefficiencies
mcp__ai-news-trader__analyze_news(
  symbol="SPY",
  lookback_hours=720,  # 30 days
  sentiment_model="enhanced",
  use_gpu=true
)

# Analyze historical patterns
./claude-flow sparc run researcher "Identify recurring market patterns in SPY options flow and news sentiment correlations"

# Store research findings
./claude-flow memory store "strategy_research_options_sentiment" "Strong correlation between unusual options activity and 2-day price movements, R²=0.68"

# Define strategy hypothesis
./claude-flow memory store "strategy_hypothesis" "Combine unusual options flow with news sentiment and neural forecasting for 2-day swing trades"
```

### Step 2: Implementation Design
```bash
# Design strategy architecture
./claude-flow sparc run architect "Design neural-enhanced options flow trading strategy with sentiment integration"

# Create strategy configuration
STRATEGY_CONFIG={
  "name": "neural_options_sentiment",
  "instruments": ["SPY", "QQQ", "IWM"],
  "signals": {
    "options_flow": {
      "min_premium": 1000000,
      "min_volume_ratio": 3.0,
      "sentiment_threshold": 0.7
    },
    "neural_forecast": {
      "model": "nhits",
      "horizon": 48,
      "confidence_required": 0.75
    }
  },
  "risk_management": {
    "position_size": 0.15,
    "stop_loss": 0.025,
    "take_profit": 0.05,
    "max_positions": 3
  }
}

# Store configuration
./claude-flow memory store "strategy_config_neural_options" "$STRATEGY_CONFIG"
```

### Step 3: Test-Driven Development
```bash
# Implement strategy with TDD
./claude-flow sparc tdd "Neural options sentiment trading strategy" --config "$STRATEGY_CONFIG"

# Run unit tests
./claude-flow test unit src/strategies/neural_options_sentiment.py

# Run integration tests
./claude-flow test integration --strategy neural_options_sentiment --mock-data

# Store test results
./claude-flow memory store "strategy_tests_passed" "Unit: 42/42 passed, Integration: 18/18 passed, Coverage: 94%"
```

### Step 4: Historical Backtesting
```bash
# Comprehensive backtest across market conditions
BACKTEST_PERIODS=[
  {"start": "2020-01-01", "end": "2020-12-31", "label": "covid_volatility"},
  {"start": "2021-01-01", "end": "2021-12-31", "label": "bull_market"},
  {"start": "2022-01-01", "end": "2022-12-31", "label": "bear_market"},
  {"start": "2023-01-01", "end": "2023-12-31", "label": "recovery"},
  {"start": "2024-01-01", "end": "2024-06-27", "label": "recent"}
]

for period in "${BACKTEST_PERIODS[@]}"; do
  mcp__ai-news-trader__run_backtest(
    strategy="neural_options_sentiment",
    symbol="SPY",
    start_date=$period.start,
    end_date=$period.end,
    benchmark="sp500",
    include_costs=true,
    use_gpu=true
  )
  
  # Store period results
  ./claude-flow memory store "backtest_${period.label}" "Sharpe: X.XX, Max DD: -XX%, Win Rate: XX%"
done

# Generate performance report
mcp__ai-news-trader__performance_report(
  strategy="neural_options_sentiment",
  period_days=1825,  # 5 years
  include_benchmark=true,
  use_gpu=true
)
```

### Step 5: Strategy Optimization
```bash
# Multi-objective optimization
mcp__ai-news-trader__optimize_strategy(
  strategy="neural_options_sentiment",
  symbol="SPY",
  parameter_ranges={
    "options_min_premium": [500000, 2000000],
    "options_volume_ratio": [2.0, 5.0],
    "sentiment_threshold": [0.6, 0.85],
    "neural_confidence": [0.7, 0.9],
    "position_size": [0.1, 0.25],
    "stop_loss": [0.015, 0.035],
    "take_profit": [0.03, 0.08]
  },
  optimization_metric="sortino_ratio",  # Focus on downside risk
  max_iterations=1000,
  use_gpu=true
)

# Walk-forward optimization
for year in 2020 2021 2022 2023 2024; do
  # Optimize on previous 2 years
  mcp__ai-news-trader__optimize_strategy(
    strategy="neural_options_sentiment",
    symbol="SPY",
    parameter_ranges=$PARAM_RANGES,
    optimization_metric="sharpe_ratio",
    max_iterations=500,
    use_gpu=true
  )
  
  # Test on out-of-sample year
  mcp__ai-news-trader__run_backtest(
    strategy="neural_options_sentiment",
    symbol="SPY",
    start_date="${year}-01-01",
    end_date="${year}-12-31",
    use_gpu=true
  )
done

# Store final optimized parameters
./claude-flow memory store "neural_options_optimized_final" "Premium: 750k, Ratio: 3.5, Sentiment: 0.75, Confidence: 0.8"
```

### Step 6: Paper Trading Validation
```bash
# Deploy to paper trading
./claude-flow-neural strategy deploy neural_options_sentiment --env paper --monitoring enabled

# Monitor paper trading performance
for i in {1..30}; do  # 30 days
  # Daily performance check
  mcp__ai-news-trader__get_portfolio_status(include_analytics=true)
  
  # Log daily metrics
  ./claude-flow memory store "paper_day_${i}" "P&L: $X,XXX, Trades: XX, Win Rate: XX%, Sharpe: X.XX"
  
  # Check for strategy degradation
  if [ "$DAILY_SHARPE" -lt "1.0" ]; then
    ./claude-flow alert "Strategy underperforming: Daily Sharpe < 1.0"
  fi
  
  sleep 86400  # Wait 24 hours
done

# Final paper trading report
mcp__ai-news-trader__performance_report(
  strategy="neural_options_sentiment",
  period_days=30,
  include_benchmark=true
)
```

---

## 5. Risk Management Workflow

**Monitoring → Analysis → Alerts → Adjustments**

This workflow maintains robust risk controls for live trading systems.

### Step 1: Real-time Risk Monitoring
```bash
# Start risk monitoring daemon
./claude-flow risk-monitor start --config risk_limits.yaml &

# Monitor key risk metrics
while true; do
  # Portfolio-level risk check
  mcp__ai-news-trader__risk_analysis(
    portfolio=$LIVE_PORTFOLIO,
    time_horizon=1,
    var_confidence=0.01,  # 99% confidence for risk
    use_monte_carlo=true,
    use_gpu=true
  )
  
  # Individual position checks
  mcp__ai-news-trader__get_portfolio_status(include_analytics=true)
  
  # Check correlation risk
  mcp__ai-news-trader__correlation_analysis(
    symbols=$PORTFOLIO_SYMBOLS,
    period_days=20,  # Recent correlations
    use_gpu=true
  )
  
  # Store risk snapshot
  ./claude-flow memory store "risk_snapshot_$(date +%s)" "VaR: $X,XXX, Exposure: XX%, Max Correlation: X.XX"
  
  sleep 300  # Check every 5 minutes
done
```

### Step 2: Anomaly Detection
```bash
# Neural model performance monitoring
for model_id in "${ACTIVE_MODELS[@]}"; do
  # Check model health
  mcp__ai-news-trader__neural_model_status(model_id=$model_id)
  
  # Detect forecast anomalies
  mcp__ai-news-trader__neural_forecast(
    symbol="SPY",
    horizon=1,
    model_id=$model_id,
    confidence_level=0.99,
    use_gpu=true
  )
  
  # Compare with ensemble
  if [ "$FORECAST_DEVIATION" -gt "3_SIGMA" ]; then
    ./claude-flow alert "Model $model_id showing anomalous forecasts"
    ./claude-flow memory store "anomaly_$(date +%s)" "Model $model_id deviation: ${FORECAST_DEVIATION}σ"
  fi
done

# Market regime detection
mcp__ai-news-trader__analyze_news(
  symbol="SPY",
  lookback_hours=24,
  sentiment_model="enhanced",
  use_gpu=true
)

# Check for regime change
if [ "$SENTIMENT_SHIFT" -gt "2_STDEV" ] || [ "$VOLATILITY_SPIKE" == "true" ]; then
  ./claude-flow alert "Market regime change detected"
  RISK_ADJUSTMENT_NEEDED=true
fi
```

### Step 3: Dynamic Risk Alerts
```bash
# Set up multi-level alerts
ALERT_LEVELS={
  "info": {"var_breach": 0.8, "drawdown": 0.5},
  "warning": {"var_breach": 0.9, "drawdown": 0.7},
  "critical": {"var_breach": 1.0, "drawdown": 0.9}
}

# Check against thresholds
CURRENT_METRICS=$(mcp__ai-news-trader__get_portfolio_status(include_analytics=true))

for level in "${!ALERT_LEVELS[@]}"; do
  if [ "$CURRENT_VAR_USAGE" -gt "${ALERT_LEVELS[$level][var_breach]}" ]; then
    ./claude-flow alert --level $level "VaR usage at ${CURRENT_VAR_USAGE}%"
    
    # Store alert
    ./claude-flow memory store "alert_${level}_$(date +%s)" "VaR breach: ${CURRENT_VAR_USAGE}% of limit"
    
    # Trigger risk reduction if critical
    if [ "$level" == "critical" ]; then
      REDUCE_RISK=true
    fi
  fi
done
```

### Step 4: Automated Risk Adjustments
```bash
# Reduce risk if needed
if [ "$REDUCE_RISK" == "true" ] || [ "$RISK_ADJUSTMENT_NEEDED" == "true" ]; then
  echo "Initiating risk reduction protocol..."
  
  # Step 1: Reduce position sizes
  POSITIONS=$(mcp__ai-news-trader__get_portfolio_status(include_analytics=true))
  
  for position in $POSITIONS; do
    if [ "$position.unrealized_pnl" -lt 0 ] && [ "$position.size" -gt "$MIN_POSITION_SIZE" ]; then
      # Reduce losing positions by 50%
      REDUCE_QTY=$((position.quantity / 2))
      
      mcp__ai-news-trader__execute_trade(
        strategy="risk_reduction",
        symbol=$position.symbol,
        action="sell",
        quantity=$REDUCE_QTY,
        order_type="market"
      )
      
      ./claude-flow memory store "risk_reduction_$(date +%s)" "Reduced $position.symbol by $REDUCE_QTY shares"
    fi
  done
  
  # Step 2: Hedge portfolio
  if [ "$PORTFOLIO_DELTA" -gt "0.5" ]; then
    # Buy put options for hedging
    mcp__ai-news-trader__execute_trade(
      strategy="portfolio_hedge",
      symbol="SPY",
      action="buy_put",
      quantity=10,
      strike="ATM-2%",
      expiration="30_days"
    )
    
    ./claude-flow memory store "hedge_added_$(date +%s)" "Added SPY put hedge: 10 contracts ATM-2%"
  fi
  
  # Step 3: Increase cash allocation
  TARGET_CASH_PERCENT=40
  ./claude-flow rebalance --target-cash $TARGET_CASH_PERCENT --urgency high
fi
```

### Step 5: Risk Reporting & Learning
```bash
# Generate comprehensive risk report
mcp__ai-news-trader__performance_report(
  strategy="all",
  period_days=30,
  include_benchmark=true,
  use_gpu=true
)

# Analyze risk events
./claude-flow sparc run analyzer "Analyze all risk events from past 30 days and identify patterns"

# Update risk parameters based on learnings
RISK_LEARNINGS=$(./claude-flow memory get "risk_events_analysis")

if [ "$RISK_LEARNINGS" != "" ]; then
  # Update risk limits
  NEW_RISK_PARAMS={
    "max_var": "$ADJUSTED_VAR_LIMIT",
    "max_position_size": "$ADJUSTED_POSITION_LIMIT",
    "correlation_threshold": "$ADJUSTED_CORRELATION_LIMIT",
    "stop_loss_multiplier": "$ADJUSTED_STOP_LOSS"
  }
  
  # Store updated parameters
  ./claude-flow memory store "risk_params_v$(date +%Y%m%d)" "$NEW_RISK_PARAMS"
  
  # Apply to all strategies
  for strategy in "${ACTIVE_STRATEGIES[@]}"; do
    ./claude-flow strategy update $strategy --risk-params "$NEW_RISK_PARAMS"
  done
fi

# Schedule next risk review
echo "Risk management cycle completed. Next review in 24 hours."
```

---

## Error Handling & Recovery Patterns

### Connection Failures
```bash
# Implement exponential backoff for API calls
retry_with_backoff() {
  local max_attempts=5
  local timeout=1
  local attempt=0
  
  while [ $attempt -lt $max_attempts ]; do
    if mcp__ai-news-trader__ping(); then
      return 0
    fi
    
    echo "Connection failed, attempt $((attempt + 1))/$max_attempts"
    sleep $timeout
    timeout=$((timeout * 2))
    attempt=$((attempt + 1))
  done
  
  # Fallback to cached data
  ./claude-flow memory get "last_known_good_state"
  return 1
}
```

### Model Failures
```bash
# Fallback cascade for neural predictions
get_forecast_with_fallback() {
  local symbol=$1
  local horizon=$2
  
  # Try primary model
  if mcp__ai-news-trader__neural_forecast(
    symbol=$symbol,
    horizon=$horizon,
    model_id="primary_model",
    use_gpu=true
  ); then
    return 0
  fi
  
  # Try backup model
  if mcp__ai-news-trader__neural_forecast(
    symbol=$symbol,
    horizon=$horizon,
    model_id="backup_model",
    use_gpu=false  # CPU fallback
  ); then
    echo "WARNING: Using backup model"
    return 0
  fi
  
  # Fall back to simple technical analysis
  mcp__ai-news-trader__quick_analysis(symbol=$symbol, use_gpu=false)
  echo "ERROR: Neural models unavailable, using technical analysis"
  return 1
}
```

### Data Quality Checks
```bash
# Validate data before model training
validate_training_data() {
  local data_path=$1
  
  # Check for missing values
  if [ $(check_missing_values $data_path) -gt 0.05 ]; then
    echo "ERROR: More than 5% missing values"
    return 1
  fi
  
  # Check for outliers
  if [ $(detect_outliers $data_path) == "true" ]; then
    echo "WARNING: Outliers detected, applying winsorization"
    apply_winsorization $data_path
  fi
  
  # Verify stationarity
  if [ $(test_stationarity $data_path) == "false" ]; then
    echo "WARNING: Non-stationary data, applying differencing"
    apply_differencing $data_path
  fi
  
  return 0
}
```

---

## Best Practices Summary

1. **Always validate inputs** before executing trades or training models
2. **Store all significant events** in memory for audit trails
3. **Use GPU acceleration** for all computationally intensive operations
4. **Implement proper error handling** with fallback mechanisms
5. **Monitor continuously** and adjust parameters based on performance
6. **Test thoroughly** before deploying any strategy changes
7. **Maintain risk limits** and never override them without review
8. **Document all workflows** and keep them updated

---

## Quick Reference Card

```bash
# Essential MCP Tools
mcp__ai-news-trader__ping()                    # Check connectivity
mcp__ai-news-trader__quick_analysis()          # Fast market analysis
mcp__ai-news-trader__neural_forecast()         # AI price predictions
mcp__ai-news-trader__analyze_news()            # News sentiment analysis
mcp__ai-news-trader__risk_analysis()           # Portfolio risk assessment
mcp__ai-news-trader__execute_trade()           # Live trade execution
mcp__ai-news-trader__get_portfolio_status()    # Portfolio analytics

# Workflow Commands
./claude-flow memory store <key> <value>        # Store results
./claude-flow memory get <key>                  # Retrieve data
./claude-flow monitor                           # System monitoring
./claude-flow alert <message>                   # Send alerts
./claude-flow sparc run <mode> <task>          # AI assistance

# Common Parameters
use_gpu=true                                    # Enable GPU acceleration
include_analytics=true                          # Include detailed metrics
optimization_metric="sharpe_ratio"              # Optimization target
confidence_level=0.95                           # Statistical confidence
```