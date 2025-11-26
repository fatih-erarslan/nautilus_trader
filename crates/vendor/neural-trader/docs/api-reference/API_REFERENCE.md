# Neural Trader v2.1.0 - Complete API Reference

**103 Production-Ready Functions**

---

## Table of Contents

1. [Core Trading Functions](#core-trading-functions) (20)
2. [Neural Network Functions](#neural-network-functions) (10)
3. [Risk Management](#risk-management) (8)
4. [News & Sentiment](#news-sentiment) (10)
5. [Strategy Management](#strategy-management) (8)
6. [Sports Betting](#sports-betting) (20)
7. [Investment Syndicates](#investment-syndicates) (17)
8. [E2B Cloud Integration](#e2b-cloud-integration) (10)

---

## Core Trading Functions

### `ping()`
Simple connectivity test.

**Parameters:** None

**Returns:**
```typescript
{
  status: "ok",
  message: "Neural Trader MCP Server is running",
  timestamp: "2025-11-14T12:00:00Z"
}
```

**Example:**
```javascript
const health = await ping();
console.log(health.status); // "ok"
```

---

### `list_strategies()`
Get all available trading strategies with GPU capabilities.

**Parameters:** None

**Returns:**
```typescript
{
  strategies: Array<{
    name: string;
    description: string;
    risk_level: "low" | "medium" | "high";
    gpu_enabled: boolean;
    parameters: object;
  }>;
  count: number;
}
```

**Example:**
```javascript
const { strategies } = await list_strategies();
strategies.forEach(s => {
  console.log(`${s.name}: ${s.description}`);
});
```

---

### `get_strategy_info(strategy: string)`
Get detailed information about a specific strategy.

**Parameters:**
- `strategy` (string, required): Strategy name

**Returns:**
```typescript
{
  name: string;
  description: string;
  risk_level: "low" | "medium" | "high";
  parameters: {
    [key: string]: {
      type: string;
      default: any;
      description: string;
    }
  };
  performance_metrics: {
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
  gpu_acceleration: boolean;
}
```

**Example:**
```javascript
const info = await get_strategy_info({
  strategy: "momentum_trader"
});
console.log(`Sharpe Ratio: ${info.performance_metrics.sharpe_ratio}`);
```

---

### `quick_analysis(symbol: string, use_gpu?: boolean)`
Quick market analysis for a symbol.

**Parameters:**
- `symbol` (string, required): Trading symbol (e.g., "SPY")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  symbol: string;
  current_price: number;
  change_percent: number;
  volume: number;
  indicators: {
    rsi: number;
    macd: { value: number; signal: number; histogram: number };
    bollinger_bands: { upper: number; middle: number; lower: number };
  };
  signal: "buy" | "sell" | "hold";
  confidence: number;
  execution_time_ms: number;
}
```

**Example:**
```javascript
const analysis = await quick_analysis({
  symbol: "AAPL",
  use_gpu: true
});
console.log(`Signal: ${analysis.signal} (${analysis.confidence}% confidence)`);
```

---

### `simulate_trade(strategy: string, symbol: string, action: string, use_gpu?: boolean)`
Simulate a trade without execution.

**Parameters:**
- `strategy` (string, required): Strategy to use
- `symbol` (string, required): Trading symbol
- `action` (string, required): "buy" or "sell"
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  simulation_id: string;
  strategy: string;
  symbol: string;
  action: "buy" | "sell";
  price: number;
  quantity: number;
  expected_pnl: number;
  risk_metrics: {
    var_95: number;
    max_loss: number;
    probability_profit: number;
  };
  execution_time_ms: number;
}
```

**Example:**
```javascript
const sim = await simulate_trade({
  strategy: "neural_sentiment",
  symbol: "TSLA",
  action: "buy",
  use_gpu: true
});
console.log(`Expected P&L: $${sim.expected_pnl}`);
```

---

### `get_portfolio_status(include_analytics?: boolean)`
Get current portfolio status with analytics.

**Parameters:**
- `include_analytics` (boolean, optional): Include detailed analytics (default: true)

**Returns:**
```typescript
{
  total_value: number;
  cash: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    avg_cost: number;
    current_price: number;
    unrealized_pnl: number;
    unrealized_pnl_percent: number;
  }>;
  analytics?: {
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
  };
}
```

**Example:**
```javascript
const portfolio = await get_portfolio_status({
  include_analytics: true
});
console.log(`Portfolio Value: $${portfolio.total_value}`);
console.log(`Sharpe Ratio: ${portfolio.analytics.sharpe_ratio}`);
```

---

### `execute_trade(strategy: string, symbol: string, action: string, quantity: number)`
Execute a live trade.

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Trading symbol
- `action` (string, required): "buy" or "sell"
- `quantity` (number, required): Number of shares
- `order_type` (string, optional): "market" or "limit" (default: "market")
- `limit_price` (number, optional): Limit price if order_type is "limit"

**Returns:**
```typescript
{
  order_id: string;
  strategy: string;
  symbol: string;
  action: "buy" | "sell";
  quantity: number;
  order_type: "market" | "limit";
  status: "pending" | "filled" | "cancelled";
  filled_price?: number;
  commission: number;
  timestamp: string;
}
```

**Example:**
```javascript
const order = await execute_trade({
  strategy: "momentum_trader",
  symbol: "SPY",
  action: "buy",
  quantity: 10,
  order_type: "market"
});
console.log(`Order ${order.order_id}: ${order.status}`);
```

---

### `run_backtest(strategy: string, symbol: string, start_date: string, end_date: string)`
Run historical backtest.

**Parameters:**
- `strategy` (string, required): Strategy to test
- `symbol` (string, required): Trading symbol
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `benchmark` (string, optional): Benchmark symbol (default: "sp500")
- `include_costs` (boolean, optional): Include commissions (default: true)
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  strategy: string;
  symbol: string;
  period: { start: string; end: string };
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  trades: {
    total: number;
    winning: number;
    losing: number;
  };
  benchmark_comparison: {
    symbol: string;
    return: number;
    alpha: number;
    beta: number;
  };
}
```

**Example:**
```javascript
const backtest = await run_backtest({
  strategy: "neural_trend",
  symbol: "QQQ",
  start_date: "2024-01-01",
  end_date: "2024-12-31",
  use_gpu: true
});
console.log(`Total Return: ${backtest.total_return}%`);
console.log(`Sharpe: ${backtest.sharpe_ratio}`);
```

---

## Neural Network Functions

### `neural_train(config: object, tier?: string)`
Train a neural network model.

**Parameters:**
- `config` (object, required): Training configuration
  - `architecture` (object): Network architecture
    - `type` (string): "lstm" | "gru" | "transformer" | "cnn" | "rnn" | "gan" | "autoencoder" | "custom"
    - `layers` (array): Layer configurations
  - `training` (object): Training parameters
    - `epochs` (number): Training epochs
    - `batch_size` (number): Batch size
    - `learning_rate` (number): Learning rate
    - `optimizer` (string): "adam" | "sgd" | "rmsprop" | "adagrad"
- `tier` (string, optional): "nano" | "mini" | "small" | "medium" | "large"
- `use_gpu` (boolean, optional): Enable GPU

**Returns:**
```typescript
{
  model_id: string;
  architecture: string;
  parameters: number;
  training_time_seconds: number;
  final_loss: number;
  validation_metrics: {
    mae: number;
    rmse: number;
    r2_score: number;
  };
  gpu_used: boolean;
}
```

**Example:**
```javascript
const model = await neural_train({
  config: {
    architecture: {
      type: "lstm",
      layers: [
        { units: 128, activation: "relu" },
        { units: 64, activation: "relu" },
        { units: 32, activation: "tanh" }
      ]
    },
    training: {
      epochs: 100,
      batch_size: 32,
      learning_rate: 0.001,
      optimizer: "adam"
    }
  },
  tier: "medium",
  use_gpu: true
});
console.log(`Model trained in ${model.training_time_seconds}s`);
```

---

### `neural_predict(model_id: string, input: array)`
Run inference on trained model.

**Parameters:**
- `model_id` (string, required): Model identifier
- `input` (array, required): Input data
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  predictions: number[];
  confidence_intervals?: {
    lower: number[];
    upper: number[];
  };
  execution_time_ms: number;
}
```

**Example:**
```javascript
const prediction = await neural_predict({
  model_id: "lstm-20251114-abc123",
  input: [[1.5, 2.3, 1.8, 2.1]],
  use_gpu: true
});
console.log(`Prediction: ${prediction.predictions[0]}`);
```

---

### `neural_forecast(model_id: string, symbol: string, horizon: number)`
Generate time-series forecast.

**Parameters:**
- `model_id` (string, required): Model ID
- `symbol` (string, required): Trading symbol
- `horizon` (number, required): Forecast steps ahead
- `confidence_level` (number, optional): Confidence level (default: 0.95)
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  symbol: string;
  forecast: number[];
  dates: string[];
  confidence_bands: {
    upper: number[];
    lower: number[];
  };
  metrics: {
    mean_forecast: number;
    volatility: number;
  };
}
```

**Example:**
```javascript
const forecast = await neural_forecast({
  model_id: "lstm-spy-model",
  symbol: "SPY",
  horizon: 30,
  confidence_level: 0.95,
  use_gpu: true
});
console.log(`30-day forecast: ${forecast.forecast}`);
```

---

### `neural_evaluate(model_id: string, test_data: string)`
Evaluate model performance.

**Parameters:**
- `model_id` (string, required): Model ID
- `test_data` (string, required): Test dataset path
- `metrics` (array, optional): Metrics to compute (default: ["mae", "rmse", "mape", "r2_score"])
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  model_id: string;
  metrics: {
    mae: number;
    rmse: number;
    mape: number;
    r2_score: number;
  };
  confusion_matrix?: number[][];
  execution_time_ms: number;
}
```

**Example:**
```javascript
const eval = await neural_evaluate({
  model_id: "lstm-spy-model",
  test_data: "s3://bucket/test-data.parquet",
  metrics: ["mae", "rmse", "r2_score"],
  use_gpu: true
});
console.log(`MAE: ${eval.metrics.mae}`);
```

---

### `neural_backtest(model_id: string, start_date: string, end_date: string)`
Backtest neural model predictions.

**Parameters:**
- `model_id` (string, required): Model ID
- `start_date` (string, required): Start date
- `end_date` (string, required): End date
- `benchmark` (string, optional): Benchmark (default: "sp500")
- `rebalance_frequency` (string, optional): "daily" | "weekly" | "monthly"
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  model_id: string;
  period: { start: string; end: string };
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  benchmark_comparison: {
    return: number;
    alpha: number;
    beta: number;
  };
}
```

**Example:**
```javascript
const backtest = await neural_backtest({
  model_id: "lstm-spy-model",
  start_date: "2024-01-01",
  end_date: "2024-12-31",
  benchmark: "sp500",
  use_gpu: true
});
console.log(`Neural Strategy Return: ${backtest.total_return}%`);
```

---

### `neural_optimize(model_id: string, parameter_ranges: object)`
Optimize model hyperparameters.

**Parameters:**
- `model_id` (string, required): Model ID
- `parameter_ranges` (object, required): Parameter search space
- `trials` (number, optional): Optimization trials (default: 100)
- `optimization_metric` (string, optional): Metric to optimize (default: "mae")
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  best_parameters: object;
  best_score: number;
  optimization_history: Array<{
    trial: number;
    parameters: object;
    score: number;
  }>;
  total_time_seconds: number;
}
```

**Example:**
```javascript
const optimized = await neural_optimize({
  model_id: "lstm-spy-model",
  parameter_ranges: {
    learning_rate: [0.0001, 0.01],
    batch_size: [16, 32, 64],
    num_layers: [2, 3, 4]
  },
  trials: 100,
  use_gpu: true
});
console.log(`Best params: ${JSON.stringify(optimized.best_parameters)}`);
```

---

### `neural_model_status(model_id?: string)`
Get model status and metadata.

**Parameters:**
- `model_id` (string, optional): Specific model ID (omit for all models)

**Returns:**
```typescript
{
  models: Array<{
    id: string;
    architecture: string;
    status: "training" | "ready" | "failed";
    created_at: string;
    accuracy: number;
    size_mb: number;
  }>;
}
```

**Example:**
```javascript
const status = await neural_model_status();
status.models.forEach(m => {
  console.log(`${m.id}: ${m.status} (accuracy: ${m.accuracy})`);
});
```

---

## Risk Management

### `risk_analysis(portfolio: array)`
Comprehensive portfolio risk analysis.

**Parameters:**
- `portfolio` (array, required): Portfolio positions
- `time_horizon` (number, optional): Days ahead (default: 1)
- `var_confidence` (number, optional): VaR confidence level (default: 0.05)
- `use_monte_carlo` (boolean, optional): Enable Monte Carlo (default: true)
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  var_95: number;
  cvar_95: number;
  max_drawdown: number;
  volatility: number;
  sharpe_ratio: number;
  beta: number;
  correlation_matrix: number[][];
  stress_tests: {
    market_crash: number;
    recession: number;
    interest_rate_spike: number;
  };
}
```

**Example:**
```javascript
const risk = await risk_analysis({
  portfolio: [
    { symbol: "SPY", weight: 0.5 },
    { symbol: "QQQ", weight: 0.3 },
    { symbol: "IWM", weight: 0.2 }
  ],
  time_horizon: 252,
  var_confidence: 0.05,
  use_gpu: true
});
console.log(`VaR (95%): $${risk.var_95}`);
```

---

### `correlation_analysis(symbols: array)`
Analyze asset correlations.

**Parameters:**
- `symbols` (array, required): List of symbols
- `period_days` (number, optional): Lookback period (default: 90)
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  correlation_matrix: number[][];
  covariance_matrix: number[][];
  eigenvalues: number[];
  principal_components: number[][];
}
```

**Example:**
```javascript
const corr = await correlation_analysis({
  symbols: ["SPY", "QQQ", "IWM", "DIA"],
  period_days: 90,
  use_gpu: true
});
console.log(corr.correlation_matrix);
```

---

## News & Sentiment

### `control_news_collection(action: string)`
Control news collection service.

**Parameters:**
- `action` (string, required): "start" | "stop" | "status"
- `symbols` (array, optional): Symbols to track
- `sources` (array, optional): News sources
- `update_frequency` (number, optional): Seconds between updates
- `lookback_hours` (number, optional): Historical lookback

**Returns:**
```typescript
{
  status: "started" | "stopped" | "running";
  symbols: string[];
  sources: string[];
  update_frequency: number;
  articles_collected: number;
}
```

**Example:**
```javascript
await control_news_collection({
  action: "start",
  symbols: ["AAPL", "TSLA", "NVDA"],
  sources: ["newsapi", "finnhub"],
  update_frequency: 300
});
```

---

### `analyze_news(symbol: string)`
AI-powered news sentiment analysis.

**Parameters:**
- `symbol` (string, required): Trading symbol
- `lookback_hours` (number, optional): Hours to analyze (default: 24)
- `sentiment_model` (string, optional): "basic" | "enhanced" (default: "enhanced")
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  symbol: string;
  overall_sentiment: "bullish" | "bearish" | "neutral";
  sentiment_score: number;  // -1 to 1
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  key_topics: string[];
  signal: "buy" | "sell" | "hold";
  confidence: number;
}
```

**Example:**
```javascript
const sentiment = await analyze_news({
  symbol: "AAPL",
  lookback_hours: 24,
  sentiment_model: "enhanced",
  use_gpu: true
});
console.log(`Sentiment: ${sentiment.overall_sentiment} (${sentiment.sentiment_score})`);
```

---

## Sports Betting

### `get_sports_odds(sport: string)`
Get real-time sports odds.

**Parameters:**
- `sport` (string, required): Sport key (e.g., "americanfootball_nfl")
- `regions` (array, optional): Regions (default: ["us"])
- `markets` (array, optional): Markets (default: ["h2h"])
- `use_gpu` (boolean, optional): GPU for analysis

**Returns:**
```typescript
{
  sport: string;
  events: Array<{
    id: string;
    home_team: string;
    away_team: string;
    commence_time: string;
    bookmakers: Array<{
      name: string;
      markets: Array<{
        key: string;
        outcomes: Array<{
          name: string;
          price: number;
        }>;
      }>;
    }>;
  }>;
}
```

**Example:**
```javascript
const odds = await get_sports_odds({
  sport: "americanfootball_nfl",
  regions: ["us", "uk"],
  markets: ["h2h", "spreads", "totals"]
});
```

---

### `calculate_kelly_criterion(probability: number, odds: number, bankroll: number)`
Calculate optimal bet size.

**Parameters:**
- `probability` (number, required): Win probability (0-1)
- `odds` (number, required): Decimal odds
- `bankroll` (number, required): Total bankroll
- `confidence` (number, optional): Confidence adjustment (default: 1)

**Returns:**
```typescript
{
  optimal_stake: number;
  kelly_fraction: number;
  expected_value: number;
  risk_of_ruin: number;
}
```

**Example:**
```javascript
const bet = await calculate_kelly_criterion({
  probability: 0.55,
  odds: 2.1,
  bankroll: 10000,
  confidence: 0.75
});
console.log(`Optimal bet: $${bet.optimal_stake}`);
```

---

## Investment Syndicates

### `create_syndicate_tool(syndicate_id: string, name: string)`
Create investment syndicate.

**Parameters:**
- `syndicate_id` (string, required): Unique ID
- `name` (string, required): Syndicate name
- `description` (string, optional): Description

**Returns:**
```typescript
{
  id: string;
  name: string;
  description: string;
  created_at: string;
  total_capital: number;
  member_count: number;
}
```

**Example:**
```javascript
const syndicate = await create_syndicate_tool({
  syndicate_id: "nfl-2025",
  name: "NFL Season 2025",
  description: "Collaborative NFL betting"
});
```

---

### `allocate_syndicate_funds(syndicate_id: string, opportunities: array)`
Allocate syndicate capital.

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `opportunities` (array, required): Betting opportunities
- `strategy` (string, optional): "kelly_criterion" | "equal" | "custom"

**Returns:**
```typescript
{
  allocations: Array<{
    opportunity_id: string;
    amount: number;
    kelly_fraction: number;
    expected_value: number;
  }>;
  total_allocated: number;
  remaining_capital: number;
}
```

**Example:**
```javascript
const allocation = await allocate_syndicate_funds({
  syndicate_id: "nfl-2025",
  opportunities: [
    { id: "game1", probability: 0.6, odds: 1.8 },
    { id: "game2", probability: 0.55, odds: 2.0 }
  ],
  strategy: "kelly_criterion"
});
```

---

## E2B Cloud Integration

### `create_e2b_sandbox(name: string, template: string)`
Create cloud sandbox.

**Parameters:**
- `name` (string, required): Sandbox name
- `template` (string, optional): Template (default: "base")
- `cpu_count` (number, optional): CPUs (default: 1)
- `memory_mb` (number, optional): Memory MB (default: 512)
- `timeout` (number, optional): Timeout seconds (default: 300)

**Returns:**
```typescript
{
  id: string;
  name: string;
  template: string;
  status: "running" | "stopped";
  created_at: string;
  url: string;
}
```

**Example:**
```javascript
const sandbox = await create_e2b_sandbox({
  name: "trading-bot-1",
  template: "nodejs",
  cpu_count: 2,
  memory_mb: 2048
});
```

---

### `run_e2b_agent(sandbox_id: string, agent_type: string, symbols: array)`
Run trading agent in sandbox.

**Parameters:**
- `sandbox_id` (string, required): Sandbox ID
- `agent_type` (string, required): Agent type
- `symbols` (array, required): Trading symbols
- `strategy_params` (object, optional): Strategy parameters
- `use_gpu` (boolean, optional): GPU acceleration

**Returns:**
```typescript
{
  agent_id: string;
  sandbox_id: string;
  agent_type: string;
  status: "running" | "stopped" | "error";
  performance: {
    trades: number;
    pnl: number;
    sharpe_ratio: number;
  };
}
```

**Example:**
```javascript
const agent = await run_e2b_agent({
  sandbox_id: "sandbox-123",
  agent_type: "momentum_trader",
  symbols: ["SPY", "QQQ"],
  strategy_params: { lookback: 20 }
});
```

---

## Error Handling

All functions may throw errors with the following structure:

```typescript
{
  code: string;         // Error code
  message: string;      // Human-readable message
  details?: object;     // Additional context
  timestamp: string;    // Error timestamp
}
```

**Common Error Codes:**
- `INVALID_PARAMETERS` - Invalid function parameters
- `AUTHENTICATION_FAILED` - API authentication failure
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INSUFFICIENT_FUNDS` - Insufficient account balance
- `MODEL_NOT_FOUND` - Neural model doesn't exist
- `GPU_NOT_AVAILABLE` - GPU requested but not available
- `TIMEOUT` - Operation timeout

**Example Error Handling:**
```javascript
try {
  const result = await neural_train({ config });
} catch (error) {
  if (error.code === "GPU_NOT_AVAILABLE") {
    // Retry with CPU
    const result = await neural_train({ config, use_gpu: false });
  } else {
    console.error(`Error: ${error.message}`);
  }
}
```

---

## Rate Limits

API rate limits (per minute):

| Tier | Requests/min | GPU Calls/min |
|------|--------------|---------------|
| Free | 60 | 10 |
| Pro | 600 | 100 |
| Enterprise | Unlimited | Unlimited |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1700000000
```

---

## Pagination

Functions returning large datasets support pagination:

**Parameters:**
- `limit` (number): Items per page (default: 100, max: 1000)
- `offset` (number): Skip N items (default: 0)

**Response:**
```typescript
{
  data: any[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    has_more: boolean;
  };
}
```

---

**For complete examples and tutorials, see the [Examples](./EXAMPLES.md) documentation.**
