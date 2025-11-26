# Neural Trader MCP API Reference

Complete reference for all 107 MCP tools available through the NAPI-RS implementation.

## Table of Contents

- [Strategy Analysis Tools](#strategy-analysis-tools)
- [Neural Network Tools](#neural-network-tools)
- [Trading Execution Tools](#trading-execution-tools)
- [Portfolio Management Tools](#portfolio-management-tools)
- [Risk Management Tools](#risk-management-tools)
- [Sports Betting Tools](#sports-betting-tools)
- [Syndicate Management Tools](#syndicate-management-tools)
- [News & Sentiment Tools](#news-sentiment-tools)
- [Prediction Market Tools](#prediction-market-tools)
- [System Tools](#system-tools)
- [E2B Integration Tools](#e2b-integration-tools)
- [Broker Integration Tools](#broker-integration-tools)

---

## Strategy Analysis Tools

### ping

Simple connectivity test tool.

**Function Signature:**
```typescript
ping(): Promise<{ status: string; timestamp: string }>;
```

**Parameters:** None

**Returns:**
```typescript
{
  status: "ok",
  timestamp: "2025-01-14T10:30:00Z"
}
```

**Example (Node.js):**
```javascript
const { McpServer } = require('@neural-trader/mcp');
const server = new McpServer();
await server.start();

const result = await server.callTool('ping', {});
console.log(result);
```

**Example (curl with MCP protocol):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping","arguments":{}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### list_strategies

List all available trading strategies with GPU capabilities.

**Function Signature:**
```typescript
list_strategies(): Promise<{
  strategies: Array<{
    name: string;
    description: string;
    gpu_capable: boolean;
    parameters: Record<string, any>;
  }>;
}>;
```

**Parameters:** None

**Returns:**
```typescript
{
  strategies: [
    {
      name: "momentum",
      description: "Momentum-based strategy with GPU acceleration",
      gpu_capable: true,
      parameters: {
        lookback_period: 20,
        threshold: 0.02
      }
    },
    // ... more strategies
  ]
}
```

**Example (Node.js):**
```javascript
const result = await server.callTool('list_strategies', {});
result.strategies.forEach(s => {
  console.log(`${s.name}: GPU=${s.gpu_capable}`);
});
```

**Example (curl):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_strategies","arguments":{}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### get_strategy_info

Get detailed information about a specific trading strategy.

**Function Signature:**
```typescript
get_strategy_info(params: {
  strategy: string;
}): Promise<{
  name: string;
  description: string;
  gpu_capable: boolean;
  parameters: Record<string, any>;
  performance_metrics: {
    sharpe_ratio?: number;
    max_drawdown?: number;
    win_rate?: number;
  };
}>;
```

**Parameters:**
- `strategy` (string, required): Strategy name

**Returns:**
```typescript
{
  name: "momentum",
  description: "Momentum strategy using GPU-accelerated calculations",
  gpu_capable: true,
  parameters: {
    lookback_period: 20,
    threshold: 0.02,
    use_gpu: true
  },
  performance_metrics: {
    sharpe_ratio: 1.85,
    max_drawdown: -0.12,
    win_rate: 0.58
  }
}
```

**Example (Node.js):**
```javascript
const info = await server.callTool('get_strategy_info', {
  strategy: 'momentum'
});
console.log(`Sharpe Ratio: ${info.performance_metrics.sharpe_ratio}`);
```

**Example (curl):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"get_strategy_info","arguments":{"strategy":"momentum"}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### quick_analysis

Perform quick market analysis for a symbol with optional GPU acceleration.

**Function Signature:**
```typescript
quick_analysis(params: {
  symbol: string;
  use_gpu?: boolean;
}): Promise<{
  symbol: string;
  current_price: number;
  volume: number;
  trend: string;
  indicators: {
    rsi: number;
    macd: number;
    bollinger_bands: { upper: number; lower: number; };
  };
  recommendation: string;
  confidence: number;
}>;
```

**Parameters:**
- `symbol` (string, required): Stock symbol (e.g., "AAPL")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  symbol: "AAPL",
  current_price: 185.25,
  volume: 52340000,
  trend: "BULLISH",
  indicators: {
    rsi: 62.5,
    macd: 1.25,
    bollinger_bands: { upper: 190.50, lower: 180.00 }
  },
  recommendation: "BUY",
  confidence: 0.75
}
```

**Example (Node.js):**
```javascript
const analysis = await server.callTool('quick_analysis', {
  symbol: 'AAPL',
  use_gpu: true
});

if (analysis.recommendation === 'BUY' && analysis.confidence > 0.7) {
  console.log(`Strong buy signal for ${analysis.symbol}`);
}
```

**Example (curl):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"quick_analysis","arguments":{"symbol":"AAPL","use_gpu":true}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### simulate_trade

Simulate a trading operation with performance tracking.

**Function Signature:**
```typescript
simulate_trade(params: {
  strategy: string;
  symbol: string;
  action: 'buy' | 'sell';
  use_gpu?: boolean;
}): Promise<{
  symbol: string;
  action: string;
  entry_price: number;
  exit_price: number;
  profit_loss: number;
  profit_loss_pct: number;
  execution_time_ms: number;
}>;
```

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Stock symbol
- `action` (string, required): "buy" or "sell"
- `use_gpu` (boolean, optional): Enable GPU acceleration

**Returns:**
```typescript
{
  symbol: "AAPL",
  action: "buy",
  entry_price: 185.25,
  exit_price: 188.50,
  profit_loss: 325.00,
  profit_loss_pct: 1.75,
  execution_time_ms: 45
}
```

**Example (Node.js):**
```javascript
const simulation = await server.callTool('simulate_trade', {
  strategy: 'momentum',
  symbol: 'AAPL',
  action: 'buy',
  use_gpu: true
});

console.log(`Simulated profit: $${simulation.profit_loss}`);
```

---

## Neural Network Tools

### neural_forecast

Generate neural network price forecasts with confidence intervals.

**Function Signature:**
```typescript
neural_forecast(params: {
  symbol: string;
  horizon: number;
  confidence_level?: number;
  model_id?: string;
  use_gpu?: boolean;
}): Promise<{
  symbol: string;
  horizon: number;
  predictions: number[];
  lower_bound: number[];
  upper_bound: number[];
  confidence_level: number;
  model_type: string;
  generation_time_ms: number;
}>;
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `horizon` (number, required): Forecast horizon in periods
- `confidence_level` (number, optional): Confidence level (default: 0.95)
- `model_id` (string, optional): Specific model ID to use
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  symbol: "AAPL",
  horizon: 5,
  predictions: [186.5, 187.2, 188.0, 187.5, 188.8],
  lower_bound: [184.0, 184.5, 185.0, 184.8, 185.5],
  upper_bound: [189.0, 189.9, 191.0, 190.2, 192.1],
  confidence_level: 0.95,
  model_type: "NHITS",
  generation_time_ms: 120
}
```

**Example (Node.js):**
```javascript
const forecast = await server.callTool('neural_forecast', {
  symbol: 'AAPL',
  horizon: 5,
  confidence_level: 0.95,
  use_gpu: true
});

console.log('5-day forecast:');
forecast.predictions.forEach((pred, i) => {
  console.log(`Day ${i+1}: $${pred.toFixed(2)} ` +
    `(${forecast.lower_bound[i].toFixed(2)} - ${forecast.upper_bound[i].toFixed(2)})`);
});
```

**Example (curl):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"neural_forecast","arguments":{"symbol":"AAPL","horizon":5,"use_gpu":true}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### neural_train

Train a neural forecasting model with specified parameters.

**Function Signature:**
```typescript
neural_train(params: {
  data_path: string;
  model_type: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  validation_split?: number;
  use_gpu?: boolean;
}): Promise<{
  model_id: string;
  model_type: string;
  training_time_ms: number;
  final_train_loss: number;
  final_val_loss: number;
  best_epoch: number;
  metrics: {
    mae: number;
    rmse: number;
    mape: number;
  };
}>;
```

**Parameters:**
- `data_path` (string, required): Path to training data
- `model_type` (string, required): Model type ("NHITS", "LSTMAttention", "Transformer")
- `epochs` (number, optional): Number of training epochs (default: 100)
- `batch_size` (number, optional): Batch size (default: 32)
- `learning_rate` (number, optional): Learning rate (default: 0.001)
- `validation_split` (number, optional): Validation split ratio (default: 0.2)
- `use_gpu` (boolean, optional): Enable GPU training (default: true)

**Returns:**
```typescript
{
  model_id: "model_20250114_103045",
  model_type: "NHITS",
  training_time_ms: 45000,
  final_train_loss: 0.0125,
  final_val_loss: 0.0156,
  best_epoch: 78,
  metrics: {
    mae: 1.25,
    rmse: 1.68,
    mape: 0.85
  }
}
```

**Example (Node.js):**
```javascript
const training = await server.callTool('neural_train', {
  data_path: '/data/AAPL_historical.csv',
  model_type: 'NHITS',
  epochs: 100,
  batch_size: 32,
  use_gpu: true
});

console.log(`Model trained: ${training.model_id}`);
console.log(`Best MAE: ${training.metrics.mae}`);
```

---

### neural_evaluate

Evaluate a trained neural model on test data.

**Function Signature:**
```typescript
neural_evaluate(params: {
  model_id: string;
  test_data: string;
  metrics?: string[];
  use_gpu?: boolean;
}): Promise<{
  model_id: string;
  evaluation_metrics: {
    mae: number;
    rmse: number;
    mape: number;
    r2_score: number;
  };
  evaluation_time_ms: number;
}>;
```

**Parameters:**
- `model_id` (string, required): Model ID to evaluate
- `test_data` (string, required): Path to test data
- `metrics` (string[], optional): Metrics to compute (default: ["mae", "rmse", "mape", "r2_score"])
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  model_id: "model_20250114_103045",
  evaluation_metrics: {
    mae: 1.32,
    rmse: 1.75,
    mape: 0.88,
    r2_score: 0.92
  },
  evaluation_time_ms: 1200
}
```

**Example (Node.js):**
```javascript
const evaluation = await server.callTool('neural_evaluate', {
  model_id: 'model_20250114_103045',
  test_data: '/data/AAPL_test.csv',
  use_gpu: true
});

console.log(`Model R²: ${evaluation.evaluation_metrics.r2_score}`);
```

---

### neural_backtest

Run historical backtest of neural model against benchmark.

**Function Signature:**
```typescript
neural_backtest(params: {
  model_id: string;
  start_date: string;
  end_date: string;
  benchmark?: string;
  rebalance_frequency?: string;
  use_gpu?: boolean;
}): Promise<{
  model_id: string;
  start_date: string;
  end_date: string;
  total_return: number;
  benchmark_return: number;
  alpha: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  trades: number;
}>;
```

**Parameters:**
- `model_id` (string, required): Model ID
- `start_date` (string, required): Start date (YYYY-MM-DD)
- `end_date` (string, required): End date (YYYY-MM-DD)
- `benchmark` (string, optional): Benchmark index (default: "sp500")
- `rebalance_frequency` (string, optional): "daily", "weekly", "monthly" (default: "daily")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  model_id: "model_20250114_103045",
  start_date: "2023-01-01",
  end_date: "2023-12-31",
  total_return: 0.245,
  benchmark_return: 0.180,
  alpha: 0.065,
  sharpe_ratio: 1.85,
  max_drawdown: -0.12,
  win_rate: 0.62,
  trades: 127
}
```

**Example (Node.js):**
```javascript
const backtest = await server.callTool('neural_backtest', {
  model_id: 'model_20250114_103045',
  start_date: '2023-01-01',
  end_date: '2023-12-31',
  benchmark: 'sp500',
  use_gpu: true
});

console.log(`Total Return: ${(backtest.total_return * 100).toFixed(2)}%`);
console.log(`Alpha: ${(backtest.alpha * 100).toFixed(2)}%`);
```

---

### neural_optimize

Optimize neural model hyperparameters.

**Function Signature:**
```typescript
neural_optimize(params: {
  model_id: string;
  parameter_ranges: Record<string, any>;
  optimization_metric?: string;
  trials?: number;
  use_gpu?: boolean;
}): Promise<{
  model_id: string;
  best_parameters: Record<string, any>;
  best_score: number;
  optimization_metric: string;
  trials_completed: number;
  optimization_time_ms: number;
}>;
```

**Parameters:**
- `model_id` (string, required): Model ID
- `parameter_ranges` (object, required): Parameter search space
- `optimization_metric` (string, optional): Metric to optimize (default: "mae")
- `trials` (number, optional): Number of optimization trials (default: 100)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  model_id: "model_20250114_103045",
  best_parameters: {
    learning_rate: 0.0015,
    hidden_size: 128,
    num_layers: 3,
    dropout: 0.2
  },
  best_score: 0.98,
  optimization_metric: "mae",
  trials_completed: 100,
  optimization_time_ms: 180000
}
```

**Example (Node.js):**
```javascript
const optimization = await server.callTool('neural_optimize', {
  model_id: 'model_20250114_103045',
  parameter_ranges: {
    learning_rate: { min: 0.0001, max: 0.01 },
    hidden_size: { values: [64, 128, 256] },
    num_layers: { values: [2, 3, 4] }
  },
  optimization_metric: 'mae',
  trials: 50,
  use_gpu: true
});

console.log('Best parameters:', optimization.best_parameters);
```

---

### neural_model_status

Get status and information about neural models.

**Function Signature:**
```typescript
neural_model_status(params?: {
  model_id?: string;
}): Promise<{
  models: Array<{
    model_id: string;
    model_type: string;
    training_status: string;
    created_at: string;
    metrics: Record<string, number>;
  }>;
}>;
```

**Parameters:**
- `model_id` (string, optional): Specific model ID (returns all if not provided)

**Returns:**
```typescript
{
  models: [
    {
      model_id: "model_20250114_103045",
      model_type: "NHITS",
      training_status: "completed",
      created_at: "2025-01-14T10:30:45Z",
      metrics: {
        mae: 1.25,
        rmse: 1.68,
        r2_score: 0.92
      }
    }
  ]
}
```

**Example (Node.js):**
```javascript
// Get all models
const allModels = await server.callTool('neural_model_status', {});
console.log(`Total models: ${allModels.models.length}`);

// Get specific model
const specificModel = await server.callTool('neural_model_status', {
  model_id: 'model_20250114_103045'
});
```

---

## Trading Execution Tools

### execute_trade

Execute live trade with advanced order management.

**Function Signature:**
```typescript
execute_trade(params: {
  strategy: string;
  symbol: string;
  action: 'buy' | 'sell';
  quantity: number;
  order_type?: 'market' | 'limit' | 'stop' | 'stop_limit';
  limit_price?: number;
}): Promise<{
  order_id: string;
  symbol: string;
  action: string;
  quantity: number;
  order_type: string;
  status: string;
  filled_quantity: number;
  filled_price?: number;
  timestamp: string;
}>;
```

**Parameters:**
- `strategy` (string, required): Strategy name
- `symbol` (string, required): Stock symbol
- `action` (string, required): "buy" or "sell"
- `quantity` (number, required): Number of shares
- `order_type` (string, optional): Order type (default: "market")
- `limit_price` (number, optional): Limit price for limit orders

**Returns:**
```typescript
{
  order_id: "ORD_1234567890",
  symbol: "AAPL",
  action: "buy",
  quantity: 100,
  order_type: "market",
  status: "filled",
  filled_quantity: 100,
  filled_price: 185.25,
  timestamp: "2025-01-14T10:30:00Z"
}
```

**Example (Node.js):**
```javascript
const order = await server.callTool('execute_trade', {
  strategy: 'momentum',
  symbol: 'AAPL',
  action: 'buy',
  quantity: 100,
  order_type: 'market'
});

if (order.status === 'filled') {
  console.log(`Order filled at $${order.filled_price}`);
}
```

---

### execute_multi_asset_trade

Execute trades across multiple assets simultaneously.

**Function Signature:**
```typescript
execute_multi_asset_trade(params: {
  trades: Array<{
    symbol: string;
    action: string;
    quantity: number;
    order_type?: string;
  }>;
  strategy: string;
  execute_parallel?: boolean;
  risk_limit?: number;
}): Promise<{
  trades: Array<{
    symbol: string;
    order_id: string;
    status: string;
    filled_price?: number;
  }>;
  total_value: number;
  execution_time_ms: number;
  risk_utilized: number;
}>;
```

**Parameters:**
- `trades` (array, required): Array of trade specifications
- `strategy` (string, required): Strategy name
- `execute_parallel` (boolean, optional): Execute in parallel (default: true)
- `risk_limit` (number, optional): Maximum risk exposure

**Returns:**
```typescript
{
  trades: [
    {
      symbol: "AAPL",
      order_id: "ORD_1234567890",
      status: "filled",
      filled_price: 185.25
    },
    {
      symbol: "GOOGL",
      order_id: "ORD_1234567891",
      status: "filled",
      filled_price: 142.50
    }
  ],
  total_value: 32800.00,
  execution_time_ms: 450,
  risk_utilized: 0.15
}
```

**Example (Node.js):**
```javascript
const multiTrade = await server.callTool('execute_multi_asset_trade', {
  trades: [
    { symbol: 'AAPL', action: 'buy', quantity: 100 },
    { symbol: 'GOOGL', action: 'buy', quantity: 50 }
  ],
  strategy: 'portfolio_rebalance',
  execute_parallel: true,
  risk_limit: 10000
});

console.log(`Executed ${multiTrade.trades.length} trades`);
```

---

## Portfolio Management Tools

### get_portfolio_status

Get current portfolio status with analytics.

**Function Signature:**
```typescript
get_portfolio_status(params?: {
  include_analytics?: boolean;
}): Promise<{
  total_value: number;
  cash: number;
  positions: Array<{
    symbol: string;
    quantity: number;
    avg_price: number;
    current_price: number;
    market_value: number;
    unrealized_pnl: number;
    unrealized_pnl_pct: number;
  }>;
  daily_pnl: number;
  daily_pnl_pct: number;
  analytics?: {
    sharpe_ratio: number;
    sortino_ratio: number;
    max_drawdown: number;
    win_rate: number;
  };
}>;
```

**Parameters:**
- `include_analytics` (boolean, optional): Include performance analytics (default: true)

**Returns:**
```typescript
{
  total_value: 125000.00,
  cash: 25000.00,
  positions: [
    {
      symbol: "AAPL",
      quantity: 100,
      avg_price: 180.00,
      current_price: 185.25,
      market_value: 18525.00,
      unrealized_pnl: 525.00,
      unrealized_pnl_pct: 2.92
    }
  ],
  daily_pnl: 1250.00,
  daily_pnl_pct: 1.01,
  analytics: {
    sharpe_ratio: 1.85,
    sortino_ratio: 2.15,
    max_drawdown: -0.12,
    win_rate: 0.62
  }
}
```

**Example (Node.js):**
```javascript
const portfolio = await server.callTool('get_portfolio_status', {
  include_analytics: true
});

console.log(`Total Portfolio Value: $${portfolio.total_value.toLocaleString()}`);
console.log(`Daily P&L: $${portfolio.daily_pnl.toFixed(2)} (${portfolio.daily_pnl_pct.toFixed(2)}%)`);
```

---

### portfolio_rebalance

Calculate optimal portfolio rebalancing.

**Function Signature:**
```typescript
portfolio_rebalance(params: {
  target_allocations: Record<string, number>;
  current_portfolio?: Record<string, number>;
  rebalance_threshold?: number;
}): Promise<{
  rebalancing_needed: boolean;
  trades: Array<{
    symbol: string;
    action: string;
    quantity: number;
    value: number;
  }>;
  current_allocations: Record<string, number>;
  target_allocations: Record<string, number>;
  deviation: number;
}>;
```

**Parameters:**
- `target_allocations` (object, required): Target allocation percentages
- `current_portfolio` (object, optional): Current holdings
- `rebalance_threshold` (number, optional): Minimum deviation to trigger rebalance (default: 0.05)

**Returns:**
```typescript
{
  rebalancing_needed: true,
  trades: [
    {
      symbol: "AAPL",
      action: "buy",
      quantity: 25,
      value: 4631.25
    },
    {
      symbol: "GOOGL",
      action: "sell",
      quantity: 10,
      value: 1425.00
    }
  ],
  current_allocations: {
    "AAPL": 0.35,
    "GOOGL": 0.42,
    "MSFT": 0.23
  },
  target_allocations: {
    "AAPL": 0.40,
    "GOOGL": 0.35,
    "MSFT": 0.25
  },
  deviation: 0.08
}
```

**Example (Node.js):**
```javascript
const rebalance = await server.callTool('portfolio_rebalance', {
  target_allocations: {
    'AAPL': 0.40,
    'GOOGL': 0.35,
    'MSFT': 0.25
  },
  rebalance_threshold: 0.05
});

if (rebalance.rebalancing_needed) {
  console.log('Rebalancing trades:');
  rebalance.trades.forEach(trade => {
    console.log(`${trade.action} ${trade.quantity} ${trade.symbol}`);
  });
}
```

---

## Risk Management Tools

### risk_analysis

Comprehensive portfolio risk analysis with VaR/CVaR.

**Function Signature:**
```typescript
risk_analysis(params: {
  portfolio: Array<{ symbol: string; weight: number }>;
  time_horizon?: number;
  var_confidence?: number;
  use_monte_carlo?: boolean;
  use_gpu?: boolean;
}): Promise<{
  portfolio_var: number;
  portfolio_cvar: number;
  confidence_level: number;
  individual_var: Record<string, number>;
  correlation_matrix: number[][];
  risk_contribution: Record<string, number>;
  diversification_ratio: number;
}>;
```

**Parameters:**
- `portfolio` (array, required): Array of holdings with weights
- `time_horizon` (number, optional): Risk horizon in days (default: 1)
- `var_confidence` (number, optional): VaR confidence level (default: 0.05)
- `use_monte_carlo` (boolean, optional): Use Monte Carlo simulation (default: true)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  portfolio_var: 12500.00,
  portfolio_cvar: 15800.00,
  confidence_level: 0.05,
  individual_var: {
    "AAPL": 5200.00,
    "GOOGL": 4300.00,
    "MSFT": 3000.00
  },
  correlation_matrix: [
    [1.0, 0.75, 0.68],
    [0.75, 1.0, 0.72],
    [0.68, 0.72, 1.0]
  ],
  risk_contribution: {
    "AAPL": 0.42,
    "GOOGL": 0.34,
    "MSFT": 0.24
  },
  diversification_ratio: 1.35
}
```

**Example (Node.js):**
```javascript
const risk = await server.callTool('risk_analysis', {
  portfolio: [
    { symbol: 'AAPL', weight: 0.40 },
    { symbol: 'GOOGL', weight: 0.35 },
    { symbol: 'MSFT', weight: 0.25 }
  ],
  time_horizon: 1,
  var_confidence: 0.05,
  use_gpu: true
});

console.log(`Portfolio VaR (95%): $${risk.portfolio_var.toLocaleString()}`);
console.log(`Portfolio CVaR: $${risk.portfolio_cvar.toLocaleString()}`);
```

---

### correlation_analysis

Analyze asset correlations with GPU acceleration.

**Function Signature:**
```typescript
correlation_analysis(params: {
  symbols: string[];
  period_days?: number;
  use_gpu?: boolean;
}): Promise<{
  correlation_matrix: number[][];
  symbols: string[];
  period_days: number;
  max_correlation: { pair: string[]; correlation: number };
  min_correlation: { pair: string[]; correlation: number };
}>;
```

**Parameters:**
- `symbols` (string[], required): Array of symbols to analyze
- `period_days` (number, optional): Lookback period in days (default: 90)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: true)

**Returns:**
```typescript
{
  correlation_matrix: [
    [1.0, 0.75, 0.68],
    [0.75, 1.0, 0.72],
    [0.68, 0.72, 1.0]
  ],
  symbols: ["AAPL", "GOOGL", "MSFT"],
  period_days: 90,
  max_correlation: {
    pair: ["AAPL", "GOOGL"],
    correlation: 0.75
  },
  min_correlation: {
    pair: ["AAPL", "MSFT"],
    correlation: 0.68
  }
}
```

**Example (Node.js):**
```javascript
const correlation = await server.callTool('correlation_analysis', {
  symbols: ['AAPL', 'GOOGL', 'MSFT'],
  period_days: 90,
  use_gpu: true
});

console.log('Correlation Matrix:');
correlation.symbols.forEach((symbol, i) => {
  console.log(`${symbol}: ${correlation.correlation_matrix[i].join(', ')}`);
});
```

---

## Sports Betting Tools

### get_sports_events

Get upcoming sports events with comprehensive analysis.

**Function Signature:**
```typescript
get_sports_events(params: {
  sport: string;
  days_ahead?: number;
  use_gpu?: boolean;
}): Promise<{
  sport: string;
  events: Array<{
    event_id: string;
    home_team: string;
    away_team: string;
    start_time: string;
    market_count: number;
    estimated_edge: number;
  }>;
  total_events: number;
}>;
```

**Parameters:**
- `sport` (string, required): Sport type ("NFL", "NBA", "MLB", etc.)
- `days_ahead` (number, optional): Number of days to look ahead (default: 7)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  sport: "NFL",
  events: [
    {
      event_id: "nfl_001",
      home_team: "Chiefs",
      away_team: "Eagles",
      start_time: "2025-01-20T18:00:00Z",
      market_count: 15,
      estimated_edge: 0.045
    }
  ],
  total_events: 12
}
```

**Example (Node.js):**
```javascript
const events = await server.callTool('get_sports_events', {
  sport: 'NFL',
  days_ahead: 7
});

console.log(`Upcoming ${events.sport} events: ${events.total_events}`);
events.events.forEach(event => {
  console.log(`${event.home_team} vs ${event.away_team} - Edge: ${event.estimated_edge}`);
});
```

---

### get_sports_odds

Get real-time sports betting odds with market analysis.

**Function Signature:**
```typescript
get_sports_odds(params: {
  sport: string;
  market_types?: string[];
  regions?: string[];
  use_gpu?: boolean;
}): Promise<{
  sport: string;
  markets: Array<{
    event_id: string;
    market_type: string;
    bookmaker: string;
    odds: Record<string, number>;
    implied_probability: Record<string, number>;
    margin: number;
  }>;
  best_odds: Array<{
    event_id: string;
    outcome: string;
    bookmaker: string;
    odds: number;
  }>;
}>;
```

**Parameters:**
- `sport` (string, required): Sport type
- `market_types` (string[], optional): Market types to include
- `regions` (string[], optional): Bookmaker regions
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  sport: "NFL",
  markets: [
    {
      event_id: "nfl_001",
      market_type: "moneyline",
      bookmaker: "FanDuel",
      odds: {
        "Chiefs": 2.15,
        "Eagles": 1.75
      },
      implied_probability: {
        "Chiefs": 0.465,
        "Eagles": 0.571
      },
      margin: 0.036
    }
  ],
  best_odds: [
    {
      event_id: "nfl_001",
      outcome: "Chiefs",
      bookmaker: "DraftKings",
      odds: 2.20
    }
  ]
}
```

**Example (Node.js):**
```javascript
const odds = await server.callTool('get_sports_odds', {
  sport: 'NFL',
  market_types: ['moneyline', 'spread'],
  regions: ['us']
});

console.log('Best odds:');
odds.best_odds.forEach(bet => {
  console.log(`${bet.outcome} @ ${bet.odds} (${bet.bookmaker})`);
});
```

---

### find_sports_arbitrage

Find arbitrage opportunities in sports betting markets.

**Function Signature:**
```typescript
find_sports_arbitrage(params: {
  sport: string;
  min_profit_margin?: number;
  use_gpu?: boolean;
}): Promise<{
  sport: string;
  opportunities: Array<{
    event_id: string;
    arbitrage_type: string;
    profit_margin: number;
    stakes: Record<string, { bookmaker: string; odds: number; stake: number }>;
    total_stake: number;
    guaranteed_profit: number;
  }>;
  total_opportunities: number;
}>;
```

**Parameters:**
- `sport` (string, required): Sport type
- `min_profit_margin` (number, optional): Minimum profit margin (default: 0.01)
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  sport: "NFL",
  opportunities: [
    {
      event_id: "nfl_001",
      arbitrage_type: "moneyline",
      profit_margin: 0.025,
      stakes: {
        "Chiefs": {
          bookmaker: "FanDuel",
          odds: 2.15,
          stake: 485.00
        },
        "Eagles": {
          bookmaker: "DraftKings",
          odds: 1.80,
          stake: 515.00
        }
      },
      total_stake: 1000.00,
      guaranteed_profit: 25.00
    }
  ],
  total_opportunities: 3
}
```

**Example (Node.js):**
```javascript
const arb = await server.callTool('find_sports_arbitrage', {
  sport: 'NFL',
  min_profit_margin: 0.01
});

console.log(`Found ${arb.total_opportunities} arbitrage opportunities`);
arb.opportunities.forEach(opp => {
  console.log(`Event ${opp.event_id}: ${(opp.profit_margin * 100).toFixed(2)}% profit`);
});
```

---

### calculate_kelly_criterion

Calculate optimal bet size using Kelly Criterion.

**Function Signature:**
```typescript
calculate_kelly_criterion(params: {
  probability: number;
  odds: number;
  bankroll: number;
  confidence?: number;
}): Promise<{
  kelly_percentage: number;
  recommended_stake: number;
  expected_value: number;
  max_loss: number;
  full_kelly: number;
  fractional_kelly: number;
}>;
```

**Parameters:**
- `probability` (number, required): Win probability (0-1)
- `odds` (number, required): Decimal odds
- `bankroll` (number, required): Total bankroll
- `confidence` (number, optional): Confidence level for fractional Kelly (default: 1)

**Returns:**
```typescript
{
  kelly_percentage: 0.045,
  recommended_stake: 1125.00,
  expected_value: 50.63,
  max_loss: -1125.00,
  full_kelly: 0.045,
  fractional_kelly: 0.0225
}
```

**Example (Node.js):**
```javascript
const kelly = await server.callTool('calculate_kelly_criterion', {
  probability: 0.52,
  odds: 2.15,
  bankroll: 25000,
  confidence: 0.5  // Use half-Kelly for safety
});

console.log(`Recommended stake: $${kelly.recommended_stake.toFixed(2)}`);
console.log(`Expected value: $${kelly.expected_value.toFixed(2)}`);
```

---

### execute_sports_bet

Execute sports bet with validation and risk checks.

**Function Signature:**
```typescript
execute_sports_bet(params: {
  market_id: string;
  selection: string;
  stake: number;
  odds: number;
  bet_type?: string;
  validate_only?: boolean;
}): Promise<{
  bet_id: string;
  market_id: string;
  selection: string;
  stake: number;
  odds: number;
  potential_return: number;
  status: string;
  timestamp: string;
  validation: {
    within_limits: boolean;
    kelly_compliant: boolean;
    risk_level: string;
  };
}>;
```

**Parameters:**
- `market_id` (string, required): Market identifier
- `selection` (string, required): Bet selection
- `stake` (number, required): Bet amount
- `odds` (number, required): Odds for the bet
- `bet_type` (string, optional): Bet type (default: "back")
- `validate_only` (boolean, optional): Only validate without placing (default: true)

**Returns:**
```typescript
{
  bet_id: "BET_1234567890",
  market_id: "nfl_001_moneyline",
  selection: "Chiefs",
  stake: 1125.00,
  odds: 2.15,
  potential_return: 2418.75,
  status: "placed",
  timestamp: "2025-01-14T10:30:00Z",
  validation: {
    within_limits: true,
    kelly_compliant: true,
    risk_level: "moderate"
  }
}
```

**Example (Node.js):**
```javascript
// Validate first
const validation = await server.callTool('execute_sports_bet', {
  market_id: 'nfl_001_moneyline',
  selection: 'Chiefs',
  stake: 1125.00,
  odds: 2.15,
  validate_only: true
});

if (validation.validation.within_limits && validation.validation.kelly_compliant) {
  // Place the bet
  const bet = await server.callTool('execute_sports_bet', {
    market_id: 'nfl_001_moneyline',
    selection: 'Chiefs',
    stake: 1125.00,
    odds: 2.15,
    validate_only: false
  });

  console.log(`Bet placed: ${bet.bet_id}`);
}
```

---

## Syndicate Management Tools

### create_syndicate

Create a new investment syndicate for collaborative trading.

**Function Signature:**
```typescript
create_syndicate(params: {
  syndicate_id: string;
  name: string;
  description?: string;
}): Promise<{
  syndicate_id: string;
  name: string;
  description: string;
  created_at: string;
  member_count: number;
  total_bankroll: number;
}>;
```

**Parameters:**
- `syndicate_id` (string, required): Unique syndicate identifier
- `name` (string, required): Syndicate name
- `description` (string, optional): Syndicate description

**Returns:**
```typescript
{
  syndicate_id: "alpha-001",
  name: "Alpha Trading Syndicate",
  description: "Professional sports betting syndicate",
  created_at: "2025-01-14T10:30:00Z",
  member_count: 0,
  total_bankroll: 0
}
```

**Example (Node.js):**
```javascript
const syndicate = await server.callTool('create_syndicate', {
  syndicate_id: 'alpha-001',
  name: 'Alpha Trading Syndicate',
  description: 'Professional sports betting with Kelly Criterion'
});

console.log(`Syndicate created: ${syndicate.name}`);
```

**Example (curl):**
```bash
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"create_syndicate","arguments":{"syndicate_id":"alpha-001","name":"Alpha Trading Syndicate"}},"id":1}' | \
  npx @neural-trader/mcp
```

---

### add_syndicate_member

Add a new member to an investment syndicate.

**Function Signature:**
```typescript
add_syndicate_member(params: {
  syndicate_id: string;
  name: string;
  email: string;
  role: string;
  initial_contribution: number;
}): Promise<{
  member_id: string;
  syndicate_id: string;
  name: string;
  email: string;
  role: string;
  contribution: number;
  ownership_pct: number;
  joined_at: string;
}>;
```

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `name` (string, required): Member name
- `email` (string, required): Member email
- `role` (string, required): Member role
- `initial_contribution` (number, required): Initial capital contribution

**Returns:**
```typescript
{
  member_id: "member_001",
  syndicate_id: "alpha-001",
  name: "Alice Johnson",
  email: "alice@example.com",
  role: "senior_analyst",
  contribution: 25000,
  ownership_pct: 25.0,
  joined_at: "2025-01-14T10:30:00Z"
}
```

**Example (Node.js):**
```javascript
const member = await server.callTool('add_syndicate_member', {
  syndicate_id: 'alpha-001',
  name: 'Alice Johnson',
  email: 'alice@example.com',
  role: 'senior_analyst',
  initial_contribution: 25000
});

console.log(`Member added: ${member.name} (${member.ownership_pct}% ownership)`);
```

---

### get_syndicate_status

Get current syndicate status and statistics.

**Function Signature:**
```typescript
get_syndicate_status(params: {
  syndicate_id: string;
}): Promise<{
  syndicate_id: string;
  name: string;
  member_count: number;
  total_bankroll: number;
  available_capital: number;
  active_bets: number;
  total_bets_placed: number;
  total_profit_loss: number;
  roi: number;
  sharpe_ratio: number;
  win_rate: number;
  kelly_compliance: number;
}>;
```

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID

**Returns:**
```typescript
{
  syndicate_id: "alpha-001",
  name: "Alpha Trading Syndicate",
  member_count: 5,
  total_bankroll: 125000,
  available_capital: 95000,
  active_bets: 12,
  total_bets_placed: 247,
  total_profit_loss: 18500,
  roi: 0.148,
  sharpe_ratio: 1.65,
  win_rate: 0.58,
  kelly_compliance: 0.92
}
```

**Example (Node.js):**
```javascript
const status = await server.callTool('get_syndicate_status', {
  syndicate_id: 'alpha-001'
});

console.log(`Syndicate: ${status.name}`);
console.log(`Total Bankroll: $${status.total_bankroll.toLocaleString()}`);
console.log(`ROI: ${(status.roi * 100).toFixed(2)}%`);
console.log(`Win Rate: ${(status.win_rate * 100).toFixed(2)}%`);
```

---

### allocate_syndicate_funds

Allocate syndicate funds using Kelly Criterion.

**Function Signature:**
```typescript
allocate_syndicate_funds(params: {
  syndicate_id: string;
  opportunities: Array<{
    id: string;
    sport: string;
    odds: number;
    probability: number;
    edge: number;
  }>;
  strategy?: string;
}): Promise<{
  syndicate_id: string;
  strategy: string;
  allocations: Array<{
    opportunity_id: string;
    recommended_stake: number;
    kelly_percentage: number;
    expected_value: number;
  }>;
  total_allocated: number;
  remaining_capital: number;
  portfolio_kelly: number;
}>;
```

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `opportunities` (array, required): Betting opportunities
- `strategy` (string, optional): Allocation strategy (default: "kelly_criterion")

**Returns:**
```typescript
{
  syndicate_id: "alpha-001",
  strategy: "kelly_criterion",
  allocations: [
    {
      opportunity_id: "nfl_001",
      recommended_stake: 2500.00,
      kelly_percentage: 0.02,
      expected_value: 125.00
    }
  ],
  total_allocated: 7500.00,
  remaining_capital: 87500.00,
  portfolio_kelly: 0.06
}
```

**Example (Node.js):**
```javascript
const allocation = await server.callTool('allocate_syndicate_funds', {
  syndicate_id: 'alpha-001',
  opportunities: [
    {
      id: 'nfl_001',
      sport: 'NFL',
      odds: 2.15,
      probability: 0.52,
      edge: 0.045
    },
    {
      id: 'nba_002',
      sport: 'NBA',
      odds: 1.95,
      probability: 0.55,
      edge: 0.055
    }
  ],
  strategy: 'kelly_criterion'
});

console.log('Recommended allocations:');
allocation.allocations.forEach(alloc => {
  console.log(`${alloc.opportunity_id}: $${alloc.recommended_stake} (EV: $${alloc.expected_value})`);
});
```

---

### distribute_syndicate_profits

Distribute profits among syndicate members.

**Function Signature:**
```typescript
distribute_syndicate_profits(params: {
  syndicate_id: string;
  total_profit: number;
  model?: string;
}): Promise<{
  syndicate_id: string;
  total_profit: number;
  distribution_model: string;
  distributions: Array<{
    member_id: string;
    amount: number;
    percentage: number;
    basis: string;
  }>;
  timestamp: string;
}>;
```

**Parameters:**
- `syndicate_id` (string, required): Syndicate ID
- `total_profit` (number, required): Total profit to distribute
- `model` (string, optional): Distribution model (default: "hybrid")

**Returns:**
```typescript
{
  syndicate_id: "alpha-001",
  total_profit: 50000,
  distribution_model: "hybrid",
  distributions: [
    {
      member_id: "member_001",
      amount: 28500,
      percentage: 57.0,
      basis: "70% capital, 30% performance"
    }
  ],
  timestamp: "2025-01-14T10:30:00Z"
}
```

**Example (Node.js):**
```javascript
const distribution = await server.callTool('distribute_syndicate_profits', {
  syndicate_id: 'alpha-001',
  total_profit: 50000,
  model: 'hybrid'
});

console.log(`Distributing $${distribution.total_profit.toLocaleString()}`);
distribution.distributions.forEach(dist => {
  console.log(`${dist.member_id}: $${dist.amount.toLocaleString()} (${dist.percentage}%)`);
});
```

---

## News & Sentiment Tools

### analyze_news

AI sentiment analysis of market news.

**Function Signature:**
```typescript
analyze_news(params: {
  symbol: string;
  lookback_hours?: number;
  sentiment_model?: string;
  use_gpu?: boolean;
}): Promise<{
  symbol: string;
  sentiment_score: number;
  sentiment_label: string;
  article_count: number;
  key_topics: string[];
  market_impact: string;
  confidence: number;
}>;
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `lookback_hours` (number, optional): Hours to look back (default: 24)
- `sentiment_model` (string, optional): Sentiment model (default: "enhanced")
- `use_gpu` (boolean, optional): Enable GPU acceleration (default: false)

**Returns:**
```typescript
{
  symbol: "AAPL",
  sentiment_score: 0.75,
  sentiment_label: "POSITIVE",
  article_count: 42,
  key_topics: ["earnings", "product launch", "market share"],
  market_impact: "BULLISH",
  confidence: 0.85
}
```

**Example (Node.js):**
```javascript
const news = await server.callTool('analyze_news', {
  symbol: 'AAPL',
  lookback_hours: 24,
  use_gpu: true
});

console.log(`Sentiment: ${news.sentiment_label} (${news.sentiment_score.toFixed(2)})`);
console.log(`Articles analyzed: ${news.article_count}`);
console.log(`Market impact: ${news.market_impact}`);
```

---

### get_news_sentiment

Get real-time news sentiment for a symbol.

**Function Signature:**
```typescript
get_news_sentiment(params: {
  symbol: string;
  sources?: string[];
}): Promise<{
  symbol: string;
  overall_sentiment: number;
  sources: Array<{
    source: string;
    sentiment: number;
    article_count: number;
  }>;
  timestamp: string;
}>;
```

**Parameters:**
- `symbol` (string, required): Stock symbol
- `sources` (string[], optional): News sources to include

**Returns:**
```typescript
{
  symbol: "AAPL",
  overall_sentiment: 0.68,
  sources: [
    {
      source: "Bloomberg",
      sentiment: 0.72,
      article_count: 15
    },
    {
      source: "Reuters",
      sentiment: 0.64,
      article_count: 12
    }
  ],
  timestamp: "2025-01-14T10:30:00Z"
}
```

**Example (Node.js):**
```javascript
const sentiment = await server.callTool('get_news_sentiment', {
  symbol: 'AAPL',
  sources: ['Bloomberg', 'Reuters', 'WSJ']
});

console.log(`Overall sentiment: ${sentiment.overall_sentiment.toFixed(2)}`);
sentiment.sources.forEach(source => {
  console.log(`${source.source}: ${source.sentiment.toFixed(2)} (${source.article_count} articles)`);
});
```

---

### fetch_filtered_news

Fetch news with advanced filtering options.

**Function Signature:**
```typescript
fetch_filtered_news(params: {
  symbols: string[];
  limit?: number;
  relevance_threshold?: number;
  sentiment_filter?: string;
}): Promise<{
  articles: Array<{
    id: string;
    title: string;
    source: string;
    published_at: string;
    sentiment: number;
    relevance: number;
    symbols: string[];
  }>;
  total_count: number;
}>;
```

**Parameters:**
- `symbols` (string[], required): Symbols to filter
- `limit` (number, optional): Maximum articles (default: 50)
- `relevance_threshold` (number, optional): Minimum relevance (default: 0.5)
- `sentiment_filter` (string, optional): Sentiment filter ("positive", "negative", "neutral")

**Returns:**
```typescript
{
  articles: [
    {
      id: "article_001",
      title: "Apple Announces Record Earnings",
      source: "Bloomberg",
      published_at: "2025-01-14T09:00:00Z",
      sentiment: 0.85,
      relevance: 0.92,
      symbols: ["AAPL"]
    }
  ],
  total_count: 42
}
```

**Example (Node.js):**
```javascript
const news = await server.callTool('fetch_filtered_news', {
  symbols: ['AAPL', 'GOOGL'],
  limit: 20,
  relevance_threshold: 0.7,
  sentiment_filter: 'positive'
});

console.log(`Found ${news.total_count} relevant articles`);
news.articles.forEach(article => {
  console.log(`${article.title} - Sentiment: ${article.sentiment.toFixed(2)}`);
});
```

---

### get_news_trends

Analyze news trends over multiple time intervals.

**Function Signature:**
```typescript
get_news_trends(params: {
  symbols: string[];
  time_intervals?: number[];
}): Promise<{
  symbols: string[];
  trends: Array<{
    interval_hours: number;
    sentiment_avg: number;
    sentiment_change: number;
    article_count: number;
    volume_change: number;
  }>;
  overall_trend: string;
}>;
```

**Parameters:**
- `symbols` (string[], required): Symbols to analyze
- `time_intervals` (number[], optional): Time intervals in hours (default: [1, 6, 24])

**Returns:**
```typescript
{
  symbols: ["AAPL"],
  trends: [
    {
      interval_hours: 1,
      sentiment_avg: 0.75,
      sentiment_change: 0.05,
      article_count: 8,
      volume_change: 1.2
    },
    {
      interval_hours: 6,
      sentiment_avg: 0.68,
      sentiment_change: -0.02,
      article_count: 32,
      volume_change: 0.95
    },
    {
      interval_hours: 24,
      sentiment_avg: 0.70,
      sentiment_change: 0.10,
      article_count: 120,
      volume_change: 1.15
    }
  ],
  overall_trend: "IMPROVING"
}
```

**Example (Node.js):**
```javascript
const trends = await server.callTool('get_news_trends', {
  symbols: ['AAPL'],
  time_intervals: [1, 6, 24]
});

console.log(`Overall trend: ${trends.overall_trend}`);
trends.trends.forEach(trend => {
  console.log(`${trend.interval_hours}h: ` +
    `Sentiment ${trend.sentiment_avg.toFixed(2)} ` +
    `(${trend.sentiment_change > 0 ? '+' : ''}${trend.sentiment_change.toFixed(2)})`);
});
```

---

## System Tools

### run_benchmark

Run comprehensive benchmarks for strategy performance.

**Function Signature:**
```typescript
run_benchmark(params: {
  strategy: string;
  benchmark_type?: string;
  use_gpu?: boolean;
}): Promise<{
  strategy: string;
  benchmark_type: string;
  results: {
    execution_time_ms: number;
    throughput: number;
    memory_usage_mb: number;
    cpu_utilization: number;
    gpu_utilization?: number;
  };
  comparison: {
    vs_baseline: number;
    vs_best: number;
  };
}>;
```

**Parameters:**
- `strategy` (string, required): Strategy name
- `benchmark_type` (string, optional): Benchmark type (default: "performance")
- `use_gpu` (boolean, optional): Enable GPU benchmarking (default: true)

**Returns:**
```typescript
{
  strategy: "momentum",
  benchmark_type: "performance",
  results: {
    execution_time_ms: 450,
    throughput: 2222,
    memory_usage_mb: 128,
    cpu_utilization: 0.45,
    gpu_utilization: 0.78
  },
  comparison: {
    vs_baseline: 1.85,
    vs_best: 0.92
  }
}
```

**Example (Node.js):**
```javascript
const benchmark = await server.callTool('run_benchmark', {
  strategy: 'momentum',
  benchmark_type: 'performance',
  use_gpu: true
});

console.log(`Execution time: ${benchmark.results.execution_time_ms}ms`);
console.log(`GPU utilization: ${(benchmark.results.gpu_utilization * 100).toFixed(1)}%`);
console.log(`Performance vs baseline: ${benchmark.comparison.vs_baseline.toFixed(2)}x`);
```

---

## Broker Integration Tools

### list_broker_types

List all available broker types.

**Function Signature:**
```typescript
list_broker_types(): Promise<{
  brokers: string[];
  count: number;
}>;
```

**Parameters:** None

**Returns:**
```typescript
{
  brokers: ["alpaca", "interactive_brokers", "tastytrade", "tradier", "wealthsimple"],
  count: 5
}
```

**Example (Node.js):**
```javascript
const brokers = await server.callTool('list_broker_types', {});
console.log(`Available brokers: ${brokers.brokers.join(', ')}`);
```

---

### validate_broker_config

Validate broker configuration.

**Function Signature:**
```typescript
validate_broker_config(params: {
  broker_type: string;
  api_key: string;
  api_secret: string;
  base_url?: string;
  paper_trading: boolean;
}): Promise<{
  valid: boolean;
  broker_type: string;
  connectivity: boolean;
  errors?: string[];
}>;
```

**Parameters:**
- `broker_type` (string, required): Broker type
- `api_key` (string, required): API key
- `api_secret` (string, required): API secret
- `base_url` (string, optional): Custom base URL
- `paper_trading` (boolean, required): Paper trading mode

**Returns:**
```typescript
{
  valid: true,
  broker_type: "alpaca",
  connectivity: true,
  errors: []
}
```

**Example (Node.js):**
```javascript
const validation = await server.callTool('validate_broker_config', {
  broker_type: 'alpaca',
  api_key: process.env.ALPACA_API_KEY,
  api_secret: process.env.ALPACA_API_SECRET,
  paper_trading: true
});

if (validation.valid) {
  console.log('Broker configuration is valid');
} else {
  console.error('Validation errors:', validation.errors);
}
```

---

## Error Handling

All MCP tools return errors in the standard JSON-RPC 2.0 error format:

```typescript
{
  jsonrpc: "2.0",
  error: {
    code: number;
    message: string;
    data?: any;
  },
  id: string | number;
}
```

**Common Error Codes:**
- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
- `-32000`: Server error

**Example Error Response:**
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "param": "symbol",
      "reason": "Symbol is required"
    }
  },
  "id": 1
}
```

**Error Handling (Node.js):**
```javascript
try {
  const result = await server.callTool('quick_analysis', {
    // Missing required 'symbol' parameter
  });
} catch (error) {
  if (error.code === -32602) {
    console.error(`Invalid parameters: ${error.message}`);
  } else {
    console.error(`Error: ${error.message}`);
  }
}
```

---

## Performance Considerations

### GPU Acceleration

Most computationally intensive tools support GPU acceleration:

```javascript
// Enable GPU for neural forecasting
const forecast = await server.callTool('neural_forecast', {
  symbol: 'AAPL',
  horizon: 5,
  use_gpu: true  // 10-100x faster
});

// Enable GPU for risk analysis
const risk = await server.callTool('risk_analysis', {
  portfolio: [...],
  use_gpu: true  // Parallel VaR/CVaR calculations
});
```

### Batch Operations

Use batch operations for better performance:

```javascript
// Instead of sequential calls
const symbols = ['AAPL', 'GOOGL', 'MSFT'];
const results = await Promise.all(
  symbols.map(symbol =>
    server.callTool('quick_analysis', { symbol, use_gpu: true })
  )
);
```

### Connection Pooling

Limit concurrent connections:

```javascript
const server = new McpServer({
  transport: 'http',
  maxConnections: 50
});
```

---

## Version Information

- **API Version**: 1.0.0
- **Protocol**: MCP (Model Context Protocol) 0.5.0
- **NAPI-RS Version**: 2.16.0
- **Total Tools**: 107
- **Syndicate Tools**: 15
- **Neural Tools**: 8
- **Sports Betting Tools**: 10

---

## Related Documentation

- [MCP Integration Guide](/workspaces/neural-trader/neural-trader-rust/docs/guides/MCP_INTEGRATION.md)
- [NAPI Development Guide](/workspaces/neural-trader/neural-trader-rust/docs/development/NAPI_DEVELOPMENT.md)
- [Migration Guide](/workspaces/neural-trader/neural-trader-rust/docs/guides/PYTHON_TO_NAPI_MIGRATION.md)
- [Examples](/workspaces/neural-trader/neural-trader-rust/docs/examples/)

---

**Last Updated**: 2025-01-14
**Maintained By**: Neural Trader Team
