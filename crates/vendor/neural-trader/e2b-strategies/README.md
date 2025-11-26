# E2B Trading Strategies

Production-ready trading strategy implementations designed for isolated E2B sandbox deployment.

## 🚀 Quick Start

### Prerequisites

```bash
# Set Alpaca API credentials
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

### Deploy with Neural Trader MCP

```javascript
// Deploy momentum strategy
npx neural-trader create_e2b_sandbox({
  name: "momentum-strategy",
  template: "node",
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "3000"
  }
})
```

## 📊 Available Strategies

### 1. Momentum Trading
**Directory**: `/e2b-strategies/momentum`
**Port**: 3000

Trades based on 5-minute price momentum across liquid ETFs.

**Features**:
- Symbols: SPY, QQQ, IWM
- Buy signal: momentum > 2%
- Sell signal: momentum < -2%
- Fixed position size: 10 shares

**Endpoints**:
- `GET /health` - Health check
- `GET /status` - Current positions and account
- `POST /execute` - Manual strategy execution

### 2. Neural Forecasting
**Directory**: `/e2b-strategies/neural-forecast`
**Port**: 3001

LSTM-based price prediction with confidence-driven position sizing.

**Features**:
- Symbols: AAPL, TSLA, NVDA
- LSTM model with 60-period lookback
- Dynamic position sizing (5-50 shares)
- Confidence threshold: 70%
- Automatic model retraining

**Endpoints**:
- `GET /health` - Health check with model status
- `GET /status` - Positions and loaded models
- `POST /execute` - Run prediction and trade
- `POST /retrain/:symbol` - Retrain model for symbol

### 3. Mean Reversion
**Directory**: `/e2b-strategies/mean-reversion`
**Port**: 3002

Statistical arbitrage using z-score mean reversion on commodity ETFs.

**Features**:
- Symbols: GLD, SLV, TLT
- 20-period SMA and standard deviation
- Entry: z-score < -2 (buy) or > +2 (sell)
- Exit: |z-score| < 0.5
- Max position: 100 shares per symbol

**Endpoints**:
- `GET /health` - Health check
- `GET /status` - Positions and statistics
- `GET /statistics/:symbol` - Z-score and metrics
- `POST /execute` - Execute strategy

### 4. Risk Manager
**Directory**: `/e2b-strategies/risk-manager`
**Port**: 3003

Portfolio-wide risk monitoring and enforcement service.

**Features**:
- VaR and CVaR calculation (95% confidence)
- Maximum drawdown tracking
- Automatic stop-loss (2% per trade)
- Sharpe ratio monitoring
- Real-time risk alerts

**Endpoints**:
- `GET /health` - Service health
- `GET /metrics` - Current risk metrics
- `GET /alerts` - Active risk alerts
- `POST /monitor` - Manual risk check

### 5. Portfolio Optimizer
**Directory**: `/e2b-strategies/portfolio-optimizer`
**Port**: 3004

Sharpe ratio and risk parity portfolio optimization with automatic rebalancing.

**Features**:
- Symbols: SPY, QQQ, IWM, GLD, TLT, AAPL, TSLA
- Methods: Sharpe optimization, risk parity
- 60-day lookback for statistics
- Rebalance trigger: 5% deviation
- Daily optimization cycle

**Endpoints**:
- `GET /health` - Service health
- `GET /status` - Current and target allocations
- `POST /optimize` - Run optimization
- `POST /rebalance` - Execute rebalancing

## 🏗️ Architecture

```
e2b-strategies/
├── momentum/
│   ├── index.js           # Strategy implementation
│   └── package.json       # Dependencies
├── neural-forecast/
│   ├── index.js           # LSTM forecasting
│   └── package.json       # TensorFlow dependencies
├── mean-reversion/
│   ├── index.js           # Z-score strategy
│   └── package.json       # Dependencies
├── risk-manager/
│   ├── index.js           # Risk monitoring
│   └── package.json       # Dependencies
└── portfolio-optimizer/
    ├── index.js           # Portfolio optimization
    └── package.json       # Dependencies
```

## 📖 Documentation

Comprehensive guides available in `/docs/e2b-deployment/`:

1. **deployment-guide.md** - Complete deployment instructions
2. **integration-tests.md** - Testing patterns and examples
3. **api-integration-patterns.md** - Alpaca API best practices

## 🔧 Local Development

```bash
# Install dependencies for a strategy
cd e2b-strategies/momentum
npm install

# Run locally
npm start

# Development mode with auto-reload
npm run dev
```

## 🧪 Testing

```bash
# Run integration tests
npm test -- --testPathPattern=e2b

# Test specific strategy
npm test -- momentum.integration.test.js

# With coverage
npm test -- --coverage
```

## 🚀 Deployment Examples

### Deploy All Strategies

```javascript
const strategies = ['momentum', 'neural-forecast', 'mean-reversion', 'risk-manager', 'portfolio-optimizer'];

for (const strategy of strategies) {
  const sandbox = await createE2BSandbox({
    name: `${strategy}-strategy`,
    template: 'node',
    env_vars: {
      ALPACA_API_KEY: process.env.ALPACA_API_KEY,
      ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY
    }
  });

  await uploadStrategy(sandbox, strategy);
  await startStrategy(sandbox);
}
```

### Multi-Instance Scaling

```javascript
// Scale momentum strategy to 3 instances
npx neural-trader scale_e2b_deployment({
  deployment_id: "momentum-strategy",
  instance_count: 3,
  auto_scale: true
})
```

### With Flow-Nexus Platform

```javascript
// Deploy neural forecast with cloud monitoring
mcp__flow-nexus__sandbox_create({
  template: "node",
  name: "neural-forecast-production",
  env_vars: {
    ALPACA_API_KEY: process.env.ALPACA_API_KEY,
    ALPACA_SECRET_KEY: process.env.ALPACA_SECRET_KEY,
    PORT: "3001"
  },
  install_packages: ["@alpacahq/alpaca-trade-api", "@tensorflow/tfjs-node", "express"]
})

// Subscribe to real-time execution stream
mcp__flow-nexus__execution_stream_subscribe({
  sandbox_id: "neural-forecast-production",
  stream_type: "claude-flow-swarm"
})
```

## 📊 Monitoring

### Health Check All Strategies

```bash
for port in 3000 3001 3002 3003 3004; do
  curl http://localhost:$port/health
done
```

### Aggregate Status Dashboard

```javascript
const dashboardData = await Promise.all([
  fetch('http://momentum:3000/status'),
  fetch('http://neural-forecast:3001/status'),
  fetch('http://mean-reversion:3002/status'),
  fetch('http://risk-manager:3003/metrics'),
  fetch('http://portfolio-optimizer:3004/status')
]);
```

## 🔒 Security

1. **Never commit API keys** - Use environment variables
2. **Use paper trading** - Start with Alpaca paper account
3. **Implement rate limiting** - Built-in exponential backoff
4. **Monitor sandbox resources** - Auto-shutdown on excessive usage
5. **Validate all trades** - Pre-trade risk checks included

## 📈 Performance

- **Momentum**: ~5 second execution per cycle
- **Neural Forecast**: ~15 seconds (includes model inference)
- **Mean Reversion**: ~5 seconds per cycle
- **Risk Manager**: ~10 seconds for full portfolio analysis
- **Portfolio Optimizer**: ~30 seconds for optimization

## 🐛 Troubleshooting

### Common Issues

**API Rate Limits**
- Built-in exponential backoff handles this automatically
- Reduce execution frequency if needed

**Market Closed**
- All strategies check market hours
- Execution skipped when market closed

**TensorFlow Memory**
- Neural forecast requires 2GB+ RAM
- Increase sandbox memory allocation

**Insufficient Data**
- First execution may fail (normal)
- Strategies need historical data to calculate signals

## 💰 Cost Estimates

### E2B Sandbox Costs
- **Single strategy**: ~$0.10/hour
- **All 5 strategies**: ~$0.50/hour
- **Monthly (24/7)**: ~$360

### Optimization
- Run only during market hours: ~$125/month
- Use Flow-Nexus auto-scaling: ~$80/month

## 🤝 Contributing

Contributions welcome! To add a new strategy:

1. Create directory in `/e2b-strategies/your-strategy`
2. Implement `index.js` with standard endpoints
3. Add `package.json` with dependencies
4. Follow existing patterns for logging and error handling
5. Add tests in `/tests/integration/your-strategy.test.js`
6. Update this README

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Support

- **Documentation**: `/docs/e2b-deployment/`
- **Neural Trader Docs**: `/docs`
- **Issues**: GitHub Issues
- **E2B Docs**: https://e2b.dev/docs
- **Alpaca Docs**: https://alpaca.markets/docs
