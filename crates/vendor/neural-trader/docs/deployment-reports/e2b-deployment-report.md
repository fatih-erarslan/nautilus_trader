# E2B Sandbox Deployment Report
2025-11-14T04:57:11.478Z

## Deployment Overview

**Deployment ID:** `neural-trader-1763096012878`
**Topology:** MESH
**Status:** ✅ COMPLETED
**Success Rate:** 100.0%

## Summary Statistics

- **Total Sandboxes:** 5
- **Successful Deployments:** 5
- **Failed Deployments:** 0
- **Deployed At:** 2025-11-14T04:57:11.478Z

## Resource Allocation

### Total Resources
- **CPU Cores:** 14 cores
- **Memory:** 6656 MB (6.5 GB)

### Averages per Sandbox
- **CPU:** 2.8 cores
- **Memory:** 1331 MB

## Cost Estimation

- **CPU Cost:** $102.2/month
- **Memory Cost:** $23.73/month
- **Total Estimated Cost:** $125.93/month

*Estimate based on continuous 24/7 operation*

## Trading Coverage

- **Total Symbols:** 10
- **Symbols:** AAPL, ALL, GLD, IWM, NVDA, QQQ, SLV, SPY, TLT, TSLA

## Deployed Strategies


### 1. momentum (`momentum_trader`)

- **Sandbox ID:** `sb_bd4479c5c87e2c07003e91701f110bea`
- **Status:** ✅ Running
- **Symbols:** SPY, QQQ, IWM
- **Resources:**
  - CPU: 2 cores
  - Memory: 1024 MB
  - Timeout: 3600s (1.0h)
- **Created:** 2025-11-14T04:57:09.473Z
- **URL:** https://e2b.dev/sandboxes/sb_bd4479c5c87e2c07003e91701f110bea
- **WebSocket:** wss://e2b.dev/ws/sb_bd4479c5c87e2c07003e91701f110bea
- **API:** https://api.e2b.dev/v1/sandboxes/sb_bd4479c5c87e2c07003e91701f110bea

**Environment:**
- Node.js: 18.19.0
- NPM: 10.2.3
- OS: ubuntu-22.04



### 2. neural_forecast (`neural_forecaster`)

- **Sandbox ID:** `sb_e2a4a73d98a78a012772fab2623cd7b9`
- **Status:** ✅ Running
- **Symbols:** AAPL, TSLA, NVDA
- **Resources:**
  - CPU: 4 cores
  - Memory: 2048 MB
  - Timeout: 3600s (1.0h)
- **Created:** 2025-11-14T04:57:09.974Z
- **URL:** https://e2b.dev/sandboxes/sb_e2a4a73d98a78a012772fab2623cd7b9
- **WebSocket:** wss://e2b.dev/ws/sb_e2a4a73d98a78a012772fab2623cd7b9
- **API:** https://api.e2b.dev/v1/sandboxes/sb_e2a4a73d98a78a012772fab2623cd7b9

**Environment:**
- Node.js: 18.19.0
- NPM: 10.2.3
- OS: ubuntu-22.04



### 3. mean_reversion (`mean_reversion_trader`)

- **Sandbox ID:** `sb_a99a8f14bb8d4023cbc8e6e694682248`
- **Status:** ✅ Running
- **Symbols:** GLD, SLV, TLT
- **Resources:**
  - CPU: 2 cores
  - Memory: 1024 MB
  - Timeout: 3600s (1.0h)
- **Created:** 2025-11-14T04:57:10.475Z
- **URL:** https://e2b.dev/sandboxes/sb_a99a8f14bb8d4023cbc8e6e694682248
- **WebSocket:** wss://e2b.dev/ws/sb_a99a8f14bb8d4023cbc8e6e694682248
- **API:** https://api.e2b.dev/v1/sandboxes/sb_a99a8f14bb8d4023cbc8e6e694682248

**Environment:**
- Node.js: 18.19.0
- NPM: 10.2.3
- OS: ubuntu-22.04



### 4. risk_manager (`risk_manager`)

- **Sandbox ID:** `sb_de91d06f01835ab53143351c3c1634b5`
- **Status:** ✅ Running
- **Symbols:** ALL
- **Resources:**
  - CPU: 2 cores
  - Memory: 512 MB
  - Timeout: 7200s (2.0h)
- **Created:** 2025-11-14T04:57:10.976Z
- **URL:** https://e2b.dev/sandboxes/sb_de91d06f01835ab53143351c3c1634b5
- **WebSocket:** wss://e2b.dev/ws/sb_de91d06f01835ab53143351c3c1634b5
- **API:** https://api.e2b.dev/v1/sandboxes/sb_de91d06f01835ab53143351c3c1634b5

**Environment:**
- Node.js: 18.19.0
- NPM: 10.2.3
- OS: ubuntu-22.04



### 5. portfolio_optimizer (`portfolio_optimizer`)

- **Sandbox ID:** `sb_f02f5a06291a77c1472ef0ab9b20f3f1`
- **Status:** ✅ Running
- **Symbols:** ALL
- **Resources:**
  - CPU: 4 cores
  - Memory: 2048 MB
  - Timeout: 7200s (2.0h)
- **Created:** 2025-11-14T04:57:11.477Z
- **URL:** https://e2b.dev/sandboxes/sb_f02f5a06291a77c1472ef0ab9b20f3f1
- **WebSocket:** wss://e2b.dev/ws/sb_f02f5a06291a77c1472ef0ab9b20f3f1
- **API:** https://api.e2b.dev/v1/sandboxes/sb_f02f5a06291a77c1472ef0ab9b20f3f1

**Environment:**
- Node.js: 18.19.0
- NPM: 10.2.3
- OS: ubuntu-22.04



## Coordination Configuration

- **QUIC Enabled:** Yes
- **Sync Interval:** 5000ms
- **Distributed Memory:** Yes

## Strategy Dependencies

### Base Dependencies (All Strategies)
```json
{
  "@alpacahq/alpaca-trade-api": "^3.0.0",
  "dotenv": "^16.0.0",
  "axios": "^1.6.0"
}
```

### Neural Forecaster
```json
{
  "@tensorflow/tfjs-node": "^4.15.0",
  "mathjs": "^12.0.0"
}
```

### Momentum & Mean Reversion Traders
```json
{
  "technical-indicators": "^3.1.0"
}
```

### Risk Manager & Portfolio Optimizer
```json
{
  "mathjs": "^12.0.0",
  "optimization-js": "^2.0.0"
}
```

## Environment Variables

Each sandbox is configured with:
- `ALPACA_API_KEY`: Alpaca trading API key
- `ALPACA_API_SECRET`: Alpaca trading API secret
- `ALPACA_BASE_URL`: Paper trading endpoint
- `ANTHROPIC_API_KEY`: Claude AI API key
- `STRATEGY_NAME`: Strategy identifier
- `TRADING_SYMBOLS`: Comma-separated symbol list
- `NODE_ENV`: Production mode

## Monitoring & Health Checks

All sandboxes are configured with:
- **Auto-restart:** Enabled
- **Health check interval:** 30 seconds
- **Log retention:** 7 days

## Next Steps

1. **Verify Sandbox Status:**
   ```bash
   node scripts/deployment/verify-sandboxes.js
   ```

2. **Monitor Performance:**
   ```bash
   npx neural-trader monitor --deployment neural-trader-1763096012878
   ```

3. **View Logs:**
   ```bash
   npx neural-trader logs --sandbox <sandbox-id>
   ```

4. **Scale Resources:**
   ```bash
   npx neural-trader scale --sandbox <sandbox-id> --cpu 4 --memory 2048
   ```

## Support

For issues or questions:
- GitHub: https://github.com/ruvnet/neural-trader/issues
- E2B Docs: https://e2b.dev/docs
- Deployment ID: `neural-trader-1763096012878`

---

**Generated:** 2025-11-14T04:57:11.478Z
**Report Version:** 1.0.0
**Deployment System:** E2B Sandbox Deployer v1.0.0
