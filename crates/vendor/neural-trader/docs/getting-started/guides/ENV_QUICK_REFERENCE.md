# Environment Variables Quick Reference

## ✅ Currently Implemented Variables

### Polymarket Integration (ONLY required API)
```bash
POLYMARKET_API_KEY=your-api-key          # For real prediction market data
POLYMARKET_PRIVATE_KEY=your-private-key  # For order signing
```

### Optional Settings (have defaults)
```bash
POLYMARKET_ENVIRONMENT=production        # or staging, development
POLYMARKET_RATE_LIMIT=100               # requests per minute
CUDA_VISIBLE_DEVICES=0                  # GPU device selection
```

## 🎯 What Works WITHOUT API Keys

- **News Sentiment**: Yahoo Finance, Reuters (built-in)
- **Neural Forecasting**: NHITS, NBEATSx models
- **Trading Strategies**: All 4 strategies in demo mode
- **GPU Acceleration**: Auto-detected if available
- **27 MCP Tools**: All accessible via Claude Code

## 📊 Current System Capabilities

- **Trading Mode**: Demo only (no real money)
- **Data Sources**: Mock data + Polymarket (if configured)
- **Risk Management**: Built into strategies
- **Position Limits**: Handled by strategy parameters

## 🔧 Quick Setup Commands

1. **Copy example environment:**
   ```bash
   ./setup-env.sh
   ```

2. **Edit your configuration:**
   ```bash
   nano .env  # or use your preferred editor
   ```

3. **Test Polymarket connection:**
   ```bash
   python test_polymarket_api.py
   ```

4. **Start MCP server:**
   ```bash
   python src/mcp/mcp_server_enhanced.py
   ```

## 🎯 Implementation Notes

**Currently Active Features** (no env vars needed):
- ✅ Polymarket integration (auto-enabled if API keys present)
- ✅ Neural forecasting (always available)
- ✅ GPU acceleration (auto-detected)
- ✅ News sentiment analysis (always available)
- ✅ Demo trading (always safe mode)

**NOT Implemented** (from example.env.full):
- ❌ Real broker connections (IB, Alpaca)
- ❌ External news APIs (Alpha Vantage, NewsAPI)
- ❌ Database storage (PostgreSQL, Redis)
- ❌ Monitoring tools (Sentry, Prometheus)
- ❌ Feature flag environment variables

## 🔐 Security Tips

1. **Never commit `.env` to Git** (already in .gitignore)
2. **Use strong, unique API keys** for Polymarket
3. **Trading is ALWAYS in demo mode** (no real money risk)
4. **Polymarket keys are optional** (system works without them)

## 📖 Full Documentation

- **Current implementation**: See `example.env` (minimal, working)
- **Future roadmap**: See `example.env.full` (comprehensive) 
- **Implementation status**: See `IMPLEMENTATION_STATUS.md`
- **Polymarket setup**: See `POLYMARKET_SETUP.md`
- **General docs**: See `README.md`