# 🚀 Configuring MCP Neural Trader for Your Alpaca Account PA3MANXUAXIR

## ✅ Current Status

The MCP neural trader is **working** but using **simulated data**. To connect it to your real Alpaca paper trading account PA3MANXUAXIR, follow these steps:

---

## 📋 Step 1: Get Your Real API Keys

1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Click on "View API Keys" or "Regenerate API Keys"
3. Copy both:
   - **API Key ID** (starts with PK...)
   - **Secret Key** (keep this secure!)

---

## 📋 Step 2: Update Environment Configuration

### Option A: Update .env file (Recommended)

Edit `/workspaces/neural-trader/.env` and replace the test keys:

```bash
# Replace these lines (around line 79 and 102-103)
ALPACA_API_KEY=YOUR_REAL_API_KEY_HERE
ALPACA_SECRET_KEY=YOUR_REAL_SECRET_KEY_HERE
ALPACA_BASE_URL=https://paper-api.alpaca.markets/v2
```

### Option B: Export Environment Variables

Run these commands in your terminal:

```bash
export ALPACA_API_KEY="YOUR_REAL_API_KEY_HERE"
export ALPACA_SECRET_KEY="YOUR_REAL_SECRET_KEY_HERE"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2"
```

---

## 📋 Step 3: Restart MCP Server with Real Keys

```bash
# Stop all MCP servers
pkill -f "mcp.*neural-trader"
pkill -f "mcp_server_enhanced.py"

# Start with new configuration
cd /workspaces/neural-trader
export ALPACA_API_KEY="YOUR_REAL_API_KEY_HERE"
export ALPACA_SECRET_KEY="YOUR_REAL_SECRET_KEY_HERE"

# Restart MCP servers
claude mcp restart
```

---

## 📋 Step 4: Test Connection to PA3MANXUAXIR

Run this test command:

```bash
PYTHONPATH=/workspaces/neural-trader/src python -c "
from alpaca.alpaca_client import AlpacaClient
import os

client = AlpacaClient(
    api_key=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url='https://paper-api.alpaca.markets/v2'
)

account = client.get_account()
print(f'✅ Connected to: {account.get(\"account_number\")}')
print(f'💰 Buying Power: \${float(account.get(\"buying_power\", 0)):,.2f}')
"
```

---

## 🎯 Step 5: Execute a Test Trade

Once connected, you can execute trades via MCP tools:

### Using MCP Tools in Claude:

```python
# Quick market analysis
mcp__neural-trader__quick_analysis("AAPL")

# Execute a trade
mcp__neural-trader__execute_trade(
    strategy="momentum_trading_optimized",
    symbol="AAPL",
    action="buy"
)

# Check portfolio
mcp__neural-trader__get_portfolio_status()
```

### Using Direct Python:

```python
from alpaca.alpaca_client import AlpacaClient, OrderSide, OrderType

client = AlpacaClient()

# Place a test order
order = client.place_order(
    symbol='AAPL',
    qty=1,
    side=OrderSide.BUY,
    order_type=OrderType.MARKET
)
print(f"Order placed: {order.id}")
```

---

## ✅ Verification Checklist

After configuration, verify these work:

- [ ] `mcp__neural-trader__get_portfolio_status()` shows real account data
- [ ] Account number shows PA3MANXUAXIR
- [ ] Orders appear in Alpaca dashboard
- [ ] Real buying power and portfolio values display

---

## 🔍 Current MCP Status

**MCP Tools Available:**
- ✅ `mcp__neural-trader__list_strategies` - Working
- ✅ `mcp__neural-trader__get_portfolio_status` - Working (simulated)
- ⏳ `mcp__neural-trader__execute_trade` - Needs real API keys
- ⏳ `mcp__neural-trader__quick_analysis` - Needs real API keys

**Strategies Available:**
- mirror_trading_optimized (Sharpe: 6.01)
- momentum_trading_optimized (Sharpe: 2.84)
- mean_reversion_optimized (Sharpe: 2.9)
- swing_trading_optimized (Sharpe: 1.89)

---

## ⚠️ Important Notes

1. **Paper Trading Only**: Your account PA3MANXUAXIR is a paper trading account (safe for testing)
2. **Market Hours**: Orders only execute during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
3. **Rate Limits**: Alpaca has rate limits - don't exceed 200 requests/minute
4. **Security**: Never commit your secret key to git

---

## 🚨 Troubleshooting

If you don't see orders in your Alpaca dashboard:

1. **Check API Keys**: Ensure you're using the correct keys for PA3MANXUAXIR
2. **Market Hours**: Orders only work during market hours
3. **Check Logs**: `tail -f /tmp/neural-trader-mcp.log`
4. **Restart MCP**: `claude mcp restart`
5. **Verify Connection**: Run the test script again

---

## 📞 Next Steps

Once configured with real keys:

1. Execute a small test trade (1 share of AAPL)
2. Verify it appears in your Alpaca dashboard
3. Test neural trading strategies
4. Monitor performance in real-time

Your account PA3MANXUAXIR should then show real trading activity!