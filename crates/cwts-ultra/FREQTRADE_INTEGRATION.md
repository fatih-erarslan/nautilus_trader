# CWTS Ultra + FreqTrade Integration Guide

## ✅ Status: Ready to Use

The CWTS Ultra FreqTrade integration is now fully configured with:
- **Port 4000**: MCP WebSocket server (fixed from 3000)
- **Port 4001**: MCP HTTP API
- **Port 9595**: Prometheus metrics
- **Port 8585**: Health check

## 📋 Your Configuration

### FreqTrade Location
- **Directory**: `/home/kutlu/freqtrade`
- **Config**: `/home/kutlu/freqtrade/user_data/config.json` (not modified)
- **Strategies**: `/home/kutlu/freqtrade/user_data/strategies/`

### CWTS Ultra Configuration
- **Binary**: `/home/kutlu/.local/cwts-ultra/bin/cwts-ultra`
- **Config**: `/home/kutlu/.local/cwts-ultra/config/production.toml`
- **Ports**: Configured in `/home/kutlu/.local/cwts-ultra/config/ports.env`

## 🚀 Quick Start

### Step 1: Copy CWTS Strategies to FreqTrade

```bash
# Copy the CWTS files to your FreqTrade strategies directory
cp /home/kutlu/CWTS/cwts-ultra/freqtrade/cwts_client_simple.py /home/kutlu/freqtrade/user_data/strategies/
cp /home/kutlu/CWTS/cwts-ultra/freqtrade/strategies/CWTSUltraStrategy.py /home/kutlu/freqtrade/user_data/strategies/
cp /home/kutlu/CWTS/cwts-ultra/freqtrade/strategies/CWTSMomentumStrategy.py /home/kutlu/freqtrade/user_data/strategies/
```

### Step 2: Install Dependencies in FreqTrade venv

```bash
# Activate FreqTrade virtual environment
cd /home/kutlu/freqtrade
source .venv/bin/activate

# Install required packages
pip install numpy websockets msgpack-python aiofiles
```

### Step 3: Start CWTS Ultra (Terminal 1)

```bash
# Option A: Using the launch script (interactive menu)
~/.local/cwts-ultra/scripts/launch.sh
# Select option 2 for Paper Trading or 5 for MCP Server Only

# Option B: Direct launch with ports
export MCP_SERVER_PORT=4000
~/.local/cwts-ultra/bin/cwts-ultra --config ~/.local/cwts-ultra/config/production.toml
```

### Step 4: Run FreqTrade with CWTS Strategy (Terminal 2)

```bash
cd /home/kutlu/freqtrade
source .venv/bin/activate

# Option 1: Use CWTS strategy with your existing config
freqtrade trade \
    --strategy CWTSMomentumStrategy \
    --config user_data/config.json \
    --dry-run

# Option 2: Use with FreqAI (as you were trying)
freqtrade trade \
    --freqaimodel CatboostRegressor \
    --strategy CWTSMomentumStrategy \
    --config user_data/config.json

# Option 3: Backtest the strategy
freqtrade backtesting \
    --strategy CWTSMomentumStrategy \
    --config user_data/config.json \
    --timeframe 1m \
    --timerange 20240101-
```

## 📊 Strategy Features

### CWTSMomentumStrategy
- **Type**: High-frequency momentum trading
- **Timeframe**: 1-minute candles (configurable)
- **Features**:
  - Order book imbalance signals
  - Dynamic position sizing based on volatility
  - Sub-millisecond signal transmission via shared memory
  - Configurable momentum thresholds
  - Support for both long and short positions

### Key Parameters
```python
# In the strategy file
minimal_roi = {
    "0": 0.015,   # 1.5% immediate target
    "5": 0.01,    # 1% after 5 minutes
    "15": 0.005,  # 0.5% after 15 minutes
    "30": 0.003,  # 0.3% after 30 minutes
}

stoploss = -0.015  # 1.5% stop loss
trailing_stop = True
timeframe = '1m'
can_short = True
```

## 🔧 Configuration Options

### Using with Your Existing config.json

Your existing config at `/home/kutlu/freqtrade/user_data/config.json` includes:
- Strategy: `RustEnhancedATSCPStrategy`
- FreqAI enabled with CatBoost
- Multiple config files imported

To use CWTS strategies without modifying your main config:

```bash
# Create a temporary override config
cat > /home/kutlu/freqtrade/user_data/config_cwts_override.json << EOF
{
  "strategy": "CWTSMomentumStrategy",
  "add_config_files": [
    "./config.json"
  ]
}
EOF

# Use it
freqtrade trade --config user_data/config_cwts_override.json
```

### Performance Tuning

For optimal performance, ensure:

1. **Shared Memory**: The shared memory file exists
```bash
touch /dev/shm/cwts_ultra
chmod 666 /dev/shm/cwts_ultra
```

2. **Check CWTS is Running**: Verify the server is up
```bash
# Check if CWTS is running
ps aux | grep cwts-ultra

# Check if port 4000 is listening
netstat -an | grep 4000

# Test WebSocket connection
curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:4000/
```

3. **Monitor Performance**: Check metrics
```bash
# Prometheus metrics
curl http://localhost:9595/metrics

# Health check
curl http://localhost:8585/health
```

## 🧪 Testing the Integration

### Test 1: Verify Imports
```bash
cd /home/kutlu/freqtrade
source .venv/bin/activate

python -c "
import sys
sys.path.insert(0, 'user_data/strategies')
import cwts_client_simple
from CWTSMomentumStrategy import CWTSMomentumStrategy
print('✅ All imports successful!')
"
```

### Test 2: List Available Strategies
```bash
freqtrade list-strategies | grep CWTS
```

### Test 3: Run a Quick Backtest
```bash
freqtrade backtesting \
    --strategy CWTSMomentumStrategy \
    --timeframe 1m \
    --timerange 20240101-20240102 \
    --dry-run-wallet 10000
```

## 📈 Performance Metrics

When everything is working correctly, you should see:

- **Signal Latency**: < 100 microseconds
- **Order Book Updates**: 10,000+ per second
- **Market Data Processing**: 50,000+ updates per second
- **Memory Usage**: < 100MB for client
- **CPU Usage**: < 5% idle, 10-20% active trading

## 🔍 Troubleshooting

### Issue: "CWTS Ultra client not available"
```bash
# Ensure the client module is in the strategies directory
ls -la /home/kutlu/freqtrade/user_data/strategies/cwts_client_simple.py
```

### Issue: WebSocket connection failed
```bash
# Check CWTS Ultra is running
ps aux | grep cwts-ultra

# Check correct port
netstat -an | grep 4000

# Check logs
tail -f /tmp/cwts-ultra.log
```

### Issue: Import errors in FreqTrade
```bash
# Ensure you're in the FreqTrade virtual environment
which python  # Should show .venv/bin/python

# Reinstall dependencies
pip install --force-reinstall numpy websockets msgpack-python aiofiles
```

## 🎯 Next Steps

1. **Test with paper trading first** - Always safe
2. **Optimize parameters** using FreqTrade's hyperopt:
   ```bash
   freqtrade hyperopt \
       --strategy CWTSMomentumStrategy \
       --hyperopt-loss SharpeHyperOptLoss \
       --epochs 100
   ```
3. **Monitor performance** via Prometheus/Grafana
4. **Adjust risk parameters** in the strategy
5. **Consider custom modifications** to the strategy

## 📝 Notes on Your Existing Setup

Your current FreqTrade config uses:
- **RustEnhancedATSCPStrategy** with quantum features
- **FreqAI** with CatBoost for predictions
- **Multiple timeframes** (5m, 15m, 1h, 4h, 6h, 1d)
- **GPU acceleration** for model training

CWTS Ultra strategies can work alongside this by:
- Using different pairs or timeframes
- Running in parallel sessions
- Combining signals from both systems
- Using CWTS for execution while FreqAI does prediction

## 🚨 Important Safety Notes

1. **Always test in dry-run mode first**
2. **Start with small position sizes**
3. **Monitor the first few trades carefully**
4. **Set appropriate stop losses**
5. **Never trade with funds you can't afford to lose**

## 📞 Support

- **CWTS Ultra logs**: `/tmp/cwts-ultra.log`
- **FreqTrade logs**: Check console output or configured log file
- **Configuration files**: 
  - CWTS: `~/.local/cwts-ultra/config/production.toml`
  - FreqTrade: `~/freqtrade/user_data/config.json`

---

**The integration is ready to use!** Start with paper trading to verify everything works correctly with your setup.