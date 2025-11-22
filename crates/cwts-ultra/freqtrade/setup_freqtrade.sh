#!/bin/bash

# CWTS Ultra FreqTrade Setup Helper
# This script helps you properly set up the FreqTrade integration

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CWTS Ultra FreqTrade Setup Helper    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Step 1: Find FreqTrade installation
echo -e "${YELLOW}Step 1: Locating FreqTrade installation...${NC}"

# Common FreqTrade locations
POSSIBLE_PATHS=(
    "$HOME/freqtrade"
    "$HOME/.freqtrade"
    "$HOME/ft"
    "$HOME/trading/freqtrade"
    "$HOME/bots/freqtrade"
    "/opt/freqtrade"
    "$(pwd)"
)

FREQTRADE_DIR=""
for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -d "$path/user_data" ]; then
        FREQTRADE_DIR="$path"
        echo -e "${GREEN}✓ Found FreqTrade at: $FREQTRADE_DIR${NC}"
        break
    fi
done

if [ -z "$FREQTRADE_DIR" ]; then
    echo -e "${YELLOW}Could not auto-detect FreqTrade installation.${NC}"
    echo "Please enter the path to your FreqTrade directory:"
    echo "(The directory that contains 'user_data' folder)"
    read -p "Path: " FREQTRADE_DIR
    
    if [ ! -d "$FREQTRADE_DIR/user_data" ]; then
        echo -e "${RED}Error: user_data directory not found in $FREQTRADE_DIR${NC}"
        echo "Creating user_data structure..."
        mkdir -p "$FREQTRADE_DIR/user_data/strategies"
    fi
fi

STRATEGIES_DIR="$FREQTRADE_DIR/user_data/strategies"
echo -e "${GREEN}✓ Strategies directory: $STRATEGIES_DIR${NC}"

# Step 2: Copy or link files
echo
echo -e "${YELLOW}Step 2: Setting up strategy files...${NC}"
echo "Choose installation method:"
echo "  1) Copy files (recommended for production)"
echo "  2) Create symbolic links (recommended for development)"
read -p "Choice [1/2]: " INSTALL_METHOD

if [ "$INSTALL_METHOD" == "2" ]; then
    # Create symbolic links
    echo -e "${YELLOW}Creating symbolic links...${NC}"
    ln -sf "$SCRIPT_DIR/strategies/CWTSUltraStrategy.py" "$STRATEGIES_DIR/" 2>/dev/null || true
    ln -sf "$SCRIPT_DIR/strategies/CWTSMomentumStrategy.py" "$STRATEGIES_DIR/" 2>/dev/null || true
    ln -sf "$SCRIPT_DIR/cwts_client_simple.py" "$STRATEGIES_DIR/" 2>/dev/null || true
    echo -e "${GREEN}✓ Symbolic links created${NC}"
else
    # Copy files
    echo -e "${YELLOW}Copying files...${NC}"
    cp "$SCRIPT_DIR/strategies/CWTSUltraStrategy.py" "$STRATEGIES_DIR/"
    cp "$SCRIPT_DIR/strategies/CWTSMomentumStrategy.py" "$STRATEGIES_DIR/"
    cp "$SCRIPT_DIR/cwts_client_simple.py" "$STRATEGIES_DIR/"
    echo -e "${GREEN}✓ Files copied${NC}"
fi

# Step 3: Install Python dependencies
echo
echo -e "${YELLOW}Step 3: Installing Python dependencies...${NC}"

# Check if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}✓ Virtual environment detected: $VIRTUAL_ENV${NC}"
    PIP_CMD="pip"
elif [ -f "$FREQTRADE_DIR/.venv/bin/pip" ]; then
    echo -e "${GREEN}✓ FreqTrade virtual environment found${NC}"
    PIP_CMD="$FREQTRADE_DIR/.venv/bin/pip"
elif [ -f "$FREQTRADE_DIR/venv/bin/pip" ]; then
    echo -e "${GREEN}✓ FreqTrade virtual environment found${NC}"
    PIP_CMD="$FREQTRADE_DIR/venv/bin/pip"
else
    echo -e "${YELLOW}No virtual environment found. Using system pip.${NC}"
    PIP_CMD="pip3"
fi

echo "Installing required packages..."
$PIP_CMD install numpy websockets msgpack-python aiofiles 2>/dev/null || {
    echo -e "${YELLOW}Some packages may already be installed${NC}"
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 4: Create configuration template
echo
echo -e "${YELLOW}Step 4: Creating configuration template...${NC}"

cat > "$FREQTRADE_DIR/user_data/config_cwts.json" << 'EOF'
{
    "max_open_trades": 3,
    "stake_currency": "USDT",
    "stake_amount": 100,
    "tradable_balance_ratio": 0.99,
    "fiat_display_currency": "USD",
    "dry_run": true,
    "dry_run_wallet": 10000,
    "cancel_open_orders_on_exit": false,
    "unfilledtimeout": {
        "entry": 10,
        "exit": 10,
        "exit_timeout_count": 0,
        "unit": "minutes"
    },
    "entry_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": false,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "same",
        "use_order_book": true,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": "your_api_key_here",
        "secret": "your_api_secret_here",
        "ccxt_config": {},
        "ccxt_async_config": {},
        "pair_whitelist": [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT"
        ],
        "pair_blacklist": []
    },
    "edge": {
        "enabled": false
    },
    "telegram": {
        "enabled": false
    },
    "api_server": {
        "enabled": true,
        "listen_ip_address": "127.0.0.1",
        "listen_port": 8080,
        "verbosity": "error",
        "enable_openapi": false,
        "jwt_secret_key": "somethingrandom",
        "ws_token": "your_ws_token",
        "CORS_origins": [],
        "username": "freqtrader",
        "password": "freqtrader"
    },
    "bot_name": "CWTS_Ultra_Bot",
    "initial_state": "running",
    "force_entry_enable": false,
    "internals": {
        "process_throttle_secs": 5
    },
    "strategy": "CWTSMomentumStrategy",
    "strategy_path": "user_data/strategies",
    "dataformat_ohlcv": "json",
    "dataformat_trades": "jsongz"
}
EOF

echo -e "${GREEN}✓ Configuration template created at: $FREQTRADE_DIR/user_data/config_cwts.json${NC}"

# Step 5: Test the setup
echo
echo -e "${YELLOW}Step 5: Testing the setup...${NC}"

# Create a test script
cat > "$STRATEGIES_DIR/test_cwts_import.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os

print("Testing CWTS Ultra import...")

try:
    import cwts_client_simple
    print("✅ cwts_client_simple imported successfully")
except ImportError as e:
    print(f"❌ Failed to import cwts_client_simple: {e}")
    sys.exit(1)

try:
    from CWTSUltraStrategy import CWTSUltraStrategy
    print("✅ CWTSUltraStrategy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CWTSUltraStrategy: {e}")
    sys.exit(1)

try:
    from CWTSMomentumStrategy import CWTSMomentumStrategy
    print("✅ CWTSMomentumStrategy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CWTSMomentumStrategy: {e}")
    sys.exit(1)

print("\n✅ All imports successful! Setup is complete.")
EOF

chmod +x "$STRATEGIES_DIR/test_cwts_import.py"

# Run the test
if [ -f "$FREQTRADE_DIR/.venv/bin/python" ]; then
    TEST_PYTHON="$FREQTRADE_DIR/.venv/bin/python"
elif [ -f "$FREQTRADE_DIR/venv/bin/python" ]; then
    TEST_PYTHON="$FREQTRADE_DIR/venv/bin/python"
else
    TEST_PYTHON="python3"
fi

cd "$STRATEGIES_DIR"
$TEST_PYTHON test_cwts_import.py

# Step 6: Create launcher scripts
echo
echo -e "${YELLOW}Step 6: Creating launcher scripts...${NC}"

# Create dry-run launcher
cat > "$FREQTRADE_DIR/launch_cwts_dryrun.sh" << 'EOF'
#!/bin/bash

# Start CWTS Ultra first
echo "Starting CWTS Ultra..."
~/.local/cwts-ultra/scripts/launch.sh &
CWTS_PID=$!

# Wait for CWTS to initialize
sleep 3

# Start FreqTrade
echo "Starting FreqTrade with CWTS Momentum Strategy..."
freqtrade trade \
    --strategy CWTSMomentumStrategy \
    --config user_data/config_cwts.json \
    --dry-run

# Clean up
kill $CWTS_PID 2>/dev/null
EOF

chmod +x "$FREQTRADE_DIR/launch_cwts_dryrun.sh"

# Create backtest launcher
cat > "$FREQTRADE_DIR/backtest_cwts.sh" << 'EOF'
#!/bin/bash

# Download data if needed
echo "Downloading data for backtesting..."
freqtrade download-data \
    --config user_data/config_cwts.json \
    --days 30 \
    --timeframe 1m 5m 15m 1h

# Run backtest
echo "Running backtest..."
freqtrade backtesting \
    --strategy CWTSMomentumStrategy \
    --config user_data/config_cwts.json \
    --timeframe 1m \
    --timerange 20240101-
EOF

chmod +x "$FREQTRADE_DIR/backtest_cwts.sh"

echo -e "${GREEN}✓ Launcher scripts created${NC}"

# Final summary
echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "${YELLOW}Files installed:${NC}"
echo "  ✓ $STRATEGIES_DIR/CWTSUltraStrategy.py"
echo "  ✓ $STRATEGIES_DIR/CWTSMomentumStrategy.py"
echo "  ✓ $STRATEGIES_DIR/cwts_client_simple.py"
echo "  ✓ $FREQTRADE_DIR/user_data/config_cwts.json"
echo "  ✓ $FREQTRADE_DIR/launch_cwts_dryrun.sh"
echo "  ✓ $FREQTRADE_DIR/backtest_cwts.sh"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo
echo "1. Edit configuration:"
echo "   vim $FREQTRADE_DIR/user_data/config_cwts.json"
echo
echo "2. Test with dry-run:"
echo "   cd $FREQTRADE_DIR"
echo "   ./launch_cwts_dryrun.sh"
echo
echo "3. Run backtest:"
echo "   cd $FREQTRADE_DIR"
echo "   ./backtest_cwts.sh"
echo
echo "4. Use with existing FreqTrade:"
echo "   freqtrade trade --strategy CWTSMomentumStrategy --config user_data/config_cwts.json"
echo
echo -e "${GREEN}Happy Trading!${NC} 🚀"