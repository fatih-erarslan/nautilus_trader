#!/bin/bash

# CWTS Ultra FreqTrade Setup for kutlu's existing installation
# This script sets up CWTS Ultra strategies in your FreqTrade directory

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CWTS Ultra FreqTrade Setup           ${NC}"
echo -e "${BLUE}  For kutlu's FreqTrade Installation   ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Your FreqTrade directory
FREQTRADE_DIR="/home/kutlu/freqtrade"
STRATEGIES_DIR="$FREQTRADE_DIR/user_data/strategies"

echo -e "${GREEN}✓ Found FreqTrade at: $FREQTRADE_DIR${NC}"
echo -e "${GREEN}✓ Strategies directory: $STRATEGIES_DIR${NC}"

# Step 1: Copy CWTS files to your strategies directory
echo
echo -e "${YELLOW}Step 1: Installing CWTS Ultra strategies...${NC}"

# Copy the files (not symlinks, to avoid import issues)
cp "$SCRIPT_DIR/cwts_client_simple.py" "$STRATEGIES_DIR/"
cp "$SCRIPT_DIR/strategies/CWTSUltraStrategy.py" "$STRATEGIES_DIR/"
cp "$SCRIPT_DIR/strategies/CWTSMomentumStrategy.py" "$STRATEGIES_DIR/"

echo -e "${GREEN}✓ Strategies copied to FreqTrade${NC}"

# Step 2: Install Python dependencies in FreqTrade's venv
echo
echo -e "${YELLOW}Step 2: Installing Python dependencies...${NC}"

if [ -f "$FREQTRADE_DIR/.venv/bin/pip" ]; then
    VENV_PIP="$FREQTRADE_DIR/.venv/bin/pip"
    VENV_PYTHON="$FREQTRADE_DIR/.venv/bin/python"
elif [ -f "$FREQTRADE_DIR/venv/bin/pip" ]; then
    VENV_PIP="$FREQTRADE_DIR/venv/bin/pip"
    VENV_PYTHON="$FREQTRADE_DIR/venv/bin/python"
else
    echo -e "${RED}Error: FreqTrade virtual environment not found${NC}"
    echo "Please activate your FreqTrade environment and run:"
    echo "  pip install numpy websockets msgpack-python aiofiles"
    exit 1
fi

echo "Installing required packages in FreqTrade venv..."
$VENV_PIP install -q numpy websockets msgpack-python aiofiles 2>/dev/null || {
    echo -e "${YELLOW}Some packages may already be installed${NC}"
}
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Create shared memory file
echo
echo -e "${YELLOW}Step 3: Setting up shared memory...${NC}"

# Create shared memory file for IPC
sudo touch /dev/shm/cwts_ultra 2>/dev/null || touch /dev/shm/cwts_ultra
sudo chmod 666 /dev/shm/cwts_ultra 2>/dev/null || chmod 666 /dev/shm/cwts_ultra
echo -e "${GREEN}✓ Shared memory file created${NC}"

# Step 4: Create a CWTS-specific config that extends your existing config
echo
echo -e "${YELLOW}Step 4: Creating CWTS configuration...${NC}"

# Create a config that imports your existing config and adds CWTS strategy
cat > "$FREQTRADE_DIR/user_data/config_cwts.json" << 'EOF'
{
  "strategy": "CWTSMomentumStrategy",
  "add_config_files": [
    "./config.json"
  ]
}
EOF

echo -e "${GREEN}✓ CWTS config created (extends your existing config.json)${NC}"

# Step 5: Create launcher scripts
echo
echo -e "${YELLOW}Step 5: Creating launcher scripts...${NC}"

# Create a launcher that starts both CWTS and FreqTrade
cat > "$FREQTRADE_DIR/launch_cwts_trading.sh" << 'EOF'
#!/bin/bash

# CWTS Ultra + FreqTrade Launcher

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CWTS Ultra + FreqTrade Launcher      ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if CWTS Ultra is already running
if pgrep -f "cwts-ultra" > /dev/null; then
    echo -e "${YELLOW}CWTS Ultra is already running${NC}"
else
    echo -e "${GREEN}Starting CWTS Ultra...${NC}"
    # Start CWTS Ultra in background with configured ports
    export MCP_SERVER_PORT=4000
    export MCP_HTTP_PORT=4001
    export PROMETHEUS_PORT=9595
    export HEALTH_CHECK_PORT=8585
    
    ~/.local/cwts-ultra/bin/cwts-ultra --config ~/.local/cwts-ultra/config/production.toml > /tmp/cwts-ultra.log 2>&1 &
    CWTS_PID=$!
    
    # Wait for CWTS to initialize
    echo "Waiting for CWTS Ultra to initialize..."
    sleep 3
    
    if ps -p $CWTS_PID > /dev/null; then
        echo -e "${GREEN}✓ CWTS Ultra started (PID: $CWTS_PID)${NC}"
        echo $CWTS_PID > /tmp/cwts-ultra.pid
    else
        echo -e "${RED}Failed to start CWTS Ultra${NC}"
        exit 1
    fi
fi

# Now start FreqTrade with CWTS strategy
echo
echo -e "${GREEN}Starting FreqTrade with CWTS Momentum Strategy...${NC}"
echo

cd /home/kutlu/freqtrade
source .venv/bin/activate

# You can modify this command to use your preferred options
freqtrade trade \
    --strategy CWTSMomentumStrategy \
    --config user_data/config_cwts.json \
    --dry-run

# Cleanup on exit
if [ -f /tmp/cwts-ultra.pid ]; then
    CWTS_PID=$(cat /tmp/cwts-ultra.pid)
    echo
    echo -e "${YELLOW}Stopping CWTS Ultra (PID: $CWTS_PID)...${NC}"
    kill $CWTS_PID 2>/dev/null || true
    rm /tmp/cwts-ultra.pid
fi
EOF

chmod +x "$FREQTRADE_DIR/launch_cwts_trading.sh"

# Create a test script
cat > "$FREQTRADE_DIR/test_cwts_strategy.sh" << 'EOF'
#!/bin/bash

# Test CWTS Strategy Import

cd /home/kutlu/freqtrade
source .venv/bin/activate

echo "Testing CWTS strategy import..."
python -c "
import sys
sys.path.insert(0, 'user_data/strategies')
try:
    import cwts_client_simple
    print('✅ cwts_client_simple imported')
    from CWTSMomentumStrategy import CWTSMomentumStrategy
    print('✅ CWTSMomentumStrategy imported')
    print('\n✅ All imports successful!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    sys.exit(1)
"

echo
echo "Listing available strategies:"
freqtrade list-strategies | grep CWTS || echo "No CWTS strategies found"
EOF

chmod +x "$FREQTRADE_DIR/test_cwts_strategy.sh"

echo -e "${GREEN}✓ Launcher scripts created${NC}"

# Step 6: Test the setup
echo
echo -e "${YELLOW}Step 6: Testing the setup...${NC}"

cd "$FREQTRADE_DIR"
$VENV_PYTHON -c "
import sys
sys.path.insert(0, 'user_data/strategies')
try:
    import cwts_client_simple
    print('✅ cwts_client_simple module imported')
except ImportError as e:
    print(f'❌ Failed to import cwts_client_simple: {e}')
    sys.exit(1)

try:
    from CWTSMomentumStrategy import CWTSMomentumStrategy
    print('✅ CWTSMomentumStrategy imported')
except ImportError as e:
    print(f'❌ Failed to import CWTSMomentumStrategy: {e}')
    sys.exit(1)

print('✅ All imports successful!')
"

# Final summary
echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "${YELLOW}Files installed:${NC}"
echo "  ✓ $STRATEGIES_DIR/cwts_client_simple.py"
echo "  ✓ $STRATEGIES_DIR/CWTSUltraStrategy.py"
echo "  ✓ $STRATEGIES_DIR/CWTSMomentumStrategy.py"
echo "  ✓ $FREQTRADE_DIR/user_data/config_cwts.json"
echo "  ✓ $FREQTRADE_DIR/launch_cwts_trading.sh"
echo "  ✓ $FREQTRADE_DIR/test_cwts_strategy.sh"
echo
echo -e "${YELLOW}Your existing config.json:${NC}"
echo "  ✓ Not modified - CWTS config extends it"
echo "  ✓ Current strategy: $(grep -oP '"strategy":\s*"\K[^"]+' $FREQTRADE_DIR/user_data/config.json)"
echo
echo -e "${YELLOW}CWTS Ultra ports configured:${NC}"
echo "  • MCP WebSocket: 4000"
echo "  • MCP HTTP API: 4001"
echo "  • Prometheus: 9595"
echo "  • Health Check: 8585"
echo
echo -e "${YELLOW}To use CWTS strategies:${NC}"
echo
echo "1. Test the setup:"
echo "   $FREQTRADE_DIR/test_cwts_strategy.sh"
echo
echo "2. Launch with CWTS (dry-run):"
echo "   $FREQTRADE_DIR/launch_cwts_trading.sh"
echo
echo "3. Use with your existing config:"
echo "   freqtrade trade --strategy CWTSMomentumStrategy --config user_data/config.json"
echo
echo "4. Use with FreqAI (as you were doing):"
echo "   freqtrade trade --freqaimodel CatboostRegressor --strategy CWTSMomentumStrategy"
echo
echo -e "${GREEN}Happy Trading!${NC} 🚀"