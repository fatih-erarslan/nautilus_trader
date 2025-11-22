#!/bin/bash

# CWTS Ultra FreqTrade Integration Installer

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CWTS Ultra FreqTrade Integration     ${NC}"
echo -e "${BLUE}           Installer                    ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$SCRIPT_DIR/venv"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip setuptools wheel >/dev/null 2>&1
pip install numpy websockets msgpack-python aiofiles >/dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create symbolic link destination prompt
echo
echo -e "${YELLOW}Where is your FreqTrade user_data directory?${NC}"
echo -e "Example: /home/$USER/freqtrade/user_data"
read -p "Path: " FREQTRADE_PATH

# Validate path
if [ ! -d "$FREQTRADE_PATH" ]; then
    echo -e "${RED}Directory does not exist: $FREQTRADE_PATH${NC}"
    echo -e "${YELLOW}Creating directory...${NC}"
    mkdir -p "$FREQTRADE_PATH/strategies"
fi

# Create strategies directory if it doesn't exist
if [ ! -d "$FREQTRADE_PATH/strategies" ]; then
    mkdir -p "$FREQTRADE_PATH/strategies"
    echo -e "${GREEN}✓ Created strategies directory${NC}"
fi

# Create symbolic links
echo -e "${YELLOW}Creating symbolic links...${NC}"

# Link the strategies
ln -sf "$SCRIPT_DIR/strategies/CWTSUltraStrategy.py" "$FREQTRADE_PATH/strategies/" 2>/dev/null || true
ln -sf "$SCRIPT_DIR/strategies/CWTSMomentumStrategy.py" "$FREQTRADE_PATH/strategies/" 2>/dev/null || true

# Link the client module
ln -sf "$SCRIPT_DIR/cwts_client_simple.py" "$FREQTRADE_PATH/strategies/" 2>/dev/null || true

echo -e "${GREEN}✓ Symbolic links created${NC}"

# Create a launcher script
echo -e "${YELLOW}Creating launcher script...${NC}"
cat > "$FREQTRADE_PATH/launch_cwts_strategy.sh" << 'EOF'
#!/bin/bash

# Activate CWTS virtual environment for dependencies
CWTS_DIR="$(dirname "$(readlink -f "$0")")/../cwts-ultra/freqtrade"
if [ -d "$CWTS_DIR/venv" ]; then
    source "$CWTS_DIR/venv/bin/activate"
fi

# Start FreqTrade with CWTS strategy
freqtrade trade \
    --strategy CWTSMomentumStrategy \
    --config config.json \
    "$@"
EOF

chmod +x "$FREQTRADE_PATH/launch_cwts_strategy.sh"
echo -e "${GREEN}✓ Launcher script created${NC}"

# Test the installation
echo
echo -e "${YELLOW}Testing installation...${NC}"
python "$SCRIPT_DIR/test_integration.py" | grep -E "✅|⚠️|❌" | head -5

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "${YELLOW}Symbolic links created:${NC}"
echo "  • $FREQTRADE_PATH/strategies/CWTSUltraStrategy.py"
echo "  • $FREQTRADE_PATH/strategies/CWTSMomentumStrategy.py"
echo "  • $FREQTRADE_PATH/strategies/cwts_client_simple.py"
echo
echo -e "${YELLOW}To use the strategies:${NC}"
echo
echo "1. Start CWTS Ultra:"
echo "   ~/.local/cwts-ultra/scripts/launch.sh"
echo
echo "2. Configure FreqTrade (config.json):"
echo '   {
     "strategy": "CWTSMomentumStrategy",
     "strategy_path": "user_data/strategies"
   }'
echo
echo "3. Run FreqTrade:"
echo "   freqtrade trade --strategy CWTSMomentumStrategy"
echo
echo -e "${GREEN}Happy Trading!${NC}"