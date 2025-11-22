#!/bin/bash

# CWTS Ultra FreqTrade Integration Build Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CWTS Ultra FreqTrade Integration     ${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python3 --version

# Check for required packages
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Cython
if ! python3 -c "import cython" 2>/dev/null; then
    echo -e "${RED}Cython not found. Installing...${NC}"
    pip3 install cython
else
    echo -e "${GREEN}✓ Cython found${NC}"
fi

# Check NumPy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo -e "${RED}NumPy not found. Installing...${NC}"
    pip3 install numpy
else
    echo -e "${GREEN}✓ NumPy found${NC}"
fi

# Check FreqTrade
if ! python3 -c "import freqtrade" 2>/dev/null; then
    echo -e "${YELLOW}FreqTrade not found. Install with:${NC}"
    echo "pip3 install freqtrade"
else
    echo -e "${GREEN}✓ FreqTrade found${NC}"
fi

# Clean previous builds
echo -e "${YELLOW}Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info
rm -f cwts_client*.so cwts_client*.c cwts_client*.html

# Build Cython extension
echo -e "${YELLOW}Building Cython extension...${NC}"
python3 setup.py build_ext --inplace

# Check if build succeeded
if [ -f cwts_client*.so ]; then
    echo -e "${GREEN}✓ Build successful!${NC}"
    
    # Test import
    echo -e "${YELLOW}Testing import...${NC}"
    if python3 -c "import cwts_client; print('✓ CWTS client imported successfully')" 2>/dev/null; then
        echo -e "${GREEN}✓ Import test passed${NC}"
    else
        echo -e "${RED}✗ Import test failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Create shared memory file
echo -e "${YELLOW}Setting up shared memory...${NC}"
SHM_PATH="/dev/shm/cwts_ultra"
if [ ! -f "$SHM_PATH" ]; then
    touch "$SHM_PATH"
    chmod 666 "$SHM_PATH"
    echo -e "${GREEN}✓ Shared memory file created${NC}"
else
    echo -e "${GREEN}✓ Shared memory file exists${NC}"
fi

# Install in development mode (optional)
echo -e "${YELLOW}Install package for development? (y/n)${NC}"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip3 install -e .
    echo -e "${GREEN}✓ Package installed in development mode${NC}"
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Start CWTS Ultra: ~/.local/cwts-ultra/scripts/launch.sh"
echo "2. Configure FreqTrade: Add strategy path to config.json"
echo "3. Test strategy: freqtrade backtesting --strategy CWTSMomentumStrategy"
echo "4. Run live: freqtrade trade --strategy CWTSMomentumStrategy --dry-run"
echo
echo -e "${BLUE}Performance tips:${NC}"
echo "- Use 'ultra' latency mode for production"
echo "- Enable huge pages: sudo sysctl -w vm.nr_hugepages=128"
echo "- Pin CPU cores: taskset -c 4-7 freqtrade trade"
echo
echo -e "${GREEN}Happy trading!${NC}"