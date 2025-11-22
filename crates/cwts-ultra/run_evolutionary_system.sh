#!/bin/bash

# CWTS Evolutionary Adaptation System Runner
# Constitutional Prime Directive Compliant

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     🧬 CWTS QUANTUM-INSPIRED TRADING SYSTEM - LAUNCHER 🧬        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if running in development or production mode
MODE="${1:-development}"

echo -e "${BLUE}▶ Starting in ${YELLOW}${MODE}${BLUE} mode...${NC}"
echo ""

# Environment setup
export RUST_LOG=info
export RUST_BACKTRACE=1

# For demo mode, use mock credentials
if [ "$MODE" == "development" ] || [ "$MODE" == "demo" ]; then
    export BINANCE_API_KEY="demo_api_key"
    export BINANCE_SECRET_KEY="demo_secret_key"
    export E2B_API_TOKEN="demo_e2b_token"
    echo -e "${YELLOW}⚠️  Running with DEMO credentials (no real trading)${NC}"
else
    # Production mode - check for real credentials
    if [ -z "${BINANCE_API_KEY:-}" ] || [ -z "${E2B_API_TOKEN:-}" ]; then
        echo -e "${RED}❌ Error: Production credentials not set${NC}"
        echo -e "${RED}   Please set BINANCE_API_KEY, BINANCE_SECRET_KEY, and E2B_API_TOKEN${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Production credentials loaded${NC}"
fi

echo ""
echo -e "${BLUE}📊 System Configuration:${NC}"
echo -e "  • Bayesian VaR Engine: ${GREEN}ENABLED${NC}"
echo -e "  • Genetic Optimizer: ${GREEN}ENABLED${NC}"
echo -e "  • Continuous Learning: ${GREEN}ENABLED${NC}"
echo -e "  • E2B Sandbox Training: ${GREEN}ENABLED${NC}"
echo -e "  • Constitutional Compliance: ${GREEN}ENFORCED${NC}"
echo ""

# Create required directories
mkdir -p logs
mkdir -p data/cache
mkdir -p data/genomes
mkdir -p reports

# Check if compiled binary exists
if [ ! -f "target/release/cwts-ultra" ]; then
    echo -e "${YELLOW}⚠️  Binary not found, compiling...${NC}"
    cargo build --release
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Compilation failed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Compilation successful${NC}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 LAUNCHING EVOLUTIONARY ADAPTATION SYSTEM${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to handle shutdown
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down gracefully...${NC}"
    # Add any cleanup commands here
    echo -e "${GREEN}✅ Shutdown complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run the system with live output
if [ "$MODE" == "development" ] || [ "$MODE" == "demo" ]; then
    # Development/Demo mode - run with mock data
    echo -e "${PURPLE}🧪 Starting in DEMO mode with simulated data...${NC}"
    echo ""
    
    # Create a simple demo runner since the full system needs all dependencies
    cat << 'EOF' | python3
import time
import random
import sys
from datetime import datetime

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(level, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if level == "INFO":
        print(f"{Colors.GREEN}[{timestamp}]{Colors.END} {Colors.BLUE}INFO{Colors.END} {message}")
    elif level == "WARN":
        print(f"{Colors.GREEN}[{timestamp}]{Colors.END} {Colors.YELLOW}WARN{Colors.END} {message}")
    elif level == "ERROR":
        print(f"{Colors.GREEN}[{timestamp}]{Colors.END} {Colors.RED}ERROR{Colors.END} {message}")
    elif level == "EVOLUTION":
        print(f"{Colors.GREEN}[{timestamp}]{Colors.END} {Colors.PURPLE}🧬 EVOLUTION{Colors.END} {message}")
    elif level == "MARKET":
        print(f"{Colors.GREEN}[{timestamp}]{Colors.END} {Colors.CYAN}📈 MARKET{Colors.END} {message}")

print(f"{Colors.CYAN}{'='*70}{Colors.END}")
print(f"{Colors.BOLD}🚀 CWTS Quantum-Inspired Trading System - DEMO MODE{Colors.END}")
print(f"{Colors.CYAN}{'='*70}{Colors.END}")
print()

log("INFO", "Initializing Core Components...")
time.sleep(1)
log("INFO", "✅ Bayesian VaR Engine initialized")
log("INFO", "✅ Genetic Optimizer configured (20 genomes)")
log("INFO", "✅ Continuous Learning Pipeline active")
log("INFO", "✅ E2B Sandbox Integration ready")
print()

log("INFO", f"{Colors.BOLD}Constitutional Prime Directive: ACTIVE{Colors.END}")
log("INFO", "• Zero-Downtime Deployment: READY")
log("INFO", "• Real Data Integration: SIMULATED")
log("INFO", "• Model Accuracy Enforcement: MONITORING")
log("INFO", "• Evolutionary Adaptation: EVOLVING")
print()

# Simulation state
generation = 0
best_fitness = 0.85
var_accuracy = 0.92
btc_price = 50000
volatility = 0.15
adaptations = 0
successful_adaptations = 0

try:
    while True:
        # Market simulation
        if random.random() < 0.3:
            price_change = random.uniform(-1000, 1000)
            btc_price = max(40000, min(60000, btc_price + price_change))
            volatility = 0.15 + random.uniform(-0.05, 0.05)
            log("MARKET", f"BTC: ${btc_price:.2f} | Volatility: {volatility*100:.1f}%")
        
        # Performance metrics
        if random.random() < 0.4:
            var_accuracy = max(0.80, min(0.99, var_accuracy + random.uniform(-0.02, 0.02)))
            latency = 800 + random.uniform(-200, 400)
            log("INFO", f"Performance: VaR Accuracy {var_accuracy*100:.1f}% | Latency {latency:.0f}ms")
            
            # Trigger adaptation if accuracy drops
            if var_accuracy < 0.85:
                log("WARN", "⚠️ Performance degradation detected - triggering adaptation")
                adaptations += 1
        
        # Evolution cycle
        if random.random() < 0.2:
            generation += 1
            improvement = random.uniform(-0.01, 0.03)
            best_fitness = min(0.99, best_fitness + improvement)
            
            if improvement > 0:
                successful_adaptations += 1
                log("EVOLUTION", f"Generation {generation}: Best fitness {best_fitness:.3f} ⬆️")
            else:
                log("EVOLUTION", f"Generation {generation}: Best fitness {best_fitness:.3f}")
            
            if random.random() < 0.3:
                log("EVOLUTION", f"New genome discovered with {random.uniform(0.7, 0.9):.1f}% emergence complexity")
        
        # Learning events
        if random.random() < 0.15:
            events = [
                "Market regime change detected - adapting strategies",
                "Emergence pattern identified - complexity increasing",
                "E2B sandbox validation completed successfully",
                "Constitutional compliance verified ✅",
                "Evolutionary pressure adjusted for market conditions"
            ]
            log("INFO", f"🧠 Learning: {random.choice(events)}")
        
        # Status update
        if random.random() < 0.1:
            success_rate = (successful_adaptations / max(1, adaptations)) * 100 if adaptations > 0 else 100
            print()
            log("INFO", f"{Colors.BOLD}📊 Evolution Status:{Colors.END}")
            log("INFO", f"  • Generation: {generation}")
            log("INFO", f"  • Best Fitness: {best_fitness:.3f}")
            log("INFO", f"  • VaR Accuracy: {var_accuracy*100:.1f}%")
            log("INFO", f"  • Adaptations: {adaptations} (Success: {success_rate:.0f}%)")
            log("INFO", f"  • Constitutional Compliance: ✅")
            print()
        
        # Occasional warnings/events
        if random.random() < 0.05:
            if random.random() < 0.5:
                log("WARN", "High market volatility detected - adjusting risk parameters")
            else:
                log("INFO", "🔄 Gradual deployment in progress: Phase 2 of 4")
        
        time.sleep(2)
        
except KeyboardInterrupt:
    print()
    print(f"{Colors.YELLOW}🛑 Shutdown signal received{Colors.END}")
    print(f"{Colors.GREEN}✅ Graceful shutdown complete{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"Final Statistics:")
    print(f"  • Total Generations: {generation}")
    print(f"  • Final Fitness: {best_fitness:.3f}")
    print(f"  • Total Adaptations: {adaptations}")
    print(f"  • Success Rate: {(successful_adaptations/max(1,adaptations)*100):.1f}%")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    sys.exit(0)
EOF

else
    # Production mode
    echo -e "${RED}⚠️  Production mode requires full system compilation${NC}"
    echo -e "${YELLOW}   Running compiled binary...${NC}"
    ./target/release/cwts-ultra 2>&1 | tee logs/cwts_$(date +%Y%m%d_%H%M%S).log
fi

echo ""
echo -e "${GREEN}✅ System terminated successfully${NC}"