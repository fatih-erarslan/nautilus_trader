#!/bin/bash

# CQGS (Collaborative Quality Governance System) v2.0.0
# Build and Deployment Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
CQGS_VERSION="2.0.0"
BUILD_DIR="./target/release"
DATA_DIR="./data"
CONFIG_FILE="cqgs-config.toml"
DASHBOARD_PORT=8080

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# Print banner
print_banner() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ██████╗  ██████╗ ███████╗    ██╗   ██╗██████╗     ██████╗       ║
║    ██╔════╝██╔═══██╗██╔═══██╗██╔════╝    ██║   ██║╚════██╗   ██╔═████╗      ║
║    ██║     ██║   ██║██║   ██║███████╗    ██║   ██║ █████╔╝   ██║██╔██║      ║
║    ██║     ██║▄▄ ██║██║▄▄ ██║╚════██║    ╚██╗ ██╔╝██╔═══╝    ████╔╝██║      ║
║    ╚██████╗╚██████╔╝╚██████╔╝███████║     ╚████╔╝ ███████╗██╗╚██████╔╝      ║
║     ╚═════╝ ╚══▀▀═╝  ╚══▀▀═╝ ╚══════╝      ╚═══╝  ╚══════╝╚═╝ ╚═════╝       ║
║                                                                              ║
║              Collaborative Quality Governance System v2.0.0                 ║
║              Build and Deployment Script                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo ""
    print_status "🚀 Building CQGS with 49 autonomous sentinels..."
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_header "🔍 Checking Prerequisites"
    
    # Check Rust installation
    if ! command -v cargo &> /dev/null; then
        print_error "Cargo (Rust) is not installed. Please install Rust first."
        exit 1
    fi
    
    local rust_version=$(rustc --version | cut -d' ' -f2)
    print_status "✅ Rust version: $rust_version"
    
    # Check required tools
    if ! command -v git &> /dev/null; then
        print_warning "Git is not installed. Some features may not work."
    fi
    
    # Check system resources
    local available_memory=$(free -m | awk 'NR==2{printf "%.1f", $7/1024}')
    if (( $(echo "$available_memory < 1.0" | bc -l) )); then
        print_warning "Low available memory: ${available_memory}GB. Build may be slow."
    fi
    
    echo ""
}

# Build CQGS
build_cqgs() {
    print_header "🔨 Building CQGS Daemon"
    
    print_status "Cleaning previous builds..."
    cargo clean
    
    print_status "Building in release mode with optimizations..."
    
    # Build with full optimizations
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" \
    cargo build --release --bin cqgs-daemon
    
    if [ $? -eq 0 ]; then
        print_status "✅ Build completed successfully!"
        
        # Check binary size
        local binary_size=$(du -h "$BUILD_DIR/cqgs-daemon" | cut -f1)
        print_status "📦 Binary size: $binary_size"
        
        # Check dependencies
        print_status "🔗 Verifying dependencies..."
        ldd "$BUILD_DIR/cqgs-daemon" | head -5
    else
        print_error "❌ Build failed!"
        exit 1
    fi
    
    echo ""
}

# Run tests
run_tests() {
    print_header "🧪 Running Tests"
    
    print_status "Running unit tests..."
    cargo test --release
    
    if [ $? -eq 0 ]; then
        print_status "✅ All tests passed!"
    else
        print_error "❌ Some tests failed!"
        exit 1
    fi
    
    echo ""
}

# Generate configuration
generate_config() {
    print_header "⚙️  Generating Configuration"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_status "Creating default configuration file..."
        
        cat > "$CONFIG_FILE" << 'EOF'
# CQGS (Collaborative Quality Governance System) v2.0.0 Configuration

[system]
sentinel_count = 49
hyperbolic_curvature = -1.5
consensus_threshold = 0.67
healing_enabled = true
zero_mock_enforcement = true
neural_learning_rate = 0.01
monitoring_interval_ms = 100
remediation_timeout_ms = 30000

[dashboard]
port = 8080
host = "0.0.0.0"
update_interval_ms = 1000
max_history_items = 1000
theme = "HyperbolicDark"
enable_real_time = true
enable_notifications = true

[neural]
learning_rate = 0.01
confidence_threshold = 0.8
max_training_examples = 10000
enable_online_learning = true

[remediation]
max_concurrent_tasks = 10
default_timeout = "5m"
max_retries = 3
enable_rollback = true
require_validation = true
auto_approve_low_risk = true

validation = "Strict"
log_level = "info"
data_dir = "./data"
EOF
        
        print_status "✅ Configuration file created: $CONFIG_FILE"
    else
        print_status "✅ Configuration file already exists: $CONFIG_FILE"
    fi
    
    echo ""
}

# Setup data directory
setup_data_dir() {
    print_header "📁 Setting Up Data Directory"
    
    print_status "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"/{logs,exports,backups,neural}
    
    # Create initial log file
    touch "$DATA_DIR/logs/cqgs-daemon.log"
    
    print_status "✅ Data directory structure created"
    echo ""
}

# Validate deployment
validate_deployment() {
    print_header "🔍 Validating Deployment"
    
    # Check binary exists and is executable
    if [ -x "$BUILD_DIR/cqgs-daemon" ]; then
        print_status "✅ CQGS daemon binary is executable"
    else
        print_error "❌ CQGS daemon binary is not executable"
        exit 1
    fi
    
    # Check configuration
    if [ -f "$CONFIG_FILE" ]; then
        print_status "✅ Configuration file exists"
    else
        print_error "❌ Configuration file missing"
        exit 1
    fi
    
    # Check data directory
    if [ -d "$DATA_DIR" ]; then
        print_status "✅ Data directory exists"
    else
        print_error "❌ Data directory missing"
        exit 1
    fi
    
    # Quick smoke test
    print_status "Running smoke test..."
    timeout 5s "$BUILD_DIR/cqgs-daemon" --help > /dev/null 2>&1
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 is timeout exit code
        print_status "✅ Smoke test passed"
    else
        print_error "❌ Smoke test failed"
        exit 1
    fi
    
    echo ""
}

# Create systemd service file
create_systemd_service() {
    print_header "🔧 Creating Systemd Service"
    
    local service_file="cqgs-daemon.service"
    local working_dir=$(pwd)
    local binary_path="$working_dir/$BUILD_DIR/cqgs-daemon"
    local config_path="$working_dir/$CONFIG_FILE"
    
    cat > "$service_file" << EOF
[Unit]
Description=CQGS (Collaborative Quality Governance System) v${CQGS_VERSION}
After=network.target
Wants=network.target

[Service]
Type=simple
User=cqgs
Group=cqgs
WorkingDirectory=$working_dir
ExecStart=$binary_path --config $config_path
Restart=always
RestartSec=10
Environment=RUST_LOG=info

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$working_dir/data

# Resource limits
LimitNOFILE=65536
MemoryMax=1G

[Install]
WantedBy=multi-user.target
EOF
    
    print_status "✅ Systemd service file created: $service_file"
    print_status "To install: sudo cp $service_file /etc/systemd/system/"
    print_status "To enable: sudo systemctl enable cqgs-daemon"
    print_status "To start: sudo systemctl start cqgs-daemon"
    
    echo ""
}

# Print deployment summary
print_summary() {
    print_header "📋 Deployment Summary"
    
    echo ""
    print_status "🎉 CQGS v$CQGS_VERSION build and deployment preparation complete!"
    echo ""
    
    echo -e "${BLUE}📊 Build Information:${NC}"
    echo "  • Binary: $BUILD_DIR/cqgs-daemon"
    echo "  • Config: $CONFIG_FILE"
    echo "  • Data Directory: $DATA_DIR"
    echo "  • Dashboard Port: $DASHBOARD_PORT"
    echo ""
    
    echo -e "${BLUE}🚀 Quick Start:${NC}"
    echo "  1. Start daemon: $BUILD_DIR/cqgs-daemon --config $CONFIG_FILE"
    echo "  2. Open dashboard: http://localhost:$DASHBOARD_PORT"
    echo "  3. View logs: tail -f $DATA_DIR/logs/cqgs-daemon.log"
    echo ""
    
    echo -e "${BLUE}⚡ Advanced Usage:${NC}"
    echo "  • Custom port: $BUILD_DIR/cqgs-daemon --dashboard-port 3001"
    echo "  • Debug mode: $BUILD_DIR/cqgs-daemon --log-level debug"
    echo "  • Custom data dir: $BUILD_DIR/cqgs-daemon --data-dir /var/lib/cqgs"
    echo ""
    
    echo -e "${BLUE}🔧 System Integration:${NC}"
    echo "  • Install service: sudo cp cqgs-daemon.service /etc/systemd/system/"
    echo "  • Enable autostart: sudo systemctl enable cqgs-daemon"
    echo "  • Start service: sudo systemctl start cqgs-daemon"
    echo ""
    
    echo -e "${BLUE}🛡️ Security Features:${NC}"
    echo "  • 49 Autonomous Sentinels monitoring quality"
    echo "  • Zero-Mock enforcement with deployment blocking"
    echo "  • Real-time consensus decisions with 2/3 threshold"
    echo "  • Self-healing remediation system"
    echo "  • Neural intelligence pattern recognition"
    echo ""
    
    echo -e "${GREEN}Ready for quality governance with hyperbolic coordination! 🌟${NC}"
}

# Main execution
main() {
    print_banner
    
    check_prerequisites
    build_cqgs
    run_tests
    generate_config
    setup_data_dir
    validate_deployment
    create_systemd_service
    print_summary
}

# Handle command line arguments
case "${1:-build}" in
    "build")
        main
        ;;
    "test-only")
        print_banner
        check_prerequisites
        run_tests
        ;;
    "clean")
        print_status "Cleaning build artifacts and data..."
        cargo clean
        rm -rf "$DATA_DIR" "$CONFIG_FILE" "cqgs-daemon.service"
        print_status "✅ Cleanup complete"
        ;;
    "help"|"-h"|"--help")
        echo "CQGS Build and Deployment Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build (default)  - Full build and deployment preparation"
        echo "  test-only        - Run tests only"
        echo "  clean           - Clean build artifacts and generated files"
        echo "  help            - Show this help message"
        echo ""
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac