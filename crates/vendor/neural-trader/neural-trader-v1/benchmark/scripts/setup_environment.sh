#!/bin/bash

# AI News Trading Benchmark - Environment Setup Script
# This script sets up the development and production environment

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-development}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_NAME="${VENV_NAME:-benchmark-venv}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        log_info "Found Python ${python_version}"
        
        if [[ "$(printf '%s\n' "$PYTHON_VERSION" "$python_version" | sort -V | head -n1)" != "$PYTHON_VERSION" ]]; then
            log_error "Python ${PYTHON_VERSION} or higher required, found ${python_version}"
            return 1
        fi
    else
        log_error "Python 3 not found. Please install Python ${PYTHON_VERSION} or higher."
        return 1
    fi
    
    # Check pip
    if ! command_exists pip3; then
        log_error "pip3 not found. Please install pip."
        return 1
    fi
    
    # Check git
    if ! command_exists git; then
        log_warn "git not found. Version control features may not work."
    fi
    
    # Check system packages (Linux/macOS)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt-get; then
            log_info "Detected Ubuntu/Debian system"
            if ! dpkg -l | grep -q python3-dev; then
                log_warn "python3-dev not installed. Some packages may fail to build."
                log_info "Run: sudo apt-get install python3-dev python3-pip build-essential"
            fi
        elif command_exists yum; then
            log_info "Detected RHEL/CentOS system"
            if ! rpm -q python3-devel >/dev/null 2>&1; then
                log_warn "python3-devel not installed. Some packages may fail to build."
                log_info "Run: sudo yum install python3-devel python3-pip gcc"
            fi
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        log_info "Detected macOS system"
        if ! command_exists brew; then
            log_warn "Homebrew not found. Consider installing: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi
    fi
    
    log_info "System requirements check completed"
}

# Setup Python virtual environment
setup_virtual_environment() {
    log_info "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$VENV_NAME" ]]; then
        log_info "Creating virtual environment: $VENV_NAME"
        python3 -m venv "$VENV_NAME"
    else
        log_info "Virtual environment already exists: $VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install wheel for better package building
    pip install wheel
    
    log_info "Virtual environment setup completed"
}

# Install Python dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Make sure we're in the virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_error "Virtual environment not activated"
        return 1
    fi
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing from requirements.txt..."
        pip install -r requirements.txt
    else
        log_error "requirements.txt not found"
        return 1
    fi
    
    # Install development dependencies if in development mode
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log_info "Installing development dependencies..."
        pip install \
            black \
            isort \
            flake8 \
            mypy \
            pytest \
            pytest-cov \
            pytest-asyncio \
            jupyter \
            ipdb \
            pre-commit
        
        # Setup pre-commit hooks
        if command_exists pre-commit; then
            log_info "Setting up pre-commit hooks..."
            pre-commit install
        fi
    fi
    
    log_info "Dependencies installation completed"
}

# Setup project directories
setup_directories() {
    log_info "Setting up project directories..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    local directories=(
        "data"
        "results"
        "logs"
        "monitoring_data"
        "pipeline_data"
        "configs"
        "scripts"
        "tests"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Create .gitkeep files for empty directories
    find data results logs monitoring_data pipeline_data -type d -empty -exec touch {}/.gitkeep \;
    
    log_info "Project directories setup completed"
}

# Setup configuration files
setup_configuration() {
    log_info "Setting up configuration files..."
    
    cd "$PROJECT_ROOT"
    
    # Create default configuration if it doesn't exist
    if [[ ! -f "configs/default.yaml" ]]; then
        log_info "Creating default configuration..."
        mkdir -p configs
        
        cat > configs/default.yaml << 'EOF'
# Default configuration for AI News Trading Benchmark

benchmark:
  environment: development
  log_level: INFO
  data_dir: ./data
  results_dir: ./results
  
simulation:
  initial_capital: 100000
  commission: 0.001
  slippage: 0.0005
  
optimization:
  max_iterations: 100
  convergence_threshold: 0.001
  
monitoring:
  enabled: true
  interval: 1.0
  persistence_enabled: true
  
data_pipeline:
  max_queue_size: 1000
  worker_count: 4
  processing_rate_limit: 100
EOF
    fi
    
    # Create environment-specific configuration
    if [[ ! -f "configs/${ENVIRONMENT}.yaml" && "$ENVIRONMENT" != "default" ]]; then
        log_info "Creating ${ENVIRONMENT} configuration..."
        cp configs/default.yaml "configs/${ENVIRONMENT}.yaml"
    fi
    
    # Create .env file template
    if [[ ! -f ".env" ]]; then
        log_info "Creating .env template..."
        cat > .env << 'EOF'
# Environment variables for AI News Trading Benchmark

# Environment
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database
# POSTGRES_URL=postgresql://user:password@localhost:5432/benchmark
# REDIS_URL=redis://localhost:6379

# API Keys (uncomment and fill in as needed)
# ALPHA_VANTAGE_API_KEY=your_api_key_here
# FINNHUB_API_KEY=your_api_key_here
# NEWS_API_KEY=your_api_key_here

# Monitoring
MONITORING_ENABLED=true
PERFORMANCE_MONITORING=true
EOF
    fi
    
    log_info "Configuration setup completed"
}

# Setup Docker environment
setup_docker() {
    if command_exists docker; then
        log_info "Setting up Docker environment..."
        
        cd "$PROJECT_ROOT"
        
        # Check if docker-compose.yml exists
        if [[ -f "docker/docker-compose.yml" ]]; then
            log_info "Docker configuration found"
            
            # Build Docker images
            if [[ "${BUILD_DOCKER:-false}" == "true" ]]; then
                log_info "Building Docker images..."
                cd docker
                docker-compose build
                cd ..
            fi
        else
            log_warn "Docker configuration not found at docker/docker-compose.yml"
        fi
    else
        log_warn "Docker not installed. Docker features will not be available."
    fi
}

# Run system validation
run_validation() {
    log_info "Running system validation..."
    
    cd "$PROJECT_ROOT"
    
    # Check if we can import the main modules
    python -c "
import sys
sys.path.insert(0, '.')

try:
    from src.integration.system_orchestrator import SystemOrchestrator
    print('✓ System orchestrator import successful')
except ImportError as e:
    print(f'✗ System orchestrator import failed: {e}')
    sys.exit(1)

try:
    from src.integration.data_pipeline import DataPipeline
    print('✓ Data pipeline import successful')
except ImportError as e:
    print(f'✗ Data pipeline import failed: {e}')
    sys.exit(1)

try:
    from src.integration.performance_monitor import PerformanceMonitor
    print('✓ Performance monitor import successful')
except ImportError as e:
    print(f'✗ Performance monitor import failed: {e}')
    sys.exit(1)

print('✓ All core modules imported successfully')
"
    
    if [[ $? -eq 0 ]]; then
        log_info "System validation passed"
    else
        log_error "System validation failed"
        return 1
    fi
}

# Generate activation script
generate_activation_script() {
    log_info "Generating activation script..."
    
    cd "$PROJECT_ROOT"
    
    cat > activate_benchmark.sh << EOF
#!/bin/bash
# Activation script for AI News Trading Benchmark

# Activate virtual environment
source ${VENV_NAME}/bin/activate

# Set environment variables
export PYTHONPATH="\${PWD}:\${PYTHONPATH}"
export BENCHMARK_HOME="\${PWD}"

# Load environment variables
if [[ -f ".env" ]]; then
    export \$(cat .env | grep -v '^#' | xargs)
fi

echo "AI News Trading Benchmark environment activated"
echo "Python: \$(which python)"
echo "Virtual Environment: \$VIRTUAL_ENV"
echo "Project Root: \$BENCHMARK_HOME"

# Show quick help
echo ""
echo "Quick commands:"
echo "  python -m benchmark.cli --help     # Show CLI help"
echo "  python integration_tests.py        # Run integration tests"
echo "  ./scripts/validate_system.sh       # Validate system"
echo "  deactivate                         # Exit virtual environment"
EOF
    
    chmod +x activate_benchmark.sh
    log_info "Activation script created: activate_benchmark.sh"
}

# Print completion message
print_completion_message() {
    log_info "Environment setup completed successfully!"
    echo ""
    log_info "To activate the environment, run:"
    echo "  source activate_benchmark.sh"
    echo ""
    log_info "To test the installation, run:"
    echo "  python integration_tests.py"
    echo ""
    log_info "To start the benchmark system, run:"
    echo "  python -m benchmark.cli start"
    echo ""
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        log_info "Development environment features:"
        echo "  - Pre-commit hooks installed"
        echo "  - Development dependencies available"
        echo "  - Jupyter notebook support"
        echo ""
    fi
    
    if command_exists docker; then
        log_info "Docker environment available:"
        echo "  cd docker && docker-compose up    # Start all services"
        echo "  cd docker && docker-compose up -d # Start in background"
        echo ""
    fi
}

# Main function
main() {
    log_info "Starting AI News Trading Benchmark environment setup..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Python Version: $PYTHON_VERSION"
    log_info "Project Root: $PROJECT_ROOT"
    echo ""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment|-e)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            --venv-name)
                VENV_NAME="$2"
                shift 2
                ;;
            --build-docker)
                BUILD_DOCKER=true
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -e, --environment ENV     Set environment (development/production)"
                echo "  --python-version VERSION Python version requirement (default: 3.11)"
                echo "  --venv-name NAME         Virtual environment name (default: benchmark-venv)"
                echo "  --build-docker           Build Docker images"
                echo "  --skip-validation        Skip system validation"
                echo "  -h, --help               Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Execute setup steps
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    setup_directories
    setup_configuration
    setup_docker
    
    if [[ "${SKIP_VALIDATION:-false}" != "true" ]]; then
        run_validation
    fi
    
    generate_activation_script
    print_completion_message
    
    log_info "Setup completed successfully!"
}

# Run main function
main "$@"