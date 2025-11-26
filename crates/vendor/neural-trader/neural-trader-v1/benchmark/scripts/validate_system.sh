#!/bin/bash

# AI News Trading Benchmark - System Validation Script
# This script performs comprehensive system health checks and validation

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VALIDATION_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to add validation result
add_result() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    VALIDATION_RESULTS+=("$test_name|$status|$message")
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [[ "$status" == "PASS" ]]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        log_success "$test_name: $message"
    else
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        log_fail "$test_name: $message"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validate Python environment
validate_python_environment() {
    log_info "Validating Python environment..."
    
    # Check Python version
    if command_exists python; then
        local python_version=$(python --version 2>&1 | cut -d' ' -f2)
        if [[ "$python_version" > "3.10" ]]; then
            add_result "Python Version" "PASS" "Python $python_version (>= 3.11 required)"
        else
            add_result "Python Version" "FAIL" "Python $python_version (< 3.11)"
        fi
    else
        add_result "Python Installation" "FAIL" "Python not found"
    fi
    
    # Check virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        add_result "Virtual Environment" "PASS" "Active: $(basename $VIRTUAL_ENV)"
    else
        add_result "Virtual Environment" "FAIL" "No virtual environment active"
    fi
    
    # Check pip
    if command_exists pip; then
        local pip_version=$(pip --version | cut -d' ' -f2)
        add_result "Pip Installation" "PASS" "pip $pip_version"
    else
        add_result "Pip Installation" "FAIL" "pip not found"
    fi
}

# Validate required Python packages
validate_python_packages() {
    log_info "Validating Python packages..."
    
    # Core packages
    local required_packages=(
        "click"
        "pytest"
        "numpy"
        "pandas" 
        "psutil"
        "aiohttp"
        "redis"
        "pyyaml"
    )
    
    for package in "${required_packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            local version=$(python -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null)
            add_result "Package: $package" "PASS" "Version: $version"
        else
            add_result "Package: $package" "FAIL" "Not installed or importable"
        fi
    done
    
    # Integration modules
    local integration_modules=(
        "src.integration.system_orchestrator"
        "src.integration.component_registry"
        "src.integration.data_pipeline"
        "src.integration.performance_monitor"
    )
    
    cd "$PROJECT_ROOT"
    for module in "${integration_modules[@]}"; do
        if python -c "import $module" 2>/dev/null; then
            add_result "Module: $module" "PASS" "Successfully imported"
        else
            add_result "Module: $module" "FAIL" "Import failed"
        fi
    done
}

# Validate system resources
validate_system_resources() {
    log_info "Validating system resources..."
    
    # Check available memory
    if command_exists free; then
        local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2}')
        if [[ $total_mem -gt 4096 ]]; then
            add_result "System Memory" "PASS" "${total_mem}MB available"
        else
            add_result "System Memory" "WARN" "${total_mem}MB available (4GB+ recommended)"
        fi
    elif command_exists sysctl; then
        # macOS
        local total_mem=$(sysctl -n hw.memsize | awk '{printf "%.0f", $1/1024/1024}')
        if [[ $total_mem -gt 4096 ]]; then
            add_result "System Memory" "PASS" "${total_mem}MB available"
        else
            add_result "System Memory" "WARN" "${total_mem}MB available (4GB+ recommended)"
        fi
    fi
    
    # Check available disk space
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {printf "%.0f", $4/1024}')
    if [[ $available_space -gt 1024 ]]; then
        add_result "Disk Space" "PASS" "${available_space}MB available"
    else
        add_result "Disk Space" "WARN" "${available_space}MB available (1GB+ recommended)"
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    if [[ "$cpu_cores" != "unknown" && $cpu_cores -gt 2 ]]; then
        add_result "CPU Cores" "PASS" "$cpu_cores cores available"
    else
        add_result "CPU Cores" "WARN" "$cpu_cores cores (4+ recommended)"
    fi
}

# Validate project structure
validate_project_structure() {
    log_info "Validating project structure..."
    
    cd "$PROJECT_ROOT"
    
    # Required directories
    local required_dirs=(
        "src/integration"
        "src/benchmarks"
        "src/simulation"
        "src/optimization"
        "src/data"
        "docker"
        "scripts"
        "configs"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            add_result "Directory: $dir" "PASS" "Exists"
        else
            add_result "Directory: $dir" "FAIL" "Missing"
        fi
    done
    
    # Required files
    local required_files=(
        "requirements.txt"
        "integration_tests.py"
        "docker/Dockerfile"
        "docker/docker-compose.yml"
        "docker/entrypoint.sh"
        "src/integration/__init__.py"
        "src/integration/system_orchestrator.py"
        "src/integration/component_registry.py"
        "src/integration/data_pipeline.py"
        "src/integration/performance_monitor.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            add_result "File: $file" "PASS" "Exists"
        else
            add_result "File: $file" "FAIL" "Missing"
        fi
    done
}

# Validate configuration files
validate_configuration() {
    log_info "Validating configuration files..."
    
    cd "$PROJECT_ROOT"
    
    # Check YAML configuration files
    if [[ -f "configs/default.yaml" ]]; then
        if python -c "import yaml; yaml.safe_load(open('configs/default.yaml'))" 2>/dev/null; then
            add_result "Configuration: default.yaml" "PASS" "Valid YAML"
        else
            add_result "Configuration: default.yaml" "FAIL" "Invalid YAML"
        fi
    else
        add_result "Configuration: default.yaml" "FAIL" "Missing"
    fi
    
    # Check .env file
    if [[ -f ".env" ]]; then
        add_result "Environment File" "PASS" ".env exists"
    else
        add_result "Environment File" "WARN" ".env missing (optional)"
    fi
}

# Test basic functionality
test_basic_functionality() {
    log_info "Testing basic functionality..."
    
    cd "$PROJECT_ROOT"
    
    # Test CLI import
    if python -c "from src.cli.benchmark_cli import BenchmarkCLI" 2>/dev/null; then
        add_result "CLI Import" "PASS" "CLI modules importable"
    else
        add_result "CLI Import" "FAIL" "CLI import failed"
    fi
    
    # Test system orchestrator creation
    if python -c "
from src.integration.system_orchestrator import SystemOrchestrator
orchestrator = SystemOrchestrator()
print('SystemOrchestrator created successfully')
" 2>/dev/null; then
        add_result "Orchestrator Creation" "PASS" "Can create system orchestrator"
    else
        add_result "Orchestrator Creation" "FAIL" "Cannot create system orchestrator"
    fi
    
    # Test component registry
    if python -c "
from src.integration.component_registry import ComponentRegistry
registry = ComponentRegistry()
print('ComponentRegistry created successfully')
" 2>/dev/null; then
        add_result "Component Registry" "PASS" "Can create component registry"
    else
        add_result "Component Registry" "FAIL" "Cannot create component registry"
    fi
    
    # Test performance monitor
    if python -c "
from src.integration.performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
print('PerformanceMonitor created successfully')
" 2>/dev/null; then
        add_result "Performance Monitor" "PASS" "Can create performance monitor"
    else
        add_result "Performance Monitor" "FAIL" "Cannot create performance monitor"
    fi
}

# Test Docker setup
validate_docker_setup() {
    log_info "Validating Docker setup..."
    
    if command_exists docker; then
        add_result "Docker Installation" "PASS" "Docker available"
        
        # Check if Docker daemon is running
        if docker info >/dev/null 2>&1; then
            add_result "Docker Daemon" "PASS" "Docker daemon running"
        else
            add_result "Docker Daemon" "WARN" "Docker daemon not running"
        fi
        
        # Check docker-compose
        if command_exists docker-compose; then
            add_result "Docker Compose" "PASS" "docker-compose available"
        else
            add_result "Docker Compose" "WARN" "docker-compose not available"
        fi
        
        # Validate Dockerfile
        cd "$PROJECT_ROOT"
        if [[ -f "docker/Dockerfile" ]]; then
            if docker build -t benchmark-test -f docker/Dockerfile --target base . >/dev/null 2>&1; then
                add_result "Dockerfile Build" "PASS" "Dockerfile builds successfully"
                # Clean up test image
                docker rmi benchmark-test >/dev/null 2>&1 || true
            else
                add_result "Dockerfile Build" "FAIL" "Dockerfile build failed"
            fi
        fi
    else
        add_result "Docker Installation" "WARN" "Docker not installed (optional)"
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -f "integration_tests.py" ]]; then
        # Run a quick smoke test
        if python -c "
import integration_tests
suite = integration_tests.IntegrationTestSuite()
print('Integration test suite can be created')
" 2>/dev/null; then
            add_result "Integration Tests" "PASS" "Test suite can be created"
        else
            add_result "Integration Tests" "FAIL" "Test suite creation failed"
        fi
        
        # Try to run a basic test (if pytest is available)
        if command_exists pytest; then
            if timeout 30 python -m pytest integration_tests.py::TestIntegrationLayer::test_component_registry_functionality -v >/dev/null 2>&1; then
                add_result "Sample Test Execution" "PASS" "Sample test passed"
            else
                add_result "Sample Test Execution" "WARN" "Sample test failed or timed out"
            fi
        fi
    else
        add_result "Integration Tests" "FAIL" "integration_tests.py not found"
    fi
}

# Check network connectivity
validate_network() {
    log_info "Validating network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        add_result "Internet Connectivity" "PASS" "Can reach external servers"
    else
        add_result "Internet Connectivity" "WARN" "No internet connectivity"
    fi
    
    # Check if common ports are available
    local ports_to_check=(8000 8080 9090 3000 6379 5432)
    for port in "${ports_to_check[@]}"; do
        if ! netstat -ln 2>/dev/null | grep -q ":$port "; then
            add_result "Port $port" "PASS" "Port available"
        else
            add_result "Port $port" "WARN" "Port in use"
        fi
    done
}

# Generate validation report
generate_report() {
    log_info "Generating validation report..."
    
    local report_file="$PROJECT_ROOT/validation_report.txt"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    {
        echo "=================================="
        echo "AI News Trading Benchmark Validation Report"
        echo "=================================="
        echo "Generated: $timestamp"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "Passed: $PASSED_CHECKS"
        echo "Failed: $FAILED_CHECKS"
        echo "Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%"
        echo ""
        echo "Detailed Results:"
        echo "=================================="
        
        for result in "${VALIDATION_RESULTS[@]}"; do
            IFS='|' read -r test_name status message <<< "$result"
            printf "%-30s %-6s %s\n" "$test_name" "$status" "$message"
        done
        
        echo ""
        echo "=================================="
        
        if [[ $FAILED_CHECKS -eq 0 ]]; then
            echo "✅ All validations passed! System is ready."
        elif [[ $FAILED_CHECKS -lt 5 ]]; then
            echo "⚠️  Minor issues detected. System may work with limitations."
        else
            echo "❌ Significant issues detected. System may not function properly."
        fi
        
        echo ""
        echo "Recommendations:"
        echo "- If tests failed, run: ./scripts/setup_environment.sh"
        echo "- For Docker issues, ensure Docker is running"
        echo "- For Python issues, check virtual environment activation"
        echo "- For missing packages, run: pip install -r requirements.txt"
        
    } | tee "$report_file"
    
    log_info "Report saved to: $report_file"
}

# Print summary
print_summary() {
    echo ""
    echo "=================================="
    echo "VALIDATION SUMMARY"
    echo "=================================="
    echo "Total Checks: $TOTAL_CHECKS"
    echo "✅ Passed: $PASSED_CHECKS"
    echo "❌ Failed: $FAILED_CHECKS"
    echo "Success Rate: $(( PASSED_CHECKS * 100 / TOTAL_CHECKS ))%"
    echo "=================================="
    echo ""
    
    if [[ $FAILED_CHECKS -eq 0 ]]; then
        log_success "🎉 All validations passed! System is fully operational."
        echo ""
        echo "Next steps:"
        echo "  python -m benchmark.cli --help          # Show CLI help"
        echo "  python integration_tests.py             # Run full test suite"
        echo "  cd docker && docker-compose up         # Start with Docker"
        return 0
    elif [[ $FAILED_CHECKS -lt 5 ]]; then
        log_warn "⚠️  System has minor issues but may still function."
        echo ""
        echo "Consider running: ./scripts/setup_environment.sh"
        return 1
    else
        log_error "❌ System has significant issues and may not function properly."
        echo ""
        echo "Please run: ./scripts/setup_environment.sh"
        return 2
    fi
}

# Main function
main() {
    log_info "Starting AI News Trading Benchmark system validation..."
    echo ""
    
    # Parse command line arguments
    local quick_mode=false
    local skip_docker=false
    local skip_tests=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick|-q)
                quick_mode=true
                shift
                ;;
            --skip-docker)
                skip_docker=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -q, --quick      Quick validation (skip time-consuming tests)"
                echo "  --skip-docker    Skip Docker validation"
                echo "  --skip-tests     Skip integration tests"
                echo "  -h, --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run validation checks
    validate_python_environment
    validate_python_packages
    validate_system_resources
    validate_project_structure
    validate_configuration
    test_basic_functionality
    
    if [[ "$skip_docker" != "true" ]]; then
        validate_docker_setup
    fi
    
    if [[ "$quick_mode" != "true" ]]; then
        validate_network
        
        if [[ "$skip_tests" != "true" ]]; then
            run_integration_tests
        fi
    fi
    
    # Generate report and summary
    generate_report
    print_summary
}

# Run main function
main "$@"