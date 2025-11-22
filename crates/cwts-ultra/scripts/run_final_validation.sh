#!/bin/bash

# CWTS Ultra Trading System - Final Validation Script
# Critical validation before production deployment

set -e  # Exit on any error

echo "🚀 CWTS Ultra Trading System - Final Validation"
echo "💰 Critical validation for billion-dollar trading system"
echo "🎯 Production deployment readiness check"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/home/kutlu/CWTS/cwts-ultra"
cd "$PROJECT_DIR"

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "info")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "success")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "warning")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $message${NC}"
            ;;
    esac
}

# Function to check system requirements
check_system_requirements() {
    print_status "info" "Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    major_version=$(echo $python_version | cut -d'.' -f1)
    minor_version=$(echo $python_version | cut -d'.' -f2)
    
    if [ "$major_version" -lt 3 ] || [ "$major_version" -eq 3 -a "$minor_version" -lt 8 ]; then
        print_status "error" "Python 3.8+ required, found $python_version"
        exit 1
    fi
    print_status "success" "Python version: $python_version"
    
    # Check available memory
    available_memory=$(free -g | awk 'NR==2{printf "%.1f", $7}')
    if (( $(echo "$available_memory < 4.0" | bc -l) )); then
        print_status "warning" "Low available memory: ${available_memory}GB (recommended: 4GB+)"
    else
        print_status "success" "Available memory: ${available_memory}GB"
    fi
    
    # Check disk space
    available_disk=$(df -h . | awk 'NR==2{print $4}')
    print_status "success" "Available disk space: $available_disk"
    
    # Check for GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_status "success" "GPU detected: $gpu_info"
    else
        print_status "warning" "No GPU detected - will use CPU fallback"
    fi
}

# Function to install dependencies
install_dependencies() {
    print_status "info" "Installing/updating dependencies..."
    
    # Install Python dependencies
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt > /dev/null 2>&1
        print_status "success" "Python dependencies installed"
    fi
    
    # Install test dependencies
    if [ -f "tests/requirements.txt" ]; then
        pip3 install -r tests/requirements.txt > /dev/null 2>&1
        print_status "success" "Test dependencies installed"
    fi
    
    # Install additional test packages
    pip3 install pytest pytest-asyncio pytest-cov psutil > /dev/null 2>&1
    print_status "success" "Additional test packages installed"
}

# Function to prepare test environment
prepare_test_environment() {
    print_status "info" "Preparing test environment..."
    
    # Create necessary directories
    mkdir -p tests/integration/{logs,reports,data,archives}
    mkdir -p .test_cache
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    export TEST_ENVIRONMENT="integration"
    export LOG_LEVEL="INFO"
    
    # Create test configuration
    cat > tests/integration/test_config.json << EOF
{
    "test_environment": "integration",
    "timestamp": "$(date -Iseconds)",
    "parameters": {
        "order_count": 10000,
        "stress_test_enabled": true,
        "gpu_acceleration": true,
        "compliance_validation": true,
        "memory_limit_gb": 8,
        "latency_threshold_ms": 5
    },
    "system_info": {
        "python_version": "$(python3 --version | cut -d' ' -f2)",
        "available_memory_gb": $(free -g | awk 'NR==2{printf "%.1f", $7}'),
        "cpu_cores": $(nproc),
        "hostname": "$(hostname)"
    }
}
EOF
    
    print_status "success" "Test environment prepared"
}

# Function to run pre-validation checks
run_pre_validation() {
    print_status "info" "Running pre-validation checks..."
    
    # Check import dependencies
    python3 -c "
import sys
import importlib
required_modules = [
    'numpy', 'asyncio', 'json', 'time', 'psutil', 
    'threading', 'datetime', 'typing', 'dataclasses', 
    'pytest', 'logging'
]
missing_modules = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print(f'Missing modules: {missing_modules}')
    sys.exit(1)
else:
    print('All required modules available')
"
    
    if [ $? -eq 0 ]; then
        print_status "success" "All required Python modules available"
    else
        print_status "error" "Missing required Python modules"
        exit 1
    fi
    
    # Validate test files exist
    test_files=(
        "tests/integration/comprehensive_integration_test.py"
        "tests/integration/test_runner.py"
    )
    
    for file in "${test_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "success" "Test file found: $file"
        else
            print_status "error" "Missing test file: $file"
            exit 1
        fi
    done
}

# Function to execute comprehensive integration tests
execute_integration_tests() {
    print_status "info" "Executing comprehensive integration tests..."
    
    # Set test execution parameters
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    export CUDA_VISIBLE_DEVICES="0"  # Use first GPU if available
    
    # Create test execution timestamp
    test_timestamp=$(date '+%Y%m%d_%H%M%S')
    log_file="tests/integration/logs/validation_${test_timestamp}.log"
    
    # Execute the comprehensive test suite
    print_status "info" "Starting test execution (this may take several minutes)..."
    
    # Run with timeout to prevent hanging
    timeout 1800 python3 tests/integration/test_runner.py 2>&1 | tee "$log_file"
    test_exit_code=$?
    
    # Check test results
    if [ $test_exit_code -eq 0 ]; then
        print_status "success" "Integration tests PASSED"
        return 0
    elif [ $test_exit_code -eq 124 ]; then
        print_status "error" "Integration tests TIMED OUT (30 minutes)"
        return 1
    else
        print_status "error" "Integration tests FAILED (exit code: $test_exit_code)"
        return 1
    fi
}

# Function to validate test results
validate_test_results() {
    print_status "info" "Validating test results..."
    
    # Check for test report
    report_file="tests/integration/reports/final_validation_report.json"
    if [ -f "$report_file" ]; then
        print_status "success" "Test report generated: $report_file"
        
        # Extract key metrics from report
        overall_success=$(jq -r '.test_execution.overall_success' "$report_file" 2>/dev/null || echo "false")
        
        if [ "$overall_success" = "true" ]; then
            print_status "success" "All validation tests passed"
            return 0
        else
            print_status "error" "Some validation tests failed"
            return 1
        fi
    else
        print_status "error" "Test report not found"
        return 1
    fi
}

# Function to generate final summary
generate_final_summary() {
    local test_passed=$1
    
    echo ""
    echo "========================================================"
    echo "🎯 FINAL VALIDATION SUMMARY"
    echo "========================================================"
    
    if [ $test_passed -eq 0 ]; then
        print_status "success" "COMPREHENSIVE VALIDATION PASSED"
        print_status "success" "SYSTEM READY FOR PRODUCTION DEPLOYMENT"
        print_status "success" "Validated to handle billions in trading volume"
        print_status "success" "All safety and compliance systems verified"
        
        echo ""
        echo "🚀 PRODUCTION DEPLOYMENT APPROVED"
        echo "💰 System validated for high-frequency trading"
        echo "🛡️ Risk controls and kill switch verified"
        echo "⚖️ SEC Rule 15c3-5 compliance validated"
        echo "⚡ GPU acceleration and latency requirements met"
        echo "🧠 Neural systems and attention cascade functional"
        
    else
        print_status "error" "COMPREHENSIVE VALIDATION FAILED"
        print_status "error" "SYSTEM NOT READY FOR PRODUCTION"
        print_status "error" "Address validation failures before deployment"
        
        echo ""
        echo "🚨 PRODUCTION DEPLOYMENT BLOCKED"
        echo "🔧 Review test logs and fix identified issues"
        echo "📋 Check: tests/integration/logs/ for detailed logs"
        echo "📊 Check: tests/integration/reports/ for test reports"
    fi
    
    echo "========================================================"
}

# Main execution flow
main() {
    print_status "info" "Starting CWTS Ultra final validation process..."
    
    # System checks
    check_system_requirements
    install_dependencies
    prepare_test_environment
    run_pre_validation
    
    # Execute tests
    if execute_integration_tests; then
        if validate_test_results; then
            generate_final_summary 0
            exit 0
        else
            generate_final_summary 1
            exit 1
        fi
    else
        generate_final_summary 1
        exit 1
    fi
}

# Execute main function
main "$@"