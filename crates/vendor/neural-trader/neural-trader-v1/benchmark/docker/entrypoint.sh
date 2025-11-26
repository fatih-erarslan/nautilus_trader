#!/bin/bash

# AI News Trading Benchmark - Docker Entrypoint Script
# This script handles container startup, environment setup, and service orchestration

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
    if [[ "${LOG_LEVEL}" == "DEBUG" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Environment setup
setup_environment() {
    log_info "Setting up environment..."
    
    # Set default values
    export ENVIRONMENT=${ENVIRONMENT:-production}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export WORKERS=${WORKERS:-2}
    export BENCHMARK_HOME=${BENCHMARK_HOME:-/app}
    export PYTHONPATH="${BENCHMARK_HOME}:${PYTHONPATH}"
    
    # Create necessary directories
    mkdir -p "${BENCHMARK_HOME}/data"
    mkdir -p "${BENCHMARK_HOME}/results"
    mkdir -p "${BENCHMARK_HOME}/logs"
    mkdir -p "${BENCHMARK_HOME}/monitoring_data"
    mkdir -p "${BENCHMARK_HOME}/pipeline_data"
    
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Log Level: ${LOG_LEVEL}"
    log_info "Workers: ${WORKERS}"
}

# Health check function
health_check() {
    log_info "Performing health check..."
    
    # Check Python environment
    if ! python -c "import sys; print(f'Python {sys.version}')"; then
        log_error "Python environment check failed"
        return 1
    fi
    
    # Check required modules
    local required_modules=("asyncio" "psutil" "numpy" "pandas")
    for module in "${required_modules[@]}"; do
        if ! python -c "import ${module}"; then
            log_error "Required module ${module} not found"
            return 1
        fi
    done
    
    # Check file permissions
    if [[ ! -w "${BENCHMARK_HOME}/data" ]]; then
        log_error "Data directory not writable"
        return 1
    fi
    
    log_info "Health check passed"
    return 0
}

# Wait for dependencies
wait_for_dependencies() {
    log_info "Waiting for dependencies..."
    
    # Wait for Redis if URL is provided
    if [[ -n "${REDIS_URL}" ]]; then
        log_info "Waiting for Redis..."
        local redis_host=$(echo "${REDIS_URL}" | sed -n 's|redis://\([^:]*\):.*|\1|p')
        local redis_port=$(echo "${REDIS_URL}" | sed -n 's|redis://[^:]*:\([0-9]*\).*|\1|p')
        
        for i in {1..30}; do
            if timeout 5 bash -c "echo > /dev/tcp/${redis_host}/${redis_port}"; then
                log_info "Redis is ready"
                break
            fi
            if [[ $i -eq 30 ]]; then
                log_error "Redis connection timeout"
                return 1
            fi
            sleep 1
        done
    fi
    
    # Wait for PostgreSQL if URL is provided
    if [[ -n "${POSTGRES_URL}" ]]; then
        log_info "Waiting for PostgreSQL..."
        local postgres_host=$(echo "${POSTGRES_URL}" | sed -n 's|postgresql://[^@]*@\([^:]*\):.*|\1|p')
        local postgres_port=$(echo "${POSTGRES_URL}" | sed -n 's|postgresql://[^@]*@[^:]*:\([0-9]*\)/.*|\1|p')
        
        for i in {1..30}; do
            if timeout 5 bash -c "echo > /dev/tcp/${postgres_host}/${postgres_port}"; then
                log_info "PostgreSQL is ready"
                break
            fi
            if [[ $i -eq 30 ]]; then
                log_error "PostgreSQL connection timeout"
                return 1
            fi
            sleep 1
        done
    fi
}

# Initialize database
init_database() {
    if [[ -n "${POSTGRES_URL}" ]]; then
        log_info "Initializing database..."
        
        # Run database migrations if they exist
        if [[ -f "${BENCHMARK_HOME}/scripts/init_db.sql" ]]; then
            log_info "Running database initialization script..."
            python -c "
import psycopg2
import os
conn = psycopg2.connect(os.environ['POSTGRES_URL'])
cur = conn.cursor()
with open('${BENCHMARK_HOME}/scripts/init_db.sql', 'r') as f:
    cur.execute(f.read())
conn.commit()
conn.close()
" || log_warn "Database initialization failed (may already be initialized)"
        fi
    fi
}

# Start system orchestrator
start_orchestrator() {
    log_info "Starting system orchestrator..."
    
    # Start the main system orchestrator in the background
    python -m benchmark.integration.system_orchestrator &
    local orchestrator_pid=$!
    
    # Wait for orchestrator to be ready
    for i in {1..30}; do
        if python -c "
import requests
try:
    r = requests.get('http://localhost:8000/health', timeout=1)
    exit(0 if r.status_code == 200 else 1)
except:
    exit(1)
        "; then
            log_info "System orchestrator is ready"
            return 0
        fi
        sleep 1
    done
    
    log_error "System orchestrator failed to start"
    return 1
}

# Start workers based on type
start_worker() {
    local worker_type="${1:-pipeline}"
    log_info "Starting ${worker_type} worker..."
    
    case "${worker_type}" in
        "pipeline")
            exec python -m benchmark.integration.data_pipeline --worker
            ;;
        "optimization")
            exec python -m benchmark.optimization.optimizer --worker
            ;;
        "simulation")
            exec python -m benchmark.simulation.simulator --worker
            ;;
        "monitoring")
            exec python -m benchmark.monitoring.dashboard
            ;;
        *)
            log_error "Unknown worker type: ${worker_type}"
            exit 1
            ;;
    esac
}

# Start CLI mode
start_cli() {
    log_info "Starting CLI mode..."
    
    # If no arguments provided, show help
    if [[ $# -eq 0 ]]; then
        exec python -m benchmark.cli --help
    else
        exec python -m benchmark.cli "$@"
    fi
}

# Start web UI
start_web() {
    log_info "Starting web UI..."
    exec python -m benchmark.web.app
}

# Start monitoring dashboard
start_monitoring() {
    log_info "Starting monitoring dashboard..."
    exec python -m benchmark.monitoring.dashboard
}

# Development mode
start_development() {
    log_info "Starting development mode..."
    
    # Install development dependencies if needed
    if [[ ! -f "/app/.dev_deps_installed" ]]; then
        log_info "Installing development dependencies..."
        pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy jupyter
        touch "/app/.dev_deps_installed"
    fi
    
    # Start Jupyter Lab if requested
    if [[ "${JUPYTER_ENABLE_LAB}" == "yes" ]]; then
        log_info "Starting Jupyter Lab..."
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    else
        exec python -m benchmark.cli status
    fi
}

# Testing mode
start_testing() {
    log_info "Starting testing mode..."
    
    # Run tests with coverage
    exec pytest ${PYTEST_ARGS:--v --tb=short --cov=benchmark --cov-report=html:/app/test-results/coverage}
}

# Signal handlers
handle_sigterm() {
    log_info "Received SIGTERM, shutting down gracefully..."
    # Send shutdown signal to child processes
    pkill -TERM -P $$
    wait
    exit 0
}

handle_sigint() {
    log_info "Received SIGINT, shutting down gracefully..."
    # Send shutdown signal to child processes
    pkill -INT -P $$
    wait
    exit 0
}

# Set up signal handlers
trap handle_sigterm SIGTERM
trap handle_sigint SIGINT

# Main execution
main() {
    log_info "Starting AI News Trading Benchmark Container..."
    log_info "Arguments: $*"
    
    # Setup environment
    setup_environment
    
    # Perform health check
    if ! health_check; then
        log_error "Health check failed"
        exit 1
    fi
    
    # Wait for dependencies (only in production)
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        wait_for_dependencies
        init_database
    fi
    
    # Determine startup mode based on arguments
    case "${1:-benchmark}" in
        "orchestrator"|"start")
            start_orchestrator
            ;;
        "worker")
            start_worker "${2:-pipeline}"
            ;;
        "cli")
            shift
            start_cli "$@"
            ;;
        "web")
            start_web
            ;;
        "monitoring")
            start_monitoring
            ;;
        "development"|"dev")
            start_development
            ;;
        "test"|"testing")
            start_testing
            ;;
        "benchmark")
            # Default benchmark command
            start_cli "$@"
            ;;
        *)
            # Pass through to CLI
            start_cli "$@"
            ;;
    esac
}

# Run main function
main "$@"