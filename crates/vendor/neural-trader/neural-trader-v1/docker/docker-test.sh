#!/bin/bash
# Docker Test Script - Comprehensive testing in containerized environment
# Usage: ./scripts/docker-test.sh [OPTIONS]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$RESULTS_DIR/docker-test-$TIMESTAMP.log"

# Options
BUILD_FRESH=false
RUN_BENCHMARKS=false
SKIP_VALIDATION=false
PLATFORM="linux/amd64"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fresh|--rebuild)
            BUILD_FRESH=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARKS=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fresh, --rebuild    Force rebuild of all images"
            echo "  --benchmark           Run performance benchmarks"
            echo "  --skip-validation     Skip MCP validation checks"
            echo "  --platform PLATFORM   Target platform (default: linux/amd64)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Helper functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}" | tee -a "$LOG_FILE"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

log "Starting Docker test suite"
log "Platform: $PLATFORM"
log "Results will be saved to: $RESULTS_DIR"

# Change to project root
cd "$PROJECT_ROOT"

# Clean up previous runs
cleanup() {
    log "Cleaning up containers and volumes..."
    docker-compose -f docker-compose.yml down -v 2>/dev/null || true
}

trap cleanup EXIT

# Build images
if [ "$BUILD_FRESH" = true ]; then
    log "Building fresh Docker images..."
    docker-compose -f docker-compose.yml build --no-cache --parallel --platform "$PLATFORM"
else
    log "Building Docker images (using cache)..."
    docker-compose -f docker-compose.yml build --parallel --platform "$PLATFORM"
fi

if [ $? -eq 0 ]; then
    success "Docker images built successfully"
else
    error "Failed to build Docker images"
    exit 1
fi

# Start MCP server
log "Starting MCP server..."
docker-compose -f docker-compose.yml up -d mcp-server

# Wait for server to be healthy
log "Waiting for MCP server to be healthy..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if docker-compose -f docker-compose.yml ps mcp-server | grep -q "healthy"; then
        success "MCP server is healthy"
        break
    fi
    attempt=$((attempt + 1))
    echo -n "."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    error "MCP server failed to become healthy"
    docker-compose -f docker-compose.yml logs mcp-server
    exit 1
fi

# Run test suite
log "Running comprehensive test suite..."
docker-compose -f docker-compose.yml run --rm testing

if [ $? -eq 0 ]; then
    success "Test suite passed"
else
    error "Test suite failed"
    docker-compose -f docker-compose.yml logs testing
    exit 1
fi

# Run validation checks
if [ "$SKIP_VALIDATION" = false ]; then
    log "Running MCP 2025-11 validation checks..."
    docker-compose -f docker-compose.yml run --rm validation

    if [ $? -eq 0 ]; then
        success "Validation checks passed"
    else
        error "Validation checks failed"
        docker-compose -f docker-compose.yml logs validation
        exit 1
    fi
else
    warn "Skipping validation checks"
fi

# Run benchmarks
if [ "$RUN_BENCHMARKS" = true ]; then
    log "Running performance benchmarks..."
    docker-compose -f docker-compose.yml run --rm benchmark

    if [ $? -eq 0 ]; then
        success "Benchmarks completed"
    else
        warn "Benchmarks failed or incomplete"
    fi
fi

# Collect results
log "Collecting test results..."
docker cp neural-trader-test:/app/test-results "$RESULTS_DIR/" 2>/dev/null || true
docker cp neural-trader-validation:/app/reports "$RESULTS_DIR/" 2>/dev/null || true

# Generate summary report
SUMMARY_FILE="$RESULTS_DIR/summary-$TIMESTAMP.txt"
{
    echo "========================================"
    echo "Neural Trader Docker Test Summary"
    echo "========================================"
    echo "Timestamp: $(date)"
    echo "Platform: $PLATFORM"
    echo ""
    echo "Build Status: SUCCESS"
    echo "Test Status: PASSED"
    if [ "$SKIP_VALIDATION" = false ]; then
        echo "Validation Status: PASSED"
    fi
    if [ "$RUN_BENCHMARKS" = true ]; then
        echo "Benchmark Status: COMPLETED"
    fi
    echo ""
    echo "Logs: $LOG_FILE"
    echo "Results Directory: $RESULTS_DIR"
    echo "========================================"
} | tee "$SUMMARY_FILE"

success "All tests completed successfully!"
log "Summary saved to: $SUMMARY_FILE"

exit 0
