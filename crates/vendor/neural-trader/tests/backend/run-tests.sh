#!/bin/bash

# Neural Trader Backend Test Runner
# Convenient script for running backend tests

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_message() {
    echo -e "${2}${1}${NC}"
}

# Print header
print_header() {
    echo ""
    echo "======================================================================"
    echo "  $1"
    echo "======================================================================"
    echo ""
}

# Show usage
show_usage() {
    print_header "Neural Trader Backend Test Runner"
    echo "Usage: ./run-tests.sh [option]"
    echo ""
    echo "Options:"
    echo "  all             Run all test suites with coverage"
    echo "  unit            Run unit tests only"
    echo "  class           Run class tests only"
    echo "  integration     Run integration tests only"
    echo "  edge            Run edge case tests only"
    echo "  performance     Run performance tests only"
    echo "  coverage        Run all tests with coverage report"
    echo "  watch           Run tests in watch mode"
    echo "  quick           Run quick validation (unit + class)"
    echo "  ci              Run CI pipeline tests"
    echo "  help            Show this help message"
    echo ""
}

# Run all tests
run_all() {
    print_header "Running All Backend Tests"
    npm test -- tests/backend --coverage
}

# Run unit tests
run_unit() {
    print_header "Running Unit Tests"
    npm test -- tests/backend/unit-tests.test.js
}

# Run class tests
run_class() {
    print_header "Running Class Tests"
    npm test -- tests/backend/class-tests.test.js
}

# Run integration tests
run_integration() {
    print_header "Running Integration Tests"
    npm test -- tests/backend/integration-tests.test.js
}

# Run edge case tests
run_edge() {
    print_header "Running Edge Case Tests"
    npm test -- tests/backend/edge-cases.test.js
}

# Run performance tests
run_performance() {
    print_header "Running Performance Tests"
    npm test -- tests/backend/performance-tests.test.js
}

# Run with coverage
run_coverage() {
    print_header "Running Tests with Coverage Report"
    npm test -- tests/backend --coverage --coverageReporters=text --coverageReporters=html
    print_message "Coverage report available at: coverage/backend/lcov-report/index.html" "$GREEN"
}

# Run in watch mode
run_watch() {
    print_header "Running Tests in Watch Mode"
    npm test -- tests/backend --watch
}

# Run quick validation
run_quick() {
    print_header "Running Quick Validation (Unit + Class Tests)"
    npm test -- tests/backend/unit-tests.test.js tests/backend/class-tests.test.js
}

# Run CI pipeline
run_ci() {
    print_header "Running CI Pipeline Tests"
    print_message "Building backend..." "$BLUE"
    cd neural-trader-rust/packages/neural-trader-backend
    npm run build || true
    cd ../../..

    print_message "Running tests with coverage..." "$BLUE"
    npm test -- tests/backend --coverage --ci --maxWorkers=2

    print_message "Checking coverage thresholds..." "$BLUE"
    npm test -- tests/backend --coverage --coverageThreshold='{"global":{"branches":95,"functions":95,"lines":95,"statements":95}}'
}

# Main script
main() {
    case "${1:-all}" in
        all)
            run_all
            ;;
        unit)
            run_unit
            ;;
        class)
            run_class
            ;;
        integration)
            run_integration
            ;;
        edge)
            run_edge
            ;;
        performance)
            run_performance
            ;;
        coverage)
            run_coverage
            ;;
        watch)
            run_watch
            ;;
        quick)
            run_quick
            ;;
        ci)
            run_ci
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_message "Unknown option: $1" "$RED"
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
