#!/bin/bash
# Load Testing Script for Hive Mind Rust Financial Trading System
# Validates system performance under high-frequency trading loads

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-hive-mind-production}"
DURATION="${DURATION:-300}"
CONCURRENT_USERS="${CONCURRENT_USERS:-1000}"
TARGET_LATENCY="${TARGET_LATENCY:-1000}"  # microseconds
SERVICE_URL="${SERVICE_URL:-http://hive-mind-service:8091}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --concurrent-users)
            CONCURRENT_USERS="$2"
            shift 2
            ;;
        --target-latency)
            TARGET_LATENCY="$2"
            shift 2
            ;;
        --service-url)
            SERVICE_URL="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating load test prerequisites..."
    
    # Check required tools
    for tool in kubectl curl jq bc; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_error "$tool not found"
            exit 1
        fi
    done
    
    # Verify service is available
    if ! kubectl get service hive-mind-service -n "$NAMESPACE" >/dev/null 2>&1; then
        log_error "Hive Mind service not found in namespace $NAMESPACE"
        exit 1
    fi
    
    # Check if load testing tools are available
    if command -v hey >/dev/null 2>&1; then
        LOAD_TOOL="hey"
    elif command -v wrk >/dev/null 2>&1; then
        LOAD_TOOL="wrk"
    elif command -v ab >/dev/null 2>&1; then
        LOAD_TOOL="ab"
    else
        log_warning "No dedicated load testing tool found, using curl-based testing"
        LOAD_TOOL="curl"
    fi
    
    log_info "Using load testing tool: $LOAD_TOOL"
    log_success "Prerequisites validation passed"
}

# System baseline check
get_baseline_metrics() {
    log_info "Collecting baseline system metrics..."
    
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Get current resource usage
    kubectl top pods -n "$NAMESPACE" -l app=hive-mind-rust > baseline-resources.txt || log_warning "Could not collect resource metrics"
    
    # Get current performance metrics
    curl -s "http://$service_ip:9090/metrics" | grep -E "(trading_operation_duration|cpu_usage|memory_usage)" > baseline-metrics.txt || log_warning "Could not collect performance metrics"
    
    # Quick latency check
    local latency_sum=0
    local latency_count=0
    
    for i in {1..10}; do
        local start_time=$(date +%s%N)
        curl -s "http://$service_ip:8091/health" >/dev/null
        local end_time=$(date +%s%N)
        local latency_us=$(( (end_time - start_time) / 1000 ))
        latency_sum=$((latency_sum + latency_us))
        latency_count=$((latency_count + 1))
    done
    
    local baseline_latency=$((latency_sum / latency_count))
    echo "$baseline_latency" > baseline-latency.txt
    
    log_info "Baseline latency: ${baseline_latency}μs"
}

# Trading simulation load test
run_trading_load_test() {
    log_info "Running trading simulation load test..."
    
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    local results_file="trading-load-results.json"
    
    case "$LOAD_TOOL" in
        "hey")
            hey -n $((CONCURRENT_USERS * DURATION / 10)) \
                -c "$CONCURRENT_USERS" \
                -t 10 \
                -o csv \
                -H "Content-Type: application/json" \
                -m POST \
                -d '{"operation":"trade","symbol":"EURUSD","amount":1000,"type":"market"}' \
                "http://$service_ip:8090/api/v1/trade" > trading-load-hey.csv
            ;;
        "wrk")
            wrk -t"$CONCURRENT_USERS" \
                -c"$CONCURRENT_USERS" \
                -d"${DURATION}s" \
                --timeout 10s \
                -s trading-test.lua \
                "http://$service_ip:8090/api/v1/trade" > trading-load-wrk.txt
            ;;
        "ab")
            ab -n $((CONCURRENT_USERS * DURATION / 10)) \
                -c "$CONCURRENT_USERS" \
                -g trading-load-ab.tsv \
                -T "application/json" \
                -p trading-request.json \
                "http://$service_ip:8090/api/v1/trade" > trading-load-ab.txt
            ;;
        *)
            run_curl_load_test "$service_ip"
            ;;
    esac
}

# Consensus system load test
run_consensus_load_test() {
    log_info "Running consensus system load test..."
    
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Simulate high-frequency consensus operations
    case "$LOAD_TOOL" in
        "hey")
            hey -n $((CONCURRENT_USERS * DURATION / 20)) \
                -c "$((CONCURRENT_USERS / 2))" \
                -t 5 \
                -H "Content-Type: application/json" \
                -m POST \
                -d '{"type":"proposal","data":{"operation":"consensus_test","value":123}}' \
                "http://$service_ip:8090/api/v1/consensus/propose" > consensus-load-hey.csv
            ;;
        *)
            log_info "Running consensus load test with curl..."
            run_consensus_curl_test "$service_ip"
            ;;
    esac
}

# Neural system load test
run_neural_load_test() {
    log_info "Running neural system load test..."
    
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test neural inference performance
    case "$LOAD_TOOL" in
        "hey")
            hey -n $((CONCURRENT_USERS * DURATION / 50)) \
                -c "$((CONCURRENT_USERS / 4))" \
                -t 15 \
                -H "Content-Type: application/json" \
                -m POST \
                -d '{"data":[1,2,3,4,5,6,7,8,9,10],"model":"pattern_recognition"}' \
                "http://$service_ip:8090/api/v1/neural/inference" > neural-load-hey.csv
            ;;
        *)
            log_info "Running neural load test with curl..."
            run_neural_curl_test "$service_ip"
            ;;
    esac
}

# Memory system load test  
run_memory_load_test() {
    log_info "Running distributed memory load test..."
    
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test memory operations under load
    case "$LOAD_TOOL" in
        "hey")
            hey -n $((CONCURRENT_USERS * DURATION / 30)) \
                -c "$((CONCURRENT_USERS / 3))" \
                -t 8 \
                -H "Content-Type: application/json" \
                -m POST \
                -d '{"key":"test_key_RANDOM","value":{"data":"performance_test_data"}}' \
                "http://$service_ip:8090/api/v1/memory/store" > memory-load-hey.csv
            ;;
        *)
            log_info "Running memory load test with curl..."
            run_memory_curl_test "$service_ip"
            ;;
    esac
}

# Curl-based load testing (fallback)
run_curl_load_test() {
    local service_ip="$1"
    local results_file="curl-load-results.txt"
    
    log_info "Running curl-based load test for ${DURATION}s with ${CONCURRENT_USERS} concurrent users"
    
    # Create test data
    cat <<EOF > trading-request.json
{
    "operation": "trade",
    "symbol": "EURUSD",
    "amount": 1000,
    "type": "market",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    # Run parallel curl requests
    echo "timestamp,latency_us,status_code,response_size" > "$results_file"
    
    local end_time=$(($(date +%s) + DURATION))
    local pids=()
    
    for ((i=1; i<=CONCURRENT_USERS; i++)); do
        {
            while [[ $(date +%s) -lt $end_time ]]; do
                local start_time=$(date +%s%N)
                local response=$(curl -s -w "%{http_code},%{size_download}" \
                    -H "Content-Type: application/json" \
                    -d @trading-request.json \
                    "http://$service_ip:8090/api/v1/trade")
                local end_time_req=$(date +%s%N)
                local latency_us=$(( (end_time_req - start_time) / 1000 ))
                
                echo "$(date +%s%N),$latency_us,$response" >> "$results_file"
                
                # Brief pause to prevent overwhelming
                sleep 0.01
            done
        } &
        pids+=($!)
    done
    
    # Wait for all background processes
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    
    log_info "Curl-based load test completed"
}

# Analyze results
analyze_results() {
    log_info "Analyzing load test results..."
    
    local total_requests=0
    local total_errors=0
    local latency_sum=0
    local max_latency=0
    local min_latency=999999999
    
    # Analyze based on available result files
    if [[ -f "trading-load-hey.csv" ]]; then
        analyze_hey_results "trading-load-hey.csv"
    elif [[ -f "trading-load-wrk.txt" ]]; then
        analyze_wrk_results "trading-load-wrk.txt"
    elif [[ -f "trading-load-ab.txt" ]]; then
        analyze_ab_results "trading-load-ab.txt"
    elif [[ -f "curl-load-results.txt" ]]; then
        analyze_curl_results "curl-load-results.txt"
    else
        log_warning "No result files found to analyze"
        return 1
    fi
    
    # Generate summary report
    generate_summary_report
}

analyze_hey_results() {
    local file="$1"
    log_info "Analyzing Hey results from $file"
    
    # Hey CSV format: response-time,DNS+dialup,DNS,Request-write,Response-delay,Response-read,status-code,offset
    if [[ -f "$file" ]]; then
        local total_requests=$(tail -n +2 "$file" | wc -l)
        local successful_requests=$(tail -n +2 "$file" | awk -F',' '$7 < 400' | wc -l)
        local error_rate=$(echo "scale=2; (($total_requests - $successful_requests) / $total_requests) * 100" | bc)
        
        # Calculate latency statistics (response-time is in seconds, convert to microseconds)
        local latencies=$(tail -n +2 "$file" | awk -F',' '{print $1 * 1000000}' | sort -n)
        local p50=$(echo "$latencies" | awk 'NR==int(NR*0.5)+1')
        local p95=$(echo "$latencies" | awk 'NR==int(NR*0.95)+1')
        local p99=$(echo "$latencies" | awk 'NR==int(NR*0.99)+1')
        
        cat <<EOF > load-test-summary.txt
Load Test Results Summary
=========================
Tool: Hey
Total Requests: $total_requests
Successful Requests: $successful_requests
Error Rate: ${error_rate}%
P50 Latency: ${p50}μs
P95 Latency: ${p95}μs
P99 Latency: ${p99}μs
EOF
    fi
}

analyze_curl_results() {
    local file="$1"
    log_info "Analyzing curl results from $file"
    
    if [[ -f "$file" ]]; then
        local total_requests=$(tail -n +2 "$file" | wc -l)
        local successful_requests=$(tail -n +2 "$file" | awk -F',' '$3 < 400' | wc -l)
        local error_rate=$(echo "scale=2; (($total_requests - $successful_requests) / $total_requests) * 100" | bc)
        
        # Calculate latency statistics
        local latencies=$(tail -n +2 "$file" | awk -F',' '{print $2}' | sort -n)
        local p50=$(echo "$latencies" | awk 'NR==int(NR*0.5)+1')
        local p95=$(echo "$latencies" | awk 'NR==int(NR*0.95)+1')
        local p99=$(echo "$latencies" | awk 'NR==int(NR*0.99)+1')
        local avg=$(echo "$latencies" | awk '{sum+=$1} END {print sum/NR}')
        
        cat <<EOF > load-test-summary.txt
Load Test Results Summary
=========================
Tool: curl
Duration: ${DURATION}s
Concurrent Users: $CONCURRENT_USERS
Total Requests: $total_requests
Successful Requests: $successful_requests
Error Rate: ${error_rate}%
Average Latency: ${avg}μs
P50 Latency: ${p50}μs
P95 Latency: ${p95}μs
P99 Latency: ${p99}μs
Requests per Second: $(echo "scale=2; $total_requests / $DURATION" | bc)
EOF
    fi
}

generate_summary_report() {
    log_info "Generating comprehensive summary report..."
    
    # Get post-test metrics
    kubectl top pods -n "$NAMESPACE" -l app=hive-mind-rust > post-test-resources.txt || log_warning "Could not collect post-test resource metrics"
    
    # System health check
    local service_ip=$(kubectl get service hive-mind-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    local health_status="UNKNOWN"
    if curl -f "http://$service_ip:8091/health" >/dev/null 2>&1; then
        health_status="HEALTHY"
    else
        health_status="UNHEALTHY"
    fi
    
    # Create comprehensive report
    cat <<EOF > comprehensive-load-test-report.md
# Hive Mind Rust - Load Test Report

## Test Configuration
- **Date**: $(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)
- **Duration**: ${DURATION} seconds
- **Concurrent Users**: $CONCURRENT_USERS
- **Target Latency**: ${TARGET_LATENCY}μs
- **Namespace**: $NAMESPACE

## System Health
- **Post-Test Status**: $health_status

## Performance Results
$(cat load-test-summary.txt 2>/dev/null || echo "No detailed results available")

## Resource Usage
### Baseline Resources
$(cat baseline-resources.txt 2>/dev/null || echo "No baseline data available")

### Post-Test Resources  
$(cat post-test-resources.txt 2>/dev/null || echo "No post-test data available")

## SLA Compliance
EOF

    # Check SLA compliance
    if [[ -f "load-test-summary.txt" ]]; then
        local p99_latency=$(grep "P99 Latency" load-test-summary.txt | awk '{print $3}' | sed 's/μs//')
        local error_rate=$(grep "Error Rate" load-test-summary.txt | awk '{print $3}' | sed 's/%//')
        
        if [[ -n "$p99_latency" ]] && [[ $(echo "$p99_latency <= $TARGET_LATENCY" | bc) -eq 1 ]]; then
            echo "- ✅ **Latency SLA**: PASSED (P99: ${p99_latency}μs <= ${TARGET_LATENCY}μs)" >> comprehensive-load-test-report.md
        else
            echo "- ❌ **Latency SLA**: FAILED (P99: ${p99_latency}μs > ${TARGET_LATENCY}μs)" >> comprehensive-load-test-report.md
        fi
        
        if [[ -n "$error_rate" ]] && [[ $(echo "$error_rate <= 1.0" | bc) -eq 1 ]]; then
            echo "- ✅ **Error Rate SLA**: PASSED (${error_rate}% <= 1.0%)" >> comprehensive-load-test-report.md
        else
            echo "- ❌ **Error Rate SLA**: FAILED (${error_rate}% > 1.0%)" >> comprehensive-load-test-report.md
        fi
    fi
    
    echo "" >> comprehensive-load-test-report.md
    echo "## Recommendations" >> comprehensive-load-test-report.md
    
    # Add recommendations based on results
    if [[ "$health_status" != "HEALTHY" ]]; then
        echo "- ⚠️  System health check failed - investigate immediately" >> comprehensive-load-test-report.md
    fi
    
    echo "- Monitor system for next 24 hours for any degradation" >> comprehensive-load-test-report.md
    echo "- Review resource usage patterns for optimization opportunities" >> comprehensive-load-test-report.md
    
    log_success "Comprehensive report generated: comprehensive-load-test-report.md"
}

# Cleanup test artifacts
cleanup() {
    log_info "Cleaning up test artifacts..."
    
    rm -f trading-request.json
    rm -f baseline-*.txt
    rm -f post-test-*.txt
    rm -f *-load-*.csv
    rm -f *-load-*.txt
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting comprehensive load testing for Hive Mind Rust system"
    log_info "Configuration: Duration=${DURATION}s, Users=${CONCURRENT_USERS}, Target=${TARGET_LATENCY}μs"
    
    validate_prerequisites
    get_baseline_metrics
    
    # Run different types of load tests
    run_trading_load_test
    run_consensus_load_test  
    run_neural_load_test
    run_memory_load_test
    
    # Wait a moment for system to stabilize
    log_info "Waiting for system to stabilize..."
    sleep 30
    
    analyze_results
    generate_summary_report
    
    # Display summary
    if [[ -f "comprehensive-load-test-report.md" ]]; then
        echo ""
        echo "========================================="
        echo "         LOAD TEST SUMMARY"
        echo "========================================="
        grep -E "(PASSED|FAILED|Post-Test Status)" comprehensive-load-test-report.md || true
        echo "========================================="
        echo ""
        echo "Full report available: comprehensive-load-test-report.md"
    fi
    
    # Cleanup
    trap cleanup EXIT
    
    log_success "Load testing completed"
}

# Execute main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi