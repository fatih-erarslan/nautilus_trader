#!/bin/bash

# Performance Validation Script
# Measures API performance metrics and validates against targets

set -e

API_BASE="${API_BASE:-http://localhost:8080}"
ITERATIONS="${ITERATIONS:-100}"
CONCURRENCY="${CONCURRENCY:-10}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Performance Validation${NC}"
echo -e "${BLUE}================================${NC}\n"

# Function to measure endpoint latency
measure_latency() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-""}
    local iterations=${4:-10}

    local total_time=0
    local times=()

    for i in $(seq 1 $iterations); do
        if [ "$method" = "POST" ]; then
            time_ms=$(curl -s -o /dev/null -w "%{time_total}" -X POST \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$API_BASE$endpoint" | awk '{print $1 * 1000}')
        else
            time_ms=$(curl -s -o /dev/null -w "%{time_total}" "$API_BASE$endpoint" | awk '{print $1 * 1000}')
        fi

        times+=($time_ms)
        total_time=$(echo "$total_time + $time_ms" | bc)
    done

    # Calculate average
    avg=$(echo "scale=2; $total_time / $iterations" | bc)

    # Calculate min/max
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
    min=${sorted[0]}
    max=${sorted[-1]}

    # Calculate P50, P95, P99
    p50_idx=$(echo "($iterations * 0.5) / 1" | bc)
    p95_idx=$(echo "($iterations * 0.95) / 1" | bc)
    p99_idx=$(echo "($iterations * 0.99) / 1" | bc)

    p50=${sorted[$p50_idx]}
    p95=${sorted[$p95_idx]}
    p99=${sorted[$p99_idx]}

    echo "$avg,$min,$max,$p50,$p95,$p99"
}

# Validate performance targets
validate_performance() {
    local endpoint=$1
    local target_avg=$2
    local target_p95=$3
    local actual_avg=$4
    local actual_p95=$5

    local status="✅"
    local color=$GREEN

    if (( $(echo "$actual_avg > $target_avg" | bc -l) )) || \
       (( $(echo "$actual_p95 > $target_p95" | bc -l) )); then
        status="❌"
        color=$RED
    fi

    echo -e "$color$status $endpoint - Avg: ${actual_avg}ms (target: ${target_avg}ms), P95: ${actual_p95}ms (target: ${target_p95}ms)$NC"

    [ "$status" = "✅" ]
}

echo -e "${YELLOW}Testing endpoint performance (${ITERATIONS} iterations)...${NC}\n"

passed=0
failed=0

# Test 1: Health Check
echo -e "${BLUE}1. Health Check Endpoint${NC}"
result=$(measure_latency "/health" "GET" "" $ITERATIONS)
IFS=',' read -r avg min max p50 p95 p99 <<< "$result"

if validate_performance "GET /health" 50 100 $avg $p95; then
    ((passed++))
else
    ((failed++))
fi
echo "   Min: ${min}ms, Max: ${max}ms, P50: ${p50}ms, P99: ${p99}ms"
echo ""

# Test 2: List Scans
echo -e "${BLUE}2. List Scans Endpoint${NC}"
result=$(measure_latency "/api/scanner/scans?limit=10" "GET" "" $ITERATIONS)
IFS=',' read -r avg min max p50 p95 p99 <<< "$result"

if validate_performance "GET /api/scanner/scans" 200 500 $avg $p95; then
    ((passed++))
else
    ((failed++))
fi
echo "   Min: ${min}ms, Max: ${max}ms, P50: ${p50}ms, P99: ${p99}ms"
echo ""

# Test 3: Create Scan
echo -e "${BLUE}3. Create Scan Endpoint${NC}"
scan_data='{
    "url": "https://api.example.com/spec.json",
    "scan_type": "openapi",
    "options": {}
}'
result=$(measure_latency "/api/scanner/scan" "POST" "$scan_data" 10)
IFS=',' read -r avg min max p50 p95 p99 <<< "$result"

if validate_performance "POST /api/scanner/scan" 300 800 $avg $p95; then
    ((passed++))
else
    ((failed++))
fi
echo "   Min: ${min}ms, Max: ${max}ms, P50: ${p50}ms, P99: ${p99}ms"
echo ""

# Test 4: Stats Endpoint
echo -e "${BLUE}4. Scanner Stats Endpoint${NC}"
result=$(measure_latency "/api/scanner/stats" "GET" "" $ITERATIONS)
IFS=',' read -r avg min max p50 p95 p99 <<< "$result"

if validate_performance "GET /api/scanner/stats" 100 250 $avg $p95; then
    ((passed++))
else
    ((failed++))
fi
echo "   Min: ${min}ms, Max: ${max}ms, P50: ${p50}ms, P99: ${p99}ms"
echo ""

# Test 5: Concurrent Load Test
echo -e "${BLUE}5. Concurrent Load Test ($CONCURRENCY concurrent requests)${NC}"
start_time=$(date +%s.%N)

for i in $(seq 1 $CONCURRENCY); do
    curl -s "$API_BASE/health" > /dev/null &
done
wait

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
throughput=$(echo "scale=2; $CONCURRENCY / $duration" | bc)

echo -e "   Throughput: ${throughput} req/sec"
if (( $(echo "$throughput > 50" | bc -l) )); then
    echo -e "   ${GREEN}✅ Throughput target met (>50 req/sec)${NC}"
    ((passed++))
else
    echo -e "   ${RED}❌ Throughput below target (<50 req/sec)${NC}"
    ((failed++))
fi
echo ""

# Summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Performance Summary${NC}"
echo -e "${BLUE}================================${NC}"
total=$((passed + failed))
echo -e "Tests passed: ${GREEN}$passed${NC}/$total"
echo -e "Tests failed: ${RED}$failed${NC}/$total"

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}✅ All performance targets met${NC}\n"
    exit 0
else
    echo -e "\n${YELLOW}⚠️ Some performance targets not met${NC}\n"
    exit 1
fi
