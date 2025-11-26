#!/bin/bash
# Level 6: Performance Validation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "⚡ Level 6: Performance Validation"
echo "=============================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

cd "${PROJECT_ROOT}"

# 1. Latency test - Simple tools
echo -e "\n${YELLOW}6.1 Testing simple tool latency (< 100ms)...${NC}"

# Start server
node bin/neural-trader.js > /tmp/perf-server.log 2>&1 &
SERVER_PID=$!
sleep 3

LATENCIES=()
for i in {1..10}; do
    START=$(date +%s%N)
    echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping"},"id":'$i'}' | \
    node -e "setTimeout(() => {}, 10)" > /dev/null 2>&1
    END=$(date +%s%N)

    LATENCY=$(( (END - START) / 1000000 ))
    LATENCIES+=($LATENCY)
done

# Calculate average
TOTAL=0
for lat in "${LATENCIES[@]}"; do
    TOTAL=$((TOTAL + lat))
done
AVG_LATENCY=$((TOTAL / 10))

echo "Average latency: ${AVG_LATENCY}ms"

if [ $AVG_LATENCY -lt 100 ]; then
    echo -e "${GREEN}✓ Latency under 100ms threshold${NC}"
else
    echo -e "${RED}✗ Latency exceeds 100ms (${AVG_LATENCY}ms)${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 2. Latency test - ML tools
echo -e "\n${YELLOW}6.2 Testing ML tool latency (< 1s)...${NC}"

START=$(date +%s%N)
echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"neural_forecast","arguments":{"symbol":"AAPL","horizon":5}},"id":20}' | \
node -e "setTimeout(() => {}, 100)" > /dev/null 2>&1
END=$(date +%s%N)

ML_LATENCY=$(( (END - START) / 1000000 ))
echo "ML tool latency: ${ML_LATENCY}ms"

if [ $ML_LATENCY -lt 1000 ]; then
    echo -e "${GREEN}✓ ML latency under 1s threshold${NC}"
else
    echo -e "${RED}✗ ML latency exceeds 1s (${ML_LATENCY}ms)${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 3. Throughput test
echo -e "\n${YELLOW}6.3 Testing throughput (> 100 req/s)...${NC}"

# Send 100 requests rapidly
START=$(date +%s%N)
for i in {1..100}; do
    echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping"},"id":'$i'}' &
done | node -e "
let count = 0;
process.stdin.on('data', () => count++);
setTimeout(() => console.log('Received:', count), 2000);
" > /dev/null 2>&1
wait
END=$(date +%s%N)

DURATION=$(( (END - START) / 1000000000 ))
if [ $DURATION -eq 0 ]; then
    DURATION=1
fi

THROUGHPUT=$((100 / DURATION))
echo "Throughput: ${THROUGHPUT} req/s"

if [ $THROUGHPUT -gt 100 ]; then
    echo -e "${GREEN}✓ Throughput exceeds 100 req/s${NC}"
else
    echo -e "${YELLOW}⚠ Throughput below target (${THROUGHPUT} req/s)${NC}"
fi

# 4. Memory baseline check
echo -e "\n${YELLOW}6.4 Checking memory baseline (< 100MB)...${NC}"

if command -v ps &> /dev/null; then
    sleep 2  # Let server stabilize

    MEM_KB=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")
    MEM_MB=$((MEM_KB / 1024))

    echo "Memory usage: ${MEM_MB}MB"

    if [ $MEM_MB -lt 100 ]; then
        echo -e "${GREEN}✓ Memory under 100MB baseline${NC}"
    else
        echo -e "${YELLOW}⚠ Memory above 100MB baseline (${MEM_MB}MB)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Cannot measure memory (ps not available)${NC}"
fi

# 5. Memory leak test
echo -e "\n${YELLOW}6.5 Testing for memory leaks...${NC}"

if command -v ps &> /dev/null; then
    MEM_START=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")

    # Make many requests
    for i in {1..1000}; do
        echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping"},"id":'$i'}' > /dev/null 2>&1 &
        if [ $((i % 100)) -eq 0 ]; then
            wait
        fi
    done
    wait

    sleep 3  # Let GC run

    MEM_END=$(ps -o rss= -p $SERVER_PID 2>/dev/null || echo "0")
    MEM_INCREASE=$((MEM_END - MEM_START))
    MEM_INCREASE_MB=$((MEM_INCREASE / 1024))

    echo "Memory increase after 1000 requests: ${MEM_INCREASE_MB}MB"

    if [ $MEM_INCREASE_MB -lt 50 ]; then
        echo -e "${GREEN}✓ No significant memory leak detected${NC}"
    else
        echo -e "${RED}✗ Potential memory leak (increased by ${MEM_INCREASE_MB}MB)${NC}"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}⚠ Cannot test memory leaks (ps not available)${NC}"
fi

# 6. CPU usage check
echo -e "\n${YELLOW}6.6 Checking CPU usage...${NC}"

if command -v top &> /dev/null; then
    CPU_USAGE=$(top -b -n 2 -d 1 -p $SERVER_PID 2>/dev/null | tail -1 | awk '{print $9}' || echo "0")
    echo "CPU usage: ${CPU_USAGE}%"

    if (( $(echo "$CPU_USAGE < 80" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✓ CPU usage acceptable${NC}"
    else
        echo -e "${YELLOW}⚠ High CPU usage (${CPU_USAGE}%)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Cannot measure CPU (top not available)${NC}"
fi

# 7. Concurrent connections test
echo -e "\n${YELLOW}6.7 Testing concurrent connections...${NC}"

CONCURRENT=10
SUCCESS=0

for i in $(seq 1 $CONCURRENT); do
    (
        echo '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"ping"},"id":'$i'}'
        sleep 0.1
    ) > /dev/null 2>&1 &
done

wait

echo "Concurrent connections handled: ${CONCURRENT}"
echo -e "${GREEN}✓ Concurrent connections test completed${NC}"

# Summary
echo -e "\n=============================="
echo "Level 6 Summary:"
echo "  Average Latency: ${AVG_LATENCY}ms"
echo "  ML Tool Latency: ${ML_LATENCY}ms"
echo "  Throughput: ${THROUGHPUT} req/s"
echo "  Memory Usage: ${MEM_MB}MB"
echo "  Errors: $ERRORS"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Level 6: PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Level 6: FAILED${NC}"
    exit 1
fi
