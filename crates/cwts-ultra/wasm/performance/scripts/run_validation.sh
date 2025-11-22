#!/bin/bash

# CWTS Ultra Scientific Performance Validation Script
# Executes comprehensive benchmarking with scientific rigor

set -e

echo "🏆 CWTS Ultra Scientific Performance Validation"
echo "=============================================="
echo "📊 Validating performance claims with statistical significance"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Performance claims to validate
echo "🎯 Performance Claims to Validate:"
echo "   🔥 GPU Acceleration: 4,000,000x speedup"
echo "   ⏱️  P99 Latency: <740ns"
echo "   🚀 Throughput: 1,000,000+ ops/second"
echo "   💾 Memory Efficiency: >90%"
echo ""

# Check system requirements
echo "🔍 Checking System Requirements..."
echo "================================="

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo -e "${RED}❌ Rust compiler not found. Please install Rust.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Rust compiler found: $(rustc --version)${NC}"

# Check if Cargo is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Cargo not found. Please install Cargo.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Cargo found: $(cargo --version)${NC}"

# Check available memory
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
if [ "$TOTAL_MEM" -lt 4 ]; then
    echo -e "${YELLOW}⚠️ Warning: Less than 4GB RAM available. Performance tests may be limited.${NC}"
else
    echo -e "${GREEN}✅ Memory: ${TOTAL_MEM}GB available${NC}"
fi

# Check CPU cores
CPU_CORES=$(nproc)
if [ "$CPU_CORES" -lt 4 ]; then
    echo -e "${YELLOW}⚠️ Warning: Less than 4 CPU cores. Parallel benchmarks may be limited.${NC}"
else
    echo -e "${GREEN}✅ CPU Cores: ${CPU_CORES} available${NC}"
fi

echo ""

# Create necessary directories
echo "📁 Setting up benchmark environment..."
mkdir -p /home/kutlu/CWTS/cwts-ultra/wasm/performance/{reports,data,logs}

# Set environment variables for optimal performance
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"
export RUST_BACKTRACE=1

# Build the performance validation suite
echo "🔧 Building Performance Validation Suite..."
echo "=========================================="
cd /home/kutlu/CWTS/cwts-ultra/wasm/performance/benchmarks

if cargo build --release --bin performance_validator; then
    echo -e "${GREEN}✅ Build successful${NC}"
else
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo ""

# Run comprehensive performance validation
echo "🚀 Executing Comprehensive Performance Validation..."
echo "=================================================="
echo "⏱️ This may take several minutes for statistical accuracy"
echo ""

# Set CPU governor to performance mode (if available)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "⚡ Setting CPU governor to performance mode..."
    sudo bash -c 'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' 2>/dev/null || true
fi

# Disable CPU frequency scaling (if available)
if command -v cpupower &> /dev/null; then
    echo "🔒 Disabling CPU frequency scaling for consistent results..."
    sudo cpupower frequency-set --governor performance 2>/dev/null || true
fi

# Clear system caches for clean measurements
echo "🧹 Clearing system caches..."
sync
sudo bash -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# Set high priority for benchmark process
echo "⚡ Running performance validation with high priority..."

# Execute the validation suite
START_TIME=$(date +%s)

if nice -n -20 taskset -c 0-$((CPU_CORES-1)) ./target/release/performance_validator; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${GREEN}✅ Performance validation completed successfully${NC}"
    echo -e "${BLUE}⏱️ Total execution time: ${DURATION} seconds${NC}"
    
    # Check if report was generated
    REPORT_PATH="/home/kutlu/CWTS/cwts-ultra/wasm/performance/reports/scientific_performance_validation_report.md"
    if [ -f "$REPORT_PATH" ]; then
        echo -e "${GREEN}📄 Report generated: ${REPORT_PATH}${NC}"
        
        # Show quick summary from report
        echo ""
        echo "📊 QUICK SUMMARY:"
        echo "================"
        grep -A 10 "OVERALL RESULT" "$REPORT_PATH" 2>/dev/null || echo "Summary not found in report"
    else
        echo -e "${YELLOW}⚠️ Report file not found${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}❌ Performance validation failed${NC}"
    exit 1
fi

echo ""

# Reset CPU governor to default (if we changed it)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    echo "🔄 Resetting CPU governor to default..."
    sudo bash -c 'echo ondemand > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' 2>/dev/null || true
fi

# Generate benchmark artifacts
echo "📦 Generating Benchmark Artifacts..."
echo "=================================="

# Create performance metrics JSON
METRICS_FILE="/home/kutlu/CWTS/cwts-ultra/wasm/performance/data/validation_metrics.json"
echo "{
  \"validation_timestamp\": \"$(date -Iseconds)\",
  \"system_info\": {
    \"cpu_cores\": $CPU_CORES,
    \"total_memory_gb\": $TOTAL_MEM,
    \"rust_version\": \"$(rustc --version)\",
    \"hostname\": \"$(hostname)\"
  },
  \"validation_duration_seconds\": $DURATION,
  \"claims_tested\": [
    \"gpu_acceleration_4million_x\",
    \"p99_latency_740ns\",
    \"throughput_1million_ops\",
    \"memory_efficiency_90_percent\"
  ]
}" > "$METRICS_FILE"

echo -e "${GREEN}✅ Metrics saved to: ${METRICS_FILE}${NC}"

# Create validation summary
SUMMARY_FILE="/home/kutlu/CWTS/cwts-ultra/wasm/performance/reports/validation_summary.txt"
echo "CWTS Ultra Performance Validation Summary" > "$SUMMARY_FILE"
echo "=======================================" >> "$SUMMARY_FILE"
echo "Execution Date: $(date)" >> "$SUMMARY_FILE"
echo "Duration: ${DURATION} seconds" >> "$SUMMARY_FILE"
echo "System: $CPU_CORES cores, ${TOTAL_MEM}GB RAM" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

if [ -f "$REPORT_PATH" ]; then
    grep -A 20 "VALIDATION SUMMARY" "$REPORT_PATH" >> "$SUMMARY_FILE" 2>/dev/null || true
fi

echo -e "${GREEN}✅ Summary saved to: ${SUMMARY_FILE}${NC}"

# Final status
echo ""
echo "🏆 CWTS Ultra Scientific Performance Validation Complete"
echo "======================================================"
echo -e "${BLUE}📄 Detailed Report: ${REPORT_PATH}${NC}"
echo -e "${BLUE}📊 Metrics Data: ${METRICS_FILE}${NC}"
echo -e "${BLUE}📋 Summary: ${SUMMARY_FILE}${NC}"
echo ""
echo -e "${GREEN}✅ Validation suite execution completed successfully${NC}"
echo ""

# Optional: Open the report if GUI is available
if command -v xdg-open &> /dev/null && [ -n "$DISPLAY" ]; then
    echo "🌐 Opening performance report..."
    xdg-open "$REPORT_PATH" 2>/dev/null || true
fi

exit 0