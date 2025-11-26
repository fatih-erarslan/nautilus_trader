#!/bin/bash

# AI News Trading Benchmark - Quick Benchmark Execution Script
# This script provides easy access to common benchmark operations

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

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_TYPE="quick"
OUTPUT_FORMAT="json"
OUTPUT_FILE=""
DURATION="5m"
STRATEGIES="momentum,swing"
ASSETS="stocks"
PARALLEL_JOBS=1
VERBOSE=false

# Function to show usage
show_usage() {
    cat << EOF
AI News Trading Benchmark - Quick Execution Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  quick           Run quick benchmark suite (default)
  full            Run comprehensive benchmark suite
  latency         Run latency benchmarks only
  throughput      Run throughput benchmarks only
  strategy        Run strategy performance benchmarks
  integration     Run integration tests
  simulation      Run simulation tests
  optimization    Run optimization tests
  stress          Run stress tests
  monitoring      Start monitoring dashboard
  status          Show system status

Options:
  --duration DURATION     Benchmark duration (default: 5m)
  --strategies LIST       Comma-separated strategy list (default: momentum,swing)
  --assets ASSETS         Asset types to test (stocks,crypto,bonds)
  --output FILE           Output file path
  --format FORMAT         Output format (json,csv,html) (default: json)
  --parallel JOBS         Number of parallel jobs (default: 1)
  --verbose, -v           Verbose output
  --help, -h              Show this help

Examples:
  $0 quick --duration 1m --verbose
  $0 full --output results.json --parallel 4
  $0 latency --strategies momentum --assets stocks
  $0 integration --verbose
  $0 monitoring

EOF
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/integration_tests.py" ]]; then
        log_error "Not in project root directory or missing files"
        exit 1
    fi
    
    # Check Python environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warn "Virtual environment not activated"
        if [[ -f "$PROJECT_ROOT/activate_benchmark.sh" ]]; then
            log_info "Run: source activate_benchmark.sh"
        fi
    fi
    
    # Check if required modules can be imported
    if ! python -c "from src.integration.system_orchestrator import SystemOrchestrator" 2>/dev/null; then
        log_error "Cannot import required modules. Run setup_environment.sh first."
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Function to generate output filename
generate_output_filename() {
    local prefix="$1"
    local timestamp=$(date "+%Y%m%d_%H%M%S")
    
    if [[ -z "$OUTPUT_FILE" ]]; then
        OUTPUT_FILE="$PROJECT_ROOT/results/${prefix}_${timestamp}.${OUTPUT_FORMAT}"
        mkdir -p "$(dirname "$OUTPUT_FILE")"
    fi
    
    log_info "Output will be saved to: $OUTPUT_FILE"
}

# Function to run quick benchmark
run_quick_benchmark() {
    log_info "Running quick benchmark suite..."
    generate_output_filename "quick_benchmark"
    
    local cmd_args=(
        "python" "-m" "benchmark.cli" "benchmark"
        "--strategy" "$STRATEGIES"
        "--duration" "$DURATION"
        "--assets" "$ASSETS"
        "--output" "$OUTPUT_FILE"
        "--format" "$OUTPUT_FORMAT"
    )
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd_args+=("--verbose")
    fi
    
    log_info "Executing: ${cmd_args[*]}"
    "${cmd_args[@]}"
    
    log_info "Quick benchmark completed. Results: $OUTPUT_FILE"
}

# Function to run full benchmark suite
run_full_benchmark() {
    log_info "Running full benchmark suite..."
    generate_output_filename "full_benchmark"
    
    local strategies_array=(momentum swing mirror)
    local assets_array=(stocks crypto bonds)
    
    for strategy in "${strategies_array[@]}"; do
        for asset in "${assets_array[@]}"; do
            log_info "Running benchmark: $strategy on $asset"
            
            local strategy_output="${OUTPUT_FILE%.${OUTPUT_FORMAT}}_${strategy}_${asset}.${OUTPUT_FORMAT}"
            
            python -m benchmark.cli benchmark \
                --strategy "$strategy" \
                --duration "$DURATION" \
                --assets "$asset" \
                --output "$strategy_output" \
                --format "$OUTPUT_FORMAT" \
                ${VERBOSE:+--verbose}
        done
    done
    
    log_info "Full benchmark suite completed"
}

# Function to run latency benchmarks
run_latency_benchmark() {
    log_info "Running latency benchmarks..."
    generate_output_filename "latency_benchmark"
    
    python -c "
import sys
sys.path.insert(0, '.')
from src.benchmarks.latency_benchmark import LatencyBenchmark
from integration_tests import IntegrationTestConfig

config = IntegrationTestConfig()
benchmark = LatencyBenchmark(config)

# Run latency tests
results = {}
results['signal_generation'] = benchmark.run_sync('signal_generation')
results['data_processing'] = benchmark.run_sync('data_processing')
results['portfolio_update'] = benchmark.run_sync('portfolio_update')

# Save results
import json
with open('$OUTPUT_FILE', 'w') as f:
    json.dump({
        'benchmark_type': 'latency',
        'results': results,
        'timestamp': $(date +%s)
    }, f, indent=2, default=str)

print('Latency benchmark completed')
"
    
    log_info "Latency benchmark completed. Results: $OUTPUT_FILE"
}

# Function to run throughput benchmarks
run_throughput_benchmark() {
    log_info "Running throughput benchmarks..."
    generate_output_filename "throughput_benchmark"
    
    python -c "
import sys
import asyncio
sys.path.insert(0, '.')
from src.benchmarks.throughput_benchmark import ThroughputBenchmark
from integration_tests import IntegrationTestConfig

async def run_throughput():
    config = IntegrationTestConfig()
    benchmark = ThroughputBenchmark(config)
    
    # Run throughput tests
    results = {}
    results['signal_throughput'] = await benchmark.benchmark_signal_throughput()
    results['data_throughput'] = await benchmark.benchmark_data_processing_throughput()
    
    # Save results
    import json
    with open('$OUTPUT_FILE', 'w') as f:
        json.dump({
            'benchmark_type': 'throughput',
            'results': results,
            'timestamp': $(date +%s)
        }, f, indent=2, default=str)
    
    print('Throughput benchmark completed')

asyncio.run(run_throughput())
"
    
    log_info "Throughput benchmark completed. Results: $OUTPUT_FILE"
}

# Function to run strategy benchmarks
run_strategy_benchmark() {
    log_info "Running strategy performance benchmarks..."
    generate_output_filename "strategy_benchmark"
    
    python -c "
import sys
sys.path.insert(0, '.')
from src.benchmarks.strategy_benchmark import StrategyBenchmark
from integration_tests import IntegrationTestConfig

config = IntegrationTestConfig()
benchmark = StrategyBenchmark(config)

strategies = '$STRATEGIES'.split(',')
results = {}

for strategy in strategies:
    print(f'Testing strategy: {strategy}')
    results[strategy] = benchmark.run_strategy_benchmark(strategy, duration='$DURATION')

# Save results
import json
with open('$OUTPUT_FILE', 'w') as f:
    json.dump({
        'benchmark_type': 'strategy',
        'results': results,
        'timestamp': $(date +%s),
        'duration': '$DURATION',
        'strategies': strategies
    }, f, indent=2, default=str)

print('Strategy benchmark completed')
"
    
    log_info "Strategy benchmark completed. Results: $OUTPUT_FILE"
}

# Function to run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$VERBOSE" == "true" ]]; then
        python integration_tests.py
    else
        python integration_tests.py 2>/dev/null
    fi
    
    log_info "Integration tests completed"
}

# Function to run simulation tests
run_simulation_tests() {
    log_info "Running simulation tests..."
    generate_output_filename "simulation_test"
    
    python -c "
import sys
import asyncio
sys.path.insert(0, '.')
from src.simulation.simulator import Simulator
from integration_tests import IntegrationTestConfig

config = IntegrationTestConfig()
simulator = Simulator(config.to_dict())

# Run simulation tests
results = simulator.run_backtest(
    strategies=['$STRATEGIES'.split(',')[0]],
    start_date='2024-01-01',
    end_date='2024-01-31',
    assets=['AAPL', 'MSFT']
)

# Save results
import json
with open('$OUTPUT_FILE', 'w') as f:
    json.dump({
        'benchmark_type': 'simulation',
        'results': results,
        'timestamp': $(date +%s)
    }, f, indent=2, default=str)

print('Simulation test completed')
"
    
    log_info "Simulation test completed. Results: $OUTPUT_FILE"
}

# Function to run optimization tests
run_optimization_tests() {
    log_info "Running optimization tests..."
    generate_output_filename "optimization_test"
    
    python -c "
import sys
sys.path.insert(0, '.')
from src.optimization.optimizer import Optimizer
from integration_tests import IntegrationTestConfig

config = IntegrationTestConfig()
optimizer = Optimizer(config.to_dict())

# Mock simulation results for optimization
mock_results = {
    'performance': {
        'total_return': 0.15,
        'sharpe_ratio': 1.2,
        'max_drawdown': 0.05
    }
}

# Run optimization
results = optimizer.optimize_strategy(
    'momentum',
    mock_results,
    metric='sharpe_ratio'
)

# Save results
import json
with open('$OUTPUT_FILE', 'w') as f:
    json.dump({
        'benchmark_type': 'optimization',
        'results': results,
        'timestamp': $(date +%s)
    }, f, indent=2, default=str)

print('Optimization test completed')
"
    
    log_info "Optimization test completed. Results: $OUTPUT_FILE"
}

# Function to run stress tests
run_stress_tests() {
    log_info "Running stress tests..."
    generate_output_filename "stress_test"
    
    python -c "
import sys
import asyncio
import time
sys.path.insert(0, '.')
from src.integration.system_orchestrator import SystemOrchestrator
from src.integration.data_pipeline import DataPipeline, DataPacket, DataType

async def stress_test():
    # Create orchestrator
    orchestrator = SystemOrchestrator()
    
    try:
        # Start system
        await orchestrator.start()
        
        # Generate high load
        if orchestrator.data_pipeline:
            start_time = time.time()
            packet_count = 1000
            
            for i in range(packet_count):
                packet = DataPacket(
                    data_type=DataType.MARKET_DATA,
                    timestamp=time.time(),
                    source='stress_test',
                    data={'symbol': f'TEST{i}', 'price': 100.0 + i, 'volume': 1000}
                )
                await orchestrator.data_pipeline.enqueue_packet(packet)
            
            # Wait for processing
            await asyncio.sleep(5)
            
            # Get results
            status = orchestrator.data_pipeline.get_status()
            processing_time = time.time() - start_time
            throughput = status['metrics']['packets_processed'] / processing_time
            
            results = {
                'packets_sent': packet_count,
                'packets_processed': status['metrics']['packets_processed'],
                'processing_time': processing_time,
                'throughput': throughput,
                'error_rate': status['metrics']['error_rate']
            }
            
            # Save results
            import json
            with open('$OUTPUT_FILE', 'w') as f:
                json.dump({
                    'benchmark_type': 'stress',
                    'results': results,
                    'timestamp': int(time.time())
                }, f, indent=2)
        
        # Stop system
        await orchestrator.stop()
        
    except Exception as e:
        print(f'Stress test error: {e}')
        if orchestrator.state.value == 'running':
            await orchestrator.stop()

asyncio.run(stress_test())
print('Stress test completed')
"
    
    log_info "Stress test completed. Results: $OUTPUT_FILE"
}

# Function to start monitoring dashboard
start_monitoring() {
    log_info "Starting monitoring dashboard..."
    
    python -c "
import sys
import asyncio
sys.path.insert(0, '.')
from src.integration.performance_monitor import PerformanceMonitor

async def start_monitor():
    monitor = PerformanceMonitor()
    await monitor.start()
    
    print('Monitoring dashboard started')
    print('Press Ctrl+C to stop')
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print('Stopping monitoring...')
        await monitor.stop()

asyncio.run(start_monitor())
"
}

# Function to show system status
show_status() {
    log_info "System Status"
    echo "=============="
    
    # Python environment
    echo "Python: $(python --version 2>&1)"
    echo "Virtual Environment: ${VIRTUAL_ENV:-Not activated}"
    
    # Project info
    echo "Project Root: $PROJECT_ROOT"
    echo "Working Directory: $(pwd)"
    
    # Check if system components are importable
    if python -c "from src.integration.system_orchestrator import SystemOrchestrator" 2>/dev/null; then
        echo "✅ System Orchestrator: Available"
    else
        echo "❌ System Orchestrator: Not available"
    fi
    
    if python -c "from src.integration.data_pipeline import DataPipeline" 2>/dev/null; then
        echo "✅ Data Pipeline: Available"
    else
        echo "❌ Data Pipeline: Not available"
    fi
    
    if python -c "from src.integration.performance_monitor import PerformanceMonitor" 2>/dev/null; then
        echo "✅ Performance Monitor: Available"
    else
        echo "❌ Performance Monitor: Not available"
    fi
    
    # Recent results
    if [[ -d "$PROJECT_ROOT/results" ]]; then
        local recent_files=$(find "$PROJECT_ROOT/results" -name "*.json" -type f -mtime -1 | wc -l)
        echo "Recent Results: $recent_files files (last 24h)"
    fi
    
    echo "=============="
}

# Main function
main() {
    cd "$PROJECT_ROOT"
    
    # Parse command line arguments
    local command="quick"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            quick|full|latency|throughput|strategy|integration|simulation|optimization|stress|monitoring|status)
                command="$1"
                shift
                ;;
            --duration)
                DURATION="$2"
                shift 2
                ;;
            --strategies)
                STRATEGIES="$2"
                shift 2
                ;;
            --assets)
                ASSETS="$2"
                shift 2
                ;;
            --output)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "AI News Trading Benchmark - Quick Execution"
    log_info "Command: $command"
    log_info "Duration: $DURATION"
    log_info "Strategies: $STRATEGIES"
    log_info "Assets: $ASSETS"
    
    # Check prerequisites (except for status command)
    if [[ "$command" != "status" ]]; then
        check_prerequisites
    fi
    
    # Execute command
    case "$command" in
        quick)
            run_quick_benchmark
            ;;
        full)
            run_full_benchmark
            ;;
        latency)
            run_latency_benchmark
            ;;
        throughput)
            run_throughput_benchmark
            ;;
        strategy)
            run_strategy_benchmark
            ;;
        integration)
            run_integration_tests
            ;;
        simulation)
            run_simulation_tests
            ;;
        optimization)
            run_optimization_tests
            ;;
        stress)
            run_stress_tests
            ;;
        monitoring)
            start_monitoring
            ;;
        status)
            show_status
            ;;
        *)
            log_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
    
    log_info "Benchmark execution completed"
}

# Run main function
main "$@"