#!/bin/bash
# CLI Examples for AI News Trading Platform with Neural Forecasting
#
# This script demonstrates comprehensive usage of the command-line interfaces
# for the AI News Trading Platform including:
# - claude-flow CLI for orchestration
# - MCP server management
# - Neural forecasting operations
# - Trading strategy execution
# - Performance monitoring
#
# Usage:
#   ./cli_examples.sh [EXAMPLE_NAME]
#
# Examples:
#   ./cli_examples.sh basic_setup
#   ./cli_examples.sh neural_forecasting
#   ./cli_examples.sh trading_strategies
#   ./cli_examples.sh performance_monitoring
#   ./cli_examples.sh all

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}======================================${NC}"
}

print_step() {
    echo -e "${GREEN}Step $1: $2${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for user input
wait_for_input() {
    echo -e "${YELLOW}Press Enter to continue...${NC}"
    read
}

# Example 1: Basic System Setup and Status
example_basic_setup() {
    print_header "EXAMPLE 1: Basic System Setup and Status"
    
    print_step "1.1" "Checking system requirements"
    
    # Check Python
    if command_exists python; then
        PYTHON_VERSION=$(python --version 2>&1)
        print_success "Python available: $PYTHON_VERSION"
    else
        print_error "Python not found. Please install Python 3.8+."
        return 1
    fi
    
    # Check GPU availability
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -3
    else
        print_warning "NVIDIA GPU not detected. CPU mode will be used."
    fi
    
    # Check claude-flow
    if [ -f "./claude-flow" ]; then
        print_success "claude-flow CLI found"
    else
        print_error "claude-flow CLI not found. Please ensure you're in the project directory."
        return 1
    fi
    
    wait_for_input
    
    print_step "1.2" "Initializing claude-flow configuration"
    
    # Initialize claude-flow with neural forecasting
    echo "Initializing claude-flow with neural forecasting support..."
    ./claude-flow init --neural-forecast
    
    print_success "claude-flow initialized"
    
    wait_for_input
    
    print_step "1.3" "Checking system status"
    
    # Check overall system status
    echo "Checking system status..."
    ./claude-flow status --verbose --check-gpu --check-models
    
    print_success "System status check completed"
    
    wait_for_input
    
    print_step "1.4" "Starting core services"
    
    # Start MCP server in background
    echo "Starting MCP server..."
    python mcp_server_enhanced.py &
    MCP_PID=$!
    echo "MCP server started with PID: $MCP_PID"
    
    # Wait for server to start
    sleep 5
    
    # Test MCP server connectivity
    echo "Testing MCP server connectivity..."
    if curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"ping","id":1}' | grep -q "ok"; then
        print_success "MCP server is responding"
    else
        print_error "MCP server is not responding"
    fi
    
    # Start claude-flow orchestrator
    echo "Starting claude-flow orchestrator..."
    ./claude-flow start --neural-forecast --gpu --port 8080 &
    CLAUDE_FLOW_PID=$!
    echo "Claude-flow started with PID: $CLAUDE_FLOW_PID"
    
    sleep 3
    
    print_success "Core services started successfully"
    
    # Store PIDs for cleanup
    echo "$MCP_PID" > /tmp/mcp_server.pid
    echo "$CLAUDE_FLOW_PID" > /tmp/claude_flow.pid
    
    wait_for_input
    
    print_step "1.5" "Verifying service health"
    
    # Check service health
    echo "Checking service health..."
    
    # MCP server health
    if curl -s http://localhost:3000/health | grep -q "ok"; then
        print_success "MCP server health check passed"
    else
        print_warning "MCP server health check failed"
    fi
    
    # Claude-flow health
    if curl -s http://localhost:8080/health | grep -q "ok"; then
        print_success "Claude-flow health check passed"
    else
        print_warning "Claude-flow health check failed"
    fi
    
    print_info "Basic setup completed. Services are running in background."
    print_info "MCP Server: http://localhost:3000"
    print_info "Claude-Flow UI: http://localhost:8080"
}

# Example 2: Neural Forecasting Operations
example_neural_forecasting() {
    print_header "EXAMPLE 2: Neural Forecasting Operations"
    
    print_step "2.1" "Generating basic neural forecasts"
    
    # Generate forecast for AAPL
    echo "Generating 30-day forecast for AAPL..."
    ./claude-flow neural forecast AAPL --horizon 30 --gpu --confidence 80,95
    
    print_success "AAPL forecast completed"
    
    wait_for_input
    
    print_step "2.2" "Multi-symbol forecasting"
    
    # Generate forecasts for multiple symbols
    echo "Generating forecasts for tech portfolio..."
    ./claude-flow neural forecast AAPL,GOOGL,MSFT,NVDA --horizon 14 --gpu --output json
    
    print_success "Multi-symbol forecasting completed"
    
    wait_for_input
    
    print_step "2.3" "Advanced forecasting with custom parameters"
    
    # Advanced forecasting with custom model
    echo "Running advanced forecasting with custom parameters..."
    ./claude-flow sparc neural "Generate 30-day forecasts for TSLA using NHITS model with GPU acceleration and confidence intervals" --mode implement --gpu
    
    print_success "Advanced forecasting completed"
    
    wait_for_input
    
    print_step "2.4" "Model training and optimization"
    
    # Train custom neural model
    echo "Training custom neural forecasting model..."
    ./claude-flow neural train --model nhits --epochs 50 --batch-size 32 --gpu
    
    print_success "Model training completed"
    
    # Optimize model hyperparameters
    echo "Optimizing model hyperparameters..."
    ./claude-flow neural optimize nhits --trials 20 --gpu --metric mape
    
    print_success "Hyperparameter optimization completed"
    
    wait_for_input
    
    print_step "2.5" "Forecast evaluation and validation"
    
    # Evaluate forecast accuracy
    echo "Evaluating forecast accuracy..."
    ./claude-flow neural evaluate --model nhits --symbols AAPL --backtest-days 60
    
    print_success "Forecast evaluation completed"
    
    # Cross-validation
    echo "Running cross-validation..."
    ./claude-flow neural cross-validate --model nhits --symbols AAPL,GOOGL --folds 5
    
    print_success "Cross-validation completed"
}

# Example 3: Trading Strategy Operations
example_trading_strategies() {
    print_header "EXAMPLE 3: Trading Strategy Operations"
    
    print_step "3.1" "Listing available trading strategies"
    
    # List all available strategies
    echo "Listing available trading strategies..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"list_strategies","id":1}' | \
        python -m json.tool
    
    print_success "Strategy listing completed"
    
    wait_for_input
    
    print_step "3.2" "Getting detailed strategy information"
    
    # Get momentum strategy details
    echo "Getting momentum strategy details..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"get_strategy_info","params":{"strategy":"momentum_trading_optimized"},"id":1}' | \
        python -m json.tool
    
    print_success "Strategy information retrieved"
    
    wait_for_input
    
    print_step "3.3" "Running quick market analysis"
    
    # Quick analysis for multiple symbols
    symbols=("AAPL" "GOOGL" "MSFT" "TSLA")
    
    for symbol in "${symbols[@]}"; do
        echo "Analyzing $symbol..."
        curl -s -X POST http://localhost:3000/mcp \
            -H "Content-Type: application/json" \
            -d "{\"jsonrpc\":\"2.0\",\"method\":\"quick_analysis\",\"params\":{\"symbol\":\"$symbol\",\"use_gpu\":true},\"id\":1}" | \
            python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print(f\"$symbol Analysis:\")
    analysis = result.get('analysis', {})
    print(f\"  Trend: {analysis.get('trend', 'N/A')}\")
    print(f\"  Momentum: {analysis.get('momentum', 'N/A')}\")
    neural = result.get('neural_forecast', {})
    print(f\"  Neural Forecast: {neural.get('next_day', 'N/A')} (confidence: {neural.get('confidence', 'N/A')})\")
else:
    print(f\"Error analyzing $symbol: {data.get('error', 'Unknown error')}\")
"
    done
    
    print_success "Market analysis completed"
    
    wait_for_input
    
    print_step "3.4" "Simulating trades"
    
    # Simulate momentum trades
    echo "Simulating momentum trading strategy..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"simulate_trade","params":{"strategy":"momentum_trading_optimized","symbol":"AAPL","action":"buy","use_gpu":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Trade Simulation Results:')
    print(f\"  Symbol: {result.get('symbol', 'N/A')}\")
    print(f\"  Action: {result.get('action', 'N/A')}\")
    print(f\"  Entry Price: ${result.get('entry_price', 'N/A')}\")
    print(f\"  Expected Return: {result.get('expected_return', 'N/A')}\")
    neural = result.get('neural_forecast_support', {})
    print(f\"  Neural Forecast Alignment: {neural.get('forecast_alignment', 'N/A')}\")
else:
    print(f\"Simulation error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Trade simulation completed"
    
    wait_for_input
    
    print_step "3.5" "Portfolio status and analytics"
    
    # Get portfolio status
    echo "Getting portfolio status..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"get_portfolio_status","params":{"include_analytics":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    portfolio = result.get('portfolio', {})
    print('Portfolio Status:')
    print(f\"  Total Value: ${portfolio.get('total_value', 'N/A')}\")
    print(f\"  Cash: ${portfolio.get('cash', 'N/A')}\")
    
    analytics = result.get('analytics', {})
    if analytics:
        print('  Analytics:')
        print(f\"    Total Return: {analytics.get('total_return', 'N/A')}\")
        print(f\"    Sharpe Ratio: {analytics.get('sharpe_ratio', 'N/A')}\")
        print(f\"    Max Drawdown: {analytics.get('max_drawdown', 'N/A')}\")
else:
    print(f\"Portfolio error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Portfolio analysis completed"
}

# Example 4: Performance Monitoring and Benchmarking
example_performance_monitoring() {
    print_header "EXAMPLE 4: Performance Monitoring and Benchmarking"
    
    print_step "4.1" "Real-time system monitoring"
    
    # Start real-time monitoring
    echo "Starting real-time system monitoring (30 seconds)..."
    timeout 30s ./claude-flow monitor --neural-metrics --gpu-metrics --refresh 2 || true
    
    print_success "Monitoring session completed"
    
    wait_for_input
    
    print_step "4.2" "Running performance benchmarks"
    
    # Run neural forecasting benchmarks
    echo "Running neural forecasting benchmarks..."
    cd benchmark/
    python benchmark_cli.py neural --models nhits,nbeats --symbols AAPL,GOOGL --gpu --quick-test
    cd ..
    
    print_success "Neural forecasting benchmarks completed"
    
    wait_for_input
    
    print_step "4.3" "GPU performance testing"
    
    # Test GPU performance
    echo "Testing GPU performance..."
    cd benchmark/
    python benchmark_cli.py gpu --latency-test --memory-test --throughput-test
    cd ..
    
    print_success "GPU performance testing completed"
    
    wait_for_input
    
    print_step "4.4" "Strategy performance comparison"
    
    # Compare strategy performance
    echo "Comparing strategy performance..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"run_benchmark","params":{"strategy":"momentum_trading_optimized","benchmark_type":"performance","use_gpu":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Strategy Benchmark Results:')
    
    system_perf = result.get('system_performance', {})
    print('  System Performance:')
    print(f\"    GPU Utilization: {system_perf.get('gpu_utilization', 'N/A')}\")
    print(f\"    Inference Latency: {system_perf.get('inference_latency_ms', 'N/A')}ms\")
    print(f\"    Throughput: {system_perf.get('throughput_ops_sec', 'N/A')} ops/sec\")
    
    strategy_bench = result.get('strategy_benchmarks', {})
    print('  Strategy Benchmarks:')
    print(f\"    Sharpe Ratio Rank: {strategy_bench.get('sharpe_ratio_rank', 'N/A')}\")
    print(f\"    Overall Score: {strategy_bench.get('overall_score', 'N/A')}\")
else:
    print(f\"Benchmark error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Strategy benchmarking completed"
    
    wait_for_input
    
    print_step "4.5" "Generating performance reports"
    
    # Generate comprehensive performance report
    echo "Generating performance report..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"performance_report","params":{"strategy":"momentum_trading_optimized","period_days":30,"include_benchmark":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Performance Report:')
    
    metrics = result.get('performance_metrics', {})
    print('  Performance Metrics:')
    print(f\"    Total Return: {metrics.get('total_return', 'N/A')}\")
    print(f\"    Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}\")
    print(f\"    Max Drawdown: {metrics.get('max_drawdown', 'N/A')}\")
    
    neural = result.get('neural_forecast_impact', {})
    print('  Neural Forecast Impact:')
    print(f\"    Accuracy: {neural.get('forecast_accuracy', 'N/A')}\")
    print(f\"    Signal Enhancement: {neural.get('signal_enhancement', 'N/A')}\")
    
    benchmark = result.get('benchmark_comparison', {})
    print('  Benchmark Comparison:')
    print(f\"    Excess Return: {benchmark.get('excess_return', 'N/A')}\")
    print(f\"    Information Ratio: {benchmark.get('information_ratio', 'N/A')}\")
else:
    print(f\"Report error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Performance report generated"
}

# Example 5: Advanced Operations and Automation
example_advanced_operations() {
    print_header "EXAMPLE 5: Advanced Operations and Automation"
    
    print_step "5.1" "Automated agent coordination"
    
    # Spawn multiple agents for coordinated analysis
    echo "Spawning coordinated agent swarm..."
    ./claude-flow swarm "Analyze current market conditions and generate trading recommendations using neural forecasting" \
        --strategy analysis --mode distributed --max-agents 3 --parallel --monitor
    
    print_success "Agent swarm analysis completed"
    
    wait_for_input
    
    print_step "5.2" "Memory-driven coordination"
    
    # Store market analysis in memory
    echo "Storing market analysis in memory..."
    ./claude-flow memory store "market_analysis_$(date +%Y%m%d)" "Current market shows bullish momentum with neural forecasts supporting upward trend"
    
    # Retrieve and use stored analysis
    echo "Retrieving stored analysis..."
    ./claude-flow memory get "market_analysis_$(date +%Y%m%d)"
    
    print_success "Memory operations completed"
    
    wait_for_input
    
    print_step "5.3" "Workflow automation"
    
    # Create automated trading workflow
    cat > /tmp/trading_workflow.yaml << EOF
name: "Daily Trading Analysis"
description: "Automated daily analysis and trading signal generation"
schedule: "0 9 * * 1-5"  # Weekdays at 9 AM
steps:
  - name: "market_analysis"
    action: "neural_forecast"
    symbols: ["AAPL", "GOOGL", "MSFT", "TSLA"]
    horizon: 5
  - name: "generate_signals"
    action: "trading_strategy"
    strategy: "momentum_trading_optimized"
    depends_on: "market_analysis"
  - name: "risk_check"
    action: "risk_analysis"
    portfolio: "main"
    depends_on: "generate_signals"
  - name: "execute_trades"
    action: "conditional_execution"
    conditions: 
      - "risk_check.within_limits == true"
      - "generate_signals.confidence > 0.7"
    depends_on: "risk_check"
EOF
    
    echo "Created automated trading workflow..."
    ./claude-flow workflow /tmp/trading_workflow.yaml --dry-run
    
    print_success "Workflow automation configured"
    
    wait_for_input
    
    print_step "5.4" "Backtesting with neural forecasts"
    
    # Run comprehensive backtest
    echo "Running comprehensive backtest..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"run_backtest","params":{"strategy":"momentum_trading_optimized","symbol":"AAPL","start_date":"2023-01-01","end_date":"2024-06-01","use_gpu":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Backtest Results:')
    
    performance = result.get('performance', {})
    print('  Performance:')
    print(f\"    Total Return: {performance.get('total_return', 'N/A')}\")
    print(f\"    Annual Return: {performance.get('annual_return', 'N/A')}\")
    print(f\"    Sharpe Ratio: {performance.get('sharpe_ratio', 'N/A')}\")
    print(f\"    Max Drawdown: {performance.get('max_drawdown', 'N/A')}\")
    
    neural = result.get('neural_forecast_contribution', {})
    print('  Neural Forecast Contribution:')
    print(f\"    Accuracy Improvement: {neural.get('accuracy_improvement', 'N/A')}\")
    print(f\"    Return Enhancement: {neural.get('return_enhancement', 'N/A')}\")
    
    gpu = result.get('gpu_acceleration', {})
    print('  GPU Performance:')
    print(f\"    Speedup: {gpu.get('speedup', 'N/A')}\")
    print(f\"    GPU Utilization: {gpu.get('gpu_utilization', 'N/A')}\")
else:
    print(f\"Backtest error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Backtesting completed"
    
    wait_for_input
    
    print_step "5.5" "Model optimization and deployment"
    
    # Optimize strategy parameters
    echo "Optimizing strategy parameters..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"optimize_strategy","params":{"strategy":"momentum_trading_optimized","symbol":"AAPL","parameter_ranges":{"momentum_threshold":[0.1,0.8],"lookback_period":[5,60]},"use_gpu":true,"max_iterations":50},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Strategy Optimization Results:')
    
    best_params = result.get('best_parameters', {})
    print('  Best Parameters:')
    for param, value in best_params.items():
        print(f\"    {param}: {value}\")
    
    opt_results = result.get('optimization_results', {})
    print('  Optimization Results:')
    print(f\"    Best Sharpe: {opt_results.get('best_sharpe', 'N/A')}\")
    print(f\"    Best Return: {opt_results.get('best_return', 'N/A')}\")
    
    improvement = opt_results.get('improvement_vs_baseline', {})
    print('  Improvement vs Baseline:')
    print(f\"    Sharpe Improvement: {improvement.get('sharpe_improvement', 'N/A')}\")
    print(f\"    Return Improvement: {improvement.get('return_improvement', 'N/A')}\")
else:
    print(f\"Optimization error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Strategy optimization completed"
}

# Example 6: Risk Management and Analytics
example_risk_management() {
    print_header "EXAMPLE 6: Risk Management and Analytics"
    
    print_step "6.1" "Portfolio risk analysis"
    
    # Define sample portfolio
    portfolio='[
        {"symbol": "AAPL", "quantity": 100, "price": 150.0},
        {"symbol": "GOOGL", "quantity": 50, "price": 2500.0},
        {"symbol": "MSFT", "quantity": 75, "price": 300.0}
    ]'
    
    # Run comprehensive risk analysis
    echo "Running comprehensive risk analysis..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"risk_analysis\",\"params\":{\"portfolio\":$portfolio,\"use_gpu\":true,\"use_monte_carlo\":true},\"id\":1}" | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Risk Analysis Results:')
    
    portfolio = result.get('portfolio_summary', {})
    print('  Portfolio Summary:')
    print(f\"    Total Value: ${portfolio.get('total_value', 'N/A')}\")
    print(f\"    Position Count: {portfolio.get('position_count', 'N/A')}\")
    print(f\"    Concentration Risk: {portfolio.get('concentration_risk', 'N/A')}\")
    
    var = result.get('var_analysis', {})
    print('  VaR Analysis:')
    print(f\"    VaR 95%: {var.get('var_95', 'N/A')}\")
    print(f\"    Expected Shortfall 95%: {var.get('expected_shortfall_95', 'N/A')}\")
    
    neural = result.get('neural_risk_modeling', {})
    print('  Neural Risk Modeling:')
    print(f\"    Tail Risk Forecast: {neural.get('tail_risk_forecast', 'N/A')}\")
    print(f\"    Volatility Forecast: {neural.get('volatility_forecast', 'N/A')}\")
    print(f\"    Confidence: {neural.get('confidence', 'N/A')}\")
else:
    print(f\"Risk analysis error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Risk analysis completed"
    
    wait_for_input
    
    print_step "6.2" "Correlation analysis"
    
    # Analyze asset correlations
    echo "Analyzing asset correlations..."
    curl -s -X POST http://localhost:3000/mcp \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"correlation_analysis","params":{"symbols":["AAPL","GOOGL","MSFT","TSLA"],"use_gpu":true},"id":1}' | \
        python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print('Correlation Analysis Results:')
    
    rolling = result.get('rolling_correlations', {})
    print('  Rolling Correlations:')
    print(f\"    Average Correlation: {rolling.get('avg_correlation', 'N/A')}\")
    print(f\"    Max Correlation: {rolling.get('max_correlation', 'N/A')}\")
    print(f\"    Correlation Stability: {rolling.get('correlation_stability', 'N/A')}\")
    
    neural = result.get('neural_correlation_forecast', {})
    print('  Neural Correlation Forecast:')
    next_corr = neural.get('next_period_correlations', {})
    for pair, corr in next_corr.items():
        if pair != 'confidence':
            print(f\"    {pair}: {corr}\")
else:
    print(f\"Correlation analysis error: {data.get('error', 'Unknown error')}\")
"
    
    print_success "Correlation analysis completed"
    
    wait_for_input
    
    print_step "6.3" "News sentiment analysis"
    
    # Analyze news sentiment for risk assessment
    symbols=("AAPL" "GOOGL" "MSFT")
    
    for symbol in "${symbols[@]}"; do
        echo "Analyzing news sentiment for $symbol..."
        curl -s -X POST http://localhost:3000/mcp \
            -H "Content-Type: application/json" \
            -d "{\"jsonrpc\":\"2.0\",\"method\":\"analyze_news\",\"params\":{\"symbol\":\"$symbol\",\"use_gpu\":true},\"id\":1}" | \
            python -c "
import json, sys
data = json.load(sys.stdin)
if 'result' in data:
    result = data['result']
    print(f\"$symbol News Sentiment:\")
    
    sentiment = result.get('sentiment_analysis', {})
    print(f\"  Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}\")
    print(f\"  Sentiment Score: {sentiment.get('sentiment_score', 'N/A')}\")
    print(f\"  Confidence: {sentiment.get('confidence', 'N/A')}\")
    
    impact = result.get('impact_prediction', {})
    print(f\"  Impact Prediction:\")
    print(f\"    Price Impact: {impact.get('price_impact', 'N/A')}\")
    print(f\"    Time Horizon: {impact.get('time_horizon', 'N/A')}\")
else:
    print(f\"Sentiment analysis error for $symbol: {data.get('error', 'Unknown error')}\")
"
    done
    
    print_success "News sentiment analysis completed"
}

# Cleanup function
cleanup() {
    print_header "Cleaning up background processes"
    
    # Kill background processes
    if [ -f /tmp/mcp_server.pid ]; then
        MCP_PID=$(cat /tmp/mcp_server.pid)
        if kill -0 $MCP_PID 2>/dev/null; then
            echo "Stopping MCP server (PID: $MCP_PID)..."
            kill $MCP_PID
        fi
        rm -f /tmp/mcp_server.pid
    fi
    
    if [ -f /tmp/claude_flow.pid ]; then
        CLAUDE_FLOW_PID=$(cat /tmp/claude_flow.pid)
        if kill -0 $CLAUDE_FLOW_PID 2>/dev/null; then
            echo "Stopping Claude-flow (PID: $CLAUDE_FLOW_PID)..."
            kill $CLAUDE_FLOW_PID
        fi
        rm -f /tmp/claude_flow.pid
    fi
    
    # Clean up temporary files
    rm -f /tmp/trading_workflow.yaml
    
    print_success "Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main function
main() {
    local example_name=${1:-"all"}
    
    echo -e "${PURPLE}🚀 AI News Trading Platform - CLI Examples${NC}"
    echo -e "${PURPLE}==============================================${NC}"
    echo ""
    
    case $example_name in
        "basic_setup")
            example_basic_setup
            ;;
        "neural_forecasting")
            example_neural_forecasting
            ;;
        "trading_strategies")
            example_trading_strategies
            ;;
        "performance_monitoring")
            example_performance_monitoring
            ;;
        "advanced_operations")
            example_advanced_operations
            ;;
        "risk_management")
            example_risk_management
            ;;
        "all")
            example_basic_setup
            example_neural_forecasting
            example_trading_strategies
            example_performance_monitoring
            example_advanced_operations
            example_risk_management
            ;;
        *)
            print_error "Unknown example: $example_name"
            echo "Available examples:"
            echo "  basic_setup"
            echo "  neural_forecasting"
            echo "  trading_strategies"
            echo "  performance_monitoring"
            echo "  advanced_operations"
            echo "  risk_management"
            echo "  all"
            exit 1
            ;;
    esac
    
    echo ""
    print_header "All Examples Completed Successfully!"
    echo ""
    print_info "For more information:"
    print_info "  📚 Documentation: docs/"
    print_info "  🔧 API Reference: docs/api/"
    print_info "  📖 Tutorials: docs/tutorials/"
    print_info "  🎯 User Guides: docs/guides/"
    echo ""
    print_success "Thank you for using the AI News Trading Platform!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi