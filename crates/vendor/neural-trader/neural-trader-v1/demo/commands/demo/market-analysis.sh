#!/bin/bash
# Claude Flow Command: Demo - Market Analysis

echo "📊 Claude Flow: Market Analysis Demo"
echo "===================================="
echo ""
echo "This demo showcases real-time market analysis with AI neural forecasting."
echo ""

# Function to display MCP tool usage
show_mcp_tool() {
    echo "🔧 MCP Tool: $1"
    echo "📋 Parameters:"
    echo "$2"
    echo ""
}

echo "### Step 1: Quick Market Analysis"
echo "Analyze multiple symbols with GPU acceleration for real-time insights."
echo ""
show_mcp_tool "mcp__ai-news-trader__quick_analysis" "  symbol: \"AAPL\"
  use_gpu: true"

echo "Expected Output:"
echo "- Current price and trend direction"
echo "- Technical indicators (RSI, MACD, Bollinger Bands)"
echo "- Buy/Sell/Hold recommendation"
echo "- Processing time with GPU acceleration"
echo ""
echo "---"
echo ""

echo "### Step 2: Neural Price Forecasting"
echo "Generate AI-powered price predictions with confidence intervals."
echo ""
show_mcp_tool "mcp__ai-news-trader__neural_forecast" "  symbol: \"NVDA\"
  horizon: 7
  confidence_level: 0.95
  use_gpu: true"

echo "Expected Output:"
echo "- 7-day price predictions with daily granularity"
echo "- Upper/lower confidence bounds (95% interval)"
echo "- Overall trend analysis (bullish/bearish/neutral)"
echo "- Model performance metrics (MAE, RMSE, R²)"
echo ""
echo "---"
echo ""

echo "### Step 3: System Performance Check"
echo "Monitor system resources and GPU utilization."
echo ""
show_mcp_tool "mcp__ai-news-trader__get_system_metrics" "  metrics: [\"cpu\", \"memory\", \"latency\", \"throughput\"]
  include_history: true
  time_range_minutes: 60"

echo "Expected Output:"
echo "- CPU and memory usage percentages"
echo "- API latency statistics (mean, p95, p99)"
echo "- Throughput metrics (requests/sec, trades/min)"
echo "- Historical trends for capacity planning"
echo ""
echo "---"
echo ""

echo "### Complete Market Analysis Workflow"
echo ""
cat << 'EOF'
# In Claude Code, execute these tools in sequence:

# 1. Analyze tech giants
for symbol in AAPL NVDA MSFT GOOGL TSLA:
  Use: mcp__ai-news-trader__quick_analysis
  Parameters: 
    symbol: $symbol
    use_gpu: true

# 2. Deep dive on top performers
Use: mcp__ai-news-trader__neural_forecast
Parameters:
  symbol: "NVDA"  # Or highest momentum stock
  horizon: 30      # 30-day forecast
  confidence_level: 0.95
  model_id: "transformer_v2"  # Latest model
  use_gpu: true

# 3. Check neural model health
Use: mcp__ai-news-trader__neural_model_status
Parameters:
  model_id: null  # Check all models

# 4. System diagnostics
Use: mcp__ai-news-trader__get_system_metrics
Parameters:
  metrics: ["cpu", "memory", "latency", "throughput", "gpu_utilization"]
  include_history: true
EOF

echo ""
echo "💡 Pro Tips:"
echo "- Enable GPU for 1000x faster neural predictions"
echo "- Combine with news sentiment for stronger signals"
echo "- Monitor system metrics during high-volume periods"
echo "- Use longer horizons (30+ days) for position trading"
echo ""
echo "📚 Full documentation: /workspaces/ai-news-trader/demo/guides/market_analysis_demo.md"