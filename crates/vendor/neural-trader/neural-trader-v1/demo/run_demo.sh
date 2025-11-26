#!/bin/bash
# AI News Trading Platform - Demo Launcher

echo "🚀 AI News Trading Platform - Demo Suite"
echo "========================================"
echo ""
echo "Select a demo option:"
echo ""
echo "1) 📊 View Demo Index (start here)"
echo "2) 🤖 Run Parallel Agent Demo (5 agents)"
echo "3) 📈 Market Analysis Demo"
echo "4) 📰 News Sentiment Demo"
echo "5) 🎯 Strategy Optimization Demo"
echo "6) ⚠️  Risk Management Demo"
echo "7) 💹 Trading Execution Demo"
echo "8) 📁 Browse Demo Files"
echo ""
read -p "Enter your choice (1-8): " choice

case $choice in
    1)
        echo "Opening Demo Index..."
        cat docs/DEMO_INDEX.md
        ;;
    2)
        echo "Launching Parallel Agent Demo..."
        cd scripts && ./run_parallel_demo.sh
        ;;
    3)
        echo "Opening Market Analysis Demo..."
        cat guides/market_analysis_demo.md
        ;;
    4)
        echo "Opening News Sentiment Demo..."
        cat guides/news_analysis_demo.md
        ;;
    5)
        echo "Opening Strategy Optimization Demo..."
        cat guides/strategy_optimization_demo.md
        ;;
    6)
        echo "Opening Risk Management Demo..."
        cat guides/risk_management_demo.md
        ;;
    7)
        echo "Opening Trading Execution Demo..."
        cat guides/trading_execution_demo.md
        ;;
    8)
        echo ""
        echo "Demo Files Structure:"
        echo "-------------------"
        tree -L 2 .
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-8."
        ;;
esac

echo ""
echo "---"
echo "To use MCP tools directly in Claude Code, use the prefix:"
echo "mcp__ai-news-trader__[tool_name]"
echo ""
echo "Example: mcp__ai-news-trader__quick_analysis"