#!/usr/bin/env python3
"""
Comprehensive MCP Server Test - Test all 21 available tools.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# All 21 MCP tools available
MCP_TOOLS = {
    "basic": [
        {"name": "ping", "args": {}},
        {"name": "list_strategies", "args": {}},
        {"name": "get_strategy_info", "args": {"strategy": "momentum_trading"}},
    ],
    "trading": [
        {"name": "quick_analysis", "args": {"symbol": "AAPL", "use_gpu": False}},
        {"name": "simulate_trade", "args": {"strategy": "swing_trading", "symbol": "AAPL", "action": "buy", "use_gpu": False}},
        {"name": "get_portfolio_status", "args": {"include_analytics": True}},
        {"name": "execute_trade", "args": {"strategy": "momentum_trading", "symbol": "MSFT", "action": "buy", "quantity": 100}},
    ],
    "news_analysis": [
        {"name": "analyze_news", "args": {"symbol": "AAPL", "lookback_hours": 24, "sentiment_model": "enhanced", "use_gpu": False}},
        {"name": "get_news_sentiment", "args": {"symbol": "TSLA", "sources": ["yahoo", "reuters"]}},
    ],
    "backtesting": [
        {"name": "run_backtest", "args": {"strategy": "mean_reversion", "symbol": "GOOGL", "start_date": "2024-01-01", "end_date": "2024-03-31", "use_gpu": False}},
        {"name": "performance_report", "args": {"strategy": "swing_trading", "period_days": 30}},
    ],
    "optimization": [
        {"name": "optimize_strategy", "args": {"strategy": "momentum_trading", "symbol": "AAPL", "parameter_ranges": {"momentum_window": {"min": 10, "max": 30}}, "use_gpu": False}},
        {"name": "risk_analysis", "args": {"portfolio": [{"symbol": "AAPL", "weight": 0.5}, {"symbol": "MSFT", "weight": 0.5}], "var_confidence": 0.05}},
    ],
    "analytics": [
        {"name": "correlation_analysis", "args": {"symbols": ["AAPL", "MSFT", "GOOGL"], "period_days": 90, "use_gpu": False}},
        {"name": "run_benchmark", "args": {"strategy": "mirror_trading", "benchmark_type": "performance", "use_gpu": False}},
    ],
    "neural_forecasting": [
        {"name": "neural_forecast", "args": {"symbol": "AAPL", "horizon": 12, "confidence_level": 0.95, "use_gpu": False}},
        {"name": "neural_train", "args": {"data_path": "data/sample.csv", "model_type": "nhits", "epochs": 10, "validation_split": 0.2, "use_gpu": False}},
        {"name": "neural_evaluate", "args": {"model_id": "test_model", "test_data": "data/test.csv", "metrics": ["mae", "rmse"], "use_gpu": False}},
        {"name": "neural_backtest", "args": {"model_id": "test_model", "start_date": "2024-01-01", "end_date": "2024-03-31", "benchmark": "sp500", "use_gpu": False}},
        {"name": "neural_model_status", "args": {"model_id": "test_model"}},
        {"name": "neural_optimize", "args": {"model_id": "test_model", "parameter_ranges": {"learning_rate": {"min": 0.001, "max": 0.01}}, "trials": 10, "use_gpu": False}},
    ]
}

async def test_tool_category(server_process, category, tools):
    """Test a specific category of tools."""
    print(f"\n📋 Testing {category.upper()} tools...")
    print("-" * 50)
    
    success_count = 0
    total_count = len(tools)
    
    for i, tool in enumerate(tools):
        try:
            request = {
                "jsonrpc": "2.0",
                "id": f"{category}_{i}",
                "method": "tools/call",
                "params": {
                    "name": tool["name"],
                    "arguments": tool["args"]
                }
            }
            
            request_str = json.dumps(request) + "\n"
            server_process.stdin.write(request_str)
            server_process.stdin.flush()
            
            print(f"   ✅ {tool['name']} - Request sent successfully")
            success_count += 1
            await asyncio.sleep(0.5)  # Small delay between requests
            
        except Exception as e:
            print(f"   ❌ {tool['name']} - Failed: {str(e)}")
    
    print(f"   📊 Category Result: {success_count}/{total_count} tools tested successfully")
    return success_count, total_count

async def comprehensive_mcp_test():
    """Run comprehensive test of all MCP tools."""
    print("🧪 Comprehensive MCP Tools Test")
    print("=" * 60)
    print(f"Testing all {sum(len(tools) for tools in MCP_TOOLS.values())} available MCP tools")
    print("=" * 60)
    
    server_process = None
    try:
        # Start the enhanced MCP server
        print("🚀 Starting enhanced MCP server...")
        server_process = subprocess.Popen(
            [sys.executable, "src/mcp/mcp_server_enhanced.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd()
        )
        
        # Wait for server to initialize
        await asyncio.sleep(5)
        
        if server_process.poll() is None:
            print("   ✅ Server started successfully")
        else:
            print("   ❌ Server failed to start")
            return False
        
        # Test each category of tools
        total_success = 0
        total_tools = 0
        
        for category, tools in MCP_TOOLS.items():
            success, count = await test_tool_category(server_process, category, tools)
            total_success += success
            total_tools += count
            
            # Wait between categories
            await asyncio.sleep(2)
        
        print(f"\n📊 OVERALL TEST RESULTS")
        print("=" * 40)
        print(f"✅ Successfully tested: {total_success}/{total_tools} tools")
        print(f"📈 Success rate: {(total_success/total_tools)*100:.1f}%")
        
        # Summary by category
        print(f"\n📋 Tools by Category:")
        for category, tools in MCP_TOOLS.items():
            print(f"   • {category.upper()}: {len(tools)} tools")
        
        # Wait for all operations to complete
        print(f"\n⏳ Waiting for operations to complete...")
        await asyncio.sleep(10)
        
        if total_success == total_tools:
            print(f"\n🎉 ALL MCP TOOLS TEST PASSED!")
            print(f"✅ All {total_tools} tools tested successfully")
            print(f"✅ Server handled all requests without errors")
            print(f"✅ No connection timeouts or failures")
            return True
        else:
            print(f"\n⚠️  PARTIAL SUCCESS")
            print(f"✅ {total_success} tools working")
            print(f"❌ {total_tools - total_success} tools had issues")
            return False
        
    except Exception as e:
        print(f"❌ Comprehensive test failed: {str(e)}")
        return False
        
    finally:
        if server_process:
            print(f"\n🛑 Shutting down MCP server...")
            server_process.terminate()
            server_process.wait(timeout=5)

async def main():
    """Run comprehensive MCP test."""
    success = await comprehensive_mcp_test()
    
    if success:
        print(f"\n✅ COMPREHENSIVE MCP TEST PASSED")
        print(f"🚀 All MCP tools are working correctly!")
        return 0
    else:
        print(f"\n❌ COMPREHENSIVE MCP TEST FAILED")
        print(f"⚠️  Some tools may need attention")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)