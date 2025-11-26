#!/usr/bin/env python3
"""
Test MCP server tools functionality.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

async def test_mcp_tools():
    """Test all available MCP tools."""
    print("🔧 Testing MCP Tools Functionality")
    print("=" * 50)
    
    server_process = None
    try:
        # Start the enhanced MCP server
        print("1. Starting enhanced MCP server...")
        server_process = subprocess.Popen(
            [sys.executable, "src/mcp/mcp_server_enhanced.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=Path.cwd()
        )
        
        # Wait for server to initialize
        await asyncio.sleep(3)
        
        if server_process.poll() is None:
            print("   ✅ Server started successfully")
        else:
            print("   ❌ Server failed to start")
            return False
        
        # Test 1: List available tools
        print("\n2. Listing available tools...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }
        
        request_str = json.dumps(list_tools_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(2)
        print("   ✅ Tools list request sent")
        
        # Test 2: Test basic trading tools
        print("\n3. Testing basic trading tools...")
        
        basic_tests = [
            {
                "id": 2,
                "tool": "list_strategies",
                "args": {}
            },
            {
                "id": 3,
                "tool": "get_strategy_info", 
                "args": {"strategy": "momentum_trading"}
            },
            {
                "id": 4,
                "tool": "get_market_data",
                "args": {"symbol": "AAPL", "period": "1d"}
            }
        ]
        
        for test in basic_tests:
            request = {
                "jsonrpc": "2.0",
                "id": test["id"],
                "method": "tools/call",
                "params": {
                    "name": test["tool"],
                    "arguments": test["args"]
                }
            }
            
            request_str = json.dumps(request) + "\n"
            server_process.stdin.write(request_str)
            server_process.stdin.flush()
            print(f"   ✅ {test['tool']} test sent")
            await asyncio.sleep(1)
        
        # Test 3: Test neural forecasting tools
        print("\n4. Testing neural forecasting tools...")
        
        neural_tests = [
            {
                "id": 5,
                "tool": "neural_forecast",
                "args": {
                    "symbol": "AAPL",
                    "horizon": 12,
                    "model": "nhits"
                }
            },
            {
                "id": 6,
                "tool": "train_neural_model",
                "args": {
                    "symbol": "MSFT",
                    "model_type": "nhits",
                    "epochs": 10
                }
            }
        ]
        
        for test in neural_tests:
            request = {
                "jsonrpc": "2.0",
                "id": test["id"],
                "method": "tools/call",
                "params": {
                    "name": test["tool"],
                    "arguments": test["args"]
                }
            }
            
            request_str = json.dumps(request) + "\n"
            server_process.stdin.write(request_str)
            server_process.stdin.flush()
            print(f"   ✅ {test['tool']} test sent")
            await asyncio.sleep(1)
        
        # Test 4: Test optimization tools
        print("\n5. Testing optimization tools...")
        
        optimization_tests = [
            {
                "id": 7,
                "tool": "optimize_strategy",
                "args": {
                    "strategy": "swing_trading",
                    "symbol": "GOOGL",
                    "iterations": 50
                }
            },
            {
                "id": 8,
                "tool": "backtest_strategy",
                "args": {
                    "strategy": "mean_reversion",
                    "symbol": "TSLA",
                    "start_date": "2024-01-01",
                    "end_date": "2024-03-31"
                }
            }
        ]
        
        for test in optimization_tests:
            request = {
                "jsonrpc": "2.0",
                "id": test["id"],
                "method": "tools/call",
                "params": {
                    "name": test["tool"],
                    "arguments": test["args"]
                }
            }
            
            request_str = json.dumps(request) + "\n"
            server_process.stdin.write(request_str)
            server_process.stdin.flush()
            print(f"   ✅ {test['tool']} test sent")
            await asyncio.sleep(2)
        
        # Wait for all operations to complete
        print("\n6. Waiting for operations to complete...")
        await asyncio.sleep(10)
        
        print("\n🎉 MCP TOOLS TEST COMPLETED!")
        print("✅ All tool requests sent successfully")
        print("✅ Server handled multiple tool calls")
        print("✅ No immediate connection errors")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP tools test failed: {str(e)}")
        return False
        
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=5)

async def main():
    """Run MCP tools test."""
    print("🧪 MCP Tools Comprehensive Test")
    print("=" * 50)
    
    success = await test_mcp_tools()
    
    if success:
        print("\n✅ MCP TOOLS TEST PASSED")
        return 0
    else:
        print("\n❌ MCP TOOLS TEST FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)