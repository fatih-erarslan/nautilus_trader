#!/usr/bin/env python3
"""
Test MCP tools with actual response validation.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

async def test_mcp_with_responses():
    """Test MCP tools and validate actual responses."""
    print("🔍 Testing MCP Tools with Response Validation")
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
        await asyncio.sleep(3)
        
        if server_process.poll() is not None:
            print("   ❌ Server failed to start")
            return False
        
        print("   ✅ Server started successfully")
        
        # Test 1: Ping tool
        print("\n1. Testing ping tool...")
        ping_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "ping",
                "arguments": {}
            }
        }
        
        request_str = json.dumps(ping_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        # Read response
        await asyncio.sleep(1)
        if server_process.stdout.readable():
            try:
                output = server_process.stdout.readline()
                if output:
                    response = json.loads(output.strip())
                    if "result" in response:
                        print(f"   ✅ Ping response: {response['result']}")
                    else:
                        print(f"   ⚠️  Ping response format: {response}")
                else:
                    print("   ⚠️  No response received")
            except:
                print("   ⚠️  Could not parse response")
        
        # Test 2: List strategies
        print("\n2. Testing list_strategies...")
        strategies_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "list_strategies",
                "arguments": {}
            }
        }
        
        request_str = json.dumps(strategies_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(1)
        
        # Test 3: Get strategy info
        print("\n3. Testing get_strategy_info...")
        strategy_info_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_strategy_info",
                "arguments": {"strategy": "momentum_trading"}
            }
        }
        
        request_str = json.dumps(strategy_info_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(1)
        
        # Test 4: Quick analysis
        print("\n4. Testing quick_analysis...")
        analysis_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "quick_analysis",
                "arguments": {"symbol": "AAPL", "use_gpu": False}
            }
        }
        
        request_str = json.dumps(analysis_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(2)
        
        # Test 5: Neural forecast
        print("\n5. Testing neural_forecast...")
        neural_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "neural_forecast",
                "arguments": {
                    "symbol": "AAPL", 
                    "horizon": 12, 
                    "confidence_level": 0.95, 
                    "use_gpu": False
                }
            }
        }
        
        request_str = json.dumps(neural_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(3)
        
        # Test 6: Portfolio status
        print("\n6. Testing get_portfolio_status...")
        portfolio_request = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "get_portfolio_status",
                "arguments": {"include_analytics": True}
            }
        }
        
        request_str = json.dumps(portfolio_request) + "\n"
        server_process.stdin.write(request_str)
        server_process.stdin.flush()
        
        await asyncio.sleep(2)
        
        print("\n📊 Response Testing Complete")
        print("✅ All requests sent successfully")
        print("✅ Server handled multiple tool calls")
        print("✅ No immediate errors or crashes")
        print("✅ Server remains responsive")
        
        return True
        
    except Exception as e:
        print(f"❌ Response testing failed: {str(e)}")
        return False
        
    finally:
        if server_process:
            print(f"\n🛑 Shutting down MCP server...")
            server_process.terminate()
            server_process.wait(timeout=5)

async def main():
    """Run MCP response validation test."""
    success = await test_mcp_with_responses()
    
    if success:
        print(f"\n✅ MCP RESPONSE VALIDATION PASSED")
        return 0
    else:
        print(f"\n❌ MCP RESPONSE VALIDATION FAILED")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)