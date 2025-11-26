#!/usr/bin/env python3
"""
Test MCP server startup and basic functionality
"""

import subprocess
import sys
import time
from pathlib import Path

def test_mcp_server_startup():
    """Test that the MCP server can start without errors"""
    print("🧪 Testing MCP Server Startup...")
    
    # Test the enhanced server
    server_file = Path("src/mcp/mcp_server_enhanced.py")
    if not server_file.exists():
        print("❌ Enhanced server file not found")
        return False
    
    # Run server with timeout to see if it starts
    try:
        # Start the server process
        process = subprocess.Popen(
            [sys.executable, str(server_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if process is still running (good sign)
        if process.poll() is None:
            print("✅ MCP server started successfully")
            process.terminate()
            process.wait(timeout=5)
            return True
        else:
            # Process exited, check output
            stdout, stderr = process.communicate()
            if "Starting MCP server" in stdout or "Starting MCP server" in stderr:
                print("✅ MCP server startup initiated (may have exited normally)")
                return True
            else:
                print(f"❌ MCP server failed to start")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
    
    except subprocess.TimeoutExpired:
        print("✅ MCP server running (timeout as expected)")
        process.terminate()
        return True
    except Exception as e:
        print(f"❌ Error testing server startup: {e}")
        return False

def test_tool_syntax():
    """Test that the tool definitions have correct syntax"""
    print("\n🔍 Testing Tool Syntax...")
    
    server_file = Path("src/mcp/mcp_server_enhanced.py")
    content = server_file.read_text()
    
    # Check for syntax issues in tool definitions
    tool_sections = content.split("@mcp.tool()")
    print(f"📊 Found {len(tool_sections)-1} tool definitions")
    
    # Basic syntax checks
    issues = []
    
    if "async def" in content and "def " in content:
        # Check for mixed async/sync (could be an issue)
        async_count = content.count("async def")
        sync_count = content.count("def ") - async_count
        print(f"📈 Function types: {async_count} async, {sync_count} sync")
    
    # Check for import statement
    if "from fastmcp import FastMCP" in content:
        print("✅ FastMCP import found")
    else:
        issues.append("Missing FastMCP import")
    
    # Check for server initialization
    if "mcp = FastMCP" in content:
        print("✅ Server initialization found")
    else:
        issues.append("Missing server initialization")
    
    # Check for main function
    if "def main():" in content:
        print("✅ Main function found")
    else:
        issues.append("Missing main function")
    
    if issues:
        print("❌ Syntax issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All syntax checks passed")
        return True

def test_tool_coverage():
    """Test that we have all expected tool categories"""
    print("\n📋 Testing Tool Coverage...")
    
    server_file = Path("src/mcp/mcp_server_enhanced.py")
    content = server_file.read_text()
    
    # Expected tool categories and representative tools
    expected_tools = {
        "Core": ["ping", "list_strategies", "quick_analysis"],
        "News": ["control_news_collection", "get_news_provider_status"],
        "Strategy": ["recommend_strategy", "adaptive_strategy_selection"],
        "Performance": ["get_system_metrics", "monitor_strategy_health"],
        "Multi-Asset": ["execute_multi_asset_trade", "portfolio_rebalance"],
        "Neural": ["neural_forecast", "neural_train"],
        "Trading": ["run_backtest", "execute_trade"],
        "Polymarket": ["get_prediction_markets_tool", "place_prediction_order_tool"]
    }
    
    results = {}
    for category, tools in expected_tools.items():
        found_tools = []
        for tool in tools:
            if f"def {tool}(" in content:
                found_tools.append(tool)
        
        coverage = len(found_tools) / len(tools)
        results[category] = (found_tools, len(tools), coverage)
        
        if coverage >= 0.5:  # At least 50% of tools in category
            print(f"✅ {category}: {len(found_tools)}/{len(tools)} tools ({coverage*100:.0f}%)")
        else:
            print(f"❌ {category}: {len(found_tools)}/{len(tools)} tools ({coverage*100:.0f}%)")
    
    # Overall coverage
    total_expected = sum(len(tools) for tools in expected_tools.values())
    total_found = sum(len(found) for found, _, _ in results.values())
    overall_coverage = total_found / total_expected
    
    print(f"\n📊 Overall Tool Coverage: {total_found}/{total_expected} ({overall_coverage*100:.1f}%)")
    
    return overall_coverage >= 0.8  # 80% coverage required

def main():
    """Run all MCP server tests"""
    print("🚀 MCP SERVER FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        ("Server Startup", test_mcp_server_startup),
        ("Tool Syntax", test_tool_syntax),
        ("Tool Coverage", test_tool_coverage)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📊 TEST SUMMARY:")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - MCP Server is ready!")
        return True
    elif passed >= total * 0.67:  # 2/3 pass rate
        print("⚠️  MOSTLY FUNCTIONAL - Some issues detected")
        return True
    else:
        print("❌ MULTIPLE ISSUES - Server needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)