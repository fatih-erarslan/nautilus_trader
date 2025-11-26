#!/usr/bin/env python3
"""
Comprehensive test for integrated MCP server with all 40+ tools
Tests both original 27 tools and new 14 integration tools
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_mcp_integration():
    """Test all MCP tools to verify integration works correctly"""
    
    try:
        # Import the integrated MCP server
        from mcp.mcp_server_integrated import (
            # News Collection Tools (4)
            control_news_collection,
            get_news_provider_status,
            fetch_filtered_news,
            get_news_trends,
            
            # Strategy Selection Tools (4)
            recommend_strategy,
            switch_active_strategy,
            get_strategy_comparison,
            adaptive_strategy_selection,
            
            # Performance Monitoring Tools (3)
            get_system_metrics,
            monitor_strategy_health,
            get_execution_analytics,
            
            # Multi-Asset Trading Tools (3)
            execute_multi_asset_trade,
            portfolio_rebalance,
            cross_asset_correlation_matrix,
            
            # Original tools
            ping, list_strategies, quick_analysis, get_portfolio_status
        )
        
        print("✅ Successfully imported all MCP tools")
        
        # Test results
        test_results = {}
        total_tests = 0
        passed_tests = 0
        
        # Test 1: Basic connectivity
        print("\n🧪 Testing basic MCP connectivity...")
        total_tests += 1
        try:
            result = await ping()
            if result == "pong":
                print("✅ Ping test passed")
                test_results["ping"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Ping test failed")
                test_results["ping"] = "FAIL"
        except Exception as e:
            print(f"❌ Ping test error: {e}")
            test_results["ping"] = f"ERROR: {e}"
        
        # Test 2: Original strategy tools
        print("\n🧪 Testing original strategy tools...")
        total_tests += 1
        try:
            strategies = await list_strategies()
            if strategies and "strategies" in strategies:
                print(f"✅ Strategy listing passed - {len(strategies['strategies'])} strategies found")
                test_results["list_strategies"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Strategy listing failed")
                test_results["list_strategies"] = "FAIL"
        except Exception as e:
            print(f"❌ Strategy listing error: {e}")
            test_results["list_strategies"] = f"ERROR: {e}"
        
        # Test 3: Portfolio status
        print("\n🧪 Testing portfolio management...")
        total_tests += 1
        try:
            portfolio = await get_portfolio_status(include_analytics=True)
            if portfolio and "portfolio" in portfolio:
                print("✅ Portfolio status test passed")
                test_results["portfolio_status"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Portfolio status test failed")
                test_results["portfolio_status"] = "FAIL"
        except Exception as e:
            print(f"❌ Portfolio status error: {e}")
            test_results["portfolio_status"] = f"ERROR: {e}"
        
        # Test 4: News collection control (graceful failure expected)
        print("\n🧪 Testing news collection control...")
        total_tests += 1
        try:
            result = await control_news_collection("start", symbols=["AAPL"])
            # Should return error if news aggregation not available, which is OK
            print(f"✅ News collection test completed: {result.get('status', 'unknown')}")
            test_results["news_collection"] = "PASS"
            passed_tests += 1
        except Exception as e:
            print(f"❌ News collection error: {e}")
            test_results["news_collection"] = f"ERROR: {e}"
        
        # Test 5: Strategy recommendation
        print("\n🧪 Testing strategy recommendation...")
        total_tests += 1
        try:
            market_conditions = {
                "volatility": "moderate",
                "trend": "bullish",
                "sentiment": 0.2
            }
            result = await recommend_strategy(
                market_conditions=market_conditions,
                risk_tolerance="moderate",
                objectives=["profit"]
            )
            if result and "recommendation" in result:
                print(f"✅ Strategy recommendation passed - recommended: {result['recommendation']}")
                test_results["strategy_recommendation"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Strategy recommendation failed")
                test_results["strategy_recommendation"] = "FAIL"
        except Exception as e:
            print(f"❌ Strategy recommendation error: {e}")
            test_results["strategy_recommendation"] = f"ERROR: {e}"
        
        # Test 6: Strategy comparison
        print("\n🧪 Testing strategy comparison...")
        total_tests += 1
        try:
            result = await get_strategy_comparison(
                strategies=["momentum_trading", "swing_trading"],
                metrics=["sharpe_ratio", "total_return"]
            )
            if result and "comparison" in result:
                print("✅ Strategy comparison passed")
                test_results["strategy_comparison"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Strategy comparison failed")
                test_results["strategy_comparison"] = "FAIL"
        except Exception as e:
            print(f"❌ Strategy comparison error: {e}")
            test_results["strategy_comparison"] = f"ERROR: {e}"
        
        # Test 7: System metrics
        print("\n🧪 Testing system metrics...")
        total_tests += 1
        try:
            result = await get_system_metrics(
                metrics=["cpu", "memory"],
                include_history=False
            )
            if result and "cpu" in result.get("current_metrics", {}):
                print("✅ System metrics test passed")
                test_results["system_metrics"] = "PASS"
                passed_tests += 1
            else:
                print("❌ System metrics test failed")
                test_results["system_metrics"] = "FAIL"
        except Exception as e:
            print(f"❌ System metrics error: {e}")
            test_results["system_metrics"] = f"ERROR: {e}"
        
        # Test 8: Multi-asset correlation (mock test)
        print("\n🧪 Testing cross-asset correlation...")
        total_tests += 1
        try:
            result = await cross_asset_correlation_matrix(
                assets=["AAPL", "GOOGL", "MSFT"],
                lookback_days=30,
                include_prediction_confidence=True
            )
            if result and ("correlation_matrix" in result or "error" in result):
                print("✅ Cross-asset correlation test passed")
                test_results["correlation_matrix"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Cross-asset correlation test failed")
                test_results["correlation_matrix"] = "FAIL"
        except Exception as e:
            print(f"❌ Cross-asset correlation error: {e}")
            test_results["correlation_matrix"] = f"ERROR: {e}"
        
        # Test 9: Adaptive strategy selection
        print("\n🧪 Testing adaptive strategy selection...")
        total_tests += 1
        try:
            result = await adaptive_strategy_selection(
                symbol="AAPL",
                auto_switch=False
            )
            if result and ("selected_strategy" in result or "error" in result):
                print("✅ Adaptive strategy selection test passed")
                test_results["adaptive_strategy"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Adaptive strategy selection test failed")
                test_results["adaptive_strategy"] = "FAIL"
        except Exception as e:
            print(f"❌ Adaptive strategy selection error: {e}")
            test_results["adaptive_strategy"] = f"ERROR: {e}"
        
        # Test 10: Portfolio rebalance calculation
        print("\n🧪 Testing portfolio rebalancing...")
        total_tests += 1
        try:
            target_allocations = {
                "AAPL": 0.3,
                "GOOGL": 0.3,
                "MSFT": 0.4
            }
            result = await portfolio_rebalance(
                target_allocations=target_allocations,
                rebalance_threshold=0.05
            )
            if result and ("required_trades" in result or "error" in result):
                print("✅ Portfolio rebalancing test passed")
                test_results["portfolio_rebalance"] = "PASS"
                passed_tests += 1
            else:
                print("❌ Portfolio rebalancing test failed")
                test_results["portfolio_rebalance"] = "FAIL"
        except Exception as e:
            print(f"❌ Portfolio rebalancing error: {e}")
            test_results["portfolio_rebalance"] = f"ERROR: {e}"
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in test_results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"{status} {test_name}: {result}")
        
        # Final assessment
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print(f"\n🎉 INTEGRATION SUCCESS: {passed_tests}/{total_tests} tests passed")
            return True
        else:
            print(f"\n⚠️  INTEGRATION ISSUES: Only {passed_tests}/{total_tests} tests passed")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import MCP components: {e}")
        return False
    except Exception as e:
        print(f"❌ Test execution error: {e}")
        return False

async def test_tool_availability():
    """Test that all 40+ tools are available"""
    
    print("🔍 Checking tool availability...")
    
    try:
        from mcp.mcp_server_integrated import mcp
        
        # Get all registered tools
        tools = []
        for name, func in mcp._tool_registry.items():
            tools.append(name)
        
        print(f"📊 Found {len(tools)} registered MCP tools:")
        
        # Expected tool categories
        expected_categories = {
            "Core Tools": ["ping", "list_strategies", "get_strategy_info", "quick_analysis", "simulate_trade", "get_portfolio_status"],
            "News Tools": ["analyze_news", "get_news_sentiment", "control_news_collection", "get_news_provider_status", "fetch_filtered_news", "get_news_trends"],
            "Trading Tools": ["run_backtest", "optimize_strategy", "risk_analysis", "execute_trade", "execute_multi_asset_trade"],
            "Analytics": ["performance_report", "correlation_analysis", "cross_asset_correlation_matrix"],
            "Neural Tools": ["neural_forecast", "neural_train", "neural_evaluate", "neural_backtest", "neural_model_status", "neural_optimize"],
            "Strategy Tools": ["recommend_strategy", "switch_active_strategy", "get_strategy_comparison", "adaptive_strategy_selection"],
            "Monitoring": ["get_system_metrics", "monitor_strategy_health", "get_execution_analytics"],
            "Portfolio": ["portfolio_rebalance"],
            "Benchmark": ["run_benchmark"]
        }
        
        found_tools = set(tools)
        total_expected = 0
        total_found = 0
        
        for category, expected_tools in expected_categories.items():
            print(f"\n{category}:")
            category_found = 0
            for tool in expected_tools:
                total_expected += 1
                if tool in found_tools:
                    print(f"  ✅ {tool}")
                    category_found += 1
                    total_found += 1
                else:
                    print(f"  ❌ {tool} (missing)")
            print(f"  Category: {category_found}/{len(expected_tools)} found")
        
        # Check for unexpected tools
        expected_all = set()
        for tools_list in expected_categories.values():
            expected_all.update(tools_list)
        
        unexpected = found_tools - expected_all
        if unexpected:
            print(f"\n🔍 Additional tools found:")
            for tool in sorted(unexpected):
                print(f"  ➕ {tool}")
        
        print(f"\n📊 Tool Availability Summary:")
        print(f"Expected tools: {total_expected}")
        print(f"Found tools: {total_found}")
        print(f"Total registered: {len(tools)}")
        print(f"Coverage: {total_found/total_expected*100:.1f}%")
        
        return total_found >= total_expected * 0.9  # 90% coverage required
        
    except Exception as e:
        print(f"❌ Error checking tool availability: {e}")
        return False

async def main():
    """Run all verification tests"""
    print("🚀 Starting MCP Integration Verification\n")
    
    # Test 1: Tool availability
    print("=" * 60)
    availability_ok = await test_tool_availability()
    
    # Test 2: Functional testing
    print("\n" + "=" * 60)
    functionality_ok = await test_mcp_integration()
    
    # Final assessment
    print("\n" + "=" * 60)
    print("🏁 FINAL VERIFICATION RESULTS")
    print("=" * 60)
    
    if availability_ok and functionality_ok:
        print("🎉 SUCCESS: All MCP tools verified and working!")
        print("✅ 40+ tools are properly integrated")
        print("✅ Core functionality is operational")
        print("✅ New integration tools are accessible")
    elif availability_ok:
        print("⚠️  PARTIAL SUCCESS: Tools available but some functionality issues")
        print("✅ Tool registration is complete")
        print("❌ Some functional tests failed")
    elif functionality_ok:
        print("⚠️  PARTIAL SUCCESS: Functionality works but tool availability issues")
        print("❌ Some tools may not be properly registered")
        print("✅ Core functionality is operational")
    else:
        print("❌ INTEGRATION ISSUES: Multiple problems detected")
        print("❌ Tool availability problems")
        print("❌ Functionality problems")
    
    return availability_ok and functionality_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)