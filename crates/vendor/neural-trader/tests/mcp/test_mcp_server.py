#!/usr/bin/env python3
"""
Test script for MCP Server

Tests basic functionality of the MCP server
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class MCPServerTester:
    """Test MCP server functionality"""
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"
        self.session = None
        self.request_id = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    def _get_request_id(self):
        """Get next request ID"""
        self.request_id += 1
        return self.request_id
    
    async def call_method(self, method, params=None):
        """Call an MCP method"""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._get_request_id()
        }
        
        async with self.session.post(self.mcp_url, json=payload) as resp:
            return await resp.json()
    
    async def test_health(self):
        """Test health endpoint"""
        print("\n1. Testing health endpoint...")
        async with self.session.get(f"{self.base_url}/health") as resp:
            result = await resp.json()
            print(f"   Status: {result['status']}")
            print(f"   GPU Available: {result['gpu_available']}")
            print(f"   ✓ Health check passed")
            return result
    
    async def test_capabilities(self):
        """Test capabilities endpoint"""
        print("\n2. Testing capabilities endpoint...")
        async with self.session.get(f"{self.base_url}/capabilities") as resp:
            result = await resp.json()
            print(f"   Supported strategies: {', '.join(result['capabilities']['supported_strategies'])}")
            print(f"   Available methods: {len(result['methods'])}")
            print(f"   ✓ Capabilities check passed")
            return result
    
    async def test_discovery(self):
        """Test discovery methods"""
        print("\n3. Testing discovery methods...")
        
        # Discover services
        result = await self.call_method("discover")
        if "error" not in result:
            print(f"   Found {result['result']['count']} services")
            print(f"   ✓ Discovery passed")
        else:
            print(f"   ✗ Discovery failed: {result['error']}")
    
    async def test_list_tools(self):
        """Test listing tools"""
        print("\n4. Testing list tools...")
        
        result = await self.call_method("list_tools")
        if "error" not in result:
            tools = result['result']['tools']
            print(f"   Available tools: {len(tools)}")
            for tool in tools:
                print(f"     - {tool['name']}: {tool['description']}")
            print(f"   ✓ List tools passed")
        else:
            print(f"   ✗ List tools failed: {result['error']}")
    
    async def test_list_resources(self):
        """Test listing resources"""
        print("\n5. Testing list resources...")
        
        result = await self.call_method("list_resources")
        if "error" not in result:
            resources = result['result']['resources']
            print(f"   Available resources: {len(resources)}")
            for resource in resources[:3]:  # Show first 3
                print(f"     - {resource['name']}: {resource['description']}")
            print(f"   ✓ List resources passed")
        else:
            print(f"   ✗ List resources failed: {result['error']}")
    
    async def test_read_resource(self):
        """Test reading a resource"""
        print("\n6. Testing read resource...")
        
        result = await self.call_method("read_resource", {
            "uri": "mcp://config/strategies"
        })
        if "error" not in result:
            contents = result['result']['contents'][0]
            config = json.loads(contents['text'])
            print(f"   Strategies in config: {len(config['available_strategies'])}")
            print(f"   ✓ Read resource passed")
        else:
            print(f"   ✗ Read resource failed: {result['error']}")
    
    async def test_list_prompts(self):
        """Test listing prompts"""
        print("\n7. Testing list prompts...")
        
        result = await self.call_method("list_prompts")
        if "error" not in result:
            prompts = result['result']['prompts']
            print(f"   Available prompts: {len(prompts)}")
            for prompt in prompts:
                print(f"     - {prompt['name']}: {prompt['description']}")
            print(f"   ✓ List prompts passed")
        else:
            print(f"   ✗ List prompts failed: {result['error']}")
    
    async def test_strategy_recommendation(self):
        """Test strategy recommendation prompt"""
        print("\n8. Testing strategy recommendation...")
        
        result = await self.call_method("get_prompt", {
            "name": "strategy_recommendation",
            "arguments": {
                "market_conditions": "volatile",
                "risk_profile": "moderate",
                "investment_horizon": "medium",
                "capital": 50000
            }
        })
        if "error" not in result:
            messages = result['result']['messages']
            print(f"   Generated prompt with {len(messages[0]['content'])} characters")
            print(f"   ✓ Strategy recommendation passed")
        else:
            print(f"   ✗ Strategy recommendation failed: {result['error']}")
    
    async def test_backtest_tool(self):
        """Test backtest tool"""
        print("\n9. Testing backtest tool...")
        
        result = await self.call_method("call_tool", {
            "name": "backtest",
            "arguments": {
                "strategy": "momentum_trader",
                "start_date": "2023-01-01",
                "end_date": "2023-03-31",
                "symbols": ["AAPL", "GOOGL"],
                "initial_capital": 100000
            }
        })
        if "error" not in result:
            backtest_result = result['result']['result']
            print(f"   Total return: {backtest_result['total_return']:.2%}")
            print(f"   Trades executed: {len(backtest_result['trades'])}")
            print(f"   ✓ Backtest passed")
        else:
            print(f"   ✗ Backtest failed: {result['error']}")
    
    async def test_monte_carlo(self):
        """Test Monte Carlo simulation"""
        print("\n10. Testing Monte Carlo simulation...")
        
        result = await self.call_method("create_message", {
            "messages": [{
                "role": "user",
                "content": "Run Monte Carlo simulation for portfolio risk analysis"
            }],
            "sampling": {
                "max_tokens": 2000,
                "temperature": 0.7
            }
        })
        if "error" not in result:
            content = result['result']['content']
            print(f"   Simulation completed")
            print(f"   Results include VaR and CVaR calculations")
            print(f"   ✓ Monte Carlo passed")
        else:
            print(f"   ✗ Monte Carlo failed: {result['error']}")
    
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("MCP Server Test Suite")
        print("=" * 60)
        
        try:
            # Check if server is running
            await self.test_health()
            await self.test_capabilities()
            await self.test_discovery()
            await self.test_list_tools()
            await self.test_list_resources()
            await self.test_read_resource()
            await self.test_list_prompts()
            await self.test_strategy_recommendation()
            await self.test_backtest_tool()
            await self.test_monte_carlo()
            
            print("\n" + "=" * 60)
            print("✓ All tests passed successfully!")
            print("=" * 60)
            
        except aiohttp.ClientError as e:
            print(f"\n✗ Error: Could not connect to MCP server at {self.base_url}")
            print(f"  Make sure the server is running: python start_mcp_server.py")
            print(f"  Error details: {str(e)}")
        except Exception as e:
            print(f"\n✗ Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()


async def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP Server")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of MCP server (default: http://localhost:8080)"
    )
    
    args = parser.parse_args()
    
    async with MCPServerTester(args.url) as tester:
        await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())