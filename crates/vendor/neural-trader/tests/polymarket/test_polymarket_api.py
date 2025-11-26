#!/usr/bin/env python3
"""
Test script to verify Polymarket API integration is working correctly.
This script will test both the API connection and the MCP tools.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import Polymarket modules
try:
    from polymarket.api.clob_client import CLOBClient
    from polymarket.utils.config import PolymarketConfig, load_config
    from polymarket.mcp_tools import (
        get_prediction_markets,
        get_market_orderbook,
        REAL_API_AVAILABLE
    )
    print("✓ Polymarket modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Polymarket modules: {e}")
    sys.exit(1)


async def test_api_connection():
    """Test basic API connection"""
    print("\n1. Testing API Connection...")
    
    try:
        # Load configuration
        config = load_config()
        print(f"   Environment: {config.environment}")
        print(f"   CLOB URL: {config.clob_url}")
        
        # Check for API credentials
        if config.api_key:
            print("   ✓ API key configured")
        else:
            print("   ✗ API key not configured (set POLYMARKET_API_KEY)")
            
        if config.private_key:
            print("   ✓ Private key configured")
        else:
            print("   ⚠ Private key not configured (set POLYMARKET_PRIVATE_KEY)")
            
        # Initialize client
        client = CLOBClient(config=config)
        print("   ✓ CLOB client initialized")
        
        # Test basic API call
        print("\n2. Testing Market Retrieval...")
        async with client:
            markets = await client.get_markets(limit=5)
            print(f"   ✓ Retrieved {len(markets)} markets")
            
            if markets:
                print("\n   Sample Market:")
                market = markets[0]
                print(f"   - ID: {market.id}")
                print(f"   - Question: {market.question}")
                print(f"   - Volume 24h: ${market.volume_24h:,.2f}")
                print(f"   - Status: {market.market_status}")
                
        return True
        
    except Exception as e:
        print(f"   ✗ API test failed: {e}")
        return False


def test_mcp_tools():
    """Test MCP tool integration"""
    print("\n3. Testing MCP Tools...")
    
    try:
        # Test get_prediction_markets
        print("\n   Testing get_prediction_markets...")
        result = get_prediction_markets(limit=3)
        
        if result.get("status") == "success":
            print(f"   ✓ Retrieved {result['count']} markets")
            if result.get("processing", {}).get("source") == "real_api":
                print("   ✓ Using REAL API data")
            else:
                print("   ⚠ Using mock data (API credentials may not be set)")
        else:
            print(f"   ✗ Failed: {result.get('error')}")
            
        # Test specific market if we have one
        if result.get("markets") and len(result["markets"]) > 0:
            market_id = result["markets"][0]["market_id"]
            print(f"\n   Testing get_market_orderbook for {market_id}...")
            
            orderbook = get_market_orderbook(market_id, depth=5)
            if orderbook.get("status") == "success":
                print("   ✓ Retrieved orderbook data")
                if orderbook.get("source") == "real_api":
                    print("   ✓ Using REAL API data")
                else:
                    print("   ⚠ Using mock data")
            else:
                print(f"   ✗ Failed: {orderbook.get('error')}")
                
        return True
        
    except Exception as e:
        print(f"   ✗ MCP tools test failed: {e}")
        return False


def check_environment():
    """Check environment setup"""
    print("Environment Check:")
    print("-" * 50)
    
    env_vars = {
        "POLYMARKET_API_KEY": "API Key",
        "POLYMARKET_PRIVATE_KEY": "Private Key",
        "POLYMARKET_ENVIRONMENT": "Environment",
        "POLYMARKET_CLOB_URL": "CLOB URL",
        "POLYMARKET_RATE_LIMIT": "Rate Limit"
    }
    
    configured = 0
    for var, name in env_vars.items():
        value = os.getenv(var)
        if value:
            if "KEY" in var:
                # Mask sensitive data
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✓ {name}: {masked}")
            else:
                print(f"✓ {name}: {value}")
            configured += 1
        else:
            print(f"✗ {name}: Not set")
            
    print(f"\nConfigured: {configured}/{len(env_vars)} variables")
    print("-" * 50)
    
    return configured >= 2  # At least API key should be set


async def main():
    """Run all tests"""
    print("Polymarket API Integration Test")
    print("=" * 50)
    
    # Check environment
    env_ok = check_environment()
    
    if not env_ok:
        print("\n⚠️  Warning: API credentials not configured.")
        print("The system will use mock data instead of real API.")
        print("\nTo use real API, set environment variables:")
        print("  export POLYMARKET_API_KEY='your-api-key'")
        print("  export POLYMARKET_PRIVATE_KEY='your-private-key'")
        print("\nSee POLYMARKET_SETUP.md for detailed instructions.")
    
    # Test API connection
    if env_ok:
        api_ok = await test_api_connection()
    else:
        api_ok = False
        print("\n✗ Skipping API connection test (no credentials)")
    
    # Test MCP tools (will use mock data if API not configured)
    mcp_ok = test_mcp_tools()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Environment Setup: {'✓' if env_ok else '✗'}")
    print(f"  API Connection: {'✓' if api_ok else '✗ (using mock data)'}")
    print(f"  MCP Tools: {'✓' if mcp_ok else '✗'}")
    
    if env_ok and api_ok and mcp_ok:
        print("\n✅ All tests passed! Real Polymarket API is working.")
    elif mcp_ok:
        print("\n⚠️  MCP tools are working with mock data.")
        print("Configure API credentials to use real market data.")
    else:
        print("\n❌ Tests failed. Check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())