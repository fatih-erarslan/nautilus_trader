#!/usr/bin/env python3
"""
Test script to verify Canadian trading module imports and basic functionality
"""

import sys
import traceback

def test_imports():
    """Test that all imports work correctly"""
    print("Testing Canadian Trading Module Imports...")
    print("-" * 50)
    
    try:
        # Test main module import
        print("1. Testing main module import...")
        import canadian_trading
        print("   ✅ Main module imported successfully")
        
        # Test specific imports
        print("\n2. Testing specific imports...")
        from canadian_trading import (
            IBCanadaClient,
            ConnectionConfig,
            ConnectionState,
            OrderType,
            PositionInfo,
            OrderInfo
        )
        print("   ✅ All classes imported successfully")
        
        # Test broker module
        print("\n3. Testing broker submodule...")
        from canadian_trading.brokers import ib_canada
        print("   ✅ IB Canada module imported successfully")
        
        # Verify class attributes
        print("\n4. Verifying class structures...")
        
        # ConnectionConfig
        config = ConnectionConfig()
        print(f"   ✅ ConnectionConfig default port: {config.port}")
        print(f"   ✅ ConnectionConfig paper trading: {config.is_paper}")
        
        # ConnectionState enum
        print(f"   ✅ ConnectionState values: {[state.value for state in ConnectionState]}")
        
        # OrderType enum
        print(f"   ✅ OrderType values: {[order.value for order in OrderType]}")
        
        # Version info
        print(f"\n5. Module version: {canadian_trading.__version__}")
        
        print("\n✅ All imports and basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure to install dependencies:")
        print("   pip install -r requirements.txt")
        traceback.print_exc()
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        traceback.print_exc()
        return False


def test_client_creation():
    """Test client instantiation (without connection)"""
    print("\n\nTesting Client Creation...")
    print("-" * 50)
    
    try:
        from canadian_trading import IBCanadaClient, ConnectionConfig
        
        # Create config
        config = ConnectionConfig(
            host="127.0.0.1",
            port=7497,
            client_id=1,
            is_paper=True
        )
        
        # Create client
        client = IBCanadaClient(config)
        print("✅ Client created successfully")
        
        # Test contract creation methods
        shop_contract = client.create_canadian_stock("SHOP", "CAD")
        print(f"✅ Canadian stock contract created: {shop_contract.symbol} on {shop_contract.exchange}")
        
        aapl_contract = client.create_us_stock("AAPL")
        print(f"✅ US stock contract created: {aapl_contract.symbol} on {aapl_contract.exchange}")
        
        # Check event system
        test_handler = lambda data: print(f"Event received: {data}")
        client.on('test_event', test_handler)
        print("✅ Event handler registered")
        
        client.off('test_event', test_handler)
        print("✅ Event handler unregistered")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Client Creation Error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("Canadian Trading Module Test Suite")
    print("=" * 50)
    
    # Add parent directory to path
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Run tests
    import_success = test_imports()
    
    if import_success:
        client_success = test_client_creation()
        
        if client_success:
            print("\n\n✅ All tests passed! The module is ready to use.")
            print("\nNext steps:")
            print("1. Make sure TWS or IB Gateway is running")
            print("2. Enable API connections in TWS/Gateway settings")
            print("3. Run the example script: python example_ib_usage.py")
        else:
            print("\n\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n\n❌ Import tests failed. Please install dependencies first.")
        
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()