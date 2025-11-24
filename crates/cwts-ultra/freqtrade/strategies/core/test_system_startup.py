#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test system startup for the Integrated Quantum Trading System
"""

import asyncio
import logging
from datetime import datetime

try:
    from integrated_quantum_trading_system import IntegratedQuantumTradingSystem, SystemConfiguration
    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    print(f"✗ Integrated system not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemStartupTest")

async def test_system_initialization():
    """Test system initialization with mock agents."""
    print("Testing system initialization...")
    
    if not SYSTEM_AVAILABLE:
        print("✗ Cannot test - system not available")
        return False
    
    try:
        # Create configuration with all agents disabled (to test just the framework)
        config = SystemConfiguration()
        config.enable_pads = False  # Disable to avoid missing PADS implementation
        config.enable_qbmia = False  # Disable to avoid missing QBMIA implementation
        config.enable_quasar = False  # Disable to avoid missing QUASAR implementation
        config.enable_quantum_amos = False  # Disable to avoid missing Quantum AMOS implementation
        config.simulate_market_data = False  # Disable simulation
        
        # Create system
        system = IntegratedQuantumTradingSystem(config)
        print("✓ System instance created")
        
        # Test initialization
        success = await system.initialize()
        print(f"✓ System initialization: {success}")
        
        # Test status
        status = system.get_system_status()
        print(f"✓ System status retrieved: {status['running']}")
        
        return success
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_system_with_messaging():
    """Test system with messaging enabled but agents disabled."""
    print("Testing system with messaging...")
    
    if not SYSTEM_AVAILABLE:
        print("✗ Cannot test - system not available")
        return False
    
    try:
        # Create configuration with messaging enabled but agents disabled
        config = SystemConfiguration()
        config.enable_pads = False
        config.enable_qbmia = False
        config.enable_quasar = False
        config.enable_quantum_amos = False
        config.use_real_messaging = True
        config.simulate_market_data = False
        
        # Create system
        system = IntegratedQuantumTradingSystem(config)
        print("✓ System with messaging created")
        
        # Test initialization
        success = await system.initialize()
        print(f"✓ Messaging system initialization: {success}")
        
        return success
        
    except Exception as e:
        print(f"✗ Messaging test failed: {e}")
        return False

async def main():
    print("SYSTEM STARTUP TEST")
    print("=" * 40)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Messaging Integration", test_system_with_messaging)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results[test_name] = result
            print(f"Result: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results[test_name] = False
            print(f"Result: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System startup is working.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

if __name__ == "__main__":
    asyncio.run(main())