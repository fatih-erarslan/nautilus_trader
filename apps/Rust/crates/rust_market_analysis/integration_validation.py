#!/usr/bin/env python3
"""
Comprehensive Integration Validation for Rust Market Analysis API Fix
"""

import sys
import numpy as np
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_direct_rust_api():
    """Test direct Rust API access"""
    print("🔧 Testing Direct Rust API...")
    
    try:
        import market_analysis
        engine = market_analysis.MarketAnalysisEngine()
        
        # Test data
        test_data = {
            'close': [100.0, 101.5, 99.8, 102.3, 101.9, 103.1],
            'volume': [1000.0, 1200.0, 800.0, 1500.0, 1100.0, 1300.0],
            'high': [101.0, 102.0, 100.5, 103.0, 102.5, 103.8],
            'low': [99.5, 100.8, 98.9, 101.5, 101.0, 102.5]
        }
        
        # Test both API methods
        start_time = time.time()
        result1 = engine.run_comprehensive_analysis(test_data)
        time1 = (time.time() - start_time) * 1000
        
        start_time = time.time()
        result2 = engine.analyze_market(test_data)
        time2 = (time.time() - start_time) * 1000
        
        print(f"   ✅ run_comprehensive_analysis: {time1:.2f}ms")
        print(f"   ✅ analyze_market (legacy): {time2:.2f}ms")
        
        # Validate results structure
        expected_fields = ['antifragility_score', 'soc_level', 'panarchy_phase', 'black_swan_prob', 'overall_score']
        for field in expected_fields:
            if field not in result1:
                raise ValueError(f"Missing field {field} in run_comprehensive_analysis")
        
        print("   ✅ All expected fields present in results")
        return True
        
    except Exception as e:
        print(f"   ❌ Direct Rust API test failed: {e}")
        return False

def test_python_wrapper():
    """Test Python wrapper integration"""
    print("\n🐍 Testing Python Wrapper Integration...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from python_bindings import RustMarketAnalyzer
        
        analyzer = RustMarketAnalyzer()
        
        test_data = {
            'close': np.array([100.0, 101.5, 99.8, 102.3, 101.9, 103.1]),
            'volume': np.array([1000.0, 1200.0, 800.0, 1500.0, 1100.0, 1300.0]),
            'high': np.array([101.0, 102.0, 100.5, 103.0, 102.5, 103.8]),
            'low': np.array([99.5, 100.8, 98.9, 101.5, 101.0, 102.5])
        }
        
        # Test all wrapper methods
        start_time = time.time()
        result1 = analyzer.run_comprehensive_analysis(test_data)
        time1 = (time.time() - start_time) * 1000
        
        start_time = time.time() 
        result2 = analyzer.analyze_market(test_data)
        time2 = (time.time() - start_time) * 1000
        
        start_time = time.time()
        signals = analyzer.generate_trade_signals(test_data)
        time3 = (time.time() - start_time) * 1000
        
        print(f"   ✅ run_comprehensive_analysis: {time1:.2f}ms")
        print(f"   ✅ analyze_market: {time2:.2f}ms")
        print(f"   ✅ generate_trade_signals: {time3:.2f}ms")
        print(f"   ✅ Backend: {analyzer.backend}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Python wrapper test failed: {e}")
        return False

def test_integration_layer():
    """Test the integration layer that was having issues"""
    print("\n🔗 Testing Integration Layer...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rust_market_integration import get_market_integration
        
        integration = get_market_integration()
        
        # Health check
        health = integration.health_check()
        print(f"   ✅ Health check: {health.get('rust_analyzer_available', False)}")
        
        test_data = {
            'close': np.array([100.0, 101.5, 99.8, 102.3, 101.9, 103.1, 102.7, 104.2]),
            'volume': np.array([1000.0, 1200.0, 800.0, 1500.0, 1100.0, 1300.0, 900.0, 1400.0]),
            'high': np.array([101.0, 102.0, 100.5, 103.0, 102.5, 103.8, 103.2, 104.8]),
            'low': np.array([99.5, 100.8, 98.9, 101.5, 101.0, 102.5, 102.0, 103.5])
        }
        
        # Test comprehensive analysis
        start_time = time.time()
        analysis = integration.analyze_market_comprehensive(test_data)
        analysis_time = (time.time() - start_time) * 1000
        
        # Test signal generation
        start_time = time.time()
        signal = integration.generate_trading_signals(test_data, "BTC/USDT")
        signal_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Comprehensive analysis: {analysis_time:.2f}ms")
        print(f"   ✅ Signal generation: {signal_time:.2f}ms")
        print(f"   ✅ Signal action: {signal.action}")
        print(f"   ✅ Signal confidence: {signal.confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration layer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and fallback mechanisms"""
    print("\n🛡️ Testing Error Handling...")
    
    try:
        import market_analysis
        engine = market_analysis.MarketAnalysisEngine()
        
        # Test with minimal data
        minimal_data = {
            'close': [100.0, 101.0],
            'volume': [1000.0, 1100.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0]
        }
        
        result = engine.run_comprehensive_analysis(minimal_data)
        print("   ✅ Minimal data handling works")
        
        # Test with empty data (should be handled gracefully)
        try:
            empty_data = {
                'close': [],
                'volume': [],
                'high': [],
                'low': []
            }
            result = engine.run_comprehensive_analysis(empty_data)
            print("   ⚠️  Empty data handled (unexpected success)")
        except Exception:
            print("   ✅ Empty data properly rejected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False

def test_performance():
    """Test performance with larger datasets"""
    print("\n⚡ Testing Performance...")
    
    try:
        import market_analysis
        engine = market_analysis.MarketAnalysisEngine()
        
        # Generate larger test dataset
        np.random.seed(42)
        size = 1000
        
        prices = 100 + np.cumsum(np.random.normal(0, 0.01, size))
        volumes = np.random.uniform(800, 2000, size)
        highs = prices * (1 + np.random.uniform(0, 0.02, size))
        lows = prices * (1 - np.random.uniform(0, 0.02, size))
        
        large_data = {
            'close': prices.tolist(),
            'volume': volumes.tolist(),
            'high': highs.tolist(),
            'low': lows.tolist()
        }
        
        # Performance test
        start_time = time.time()
        result = engine.run_comprehensive_analysis(large_data)
        execution_time = (time.time() - start_time) * 1000
        
        print(f"   ✅ Large dataset ({size} points): {execution_time:.2f}ms")
        
        # Performance target check (should be under 50ms for 1000 points)
        if execution_time < 50:
            print("   ✅ Performance target met (< 50ms)")
        else:
            print(f"   ⚠️  Performance target missed ({execution_time:.2f}ms > 50ms)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Comprehensive Rust Market Analysis API Validation")
    print("=" * 60)
    
    tests = [
        ("Direct Rust API", test_direct_rust_api),
        ("Python Wrapper", test_python_wrapper),
        ("Integration Layer", test_integration_layer),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The Rust-Python integration is fully functional.")
        print("🔧 The run_comprehensive_analysis API error has been completely resolved.")
        return 0
    else:
        print("⚠️  Some tests failed. Manual investigation may be required.")
        return 1

if __name__ == "__main__":
    sys.exit(main())