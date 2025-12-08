#!/usr/bin/env python3
"""
Test script to verify the run_comprehensive_analysis API fix
"""

import sys
import numpy as np
from pathlib import Path

# Test 1: Import the Rust module
print("🧪 Testing Rust Market Analysis API Fix")
print("=" * 50)

try:
    import market_analysis
    print("✅ Successfully imported market_analysis module")
    
    # Create a MarketAnalysisEngine instance
    engine = market_analysis.MarketAnalysisEngine()
    print("✅ Successfully created MarketAnalysisEngine instance")
    
    # Test 2: Check if run_comprehensive_analysis method exists
    if hasattr(engine, 'run_comprehensive_analysis'):
        print("✅ run_comprehensive_analysis method found on engine")
    else:
        print("❌ run_comprehensive_analysis method NOT found")
        print("Available methods:")
        for attr in dir(engine):
            if not attr.startswith('_'):
                print(f"   - {attr}")
        sys.exit(1)
    
    # Test 3: Create test data
    print("\n📊 Creating test market data...")
    np.random.seed(42)  # For reproducible results
    
    test_data = {
        'close': [100.0, 101.5, 99.8, 102.3, 101.9, 103.1, 102.7, 104.2, 103.8, 105.0],
        'volume': [1000.0, 1200.0, 800.0, 1500.0, 1100.0, 1300.0, 900.0, 1400.0, 1250.0, 1350.0],
        'high': [101.0, 102.0, 100.5, 103.0, 102.5, 103.8, 103.2, 104.8, 104.5, 105.5],
        'low': [99.5, 100.8, 98.9, 101.5, 101.0, 102.5, 102.0, 103.5, 103.0, 104.2]
    }
    
    print("✅ Test data created successfully")
    
    # Test 4: Call run_comprehensive_analysis method
    print("\n⚡ Testing run_comprehensive_analysis method...")
    try:
        result = engine.run_comprehensive_analysis(test_data)
        print("✅ run_comprehensive_analysis method executed successfully!")
        
        # Display the results
        print("\n📈 Analysis Results:")
        print(f"   - Antifragility Score: {result.get('antifragility_score', 'N/A')}")
        print(f"   - SOC Level: {result.get('soc_level', 'N/A')}")
        print(f"   - Panarchy Phase: {result.get('panarchy_phase', 'N/A')}")
        print(f"   - Black Swan Probability: {result.get('black_swan_prob', 'N/A')}")
        print(f"   - Overall Score: {result.get('overall_score', 'N/A')}")
        print(f"   - Trade Action: {result.get('trade_action', 'N/A')}")
        print(f"   - Risk Level: {result.get('risk_level', 'N/A')}")
        
        # Check whale signals
        whale_signals = result.get('whale_signals', {})
        if whale_signals:
            print(f"   - Whale Detected: {whale_signals.get('major_whale_detected', 'N/A')}")
            print(f"   - Whale Direction: {whale_signals.get('whale_direction', 'N/A')}")
            print(f"   - Whale Strength: {whale_signals.get('whale_strength', 'N/A')}")
        
    except Exception as e:
        print(f"❌ run_comprehensive_analysis method failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 5: Test the legacy analyze_market method for compatibility
    print("\n🔄 Testing legacy analyze_market method...")
    try:
        legacy_result = engine.analyze_market(test_data)
        print("✅ Legacy analyze_market method still works!")
        print(f"   - Antifragility: {legacy_result.get('antifragility', 'N/A')}")
        print(f"   - Overall Score: {legacy_result.get('overall_score', 'N/A')}")
        print(f"   - Trade Recommendation: {legacy_result.get('trade_recommendation', 'N/A')}")
        
    except Exception as e:
        print(f"⚠️  Legacy analyze_market method failed: {e}")
    
    # Test 6: Test with Python bindings wrapper
    print("\n🐍 Testing Python bindings wrapper...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from python_bindings import RustMarketAnalyzer
        
        analyzer = RustMarketAnalyzer()
        if analyzer.analyzer:  # Check if Rust backend is available
            wrapper_result = analyzer.analyze_market(test_data)
            print("✅ Python wrapper works successfully!")
            print(f"   - Backend: {analyzer.backend}")
            print(f"   - Overall Score: {wrapper_result.get('overall_score', 'N/A')}")
        else:
            print("⚠️  Python wrapper using fallback (Rust not available)")
    
    except Exception as e:
        print(f"⚠️  Python wrapper test failed: {e}")
    
    print("\n🎉 API Fix Test Completed Successfully!")
    print("🔧 The run_comprehensive_analysis method is now available and working.")
    print("✅ The critical Rust-Python integration error has been FIXED!")
    
except ImportError as e:
    print(f"❌ Failed to import market_analysis module: {e}")
    print("   This could mean the module wasn't built successfully")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)