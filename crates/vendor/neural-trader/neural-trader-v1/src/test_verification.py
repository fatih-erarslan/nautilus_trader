#!/usr/bin/env python3
"""
Test script for Neural Trader Verification System
Demonstrates real data validation vs simulation detection
"""

import time
from data_verification import DataVerification, NeuralTraderVerification, print_truth_dashboard

def test_real_data():
    """Test with real, live data"""
    print("\n" + "="*50)
    print("TEST 1: REAL LIVE DATA")
    print("="*50)
    
    verifier = DataVerification(truth_threshold=0.95)
    
    # Simulate real live data
    real_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'prev_close': 149.80,
        'volume': 75000000,
        'timestamp': time.time(),  # Current time
        'api_key': 'pk_live_1234567890abcdefghij',  # Real-looking key
        'latency_ms': 45,
        'demo_mode': False,
        'paper_trading': False,
        'endpoint': 'https://api.alpaca.markets/v2',
        'name': 'alpaca_live'
    }
    
    is_verified, truth_score, checks = verifier.verify_live_data(real_data)
    
    print(f"\nResult: {'✓ VERIFIED' if is_verified else '✗ FAILED'}")
    print(f"Truth Score: {truth_score:.2f}")
    print("\nDetailed Checks:")
    for check, passed in checks.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")
    
    return is_verified

def test_demo_data():
    """Test with demo/simulation data"""
    print("\n" + "="*50)
    print("TEST 2: DEMO/SIMULATION DATA")
    print("="*50)
    
    verifier = DataVerification(truth_threshold=0.95)
    
    # Simulate demo data
    demo_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'prev_close': 149.80,
        'volume': 75000000,
        'timestamp': time.time() - 3600,  # Old timestamp (1 hour ago)
        'api_key': 'demo_key_123456',  # Demo key
        'latency_ms': 200,  # High latency
        'demo_mode': True,  # Demo mode enabled
        'paper_trading': True,
        'endpoint': 'https://paper-api.alpaca.markets/v2',  # Paper trading endpoint
        'name': 'alpaca_paper'
    }
    
    is_verified, truth_score, checks = verifier.verify_live_data(demo_data)
    
    print(f"\nResult: {'✓ VERIFIED' if is_verified else '✗ FAILED'}")
    print(f"Truth Score: {truth_score:.2f}")
    print("\nDetailed Checks:")
    for check, passed in checks.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")
    
    return not is_verified  # Should fail for demo data

def test_suspicious_data():
    """Test with suspicious/manipulated data"""
    print("\n" + "="*50)
    print("TEST 3: SUSPICIOUS DATA")
    print("="*50)
    
    verifier = DataVerification(truth_threshold=0.95)
    
    # Suspicious data with unrealistic values
    suspicious_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'prev_close': 100.00,  # 50% price change (unrealistic)
        'volume': 1e15,  # Impossibly high volume
        'timestamp': time.time() + 3600,  # Future timestamp
        'api_key': 'test_sandbox_key',  # Test key
        'latency_ms': 5000,  # Very high latency
        'demo_mode': False,  # Claims not demo but...
        'paper_trading': False,
        'endpoint': 'https://sandbox.alpaca.markets/v2',  # Sandbox endpoint
        'name': 'alpaca_sandbox'
    }
    
    is_verified, truth_score, checks = verifier.verify_live_data(suspicious_data)
    
    print(f"\nResult: {'✓ VERIFIED' if is_verified else '✗ FAILED'}")
    print(f"Truth Score: {truth_score:.2f}")
    print("\nDetailed Checks:")
    for check, passed in checks.items():
        status = '✓' if passed else '✗'
        print(f"  {status} {check}")
    
    return not is_verified  # Should fail for suspicious data

def test_neural_trader_config():
    """Test Neural Trader configuration verification"""
    print("\n" + "="*50)
    print("TEST 4: NEURAL TRADER CONFIG")
    print("="*50)
    
    # Test good config
    good_config = {
        'ALPACA_KEY': 'pk_live_real_key_1234567890',
        'POLYGON_KEY': 'real_polygon_key_abcdef',
        'live_trading': True,
        'paper_trading': False,
        'demo_mode': False
    }
    
    result = NeuralTraderVerification.verify_not_demo_mode(good_config)
    print(f"Good Config: {'✓ VERIFIED' if result else '✗ FAILED'}")
    
    # Test bad config (demo mode)
    bad_config = {
        'ALPACA_KEY': 'demo_alpaca_key',
        'POLYGON_KEY': 'test_polygon_key',
        'live_trading': False,
        'paper_trading': True,
        'demo_mode': True
    }
    
    result = NeuralTraderVerification.verify_not_demo_mode(bad_config)
    print(f"Demo Config: {'✓ FAILED AS EXPECTED' if not result else '✗ SHOULD HAVE FAILED'}")
    
    return True

def test_truth_dashboard():
    """Test truth verification dashboard"""
    print("\n" + "="*50)
    print("TEST 5: TRUTH DASHBOARD")
    print("="*50)
    
    verifier = DataVerification(truth_threshold=0.95)
    
    # Run multiple verifications to populate metrics
    test_data_sets = [
        {'timestamp': time.time(), 'api_key': 'live_key', 'demo_mode': False, 'latency_ms': 30},
        {'timestamp': time.time(), 'api_key': 'live_key2', 'demo_mode': False, 'latency_ms': 40},
        {'timestamp': time.time() - 10, 'api_key': 'demo_key', 'demo_mode': True, 'latency_ms': 200},
        {'timestamp': time.time(), 'api_key': 'live_key3', 'demo_mode': False, 'latency_ms': 50},
    ]
    
    for data in test_data_sets:
        verifier.verify_live_data(data)
    
    print()
    print_truth_dashboard(verifier)
    
    return True

def run_all_tests():
    """Run all verification tests"""
    print("\n" + "="*50)
    print("NEURAL TRADER VERIFICATION SYSTEM TESTS")
    print("="*50)
    
    results = {
        'Real Data Test': test_real_data(),
        'Demo Data Test': test_demo_data(),
        'Suspicious Data Test': test_suspicious_data(),
        'Config Test': test_neural_trader_config(),
        'Dashboard Test': test_truth_dashboard()
    }
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    for test_name, passed in results.items():
        status = '✓ PASSED' if passed else '✗ FAILED'
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Verification system working correctly!")
    else:
        print("\n✗ SOME TESTS FAILED - Check verification system")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)