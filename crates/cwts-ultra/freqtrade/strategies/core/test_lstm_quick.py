#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick integration test for LSTM implementations
"""

import sys
import time
import numpy as np
import torch
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Run quick tests on both LSTM implementations"""
    
    print("\n" + "="*60)
    print("QUICK LSTM INTEGRATION TEST")
    print("="*60 + "\n")
    
    results = {"passed": 0, "failed": 0}
    
    # Test 1: Import and basic functionality
    print("1. Testing imports...")
    try:
        import advanced_lstm
        import quantum_lstm
        from enhanced_lstm_integration import create_enhanced_lstm, EnhancedLSTMConfig
        print("✓ All imports successful")
        results["passed"] += 1
    except Exception as e:
        print(f"✗ Import failed: {e}")
        results["failed"] += 1
        return results
    
    # Test 2: Advanced LSTM
    print("\n2. Testing Advanced LSTM...")
    try:
        config = {
            'input_size': 10,
            'hidden_sizes': [32, 16],
            'num_heads': 2,
            'timeframes': ['1h'],  # Single timeframe for speed
            'use_biological': True
        }
        
        model = advanced_lstm.create_advanced_lstm(config)
        x = np.random.randn(2, 20, 10)  # Small batch
        
        start = time.time()
        output = model.forward(x)
        elapsed = time.time() - start
        
        print(f"✓ Advanced LSTM working - Output shape: {output.shape}, Time: {elapsed:.3f}s")
        results["passed"] += 1
        
    except Exception as e:
        print(f"✗ Advanced LSTM failed: {e}")
        results["failed"] += 1
    
    # Test 3: Quantum LSTM (minimal)
    print("\n3. Testing Quantum LSTM...")
    try:
        config = {
            'input_size': 4,
            'hidden_size': 16,
            'n_qubits': 2,  # Very small for speed
            'n_layers': 1,
            'use_biological': False
        }
        
        model = quantum_lstm.create_quantum_lstm(config)
        x = np.random.randn(1, 10, 4)  # Tiny batch
        
        start = time.time()
        output = model.forward(x)
        elapsed = time.time() - start
        
        print(f"✓ Quantum LSTM working - Output shape: {output.shape}, Time: {elapsed:.3f}s")
        results["passed"] += 1
        
    except Exception as e:
        print(f"✗ Quantum LSTM failed: {e}")
        results["failed"] += 1
    
    # Test 4: Enhanced Integration
    print("\n4. Testing Enhanced Integration...")
    try:
        config = EnhancedLSTMConfig(
            input_size=10,
            hidden_size=32,
            use_biological_activation=True,
            use_multi_timeframe=False,  # Single for speed
            use_advanced_attention=True,
            use_quantum=False  # No quantum for speed
        )
        
        model = create_enhanced_lstm(config)
        x = torch.randn(4, 30, 10)
        
        start = time.time()
        output = model(x)
        elapsed = time.time() - start
        
        stats = model.get_performance_stats()
        print(f"✓ Enhanced model working - Output shape: {output.shape}, Time: {elapsed:.3f}s")
        print(f"  Features enabled: {[k for k,v in stats.items() if v and isinstance(v, bool)]}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"✗ Enhanced integration failed: {e}")
        results["failed"] += 1
    
    # Test 5: Market data simulation
    print("\n5. Testing with market-like data...")
    try:
        # Generate simple market data
        prices = 1000 + np.cumsum(np.random.randn(100) * 5)
        volumes = np.abs(np.random.randn(100) * 1000)
        features = np.column_stack([prices, volumes, 
                                   np.roll(prices, 1), np.roll(prices, 5),
                                   np.roll(volumes, 1)])
        
        # Prepare sequences
        seq_len = 20
        sequences = []
        for i in range(len(features) - seq_len):
            sequences.append(features[i:i+seq_len])
        
        sequences = np.array(sequences)
        
        # Test with enhanced model
        x_tensor = torch.FloatTensor(sequences[:10])  # First 10 sequences
        
        model.eval()
        with torch.no_grad():
            predictions = model(x_tensor)
        
        print(f"✓ Market prediction working - Input: {x_tensor.shape}, Output: {predictions.shape}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"✗ Market data test failed: {e}")
        results["failed"] += 1
    
    # Test 6: Memory and performance features
    print("\n6. Testing special features...")
    try:
        # Test biological activation
        if hasattr(advanced_lstm, 'biological_activation'):
            x = np.array([[0.1, 0.6, 1.2]])
            y = advanced_lstm.biological_activation(x, threshold=0.5)
            print(f"✓ Biological activation: Input {x[0]}, Output {y[0]}")
        
        # Test caching
        if hasattr(advanced_lstm, 'cache'):
            cache = advanced_lstm.MemoryCache(maxsize=10)
            cache.put("test", np.array([1, 2, 3]))
            retrieved = cache.get("test")
            print(f"✓ Caching system working: Stored and retrieved {retrieved}")
        
        # Test quantum features
        encoder = quantum_lstm.QuantumStateEncoder(n_qubits=2)
        state = encoder.encode(np.array([1, 2, 3, 4]))
        print(f"✓ Quantum encoding working: State shape {state.shape}")
        
        results["passed"] += 1
        
    except Exception as e:
        print(f"✗ Special features test failed: {e}")
        results["failed"] += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"\nTests Passed: {results['passed']}/6")
    print(f"Tests Failed: {results['failed']}/6")
    
    if results['failed'] == 0:
        print("\n🎉 ALL TESTS PASSED! Both LSTM implementations are ready for use.")
    else:
        print(f"\n⚠️  {results['failed']} tests failed.")
    
    return results

def demonstrate_usage():
    """Show practical usage examples"""
    print("\n" + "="*60)
    print("USAGE DEMONSTRATION")
    print("="*60 + "\n")
    
    from enhanced_lstm_integration import create_enhanced_lstm, EnhancedLSTMConfig
    
    print("Example 1: Classical Enhanced LSTM for Trading")
    print("-" * 40)
    print("""
# Configuration for production use
config = EnhancedLSTMConfig(
    input_size=50,           # OHLCV + indicators
    hidden_size=64,
    use_biological_activation=True,    # Better neuron dynamics
    use_multi_timeframe=True,          # Multiple market cycles
    timeframes=['5m', '15m', '1h', '4h'],
    use_advanced_attention=True,       # Cached attention
    use_quantum=False                  # Start classical
)

# Create model
model = create_enhanced_lstm(config)

# Use in trading loop
for batch in market_data:
    predictions = model(batch)
    # ... trading logic ...
""")
    
    print("\nExample 2: Quantum-Enhanced LSTM (Experimental)")
    print("-" * 40)
    print("""
# Quantum configuration
quantum_config = EnhancedLSTMConfig(
    input_size=16,           # Must be small for quantum
    hidden_size=32,
    use_quantum=True,
    n_qubits=4,              # 16 quantum states
    use_quantum_attention=True,
    use_biological_activation=True
)

# Create quantum model
quantum_model = create_enhanced_lstm(quantum_config)

# Use for pattern detection
complex_patterns = extract_market_patterns(data)
quantum_predictions = quantum_model(complex_patterns)
""")
    
    print("\nExample 3: Integration with Tengri System")
    print("-" * 40)
    print("""
# In superior_engine.py, replace:
# self.lstm_transformer = OptimizedLSTMTransformer(...)

# With:
from enhanced_lstm_integration import create_enhanced_lstm, EnhancedLSTMConfig

config = EnhancedLSTMConfig(
    input_size=self.feature_dim,
    hidden_size=64,
    use_biological_activation=True,
    use_multi_timeframe=True,
    use_advanced_attention=True
)
self.lstm_transformer = create_enhanced_lstm(config)
""")

if __name__ == "__main__":
    # Run quick tests
    results = quick_test()
    
    # Show usage examples
    demonstrate_usage()
    
    # Save test timestamp
    with open("lstm_test_results.txt", "w") as f:
        f.write(f"LSTM Integration Test Results\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Passed: {results['passed']}/6\n")
        f.write(f"Failed: {results['failed']}/6\n")
    
    print(f"\nResults saved to lstm_test_results.txt")
    
    sys.exit(0 if results['failed'] == 0 else 1)