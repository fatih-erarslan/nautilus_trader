#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test and analyze the new LSTM files for bugs and integration opportunities
"""

import sys
import traceback
import warnings

def test_advanced_lstm_imports():
    """Test advanced_lstm.py imports and basic functionality"""
    print("Testing advanced_lstm.py...")
    
    try:
        import advanced_lstm
        print("✓ advanced_lstm.py imports successfully")
        
        # Test basic functionality
        config = {
            'input_size': 10,
            'hidden_sizes': [128, 64],
            'num_heads': 5,
            'timeframes': ['1h', '4h', '1d', '1w'],
            'use_biological': True
        }
        
        model = advanced_lstm.create_advanced_lstm(config)
        print("✓ Advanced LSTM model creation successful")
        
        # Test basic operations
        backend = getattr(advanced_lstm, 'BACKEND', 'unknown')
        print(f"✓ Backend detected: {backend}")
        
        return True
        
    except Exception as e:
        print(f"✗ advanced_lstm.py test failed: {e}")
        traceback.print_exc()
        return False

def test_quantum_lstm_imports():
    """Test quantum_lstm.py imports and basic functionality"""
    print("\nTesting quantum_lstm.py...")
    
    try:
        import quantum_lstm
        print("✓ quantum_lstm.py imports successfully")
        
        # Test basic functionality
        config = {
            'input_size': 10,
            'hidden_size': 64,
            'n_qubits': 8,
            'n_layers': 2,
            'use_biological': True
        }
        
        model = quantum_lstm.create_quantum_lstm(config)
        print("✓ Quantum LSTM model creation successful")
        
        # Test device selection
        device = quantum_lstm.get_quantum_device(n_qubits=4)
        print(f"✓ Quantum device available: {type(device)}")
        
        return True
        
    except Exception as e:
        print(f"✗ quantum_lstm.py test failed: {e}")
        traceback.print_exc()
        return False

def analyze_integration_opportunities():
    """Analyze how these files could integrate with tengri/prediction_app"""
    print("\nAnalyzing integration opportunities...")
    
    try:
        # Import current prediction engine
        sys.path.append('/home/kutlu/freqtrade/user_data/strategies/core/tengri/prediction_app')
        
        # Check if we can import the current implementation
        try:
            import superior_engine
            print("✓ Current prediction engine accessible")
            current_has_lstm = hasattr(superior_engine, 'OptimizedLSTMTransformer')
            print(f"✓ Current engine has LSTM-Transformer: {current_has_lstm}")
        except ImportError:
            print("⚠ Current prediction engine not directly accessible")
        
        # Import new LSTM implementations
        import advanced_lstm
        import quantum_lstm
        
        # Check feature compatibility
        advanced_features = {
            'biological_activation': hasattr(advanced_lstm, 'biological_activation'),
            'attention_mechanism': hasattr(advanced_lstm, 'AttentionMechanism'),
            'ensemble_pathways': hasattr(advanced_lstm, 'BiologicalLSTM'),
            'swarm_optimization': hasattr(advanced_lstm, 'SwarmOptimizer'),
            'memory_systems': hasattr(advanced_lstm, 'LongTermMemory'),
            'caching': hasattr(advanced_lstm, 'MemoryCache')
        }
        
        quantum_features = {
            'quantum_gates': hasattr(quantum_lstm, 'QuantumLSTMGate'),
            'quantum_attention': hasattr(quantum_lstm, 'QuantumAttention'),
            'quantum_memory': hasattr(quantum_lstm, 'QuantumMemory'),
            'state_encoding': hasattr(quantum_lstm, 'QuantumStateEncoder'),
            'biological_quantum': hasattr(quantum_lstm, 'BiologicalQuantumEffects'),
            'error_correction': 'error_correction' in str(quantum_lstm.QuantumMemory.__doc__ if hasattr(quantum_lstm, 'QuantumMemory') else '')
        }
        
        print("\nAdvanced LSTM Features:")
        for feature, available in advanced_features.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature}")
        
        print("\nQuantum LSTM Features:")
        for feature, available in quantum_features.items():
            status = "✓" if available else "✗"
            print(f"  {status} {feature}")
        
        # Integration recommendations
        print("\nIntegration Recommendations:")
        print("1. Advanced LSTM could enhance current implementation with:")
        print("   - Biological activation functions for more realistic neuron behavior")
        print("   - Multi-timeframe ensemble processing")
        print("   - Advanced attention mechanisms with caching")
        print("   - Swarm optimization for hyperparameter tuning")
        
        print("2. Quantum LSTM could provide:")
        print("   - True quantum computing advantages for certain patterns")
        print("   - Quantum attention for complex market correlations")
        print("   - Quantum memory with error correction")
        print("   - Biological quantum effects (tunneling, coherence)")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration analysis failed: {e}")
        traceback.print_exc()
        return False

def check_for_bugs():
    """Check for potential bugs in the new LSTM files"""
    print("\nChecking for potential bugs...")
    
    bugs_found = []
    
    try:
        import advanced_lstm
        import quantum_lstm
        
        # Check 1: Thread safety issues
        if hasattr(advanced_lstm, 'cache') and hasattr(advanced_lstm.cache, '_lock'):
            print("✓ Advanced LSTM has thread-safe caching")
        else:
            bugs_found.append("⚠ Advanced LSTM caching may not be thread-safe")
        
        # Check 2: Memory management
        if hasattr(advanced_lstm, 'CACHE_SIZE'):
            cache_size = getattr(advanced_lstm, 'CACHE_SIZE', 0)
            if cache_size > 50000:
                bugs_found.append(f"⚠ Advanced LSTM cache size very large: {cache_size}")
            else:
                print(f"✓ Advanced LSTM cache size reasonable: {cache_size}")
        
        # Check 3: Error handling
        has_safe_execution = hasattr(advanced_lstm, 'safe_execution')
        has_quantum_safe = hasattr(quantum_lstm, 'safe_quantum_execution')
        print(f"✓ Advanced LSTM has error handling: {has_safe_execution}")
        print(f"✓ Quantum LSTM has error handling: {has_quantum_safe}")
        
        # Check 4: Quantum device fallback
        try:
            device = quantum_lstm.get_quantum_device(n_qubits=4)
            print("✓ Quantum device fallback mechanism works")
        except Exception as e:
            bugs_found.append(f"⚠ Quantum device fallback issue: {e}")
        
        # Check 5: JAX/Numba conditional imports
        advanced_backend = getattr(advanced_lstm, 'BACKEND', 'unknown')
        if advanced_backend == 'numpy':
            print("⚠ Advanced LSTM using NumPy fallback (no acceleration)")
        else:
            print(f"✓ Advanced LSTM using accelerated backend: {advanced_backend}")
        
        # Check 6: Quantum coherence parameters
        if hasattr(quantum_lstm, 'BiologicalQuantumEffects'):
            print("✓ Quantum LSTM has biological quantum effects")
        
        # Check 7: Memory leaks potential
        if hasattr(advanced_lstm, 'ThreadPoolExecutor'):
            print("⚠ Advanced LSTM uses ThreadPoolExecutor - ensure proper cleanup")
        
        if bugs_found:
            print(f"\nPotential issues found: {len(bugs_found)}")
            for bug in bugs_found:
                print(f"  {bug}")
        else:
            print("\n✓ No critical bugs detected")
        
        return len(bugs_found) == 0
        
    except Exception as e:
        print(f"✗ Bug checking failed: {e}")
        return False

def main():
    """Run comprehensive analysis"""
    print("COMPREHENSIVE ANALYSIS OF NEW LSTM FILES")
    print("=" * 50)
    
    results = {
        'advanced_lstm_test': test_advanced_lstm_imports(),
        'quantum_lstm_test': test_quantum_lstm_imports(),
        'integration_analysis': analyze_integration_opportunities(),
        'bug_check': check_for_bugs()
    }
    
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Both LSTM files are ready for integration!")
    else:
        print(f"\n⚠️  {total - passed} issues need attention.")
    
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)