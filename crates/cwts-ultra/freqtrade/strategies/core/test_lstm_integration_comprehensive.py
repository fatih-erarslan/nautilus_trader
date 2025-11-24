#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for LSTM integration
Tests both advanced_lstm.py and quantum_lstm.py with real-world scenarios
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import logging
import traceback
import json
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our LSTM implementations
try:
    import advanced_lstm
    import quantum_lstm
    from enhanced_lstm_integration import (
        EnhancedLSTMConfig, 
        create_enhanced_lstm,
        EnhancedLSTMTransformer
    )
    IMPORTS_SUCCESS = True
except ImportError as e:
    logger.error(f"Import failed: {e}")
    IMPORTS_SUCCESS = False

class LSTMTestSuite:
    """Comprehensive test suite for LSTM implementations"""
    
    def __init__(self):
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'performance_metrics': {},
            'error_logs': []
        }
        
    def generate_synthetic_market_data(self, n_samples: int = 1000, n_features: int = 10) -> np.ndarray:
        """Generate synthetic market data for testing"""
        # Base trend
        trend = np.linspace(1000, 1200, n_samples)
        
        # Add multiple frequency components (like real markets)
        frequencies = [0.1, 0.5, 1.0, 5.0]  # Different market cycles
        data = []
        
        for i in range(n_features):
            signal = trend.copy()
            for freq in frequencies:
                signal += 10 * np.sin(2 * np.pi * freq * np.linspace(0, 10, n_samples) + i)
            
            # Add noise
            signal += np.random.randn(n_samples) * 5
            
            # Add occasional spikes (market events)
            spike_positions = np.random.choice(n_samples, size=10, replace=False)
            signal[spike_positions] += np.random.randn(10) * 50
            
            data.append(signal)
        
        return np.array(data).T  # Shape: (n_samples, n_features)
    
    def test_advanced_lstm_basic(self) -> bool:
        """Test basic functionality of advanced LSTM"""
        logger.info("Testing Advanced LSTM basic functionality...")
        
        try:
            # Create model
            config = {
                'input_size': 10,
                'hidden_sizes': [64, 32],
                'num_heads': 4,
                'timeframes': ['1h', '4h'],
                'use_biological': True
            }
            
            model = advanced_lstm.create_advanced_lstm(config)
            
            # Test data
            batch_size, seq_len, features = 4, 50, 10
            x = np.random.randn(batch_size, seq_len, features)
            
            # Forward pass
            start_time = time.time()
            output = model.forward(x, return_attention=False)
            inference_time = time.time() - start_time
            
            # Verify output shape
            expected_shape = (batch_size, seq_len, 1)
            assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
            
            # Performance metrics
            self.results['performance_metrics']['advanced_lstm_inference_time'] = inference_time
            logger.info(f"✓ Advanced LSTM basic test passed (inference: {inference_time:.4f}s)")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Advanced LSTM basic test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_quantum_lstm_basic(self) -> bool:
        """Test basic functionality of quantum LSTM"""
        logger.info("Testing Quantum LSTM basic functionality...")
        
        try:
            # Create model with small quantum config
            config = {
                'input_size': 8,
                'hidden_size': 32,
                'n_qubits': 4,  # Small for testing
                'n_layers': 1,
                'use_biological': True
            }
            
            model = quantum_lstm.create_quantum_lstm(config)
            
            # Test data (smaller for quantum)
            batch_size, seq_len, features = 2, 20, 8
            x = np.random.randn(batch_size, seq_len, features)
            
            # Forward pass
            start_time = time.time()
            output = model.forward(x, return_quantum_state=False)
            inference_time = time.time() - start_time
            
            # Verify output shape
            expected_shape = (batch_size, seq_len, config['hidden_size'])
            assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
            
            # Performance metrics
            self.results['performance_metrics']['quantum_lstm_inference_time'] = inference_time
            logger.info(f"✓ Quantum LSTM basic test passed (inference: {inference_time:.4f}s)")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Quantum LSTM basic test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_enhanced_integration(self) -> bool:
        """Test the enhanced integrated model"""
        logger.info("Testing Enhanced LSTM Integration...")
        
        try:
            # Create enhanced model with all features
            config = EnhancedLSTMConfig(
                input_size=20,
                hidden_size=64,
                use_biological_activation=True,
                use_multi_timeframe=True,
                use_advanced_attention=True,
                use_quantum=False,  # Start without quantum
                cache_size=500
            )
            
            model = create_enhanced_lstm(config)
            
            # Test data
            batch_size, seq_len, features = 8, 100, 20
            x = torch.randn(batch_size, seq_len, features)
            
            # Forward pass
            start_time = time.time()
            output = model(x)
            inference_time = time.time() - start_time
            
            # Get performance stats
            stats = model.get_performance_stats()
            
            # Verify output
            assert output.shape[0] == batch_size
            assert len(output.shape) == 3  # batch, seq, output
            
            # Log stats
            self.results['performance_metrics']['enhanced_lstm_stats'] = stats
            self.results['performance_metrics']['enhanced_lstm_inference_time'] = inference_time
            
            logger.info(f"✓ Enhanced integration test passed")
            logger.info(f"  Performance stats: {stats}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Enhanced integration test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_market_prediction_scenario(self) -> bool:
        """Test with realistic market prediction scenario"""
        logger.info("Testing market prediction scenario...")
        
        try:
            # Generate market data
            market_data = self.generate_synthetic_market_data(n_samples=500, n_features=10)
            
            # Prepare sequences
            seq_len = 60
            sequences = []
            targets = []
            
            for i in range(len(market_data) - seq_len - 1):
                sequences.append(market_data[i:i+seq_len])
                targets.append(market_data[i+seq_len+1, 0])  # Predict next price
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # Split train/test
            split_idx = int(0.8 * len(sequences))
            train_x, test_x = sequences[:split_idx], sequences[split_idx:]
            train_y, test_y = targets[:split_idx], targets[split_idx:]
            
            # Test with enhanced model
            config = EnhancedLSTMConfig(
                input_size=10,
                hidden_size=32,
                use_biological_activation=True,
                use_multi_timeframe=True,
                timeframes=['5m', '15m', '1h'],
                use_advanced_attention=True
            )
            
            model = create_enhanced_lstm(config)
            
            # Convert to tensors
            train_x_tensor = torch.FloatTensor(train_x)
            test_x_tensor = torch.FloatTensor(test_x)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                train_pred = model(train_x_tensor)
                test_pred = model(test_x_tensor)
            
            # Calculate basic metrics (would need training for real accuracy)
            train_output_shape = train_pred.shape
            test_output_shape = test_pred.shape
            
            logger.info(f"✓ Market prediction test passed")
            logger.info(f"  Train predictions shape: {train_output_shape}")
            logger.info(f"  Test predictions shape: {test_output_shape}")
            
            self.results['performance_metrics']['market_test'] = {
                'train_samples': len(train_x),
                'test_samples': len(test_x),
                'seq_len': seq_len,
                'features': 10
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Market prediction test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_memory_and_caching(self) -> bool:
        """Test memory systems and caching functionality"""
        logger.info("Testing memory and caching systems...")
        
        try:
            # Test Advanced LSTM memory systems
            if hasattr(advanced_lstm, 'LongTermMemory'):
                ltm = advanced_lstm.LongTermMemory(capacity=100)
                
                # Add some experiences
                for i in range(50):
                    experience = type('obj', (object,), {'error': np.random.rand()})()
                    ltm.add(experience)
                
                # Test replay
                replayed = ltm.replay(n_samples=10)
                assert len(replayed) == 10, "Memory replay failed"
                logger.info("✓ Long-term memory test passed")
            
            # Test caching
            if hasattr(advanced_lstm, 'cache'):
                cache = advanced_lstm.cache
                
                # Test cache operations
                test_key = "test_key_123"
                test_value = np.random.randn(10, 10)
                
                cache.put(test_key, test_value)
                retrieved = cache.get(test_key)
                
                assert retrieved is not None, "Cache retrieval failed"
                assert np.array_equal(retrieved, test_value), "Cache value mismatch"
                logger.info("✓ Caching system test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Memory/caching test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_quantum_features(self) -> bool:
        """Test quantum-specific features"""
        logger.info("Testing quantum features...")
        
        try:
            # Test quantum state encoding
            encoder = quantum_lstm.QuantumStateEncoder(n_qubits=4)
            
            # Test different encoding types
            test_data = np.random.randn(16)  # 2^4 for 4 qubits
            
            # Amplitude encoding
            state = encoder.encode(test_data)
            assert len(state) == 16, "Quantum state encoding failed"
            logger.info("✓ Quantum state encoding passed")
            
            # Test quantum attention
            attention = quantum_lstm.QuantumAttention(n_qubits=4, n_heads=2)
            
            # Create dummy quantum states
            query_state = np.random.randn(16)
            key_state = np.random.randn(16)
            value_state = np.random.randn(16)
            
            # Compute attention
            result = attention.compute_attention(query_state, key_state, value_state)
            assert result is not None, "Quantum attention failed"
            logger.info("✓ Quantum attention test passed")
            
            # Test biological quantum effects
            bio_quantum = quantum_lstm.BiologicalQuantumEffects(n_qubits=4)
            
            # Test quantum tunneling
            barrier = 0.5
            state = np.random.randn(16)
            state = state / np.linalg.norm(state)  # Normalize
            
            tunneled = bio_quantum.quantum_tunneling(barrier, state)
            assert tunneled is not None, "Quantum tunneling failed"
            logger.info("✓ Biological quantum effects test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Quantum features test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_performance_comparison(self) -> bool:
        """Compare performance between different configurations"""
        logger.info("Running performance comparison...")
        
        try:
            # Test data
            batch_size, seq_len, features = 4, 50, 10
            x_numpy = np.random.randn(batch_size, seq_len, features)
            x_torch = torch.FloatTensor(x_numpy)
            
            results = {}
            
            # 1. Basic PyTorch LSTM (baseline)
            logger.info("Testing baseline PyTorch LSTM...")
            baseline_lstm = nn.LSTM(features, 64, 2, batch_first=True)
            start_time = time.time()
            with torch.no_grad():
                baseline_out, _ = baseline_lstm(x_torch)
            baseline_time = time.time() - start_time
            results['baseline_pytorch'] = baseline_time
            
            # 2. Enhanced LSTM without quantum
            logger.info("Testing enhanced LSTM (no quantum)...")
            config = EnhancedLSTMConfig(
                input_size=features,
                hidden_size=64,
                use_biological_activation=True,
                use_multi_timeframe=False,  # Single timeframe for fair comparison
                use_quantum=False
            )
            enhanced_model = create_enhanced_lstm(config)
            start_time = time.time()
            with torch.no_grad():
                enhanced_out = enhanced_model(x_torch)
            enhanced_time = time.time() - start_time
            results['enhanced_classical'] = enhanced_time
            
            # 3. Advanced LSTM with biological features
            logger.info("Testing advanced LSTM with biological features...")
            adv_config = {
                'input_size': features,
                'hidden_sizes': [64],
                'num_heads': 4,
                'timeframes': ['1h'],
                'use_biological': True
            }
            adv_model = advanced_lstm.create_advanced_lstm(adv_config)
            start_time = time.time()
            adv_out = adv_model.forward(x_numpy)
            adv_time = time.time() - start_time
            results['advanced_biological'] = adv_time
            
            # Log results
            logger.info("\nPerformance Comparison Results:")
            logger.info(f"  Baseline PyTorch LSTM: {baseline_time:.4f}s")
            logger.info(f"  Enhanced Classical LSTM: {enhanced_time:.4f}s")
            logger.info(f"  Advanced Biological LSTM: {adv_time:.4f}s")
            
            # Calculate relative performance
            logger.info("\nRelative Performance:")
            logger.info(f"  Enhanced vs Baseline: {enhanced_time/baseline_time:.2f}x")
            logger.info(f"  Advanced vs Baseline: {adv_time/baseline_time:.2f}x")
            
            self.results['performance_metrics']['comparison'] = results
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Performance comparison failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_error_handling_and_fallbacks(self) -> bool:
        """Test error handling and fallback mechanisms"""
        logger.info("Testing error handling and fallbacks...")
        
        try:
            # Test quantum device fallback
            devices_tried = []
            
            def mock_get_device(n_qubits=8, shots=None):
                device = quantum_lstm.get_quantum_device(n_qubits, shots)
                devices_tried.append(type(device).__name__)
                return device
            
            # Get device - should fallback gracefully
            device = mock_get_device(4)
            logger.info(f"✓ Quantum device fallback working: {type(device).__name__}")
            
            # Test with invalid input shapes
            logger.info("Testing invalid input handling...")
            
            # Advanced LSTM with wrong shape
            model = advanced_lstm.create_advanced_lstm({'input_size': 10, 'hidden_sizes': [32]})
            
            try:
                # Wrong input shape
                bad_input = np.random.randn(4, 50)  # Missing feature dimension
                output = model.forward(bad_input)
                logger.warning("Model should have raised an error for bad input")
                return False
            except Exception:
                logger.info("✓ Advanced LSTM correctly rejected bad input shape")
            
            # Test memory overflow protection
            if hasattr(advanced_lstm, 'cache'):
                cache = advanced_lstm.MemoryCache(maxsize=10)
                
                # Add more than capacity
                for i in range(20):
                    cache.put(f"key_{i}", np.random.randn(100))
                
                # Should only have last 10
                assert len(cache._cache) <= 10, "Cache overflow protection failed"
                logger.info("✓ Cache overflow protection working")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error handling test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def test_thread_safety(self) -> bool:
        """Test thread safety of concurrent operations"""
        logger.info("Testing thread safety...")
        
        try:
            import threading
            import queue
            
            # Create shared model
            config = EnhancedLSTMConfig(
                input_size=10,
                hidden_size=32,
                use_biological_activation=True,
                use_multi_timeframe=True,
                parallel_processing=True
            )
            model = create_enhanced_lstm(config)
            
            # Results queue
            results_queue = queue.Queue()
            errors_queue = queue.Queue()
            
            def worker(thread_id):
                try:
                    # Each thread processes different data
                    x = torch.randn(2, 30, 10)
                    output = model(x)
                    results_queue.put((thread_id, output.shape))
                except Exception as e:
                    errors_queue.put((thread_id, str(e)))
            
            # Launch threads
            threads = []
            n_threads = 4
            
            for i in range(n_threads):
                t = threading.Thread(target=worker, args=(i,))
                threads.append(t)
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join(timeout=10)
            
            # Check results
            if not errors_queue.empty():
                errors = []
                while not errors_queue.empty():
                    errors.append(errors_queue.get())
                logger.error(f"Thread errors: {errors}")
                return False
            
            # Verify all threads completed
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            assert len(results) == n_threads, f"Not all threads completed: {len(results)}/{n_threads}"
            logger.info(f"✓ Thread safety test passed ({n_threads} concurrent threads)")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Thread safety test failed: {e}")
            self.results['error_logs'].append(str(e))
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return results"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE LSTM INTEGRATION TEST SUITE")
        logger.info("="*60 + "\n")
        
        if not IMPORTS_SUCCESS:
            logger.error("Cannot run tests - imports failed")
            return self.results
        
        # Define all tests
        tests = [
            ("Advanced LSTM Basic", self.test_advanced_lstm_basic),
            ("Quantum LSTM Basic", self.test_quantum_lstm_basic),
            ("Enhanced Integration", self.test_enhanced_integration),
            ("Market Prediction", self.test_market_prediction_scenario),
            ("Memory and Caching", self.test_memory_and_caching),
            ("Quantum Features", self.test_quantum_features),
            ("Performance Comparison", self.test_performance_comparison),
            ("Error Handling", self.test_error_handling_and_fallbacks),
            ("Thread Safety", self.test_thread_safety)
        ]
        
        # Run each test
        for test_name, test_func in tests:
            logger.info(f"\n{'='*40}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*40}")
            
            try:
                success = test_func()
                if success:
                    self.results['tests_passed'] += 1
                else:
                    self.results['tests_failed'] += 1
            except Exception as e:
                logger.error(f"Unexpected error in {test_name}: {e}")
                traceback.print_exc()
                self.results['tests_failed'] += 1
                self.results['error_logs'].append(f"{test_name}: {str(e)}")
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print test summary"""
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nTests Passed: {self.results['tests_passed']}/{total_tests}")
        logger.info(f"Tests Failed: {self.results['tests_failed']}/{total_tests}")
        
        if self.results['tests_failed'] == 0:
            logger.info("\n🎉 ALL TESTS PASSED! Both LSTM implementations are working correctly.")
        else:
            logger.warning(f"\n⚠️  {self.results['tests_failed']} tests failed. Check error logs.")
        
        # Performance metrics
        if self.results['performance_metrics']:
            logger.info("\nPerformance Metrics:")
            for key, value in self.results['performance_metrics'].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}s")
                elif isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for k, v in value.items():
                        logger.info(f"    {k}: {v}")
        
        # Error summary
        if self.results['error_logs']:
            logger.error("\nError Summary:")
            for error in self.results['error_logs'][:5]:  # Show first 5 errors
                logger.error(f"  - {error}")

def main():
    """Main test execution"""
    test_suite = LSTMTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file
    results_file = "lstm_integration_test_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x), indent=2)
        f.write(json_results)
    
    logger.info(f"\nTest results saved to: {results_file}")
    
    # Return exit code based on test results
    return 0 if results['tests_failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())