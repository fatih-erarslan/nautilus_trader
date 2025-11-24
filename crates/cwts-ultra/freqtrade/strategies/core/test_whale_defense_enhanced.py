#!/usr/bin/env python3
"""
Test Enhanced Quantum Whale Defense System
Tests multi-GPU support, detection accuracy, and performance
"""

import sys
import os
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced whale defense
from quantum_whale_defense_enhanced import (
    EnhancedQuantumWhaleDefense,
    WhaleDetectionResult,
    DefenseStrategy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhaleDefenseTestSuite:
    """Comprehensive test suite for whale defense system"""
    
    def __init__(self):
        """Initialize test suite"""
        self.defense = EnhancedQuantumWhaleDefense()
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("=== Starting Enhanced Whale Defense Test Suite ===")
        
        # Test 1: GPU Detection and Acceleration
        await self.test_gpu_detection()
        
        # Test 2: Basic Whale Detection
        await self.test_basic_detection()
        
        # Test 3: Multi-GPU Performance
        await self.test_multi_gpu_performance()
        
        # Test 4: Detection Accuracy
        await self.test_detection_accuracy()
        
        # Test 5: Defense Strategy Generation
        await self.test_defense_strategies()
        
        # Test 6: Real-time Performance
        await self.test_realtime_performance()
        
        # Test 7: Stress Test
        await self.test_stress_detection()
        
        # Print summary
        self.print_test_summary()
        
    async def test_gpu_detection(self):
        """Test GPU detection and status"""
        logger.info("\n--- Test 1: GPU Detection ---")
        
        try:
            gpu_status = self.defense.accelerator.get_gpu_status()
            
            logger.info(f"Available GPUs: {gpu_status['available_gpus']}")
            logger.info(f"Primary GPU: {gpu_status['primary_gpu']}")
            logger.info(f"CUDA available: {gpu_status['cuda_available']}")
            logger.info(f"OpenCL available: {gpu_status['opencl_available']}")
            logger.info(f"ROCm available: {gpu_status['rocm_available']}")
            
            # Benchmark GPUs
            logger.info("\nBenchmarking GPUs...")
            benchmark_results = self.defense.accelerator.benchmark_gpus(100000)
            
            for gpu, time_sec in benchmark_results.items():
                throughput = 100000 / time_sec if time_sec > 0 else 0
                logger.info(f"{gpu}: {time_sec:.3f}s ({throughput:.0f} elements/sec)")
                
            self.test_results.append({
                'test': 'GPU Detection',
                'status': 'PASSED',
                'gpus_found': len(gpu_status['available_gpus'])
            })
            
        except Exception as e:
            logger.error(f"GPU detection test failed: {e}")
            self.test_results.append({
                'test': 'GPU Detection',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_basic_detection(self):
        """Test basic whale detection functionality"""
        logger.info("\n--- Test 2: Basic Whale Detection ---")
        
        try:
            # Generate test data with whale pattern
            market_data = self._generate_whale_data('pump', size=1000)
            
            # Run detection
            start_time = time.time()
            result = await self.defense.detect_whale_activity(market_data)
            detection_time = (time.time() - start_time) * 1000
            
            logger.info(f"Detection completed in {detection_time:.2f}ms")
            logger.info(f"Threat Level: {result.threat_level}")
            logger.info(f"Confidence: {result.confidence:.2%}")
            logger.info(f"Whale Size Estimate: ${result.whale_size_estimate:,.2f}")
            logger.info(f"GPU Used: {result.gpu_used}")
            
            # Verify detection
            assert result.threat_level in ['HIGH', 'CRITICAL'], "Failed to detect whale"
            assert result.confidence > 0.7, "Low confidence in detection"
            assert detection_time < 100, "Detection too slow"
            
            self.test_results.append({
                'test': 'Basic Detection',
                'status': 'PASSED',
                'detection_time_ms': detection_time,
                'threat_level': result.threat_level
            })
            
        except Exception as e:
            logger.error(f"Basic detection test failed: {e}")
            self.test_results.append({
                'test': 'Basic Detection',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_multi_gpu_performance(self):
        """Test performance across different GPUs"""
        logger.info("\n--- Test 3: Multi-GPU Performance ---")
        
        try:
            gpu_times = {}
            
            # Test each available GPU
            for gpu in self.defense.accelerator.available_gpus:
                if gpu.gpu_type.value == 'CPU':
                    continue
                    
                logger.info(f"\nTesting {gpu.name}...")
                
                # Force specific GPU
                original_primary = self.defense.accelerator.primary_gpu
                self.defense.accelerator.primary_gpu = gpu
                
                # Generate test data
                market_data = self._generate_whale_data('squeeze', size=5000)
                
                # Run detection
                start_time = time.time()
                result = await self.defense.detect_whale_activity(market_data)
                detection_time = (time.time() - start_time) * 1000
                
                gpu_times[gpu.name] = detection_time
                logger.info(f"{gpu.name}: {detection_time:.2f}ms")
                
                # Restore original GPU
                self.defense.accelerator.primary_gpu = original_primary
                
            # Compare performance
            if gpu_times:
                fastest_gpu = min(gpu_times, key=gpu_times.get)
                slowest_gpu = max(gpu_times, key=gpu_times.get)
                speedup = gpu_times[slowest_gpu] / gpu_times[fastest_gpu]
                
                logger.info(f"\nFastest GPU: {fastest_gpu} ({gpu_times[fastest_gpu]:.2f}ms)")
                logger.info(f"Slowest GPU: {slowest_gpu} ({gpu_times[slowest_gpu]:.2f}ms)")
                logger.info(f"Speedup: {speedup:.2f}x")
                
            self.test_results.append({
                'test': 'Multi-GPU Performance',
                'status': 'PASSED',
                'gpu_times': gpu_times
            })
            
        except Exception as e:
            logger.error(f"Multi-GPU performance test failed: {e}")
            self.test_results.append({
                'test': 'Multi-GPU Performance',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_detection_accuracy(self):
        """Test detection accuracy with various whale patterns"""
        logger.info("\n--- Test 4: Detection Accuracy ---")
        
        patterns = ['pump', 'dump', 'squeeze', 'normal']
        results = {}
        
        try:
            for pattern in patterns:
                logger.info(f"\nTesting {pattern} pattern...")
                
                # Generate data
                market_data = self._generate_whale_data(
                    pattern, 
                    size=1000,
                    whale_size=10_000_000 if pattern != 'normal' else 0
                )
                
                # Run detection
                result = await self.defense.detect_whale_activity(market_data)
                
                # Check accuracy
                if pattern == 'normal':
                    accurate = result.threat_level in ['NONE', 'LOW']
                else:
                    accurate = result.threat_level in ['MEDIUM', 'HIGH', 'CRITICAL']
                    
                results[pattern] = {
                    'threat_level': result.threat_level,
                    'confidence': result.confidence,
                    'accurate': accurate
                }
                
                logger.info(f"Pattern: {pattern}, Detected: {result.threat_level}, "
                          f"Confidence: {result.confidence:.2%}, Accurate: {accurate}")
                
            # Calculate overall accuracy
            accuracy = sum(1 for r in results.values() if r['accurate']) / len(results)
            logger.info(f"\nOverall Accuracy: {accuracy:.2%}")
            
            assert accuracy >= 0.75, f"Accuracy too low: {accuracy:.2%}"
            
            self.test_results.append({
                'test': 'Detection Accuracy',
                'status': 'PASSED',
                'accuracy': accuracy,
                'pattern_results': results
            })
            
        except Exception as e:
            logger.error(f"Detection accuracy test failed: {e}")
            self.test_results.append({
                'test': 'Detection Accuracy',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_defense_strategies(self):
        """Test defense strategy generation"""
        logger.info("\n--- Test 5: Defense Strategy Generation ---")
        
        try:
            # Generate whale attack scenario
            market_data = self._generate_whale_data('dump', size=500, whale_size=20_000_000)
            
            # Detect whale
            detection = await self.defense.detect_whale_activity(market_data)
            
            # Current position
            position = {
                'symbol': 'BTC/USDT',
                'size': 10.0,
                'entry_price': 50000,
                'current_price': 49000  # Price dropping due to dump
            }
            
            # Generate defense strategy
            strategy = self.defense.generate_defense_strategy(detection, position)
            
            logger.info(f"\nDefense Strategy:")
            logger.info(f"Type: {strategy.strategy_type}")
            logger.info(f"Position Adjustment: {strategy.position_adjustment:.1%}")
            logger.info(f"Estimated Impact: {strategy.estimated_impact:.3%}")
            logger.info(f"Confidence: {strategy.confidence:.2%}")
            
            # Verify strategy is appropriate
            assert strategy.strategy_type in ['aggressive', 'balanced'], \
                "Expected aggressive response to dump"
            assert strategy.position_adjustment < 0, \
                "Expected position reduction"
                
            # Log order modifications
            logger.info("\nOrder Modifications:")
            for mod in strategy.order_modifications:
                logger.info(f"  - {mod}")
                
            self.test_results.append({
                'test': 'Defense Strategies',
                'status': 'PASSED',
                'strategy_type': strategy.strategy_type,
                'position_adjustment': strategy.position_adjustment
            })
            
        except Exception as e:
            logger.error(f"Defense strategy test failed: {e}")
            self.test_results.append({
                'test': 'Defense Strategies',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_realtime_performance(self):
        """Test real-time performance requirements"""
        logger.info("\n--- Test 6: Real-time Performance ---")
        
        try:
            detection_times = []
            target_latency = 50  # ms
            
            # Run multiple detections
            for i in range(10):
                market_data = self._generate_whale_data(
                    'pump' if i % 2 == 0 else 'dump',
                    size=500
                )
                
                start_time = time.time()
                result = await self.defense.detect_whale_activity(market_data)
                detection_time = (time.time() - start_time) * 1000
                
                detection_times.append(detection_time)
                
            # Calculate statistics
            avg_time = np.mean(detection_times)
            max_time = np.max(detection_times)
            min_time = np.min(detection_times)
            std_time = np.std(detection_times)
            
            logger.info(f"\nPerformance Statistics:")
            logger.info(f"Average: {avg_time:.2f}ms")
            logger.info(f"Min: {min_time:.2f}ms")
            logger.info(f"Max: {max_time:.2f}ms")
            logger.info(f"Std Dev: {std_time:.2f}ms")
            logger.info(f"Target: <{target_latency}ms")
            
            # Check if meets requirements
            meets_target = avg_time < target_latency
            logger.info(f"Meets target: {'✓' if meets_target else '✗'}")
            
            self.test_results.append({
                'test': 'Real-time Performance',
                'status': 'PASSED' if meets_target else 'WARNING',
                'avg_latency_ms': avg_time,
                'max_latency_ms': max_time
            })
            
        except Exception as e:
            logger.error(f"Real-time performance test failed: {e}")
            self.test_results.append({
                'test': 'Real-time Performance',
                'status': 'FAILED',
                'error': str(e)
            })
            
    async def test_stress_detection(self):
        """Stress test with high-frequency detections"""
        logger.info("\n--- Test 7: Stress Test ---")
        
        try:
            num_detections = 100
            concurrent_detections = 10
            
            logger.info(f"Running {num_detections} detections "
                       f"with {concurrent_detections} concurrent...")
            
            start_time = time.time()
            
            # Create detection tasks
            tasks = []
            for i in range(num_detections):
                market_data = self._generate_whale_data(
                    'pump' if i % 3 == 0 else 'normal',
                    size=200
                )
                task = self.defense.detect_whale_activity(market_data)
                tasks.append(task)
                
                # Run in batches
                if len(tasks) >= concurrent_detections:
                    await asyncio.gather(*tasks)
                    tasks = []
                    
            # Run remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
                
            total_time = time.time() - start_time
            detections_per_second = num_detections / total_time
            
            logger.info(f"\nStress Test Results:")
            logger.info(f"Total detections: {num_detections}")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Detections/second: {detections_per_second:.2f}")
            
            # Get final performance report
            report = self.defense.get_performance_report()
            logger.info(f"System health: {report['system_health']['overall_status']}")
            
            self.test_results.append({
                'test': 'Stress Test',
                'status': 'PASSED',
                'detections_per_second': detections_per_second,
                'system_health': report['system_health']['overall_status']
            })
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            self.test_results.append({
                'test': 'Stress Test',
                'status': 'FAILED',
                'error': str(e)
            })
            
    def _generate_whale_data(self, pattern='pump', size=1000, whale_size=10_000_000):
        """Generate synthetic market data with whale patterns"""
        dates = pd.date_range(end=datetime.now(), periods=size, freq='1min')
        
        # Base price with noise
        base_price = 50000
        prices = base_price + np.cumsum(np.random.randn(size) * 50)
        
        # Add whale pattern
        attack_start = size - 100
        
        if pattern == 'pump':
            # Sudden price pump
            pump_magnitude = whale_size / 1e6 * 0.01  # 1% per million
            prices[attack_start:] += np.linspace(0, base_price * pump_magnitude, 100)
            
            # Add oscillations (whale's footprint)
            frequency = 0.1
            amplitude = base_price * pump_magnitude * 0.1
            oscillations = amplitude * np.sin(frequency * np.arange(100))
            prices[attack_start:] += oscillations
            
        elif pattern == 'dump':
            # Sudden price dump
            dump_magnitude = whale_size / 1e6 * 0.01
            prices[attack_start:] -= np.linspace(0, base_price * dump_magnitude, 100)
            
        elif pattern == 'squeeze':
            # Liquidity squeeze pattern
            squeeze_factor = whale_size / 1e7
            for i in range(attack_start, size):
                prices[i] += np.random.randn() * 200 * squeeze_factor
                
        # Generate volume with whale activity
        volumes = np.random.exponential(1000, size)
        if pattern != 'normal':
            volumes[attack_start:] *= (1 + whale_size / 1e6)  # Increased volume
            
        return pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': volumes,
            'high': prices + np.random.rand(size) * 100,
            'low': prices - np.random.rand(size) * 100
        })
        
    def print_test_summary(self):
        """Print test summary"""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAILED')
        warnings = sum(1 for r in self.test_results if r['status'] == 'WARNING')
        
        for result in self.test_results:
            status_symbol = {
                'PASSED': '✓',
                'FAILED': '✗',
                'WARNING': '⚠'
            }.get(result['status'], '?')
            
            logger.info(f"{status_symbol} {result['test']}: {result['status']}")
            if 'error' in result:
                logger.info(f"  Error: {result['error']}")
                
        logger.info("\n" + "-"*60)
        logger.info(f"Total Tests: {len(self.test_results)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Warnings: {warnings}")
        logger.info(f"Success Rate: {passed/len(self.test_results)*100:.1f}%")
        logger.info("="*60)


async def main():
    """Main test function"""
    test_suite = WhaleDefenseTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())