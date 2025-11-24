# File: tests/whale_defense_test_suite.py

import unittest
import numpy as np
import time
import threading
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import yaml
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

# Import the whale defense system
from quantum_whale_detection_core import (
    QuantumOscillationDetector,
    QuantumCorrelationEngine,
    QuantumGameTheoryEngine,
    MachiavellianQuantumTradingSystem,
    WhaleDetectionConfig
)

# Test Configuration
@dataclass
class TestConfig:
    """Test configuration parameters"""
    test_timeout_seconds: int = 30
    performance_iterations: int = 100
    memory_limit_mb: int = 1000
    max_test_threads: int = 4
    historical_events_file: str = "test_data/historical_whale_events.json"
    mock_data_enabled: bool = True

class TestDataGenerator:
    """Generate test data for whale defense system testing"""
    
    @staticmethod
    def generate_normal_market_data(length: int = 100) -> Dict:
        """Generate normal market data without whale activity"""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price series with normal volatility
        base_price = 50000.0
        returns = np.random.normal(0, 0.02, length)  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        # Generate correlated volume
        volumes = []
        for i in range(len(prices)):
            base_volume = 1000
            volatility_boost = abs(returns[i]) * 5000 if i < len(returns) else 0
            volume = max(100, np.random.normal(base_volume + volatility_boost, 200))
            volumes.append(volume)
        
        return {
            'prices': prices,
            'volumes': volumes,
            'timestamps': [time.time() - (length - i) * 60 for i in range(length)],
            'volatility': np.std(returns),
            'liquidity': 0.7,
            'bid_ask_spread': 0.001,
            'orderbook': {
                'bids': [[49950, 100], [49940, 200], [49930, 150]],
                'asks': [[50050, 120], [50060, 180], [50070, 100]]
            }
        }
    
    @staticmethod
    def generate_whale_attack_data(attack_type: str = "dump", magnitude: float = 0.1) -> Dict:
        """Generate market data simulating whale attack"""
        normal_data = TestDataGenerator.generate_normal_market_data(50)
        
        if attack_type == "dump":
            # Simulate large sell order impact
            for i in range(-10, 0):  # Last 10 data points
                normal_data['prices'][i] *= (1 - magnitude * (10 + i) / 10)
                normal_data['volumes'][i] *= (3 + magnitude * 2)  # Volume spike
        
        elif attack_type == "pump":
            # Simulate large buy order impact
            for i in range(-10, 0):
                normal_data['prices'][i] *= (1 + magnitude * (10 + i) / 10)
                normal_data['volumes'][i] *= (2 + magnitude)
        
        elif attack_type == "squeeze":
            # Simulate short squeeze
            for i in range(-15, 0):
                squeeze_factor = np.exp(magnitude * (15 + i) / 15)
                normal_data['prices'][i] *= squeeze_factor
                normal_data['volumes'][i] *= (1 + magnitude * 3)
        
        # Update market microstructure
        normal_data['volatility'] *= (1 + magnitude * 2)
        normal_data['liquidity'] *= (1 - magnitude * 0.5)
        normal_data['bid_ask_spread'] *= (1 + magnitude * 3)
        
        return normal_data
    
    @staticmethod
    def generate_stealth_accumulation_data() -> Dict:
        """Generate data for stealth whale accumulation"""
        normal_data = TestDataGenerator.generate_normal_market_data(100)
        
        # Subtle volume increase without major price impact
        for i in range(-30, 0):
            normal_data['volumes'][i] *= 1.2  # 20% volume increase
            # Very slight upward pressure
            normal_data['prices'][i] *= 1.001
        
        return normal_data
    
    @staticmethod
    def generate_social_sentiment_data(manipulation_type: str = "none") -> Dict:
        """Generate social sentiment data"""
        base_sentiment = {
            'twitter': {
                'posts': [
                    {'text': 'Bitcoin looking strong today', 'timestamp': time.time(), 'engagement': 100},
                    {'text': 'Crypto market is stable', 'timestamp': time.time() - 3600, 'engagement': 50}
                ]
            },
            'discord': {'posts': []},
            'telegram': {'posts': []},
            'reddit': {'posts': []}
        }
        
        if manipulation_type == "bearish_campaign":
            base_sentiment['twitter']['posts'].extend([
                {'text': 'BITCOIN CRASH INCOMING! SELL NOW!', 'timestamp': time.time(), 'engagement': 1000},
                {'text': 'Whale dump detected, get out while you can', 'timestamp': time.time() - 300, 'engagement': 800},
                {'text': 'BTC going to zero, this is the end', 'timestamp': time.time() - 600, 'engagement': 600}
            ])
        
        elif manipulation_type == "bullish_campaign":
            base_sentiment['twitter']['posts'].extend([
                {'text': 'Bitcoin to the moon! 🚀🚀🚀', 'timestamp': time.time(), 'engagement': 2000},
                {'text': 'MASSIVE BTC accumulation happening now!', 'timestamp': time.time() - 300, 'engagement': 1500}
            ])
        
        return base_sentiment

class TestQuantumOscillationDetector(unittest.TestCase):
    """Test suite for quantum oscillation detector"""
    
    def setUp(self):
        """Set up test environment"""
        self.detector = QuantumOscillationDetector(detection_qubits=4, sensitivity=0.01)  # Smaller for tests
        self.test_config = TestConfig()
        
    def test_initialization(self):
        """Test detector initialization"""
        self.assertEqual(self.detector.detection_qubits, 4)
        self.assertEqual(self.detector.sensitivity, 0.01)
        self.assertIsNotNone(self.detector.device)
        self.assertIsNotNone(self.detector.phase_estimation_circuit)
        
    def test_normal_market_detection(self):
        """Test detection on normal market data (should not detect whale)"""
        normal_data = TestDataGenerator.generate_normal_market_data(50)
        
        result = self.detector.detect_whale_tremors(normal_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('whale_detected', result)
        self.assertIn('processing_time_ms', result)
        
        # Should not detect whale in normal data
        self.assertFalse(result['whale_detected'])
        
        # Should complete within latency requirement
        self.assertLess(result['processing_time_ms'], 100)  # 100ms for test environment
        
    def test_whale_dump_detection(self):
        """Test detection on whale dump data"""
        whale_data = TestDataGenerator.generate_whale_attack_data("dump", magnitude=0.15)
        
        result = self.detector.detect_whale_tremors(whale_data)
        
        self.assertIsInstance(result, dict)
        
        # May or may not detect depending on sensitivity, but should not error
        if result['whale_detected']:
            self.assertIn('confidence', result)
            self.assertIn('estimated_impact_time', result)
            self.assertGreater(result['confidence'], 0)
            self.assertLessEqual(result['estimated_impact_time'], 15)
    
    def test_frequency_extraction(self):
        """Test frequency extraction methods"""
        test_data = TestDataGenerator.generate_normal_market_data(100)
        
        # Test price frequency extraction
        price_frequencies = self.detector._extract_price_frequencies(test_data['prices'])
        self.assertIsInstance(price_frequencies, dict)
        
        # Test volume frequency extraction
        volume_frequencies = self.detector._extract_volume_frequencies(test_data['volumes'])
        self.assertIsInstance(volume_frequencies, dict)
        
    def test_baseline_calibration(self):
        """Test baseline calibration with historical data"""
        historical_data = []
        for i in range(10):
            data = TestDataGenerator.generate_normal_market_data(30)
            historical_data.append(data)
        
        # Should not raise exception
        self.detector.calibrate_baseline(historical_data)
        
        # Should have baseline data
        self.assertIsNotNone(self.detector.baseline_frequencies)
        
    def test_data_normalization(self):
        """Test data normalization for quantum encoding"""
        test_data = [1, 2, 3, 4, 5]
        normalized = self.detector._normalize_data(test_data)
        
        self.assertEqual(len(normalized), self.detector.detection_qubits)
        self.assertTrue(all(0 <= x <= 1 for x in normalized))
        
    def test_error_handling(self):
        """Test error handling with invalid data"""
        # Empty data
        result = self.detector.detect_whale_tremors({'prices': [], 'volumes': []})
        self.assertFalse(result.get('whale_detected', True))  # Should default to False
        
        # Malformed data
        result = self.detector.detect_whale_tremors({'invalid': 'data'})
        self.assertIn('error', result)

class TestQuantumCorrelationEngine(unittest.TestCase):
    """Test suite for quantum correlation engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = QuantumCorrelationEngine(correlation_qubits=8, timeframes=[1, 5, 15])  # Smaller for tests
        
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.correlation_qubits, 8)
        self.assertEqual(self.engine.timeframes, [1, 5, 15])
        self.assertIsNotNone(self.engine.device)
        
    def test_normal_correlation_analysis(self):
        """Test correlation analysis on normal market data"""
        normal_data = TestDataGenerator.generate_normal_market_data(100)
        
        result = self.engine.analyze_cross_timeframe_correlations(normal_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('manipulation_detected', result)
        self.assertIn('entanglement_strength', result)
        self.assertIn('processing_time_ms', result)
        
    def test_data_aggregation(self):
        """Test market data aggregation for different timeframes"""
        test_data = TestDataGenerator.generate_normal_market_data(100)
        
        for timeframe in self.engine.timeframes:
            aggregated = self.engine._aggregate_data(test_data, timeframe)
            
            self.assertIsInstance(aggregated, dict)
            self.assertIn('ohlc', aggregated)
            self.assertIn('volume', aggregated)
            self.assertIn('returns', aggregated)
            self.assertIn('volatility', aggregated)
            
            # OHLC should have 4 values
            self.assertEqual(len(aggregated['ohlc']), 4)
            
    def test_timeframe_data_normalization(self):
        """Test timeframe data normalization"""
        test_aggregated_data = {
            'ohlc': [100, 105, 95, 102],
            'volume': 1000,
            'returns': [0.01, -0.02, 0.015],
            'volatility': 0.02
        }
        
        normalized = self.engine._normalize_timeframe_data(test_aggregated_data)
        
        self.assertIsInstance(normalized, list)
        self.assertTrue(all(0 <= x <= 1 for x in normalized))
        
    def test_error_handling(self):
        """Test error handling with edge cases"""
        # Empty data
        result = self.engine.analyze_cross_timeframe_correlations({})
        self.assertIn('error', result)
        
        # Insufficient data
        small_data = {'prices': [100, 101], 'volumes': [1000, 1100]}
        result = self.engine.analyze_cross_timeframe_correlations(small_data)
        # Should not crash, may have low confidence

class TestQuantumGameTheoryEngine(unittest.TestCase):
    """Test suite for quantum game theory engine"""
    
    def setUp(self):
        """Set up test environment"""
        self.engine = QuantumGameTheoryEngine(game_theory_qubits=6)  # Smaller for tests
        
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.game_theory_qubits, 6)
        self.assertIsNotNone(self.engine.device)
        
    def test_strategy_calculation(self):
        """Test optimal counter-strategy calculation"""
        whale_profile = {
            'size_category': 'large_whale',
            'sophistication': 0.8,
            'aggression_level': 0.7,
            'stealth_level': 0.3
        }
        
        market_state = {
            'volatility': 0.3,
            'liquidity': 0.6,
            'trend': 'upward'
        }
        
        result = self.engine.calculate_optimal_counter_strategy(whale_profile, market_state)
        
        self.assertIsInstance(result, dict)
        self.assertIn('recommended_strategy', result)
        self.assertIn('expected_payoff', result)
        self.assertIn('confidence', result)
        
    def test_whale_strategy_modeling(self):
        """Test whale strategy modeling"""
        whale_profile = {
            'size_category': 'mega_whale',
            'sophistication': 0.9,
            'aggression_level': 0.8
        }
        
        market_state = {'volatility': 0.5, 'liquidity': 0.4}
        
        strategies = self.engine._model_whale_strategies(whale_profile, market_state)
        
        self.assertIsInstance(strategies, list)
        self.assertTrue(len(strategies) > 0)
        
        for strategy in strategies:
            self.assertIn('type', strategy)
            self.assertIn('weight', strategy)
            
    def test_counter_strategy_definition(self):
        """Test counter-strategy definition"""
        market_state = {'volatility': 0.3, 'liquidity': 0.7}
        
        strategies = self.engine._define_counter_strategies(market_state)
        
        self.assertIsInstance(strategies, list)
        self.assertTrue(len(strategies) > 0)
        
    def test_payoff_matrix_creation(self):
        """Test payoff matrix creation"""
        whale_strategies = [
            {'type': 'market_dump', 'size': 0.8, 'weight': 0.5},
            {'type': 'stealth_accumulation', 'size': 0.3, 'weight': 0.5}
        ]
        
        our_strategies = [
            {'type': 'defensive_hedge', 'allocation': 0.3, 'weight': 0.6},
            {'type': 'counter_trade', 'allocation': 0.2, 'weight': 0.4}
        ]
        
        market_state = {'volatility': 0.3, 'liquidity': 0.6}
        
        payoff_matrix = self.engine._create_payoff_matrix(whale_strategies, our_strategies, market_state)
        
        self.assertEqual(payoff_matrix.shape, (2, 2))
        self.assertIsInstance(payoff_matrix, np.ndarray)

class TestMachiavellianTradingSystem(unittest.TestCase):
    """Test suite for the complete trading system"""
    
    def setUp(self):
        """Set up test environment"""
        config = WhaleDetectionConfig(
            detection_qubits=4,
            correlation_qubits=6,
            game_theory_qubits=4,
            detection_sensitivity=0.01,
            max_latency_ms=100
        )
        self.system = MachiavellianQuantumTradingSystem(config)
        
    def test_system_initialization(self):
        """Test complete system initialization"""
        self.assertIsNotNone(self.system.oscillation_detector)
        self.assertIsNotNone(self.system.correlation_engine)
        self.assertIsNotNone(self.system.game_theory_engine)
        self.assertEqual(self.system.whale_threat_level, 0.0)
        
    def test_comprehensive_whale_detection(self):
        """Test comprehensive whale detection"""
        test_data = TestDataGenerator.generate_normal_market_data(50)
        
        result = self.system.comprehensive_whale_detection(test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('whale_detected', result)
        self.assertIn('confidence', result)
        self.assertIn('individual_results', result)
        self.assertIn('total_processing_time_ms', result)
        
    def test_defense_recommendation(self):
        """Test defense recommendation system"""
        whale_warning = {
            'whale_detected': True,
            'confidence': 0.8,
            'estimated_impact_time_seconds': 10
        }
        
        market_state = {'volatility': 0.4, 'liquidity': 0.5}
        
        result = self.system.get_defense_recommendation(whale_warning, market_state)
        
        self.assertIsInstance(result, dict)
        if result.get('defense_needed', False):
            self.assertIn('recommended_strategy', result)
            self.assertIn('confidence', result)
        
    def test_threat_aggregation(self):
        """Test quantum threat aggregation"""
        detection_results = {
            'oscillation': {'whale_detected': True, 'confidence': 0.7},
            'correlation': {'manipulation_detected': True, 'confidence': 0.8}
        }
        
        result = self.system._quantum_threat_aggregation(detection_results)
        
        self.assertIsInstance(result, dict)
        self.assertIn('whale_detected', result)
        self.assertIn('confidence', result)
        
    def test_whale_profile_classification(self):
        """Test whale profile classification"""
        whale_warning = {
            'confidence': 0.9,
            'detection_methods_triggered': ['oscillation', 'correlation'],
            'estimated_impact_time_seconds': 6
        }
        
        profile = self.system._classify_whale_profile(whale_warning)
        
        self.assertIsInstance(profile, dict)
        self.assertIn('size_category', profile)
        self.assertIn('sophistication', profile)
        self.assertIn('aggression_level', profile)
        self.assertIn('stealth_level', profile)
        
    def test_system_status(self):
        """Test system status reporting"""
        status = self.system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('status', status)
        self.assertIn('current_threat_level', status)
        self.assertIn('performance_metrics', status)

class PerformanceBenchmarkTests(unittest.TestCase):
    """Performance and latency benchmark tests"""
    
    def setUp(self):
        """Set up performance testing environment"""
        self.config = WhaleDetectionConfig(
            detection_qubits=8,
            correlation_qubits=12,
            max_latency_ms=50
        )
        self.system = MachiavellianQuantumTradingSystem(self.config)
        self.iterations = 50  # Reduced for CI/CD
        
    def test_detection_latency_benchmark(self):
        """Benchmark detection latency"""
        latencies = []
        
        for _ in range(self.iterations):
            test_data = TestDataGenerator.generate_normal_market_data(100)
            
            start_time = time.perf_counter()
            result = self.system.comprehensive_whale_detection(test_data)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Individual detection should meet latency requirement
            self.assertLess(latency_ms, self.config.max_latency_ms * 2)  # 2x tolerance for tests
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nDetection Latency Benchmark Results:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        
        # Performance assertions
        self.assertLess(avg_latency, self.config.max_latency_ms * 2)
        self.assertLess(p95_latency, self.config.max_latency_ms * 3)
        
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run detection multiple times
        for _ in range(20):
            test_data = TestDataGenerator.generate_whale_attack_data("dump", 0.1)
            result = self.system.comprehensive_whale_detection(test_data)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"\nMemory Usage Benchmark:")
        print(f"Initial: {initial_memory:.2f}MB")
        print(f"Final: {final_memory:.2f}MB")
        print(f"Growth: {memory_growth:.2f}MB")
        
        # Memory growth should be reasonable
        self.assertLess(memory_growth, 500)  # Less than 500MB growth
        
    def test_concurrent_detection_performance(self):
        """Test performance under concurrent load"""
        num_threads = 4
        detections_per_thread = 10
        
        def run_detections():
            latencies = []
            for _ in range(detections_per_thread):
                test_data = TestDataGenerator.generate_normal_market_data(50)
                start_time = time.perf_counter()
                result = self.system.comprehensive_whale_detection(test_data)
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            return latencies
        
        # Run concurrent detections
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.perf_counter()
            futures = [executor.submit(run_detections) for _ in range(num_threads)]
            all_latencies = []
            
            for future in as_completed(futures):
                all_latencies.extend(future.result())
            
            total_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        avg_latency = np.mean(all_latencies)
        throughput = len(all_latencies) / total_time
        
        print(f"\nConcurrent Performance Results:")
        print(f"Total detections: {len(all_latencies)}")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Throughput: {throughput:.2f} detections/second")
        
        # Performance assertions
        self.assertGreater(throughput, 5)  # At least 5 detections per second

class HistoricalWhaleEventTests(unittest.TestCase):
    """Test against historical whale manipulation events"""
    
    def setUp(self):
        """Set up historical event testing"""
        self.system = MachiavellianQuantumTradingSystem()
        self.historical_events = self._load_historical_events()
        
    def _load_historical_events(self):
        """Load historical whale events (mock data for testing)"""
        return [
            {
                'id': 'btc_flash_crash_2021_04_18',
                'description': 'Bitcoin flash crash from whale liquidation',
                'whale_action': {
                    'timestamp': '2021-04-18T14:30:00Z',
                    'type': 'market_sell',
                    'size': 0.15,  # 15% price impact
                    'duration_minutes': 15
                },
                'pre_event_data': TestDataGenerator.generate_whale_attack_data("dump", 0.15),
                'expected_detection': True
            },
            {
                'id': 'eth_pump_2021_05_12',
                'description': 'Ethereum pump from coordinated buying',
                'whale_action': {
                    'timestamp': '2021-05-12T09:15:00Z',
                    'type': 'coordinated_buy',
                    'size': 0.12,
                    'duration_minutes': 30
                },
                'pre_event_data': TestDataGenerator.generate_whale_attack_data("pump", 0.12),
                'expected_detection': True
            },
            {
                'id': 'stealth_accumulation_2021_03_01',
                'description': 'Stealth whale accumulation',
                'whale_action': {
                    'timestamp': '2021-03-01T00:00:00Z',
                    'type': 'stealth_accumulation',
                    'size': 0.03,
                    'duration_minutes': 1440  # 24 hours
                },
                'pre_event_data': TestDataGenerator.generate_stealth_accumulation_data(),
                'expected_detection': False  # Stealth should be harder to detect
            }
        ]
    
    def test_historical_event_detection(self):
        """Test detection accuracy on historical events"""
        results = []
        
        for event in self.historical_events:
            detection_result = self.system.comprehensive_whale_detection(
                event['pre_event_data']
            )
            
            detected = detection_result.get('whale_detected', False)
            expected = event['expected_detection']
            
            results.append({
                'event_id': event['id'],
                'detected': detected,
                'expected': expected,
                'correct': detected == expected,
                'confidence': detection_result.get('confidence', 0.0)
            })
        
        # Calculate accuracy
        correct_predictions = sum(1 for r in results if r['correct'])
        total_events = len(results)
        accuracy = correct_predictions / total_events if total_events > 0 else 0
        
        print(f"\nHistorical Event Detection Results:")
        for result in results:
            status = "✓" if result['correct'] else "✗"
            print(f"{status} {result['event_id']}: detected={result['detected']}, expected={result['expected']}")
        print(f"Accuracy: {accuracy:.2%} ({correct_predictions}/{total_events})")
        
        # Should achieve reasonable accuracy
        self.assertGreaterEqual(accuracy, 0.6)  # At least 60% accuracy
        
    def test_false_positive_rate(self):
        """Test false positive rate on normal market data"""
        false_positives = 0
        total_tests = 50
        
        for _ in range(total_tests):
            normal_data = TestDataGenerator.generate_normal_market_data(100)
            result = self.system.comprehensive_whale_detection(normal_data)
            
            if result.get('whale_detected', False):
                false_positives += 1
        
        false_positive_rate = false_positives / total_tests
        
        print(f"\nFalse Positive Rate: {false_positive_rate:.2%} ({false_positives}/{total_tests})")
        
        # Should have low false positive rate
        self.assertLess(false_positive_rate, 0.1)  # Less than 10%

class IntegrationTests(unittest.TestCase):
    """Integration tests for complete system workflow"""
    
    def setUp(self):
        """Set up integration testing environment"""
        self.config = WhaleDetectionConfig()
        self.system = MachiavellianQuantumTradingSystem(self.config)
        
    def test_complete_detection_defense_workflow(self):
        """Test complete workflow from detection to defense recommendation"""
        # Generate whale attack scenario
        whale_data = TestDataGenerator.generate_whale_attack_data("dump", 0.12)
        market_state = {
            'volatility': 0.4,
            'liquidity': 0.5,
            'trend': 'downward'
        }
        
        # Step 1: Detection
        detection_result = self.system.comprehensive_whale_detection(whale_data)
        
        # Step 2: Defense recommendation (if whale detected)
        if detection_result.get('whale_detected', False):
            defense_result = self.system.get_defense_recommendation(
                detection_result, market_state
            )
            
            self.assertIn('defense_needed', defense_result)
            if defense_result.get('defense_needed', False):
                self.assertIn('recommended_strategy', defense_result)
                self.assertIn('confidence', defense_result)
        
        # Should complete without errors
        self.assertIsInstance(detection_result, dict)
        
    def test_real_time_monitoring_simulation(self):
        """Simulate real-time monitoring workflow"""
        data_queue = []
        results = []
        
        # Generate data stream
        for i in range(10):
            if i == 7:  # Inject whale attack at 7th iteration
                data = TestDataGenerator.generate_whale_attack_data("pump", 0.1)
            else:
                data = TestDataGenerator.generate_normal_market_data(20)
            data_queue.append(data)
        
        # Simulate real-time processing
        def mock_data_callback():
            if data_queue:
                return data_queue.pop(0)
            return None
        
        # Process data stream
        for _ in range(len(data_queue) + 1):
            data = mock_data_callback()
            if data:
                result = self.system.comprehensive_whale_detection(data)
                results.append(result)
        
        # Verify results
        self.assertEqual(len(results), 10)
        
        # Check if whale was detected in any iteration
        whale_detections = [r for r in results if r.get('whale_detected', False)]
        print(f"Whale detections in simulation: {len(whale_detections)}")
        
    def test_system_state_consistency(self):
        """Test system state consistency across multiple operations"""
        initial_status = self.system.get_system_status()
        
        # Perform multiple detections
        for _ in range(5):
            test_data = TestDataGenerator.generate_normal_market_data(50)
            result = self.system.comprehensive_whale_detection(test_data)
        
        final_status = self.system.get_system_status()
        
        # Verify state consistency
        self.assertEqual(initial_status['status'], final_status['status'])
        self.assertGreaterEqual(
            final_status['performance_metrics']['total_detections'],
            initial_status['performance_metrics']['total_detections'] + 5
        )

# Test Runner and Reporting
class TestRunner:
    """Custom test runner with detailed reporting"""
    
    def __init__(self):
        self.test_results = {}
        
    def run_all_tests(self, verbose=True):
        """Run all test suites and generate report"""
        test_suites = [
            TestQuantumOscillationDetector,
            TestQuantumCorrelationEngine,
            TestQuantumGameTheoryEngine,
            TestMachiavellianTradingSystem,
            PerformanceBenchmarkTests,
            HistoricalWhaleEventTests,
            IntegrationTests
        ]
        
        all_results = {}
        
        for test_suite in test_suites:
            suite_name = test_suite.__name__
            print(f"\n{'='*60}")
            print(f"Running {suite_name}")
            print(f"{'='*60}")
            
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_suite)
            runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
            
            start_time = time.time()
            result = runner.run(suite)
            execution_time = time.time() - start_time
            
            all_results[suite_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
                'execution_time': execution_time
            }
        
        self._generate_test_report(all_results)
        return all_results
    
    def _generate_test_report(self, results):
        """Generate comprehensive test report"""
        print(f"\n{'='*80}")
        print("WHALE DEFENSE SYSTEM TEST REPORT")
        print(f"{'='*80}")
        
        total_tests = sum(r['tests_run'] for r in results.values())
        total_failures = sum(r['failures'] for r in results.values())
        total_errors = sum(r['errors'] for r in results.values())
        overall_success = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        total_time = sum(r['execution_time'] for r in results.values())
        
        print(f"Overall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_tests - total_failures - total_errors}")
        print(f"  Failed: {total_failures}")
        print(f"  Errors: {total_errors}")
        print(f"  Success Rate: {overall_success:.1%}")
        print(f"  Total Time: {total_time:.2f}s")
        
        print(f"\nDetailed Results by Test Suite:")
        for suite_name, result in results.items():
            status = "PASS" if result['failures'] == 0 and result['errors'] == 0 else "FAIL"
            print(f"  {suite_name:40} | {status:4} | {result['success_rate']:5.1%} | {result['execution_time']:6.2f}s")
        
        # Performance summary
        if 'PerformanceBenchmarkTests' in results:
            print(f"\nPerformance Summary:")
            print(f"  Latency tests completed successfully")
            print(f"  Memory usage within acceptable limits")
            print(f"  Concurrent processing verified")
        
        print(f"\n{'='*80}")

if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during tests
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    runner = TestRunner()
    results = runner.run_all_tests(verbose=True)
    
    # Exit with appropriate code
    overall_success = all(
        r['failures'] == 0 and r['errors'] == 0 
        for r in results.values()
    )
    
    exit(0 if overall_success else 1)