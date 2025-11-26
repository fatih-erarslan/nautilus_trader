"""
Data quality regression test suite for AI News Trading benchmark system.

This module tests that data feed quality remains consistent and validates data integrity.
Tests include:
- Data feed quality validation
- Data consistency across sources
- Data latency regression detection
- Data completeness validation
- Data format consistency
"""

import asyncio
import time
import statistics
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.data.data_validator import DataQualityValidator
from benchmark.src.data.feeds import *
from benchmark.src.analysis.data_quality_analyzer import DataQualityAnalyzer
from benchmark.src.profiling.profiler import DataQualityProfiler


class TestDataFeedQualityValidation:
    """Test data feed quality validation and regression detection."""
    
    @pytest.fixture
    async def data_quality_system(self):
        """Create system for data quality testing."""
        config = {
            'data_quality_monitoring': True,
            'quality_thresholds': {
                'completeness': 0.98,
                'timeliness': 0.95,
                'accuracy': 0.99,
                'consistency': 0.97
            },
            'regression_detection': True,
            'anomaly_detection': True
        }
        
        system = {
            'data_manager': RealtimeManager(config),
            'quality_validator': DataQualityValidator(config),
            'quality_analyzer': DataQualityAnalyzer(config),
            'profiler': DataQualityProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.fixture
    def sample_market_data_feed(self):
        """Create sample market data with quality issues for testing."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        data_points = []
        
        base_time = time.time()
        
        for i in range(1000):
            symbol = symbols[i % len(symbols)]
            timestamp = base_time + i * 1  # 1 second intervals
            
            # Introduce various quality issues
            data_point = {
                'symbol': symbol,
                'timestamp': timestamp,
                'price': 100 + np.random.normal(0, 5),
                'volume': max(1, int(np.random.normal(1000, 200))),
                'bid': 100 + np.random.normal(0, 5),
                'ask': 100 + np.random.normal(0, 5) + 0.01
            }
            
            # Introduce specific quality issues
            if i % 100 == 0:  # Missing data (1% rate)
                continue
            elif i % 50 == 0:  # Stale data (duplicate timestamp)
                data_point['timestamp'] = timestamp - 1
            elif i % 75 == 0:  # Invalid price
                data_point['price'] = -1
            elif i % 60 == 0:  # Zero volume
                data_point['volume'] = 0
            elif i % 80 == 0:  # Bid > Ask (invalid spread)
                data_point['bid'] = data_point['ask'] + 0.05
            elif i % 90 == 0:  # Late data (timestamp in future)
                data_point['timestamp'] = timestamp + 300  # 5 minutes in future
            
            data_points.append(data_point)
        
        return data_points
    
    @pytest.mark.asyncio
    async def test_data_completeness_validation(self, data_quality_system, sample_market_data_feed):
        """Test data completeness validation and regression detection."""
        data_manager = data_quality_system['data_manager']
        quality_validator = data_quality_system['quality_validator']
        profiler = data_quality_system['profiler']
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        with profiler.measure('data_completeness_validation'):
            # Process sample data feed
            processed_count = 0
            validation_results = []
            
            for data_point in sample_market_data_feed:
                # Validate data point
                validation_result = await quality_validator.validate_data_point(data_point)
                validation_results.append(validation_result)
                
                # Process if valid
                if validation_result['valid']:
                    await data_manager.process_market_update(data_point)
                    processed_count += 1
        
        # Analyze data completeness
        completeness_analysis = await quality_validator.analyze_data_completeness(
            expected_symbols=symbols,
            time_window=1000,  # 1000 seconds
            expected_frequency=1  # 1 second intervals
        )
        
        # Validate completeness metrics
        assert 'overall_completeness' in completeness_analysis
        assert 'symbol_completeness' in completeness_analysis
        assert 'temporal_completeness' in completeness_analysis
        
        overall_completeness = completeness_analysis['overall_completeness']
        
        # Should detect missing data but maintain reasonable completeness
        assert 0.85 < overall_completeness < 1.0, \
            f"Overall completeness {overall_completeness:.3f} outside expected range"
        
        # Check per-symbol completeness
        symbol_completeness = completeness_analysis['symbol_completeness']
        for symbol in symbols:
            symbol_comp = symbol_completeness.get(symbol, 0)
            assert symbol_comp > 0.8, f"Symbol {symbol} completeness {symbol_comp:.3f} < 0.8"
        
        # Check temporal gaps
        temporal_analysis = completeness_analysis['temporal_completeness']
        assert 'gap_count' in temporal_analysis
        assert 'max_gap_seconds' in temporal_analysis
        assert 'avg_gap_seconds' in temporal_analysis
        
        # Validate data quality issues were detected
        invalid_data_count = sum(1 for result in validation_results if not result['valid'])
        expected_invalid_rate = 0.1  # Expect ~10% invalid data from our sample
        actual_invalid_rate = invalid_data_count / len(validation_results)
        
        assert 0.05 < actual_invalid_rate < 0.2, \
            f"Invalid data rate {actual_invalid_rate:.3f} outside expected range"
        
        print(f"\nData Completeness Validation:")
        print(f"  Overall completeness: {overall_completeness:.3f}")
        print(f"  Invalid data rate: {actual_invalid_rate:.3f}")
        print(f"  Temporal gaps: {temporal_analysis['gap_count']}")
    
    @pytest.mark.asyncio
    async def test_data_timeliness_validation(self, data_quality_system):
        """Test data timeliness validation and latency regression detection."""
        data_manager = data_quality_system['data_manager']
        quality_validator = data_quality_system['quality_validator']
        
        # Simulate real-time data feed with various latency characteristics
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        latency_measurements = []
        timeliness_violations = []
        
        current_time = time.time()
        
        for i in range(500):
            symbol = symbols[i % len(symbols)]
            
            # Simulate different latency scenarios
            if i < 100:  # Normal latency (0-50ms)
                latency = np.random.uniform(0, 0.05)
            elif i < 200:  # Moderate latency (50-100ms)
                latency = np.random.uniform(0.05, 0.1)
            elif i < 300:  # High latency (100-500ms)
                latency = np.random.uniform(0.1, 0.5)
            elif i < 400:  # Very high latency (500ms-2s)
                latency = np.random.uniform(0.5, 2.0)
            else:  # Extreme latency (2s+)
                latency = np.random.uniform(2.0, 5.0)
            
            # Create data point with simulated latency
            data_timestamp = current_time + i * 0.1
            processing_timestamp = data_timestamp + latency
            
            data_point = {
                'symbol': symbol,
                'price': 100 + np.random.normal(0, 2),
                'volume': np.random.randint(100, 2000),
                'timestamp': data_timestamp,
                'processing_timestamp': processing_timestamp
            }
            
            # Validate timeliness
            timeliness_result = await quality_validator.validate_data_timeliness(
                data_point,
                max_latency_ms=100  # 100ms threshold
            )
            
            latency_ms = latency * 1000
            latency_measurements.append(latency_ms)
            
            if not timeliness_result['timely']:
                timeliness_violations.append({
                    'symbol': symbol,
                    'latency_ms': latency_ms,
                    'threshold_ms': 100,
                    'violation_severity': timeliness_result.get('severity', 'unknown')
                })
            
            # Process data
            await data_manager.process_market_update(data_point)
        
        # Analyze timeliness performance
        timeliness_analysis = await quality_validator.analyze_timeliness_performance(
            latency_measurements,
            thresholds={'acceptable': 50, 'warning': 100, 'critical': 500}
        )
        
        # Validate timeliness analysis
        assert 'avg_latency_ms' in timeliness_analysis
        assert 'p95_latency_ms' in timeliness_analysis
        assert 'p99_latency_ms' in timeliness_analysis
        assert 'violation_rate' in timeliness_analysis
        
        avg_latency = timeliness_analysis['avg_latency_ms']
        p95_latency = timeliness_analysis['p95_latency_ms']
        p99_latency = timeliness_analysis['p99_latency_ms']
        violation_rate = timeliness_analysis['violation_rate']
        
        # Should detect latency regression
        assert violation_rate > 0.5, f"Violation rate {violation_rate:.3f} should be > 0.5 with simulated high latency"
        assert p99_latency > 1000, f"P99 latency {p99_latency:.1f}ms should be > 1000ms with simulated delays"
        
        # Check violation severity distribution
        severity_counts = {}
        for violation in timeliness_violations:
            severity = violation['violation_severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        print(f"\nData Timeliness Validation:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  P95 latency: {p95_latency:.1f}ms")
        print(f"  P99 latency: {p99_latency:.1f}ms")
        print(f"  Violation rate: {violation_rate:.3f}")
        print(f"  Violations by severity: {severity_counts}")
    
    @pytest.mark.asyncio
    async def test_data_accuracy_validation(self, data_quality_system):
        """Test data accuracy validation and anomaly detection."""
        quality_validator = data_quality_system['quality_validator']
        quality_analyzer = data_quality_system['quality_analyzer']
        
        # Create dataset with accuracy issues
        symbols = ['AAPL', 'GOOGL']
        test_data = []
        
        for i in range(200):
            symbol = symbols[i % len(symbols)]
            base_price = 150 if symbol == 'AAPL' else 2800
            
            # Normal data
            price = base_price + np.random.normal(0, base_price * 0.01)
            volume = max(1, int(np.random.normal(1000, 200)))
            
            # Introduce accuracy issues
            if i % 25 == 0:  # Price spike (10x normal)
                price = base_price * 10
            elif i % 30 == 0:  # Price crash (10% of normal)
                price = base_price * 0.1
            elif i % 35 == 0:  # Volume spike (100x normal)
                volume = volume * 100
            elif i % 40 == 0:  # Impossible price (negative)
                price = -price
            
            data_point = {
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'timestamp': time.time() + i,
                'bid': price - 0.01,
                'ask': price + 0.01
            }
            
            test_data.append(data_point)
        
        # Validate data accuracy
        accuracy_results = []
        
        for data_point in test_data:
            accuracy_result = await quality_validator.validate_data_accuracy(
                data_point,
                historical_context=test_data[:10],  # Use first 10 as baseline
                deviation_threshold=3.0  # 3 standard deviations
            )
            
            accuracy_results.append(accuracy_result)
        
        # Analyze accuracy issues
        accuracy_analysis = await quality_analyzer.analyze_accuracy_patterns(
            accuracy_results,
            data_points=test_data
        )
        
        # Validate accuracy detection
        assert 'overall_accuracy' in accuracy_analysis
        assert 'anomaly_count' in accuracy_analysis
        assert 'anomaly_types' in accuracy_analysis
        
        overall_accuracy = accuracy_analysis['overall_accuracy']
        anomaly_count = accuracy_analysis['anomaly_count']
        anomaly_types = accuracy_analysis['anomaly_types']
        
        # Should detect anomalies but maintain reasonable overall accuracy
        assert 0.7 < overall_accuracy < 0.95, \
            f"Overall accuracy {overall_accuracy:.3f} outside expected range"
        
        assert anomaly_count > 10, f"Should detect at least 10 anomalies, found {anomaly_count}"
        
        # Check anomaly type detection
        expected_anomaly_types = ['price_spike', 'price_crash', 'volume_spike', 'negative_value']
        detected_types = set(anomaly_types.keys())
        
        # Should detect at least some expected anomaly types
        common_types = detected_types.intersection(expected_anomaly_types)
        assert len(common_types) > 0, f"Should detect some expected anomaly types, got {detected_types}"
        
        print(f"\nData Accuracy Validation:")
        print(f"  Overall accuracy: {overall_accuracy:.3f}")
        print(f"  Anomalies detected: {anomaly_count}")
        print(f"  Anomaly types: {list(anomaly_types.keys())}")


class TestDataConsistencyAcrossSources:
    """Test data consistency across multiple data sources."""
    
    @pytest.fixture
    async def multi_source_system(self):
        """Create system with multiple data sources for consistency testing."""
        config = {
            'multi_source_mode': True,
            'consistency_checking': True,
            'source_arbitration': True,
            'cross_validation': True
        }
        
        # Mock multiple data sources
        sources = ['source_a', 'source_b', 'source_c']
        
        system = {
            'data_manager': RealtimeManager(config),
            'quality_analyzer': DataQualityAnalyzer(config),
            'sources': sources
        }
        
        await system['data_manager'].initialize()
        await system['quality_analyzer'].initialize()
        
        yield system
        
        await system['data_manager'].shutdown()
        await system['quality_analyzer'].shutdown()
    
    @pytest.mark.asyncio
    async def test_cross_source_price_consistency(self, multi_source_system):
        """Test price consistency across multiple data sources."""
        data_manager = multi_source_system['data_manager']
        quality_analyzer = multi_source_system['quality_analyzer']
        sources = multi_source_system['sources']
        
        symbol = 'AAPL'
        base_price = 150.0
        
        # Generate data from multiple sources with slight variations
        source_data = {}
        
        for source in sources:
            source_data[source] = []
            
            for i in range(100):
                # Add source-specific bias and noise
                source_bias = {'source_a': 0, 'source_b': 0.02, 'source_c': -0.01}[source]
                source_noise = {'source_a': 0.01, 'source_b': 0.015, 'source_c': 0.008}[source]
                
                price = base_price + source_bias + np.random.normal(0, source_noise)
                
                # Occasional larger deviations
                if i % 20 == 0:
                    price += np.random.normal(0, 0.1)  # Larger deviation
                
                data_point = {
                    'symbol': symbol,
                    'price': price,
                    'volume': np.random.randint(500, 2000),
                    'timestamp': time.time() + i,
                    'source': source
                }
                
                source_data[source].append(data_point)
        
        # Process data from all sources
        for source in sources:
            for data_point in source_data[source]:
                await data_manager.process_market_update(data_point)
        
        # Analyze cross-source consistency
        consistency_analysis = await quality_analyzer.analyze_cross_source_consistency(
            symbol=symbol,
            sources=sources,
            time_window=100,
            consistency_threshold=0.02  # 2% price difference threshold
        )
        
        # Validate consistency analysis
        assert 'overall_consistency' in consistency_analysis
        assert 'source_pair_consistency' in consistency_analysis
        assert 'consistency_violations' in consistency_analysis
        assert 'source_reliability_scores' in consistency_analysis
        
        overall_consistency = consistency_analysis['overall_consistency']
        source_reliability = consistency_analysis['source_reliability_scores']
        violations = consistency_analysis['consistency_violations']
        
        # Should show reasonable consistency despite variations
        assert 0.8 < overall_consistency < 1.0, \
            f"Overall consistency {overall_consistency:.3f} outside expected range"
        
        # All sources should have reliability scores
        for source in sources:
            assert source in source_reliability, f"Missing reliability score for {source}"
            reliability = source_reliability[source]
            assert 0.5 < reliability < 1.0, f"Source {source} reliability {reliability:.3f} outside range"
        
        # Should detect some violations due to introduced noise
        assert len(violations) > 0, "Should detect some consistency violations"
        
        print(f"\nCross-Source Price Consistency:")
        print(f"  Overall consistency: {overall_consistency:.3f}")
        print(f"  Consistency violations: {len(violations)}")
        for source, reliability in source_reliability.items():
            print(f"  {source} reliability: {reliability:.3f}")
    
    @pytest.mark.asyncio
    async def test_source_arbitration_effectiveness(self, multi_source_system):
        """Test effectiveness of source arbitration for conflicting data."""
        data_manager = multi_source_system['data_manager']
        quality_analyzer = multi_source_system['quality_analyzer']
        sources = multi_source_system['sources']
        
        symbol = 'GOOGL'
        base_price = 2800.0
        
        # Create conflicting data scenarios
        conflict_scenarios = [
            {
                'name': 'price_divergence',
                'source_a': base_price,
                'source_b': base_price * 1.05,  # 5% higher
                'source_c': base_price * 0.98   # 2% lower
            },
            {
                'name': 'stale_data',
                'source_a': base_price,
                'source_b': base_price,  # Current
                'source_c': base_price * 0.95  # Stale (old price)
            },
            {
                'name': 'outlier_data',
                'source_a': base_price,
                'source_b': base_price,
                'source_c': base_price * 2.0  # Obvious outlier
            }
        ]
        
        arbitration_results = []
        
        for scenario in conflict_scenarios:
            scenario_data = []
            
            # Create conflicting data points
            timestamp = time.time()
            for source in sources:
                price = scenario[source]
                
                data_point = {
                    'symbol': symbol,
                    'price': price,
                    'volume': 1000,
                    'timestamp': timestamp,
                    'source': source
                }
                
                scenario_data.append(data_point)
                await data_manager.process_market_update(data_point)
            
            # Perform source arbitration
            arbitration_result = await quality_analyzer.arbitrate_conflicting_sources(
                symbol=symbol,
                conflicting_data=scenario_data,
                arbitration_method='weighted_consensus'
            )
            
            arbitration_results.append({
                'scenario': scenario['name'],
                'result': arbitration_result,
                'input_prices': [scenario[source] for source in sources]
            })
        
        # Validate arbitration effectiveness
        for result in arbitration_results:
            arbitration = result['result']
            scenario_name = result['scenario']
            
            assert 'consensus_value' in arbitration
            assert 'confidence_score' in arbitration
            assert 'excluded_sources' in arbitration
            assert 'arbitration_method' in arbitration
            
            consensus_price = arbitration['consensus_value']
            confidence = arbitration['confidence_score']
            excluded = arbitration['excluded_sources']
            
            # Consensus should be reasonable
            input_prices = result['input_prices']
            min_price = min(input_prices)
            max_price = max(input_prices)
            
            assert min_price <= consensus_price <= max_price, \
                f"Consensus price {consensus_price} outside input range [{min_price}, {max_price}]"
            
            # Outlier scenario should exclude the outlier source
            if scenario_name == 'outlier_data':
                assert len(excluded) > 0, f"Should exclude outlier source in {scenario_name}"
                assert confidence > 0.8, f"Confidence {confidence:.3f} should be high for outlier exclusion"
            
            print(f"  {scenario_name}: consensus ${consensus_price:.2f}, confidence {confidence:.3f}")


class TestDataFormatConsistency:
    """Test data format consistency and schema validation."""
    
    @pytest.mark.asyncio
    async def test_data_schema_validation(self):
        """Test data schema validation and format consistency."""
        config = {
            'schema_validation': True,
            'format_consistency': True,
            'type_checking': True
        }
        
        quality_validator = DataQualityValidator(config)
        await quality_validator.initialize()
        
        # Define expected schema
        expected_schema = {
            'symbol': {'type': 'string', 'required': True, 'max_length': 10},
            'price': {'type': 'float', 'required': True, 'min': 0},
            'volume': {'type': 'integer', 'required': True, 'min': 0},
            'timestamp': {'type': 'float', 'required': True},
            'bid': {'type': 'float', 'required': False, 'min': 0},
            'ask': {'type': 'float', 'required': False, 'min': 0}
        }
        
        await quality_validator.set_schema('market_data', expected_schema)
        
        # Test data with various schema violations
        test_cases = [
            # Valid data
            {
                'name': 'valid_data',
                'data': {
                    'symbol': 'AAPL',
                    'price': 150.25,
                    'volume': 1000,
                    'timestamp': time.time(),
                    'bid': 150.24,
                    'ask': 150.26
                },
                'should_pass': True
            },
            # Missing required field
            {
                'name': 'missing_symbol',
                'data': {
                    'price': 150.25,
                    'volume': 1000,
                    'timestamp': time.time()
                },
                'should_pass': False
            },
            # Wrong type
            {
                'name': 'wrong_price_type',
                'data': {
                    'symbol': 'AAPL',
                    'price': 'not_a_number',
                    'volume': 1000,
                    'timestamp': time.time()
                },
                'should_pass': False
            },
            # Value out of range
            {
                'name': 'negative_price',
                'data': {
                    'symbol': 'AAPL',
                    'price': -150.25,
                    'volume': 1000,
                    'timestamp': time.time()
                },
                'should_pass': False
            },
            # Symbol too long
            {
                'name': 'long_symbol',
                'data': {
                    'symbol': 'VERYLONGSYMBOL',
                    'price': 150.25,
                    'volume': 1000,
                    'timestamp': time.time()
                },
                'should_pass': False
            }
        ]
        
        schema_validation_results = []
        
        for test_case in test_cases:
            validation_result = await quality_validator.validate_schema(
                data=test_case['data'],
                schema_name='market_data'
            )
            
            schema_validation_results.append({
                'test_case': test_case['name'],
                'expected_pass': test_case['should_pass'],
                'actual_pass': validation_result['valid'],
                'validation_errors': validation_result.get('errors', [])
            })
        
        # Validate schema validation results
        for result in schema_validation_results:
            test_name = result['test_case']
            expected = result['expected_pass']
            actual = result['actual_pass']
            
            assert expected == actual, \
                f"Test {test_name}: expected {expected}, got {actual}. Errors: {result['validation_errors']}"
        
        # Check error messages for failed validations
        failed_results = [r for r in schema_validation_results if not r['actual_pass']]
        
        for result in failed_results:
            assert len(result['validation_errors']) > 0, \
                f"Failed validation {result['test_case']} should have error messages"
        
        print(f"\nData Schema Validation:")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Passed validations: {sum(1 for r in schema_validation_results if r['actual_pass'])}")
        print(f"  Failed validations: {sum(1 for r in schema_validation_results if not r['actual_pass'])}")
        
        await quality_validator.shutdown()
    
    @pytest.mark.asyncio
    async def test_data_format_consistency_regression(self):
        """Test for regressions in data format consistency."""
        config = {
            'format_tracking': True,
            'regression_detection': True
        }
        
        quality_analyzer = DataQualityAnalyzer(config)
        await quality_analyzer.initialize()
        
        # Simulate historical data format patterns
        historical_formats = [
            {
                'date': '2024-01-01',
                'format_signature': {
                    'field_count': 6,
                    'field_types': {'symbol': 'str', 'price': 'float', 'volume': 'int', 'timestamp': 'float'},
                    'precision': {'price': 2, 'timestamp': 3},
                    'null_rates': {'bid': 0.05, 'ask': 0.05}
                }
            },
            {
                'date': '2024-01-15',
                'format_signature': {
                    'field_count': 6,
                    'field_types': {'symbol': 'str', 'price': 'float', 'volume': 'int', 'timestamp': 'float'},
                    'precision': {'price': 2, 'timestamp': 3},
                    'null_rates': {'bid': 0.04, 'ask': 0.04}
                }
            }
        ]
        
        # Current data format (with some changes)
        current_format = {
            'date': '2024-02-01',
            'format_signature': {
                'field_count': 7,  # Added new field
                'field_types': {'symbol': 'str', 'price': 'float', 'volume': 'int', 'timestamp': 'float', 'exchange': 'str'},
                'precision': {'price': 3, 'timestamp': 3},  # Changed price precision
                'null_rates': {'bid': 0.15, 'ask': 0.15}  # Increased null rates
            }
        }
        
        # Analyze format consistency
        format_analysis = await quality_analyzer.analyze_format_consistency(
            historical_formats=historical_formats,
            current_format=current_format,
            consistency_thresholds={
                'field_count_change': 0.1,  # 10% change threshold
                'precision_change': 1,      # 1 decimal place change
                'null_rate_change': 0.1     # 10% null rate change
            }
        )
        
        # Validate format analysis
        assert 'format_changes' in format_analysis
        assert 'consistency_score' in format_analysis
        assert 'regression_indicators' in format_analysis
        
        format_changes = format_analysis['format_changes']
        consistency_score = format_analysis['consistency_score']
        regressions = format_analysis['regression_indicators']
        
        # Should detect format changes
        expected_changes = ['field_count', 'precision', 'null_rates']
        detected_changes = set(format_changes.keys())
        
        for expected_change in expected_changes:
            assert expected_change in detected_changes, f"Should detect {expected_change} change"
        
        # Consistency score should reflect the changes
        assert 0.5 < consistency_score < 0.9, \
            f"Consistency score {consistency_score:.3f} should reflect format changes"
        
        # Should flag some changes as potential regressions
        assert len(regressions) > 0, "Should detect potential format regressions"
        
        # Check specific regression indicators
        regression_types = [r['type'] for r in regressions]
        assert 'null_rate_increase' in regression_types, "Should detect null rate increase as regression"
        
        print(f"\nData Format Consistency Regression:")
        print(f"  Consistency score: {consistency_score:.3f}")
        print(f"  Format changes detected: {len(format_changes)}")
        print(f"  Regression indicators: {len(regressions)}")
        for regression in regressions:
            print(f"    - {regression['type']}: {regression.get('description', 'No description')}")
        
        await quality_analyzer.shutdown()


if __name__ == '__main__':
    pytest.main([__file__])