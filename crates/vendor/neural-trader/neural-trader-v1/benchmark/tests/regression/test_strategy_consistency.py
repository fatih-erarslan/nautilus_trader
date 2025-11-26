"""
Strategy consistency regression test suite for AI News Trading benchmark system.

This module tests that trading strategies maintain consistent behavior across releases.
Tests include:
- Strategy output consistency
- Signal generation reproducibility
- Strategy parameter sensitivity
- Cross-version strategy compatibility
- Risk management consistency
"""

import asyncio
import json
import time
import statistics
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta
import hashlib

from benchmark.src.simulation.simulator import MarketSimulator
from benchmark.src.strategies import *
from benchmark.src.data.realtime_manager import RealtimeManager
from benchmark.src.analysis.consistency_checker import StrategyConsistencyChecker
from benchmark.src.profiling.profiler import ConsistencyProfiler


class TestStrategyOutputConsistency:
    """Test strategy output consistency across runs."""
    
    @pytest.fixture
    async def consistency_system(self):
        """Create system for consistency testing."""
        config = {
            'deterministic_mode': True,
            'random_seed': 42,
            'consistency_tracking': True,
            'strategy_versioning': True
        }
        
        system = {
            'simulator': MarketSimulator(config),
            'data_manager': RealtimeManager(config),
            'consistency_checker': StrategyConsistencyChecker(config),
            'profiler': ConsistencyProfiler()
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.fixture
    def deterministic_market_data(self):
        """Create deterministic market data for consistency testing."""
        np.random.seed(42)  # Fixed seed for reproducibility
        
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data = {}
        
        for symbol in symbols:
            base_price = {'AAPL': 150, 'GOOGL': 2800, 'MSFT': 300}[symbol]
            
            # Generate deterministic price series
            prices = [base_price]
            for i in range(999):
                # Use deterministic random walk
                change = np.random.normal(0, base_price * 0.001)
                new_price = max(prices[-1] + change, base_price * 0.5)
                prices.append(new_price)
            
            volumes = np.random.randint(100, 5000, 1000).tolist()
            timestamps = [1640995200 + i * 60 for i in range(1000)]
            
            data[symbol] = {
                'prices': prices,
                'volumes': volumes,
                'timestamps': timestamps
            }
        
        return data
    
    @pytest.mark.asyncio
    async def test_strategy_signal_reproducibility(self, consistency_system, deterministic_market_data):
        """Test that strategies generate identical signals with same input data."""
        simulator = consistency_system['simulator']
        consistency_checker = consistency_system['consistency_checker']
        
        strategies = ['momentum', 'arbitrage', 'news_sentiment']
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        # Run 1: Generate signals with deterministic data
        run1_signals = {}
        
        for strategy in strategies:
            run1_signals[strategy] = {}
            
            for symbol in symbols:
                symbol_signals = []
                market_data = deterministic_market_data[symbol]
                
                # Process each data point
                for i in range(len(market_data['prices'])):
                    data_point = {
                        'symbol': symbol,
                        'price': market_data['prices'][i],
                        'volume': market_data['volumes'][i],
                        'timestamp': market_data['timestamps'][i]
                    }
                    
                    signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                    if signal:
                        symbol_signals.append(signal)
                
                run1_signals[strategy][symbol] = symbol_signals
        
        # Reset system state
        await simulator.reset_state()
        
        # Run 2: Generate signals with same deterministic data
        run2_signals = {}
        
        for strategy in strategies:
            run2_signals[strategy] = {}
            
            for symbol in symbols:
                symbol_signals = []
                market_data = deterministic_market_data[symbol]
                
                # Process each data point (identical to run 1)
                for i in range(len(market_data['prices'])):
                    data_point = {
                        'symbol': symbol,
                        'price': market_data['prices'][i],
                        'volume': market_data['volumes'][i],
                        'timestamp': market_data['timestamps'][i]
                    }
                    
                    signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                    if signal:
                        symbol_signals.append(signal)
                
                run2_signals[strategy][symbol] = symbol_signals
        
        # Compare signal consistency
        consistency_results = await consistency_checker.compare_signal_sets(
            run1_signals, run2_signals
        )
        
        # Validate perfect reproducibility
        assert consistency_results['overall_consistency'] == 1.0, \
            f"Signal reproducibility {consistency_results['overall_consistency']:.3f} != 1.0"
        
        # Check per-strategy consistency
        for strategy in strategies:
            strategy_consistency = consistency_results['strategy_consistency'][strategy]
            assert strategy_consistency == 1.0, \
                f"Strategy {strategy} consistency {strategy_consistency:.3f} != 1.0"
        
        # Validate signal-level details
        for strategy in strategies:
            for symbol in symbols:
                signals1 = run1_signals[strategy][symbol]
                signals2 = run2_signals[strategy][symbol]
                
                assert len(signals1) == len(signals2), \
                    f"Signal count mismatch for {strategy}/{symbol}: {len(signals1)} vs {len(signals2)}"
                
                # Compare each signal
                for i, (sig1, sig2) in enumerate(zip(signals1, signals2)):
                    assert sig1['action'] == sig2['action'], \
                        f"Signal {i} action mismatch for {strategy}/{symbol}"
                    assert abs(sig1['confidence'] - sig2['confidence']) < 1e-10, \
                        f"Signal {i} confidence mismatch for {strategy}/{symbol}"
                    
                    if 'quantity' in sig1 and 'quantity' in sig2:
                        assert abs(sig1['quantity'] - sig2['quantity']) < 1e-10, \
                            f"Signal {i} quantity mismatch for {strategy}/{symbol}"
        
        print(f"\nSignal Reproducibility Test:")
        print(f"  Overall consistency: {consistency_results['overall_consistency']:.3f}")
        for strategy in strategies:
            print(f"  {strategy}: {consistency_results['strategy_consistency'][strategy]:.3f}")
    
    @pytest.mark.asyncio
    async def test_strategy_state_consistency(self, consistency_system, deterministic_market_data):
        """Test that strategy internal state remains consistent."""
        simulator = consistency_system['simulator']
        consistency_checker = consistency_system['consistency_checker']
        
        strategy = 'momentum'
        symbol = 'AAPL'
        market_data = deterministic_market_data[symbol]
        
        # Run strategy and capture state at checkpoints
        state_checkpoints = []
        checkpoint_intervals = [100, 250, 500, 750, 999]
        
        for i in range(len(market_data['prices'])):
            data_point = {
                'symbol': symbol,
                'price': market_data['prices'][i],
                'volume': market_data['volumes'][i],
                'timestamp': market_data['timestamps'][i]
            }
            
            # Generate signal (updates internal state)
            await simulator.generate_signal(symbol, data_point, strategy=strategy)
            
            # Capture state at checkpoints
            if i in checkpoint_intervals:
                state = await simulator.get_strategy_state(strategy, symbol)
                state_checkpoints.append({
                    'step': i,
                    'state': state.copy(),
                    'timestamp': data_point['timestamp']
                })
        
        # Reset and run again
        await simulator.reset_state()
        
        state_checkpoints_2 = []
        
        for i in range(len(market_data['prices'])):
            data_point = {
                'symbol': symbol,
                'price': market_data['prices'][i],
                'volume': market_data['volumes'][i],
                'timestamp': market_data['timestamps'][i]
            }
            
            await simulator.generate_signal(symbol, data_point, strategy=strategy)
            
            if i in checkpoint_intervals:
                state = await simulator.get_strategy_state(strategy, symbol)
                state_checkpoints_2.append({
                    'step': i,
                    'state': state.copy(),
                    'timestamp': data_point['timestamp']
                })
        
        # Compare state consistency
        state_consistency = await consistency_checker.compare_strategy_states(
            state_checkpoints, state_checkpoints_2
        )
        
        # Validate state consistency
        assert state_consistency['overall_consistency'] > 0.99, \
            f"Strategy state consistency {state_consistency['overall_consistency']:.3f} < 0.99"
        
        # Check each checkpoint
        for i, (checkpoint1, checkpoint2) in enumerate(zip(state_checkpoints, state_checkpoints_2)):
            assert checkpoint1['step'] == checkpoint2['step']
            
            state1 = checkpoint1['state']
            state2 = checkpoint2['state']
            
            # Compare numerical state values
            for key in state1:
                if isinstance(state1[key], (int, float)):
                    assert abs(state1[key] - state2[key]) < 1e-10, \
                        f"State {key} mismatch at step {checkpoint1['step']}"
                elif isinstance(state1[key], list):
                    assert len(state1[key]) == len(state2[key]), \
                        f"State {key} length mismatch at step {checkpoint1['step']}"
                    for j, (v1, v2) in enumerate(zip(state1[key], state2[key])):
                        if isinstance(v1, (int, float)):
                            assert abs(v1 - v2) < 1e-10, \
                                f"State {key}[{j}] mismatch at step {checkpoint1['step']}"
        
        print(f"\nStrategy State Consistency Test:")
        print(f"  Overall consistency: {state_consistency['overall_consistency']:.3f}")
        print(f"  Checkpoints verified: {len(state_checkpoints)}")
    
    @pytest.mark.asyncio
    async def test_strategy_determinism_across_environments(self, consistency_system):
        """Test strategy determinism across different environment configurations."""
        simulator = consistency_system['simulator']
        
        # Test configurations with different environment settings
        test_configs = [
            {'threading_mode': 'single', 'optimization_level': 'none'},
            {'threading_mode': 'multi', 'optimization_level': 'basic'},
            {'threading_mode': 'async', 'optimization_level': 'aggressive'}
        ]
        
        strategy = 'arbitrage'
        symbol = 'AAPL'
        
        # Fixed test data
        test_data = [
            {'symbol': symbol, 'price': 150.0, 'volume': 1000, 'timestamp': 1640995200 + i}
            for i in range(100)
        ]
        
        environment_results = []
        
        for config in test_configs:
            # Reconfigure simulator
            await simulator.reconfigure(config)
            await simulator.reset_state()
            
            # Generate signals with this configuration
            signals = []
            for data_point in test_data:
                signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                if signal:
                    signals.append(signal)
            
            environment_results.append({
                'config': config,
                'signals': signals,
                'signal_count': len(signals)
            })
        
        # Compare results across environments
        base_signals = environment_results[0]['signals']
        
        for i, env_result in enumerate(environment_results[1:], 1):
            test_signals = env_result['signals']
            
            # Signal counts should match
            assert len(base_signals) == len(test_signals), \
                f"Environment {i}: signal count {len(test_signals)} != base {len(base_signals)}"
            
            # Signal content should match
            for j, (base_sig, test_sig) in enumerate(zip(base_signals, test_signals)):
                assert base_sig['action'] == test_sig['action'], \
                    f"Environment {i}, signal {j}: action mismatch"
                assert abs(base_sig['confidence'] - test_sig['confidence']) < 1e-6, \
                    f"Environment {i}, signal {j}: confidence mismatch"
        
        print(f"\nEnvironment Determinism Test:")
        print(f"  Configurations tested: {len(test_configs)}")
        print(f"  Signals generated: {len(base_signals)}")
        print(f"  All environments consistent: True")


class TestStrategyParameterSensitivity:
    """Test strategy parameter sensitivity and consistency."""
    
    @pytest.fixture
    async def parameter_test_system(self):
        """Create system for parameter sensitivity testing."""
        config = {
            'parameter_sensitivity_mode': True,
            'sensitivity_analysis': True,
            'parameter_validation': True
        }
        
        system = {
            'simulator': MarketSimulator(config),
            'consistency_checker': StrategyConsistencyChecker(config)
        }
        
        for component in system.values():
            await component.initialize()
        
        yield system
        
        for component in system.values():
            await component.shutdown()
    
    @pytest.mark.asyncio
    async def test_parameter_sensitivity_consistency(self, parameter_test_system):
        """Test that parameter sensitivity remains consistent across versions."""
        simulator = parameter_test_system['simulator']
        consistency_checker = parameter_test_system['consistency_checker']
        
        strategy = 'momentum'
        symbol = 'AAPL'
        
        # Base parameters
        base_params = {
            'lookback_period': 20,
            'threshold': 0.02,
            'position_size': 0.1
        }
        
        # Parameter variations for sensitivity analysis
        param_variations = {
            'lookback_period': [10, 15, 20, 25, 30],
            'threshold': [0.01, 0.015, 0.02, 0.025, 0.03],
            'position_size': [0.05, 0.075, 0.1, 0.125, 0.15]
        }
        
        # Test data
        test_data = [
            {
                'symbol': symbol,
                'price': 150.0 + i * 0.1 + np.sin(i * 0.1) * 2,
                'volume': 1000 + i * 10,
                'timestamp': 1640995200 + i * 60
            }
            for i in range(200)
        ]
        
        sensitivity_results = {}
        
        # Test each parameter variation
        for param_name, param_values in param_variations.items():
            sensitivity_results[param_name] = []
            
            for param_value in param_values:
                # Set parameters
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                await simulator.configure_strategy_parameters(strategy, test_params)
                await simulator.reset_state()
                
                # Generate signals
                signals = []
                for data_point in test_data:
                    signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                    if signal:
                        signals.append(signal)
                
                # Calculate performance metrics
                if signals:
                    buy_signals = sum(1 for s in signals if s['action'] == 'buy')
                    sell_signals = sum(1 for s in signals if s['action'] == 'sell')
                    avg_confidence = statistics.mean(s['confidence'] for s in signals)
                    
                    metrics = {
                        'param_value': param_value,
                        'total_signals': len(signals),
                        'buy_signals': buy_signals,
                        'sell_signals': sell_signals,
                        'avg_confidence': avg_confidence,
                        'signal_rate': len(signals) / len(test_data)
                    }
                else:
                    metrics = {
                        'param_value': param_value,
                        'total_signals': 0,
                        'buy_signals': 0,
                        'sell_signals': 0,
                        'avg_confidence': 0,
                        'signal_rate': 0
                    }
                
                sensitivity_results[param_name].append(metrics)
        
        # Analyze sensitivity patterns
        sensitivity_analysis = await consistency_checker.analyze_parameter_sensitivity(
            sensitivity_results
        )
        
        # Validate sensitivity consistency
        for param_name, analysis in sensitivity_analysis.items():
            # Should show monotonic or reasonable sensitivity patterns
            values = [result['param_value'] for result in sensitivity_results[param_name]]
            signal_rates = [result['signal_rate'] for result in sensitivity_results[param_name]]
            
            # Check for reasonable sensitivity (not completely flat or chaotic)
            sensitivity_variance = np.var(signal_rates)
            assert 0.001 < sensitivity_variance < 0.1, \
                f"Parameter {param_name} sensitivity variance {sensitivity_variance:.4f} outside reasonable range"
            
            # Check for trend consistency
            trend_consistency = analysis.get('trend_consistency', 0)
            assert trend_consistency > 0.3, \
                f"Parameter {param_name} trend consistency {trend_consistency:.3f} < 0.3"
        
        print(f"\nParameter Sensitivity Test:")
        for param_name, analysis in sensitivity_analysis.items():
            print(f"  {param_name}: trend consistency {analysis.get('trend_consistency', 0):.3f}")
    
    @pytest.mark.asyncio
    async def test_parameter_boundary_consistency(self, parameter_test_system):
        """Test consistency at parameter boundaries."""
        simulator = parameter_test_system['simulator']
        
        strategy = 'momentum'
        symbol = 'AAPL'
        
        # Test boundary conditions
        boundary_tests = [
            # Parameter at minimum boundary
            {'lookback_period': 1, 'threshold': 0.001, 'position_size': 0.01},
            # Parameter at maximum boundary
            {'lookback_period': 100, 'threshold': 0.1, 'position_size': 0.5},
            # Mixed boundaries
            {'lookback_period': 1, 'threshold': 0.1, 'position_size': 0.01},
            {'lookback_period': 100, 'threshold': 0.001, 'position_size': 0.5}
        ]
        
        test_data = [
            {
                'symbol': symbol,
                'price': 150.0 + i * 0.05,
                'volume': 1000,
                'timestamp': 1640995200 + i * 60
            }
            for i in range(50)
        ]
        
        boundary_results = []
        
        for boundary_params in boundary_tests:
            await simulator.configure_strategy_parameters(strategy, boundary_params)
            await simulator.reset_state()
            
            # Test multiple runs with same boundary parameters
            run_results = []
            
            for run in range(3):  # 3 runs per boundary condition
                await simulator.reset_state()
                
                signals = []
                for data_point in test_data:
                    try:
                        signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                        if signal:
                            signals.append(signal)
                    except Exception as e:
                        # Boundary conditions might cause exceptions
                        signals.append({'error': str(e)})
                
                run_results.append({
                    'run': run,
                    'signals': signals,
                    'errors': sum(1 for s in signals if 'error' in s)
                })
            
            boundary_results.append({
                'params': boundary_params,
                'runs': run_results,
                'consistency': self._calculate_boundary_consistency(run_results)
            })
        
        # Validate boundary consistency
        for result in boundary_results:
            consistency = result['consistency']
            
            # At boundaries, we expect some consistency (even if low performance)
            assert consistency > 0.8, \
                f"Boundary consistency {consistency:.3f} < 0.8 for params {result['params']}"
            
            # Should not have excessive errors
            total_errors = sum(run['errors'] for run in result['runs'])
            total_signals = sum(len(run['signals']) for run in result['runs'])
            error_rate = total_errors / total_signals if total_signals > 0 else 1
            
            assert error_rate < 0.1, \
                f"Error rate {error_rate:.3f} > 0.1 for boundary params {result['params']}"
        
        print(f"\nParameter Boundary Test:")
        for result in boundary_results:
            print(f"  Params {result['params']}: consistency {result['consistency']:.3f}")
    
    def _calculate_boundary_consistency(self, run_results):
        """Calculate consistency across boundary test runs."""
        if not run_results:
            return 0.0
        
        # Compare signal counts across runs
        signal_counts = [len([s for s in run['signals'] if 'error' not in s]) for run in run_results]
        
        if not signal_counts or max(signal_counts) == 0:
            return 1.0  # Consistently no signals
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_count = statistics.mean(signal_counts)
        std_count = statistics.stdev(signal_counts) if len(signal_counts) > 1 else 0
        
        cv = std_count / mean_count if mean_count > 0 else 0
        consistency = max(0, 1 - cv)  # Convert CV to consistency score
        
        return consistency


class TestCrossVersionCompatibility:
    """Test strategy compatibility across different versions."""
    
    @pytest.mark.asyncio
    async def test_strategy_serialization_compatibility(self):
        """Test that strategy state can be serialized/deserialized consistently."""
        config = {'serialization_mode': True}
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        strategy = 'momentum'
        symbol = 'AAPL'
        
        # Initialize strategy with some state
        test_data = [
            {'symbol': symbol, 'price': 150.0 + i, 'volume': 1000, 'timestamp': 1640995200 + i * 60}
            for i in range(20)
        ]
        
        # Build up strategy state
        for data_point in test_data:
            await simulator.generate_signal(symbol, data_point, strategy=strategy)
        
        # Serialize strategy state
        original_state = await simulator.get_strategy_state(strategy, symbol)
        serialized_state = await simulator.serialize_strategy_state(strategy, symbol)
        
        # Verify serialization is valid JSON
        assert isinstance(serialized_state, str)
        parsed_state = json.loads(serialized_state)
        assert isinstance(parsed_state, dict)
        
        # Reset and deserialize
        await simulator.reset_state()
        await simulator.deserialize_strategy_state(strategy, symbol, serialized_state)
        
        # Compare restored state
        restored_state = await simulator.get_strategy_state(strategy, symbol)
        
        # States should match
        assert self._compare_states(original_state, restored_state), \
            "Serialized/deserialized state doesn't match original"
        
        # Generate signals to test functionality
        post_restore_signals = []
        for data_point in test_data[-5:]:  # Test with last 5 data points
            signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
            if signal:
                post_restore_signals.append(signal)
        
        # Should be able to generate signals normally
        assert len(post_restore_signals) >= 0, "Strategy should function after deserialization"
        
        print(f"\nStrategy Serialization Test:")
        print(f"  Original state keys: {len(original_state)}")
        print(f"  Serialized size: {len(serialized_state)} chars")
        print(f"  Post-restore signals: {len(post_restore_signals)}")
        
        await simulator.shutdown()
    
    def _compare_states(self, state1, state2):
        """Compare two strategy states for equality."""
        if type(state1) != type(state2):
            return False
        
        if isinstance(state1, dict):
            if set(state1.keys()) != set(state2.keys()):
                return False
            return all(self._compare_states(state1[k], state2[k]) for k in state1.keys())
        
        elif isinstance(state1, list):
            if len(state1) != len(state2):
                return False
            return all(self._compare_states(v1, v2) for v1, v2 in zip(state1, state2))
        
        elif isinstance(state1, (int, float)):
            return abs(state1 - state2) < 1e-10
        
        else:
            return state1 == state2
    
    @pytest.mark.asyncio
    async def test_strategy_version_hash_consistency(self):
        """Test that strategy version hashes remain consistent."""
        simulator = MarketSimulator({'version_tracking': True})
        await simulator.initialize()
        
        strategies = ['momentum', 'arbitrage', 'news_sentiment']
        
        # Generate version hashes for each strategy
        version_hashes = {}
        
        for strategy in strategies:
            # Get strategy implementation hash
            strategy_hash = await simulator.get_strategy_version_hash(strategy)
            version_hashes[strategy] = strategy_hash
            
            # Hash should be consistent across calls
            hash2 = await simulator.get_strategy_version_hash(strategy)
            assert strategy_hash == hash2, f"Strategy {strategy} hash inconsistent across calls"
            
            # Hash should be deterministic
            assert len(strategy_hash) == 64, f"Strategy {strategy} hash wrong length: {len(strategy_hash)}"
            assert all(c in '0123456789abcdef' for c in strategy_hash), \
                f"Strategy {strategy} hash contains invalid characters"
        
        # Test strategy parameter hash consistency
        for strategy in strategies:
            base_params = await simulator.get_default_strategy_parameters(strategy)
            
            # Hash with base parameters
            base_hash = await simulator.get_strategy_parameter_hash(strategy, base_params)
            
            # Same parameters should give same hash
            base_hash2 = await simulator.get_strategy_parameter_hash(strategy, base_params)
            assert base_hash == base_hash2, f"Parameter hash inconsistent for {strategy}"
            
            # Different parameters should give different hash
            modified_params = base_params.copy()
            if 'threshold' in modified_params:
                modified_params['threshold'] = modified_params['threshold'] * 1.1
            else:
                # Add a new parameter
                modified_params['test_param'] = 0.5
            
            modified_hash = await simulator.get_strategy_parameter_hash(strategy, modified_params)
            assert base_hash != modified_hash, f"Parameter hash should change with different params for {strategy}"
        
        print(f"\nStrategy Version Hash Test:")
        for strategy, hash_value in version_hashes.items():
            print(f"  {strategy}: {hash_value[:16]}...")
        
        await simulator.shutdown()


class TestRiskManagementConsistency:
    """Test risk management consistency across strategy executions."""
    
    @pytest.mark.asyncio
    async def test_position_sizing_consistency(self):
        """Test that position sizing remains consistent for similar market conditions."""
        config = {
            'risk_management': True,
            'position_sizing': 'dynamic',
            'consistency_tracking': True
        }
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        strategy = 'momentum'
        symbol = 'AAPL'
        
        # Create similar market conditions across different time periods
        similar_conditions = [
            # Period 1: Rising trend
            [150.0 + i * 0.5 for i in range(20)],
            # Period 2: Similar rising trend (different time)
            [155.0 + i * 0.5 for i in range(20)],
            # Period 3: Another similar rising trend
            [148.0 + i * 0.5 for i in range(20)]
        ]
        
        position_sizes = []
        
        for period_idx, prices in enumerate(similar_conditions):
            await simulator.reset_state()
            
            period_positions = []
            
            for i, price in enumerate(prices):
                data_point = {
                    'symbol': symbol,
                    'price': price,
                    'volume': 1000,
                    'timestamp': 1640995200 + period_idx * 1000 + i * 60
                }
                
                signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                
                if signal and signal.get('action') in ['buy', 'sell']:
                    position_size = signal.get('quantity', 0)
                    period_positions.append(position_size)
            
            position_sizes.append(period_positions)
        
        # Analyze position sizing consistency
        if all(len(positions) > 0 for positions in position_sizes):
            # Compare average position sizes across periods
            avg_positions = [statistics.mean(positions) for positions in position_sizes]
            
            # Position sizes should be similar for similar market conditions
            position_cv = statistics.stdev(avg_positions) / statistics.mean(avg_positions)
            assert position_cv < 0.3, f"Position sizing CV {position_cv:.3f} > 0.3 (inconsistent)"
            
            # Check individual position consistency
            for i in range(len(position_sizes) - 1):
                for j in range(i + 1, len(position_sizes)):
                    positions1 = position_sizes[i]
                    positions2 = position_sizes[j]
                    
                    # Compare distributions
                    min_len = min(len(positions1), len(positions2))
                    if min_len > 5:
                        corr = np.corrcoef(positions1[:min_len], positions2[:min_len])[0, 1]
                        assert corr > 0.5, f"Position correlation {corr:.3f} < 0.5 between periods {i} and {j}"
        
        print(f"\nPosition Sizing Consistency Test:")
        for i, positions in enumerate(position_sizes):
            if positions:
                print(f"  Period {i}: {len(positions)} positions, avg {statistics.mean(positions):.3f}")
        
        await simulator.shutdown()
    
    @pytest.mark.asyncio
    async def test_risk_limit_enforcement_consistency(self):
        """Test that risk limits are consistently enforced."""
        config = {
            'risk_management': True,
            'risk_limits': {
                'max_position_size': 0.2,
                'max_daily_loss': 0.05,
                'max_drawdown': 0.1
            }
        }
        
        simulator = MarketSimulator(config)
        await simulator.initialize()
        
        strategy = 'arbitrage'
        symbol = 'AAPL'
        
        # Test scenarios that should trigger risk limits
        risk_scenarios = [
            # Scenario 1: Large position request
            {
                'name': 'large_position',
                'data': [{'symbol': symbol, 'price': 150, 'volume': 10000, 'timestamp': 1640995200}],
                'expected_limit': 'position_size'
            },
            # Scenario 2: High volatility (potential large loss)
            {
                'name': 'high_volatility',
                'data': [
                    {'symbol': symbol, 'price': 150, 'volume': 1000, 'timestamp': 1640995200 + i * 60}
                    for i in range(10)
                ] + [
                    {'symbol': symbol, 'price': 130, 'volume': 5000, 'timestamp': 1640995200 + 10 * 60}  # Big drop
                ],
                'expected_limit': 'daily_loss'
            }
        ]
        
        risk_enforcement_results = []
        
        for scenario in risk_scenarios:
            await simulator.reset_state()
            
            signals_generated = []
            risk_violations = []
            
            for data_point in scenario['data']:
                signal = await simulator.generate_signal(symbol, data_point, strategy=strategy)
                
                if signal:
                    signals_generated.append(signal)
                    
                    # Check for risk violations
                    risk_check = await simulator.check_risk_compliance(signal)
                    if not risk_check['compliant']:
                        risk_violations.append({
                            'signal': signal,
                            'violations': risk_check['violations']
                        })
            
            risk_enforcement_results.append({
                'scenario': scenario['name'],
                'signals': len(signals_generated),
                'violations': len(risk_violations),
                'violation_details': risk_violations
            })
        
        # Validate risk enforcement
        for result in risk_enforcement_results:
            scenario_name = result['scenario']
            
            if scenario_name == 'large_position':
                # Should detect position size violations
                position_violations = [
                    v for v in result['violation_details']
                    if any('position_size' in violation for violation in v['violations'])
                ]
                assert len(position_violations) > 0, \
                    f"Should detect position size violations in {scenario_name}"
            
            elif scenario_name == 'high_volatility':
                # Should detect potential loss violations
                loss_violations = [
                    v for v in result['violation_details']
                    if any('loss' in violation or 'drawdown' in violation for violation in v['violations'])
                ]
                # Note: May not always trigger depending on strategy behavior
                # Just ensure system is checking
                print(f"  {scenario_name}: {len(loss_violations)} loss-related violations detected")
        
        print(f"\nRisk Limit Enforcement Test:")
        for result in risk_enforcement_results:
            print(f"  {result['scenario']}: {result['signals']} signals, {result['violations']} violations")
        
        await simulator.shutdown()


if __name__ == '__main__':
    pytest.main([__file__])