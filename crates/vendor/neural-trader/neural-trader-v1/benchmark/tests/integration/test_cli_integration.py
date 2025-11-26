"""
CLI integration test suite for AI News Trading benchmark system.

This module tests the CLI interface integration with all components:
- Benchmark command execution
- Simulation command integration
- Optimization command workflow
- Report generation
- Configuration handling
"""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import yaml

from benchmark.src.cli.commands import (
    BenchmarkCommand,
    SimulateCommand,
    OptimizeCommand,
    ReportCommand,
    CLIParser
)
from benchmark.benchmark_cli import main as cli_main


class TestCLIBenchmarkIntegration:
    """Test CLI benchmark command integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for CLI tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create config directory
            config_dir = workspace / 'configs'
            config_dir.mkdir()
            
            # Create results directory
            results_dir = workspace / 'results'
            results_dir.mkdir()
            
            # Create test configuration
            config = {
                'global': {
                    'output_dir': str(results_dir),
                    'log_level': 'INFO',
                    'parallel_workers': 2
                },
                'benchmark': {
                    'default_suite': 'quick',
                    'warmup_duration': 5,
                    'measurement_duration': 15
                },
                'simulation': {
                    'data_source': 'synthetic',
                    'tick_resolution': '100ms',
                    'symbols': ['AAPL', 'GOOGL', 'MSFT']
                },
                'strategies': {
                    'momentum': {'enabled': True},
                    'arbitrage': {'enabled': True}
                }
            }
            
            config_path = config_dir / 'test_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            yield {
                'workspace': workspace,
                'config_path': config_path,
                'results_dir': results_dir
            }
    
    @pytest.mark.asyncio
    async def test_benchmark_command_integration(self, temp_workspace):
        """Test benchmark command end-to-end execution."""
        config_path = temp_workspace['config_path']
        results_dir = temp_workspace['results_dir']
        
        # Test quick benchmark suite
        cmd = BenchmarkCommand()
        args = Mock()
        args.config = str(config_path)
        args.suite = 'quick'
        args.strategy = None
        args.duration = 15
        args.parallel = 2
        args.metrics = None
        args.baseline = None
        args.save_baseline = False
        args.format = 'json'
        args.output = str(results_dir / 'benchmark_results.json')
        
        with patch('benchmark.src.benchmarks.runner.BenchmarkRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            mock_runner.run_suite.return_value = {
                'status': 'success',
                'suite': 'quick',
                'results': {
                    'momentum': {
                        'latency_p99': 45.2,
                        'throughput': 1250,
                        'memory_mb': 512,
                        'total_return': 0.15,
                        'sharpe_ratio': 1.8
                    },
                    'arbitrage': {
                        'latency_p99': 32.1,
                        'throughput': 1800,
                        'memory_mb': 320,
                        'total_return': 0.08,
                        'sharpe_ratio': 2.1
                    }
                },
                'performance_summary': {
                    'avg_latency_p99': 38.65,
                    'avg_throughput': 1525,
                    'total_memory_mb': 832,
                    'all_targets_met': True
                }
            }
            
            result = await cmd.execute(args)
            
            # Validate command execution
            assert result['status'] == 'success'
            assert 'results' in result
            assert result['performance_summary']['all_targets_met'] is True
            
            # Verify runner was called correctly
            mock_runner.run_suite.assert_called_once_with('quick')
    
    @pytest.mark.asyncio
    async def test_benchmark_command_with_strategies(self, temp_workspace):
        """Test benchmark command with specific strategies."""
        config_path = temp_workspace['config_path']
        
        cmd = BenchmarkCommand()
        args = Mock()
        args.config = str(config_path)
        args.suite = None
        args.strategy = ['momentum', 'news_sentiment']
        args.duration = 30
        args.parallel = 4
        args.metrics = 'latency,throughput,memory'
        args.baseline = None
        args.save_baseline = True
        args.format = 'json'
        args.output = None
        
        with patch('benchmark.src.benchmarks.runner.BenchmarkRunner') as mock_runner_class:
            mock_runner = Mock()
            mock_runner_class.return_value = mock_runner
            mock_runner.run_strategies.return_value = {
                'status': 'success',
                'strategies': ['momentum', 'news_sentiment'],
                'results': {
                    'momentum': {'latency_p99': 55, 'throughput': 1100},
                    'news_sentiment': {'latency_p99': 85, 'throughput': 800}
                }
            }
            
            result = await cmd.execute(args)
            
            assert result['status'] == 'success'
            mock_runner.run_strategies.assert_called_once_with(
                ['momentum', 'news_sentiment'],
                duration=30,
                parallel=4,
                metrics=['latency', 'throughput', 'memory']
            )
    
    @pytest.mark.asyncio
    async def test_cli_parser_integration(self):
        """Test CLI parser with realistic command combinations."""
        parser = CLIParser()
        
        # Test complex benchmark command
        args = parser.parse_args([
            '--config', 'custom.yaml',
            '--verbose', '--verbose',
            '--format', 'json',
            '--output', 'results.json',
            'benchmark',
            '--suite', 'comprehensive',
            '--strategy', 'momentum',
            '--strategy', 'arbitrage',
            '--duration', '300',
            '--parallel', '8',
            '--metrics', 'latency,throughput,memory',
            '--save-baseline'
        ])
        
        assert args.config == 'custom.yaml'
        assert args.verbose == 2
        assert args.format == 'json'
        assert args.output == 'results.json'
        assert args.command == 'benchmark'
        assert args.suite == 'comprehensive'
        assert args.strategy == ['momentum', 'arbitrage']
        assert args.duration == 300
        assert args.parallel == 8
        assert args.metrics == 'latency,throughput,memory'
        assert args.save_baseline is True


class TestCLISimulationIntegration:
    """Test CLI simulation command integration."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for simulation tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create simulation data
            data_dir = workspace / 'data'
            data_dir.mkdir()
            
            # Create mock historical data
            historical_data = {
                'AAPL': {
                    'prices': [150 + i * 0.1 for i in range(1000)],
                    'volumes': [1000 + i * 10 for i in range(1000)],
                    'timestamps': [1640995200 + i * 60 for i in range(1000)]
                }
            }
            
            with open(data_dir / 'historical.json', 'w') as f:
                json.dump(historical_data, f)
            
            yield {
                'workspace': workspace,
                'data_dir': data_dir
            }
    
    @pytest.mark.asyncio
    async def test_simulation_command_integration(self, temp_workspace):
        """Test simulation command execution."""
        data_dir = temp_workspace['data_dir']
        
        cmd = SimulateCommand()
        args = Mock()
        args.scenario = 'historical'
        args.start_date = '2024-01-01'
        args.end_date = '2024-01-31'
        args.assets = 'AAPL,GOOGL,MSFT'
        args.strategies = ['momentum', 'arbitrage']
        args.capital = 100000.0
        args.threads = 4
        args.speed = 10.0
        args.live = False
        args.record = True
        args.config = str(data_dir / 'config.yaml')
        args.format = 'json'
        args.output = str(data_dir / 'simulation_results.json')
        
        with patch('benchmark.src.simulation.simulator.MarketSimulator') as mock_sim_class:
            mock_simulator = Mock()
            mock_sim_class.return_value = mock_simulator
            mock_simulator.run.return_value = {
                'status': 'success',
                'scenario': 'historical',
                'results': {
                    'total_return': 0.12,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': 0.08,
                    'trades_executed': 156,
                    'win_rate': 0.62
                },
                'performance': {
                    'avg_latency_ms': 35,
                    'max_latency_ms': 78,
                    'throughput': 1400
                }
            }
            
            result = await cmd.execute(args)
            
            assert result['status'] == 'success'
            assert result['results']['total_return'] == 0.12
            assert result['performance']['avg_latency_ms'] < 100
            
            # Verify simulator configuration
            mock_simulator.configure.assert_called_once()
            mock_simulator.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_live_simulation_integration(self, temp_workspace):
        """Test live simulation mode."""
        cmd = SimulateCommand()
        args = Mock()
        args.scenario = 'live'
        args.assets = 'AAPL,BTC-USD'
        args.strategies = ['momentum']
        args.capital = 50000.0
        args.live = True
        args.record = True
        args.speed = 1.0
        
        with patch('benchmark.src.simulation.simulator.MarketSimulator') as mock_sim_class:
            mock_simulator = Mock()
            mock_sim_class.return_value = mock_simulator
            mock_simulator.run_live.return_value = {
                'status': 'running',
                'mode': 'live',
                'session_id': 'live_session_123'
            }
            
            result = await cmd.execute(args)
            
            assert result['status'] == 'running'
            assert result['mode'] == 'live'
            mock_simulator.run_live.assert_called_once()


class TestCLIOptimizationIntegration:
    """Test CLI optimization command integration."""
    
    @pytest.mark.asyncio
    async def test_optimization_command_integration(self):
        """Test optimization command execution."""
        cmd = OptimizeCommand()
        args = Mock()
        args.algorithm = 'bayesian'
        args.objective = 'sharpe'
        args.constraints = None
        args.parameters = 'benchmark/configs/parameters.yaml'
        args.trials = 50
        args.timeout = 300
        args.parallel = True
        args.resume = None
        args.config = 'benchmark.yaml'
        args.format = 'json'
        args.output = 'optimization_results.json'
        
        with patch('benchmark.src.optimization.optimizer.StrategyOptimizer') as mock_opt_class:
            mock_optimizer = Mock()
            mock_opt_class.return_value = mock_optimizer
            mock_optimizer.optimize.return_value = {
                'status': 'success',
                'algorithm': 'bayesian',
                'best_params': {
                    'momentum_period': 15,
                    'threshold': 0.025,
                    'position_size': 0.12
                },
                'best_score': 2.34,
                'trials_completed': 50,
                'convergence_achieved': True,
                'optimization_time': 245.6
            }
            
            result = await cmd.execute(args)
            
            assert result['status'] == 'success'
            assert result['convergence_achieved'] is True
            assert result['best_score'] > 2.0
            
            # Verify optimizer was configured correctly
            mock_optimizer.configure.assert_called_once()
            mock_optimizer.optimize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self):
        """Test multi-objective optimization."""
        cmd = OptimizeCommand()
        args = Mock()
        args.algorithm = 'genetic'
        args.objective = 'multi'  # Multi-objective
        args.constraints = 'benchmark/configs/constraints.yaml'
        args.parameters = 'benchmark/configs/parameters.yaml'
        args.trials = 100
        args.timeout = 600
        args.parallel = True
        
        with patch('benchmark.src.optimization.optimizer.StrategyOptimizer') as mock_opt_class:
            mock_optimizer = Mock()
            mock_opt_class.return_value = mock_optimizer
            mock_optimizer.optimize_multi_objective.return_value = {
                'status': 'success',
                'pareto_front': [
                    {'sharpe': 2.1, 'return': 0.15, 'drawdown': 0.05},
                    {'sharpe': 1.9, 'return': 0.18, 'drawdown': 0.07},
                    {'sharpe': 1.7, 'return': 0.22, 'drawdown': 0.10}
                ],
                'best_compromise': {'sharpe': 2.1, 'return': 0.15, 'drawdown': 0.05}
            }
            
            result = await cmd.execute(args)
            
            assert result['status'] == 'success'
            assert 'pareto_front' in result
            assert len(result['pareto_front']) == 3


class TestCLIReportIntegration:
    """Test CLI report generation integration."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results for reporting."""
        return {
            'benchmark_results': {
                'momentum': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.8,
                    'max_drawdown': 0.06,
                    'win_rate': 0.65,
                    'latency_p99': 45,
                    'throughput': 1200
                },
                'arbitrage': {
                    'total_return': 0.08,
                    'sharpe_ratio': 2.2,
                    'max_drawdown': 0.03,
                    'win_rate': 0.78,
                    'latency_p99': 28,
                    'throughput': 1800
                }
            },
            'performance_summary': {
                'avg_latency_p99': 36.5,
                'avg_throughput': 1500,
                'memory_efficiency': 0.85,
                'all_targets_met': True
            },
            'timestamp': '2024-01-15T10:30:00Z',
            'duration': 300
        }
    
    @pytest.mark.asyncio
    async def test_dashboard_report_generation(self, sample_results):
        """Test dashboard report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_file = Path(temp_dir) / 'results.json'
            with open(results_file, 'w') as f:
                json.dump(sample_results, f)
            
            cmd = ReportCommand()
            args = Mock()
            args.type = 'dashboard'
            args.input = [str(results_file)]
            args.template = None
            args.charts = True
            args.export = 'html'
            args.serve = None
            args.output = str(Path(temp_dir) / 'dashboard.html')
            
            with patch('benchmark.src.reporting.dashboard.DashboardGenerator') as mock_dash:
                mock_generator = Mock()
                mock_dash.return_value = mock_generator
                mock_generator.generate.return_value = {
                    'status': 'success',
                    'output_file': args.output,
                    'charts_generated': 5,
                    'file_size_kb': 245
                }
                
                result = await cmd.execute(args)
                
                assert result['status'] == 'success'
                assert result['charts_generated'] == 5
                mock_generator.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comparison_report_generation(self, sample_results):
        """Test comparison report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create baseline and current results
            baseline_file = Path(temp_dir) / 'baseline.json'
            current_file = Path(temp_dir) / 'current.json'
            
            # Baseline results (slightly different)
            baseline_results = sample_results.copy()
            baseline_results['benchmark_results']['momentum']['latency_p99'] = 50
            baseline_results['benchmark_results']['arbitrage']['throughput'] = 1600
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_results, f)
            with open(current_file, 'w') as f:
                json.dump(sample_results, f)
            
            cmd = ReportCommand()
            args = Mock()
            args.type = 'comparison'
            args.input = [str(baseline_file), str(current_file)]
            args.charts = True
            args.export = 'pdf'
            args.output = str(Path(temp_dir) / 'comparison.pdf')
            
            with patch('benchmark.src.reporting.comparator.ResultsComparator') as mock_comp:
                mock_comparator = Mock()
                mock_comp.return_value = mock_comparator
                mock_comparator.compare.return_value = {
                    'status': 'success',
                    'improvements': {
                        'momentum_latency': {'baseline': 50, 'current': 45, 'improvement': '10%'},
                        'arbitrage_throughput': {'baseline': 1600, 'current': 1800, 'improvement': '12.5%'}
                    },
                    'regressions': {},
                    'overall_score': 'improved'
                }
                
                result = await cmd.execute(args)
                
                assert result['status'] == 'success'
                assert result['overall_score'] == 'improved'
                assert len(result['improvements']) == 2


class TestCLIWorkflowIntegration:
    """Test complete CLI workflow integration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_integration(self):
        """Test complete benchmark workflow via CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Step 1: Run benchmark
            benchmark_results = workspace / 'benchmark_results.json'
            
            with patch('benchmark.src.benchmarks.runner.BenchmarkRunner') as mock_runner:
                mock_runner.return_value.run_suite.return_value = {
                    'status': 'success',
                    'results': {'momentum': {'latency_p99': 45}}
                }
                
                benchmark_cmd = BenchmarkCommand()
                benchmark_args = Mock()
                benchmark_args.suite = 'quick'
                benchmark_args.output = str(benchmark_results)
                benchmark_args.format = 'json'
                
                benchmark_result = await benchmark_cmd.execute(benchmark_args)
                assert benchmark_result['status'] == 'success'
            
            # Step 2: Run optimization
            optimization_results = workspace / 'optimization_results.json'
            
            with patch('benchmark.src.optimization.optimizer.StrategyOptimizer') as mock_opt:
                mock_opt.return_value.optimize.return_value = {
                    'status': 'success',
                    'best_params': {'period': 20}
                }
                
                opt_cmd = OptimizeCommand()
                opt_args = Mock()
                opt_args.algorithm = 'bayesian'
                opt_args.objective = 'sharpe'
                opt_args.output = str(optimization_results)
                
                opt_result = await opt_cmd.execute(opt_args)
                assert opt_result['status'] == 'success'
            
            # Step 3: Run simulation with optimized parameters
            simulation_results = workspace / 'simulation_results.json'
            
            with patch('benchmark.src.simulation.simulator.MarketSimulator') as mock_sim:
                mock_sim.return_value.run.return_value = {
                    'status': 'success',
                    'results': {'total_return': 0.18}
                }
                
                sim_cmd = SimulateCommand()
                sim_args = Mock()
                sim_args.scenario = 'historical'
                sim_args.output = str(simulation_results)
                
                sim_result = await sim_cmd.execute(sim_args)
                assert sim_result['status'] == 'success'
            
            # Step 4: Generate comprehensive report
            with patch('benchmark.src.reporting.dashboard.DashboardGenerator') as mock_dash:
                mock_dash.return_value.generate.return_value = {
                    'status': 'success',
                    'output_file': 'dashboard.html'
                }
                
                report_cmd = ReportCommand()
                report_args = Mock()
                report_args.type = 'dashboard'
                report_args.input = [str(benchmark_results), str(optimization_results), str(simulation_results)]
                report_args.export = 'html'
                
                report_result = await report_cmd.execute(report_args)
                assert report_result['status'] == 'success'


if __name__ == '__main__':
    pytest.main([__file__])