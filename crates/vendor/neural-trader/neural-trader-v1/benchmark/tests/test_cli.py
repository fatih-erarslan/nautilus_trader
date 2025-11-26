"""
Comprehensive tests for the benchmark CLI tool.
Following TDD methodology - RED phase: Writing failing tests first.
"""

import pytest
from click.testing import CliRunner
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os


class TestCLIBasics:
    """Test basic CLI functionality"""
    
    def test_cli_entry_point_exists(self):
        """Test that the CLI entry point exists and is callable"""
        from benchmark.cli import cli
        assert cli is not None
        
    def test_cli_help_command(self):
        """Test that help command works"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'Usage:' in result.output
        assert 'Commands:' in result.output
        
    def test_cli_version_command(self):
        """Test version display"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0
        assert 'version' in result.output.lower()


class TestBenchmarkCommand:
    """Test the benchmark command functionality"""
    
    def test_benchmark_command_exists(self):
        """Test that benchmark command is available"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['benchmark', '--help'])
        assert result.exit_code == 0
        assert '--strategy' in result.output
        assert '--duration' in result.output
        assert '--assets' in result.output
        
    def test_benchmark_momentum_strategy(self):
        """Test benchmarking momentum strategy"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'momentum',
            '--duration', '1h',
            '--assets', 'stocks'
        ])
        assert result.exit_code == 0
        assert 'Benchmarking momentum strategy' in result.output
        assert 'Duration: 1h' in result.output
        assert 'Assets: stocks' in result.output
        
    def test_benchmark_all_strategies(self):
        """Test benchmarking all strategies"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'all',
            '--duration', '30m',
            '--assets', 'stocks,bonds'
        ])
        assert result.exit_code == 0
        assert 'momentum' in result.output.lower()
        assert 'swing' in result.output.lower()
        assert 'mirror' in result.output.lower()
        
    def test_benchmark_with_progress_bar(self):
        """Test that progress bar is shown during benchmarking"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'swing',
            '--duration', '5m',
            '--progress'
        ])
        assert result.exit_code == 0
        # Progress bar output might be cleared, check for completion message
        assert 'completed' in result.output.lower()
        
    def test_benchmark_output_formats(self):
        """Test different output formats"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON output
            json_file = os.path.join(tmpdir, 'results.json')
            result = runner.invoke(cli, [
                'benchmark',
                '--strategy', 'momentum',
                '--duration', '1m',
                '--output', json_file,
                '--format', 'json'
            ])
            assert result.exit_code == 0
            assert os.path.exists(json_file)
            with open(json_file) as f:
                data = json.load(f)
                assert 'results' in data
                assert 'metrics' in data
                
    def test_benchmark_concurrent_execution(self):
        """Test concurrent execution support"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'all',
            '--duration', '10m',
            '--concurrent',
            '--workers', '4'
        ])
        assert result.exit_code == 0
        assert 'concurrent' in result.output.lower() or 'parallel' in result.output.lower()


class TestSimulateCommand:
    """Test the simulate command functionality"""
    
    def test_simulate_command_exists(self):
        """Test that simulate command is available"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['simulate', '--help'])
        assert result.exit_code == 0
        assert '--historical' in result.output
        assert '--start' in result.output
        assert '--end' in result.output
        
    def test_simulate_historical_data(self):
        """Test historical simulation"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'simulate',
            '--historical',
            '--start', '2024-01-01',
            '--end', '2024-12-31'
        ])
        assert result.exit_code == 0
        assert 'Simulating from 2024-01-01 to 2024-12-31' in result.output
        
    def test_simulate_with_specific_assets(self):
        """Test simulation with specific assets"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'simulate',
            '--historical',
            '--start', '2024-06-01',
            '--end', '2024-06-30',
            '--assets', 'BTC,ETH,AAPL'
        ])
        assert result.exit_code == 0
        assert 'BTC' in result.output
        assert 'ETH' in result.output
        assert 'AAPL' in result.output
        
    def test_simulate_realtime_mode(self):
        """Test real-time simulation mode"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'simulate',
            '--realtime',
            '--duration', '1m'
        ])
        assert result.exit_code == 0
        assert 'real-time' in result.output.lower()
        
    def test_simulate_with_config_file(self):
        """Test simulation with config file"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'simulation': {
                    'start_date': '2024-01-01',
                    'end_date': '2024-03-31',
                    'initial_capital': 100000,
                    'strategies': ['momentum', 'swing'],
                    'assets': ['stocks', 'bonds']
                }
            }
            yaml.dump(config, f)
            config_file = f.name
            
        try:
            result = runner.invoke(cli, [
                'simulate',
                '--config', config_file
            ])
            assert result.exit_code == 0
            assert '100000' in result.output or '100,000' in result.output
        finally:
            os.unlink(config_file)


class TestOptimizeCommand:
    """Test the optimize command functionality"""
    
    def test_optimize_command_exists(self):
        """Test that optimize command is available"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['optimize', '--help'])
        assert result.exit_code == 0
        assert '--strategy' in result.output
        assert '--metric' in result.output
        assert '--iterations' in result.output
        
    def test_optimize_single_strategy(self):
        """Test optimizing a single strategy"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'optimize',
            '--strategy', 'momentum',
            '--metric', 'sharpe',
            '--iterations', '100'
        ])
        assert result.exit_code == 0
        assert 'Optimizing momentum strategy' in result.output
        assert 'sharpe' in result.output.lower()
        assert '100 iterations' in result.output
        
    def test_optimize_all_strategies(self):
        """Test optimizing all strategies"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'optimize',
            '--strategy', 'all',
            '--metric', 'returns',
            '--iterations', '50'
        ])
        assert result.exit_code == 0
        assert 'all strategies' in result.output.lower()
        
    def test_optimize_multiple_metrics(self):
        """Test optimization with multiple metrics"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'optimize',
            '--strategy', 'swing',
            '--metric', 'sharpe,sortino,calmar',
            '--iterations', '200'
        ])
        assert result.exit_code == 0
        assert 'sharpe' in result.output.lower()
        assert 'sortino' in result.output.lower()
        assert 'calmar' in result.output.lower()
        
    def test_optimize_with_constraints(self):
        """Test optimization with constraints"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'optimize',
            '--strategy', 'momentum',
            '--metric', 'sharpe',
            '--iterations', '100',
            '--max-drawdown', '0.2',
            '--min-trades', '50'
        ])
        assert result.exit_code == 0
        assert 'constraints' in result.output.lower()


class TestReportCommand:
    """Test the report command functionality"""
    
    def test_report_command_exists(self):
        """Test that report command is available"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['report', '--help'])
        assert result.exit_code == 0
        assert '--format' in result.output
        assert '--output' in result.output
        
    def test_report_html_format(self):
        """Test HTML report generation"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'report.html')
            result = runner.invoke(cli, [
                'report',
                '--format', 'html',
                '--output', output_file
            ])
            assert result.exit_code == 0
            assert os.path.exists(output_file)
            with open(output_file) as f:
                content = f.read()
                assert '<html>' in content
                assert 'Performance Report' in content
                
    def test_report_json_format(self):
        """Test JSON report generation"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'report.json')
            result = runner.invoke(cli, [
                'report',
                '--format', 'json',
                '--output', output_file
            ])
            assert result.exit_code == 0
            assert os.path.exists(output_file)
            with open(output_file) as f:
                data = json.load(f)
                assert 'summary' in data
                assert 'strategies' in data
                
    def test_report_pdf_format(self):
        """Test PDF report generation"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, 'report.pdf')
            result = runner.invoke(cli, [
                'report',
                '--format', 'pdf',
                '--output', output_file
            ])
            # PDF generation might require additional dependencies
            # Check if it fails gracefully or succeeds
            if result.exit_code == 0:
                assert os.path.exists(output_file)
            else:
                assert 'pdf' in result.output.lower()
                
    def test_report_with_date_range(self):
        """Test report generation with date range"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'report',
            '--start', '2024-01-01',
            '--end', '2024-12-31',
            '--format', 'text'
        ])
        assert result.exit_code == 0
        assert '2024' in result.output


class TestConfigurationManagement:
    """Test configuration management functionality"""
    
    def test_config_from_yaml_file(self):
        """Test loading configuration from YAML file"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'benchmark': {
                    'default_strategy': 'momentum',
                    'default_duration': '1h',
                    'default_assets': ['stocks', 'bonds']
                }
            }
            yaml.dump(config, f)
            config_file = f.name
            
        try:
            result = runner.invoke(cli, [
                '--config', config_file,
                'benchmark'
            ])
            assert result.exit_code == 0
        finally:
            os.unlink(config_file)
            
    def test_config_from_json_file(self):
        """Test loading configuration from JSON file"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                'benchmark': {
                    'default_strategy': 'swing',
                    'default_duration': '30m'
                }
            }
            json.dump(config, f)
            config_file = f.name
            
        try:
            result = runner.invoke(cli, [
                '--config', config_file,
                'benchmark'
            ])
            assert result.exit_code == 0
        finally:
            os.unlink(config_file)
            
    def test_environment_variable_override(self):
        """Test environment variable configuration override"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        env = {
            'BENCHMARK_STRATEGY': 'mirror',
            'BENCHMARK_DURATION': '2h'
        }
        
        result = runner.invoke(cli, ['benchmark'], env=env)
        # Should use environment variables if no explicit args provided
        assert result.exit_code == 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_strategy_name(self):
        """Test handling of invalid strategy name"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'invalid_strategy'
        ])
        assert result.exit_code != 0
        assert 'invalid' in result.output.lower() or 'error' in result.output.lower()
        
    def test_invalid_date_format(self):
        """Test handling of invalid date format"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, [
            'simulate',
            '--start', 'not-a-date',
            '--end', '2024-12-31'
        ])
        assert result.exit_code != 0
        assert 'date' in result.output.lower() or 'error' in result.output.lower()
        
    def test_missing_required_arguments(self):
        """Test handling of missing required arguments"""
        from benchmark.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['optimize'])
        assert result.exit_code != 0
        assert 'required' in result.output.lower() or 'missing' in result.output.lower()
        
    def test_file_permission_error(self):
        """Test handling of file permission errors"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        # Try to write to a read-only location
        result = runner.invoke(cli, [
            'report',
            '--output', '/root/report.html'
        ])
        assert result.exit_code != 0
        assert 'permission' in result.output.lower() or 'error' in result.output.lower()


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_benchmark_to_report_workflow(self):
        """Test complete workflow from benchmark to report"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run benchmark
            result = runner.invoke(cli, [
                'benchmark',
                '--strategy', 'all',
                '--duration', '5m',
                '--output', os.path.join(tmpdir, 'benchmark_results.json')
            ])
            assert result.exit_code == 0
            
            # Generate report from results
            result = runner.invoke(cli, [
                'report',
                '--input', os.path.join(tmpdir, 'benchmark_results.json'),
                '--format', 'html',
                '--output', os.path.join(tmpdir, 'report.html')
            ])
            assert result.exit_code == 0
            assert os.path.exists(os.path.join(tmpdir, 'report.html'))
            
    def test_simulate_optimize_workflow(self):
        """Test workflow from simulation to optimization"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run simulation
            sim_output = os.path.join(tmpdir, 'simulation.json')
            result = runner.invoke(cli, [
                'simulate',
                '--historical',
                '--start', '2024-01-01',
                '--end', '2024-03-31',
                '--output', sim_output
            ])
            assert result.exit_code == 0
            
            # Optimize based on simulation
            result = runner.invoke(cli, [
                'optimize',
                '--input', sim_output,
                '--strategy', 'all',
                '--metric', 'sharpe',
                '--iterations', '50'
            ])
            assert result.exit_code == 0


class TestPerformanceAndCaching:
    """Test performance features and caching"""
    
    def test_result_caching(self):
        """Test that results are cached appropriately"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, '.cache')
            
            # First run should create cache
            result = runner.invoke(cli, [
                'benchmark',
                '--strategy', 'momentum',
                '--duration', '1m',
                '--cache-dir', cache_dir
            ])
            assert result.exit_code == 0
            assert os.path.exists(cache_dir)
            
            # Second run should use cache
            result = runner.invoke(cli, [
                'benchmark',
                '--strategy', 'momentum',
                '--duration', '1m',
                '--cache-dir', cache_dir,
                '--use-cache'
            ])
            assert result.exit_code == 0
            assert 'cache' in result.output.lower()
            
    def test_performance_profiling(self):
        """Test performance profiling feature"""
        from benchmark.cli import cli
        runner = CliRunner()
        
        result = runner.invoke(cli, [
            'benchmark',
            '--strategy', 'momentum',
            '--duration', '30s',
            '--profile'
        ])
        assert result.exit_code == 0
        assert 'profiling' in result.output.lower() or 'performance' in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])