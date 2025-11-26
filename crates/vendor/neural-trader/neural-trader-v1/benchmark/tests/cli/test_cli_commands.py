"""Tests for CLI command parsing and execution."""

import argparse
import sys
import unittest
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from benchmark.src.cli.commands import (
    BenchmarkCommand,
    CLIParser,
    CompareCommand,
    OptimizeCommand,
    ProfileCommand,
    ReportCommand,
    SimulateCommand,
)


class TestCLIParser(unittest.TestCase):
    """Test cases for CLI parser."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = CLIParser()

    def test_parse_global_options(self):
        """Test parsing global options."""
        args = self.parser.parse_args([
            "--config", "test.yaml",
            "--verbose", "--verbose",
            "--format", "json",
            "--output", "results.json",
            "benchmark",
            "--suite", "quick"
        ])
        
        assert args.config == "test.yaml"
        assert args.verbose == 2
        assert args.format == "json"
        assert args.output == "results.json"
        assert args.command == "benchmark"
        assert args.suite == "quick"

    def test_parse_benchmark_command(self):
        """Test parsing benchmark command."""
        args = self.parser.parse_args([
            "benchmark",
            "--suite", "comprehensive",
            "--strategy", "momentum",
            "--strategy", "arbitrage",
            "--duration", "600",
            "--parallel", "4",
            "--metrics", "latency,throughput,memory",
            "--baseline", "baseline.json",
            "--save-baseline"
        ])
        
        assert args.command == "benchmark"
        assert args.suite == "comprehensive"
        assert args.strategy == ["momentum", "arbitrage"]
        assert args.duration == 600
        assert args.parallel == 4
        assert args.metrics == "latency,throughput,memory"
        assert args.baseline == "baseline.json"
        assert args.save_baseline is True

    def test_parse_simulate_command(self):
        """Test parsing simulate command."""
        args = self.parser.parse_args([
            "simulate",
            "--scenario", "historical",
            "--start-date", "2024-01-01",
            "--end-date", "2024-12-31",
            "--assets", "AAPL,GOOGL,MSFT",
            "--capital", "50000",
            "--speed", "10.0",
            "--live",
            "--record"
        ])
        
        assert args.command == "simulate"
        assert args.scenario == "historical"
        assert args.start_date == "2024-01-01"
        assert args.end_date == "2024-12-31"
        assert args.assets == "AAPL,GOOGL,MSFT"
        assert args.capital == 50000.0
        assert args.speed == 10.0
        assert args.live is True
        assert args.record is True

    def test_parse_optimize_command(self):
        """Test parsing optimize command."""
        args = self.parser.parse_args([
            "optimize",
            "--algorithm", "bayesian",
            "--objective", "sharpe",
            "--constraints", "limits.yaml",
            "--parameters", "params.json",
            "--trials", "1000",
            "--timeout", "60",
            "--parallel",
            "--resume", "checkpoint.pkl"
        ])
        
        assert args.command == "optimize"
        assert args.algorithm == "bayesian"
        assert args.objective == "sharpe"
        assert args.constraints == "limits.yaml"
        assert args.parameters == "params.json"
        assert args.trials == 1000
        assert args.timeout == 60
        assert args.parallel is True
        assert args.resume == "checkpoint.pkl"

    def test_parse_report_command(self):
        """Test parsing report command."""
        args = self.parser.parse_args([
            "report",
            "--type", "dashboard",
            "--input", "results1.json",
            "--input", "results2.json",
            "--template", "custom.html",
            "--charts",
            "--export", "html",
            "--serve", "8080"
        ])
        
        assert args.command == "report"
        assert args.type == "dashboard"
        assert args.input == ["results1.json", "results2.json"]
        assert args.template == "custom.html"
        assert args.charts is True
        assert args.export == "html"
        assert args.serve == 8080

    def test_parse_profile_command(self):
        """Test parsing profile command."""
        args = self.parser.parse_args([
            "profile",
            "--target", "cpu",
            "--component", "signal_generator",
            "--duration", "120",
            "--sampling-rate", "100",
            "--flame-graph"
        ])
        
        assert args.command == "profile"
        assert args.target == "cpu"
        assert args.component == "signal_generator"
        assert args.duration == 120
        assert args.sampling_rate == 100
        assert args.flame_graph is True

    def test_parse_compare_command(self):
        """Test parsing compare command."""
        args = self.parser.parse_args([
            "compare",
            "--baseline", "v1.0.json",
            "--current", "v1.1.json",
            "--threshold", "5",
            "--metrics", "latency,throughput",
            "--visualize"
        ])
        
        assert args.command == "compare"
        assert args.baseline == "v1.0.json"
        assert args.current == "v1.1.json"
        assert args.threshold == 5.0
        assert args.metrics == "latency,throughput"
        assert args.visualize is True

    def test_quiet_verbose_mutual_exclusion(self):
        """Test that quiet and verbose options are mutually exclusive."""
        with pytest.raises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                self.parser.parse_args(["--quiet", "--verbose", "benchmark"])

    def test_help_output(self):
        """Test help output."""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                self.parser.parse_args(["--help"])
        
        assert exc_info.value.code == 0
        help_text = mock_stdout.getvalue()
        assert "ai-benchmark" in help_text
        assert "benchmark" in help_text
        assert "simulate" in help_text

    def test_command_help(self):
        """Test command-specific help."""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
                self.parser.parse_args(["benchmark", "--help"])
        
        assert exc_info.value.code == 0
        help_text = mock_stdout.getvalue()
        assert "--suite" in help_text
        assert "--strategy" in help_text

    def test_invalid_command(self):
        """Test invalid command handling."""
        with pytest.raises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                self.parser.parse_args(["invalid-command"])

    def test_missing_required_args(self):
        """Test missing required arguments."""
        with pytest.raises(SystemExit):
            with patch('sys.stderr', new_callable=StringIO):
                # Compare command requires baseline and current
                self.parser.parse_args(["compare", "--threshold", "5"])


class TestBenchmarkCommand(unittest.TestCase):
    """Test cases for BenchmarkCommand."""

    def setUp(self):
        """Set up test fixtures."""
        self.command = BenchmarkCommand()

    @patch('benchmark.src.benchmarks.runner.BenchmarkRunner')
    def test_execute_quick_suite(self, mock_runner_class):
        """Test executing quick benchmark suite."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run_suite.return_value = {"status": "success"}
        
        args = argparse.Namespace(
            suite="quick",
            strategy=None,
            duration=300,
            parallel=None,
            metrics=None,
            baseline=None,
            save_baseline=False,
            config="benchmark.yaml",
            format="json",
            output=None
        )
        
        result = self.command.execute(args)
        
        assert result["status"] == "success"
        mock_runner.run_suite.assert_called_once_with("quick")

    @patch('benchmark.src.benchmarks.runner.BenchmarkRunner')
    def test_execute_with_strategies(self, mock_runner_class):
        """Test executing with specific strategies."""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        mock_runner.run_strategies.return_value = {"status": "success"}
        
        args = argparse.Namespace(
            suite=None,
            strategy=["momentum", "arbitrage"],
            duration=600,
            parallel=4,
            metrics="latency,throughput",
            baseline=None,
            save_baseline=False,
            config="benchmark.yaml",
            format="json",
            output=None
        )
        
        result = self.command.execute(args)
        
        assert result["status"] == "success"
        mock_runner.run_strategies.assert_called_once_with(
            ["momentum", "arbitrage"],
            duration=600,
            parallel=4,
            metrics=["latency", "throughput"]
        )

    def test_validate_args_invalid_suite(self):
        """Test argument validation with invalid suite."""
        args = argparse.Namespace(
            suite="invalid",
            strategy=None,
            duration=300,
            parallel=None,
            metrics=None,
            baseline=None,
            save_baseline=False
        )
        
        with pytest.raises(ValueError, match="Invalid suite"):
            self.command.validate_args(args)

    def test_validate_args_negative_duration(self):
        """Test argument validation with negative duration."""
        args = argparse.Namespace(
            suite="quick",
            strategy=None,
            duration=-10,
            parallel=None,
            metrics=None,
            baseline=None,
            save_baseline=False
        )
        
        with pytest.raises(ValueError, match="Duration must be positive"):
            self.command.validate_args(args)


class TestSimulateCommand(unittest.TestCase):
    """Test cases for SimulateCommand."""

    def setUp(self):
        """Set up test fixtures."""
        self.command = SimulateCommand()

    @patch('benchmark.src.simulation.simulator.MarketSimulator')
    def test_execute_historical_simulation(self, mock_simulator_class):
        """Test executing historical simulation."""
        mock_simulator = Mock()
        mock_simulator_class.return_value = mock_simulator
        mock_simulator.run.return_value = {"status": "success"}
        
        args = argparse.Namespace(
            scenario="historical",
            start_date="2024-01-01",
            end_date="2024-12-31",
            assets="AAPL,GOOGL",
            strategies=None,
            capital=100000.0,
            threads=4,
            speed=1.0,
            live=False,
            record=False,
            config="benchmark.yaml",
            format="json",
            output=None
        )
        
        result = self.command.execute(args)
        
        assert result["status"] == "success"
        mock_simulator.run.assert_called_once()

    def test_validate_args_invalid_scenario(self):
        """Test argument validation with invalid scenario."""
        args = argparse.Namespace(
            scenario="invalid",
            start_date="2024-01-01",
            end_date="2024-12-31",
            assets="AAPL",
            capital=100000.0,
            speed=1.0
        )
        
        with pytest.raises(ValueError, match="Invalid scenario"):
            self.command.validate_args(args)

    def test_validate_args_invalid_date_order(self):
        """Test argument validation with invalid date order."""
        args = argparse.Namespace(
            scenario="historical",
            start_date="2024-12-31",
            end_date="2024-01-01",
            assets="AAPL",
            capital=100000.0,
            speed=1.0
        )
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            self.command.validate_args(args)


class TestOptimizeCommand(unittest.TestCase):
    """Test cases for OptimizeCommand."""

    def setUp(self):
        """Set up test fixtures."""
        self.command = OptimizeCommand()

    @patch('benchmark.src.optimization.optimizer.StrategyOptimizer')
    def test_execute_bayesian_optimization(self, mock_optimizer_class):
        """Test executing Bayesian optimization."""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = {"best_params": {"x": 1.0}}
        
        args = argparse.Namespace(
            algorithm="bayesian",
            objective="sharpe",
            constraints=None,
            parameters="benchmark/configs/parameters.yaml",
            trials=100,
            timeout=60,
            parallel=True,
            resume=None,
            config="benchmark.yaml",
            format="json",
            output=None
        )
        
        result = self.command.execute(args)
        
        assert result["best_params"]["x"] == 1.0
        mock_optimizer.optimize.assert_called_once()

    def test_validate_args_invalid_algorithm(self):
        """Test argument validation with invalid algorithm."""
        args = argparse.Namespace(
            algorithm="invalid",
            objective="sharpe",
            parameters="params.json",
            trials=100,
            timeout=60
        )
        
        with pytest.raises(ValueError, match="Invalid algorithm"):
            self.command.validate_args(args)


if __name__ == "__main__":
    unittest.main()