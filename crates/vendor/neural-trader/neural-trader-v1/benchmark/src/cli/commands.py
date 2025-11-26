"""CLI command implementations for benchmark tool."""

import argparse
import json
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .config import ConfigManager


class BaseCommand(ABC):
    """Base class for CLI commands."""
    
    @abstractmethod
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute the command."""
        pass
    
    @abstractmethod
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate command arguments."""
        pass


class BenchmarkCommand(BaseCommand):
    """Benchmark command implementation."""
    
    VALID_SUITES = ["quick", "standard", "comprehensive", "custom"]
    VALID_METRICS = ["latency", "throughput", "memory", "cpu", "io"]
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute benchmark command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from ..benchmarks.runner import BenchmarkRunner
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create benchmark runner
        runner = BenchmarkRunner(config)
        
        # Run benchmarks based on arguments
        if args.suite:
            results = runner.run_suite(args.suite)
        elif args.strategy:
            metrics = args.metrics.split(",") if args.metrics else None
            results = runner.run_strategies(
                args.strategy,
                duration=args.duration,
                parallel=args.parallel,
                metrics=metrics
            )
        else:
            # Default to standard suite
            results = runner.run_suite("standard")
        
        # Handle baseline comparison
        if args.baseline:
            results = self._compare_with_baseline(results, args.baseline)
        
        # Save as new baseline if requested
        if args.save_baseline:
            self._save_baseline(results)
        
        return results
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate benchmark command arguments."""
        if args.suite and args.suite not in self.VALID_SUITES:
            raise ValueError(f"Invalid suite: {args.suite}. Valid options: {self.VALID_SUITES}")
        
        if args.duration <= 0:
            raise ValueError("Duration must be positive")
        
        if args.parallel is not None and args.parallel < 1:
            raise ValueError("Parallel workers must be at least 1")
        
        if args.metrics:
            invalid_metrics = set(args.metrics.split(",")) - set(self.VALID_METRICS)
            if invalid_metrics:
                raise ValueError(f"Invalid metrics: {invalid_metrics}")
    
    def _compare_with_baseline(self, results: Dict[str, Any], baseline_path: str) -> Dict[str, Any]:
        """Compare results with baseline."""
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
        
        # Add comparison data
        results["baseline_comparison"] = {
            "baseline_file": baseline_path,
            "improvements": {},
            "regressions": {}
        }
        
        return results
    
    def _save_baseline(self, results: Dict[str, Any]) -> None:
        """Save results as new baseline."""
        baseline_path = Path("baseline.json")
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2)


class SimulateCommand(BaseCommand):
    """Simulate command implementation."""
    
    VALID_SCENARIOS = ["historical", "synthetic", "stress-test"]
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute simulation command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from ..simulation.simulator import MarketSimulator
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create simulator
        simulator = MarketSimulator(config)
        
        # Parse assets
        assets = args.assets.split(",") if args.assets else []
        
        # Run simulation
        results = simulator.run(
            scenario=args.scenario,
            start_date=args.start_date,
            end_date=args.end_date,
            assets=assets,
            strategies=args.strategies,
            capital=args.capital,
            threads=args.threads,
            speed_factor=args.speed,
            live=args.live,
            record=args.record
        )
        
        return results
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate simulation command arguments."""
        if args.scenario not in self.VALID_SCENARIOS:
            raise ValueError(f"Invalid scenario: {args.scenario}. Valid options: {self.VALID_SCENARIOS}")
        
        # Validate dates for historical scenario
        if args.scenario == "historical":
            try:
                start = datetime.fromisoformat(args.start_date)
                end = datetime.fromisoformat(args.end_date)
                if start >= end:
                    raise ValueError("Start date must be before end date")
            except ValueError as e:
                if "Invalid isoformat string" in str(e):
                    raise ValueError("Invalid date format. Use YYYY-MM-DD")
                raise
        
        if args.capital <= 0:
            raise ValueError("Capital must be positive")
        
        if args.speed <= 0:
            raise ValueError("Speed factor must be positive")


class OptimizeCommand(BaseCommand):
    """Optimize command implementation."""
    
    VALID_ALGORITHMS = ["grid", "random", "bayesian", "genetic", "ml"]
    VALID_OBJECTIVES = ["sharpe", "returns", "drawdown", "custom"]
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute optimization command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from ..optimization.optimizer import StrategyOptimizer
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create optimizer
        optimizer = StrategyOptimizer(config, algorithm=args.algorithm)
        
        # Load parameters
        parameters = self._load_parameters(args.parameters)
        
        # Load constraints if provided
        constraints = self._load_constraints(args.constraints) if args.constraints else None
        
        # Run optimization
        results = optimizer.optimize(
            objective=args.objective,
            parameters=parameters,
            constraints=constraints,
            trials=args.trials,
            timeout_minutes=args.timeout,
            parallel=args.parallel,
            resume_from=args.resume
        )
        
        return results
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate optimization command arguments."""
        if args.algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {args.algorithm}. Valid options: {self.VALID_ALGORITHMS}")
        
        if args.objective not in self.VALID_OBJECTIVES:
            raise ValueError(f"Invalid objective: {args.objective}. Valid options: {self.VALID_OBJECTIVES}")
        
        if args.trials <= 0:
            raise ValueError("Trials must be positive")
        
        if args.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if not Path(args.parameters).exists():
            raise ValueError(f"Parameters file not found: {args.parameters}")
    
    def _load_parameters(self, path: str) -> Dict[str, Any]:
        """Load parameter definitions from file."""
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def _load_constraints(self, path: str) -> Dict[str, Any]:
        """Load constraint definitions from file."""
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)


class ReportCommand(BaseCommand):
    """Report command implementation."""
    
    VALID_TYPES = ["summary", "detailed", "comparison", "dashboard"]
    VALID_EXPORTS = ["pdf", "html", "markdown", "excel"]
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute report command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from .reporters import ReportGenerator
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create report generator
        generator = ReportGenerator(config)
        
        # Load input data
        data = self._load_input_data(args.input)
        
        # Generate report
        report = generator.generate(
            report_type=args.type,
            data=data,
            template=args.template,
            include_charts=args.charts
        )
        
        # Export if requested
        if args.export:
            output_path = generator.export(report, format=args.export)
            report["exported_to"] = str(output_path)
        
        # Serve dashboard if requested
        if args.serve:
            generator.serve_dashboard(report, port=args.serve)
            report["serving_on"] = f"http://localhost:{args.serve}"
        
        return report
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate report command arguments."""
        if args.type not in self.VALID_TYPES:
            raise ValueError(f"Invalid report type: {args.type}. Valid options: {self.VALID_TYPES}")
        
        if args.export and args.export not in self.VALID_EXPORTS:
            raise ValueError(f"Invalid export format: {args.export}. Valid options: {self.VALID_EXPORTS}")
        
        if args.serve and (args.serve < 1024 or args.serve > 65535):
            raise ValueError("Port must be between 1024 and 65535")
        
        for input_file in args.input:
            if not Path(input_file).exists():
                raise ValueError(f"Input file not found: {input_file}")
    
    def _load_input_data(self, input_files: List[str]) -> List[Dict[str, Any]]:
        """Load data from input files."""
        data = []
        for file_path in input_files:
            with open(file_path, "r") as f:
                if file_path.endswith(".json"):
                    data.append(json.load(f))
                elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                    data.append(yaml.safe_load(f))
        return data


class ProfileCommand(BaseCommand):
    """Profile command implementation."""
    
    VALID_TARGETS = ["cpu", "memory", "io", "network", "all"]
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute profiling command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from ..profiling.profiler import SystemProfiler
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create profiler
        profiler = SystemProfiler(config)
        
        # Run profiling
        results = profiler.profile(
            target=args.target,
            component=args.component,
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            generate_flame_graph=args.flame_graph
        )
        
        return results
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate profile command arguments."""
        if args.target not in self.VALID_TARGETS:
            raise ValueError(f"Invalid target: {args.target}. Valid options: {self.VALID_TARGETS}")
        
        if args.duration <= 0:
            raise ValueError("Duration must be positive")
        
        if args.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")


class CompareCommand(BaseCommand):
    """Compare command implementation."""
    
    def execute(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Execute comparison command."""
        self.validate_args(args)
        
        # Import here to avoid circular imports
        from ..analysis.comparator import ResultComparator
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create comparator
        comparator = ResultComparator(config)
        
        # Parse metrics
        metrics = args.metrics.split(",") if args.metrics else None
        
        # Run comparison
        results = comparator.compare(
            baseline_path=args.baseline,
            current_path=args.current,
            threshold_pct=args.threshold,
            metrics=metrics,
            visualize=args.visualize
        )
        
        return results
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate compare command arguments."""
        if not Path(args.baseline).exists():
            raise ValueError(f"Baseline file not found: {args.baseline}")
        
        if not Path(args.current).exists():
            raise ValueError(f"Current file not found: {args.current}")
        
        if args.threshold < 0 or args.threshold > 100:
            raise ValueError("Threshold must be between 0 and 100")


class CLIParser:
    """Main CLI parser for benchmark tool."""
    
    def __init__(self):
        """Initialize CLI parser."""
        self.parser = self._create_parser()
        self.commands = {
            "benchmark": BenchmarkCommand(),
            "simulate": SimulateCommand(),
            "optimize": OptimizeCommand(),
            "report": ReportCommand(),
            "profile": ProfileCommand(),
            "compare": CompareCommand(),
        }
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="ai-benchmark",
            description="AI News Trading Platform Benchmark Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument(
            "--config",
            default="benchmark.yaml",
            help="Configuration file (default: benchmark.yaml)"
        )
        parser.add_argument(
            "--format",
            choices=["json", "csv", "html", "terminal"],
            default="terminal",
            help="Output format (default: terminal)"
        )
        parser.add_argument(
            "--output",
            help="Output file (default: stdout)"
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable profiling"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode"
        )
        
        # Mutual exclusion for quiet and verbose
        verbosity_group = parser.add_mutually_exclusive_group()
        verbosity_group.add_argument(
            "--verbose", "-v",
            action="count",
            default=0,
            help="Increase verbosity (can be repeated)"
        )
        verbosity_group.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Suppress non-essential output"
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser(
            "benchmark",
            help="Run performance benchmarks"
        )
        benchmark_parser.add_argument(
            "--suite",
            choices=["quick", "standard", "comprehensive", "custom"],
            help="Test suite to run"
        )
        benchmark_parser.add_argument(
            "--strategy",
            action="append",
            help="Strategy to benchmark (can be repeated)"
        )
        benchmark_parser.add_argument(
            "--duration",
            type=int,
            default=300,
            help="Test duration in seconds (default: 300)"
        )
        benchmark_parser.add_argument(
            "--parallel",
            type=int,
            help="Number of parallel workers"
        )
        benchmark_parser.add_argument(
            "--metrics",
            help="Comma-separated metrics to collect"
        )
        benchmark_parser.add_argument(
            "--baseline",
            help="Compare against baseline results"
        )
        benchmark_parser.add_argument(
            "--save-baseline",
            action="store_true",
            help="Save results as new baseline"
        )
        
        # Simulate command
        simulate_parser = subparsers.add_parser(
            "simulate",
            help="Run market simulations"
        )
        simulate_parser.add_argument(
            "--scenario",
            choices=["historical", "synthetic", "stress-test"],
            required=True,
            help="Market scenario"
        )
        simulate_parser.add_argument(
            "--start-date",
            help="Simulation start date (YYYY-MM-DD)"
        )
        simulate_parser.add_argument(
            "--end-date",
            help="Simulation end date (YYYY-MM-DD)"
        )
        simulate_parser.add_argument(
            "--assets",
            help="Asset list (comma-separated symbols)"
        )
        simulate_parser.add_argument(
            "--strategies",
            action="append",
            help="Strategies to simulate (can be repeated)"
        )
        simulate_parser.add_argument(
            "--capital",
            type=float,
            default=100000.0,
            help="Starting capital (default: 100000)"
        )
        simulate_parser.add_argument(
            "--threads",
            type=int,
            help="Number of simulation threads"
        )
        simulate_parser.add_argument(
            "--speed",
            type=float,
            default=1.0,
            help="Simulation speed factor (default: 1.0)"
        )
        simulate_parser.add_argument(
            "--live",
            action="store_true",
            help="Connect to live data feeds"
        )
        simulate_parser.add_argument(
            "--record",
            action="store_true",
            help="Record simulation for replay"
        )
        
        # Optimize command
        optimize_parser = subparsers.add_parser(
            "optimize",
            help="Run optimization algorithms"
        )
        optimize_parser.add_argument(
            "--algorithm",
            choices=["grid", "random", "bayesian", "genetic", "ml"],
            required=True,
            help="Optimization algorithm"
        )
        optimize_parser.add_argument(
            "--objective",
            choices=["sharpe", "returns", "drawdown", "custom"],
            required=True,
            help="Objective function"
        )
        optimize_parser.add_argument(
            "--constraints",
            help="Constraints definition file"
        )
        optimize_parser.add_argument(
            "--parameters",
            required=True,
            help="Parameters to optimize (YAML/JSON)"
        )
        optimize_parser.add_argument(
            "--trials",
            type=int,
            default=100,
            help="Number of optimization trials (default: 100)"
        )
        optimize_parser.add_argument(
            "--timeout",
            type=int,
            default=60,
            help="Optimization timeout in minutes (default: 60)"
        )
        optimize_parser.add_argument(
            "--parallel",
            action="store_true",
            help="Enable parallel optimization"
        )
        optimize_parser.add_argument(
            "--resume",
            help="Resume from previous optimization"
        )
        
        # Report command
        report_parser = subparsers.add_parser(
            "report",
            help="Generate performance reports"
        )
        report_parser.add_argument(
            "--type",
            choices=["summary", "detailed", "comparison", "dashboard"],
            default="summary",
            help="Report type (default: summary)"
        )
        report_parser.add_argument(
            "--input",
            action="append",
            required=True,
            help="Input data files (can be repeated)"
        )
        report_parser.add_argument(
            "--template",
            help="Report template"
        )
        report_parser.add_argument(
            "--charts",
            action="store_true",
            help="Include charts and visualizations"
        )
        report_parser.add_argument(
            "--export",
            choices=["pdf", "html", "markdown", "excel"],
            help="Export format"
        )
        report_parser.add_argument(
            "--serve",
            type=int,
            help="Serve interactive dashboard on port"
        )
        
        # Profile command
        profile_parser = subparsers.add_parser(
            "profile",
            help="Detailed profiling"
        )
        profile_parser.add_argument(
            "--target",
            choices=["cpu", "memory", "io", "network", "all"],
            required=True,
            help="Profiling target"
        )
        profile_parser.add_argument(
            "--component",
            help="Specific component to profile"
        )
        profile_parser.add_argument(
            "--duration",
            type=int,
            default=60,
            help="Profiling duration in seconds (default: 60)"
        )
        profile_parser.add_argument(
            "--sampling-rate",
            type=int,
            default=100,
            help="Sampling rate in Hz (default: 100)"
        )
        profile_parser.add_argument(
            "--flame-graph",
            action="store_true",
            help="Generate flame graphs"
        )
        
        # Compare command
        compare_parser = subparsers.add_parser(
            "compare",
            help="Compare multiple runs"
        )
        compare_parser.add_argument(
            "--baseline",
            required=True,
            help="Baseline results file"
        )
        compare_parser.add_argument(
            "--current",
            required=True,
            help="Current results file"
        )
        compare_parser.add_argument(
            "--threshold",
            type=float,
            default=5.0,
            help="Regression threshold percentage (default: 5.0)"
        )
        compare_parser.add_argument(
            "--metrics",
            help="Metrics to compare"
        )
        compare_parser.add_argument(
            "--visualize",
            action="store_true",
            help="Generate comparison visualizations"
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments."""
        parsed_args = self.parser.parse_args(args)
        
        # Check for mutually exclusive options
        if parsed_args.quiet and parsed_args.verbose:
            self.parser.error("--quiet and --verbose are mutually exclusive")
        
        return parsed_args
    
    def execute(self, args: Optional[List[str]] = None) -> int:
        """Parse arguments and execute command."""
        parsed_args = self.parse_args(args)
        
        # Check if a command was specified
        if not parsed_args.command:
            self.parser.print_help()
            return 1
        
        # Get the command handler
        command = self.commands.get(parsed_args.command)
        if not command:
            self.parser.error(f"Unknown command: {parsed_args.command}")
            return 1
        
        try:
            # Execute the command
            result = command.execute(parsed_args)
            
            # Format and output results
            self._output_results(result, parsed_args)
            
            return 0
        except Exception as e:
            if parsed_args.debug:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _output_results(self, results: Dict[str, Any], args: argparse.Namespace) -> None:
        """Output results in the requested format."""
        if args.format == "json":
            output = json.dumps(results, indent=2)
        elif args.format == "csv":
            # Simplified CSV output
            output = self._format_as_csv(results)
        elif args.format == "html":
            output = self._format_as_html(results)
        else:  # terminal
            output = self._format_as_terminal(results)
        
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
        else:
            print(output)
    
    def _format_as_csv(self, results: Dict[str, Any]) -> str:
        """Format results as CSV."""
        # Simplified implementation
        lines = ["metric,value"]
        for key, value in results.items():
            if isinstance(value, (int, float, str)):
                lines.append(f"{key},{value}")
        return "\n".join(lines)
    
    def _format_as_html(self, results: Dict[str, Any]) -> str:
        """Format results as HTML."""
        # Simplified implementation
        html = "<html><body><h1>Benchmark Results</h1><pre>"
        html += json.dumps(results, indent=2)
        html += "</pre></body></html>"
        return html
    
    def _format_as_terminal(self, results: Dict[str, Any]) -> str:
        """Format results for terminal output."""
        # Simplified implementation
        output = "=" * 50 + "\n"
        output += "BENCHMARK RESULTS\n"
        output += "=" * 50 + "\n\n"
        
        for key, value in results.items():
            output += f"{key}: {value}\n"
        
        return output


def main():
    """Main entry point for CLI."""
    parser = CLIParser()
    sys.exit(parser.execute())