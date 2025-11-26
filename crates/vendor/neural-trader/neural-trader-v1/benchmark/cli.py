#!/usr/bin/env python3
"""
Benchmark CLI tool for AI News Trading platform.
Provides commands for benchmarking, simulating, optimizing, and reporting.
"""

import click
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Optional, List, Dict, Any

# Set up imports properly
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Import version
try:
    from benchmark import __version__
except ImportError:
    __version__ = "0.1.0"

# Import commands
try:
    from benchmark.src.commands.benchmark_command import benchmark_command
    from benchmark.src.commands.simulate_command import simulate_command
    from benchmark.src.commands.optimize_command import optimize_command
    from benchmark.src.commands.report_command import report_command
    from benchmark.src.commands.neural_command import neural
    from benchmark.src.config import ConfigManager
except ImportError:
    # For testing, use direct imports
    from src.commands.benchmark_command import benchmark_command
    from src.commands.simulate_command import simulate_command
    from src.commands.optimize_command import optimize_command
    from src.commands.report_command import report_command
    from src.commands.neural_command import neural
    from src.config import ConfigManager


class Context:
    """CLI context object to pass configuration and state between commands."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.verbose = False
        self.debug = False


pass_context = click.make_pass_decorator(Context, ensure=True)


@click.group(invoke_without_command=True)
@click.option('--config', type=click.Path(exists=True), help='Configuration file (YAML or JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.version_option(version=__version__, prog_name='benchmark-cli')
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], verbose: bool, debug: bool):
    """
    AI News Trading Benchmark CLI
    
    A comprehensive benchmarking tool for testing and optimizing trading strategies.
    """
    # Initialize context
    ctx.obj = Context()
    ctx.obj.verbose = verbose
    ctx.obj.debug = debug
    
    # Load configuration if provided
    if config:
        ctx.obj.config_manager.load_from_file(config)
    
    # Load environment variables
    ctx.obj.config_manager.load_from_env()
    
    # Show help if no command is provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option('--strategy', '-s', 
              type=click.Choice(['momentum', 'swing', 'mirror', 'all']),
              default='all',
              help='Trading strategy to benchmark')
@click.option('--duration', '-d', 
              default='1h',
              help='Duration for benchmark (e.g., 1h, 30m, 5m)')
@click.option('--assets', '-a',
              default='stocks',
              help='Asset types to benchmark (comma-separated: stocks,bonds,crypto)')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for results')
@click.option('--format', '-f',
              type=click.Choice(['json', 'csv', 'text']),
              default='text',
              help='Output format')
@click.option('--progress', is_flag=True,
              help='Show progress bar during benchmarking')
@click.option('--concurrent', is_flag=True,
              help='Enable concurrent execution')
@click.option('--workers', '-w',
              type=int,
              default=4,
              help='Number of concurrent workers')
@click.option('--cache-dir',
              type=click.Path(),
              help='Directory for caching results')
@click.option('--use-cache', is_flag=True,
              help='Use cached results if available')
@click.option('--profile', is_flag=True,
              help='Enable performance profiling')
@pass_context
def benchmark(ctx: Context, strategy: str, duration: str, assets: str, 
              output: Optional[str], format: str, progress: bool,
              concurrent: bool, workers: int, cache_dir: Optional[str],
              use_cache: bool, profile: bool):
    """Run benchmarks on trading strategies."""
    benchmark_command(
        ctx=ctx,
        strategy=strategy,
        duration=duration,
        assets=assets.split(','),
        output=output,
        format=format,
        progress=progress,
        concurrent=concurrent,
        workers=workers,
        cache_dir=cache_dir,
        use_cache=use_cache,
        profile=profile
    )


@cli.command()
@click.option('--historical', is_flag=True,
              help='Run historical simulation')
@click.option('--realtime', is_flag=True,
              help='Run real-time simulation')
@click.option('--start', '-s',
              help='Start date (YYYY-MM-DD)')
@click.option('--end', '-e',
              help='End date (YYYY-MM-DD)')
@click.option('--duration', '-d',
              help='Duration for real-time simulation')
@click.option('--assets', '-a',
              help='Specific assets to simulate (comma-separated)')
@click.option('--config', '-c',
              type=click.Path(exists=True),
              help='Simulation configuration file')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for results')
@click.option('--format', '-f',
              type=click.Choice(['json', 'csv', 'text']),
              default='text',
              help='Output format')
@pass_context
def simulate(ctx: Context, historical: bool, realtime: bool,
             start: Optional[str], end: Optional[str], duration: Optional[str],
             assets: Optional[str], config: Optional[str],
             output: Optional[str], format: str):
    """Run trading simulations with historical or real-time data."""
    simulate_command(
        ctx=ctx,
        historical=historical,
        realtime=realtime,
        start_date=start,
        end_date=end,
        duration=duration,
        assets=assets.split(',') if assets else None,
        config_file=config,
        output=output,
        format=format
    )


@cli.command()
@click.option('--strategy', '-s',
              type=click.Choice(['momentum', 'swing', 'mirror', 'all']),
              required=True,
              help='Strategy to optimize')
@click.option('--metric', '-m',
              default='sharpe',
              help='Optimization metric (sharpe, returns, sortino, calmar)')
@click.option('--iterations', '-i',
              type=int,
              default=100,
              help='Number of optimization iterations')
@click.option('--max-drawdown',
              type=float,
              help='Maximum drawdown constraint')
@click.option('--min-trades',
              type=int,
              help='Minimum number of trades constraint')
@click.option('--input', '-in',
              type=click.Path(exists=True),
              help='Input data file from simulation')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file for optimized parameters')
@pass_context
def optimize(ctx: Context, strategy: str, metric: str, iterations: int,
             max_drawdown: Optional[float], min_trades: Optional[int],
             input: Optional[str], output: Optional[str]):
    """Optimize trading strategy parameters."""
    metrics = metric.split(',')
    constraints = {}
    if max_drawdown:
        constraints['max_drawdown'] = max_drawdown
    if min_trades:
        constraints['min_trades'] = min_trades
        
    optimize_command(
        ctx=ctx,
        strategy=strategy,
        metrics=metrics,
        iterations=iterations,
        constraints=constraints,
        input_file=input,
        output_file=output
    )


@cli.command()
@click.option('--format', '-f',
              type=click.Choice(['html', 'json', 'pdf', 'text']),
              default='html',
              help='Report format')
@click.option('--output', '-o',
              type=click.Path(),
              help='Output file path (displays to stdout if not provided)')
@click.option('--input', '-i',
              type=click.Path(exists=True),
              help='Input results file')
@click.option('--start', '-s',
              help='Start date for report (YYYY-MM-DD)')
@click.option('--end', '-e',
              help='End date for report (YYYY-MM-DD)')
@pass_context
def report(ctx: Context, format: str, output: str, input: Optional[str],
           start: Optional[str], end: Optional[str]):
    """Generate performance reports."""
    report_command(
        ctx=ctx,
        format=format,
        output_file=output,
        input_file=input,
        start_date=start,
        end_date=end
    )


# Add neural command group
cli.add_command(neural)


def main():
    """Main entry point for the CLI."""
    try:
        cli(prog_name='benchmark-cli')
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()