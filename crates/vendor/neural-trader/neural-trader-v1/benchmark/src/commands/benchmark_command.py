"""Benchmark command implementation."""

import click
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import concurrent.futures
from pathlib import Path


def parse_duration(duration_str: str) -> timedelta:
    """Parse duration string to timedelta."""
    if duration_str.endswith('h'):
        return timedelta(hours=int(duration_str[:-1]))
    elif duration_str.endswith('m'):
        return timedelta(minutes=int(duration_str[:-1]))
    elif duration_str.endswith('s'):
        return timedelta(seconds=int(duration_str[:-1]))
    else:
        raise ValueError(f"Invalid duration format: {duration_str}")


def benchmark_strategy(strategy: str, duration: timedelta, assets: List[str], 
                      progress_callback=None) -> Dict[str, Any]:
    """Benchmark a single strategy."""
    start_time = datetime.now()
    
    # Simulate benchmarking work
    total_steps = max(1, int(duration.total_seconds() / 10))  # One step per 10 seconds, min 1
    results = {
        'strategy': strategy,
        'assets': assets,
        'metrics': {
            'total_trades': 0,
            'winning_trades': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_return': 0.0
        },
        'execution_time': 0
    }
    
    # Simulate processing
    for i in range(total_steps):
        time.sleep(0.01)  # Simulate work
        
        # Update metrics (mock data)
        results['metrics']['total_trades'] += 5
        results['metrics']['winning_trades'] += 3
        results['metrics']['sharpe_ratio'] = 1.5 + (i * 0.1)
        results['metrics']['max_drawdown'] = -0.15
        results['metrics']['total_return'] = 0.25 + (i * 0.05)
        
        if progress_callback:
            progress_callback(i + 1, total_steps)
    
    results['execution_time'] = (datetime.now() - start_time).total_seconds()
    return results


def benchmark_command(ctx, strategy: str, duration: str, assets: List[str],
                     output: Optional[str], format: str, progress: bool,
                     concurrent: bool, workers: int, cache_dir: Optional[str],
                     use_cache: bool, profile: bool):
    """Execute benchmark command."""
    
    # Parse duration
    try:
        duration_td = parse_duration(duration)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
    
    # Check cache
    cache_key = f"{strategy}_{duration}_{'-'.join(sorted(assets))}"
    if use_cache and cache_dir:
        cache_file = Path(cache_dir) / f"{cache_key}.json"
        if cache_file.exists():
            click.echo(f"Using cached results from {cache_file}")
            with open(cache_file, 'r') as f:
                results = json.load(f)
                _display_results(results, format, output)
                return
    
    # Determine strategies to benchmark
    strategies = ['momentum', 'swing', 'mirror'] if strategy == 'all' else [strategy]
    
    click.echo(f"Benchmarking {strategy} strategy")
    click.echo(f"Duration: {duration}")
    click.echo(f"Assets: {', '.join(assets)}")
    
    if profile:
        click.echo("Performance profiling enabled")
    
    results = {'strategies': {}}
    
    if concurrent and len(strategies) > 1:
        click.echo(f"Running benchmarks concurrently with {workers} workers")
        
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for strat in strategies:
                future = executor.submit(benchmark_strategy, strat, duration_td, assets)
                futures[future] = strat
            
            # Process completed futures
            for future in cf.as_completed(futures):
                strat = futures[future]
                try:
                    result = future.result()
                    results['strategies'][strat] = result
                    click.echo(f"Completed: {strat}")
                except Exception as e:
                    click.echo(f"Error in {strat}: {e}", err=True)
    else:
        # Sequential execution
        for strat in strategies:
            if progress:
                with tqdm(total=100, desc=f"Benchmarking {strat}") as pbar:
                    def update_progress(current, total):
                        pbar.n = int((current / total) * 100)
                        pbar.refresh()
                    
                    result = benchmark_strategy(strat, duration_td, assets, update_progress)
                    results['strategies'][strat] = result
                    pbar.n = 100
                    pbar.refresh()
            else:
                click.echo(f"Benchmarking {strat}...")
                result = benchmark_strategy(strat, duration_td, assets)
                results['strategies'][strat] = result
                click.echo(f"Completed: {strat}")
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'duration': duration,
        'assets': assets,
        'concurrent': concurrent,
        'workers': workers if concurrent else 1
    }
    
    # Add results key for backward compatibility
    results['results'] = results['strategies']
    results['metrics'] = {
        'total_strategies': len(results['strategies']),
        'execution_mode': 'concurrent' if concurrent else 'sequential'
    }
    
    # Cache results if requested
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = Path(cache_dir) / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results cached to {cache_file}")
    
    # Display or save results
    _display_results(results, format, output)
    
    click.echo("Benchmark completed successfully")


def _display_results(results: Dict[str, Any], format: str, output: Optional[str]):
    """Display or save benchmark results."""
    
    if format == 'json':
        output_data = json.dumps(results, indent=2)
    elif format == 'csv':
        # Simple CSV format
        lines = ['Strategy,Total Trades,Winning Trades,Sharpe Ratio,Max Drawdown,Total Return']
        for strat, data in results['strategies'].items():
            metrics = data['metrics']
            lines.append(f"{strat},{metrics['total_trades']},{metrics['winning_trades']},"
                        f"{metrics['sharpe_ratio']:.2f},{metrics['max_drawdown']:.2%},"
                        f"{metrics['total_return']:.2%}")
        output_data = '\n'.join(lines)
    else:  # text format
        lines = ["Benchmark Results", "=" * 50]
        for strat, data in results['strategies'].items():
            lines.append(f"\nStrategy: {strat}")
            lines.append("-" * 30)
            metrics = data['metrics']
            lines.append(f"Total Trades: {metrics['total_trades']}")
            lines.append(f"Winning Trades: {metrics['winning_trades']}")
            lines.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            lines.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            lines.append(f"Total Return: {metrics['total_return']:.2%}")
            lines.append(f"Execution Time: {data['execution_time']:.2f}s")
        output_data = '\n'.join(lines)
    
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(output_data)