"""Simulate command implementation."""

import click
import json
import yaml
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
from tqdm import tqdm


def simulate_command(ctx, historical: bool, realtime: bool,
                    start_date: Optional[str], end_date: Optional[str],
                    duration: Optional[str], assets: Optional[List[str]],
                    config_file: Optional[str], output: Optional[str],
                    format: str):
    """Execute simulation command."""
    
    # Validate mode
    if not historical and not realtime:
        historical = True  # Default to historical
    
    if historical and realtime:
        click.echo("Error: Cannot use both --historical and --realtime", err=True)
        ctx.exit(1)
    
    # Load config if provided
    config = {}
    if config_file:
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    if historical:
        # Parse dates
        if not start_date and 'simulation' in config:
            start_date = config['simulation'].get('start_date')
        if not end_date and 'simulation' in config:
            end_date = config['simulation'].get('end_date')
            
        if not start_date or not end_date:
            click.echo("Error: --start and --end dates required for historical simulation", err=True)
            ctx.exit(1)
            
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            click.echo("Error: Invalid date format. Use YYYY-MM-DD", err=True)
            ctx.exit(1)
            
        click.echo(f"Simulating from {start_date} to {end_date}")
        results = run_historical_simulation(start_dt, end_dt, assets, config)
        
    else:  # realtime
        if not duration:
            duration = '1h'
        
        click.echo(f"Running real-time simulation for {duration}")
        results = run_realtime_simulation(duration, assets, config)
    
    # Display or save results
    _display_simulation_results(results, format, output)


def run_historical_simulation(start_date: datetime, end_date: datetime,
                            assets: Optional[List[str]], config: Dict) -> Dict[str, Any]:
    """Run historical simulation."""
    
    # Get configuration
    initial_capital = config.get('simulation', {}).get('initial_capital', 10000)
    strategies = config.get('simulation', {}).get('strategies', ['momentum', 'swing'])
    
    if not assets:
        assets = config.get('simulation', {}).get('assets', ['stocks'])
        if isinstance(assets[0], str) and ',' in assets[0]:
            assets = assets[0].split(',')
    
    # Simulate processing
    total_days = (end_date - start_date).days
    results = {
        'type': 'historical',
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'initial_capital': initial_capital,
        'strategies': strategies,
        'assets': assets,
        'daily_results': [],
        'trades': []
    }
    
    current_capital = initial_capital
    
    # Simulate daily processing
    with tqdm(total=total_days, desc="Simulating") as pbar:
        current_date = start_date
        while current_date <= end_date:
            # Simulate daily trading
            daily_pnl = _simulate_daily_trading(current_date, assets, strategies)
            current_capital += daily_pnl
            
            results['daily_results'].append({
                'date': current_date.isoformat(),
                'pnl': daily_pnl,
                'capital': current_capital
            })
            
            # Simulate some trades
            if daily_pnl != 0:
                import random
                # Pick a random asset for the trade
                asset = random.choice(assets) if assets else 'UNKNOWN'
                strategy = random.choice(strategies) if strategies else 'UNKNOWN'
                results['trades'].append({
                    'date': current_date.isoformat(),
                    'asset': asset,
                    'strategy': strategy,
                    'pnl': daily_pnl
                })
            
            current_date += timedelta(days=1)
            pbar.update(1)
    
    # Calculate summary statistics
    results['summary'] = {
        'final_capital': current_capital,
        'total_return': (current_capital - initial_capital) / initial_capital,
        'total_trades': len(results['trades']),
        'winning_trades': sum(1 for t in results['trades'] if t['pnl'] > 0)
    }
    
    return results


def run_realtime_simulation(duration: str, assets: Optional[List[str]], 
                           config: Dict) -> Dict[str, Any]:
    """Run real-time simulation."""
    
    # Parse duration
    if duration.endswith('h'):
        seconds = int(duration[:-1]) * 3600
    elif duration.endswith('m'):
        seconds = int(duration[:-1]) * 60
    else:
        seconds = int(duration[:-1])
    
    results = {
        'type': 'realtime',
        'duration': duration,
        'assets': assets or ['BTC', 'ETH'],
        'ticks': []
    }
    
    # Simulate real-time ticks
    start_time = datetime.now()
    tick_interval = min(1, seconds / 100)  # Max 100 ticks
    
    with tqdm(total=int(seconds / tick_interval), desc="Real-time simulation") as pbar:
        while (datetime.now() - start_time).total_seconds() < seconds:
            # Simulate tick
            tick_data = {
                'timestamp': datetime.now().isoformat(),
                'prices': {asset: 100 + (time.time() % 10) for asset in results['assets']}
            }
            results['ticks'].append(tick_data)
            
            time.sleep(tick_interval)
            pbar.update(1)
    
    return results


def _simulate_daily_trading(date: datetime, assets: List[str], 
                           strategies: List[str]) -> float:
    """Simulate daily trading returns."""
    # Mock implementation - returns random P&L
    import random
    return random.normalvariate(0, 100)


def _display_simulation_results(results: Dict[str, Any], format: str, 
                               output: Optional[str]):
    """Display or save simulation results."""
    
    if format == 'json':
        output_data = json.dumps(results, indent=2)
    elif format == 'csv':
        # Simple CSV for historical results
        if results['type'] == 'historical':
            lines = ['Date,PnL,Capital']
            for daily in results.get('daily_results', []):
                lines.append(f"{daily['date']},{daily['pnl']:.2f},{daily['capital']:.2f}")
            output_data = '\n'.join(lines)
        else:
            output_data = "Timestamp,Asset,Price\n"
            for tick in results.get('ticks', []):
                for asset, price in tick['prices'].items():
                    output_data += f"{tick['timestamp']},{asset},{price}\n"
    else:  # text format
        lines = ["Simulation Results", "=" * 50]
        lines.append(f"Type: {results['type']}")
        
        if results['type'] == 'historical':
            lines.append(f"Period: {results['start_date']} to {results['end_date']}")
            lines.append(f"Assets: {', '.join(results.get('assets', []))}")
            lines.append(f"Strategies: {', '.join(results.get('strategies', []))}")
            if 'summary' in results:
                summary = results['summary']
                lines.append(f"Initial Capital: {results['initial_capital']:,.2f}")
                lines.append(f"Final Capital: {summary['final_capital']:,.2f}")
                lines.append(f"Total Return: {summary['total_return']:.2%}")
                lines.append(f"Total Trades: {summary['total_trades']}")
                lines.append(f"Winning Trades: {summary['winning_trades']}")
                
            # Show some trades if available
            if results.get('trades'):
                lines.append("\nRecent Trades:")
                for trade in results['trades'][-5:]:  # Show last 5 trades
                    lines.append(f"  {trade['date']}: {trade['asset']} - P&L: {trade['pnl']:.2f}")
        else:
            lines.append(f"Duration: {results['duration']}")
            lines.append(f"Assets: {', '.join(results.get('assets', []))}")
            lines.append(f"Total Ticks: {len(results.get('ticks', []))}")
        
        output_data = '\n'.join(lines)
    
    if output:
        with open(output, 'w') as f:
            f.write(output_data)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(output_data)