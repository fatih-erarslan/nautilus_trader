#!/usr/bin/env python3
"""
Test runner script for Alpaca trading system.
"""

import sys
import subprocess
import argparse


def run_tests(test_type='all', verbose=False, coverage=False):
    """Run tests with specified options."""
    cmd = ['pytest']
    
    if verbose:
        cmd.append('-vv')
    else:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html'])
    
    if test_type == 'unit':
        cmd.extend(['-m', 'not integration'])
    elif test_type == 'integration':
        cmd.extend(['-m', 'integration'])
    elif test_type == 'websocket':
        cmd.append('tests/alpaca_trading/test_websocket_client.py')
    elif test_type == 'strategies':
        cmd.append('tests/alpaca_trading/test_strategies.py')
    elif test_type == 'execution':
        cmd.append('tests/alpaca_trading/test_execution.py')
    elif test_type == 'risk':
        cmd.append('tests/alpaca_trading/test_risk_management.py')
    elif test_type == 'monitoring':
        cmd.extend([
            'tests/alpaca_trading/test_websocket_client.py',
            'tests/alpaca_trading/test_strategies.py',
            'tests/alpaca_trading/test_execution.py',
            'tests/alpaca_trading/test_risk_management.py'
        ])
    elif test_type == 'full':
        cmd.append('tests/alpaca_trading/integration_test.py')
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description='Run Alpaca trading tests')
    parser.add_argument(
        'type',
        choices=['all', 'unit', 'integration', 'websocket', 'strategies', 
                 'execution', 'risk', 'monitoring', 'full'],
        default='all',
        nargs='?',
        help='Type of tests to run'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    
    args = parser.parse_args()
    
    sys.exit(run_tests(args.type, args.verbose, args.coverage))


if __name__ == '__main__':
    main()