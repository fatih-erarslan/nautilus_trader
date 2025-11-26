#!/usr/bin/env python3
"""
AI News Trading Platform Benchmark CLI

A comprehensive benchmarking tool for measuring performance, running simulations,
and optimizing trading strategies.

Usage:
    ./benchmark_cli.py [global-options] <command> [command-options]

Commands:
    benchmark   Run performance benchmarks
    simulate    Run market simulations
    optimize    Run optimization algorithms
    report      Generate performance reports
    profile     Detailed profiling
    compare     Compare multiple runs

Examples:
    # Run quick benchmark suite
    ./benchmark_cli.py benchmark --suite quick
    
    # Run historical simulation
    ./benchmark_cli.py simulate --scenario historical --start-date 2024-01-01 --end-date 2024-12-31
    
    # Optimize strategy parameters
    ./benchmark_cli.py optimize --algorithm bayesian --objective sharpe --parameters params.yaml
    
    # Generate HTML report
    ./benchmark_cli.py report --type dashboard --input results.json --export html

For more information on a specific command:
    ./benchmark_cli.py <command> --help
"""

import sys
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.src.cli.commands import main

if __name__ == "__main__":
    main()