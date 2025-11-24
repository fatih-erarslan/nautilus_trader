# Tengri Trading System

## Overview

The Tengri Trading System is a comprehensive algorithmic trading platform that leverages advanced data analysis, machine learning, and quantum-inspired algorithms to generate trading signals and execute trades.

This system consists of five core applications:

1. **CDFA (Cognitive Diversity Fusion Analysis)**: Advanced market data analysis using wavelets, neuromorphic computing, and cross-asset correlations.
2. **RL (Reinforcement Learning)**: Adaptive trading strategy using Q* learning to optimize trade decisions over time.
3. **Decision**: Quantum-inspired decision making framework for evaluating trading opportunities.
4. **Pairlist**: Dynamic trading pair selection based on volume, volatility, and market conditions.
5. **Optimization**: Model performance optimization using hardware acceleration, TorchScript, and specialized algorithms.

## Integration

The `tengri_integration.py` script demonstrates how all five applications work together as a complete trading system. It follows this workflow:

1. Generate optimal trading pairs
2. Optimize models for performance
3. Analyze market data for patterns
4. Update RL agent with new market information
5. Generate trading decisions
6. Process or simulate trades

## Requirements

- Python 3.8+
- Redis server (optional, for inter-component communication)
- PyTorch (for optimization and ML components)
- FreqTrade (to execute actual trades)

## Configuration

The system is configured via a JSON file at `config/tengri_integration.json`. Key configuration sections:

- `redis`: Settings for Redis communication
- `api`: Base API URLs for services
- `pairlist`: Parameters for pair selection
- `optimization`: Hardware acceleration and optimization settings
- `cdfa`: Market analysis parameters
- `rl`: Reinforcement learning hyperparameters
- `decision`: Risk management and position sizing
- `market_data`: Data fetching parameters

## Usage

To start the integrated system:

```bash
python tengri_integration.py --config /path/to/config.json
```

Options:
- `--config`: Path to configuration file (optional)
- `--debug`: Enable debug mode with additional logging

## Architecture

The Tengri system follows a modular architecture with clear separation of concerns:

1. **Core Logic**: Each app has a core module implementing the business logic
2. **Client Interfaces**: Client modules provide APIs to interact with the core functionality
3. **Servers**: FastAPI servers expose the functionality via RESTful APIs
4. **Integration**: The integration layer coordinates the workflow between components

## Performance Optimization

The system leverages several performance optimization techniques:

- Hardware acceleration (GPU/CPU) for compute-intensive tasks
- TorchScript compilation for PyTorch models
- Caching of market data and analysis results
- Asynchronous processing with threading
- Redis pub/sub for efficient inter-component communication

## Contributing

Please consult the individual README files for each component for detailed documentation on extending or modifying functionality.

## License

Proprietary - All rights reserved.