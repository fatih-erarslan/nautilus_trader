# Repository Structure

This document describes the organized directory structure of the AI News Trader repository.

## Root Directory Structure

```
/
├── src/                    # Source code modules
├── tests/                  # Test files organized by module
├── docs/                   # Documentation
├── config/                 # Configuration files
├── scripts/                # Utility and deployment scripts  
├── benchmarks/             # Performance benchmarks and analysis
├── deployment/             # Deployment configurations
├── results/                # Generated results and analysis data
├── models/                 # Trained models and configurations
├── memory/                 # System memory and cache files
├── plans/                  # Development and implementation plans
└── examples/               # Example usage and demonstrations

```

## Source Code Organization (`src/`)

### Core Modules
- `src/main.py` - Main application entry point
- `src/mcp/` - MCP (Model Context Protocol) server implementation
- `src/neural_forecast/` - Neural forecasting models and predictions
- `src/trading/` - Trading strategies and market operations
- `src/news/` - News collection and analysis modules
- `src/optimization/` - Parameter optimization algorithms
- `src/gpu_data_processing/` - GPU acceleration modules
- `src/trading-platform/` - Symbolic trading platform

### MCP Module (`src/mcp/`)
```
src/mcp/
├── server.py                      # Main MCP server
├── discovery.py                   # Service discovery
├── handlers/                      # Request handlers
│   ├── prompts.py
│   ├── resources.py
│   ├── sampling.py
│   └── tools.py
├── trading/                       # Trading-specific MCP components
│   ├── model_loader.py
│   └── strategy_manager.py
├── mcp_server_official.py         # Official MCP server implementation
├── mcp_server_enhanced.py         # Enhanced server with GPU support
├── mcp_server_claude_optimized.py # Claude-optimized server
└── mcp_server_timeout_fixed.py    # Timeout-fixed server
```

### Trading Module (`src/trading/`)
```
src/trading/
├── strategies/                    # Trading strategy implementations
│   ├── mean_reversion_trader.py
│   ├── momentum_trader.py
│   ├── swing_trader.py
│   ├── mirror_trader.py
│   └── enhanced_*.py             # Enhanced versions with optimizations
├── stocks/                       # Stock market operations
│   ├── data_collector.py
│   ├── indicators.py
│   └── price_levels.py
├── bonds/                        # Bond market operations
│   ├── etf_analyzer.py
│   ├── yield_collector.py
│   └── yield_curve.py
├── allocation/                   # Portfolio allocation
└── performance/                  # Performance tracking and analysis
```

### News Module (`src/news/`)
```
src/news/
├── models.py                     # News data models
├── sources/                      # News source implementations
│   ├── reuters.py
│   ├── yahoo_finance.py
│   ├── federal_reserve.py
│   ├── bond_market.py
│   └── treasury.py
└── trading_strategies/           # News-based trading strategies
```

## Test Organization (`tests/`)

Tests are organized to mirror the source code structure:

```
tests/
├── mcp/                          # MCP server and protocol tests
├── neural/                       # Neural forecast model tests
├── trading/                      # Trading strategy tests
│   ├── strategies/               # Individual strategy tests
│   ├── stocks/                   # Stock market operation tests
│   ├── bonds/                    # Bond market operation tests
│   ├── allocation/               # Portfolio allocation tests
│   └── performance/              # Performance tracking tests
└── news/                         # News collection and analysis tests
```

## Documentation (`docs/`)

```
docs/
├── api/                          # API documentation
├── guides/                       # User and developer guides
├── implementation/               # Implementation details and reports
├── mcp/                         # MCP-specific documentation
├── optimization/                # Optimization reports and analysis
├── reports/                     # Performance and analysis reports
└── REPOSITORY_STRUCTURE.md      # This file
```

## Configuration (`config/`)

```
config/
├── mcp/                         # MCP server configurations
│   ├── requirements-mcp-official.txt
│   └── requirements-mcp.txt
├── trading/                     # Trading system configurations
│   └── benchmark.yaml
└── deployment/                  # Deployment configurations
```

## Scripts (`scripts/`)

Utility scripts for development and deployment:

```
scripts/
├── start_mcp_server.py          # MCP server launcher
├── start_mcp_server_fastmcp.py  # FastMCP server launcher
├── claude-monitor.py            # System monitoring
├── debug_timeframe.py           # Debugging utilities
├── mcp_client_example.py        # MCP client example
├── validate_mcp_working.py      # MCP validation
└── memory_store_command.txt     # Memory management commands
```

## Results (`results/`)

Generated results and analysis data:

```
results/
├── analysis/                    # Analysis results
├── optimization/                # Optimization results
├── parameters/                  # Parameter configurations
├── memory/                      # Memory and state files
└── session_*/                   # Session-based results
```

## Models (`models/`)

Trained models and configurations:

```
models/
├── all_optimized_models.json    # Complete model registry
├── deployment_manifest.json     # Deployment configuration
├── mcp_config.json              # MCP model configuration
├── mean_reversion_optimized.json
├── mirror_trading_optimized.json
├── momentum_trading_optimized.json
└── swing_trading_optimized.json
```

## Benchmarks (`benchmarks/`)

Performance benchmarking and analysis tools. This is a comprehensive benchmarking suite with its own internal structure for testing system performance, strategy optimization, and analysis validation.

## Migration Notes

### File Moves Performed
1. **Documentation**: All loose `.md` files moved from root to appropriate `docs/` subdirectories
2. **MCP Servers**: Moved from root to `src/mcp/`
3. **Analysis Scripts**: Moved from root to `benchmarks/`
4. **Configuration**: Moved to `config/` with appropriate subdirectories
5. **Results**: Organized into `results/` with categorized subdirectories
6. **Scripts**: Utility scripts moved to `scripts/`

### Import Path Updates
- Test files updated to reference new MCP server locations
- Start scripts updated with correct path resolution
- All import paths validated for new structure

### Duplicates Removed
- Consolidated duplicate `symbolic_trading` directories
- Removed redundant temporary files
- Organized similar functionality under common directories

## Backwards Compatibility

The reorganization maintains backwards compatibility by:
- Preserving all existing functionality
- Updating import paths automatically
- Maintaining git history for moved files
- Providing clear migration path documentation

## Development Workflow

With the new structure:
1. **Source Code**: All development happens in `src/`
2. **Testing**: Tests mirror source structure in `tests/`
3. **Documentation**: All docs consolidated in `docs/`
4. **Configuration**: Centralized in `config/`
5. **Deployment**: Scripts and configs in appropriate directories
6. **Results**: Generated data organized in `results/`

This structure provides better organization, easier navigation, and clearer separation of concerns for the AI News Trader system.