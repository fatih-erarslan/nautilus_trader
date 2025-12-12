# Nautilus MCP Server

**High-Performance Trading Analytics MCP Server**

Version 1.0.0 | Rust-Bun.js Architecture | HyperPhysics Integration

## Overview

Nautilus MCP is a blazingly fast Model Context Protocol server for algorithmic trading analytics, providing:

- **Technical Indicators**: 37+ indicators (MA, RSI, MACD, Bollinger, ATR)
- **Risk Analytics**: VaR, CVaR, Kelly Criterion, position sizing
- **Portfolio Metrics**: Sharpe, Sortino, Calmar, win rate, expectancy
- **Execution Analysis**: VWAP slippage, order flow imbalance
- **Regime Detection**: pBit dynamics, Ising model, hyperbolic embeddings
- **Conformal Prediction**: Guaranteed-coverage uncertainty quantification
- **Options Greeks**: Delta, Gamma, Theta, Vega, Black-Scholes

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Bun.js Runtime                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  MCP SDK    │  │   Tools     │  │  Fallback   │          │
│  │  Transport  │  │   Router    │  │   Impls     │          │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘          │
│         │                │                                   │
│         └────────────────┼───────────────────────────────────┤
│                          │                                   │
│                    ┌─────▼─────┐                            │
│                    │   NAPI    │                            │
│                    │  Bindings │                            │
│                    └─────┬─────┘                            │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Native Rust Module                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Indicators │ Risk │ Portfolio │ Execution │ Greeks   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  HyperPhysics: pBit │ Lorentz │ Ising │ Conformal    │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Performance

| Operation | Target Latency |
|-----------|----------------|
| EMA update | <1μs |
| RSI computation | <5μs |
| VaR (parametric) | <10μs |
| Kelly sizing | <5μs |
| pBit sampling | <10μs |

## Installation

```bash
# Install dependencies
bun install

# Build TypeScript
bun run build

# Build native Rust module (recommended)
bun run build:native

# Build all
bun run build:all
```

## Usage

```bash
# Start server
bun run start

# Development mode
bun run dev
```

### MCP Configuration

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "nautilus": {
      "command": "bun",
      "args": ["run", "/path/to/nautilus-mcp/dist/index.js"]
    }
  }
}
```

## Tool Categories

### Indicators (37+ tools)
- **Moving Averages**: SMA, EMA, WMA, HMA, DEMA, VWAP
- **Momentum**: RSI, MACD, CCI, Stochastics, ROC, OBV
- **Volatility**: ATR, Bollinger, Keltner, Donchian

### Risk (15+ tools)
- `risk_var_parametric` - Gaussian VaR
- `risk_var_historical` - Historical VaR
- `risk_cvar` - Expected Shortfall
- `risk_kelly_criterion` - Optimal bet sizing
- `risk_position_size` - Fixed-risk sizing
- `risk_max_drawdown` - Maximum drawdown
- `risk_hurst_exponent` - Trend/mean-reversion

### Portfolio (21+ tools)
- `portfolio_sharpe` - Risk-adjusted return
- `portfolio_sortino` - Downside risk-adjusted
- `portfolio_calmar` - CAGR / Max DD
- `portfolio_win_rate` - Win percentage
- `portfolio_profit_factor` - Gross P/L ratio
- `portfolio_expectancy` - Expected per-trade return

### Regime Detection (HyperPhysics)
- `regime_pbit_state` - Boltzmann market state
- `regime_ising_energy` - Market coherence
- `regime_hyperbolic_embed` - H^n embedding

### Greeks & Options
- `greeks_delta`, `greeks_gamma`, `greeks_theta`, `greeks_vega`
- `options_black_scholes` - European pricing

## HyperPhysics Integration

This MCP server integrates with HyperPhysics crates:
- **hyperphysics-pbit**: pBit dynamics for regime detection
- **hyperphysics-lorentz**: Hyperbolic geometry
- **hyperphysics-plugin**: Core plugin infrastructure

## License

MIT
