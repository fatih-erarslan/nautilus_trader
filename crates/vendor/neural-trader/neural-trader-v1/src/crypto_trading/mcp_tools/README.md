# Beefy Finance MCP Tools

This module provides MCP (Model Context Protocol) tools for integrating with Beefy Finance yield farming platform.

## Features

- **Vault Discovery**: Search and filter Beefy vaults across multiple chains
- **Deep Analysis**: Risk metrics, performance analytics, and historical data
- **Investment Execution**: Deposit, withdraw, and harvest yields
- **Portfolio Management**: Track positions, calculate yields, and rebalance
- **Real-time Updates**: WebSocket support for live APY monitoring

## Available Tools

### 1. `beefy_get_vaults`
Get available Beefy Finance vaults with advanced filtering.

**Parameters:**
- `chain`: Filter by blockchain (bsc, polygon, ethereum, etc.)
- `min_apy`: Minimum APY filter (e.g., 10 for 10%)
- `max_tvl`: Maximum TVL filter in USD
- `sort_by`: Sort criteria (apy, tvl, created, name)
- `limit`: Number of results to return

### 2. `beefy_analyze_vault`
Deep analysis of a specific Beefy vault with risk metrics.

**Parameters:**
- `vault_id`: The vault ID to analyze
- `include_history`: Include historical performance data

### 3. `beefy_invest`
Execute investment into a Beefy Finance vault.

**Parameters:**
- `vault_id`: The vault ID to invest in
- `amount`: Amount to invest in USD
- `slippage`: Maximum slippage tolerance (0.01 = 1%)
- `simulate`: Simulate without executing

### 4. `beefy_harvest_yields`
Harvest yields from Beefy vaults.

**Parameters:**
- `vault_ids`: Vault IDs to harvest from (empty for all)
- `auto_compound`: Auto-compound harvested yields
- `simulate`: Simulate without executing

### 5. `beefy_rebalance_portfolio`
Rebalance Beefy portfolio based on strategy.

**Parameters:**
- `strategy`: Rebalancing strategy (equal_weight, risk_parity, max_apy, custom)
- `target_allocations`: Target allocations for custom strategy
- `simulate`: Simulate without executing

## Architecture

```
crypto_trading/mcp_tools/
├── beefy_tools.py          # Main tool handlers
├── schemas.py              # Pydantic validation schemas
├── integration.py          # MCP server integration
└── handlers/
    ├── vault_handler.py    # Vault discovery and analysis
    ├── investment_handler.py # Transaction execution
    ├── portfolio_handler.py  # Portfolio management
    └── analytics_handler.py  # Risk and performance analytics
```

## Database Schema

The tools use SQLite database with tables for:
- `vault_positions`: Track individual vault positions
- `yield_history`: Historical yield data
- `crypto_transactions`: Transaction logs
- `portfolio_summary`: Aggregated portfolio metrics

## Risk Management

Each vault is evaluated across multiple risk dimensions:
- **Liquidity Risk**: Based on TVL
- **Smart Contract Risk**: Audit status and complexity
- **Protocol Risk**: Platform reputation and track record
- **Market Risk**: APY volatility and asset correlation

## WebSocket Integration

Real-time APY updates can be subscribed to via WebSocket:
```python
await beefy_handlers.start_apy_websocket(['vault-id-1', 'vault-id-2'])
```

## Example Usage

```python
# Get high-yield vaults on BSC
vaults = await beefy_get_vaults(
    chain="bsc",
    min_apy=20,
    sort_by="apy",
    limit=10
)

# Analyze a specific vault
analysis = await beefy_analyze_vault(
    vault_id="beefy-bnb-btcb",
    include_history=True
)

# Invest in a vault (simulation)
result = await beefy_invest(
    vault_id="beefy-bnb-btcb",
    amount=1000,
    slippage=0.01,
    simulate=True
)

# Rebalance portfolio
rebalance = await beefy_rebalance_portfolio(
    strategy="risk_parity",
    simulate=False
)
```

## Security Considerations

- Private keys should be stored in environment variables
- All transactions include slippage protection
- Smart contract interactions are validated before execution
- Simulation mode available for testing

## Integration with MCP Server

The tools are automatically registered when the MCP server starts if the module is available. They appear with the `beefy_` prefix in the tool list.