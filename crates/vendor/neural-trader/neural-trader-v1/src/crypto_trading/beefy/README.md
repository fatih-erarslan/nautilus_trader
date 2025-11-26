# Beefy Finance API Integration

This module provides a complete integration with Beefy Finance, a multi-chain yield optimizer that automatically compounds cryptocurrency yields across multiple DeFi protocols.

## Features

- **Multi-Chain Support**: Ethereum, BSC, Polygon, Arbitrum, Optimism, Fantom, Avalanche
- **Real API Integration**: Uses official Beefy Finance REST APIs
- **Web3 Integration**: Full support for on-chain interactions
- **Comprehensive Data Models**: Type-safe Pydantic models for all API responses
- **Error Handling**: Robust retry logic and rate limiting
- **Gas Estimation**: Calculate transaction costs before execution
- **Vault Search**: Filter vaults by chain, APY, TVL, and more

## Architecture

```
beefy/
├── __init__.py          # Package initialization
├── beefy_client.py      # Main API client (extends TradingAPIInterface)
├── web3_manager.py      # Multi-chain Web3 connection manager
├── data_models.py       # Pydantic models for API responses
├── example_usage.py     # Usage examples
└── README.md           # This file
```

## Installation

The module requires the following dependencies:
```bash
pip install web3 aiohttp pydantic tenacity eth-account
```

## Usage

### Basic Example

```python
import asyncio
from src.crypto_trading.beefy import BeefyFinanceAPI

async def main():
    async with BeefyFinanceAPI() as api:
        # Fetch all vaults
        vaults = await api.get_vaults()
        print(f"Found {len(vaults)} vaults")
        
        # Get APY data
        apy_data = await api.get_apy()
        
        # Search for high-yield USDC vaults
        usdc_vaults = await api.search_vaults(
            query="USDC",
            chain="polygon",
            min_apy=10.0
        )

asyncio.run(main())
```

### Web3 Integration

```python
# Prepare a deposit transaction
deposit_tx = api.prepare_deposit_transaction(
    vault_id="beefy-vault-id",
    vault_address="0x...",
    token_address="0x...",
    amount=Web3.to_wei(100, 'ether'),
    chain="bsc",
    user_address="0x..."
)

# Estimate gas costs
gas_estimate = api.estimate_gas_costs(
    chain="polygon",
    vault_address="0x...",
    action="deposit",
    user_address="0x...",
    amount=amount
)
```

## API Endpoints

The integration uses the following Beefy Finance API endpoints:

- `GET /vaults` - Fetch all available vaults
- `GET /apy` - Get current APY rates
- `GET /apy/breakdown` - Detailed APY breakdown
- `GET /tvl` - Total Value Locked data
- `GET /prices` - Current token prices
- `GET /lps` - Liquidity pool data
- `GET /earnings` - Vault earnings data
- `GET /holders` - Vault holder statistics

## Data Models

### BeefyVault
Core vault information including:
- Vault ID and name
- Chain and platform
- Token addresses
- Strategy information
- Risk levels
- Status (active/paused/eol)

### VaultAPY
APY metrics including:
- Base vault APR/APY
- Trading fees APR
- Reward token APRs
- Daily and annual rates
- Performance fees

### TransactionEstimate
Gas cost estimates including:
- Estimated gas units
- Gas price in wei
- Total cost in native token
- USD equivalent

## Web3 Features

### Multi-Chain Support
The `Web3Manager` class provides:
- Connections to 7+ blockchain networks
- Automatic middleware configuration
- Connection health monitoring
- Chain-specific configurations

### Transaction Preparation
- Token approval checking
- Gas estimation
- Transaction parameter building
- Error handling for failed estimations

### Vault Interactions
- Deposit preparation
- Withdrawal preparation
- Balance checking
- Price per share queries

## Error Handling

The module includes comprehensive error handling:
- Automatic retry with exponential backoff
- Rate limiting (100 requests per minute)
- Response caching (60-second TTL)
- Graceful degradation for failed connections

## Testing

Run the example script to test the integration:
```bash
python src/crypto_trading/beefy/example_usage.py
```

## Security Considerations

- Never store private keys in code
- Use environment variables for sensitive data
- Always verify transaction parameters before signing
- Test on testnets before mainnet deployment

## Future Enhancements

- [ ] WebSocket support for real-time updates
- [ ] Historical data fetching
- [ ] Advanced portfolio analytics
- [ ] Automated compounding strategies
- [ ] Cross-chain routing optimization

## Resources

- [Beefy Finance Docs](https://docs.beefy.finance/)
- [API Documentation](https://api.beefy.finance/)
- [Smart Contracts](https://github.com/beefyfinance/beefy-contracts)
- [Web3.py Documentation](https://web3py.readthedocs.io/)