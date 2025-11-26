# Questrade API Integration

This module provides a production-ready integration with the Questrade API for the Canadian Trading Platform. It includes comprehensive OAuth2 authentication, REST API client functionality, market data streaming, and order management capabilities.

## Features

- **OAuth2 Authentication**: Secure token management with automatic refresh
- **REST API Client**: Full coverage of Questrade API endpoints
- **Market Data Streaming**: Real-time quote streaming with WebSocket support
- **Order Management**: Market, limit, stop, and bracket order support
- **Rate Limiting**: Built-in rate limiting to respect API limits (30 requests/second)
- **Connection Pooling**: Efficient HTTP connection management
- **Error Handling**: Comprehensive error handling with retry logic
- **Secure Token Storage**: Encrypted local storage for OAuth2 tokens

## Installation

```bash
pip install -r src/canadian_trading/requirements.txt
```

## Quick Start

### 1. Get Your Refresh Token

1. Log in to your Questrade account
2. Navigate to App Hub: https://www.questrade.com/api/documentation/getting-started
3. Register your application
4. Generate a manual refresh token

### 2. Basic Usage

```python
import asyncio
from src.canadian_trading.brokers import QuestradeAPI, QuestradeDataFeed

async def main():
    # Initialize API with your refresh token
    api = QuestradeAPI(refresh_token="your_refresh_token_here")
    
    try:
        # Initialize connection
        await api.initialize()
        
        # Get accounts
        accounts = await api.get_accounts()
        print(f"Found {len(accounts)} accounts")
        
        # Get market data
        data_feed = QuestradeDataFeed(api)
        quote = await data_feed.get_quote("SHOP.TO")
        print(f"SHOP.TO: ${quote['lastTradePrice']}")
        
    finally:
        await api.close()

# Run
asyncio.run(main())
```

## API Components

### QuestradeAPI

The main API client that handles authentication and raw API calls.

```python
api = QuestradeAPI(
    refresh_token="your_token",
    timeout=30,           # Request timeout in seconds
    max_retries=3        # Number of retry attempts
)
```

#### Key Methods:
- `authenticate()`: Authenticate or refresh tokens
- `get_accounts()`: Get all accounts
- `get_account_positions(account_id)`: Get positions
- `get_account_balances(account_id)`: Get account balances
- `place_order(...)`: Place various order types
- `get_quotes(symbol_ids)`: Get real-time quotes

### QuestradeDataFeed

High-level interface for market data operations.

```python
data_feed = QuestradeDataFeed(api)

# Get single quote
quote = await data_feed.get_quote("RY.TO")

# Get multiple quotes
quotes = await data_feed.get_quotes_batch(["RY.TO", "TD.TO", "SHOP.TO"])

# Get historical data
candles = await data_feed.get_historical_data(
    "SHOP.TO",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    interval="OneDay"
)

# Stream real-time quotes
async def quote_callback(quote):
    print(f"Update: {quote['symbol']} @ ${quote['lastTradePrice']}")

await data_feed.stream_quotes(["SHOP.TO", "RY.TO"], quote_callback)
```

### QuestradeOrderManager

High-level interface for order management.

```python
order_manager = QuestradeOrderManager(api)

# Place market order
result = await order_manager.place_market_order(
    account_id="12345678",
    symbol="SHOP.TO",
    quantity=100,
    action="Buy"
)

# Place limit order
result = await order_manager.place_limit_order(
    account_id="12345678",
    symbol="RY.TO",
    quantity=50,
    limit_price=150.00,
    action="Buy",
    time_in_force="Day"
)

# Place bracket order (entry + stop loss + take profit)
result = await order_manager.place_bracket_order(
    account_id="12345678",
    symbol="SHOP.TO",
    quantity=100,
    limit_price=85.00,      # Entry price
    stop_loss_price=80.00,  # Stop loss
    take_profit_price=90.00 # Take profit
)
```

## Authentication & Token Management

### OAuth2 Flow

The integration handles OAuth2 authentication automatically:

1. **Initial Authentication**: Use manual refresh token from Questrade
2. **Token Storage**: Tokens are encrypted and stored locally
3. **Automatic Refresh**: Tokens are refreshed automatically before expiry
4. **Secure Storage**: Uses Fernet encryption with password-based key derivation

### Token Storage

Tokens are stored in `~/.canadian_trading/tokens/` with encryption:

```python
from src.canadian_trading.utils import setup_broker_authentication

# Initial setup with manual token
auth_result = await setup_broker_authentication(
    "questrade",
    refresh_token="your_manual_refresh_token"
)

# Subsequent uses (automatic token loading)
auth_result = await setup_broker_authentication("questrade")
```

## Error Handling

The integration includes comprehensive error handling:

```python
from src.canadian_trading.brokers import QuestradeAPIError

try:
    result = await api.place_order(...)
except QuestradeAPIError as e:
    print(f"API Error: {e.message}")
    print(f"Error Code: {e.code}")
    print(f"Details: {e.details}")
```

## Rate Limiting

Built-in rate limiting ensures compliance with Questrade's limits:

- Maximum 30 requests per second
- Automatic request spacing
- Thread-safe for concurrent operations

## Advanced Features

### Connection Pooling

Efficient connection management for high-performance:

```python
# Connector with custom settings
api = QuestradeAPI(
    refresh_token="token",
    connector_limit=100,      # Total connections
    connector_limit_per_host=30  # Per-host limit
)
```

### Streaming Market Data

Real-time WebSocket streaming:

```python
# Define callback
async def on_quote_update(quote):
    symbol = quote['symbol']
    price = quote['lastTradePrice']
    print(f"{symbol}: ${price}")

# Start streaming
await api.stream_quotes([38526, 12345], on_quote_update)

# Stop streaming
await api.stop_streaming()
```

### Order Validation

Validate orders before placement:

```python
validation = await api.validate_order(
    account_id="12345678",
    order_data={
        "symbolId": 38526,
        "orderType": "Limit",
        "action": "Buy",
        "quantity": 100,
        "limitPrice": 85.00
    },
    impact=True  # Include market impact estimate
)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest src/canadian_trading/tests/test_questrade.py -v

# Run with coverage
pytest src/canadian_trading/tests/test_questrade.py --cov=src.canadian_trading.brokers

# Run integration tests (requires test account)
pytest src/canadian_trading/tests/test_questrade.py -m integration
```

## Security Best Practices

1. **Never commit tokens**: Keep refresh tokens out of version control
2. **Use environment variables**: Store sensitive data in `.env` files
3. **Encrypt token storage**: Tokens are automatically encrypted
4. **Secure communication**: All API calls use HTTPS
5. **Rate limit compliance**: Built-in rate limiting prevents API abuse

## Environment Variables

Set these environment variables for production:

```bash
# Token encryption password
export TOKEN_ENCRYPTION_PASSWORD="your-secure-password"

# Questrade credentials (optional)
export QUESTRADE_CLIENT_ID="your-client-id"
export QUESTRADE_CLIENT_SECRET="your-client-secret"
```

## Common Issues & Solutions

### Authentication Errors

```python
# Error: Invalid refresh token
# Solution: Get new manual token from Questrade App Hub

# Error: Token expired
# Solution: Tokens auto-refresh, but can force refresh:
await api.authenticate()
```

### Rate Limiting

```python
# Error: Too many requests
# Solution: Built-in rate limiter handles this automatically
# Can adjust if needed:
api.rate_limiter = RateLimiter(calls_per_second=20)
```

### Connection Issues

```python
# Error: Connection timeout
# Solution: Increase timeout
api = QuestradeAPI(refresh_token="token", timeout=60)

# Error: SSL verification
# Solution: Ensure system certificates are up to date
```

## Support

For issues specific to:
- **This integration**: Check the GitHub issues
- **Questrade API**: Contact Questrade API support
- **OAuth2 tokens**: Use Questrade App Hub

## License

This integration is part of the AI News Trading Platform and follows the same license terms.