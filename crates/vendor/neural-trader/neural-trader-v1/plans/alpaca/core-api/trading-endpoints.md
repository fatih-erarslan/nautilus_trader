# Alpaca API Core Trading Endpoints Research

## Overview
Alpaca's Trading API provides comprehensive endpoints for monitoring, placing, and canceling orders across equities, crypto, and options asset classes. The API supports both paper trading and live trading environments with the same interface.

## Supported Order Types (2024)

### Market Orders
- **Definition**: Request to buy or sell at currently available market price
- **Execution**: Fill nearly instantaneously
- **Risk**: Price not guaranteed but execution likelihood very high
- **Fractional Support**: Yes, for stocks only
- **API Endpoint**: `/orders` with `type: "market"`

### Limit Orders
- **Definition**: Order to buy or sell at specific price or better
- **Use Case**: Control execution price, may not fill immediately
- **Extended Hours**: Supported in pre-market (4:00-9:30 ET) and after-hours (4:00-8:00 PM ET)
- **Fractional Support**: Yes, including notional values
- **API Endpoint**: `/orders` with `type: "limit"`

### Stop Orders
- **Definition**: Becomes market order when stop price reached
- **Risk Management**: Limit losses or protect profits
- **Fractional Support**: Yes
- **API Endpoint**: `/orders` with `type: "stop"`

### Stop-Limit Orders
- **Definition**: Combines stop and limit features
- **Execution**: Becomes limit order when stop price triggered
- **Control**: Price control with activation trigger
- **API Endpoint**: `/orders` with `type: "stop_limit"`

### Trailing Stop Orders
- **Definition**: Stop price follows stock price by fixed amount/percentage
- **Use Case**: Lock in profits while allowing upside participation
- **Dynamic**: Automatically adjusts stop price
- **API Endpoint**: `/orders` with `type: "trailing_stop"`

## Python SDK Implementation (alpaca-py)

### Order Request Objects
```python
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce

# Market Order Example
market_order = MarketOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# Limit Order Example
limit_order = LimitOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    limit_price=150.00
)

# Fractional Order Example
fractional_order = MarketOrderRequest(
    symbol="AAPL",
    qty=0.5,  # Fractional shares
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)

# Notional Order Example
notional_order = MarketOrderRequest(
    symbol="AAPL",
    notional=100.00,  # $100 worth
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)
```

## Extended Hours Trading

### Pre-Market Trading
- **Hours**: 4:00 AM - 9:30 AM ET
- **Order Types**: Market, Limit, Stop, Stop-Limit
- **Implementation**: Set `extended_hours=True` in order request

### After-Hours Trading
- **Hours**: 4:00 PM - 8:00 PM ET
- **Order Types**: Market, Limit, Stop, Stop-Limit
- **Fractional Support**: Yes

### Extended Hours Example
```python
extended_hours_order = LimitOrderRequest(
    symbol="AAPL",
    qty=10,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    limit_price=150.00,
    extended_hours=True
)
```

## Multi-Asset Support

### Equities
- All major US exchanges
- Fractional shares supported
- Extended hours trading
- Options contracts

### Cryptocurrency
- 24/7 trading
- Fractional quantities
- Market, Limit, Stop-Limit orders
- Time in force: GTC, IOC

### Options
- Multi-leg strategies supported
- Same API as equities/crypto
- Complex order types available

## Order Management

### Order Status Tracking
- Real-time order status updates
- Order fill notifications
- Partial fill handling
- Order replacement/modification

### Order Lifecycle
1. **New**: Order received but not yet routed
2. **Partially Filled**: Some quantity executed
3. **Filled**: Complete execution
4. **Done for Day**: Expired at market close
5. **Canceled**: Order canceled
6. **Expired**: Time-based expiration
7. **Replaced**: Order modified
8. **Pending Cancel**: Cancellation in progress
9. **Pending Replace**: Modification in progress
10. **Rejected**: Order rejected by exchange

## Rate Limits and Best Practices

### API Limits
- **Standard**: 200 requests/minute, 10 requests/second burst
- **Unlimited Plan**: 1000 requests/minute
- **Response**: 429 status code when exceeded

### Best Practices
- Batch requests when possible
- Use WebSocket for real-time updates
- Implement exponential backoff
- Cache frequently accessed data
- Use order queuing for high-frequency strategies

## Error Handling

### Common Error Codes
- **400**: Bad Request - Invalid parameters
- **401**: Unauthorized - Authentication failed
- **403**: Forbidden - Insufficient permissions
- **422**: Unprocessable Entity - Business logic error
- **429**: Too Many Requests - Rate limit exceeded
- **500**: Internal Server Error - Server-side issue

### Retry Strategy
- Built-in retry for 429 and 504 status codes
- Default: 3 retries with 3-second intervals
- Configurable via environment variables

## 2024 Enhancements

### New Features
- Local Currency Trading API updates
- FIX API for institutional traders
- Enhanced fractional trading support
- Extended hours for limit orders
- Improved options trading capabilities

### Performance Improvements
- Reduced latency for order execution
- Enhanced reliability for high-frequency trading
- Better error messaging and debugging tools

## Integration Considerations

### Paper Trading
- Identical API to live trading
- Free real-time market data
- Reset capability for testing
- Global access available

### Live Trading
- Real money execution
- Same API endpoints as paper
- Additional compliance requirements
- Account verification needed

### Security
- API key authentication required
- Separate keys for paper/live accounts
- Environment variable storage recommended
- Regular key rotation advised