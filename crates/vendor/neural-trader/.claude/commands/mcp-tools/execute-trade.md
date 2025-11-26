# mcp__ai-news-trader__execute_trade

## Description
Execute live trade with advanced order management. This tool handles real trade execution with comprehensive order types, risk management, and real-time monitoring. While operating in demo mode by default, it provides realistic execution simulation including slippage, partial fills, and market impact.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| strategy | string | Yes | N/A | Trading strategy to use for execution |
| symbol | string | Yes | N/A | Trading symbol (e.g., "AAPL", "BTC-USD") |
| action | string | Yes | N/A | Trading action: "buy", "sell", "short", "cover" |
| quantity | integer | Yes | N/A | Number of shares/contracts to trade |
| order_type | string | No | "market" | Order type: "market", "limit", "stop", "stop_limit" |
| limit_price | number | No | null | Limit price for limit orders |

```typescript
{
  strategy: string,        // Required: Strategy identifier
  symbol: string,          // Required: Trading symbol
  action: string,          // Required: Trading action
  quantity: number,        // Required: Trade quantity (integer)
  order_type?: string,     // Optional: Order type (default: "market")
  limit_price?: number     // Optional: Limit price for limit orders
}
```

## Return Value Structure
```json
{
  "order_id": "string",
  "timestamp": "string",            // ISO timestamp
  "status": "string",               // "pending", "filled", "partial", "cancelled", "rejected"
  "strategy": "string",
  "symbol": "string",
  "action": "string",
  "order_details": {
    "order_type": "string",
    "quantity_requested": number,
    "quantity_filled": number,
    "quantity_remaining": number,
    "limit_price": number,          // null for market orders
    "stop_price": number,           // null for non-stop orders
    "time_in_force": "string",      // "DAY", "GTC", "IOC", "FOK"
    "extended_hours": boolean
  },
  "execution_details": {
    "average_fill_price": number,
    "fills": [
      {
        "timestamp": "string",
        "quantity": number,
        "price": number,
        "venue": "string"           // Exchange/venue
      }
    ],
    "total_commission": number,
    "total_fees": number,
    "slippage": number,
    "market_impact": number,
    "execution_time_ms": number
  },
  "position_update": {
    "previous_position": number,
    "current_position": number,
    "average_cost": number,
    "total_cost_basis": number,
    "unrealized_pnl": number,
    "realized_pnl": number,
    "day_pnl": number
  },
  "risk_management": {
    "stop_loss_set": boolean,
    "stop_loss_price": number,
    "take_profit_set": boolean,
    "take_profit_levels": [number],
    "position_size_check": "string", // "within_limits", "warning", "exceeded"
    "margin_requirement": number,
    "buying_power_used": number,
    "risk_score": number            // 0-100
  },
  "market_conditions": {
    "bid": number,
    "ask": number,
    "spread": number,
    "volume": number,
    "volatility": number,
    "liquidity_score": number       // 0-100
  },
  "compliance": {
    "pattern_day_trader": boolean,
    "trades_today": number,
    "buying_power_remaining": number,
    "margin_call": boolean,
    "restrictions": ["string"]
  },
  "notifications": {
    "order_placed": "string",       // Notification message
    "fill_notifications": ["string"],
    "risk_alerts": ["string"],
    "compliance_warnings": ["string"]
  },
  "next_actions": {
    "recommended_stop": number,
    "recommended_targets": [number],
    "monitoring_alerts": ["string"],
    "position_management": ["string"]
  },
  "audit_trail": {
    "strategy_signals": object,     // Strategy-specific signals
    "risk_checks": object,          // Risk validation results
    "execution_log": ["string"],    // Detailed execution steps
    "demo_mode": boolean            // Whether this was demo execution
  }
}
```

## Examples

### Example 1: Basic Market Order Execution
```python
# Execute a simple market buy order
execution = mcp__ai_news_trader__execute_trade(
    strategy="momentum_crossover",
    symbol="AAPL",
    action="buy",
    quantity=100
)

print(f"=== Order Execution Summary ===")
print(f"Order ID: {execution['order_id']}")
print(f"Status: {execution['status']}")
print(f"Filled: {execution['order_details']['quantity_filled']} shares")
print(f"Average Price: ${execution['execution_details']['average_fill_price']:.2f}")
print(f"Total Cost: ${execution['position_update']['total_cost_basis']:,.2f}")

# Check execution quality
if execution['execution_details']['slippage'] > 0.001:
    print(f"⚠️ Slippage: ${execution['execution_details']['slippage']:.4f}")

# Display risk management
rm = execution['risk_management']
print(f"\nRisk Management:")
print(f"Stop Loss: ${rm['stop_loss_price']:.2f}")
print(f"Take Profit: ${rm['take_profit_levels'][0]:.2f}")
```

### Example 2: Limit Order with Price Improvement
```python
# Execute limit order with price improvement logic
def execute_limit_order_smart(strategy, symbol, action, quantity, improvement_bps=10):
    """Execute limit order with price improvement"""
    
    # Get current market analysis
    analysis = mcp__ai_news_trader__quick_analysis(symbol=symbol)
    current_price = analysis['current_price']
    
    # Calculate limit price with improvement
    if action in ["buy", "cover"]:
        # For buys, place limit below current price
        limit_price = current_price * (1 - improvement_bps / 10000)
    else:  # sell, short
        # For sells, place limit above current price
        limit_price = current_price * (1 + improvement_bps / 10000)
    
    # Execute the order
    execution = mcp__ai_news_trader__execute_trade(
        strategy=strategy,
        symbol=symbol,
        action=action,
        quantity=quantity,
        order_type="limit",
        limit_price=round(limit_price, 2)
    )
    
    print(f"=== Smart Limit Order Execution ===")
    print(f"Symbol: {symbol}")
    print(f"Action: {action.upper()} {quantity} shares")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Limit Price: ${limit_price:.2f}")
    print(f"Price Improvement Target: {improvement_bps} bps")
    print(f"\nOrder Status: {execution['status']}")
    
    # Monitor for fills
    if execution['status'] == 'pending':
        print("Order pending... monitoring for fills")
        # In real implementation, would monitor for updates
    
    elif execution['status'] == 'filled':
        actual_price = execution['execution_details']['average_fill_price']
        improvement = abs(actual_price - current_price) / current_price * 10000
        print(f"Filled at: ${actual_price:.2f}")
        print(f"Price Improvement: {improvement:.1f} bps")
        
        # Calculate savings
        if action in ["buy", "cover"]:
            savings = (current_price - actual_price) * quantity
        else:
            savings = (actual_price - current_price) * quantity
        print(f"Execution Savings: ${savings:.2f}")
    
    return execution

# Usage
execution = execute_limit_order_smart("swing_trader", "TSLA", "buy", 50, improvement_bps=15)
```

### Example 3: Bracket Order with Stop Loss and Take Profit
```python
# Execute trade with automatic bracket orders
def execute_bracket_order(strategy, symbol, action, quantity, stop_pct=2.0, target_pct=5.0):
    """Execute main order with automatic stop loss and take profit"""
    
    # Execute main order
    main_order = mcp__ai_news_trader__execute_trade(
        strategy=strategy,
        symbol=symbol,
        action=action,
        quantity=quantity,
        order_type="market"
    )
    
    if main_order['status'] != 'filled':
        print(f"Main order not filled: {main_order['status']}")
        return main_order
    
    fill_price = main_order['execution_details']['average_fill_price']
    
    # Calculate bracket prices
    if action in ["buy", "cover"]:
        stop_price = fill_price * (1 - stop_pct / 100)
        target_price = fill_price * (1 + target_pct / 100)
        stop_action = "sell"
    else:  # sell, short
        stop_price = fill_price * (1 + stop_pct / 100)
        target_price = fill_price * (1 - target_pct / 100)
        stop_action = "cover" if action == "short" else "buy"
    
    print(f"=== Bracket Order Execution ===")
    print(f"Main Order: {action.upper()} {quantity} {symbol} @ ${fill_price:.2f}")
    print(f"Stop Loss: ${stop_price:.2f} (-{stop_pct}%)")
    print(f"Take Profit: ${target_price:.2f} (+{target_pct}%)")
    
    # Note: In real implementation, would place OCO (One-Cancels-Other) orders
    # Here we're showing the structure
    bracket_orders = {
        'main_order': main_order,
        'stop_loss': {
            'price': stop_price,
            'action': stop_action,
            'quantity': quantity,
            'order_type': 'stop'
        },
        'take_profit': {
            'price': target_price,
            'action': stop_action,
            'quantity': quantity,
            'order_type': 'limit'
        }
    }
    
    # Risk/Reward calculation
    risk_amount = abs(fill_price - stop_price) * quantity
    reward_amount = abs(target_price - fill_price) * quantity
    risk_reward_ratio = reward_amount / risk_amount
    
    print(f"\nRisk/Reward Analysis:")
    print(f"Max Risk: ${risk_amount:,.2f}")
    print(f"Target Profit: ${reward_amount:,.2f}")
    print(f"Risk/Reward Ratio: {risk_reward_ratio:.2f}:1")
    
    return bracket_orders

# Usage
bracket = execute_bracket_order("swing_trader", "NVDA", "buy", 25, stop_pct=1.5, target_pct=4.5)
```

### Example 4: Scaled Entry with Multiple Orders
```python
# Execute scaled entry with multiple price levels
def execute_scaled_entry(strategy, symbol, total_quantity, scale_levels=3):
    """Execute position with scaled entry at multiple price levels"""
    
    # Get current market data
    analysis = mcp__ai_news_trader__quick_analysis(symbol=symbol)
    current_price = analysis['current_price']
    atr = analysis['technical_indicators']['atr']
    
    # Calculate scale levels
    quantity_per_level = total_quantity // scale_levels
    remainder = total_quantity % scale_levels
    
    executions = []
    total_filled = 0
    weighted_avg_price = 0
    
    print(f"=== Scaled Entry Execution ===")
    print(f"Symbol: {symbol}")
    print(f"Total Quantity: {total_quantity}")
    print(f"Scale Levels: {scale_levels}")
    print(f"ATR: ${atr:.2f}\n")
    
    for level in range(scale_levels):
        # Calculate price for this level
        # Each level is 0.5 ATR apart
        price_offset = level * 0.5 * atr
        limit_price = current_price - price_offset  # For buy orders
        
        # Add remainder to last order
        level_quantity = quantity_per_level + (remainder if level == scale_levels - 1 else 0)
        
        # Execute order for this level
        print(f"Level {level + 1}: {level_quantity} shares @ ${limit_price:.2f}")
        
        execution = mcp__ai_news_trader__execute_trade(
            strategy=strategy,
            symbol=symbol,
            action="buy",
            quantity=level_quantity,
            order_type="limit",
            limit_price=limit_price
        )
        
        executions.append(execution)
        
        # Track fills
        if execution['status'] == 'filled':
            filled_qty = execution['order_details']['quantity_filled']
            fill_price = execution['execution_details']['average_fill_price']
            total_filled += filled_qty
            weighted_avg_price += fill_price * filled_qty
    
    # Calculate results
    if total_filled > 0:
        weighted_avg_price /= total_filled
        
    print(f"\n=== Execution Summary ===")
    print(f"Total Filled: {total_filled}/{total_quantity} shares")
    print(f"Average Price: ${weighted_avg_price:.2f}")
    print(f"Price Improvement vs Market: ${(current_price - weighted_avg_price):.2f}")
    
    # Calculate position metrics
    total_cost = weighted_avg_price * total_filled
    price_improvement_pct = ((current_price - weighted_avg_price) / current_price) * 100
    
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"Price Improvement: {price_improvement_pct:.2f}%")
    
    return {
        'executions': executions,
        'total_filled': total_filled,
        'average_price': weighted_avg_price,
        'unfilled': total_quantity - total_filled
    }

# Usage
scaled_execution = execute_scaled_entry("value_investor", "MSFT", 300, scale_levels=4)
```

### Example 5: Risk-Managed Portfolio Execution
```python
# Execute trades with portfolio-level risk management
class PortfolioExecutor:
    def __init__(self, max_portfolio_risk=0.02, max_position_risk=0.005):
        self.max_portfolio_risk = max_portfolio_risk  # 2% portfolio risk
        self.max_position_risk = max_position_risk    # 0.5% per position
        self.portfolio_value = 100000  # Example portfolio value
        self.open_risk = 0
    
    def execute_with_risk_management(self, trades):
        """Execute multiple trades with portfolio risk limits"""
        
        executions = []
        rejected_trades = []
        
        print("=== Portfolio Risk-Managed Execution ===")
        print(f"Portfolio Value: ${self.portfolio_value:,}")
        print(f"Max Portfolio Risk: {self.max_portfolio_risk*100}%")
        print(f"Max Position Risk: {self.max_position_risk*100}%\n")
        
        for trade in trades:
            # Calculate position risk
            position_value = trade['quantity'] * trade['entry_price']
            stop_distance = abs(trade['entry_price'] - trade['stop_loss'])
            position_risk = (stop_distance / trade['entry_price']) * position_value
            position_risk_pct = position_risk / self.portfolio_value
            
            # Check risk limits
            if position_risk_pct > self.max_position_risk:
                print(f"❌ {trade['symbol']}: Position risk {position_risk_pct*100:.2f}% exceeds limit")
                rejected_trades.append(trade)
                continue
            
            if self.open_risk + position_risk_pct > self.max_portfolio_risk:
                print(f"❌ {trade['symbol']}: Would exceed portfolio risk limit")
                rejected_trades.append(trade)
                continue
            
            # Execute trade
            print(f"✓ Executing {trade['symbol']}: {trade['action']} {trade['quantity']} shares")
            print(f"  Position Risk: ${position_risk:.2f} ({position_risk_pct*100:.2f}%)")
            
            execution = mcp__ai_news_trader__execute_trade(
                strategy=trade['strategy'],
                symbol=trade['symbol'],
                action=trade['action'],
                quantity=trade['quantity'],
                order_type=trade.get('order_type', 'market'),
                limit_price=trade.get('limit_price')
            )
            
            if execution['status'] in ['filled', 'partial']:
                self.open_risk += position_risk_pct
                executions.append(execution)
                print(f"  Status: {execution['status']}")
                print(f"  Fill Price: ${execution['execution_details']['average_fill_price']:.2f}")
            else:
                print(f"  Failed: {execution['status']}")
            
            print(f"  Portfolio Risk Used: {self.open_risk*100:.2f}%\n")
        
        # Summary
        print(f"\n=== Execution Summary ===")
        print(f"Executed: {len(executions)} trades")
        print(f"Rejected: {len(rejected_trades)} trades")
        print(f"Total Portfolio Risk: {self.open_risk*100:.2f}%")
        print(f"Remaining Risk Capacity: {(self.max_portfolio_risk - self.open_risk)*100:.2f}%")
        
        return {
            'executions': executions,
            'rejected': rejected_trades,
            'portfolio_risk': self.open_risk
        }
    
    def calculate_optimal_position_size(self, symbol, entry_price, stop_loss):
        """Calculate position size based on risk limits"""
        
        stop_distance = abs(entry_price - stop_loss)
        risk_per_share = stop_distance
        
        # Maximum shares based on position risk limit
        max_position_risk_dollars = self.portfolio_value * self.max_position_risk
        max_shares_by_risk = int(max_position_risk_dollars / risk_per_share)
        
        # Maximum shares based on available portfolio risk
        available_risk = self.max_portfolio_risk - self.open_risk
        available_risk_dollars = self.portfolio_value * available_risk
        max_shares_by_portfolio = int(available_risk_dollars / risk_per_share)
        
        optimal_shares = min(max_shares_by_risk, max_shares_by_portfolio)
        
        return {
            'optimal_shares': optimal_shares,
            'position_risk': (optimal_shares * risk_per_share) / self.portfolio_value,
            'max_by_position_limit': max_shares_by_risk,
            'max_by_portfolio_limit': max_shares_by_portfolio
        }

# Usage
executor = PortfolioExecutor()

# Define trades with risk parameters
trades = [
    {
        'strategy': 'momentum_crossover',
        'symbol': 'AAPL',
        'action': 'buy',
        'quantity': 100,
        'entry_price': 185.50,
        'stop_loss': 182.00
    },
    {
        'strategy': 'swing_trader',
        'symbol': 'GOOGL',
        'action': 'buy',
        'quantity': 50,
        'entry_price': 142.30,
        'stop_loss': 139.50
    },
    {
        'strategy': 'mean_reversion_bands',
        'symbol': 'MSFT',
        'action': 'buy',
        'quantity': 75,
        'entry_price': 378.20,
        'stop_loss': 371.00
    }
]

# Execute with risk management
results = executor.execute_with_risk_management(trades)
```

## Common Use Cases

1. **Live Trading**: Execute real trades with proper risk management
2. **Order Management**: Handle complex order types and conditions
3. **Portfolio Execution**: Execute multiple trades with portfolio constraints
4. **Algorithmic Trading**: Automated execution based on signals
5. **Risk Control**: Enforce position and portfolio risk limits
6. **Compliance**: Ensure trades meet regulatory requirements

## Error Handling Notes

- **Insufficient Funds**: Not enough buying power for trade
- **Position Limits**: Exceeds maximum position size
- **Market Closed**: Cannot execute outside market hours
- **Invalid Price**: Limit price outside valid range
- **Order Rejected**: Broker or risk system rejection
- **Connection Issues**: Network or API connectivity problems

### Error Handling Example:
```python
def robust_trade_execution(strategy, symbol, action, quantity, max_retries=3):
    """Execute trade with comprehensive error handling and recovery"""
    
    attempt = 0
    execution_log = []
    
    while attempt < max_retries:
        try:
            attempt += 1
            
            # Pre-execution checks
            portfolio_status = mcp__ai_news_trader__get_portfolio_status()
            buying_power = portfolio_status.get('buying_power', 0)
            
            # Estimate required capital
            analysis = mcp__ai_news_trader__quick_analysis(symbol=symbol)
            estimated_cost = analysis['current_price'] * quantity * 1.01  # 1% buffer
            
            if action in ['buy', 'cover'] and estimated_cost > buying_power:
                raise ValueError(f"Insufficient buying power. Required: ${estimated_cost:,.2f}, Available: ${buying_power:,.2f}")
            
            # Execute trade
            print(f"Attempt {attempt}: Executing {action} {quantity} {symbol}")
            
            execution = mcp__ai_news_trader__execute_trade(
                strategy=strategy,
                symbol=symbol,
                action=action,
                quantity=quantity
            )
            
            execution_log.append({
                'attempt': attempt,
                'status': execution['status'],
                'timestamp': execution['timestamp']
            })
            
            # Handle different statuses
            if execution['status'] == 'filled':
                print(f"✓ Order filled successfully at ${execution['execution_details']['average_fill_price']:.2f}")
                return {
                    'success': True,
                    'execution': execution,
                    'attempts': attempt,
                    'log': execution_log
                }
            
            elif execution['status'] == 'partial':
                filled = execution['order_details']['quantity_filled']
                remaining = execution['order_details']['quantity_remaining']
                print(f"⚠️ Partial fill: {filled}/{quantity} shares filled")
                
                # Decide whether to accept partial or retry
                if filled >= quantity * 0.75:  # Accept if 75% filled
                    return {
                        'success': True,
                        'execution': execution,
                        'attempts': attempt,
                        'log': execution_log,
                        'warning': f"Partial fill: {filled}/{quantity}"
                    }
                else:
                    # Cancel and retry
                    print("Cancelling partial fill and retrying...")
                    # In real implementation, would cancel order
                    
            elif execution['status'] == 'rejected':
                reason = execution.get('rejection_reason', 'Unknown')
                print(f"❌ Order rejected: {reason}")
                
                # Handle specific rejection reasons
                if 'margin' in reason.lower():
                    raise ValueError("Margin requirements not met")
                elif 'pattern day' in reason.lower():
                    raise ValueError("Pattern day trader restriction")
                elif 'position limit' in reason.lower():
                    # Try with reduced quantity
                    new_quantity = int(quantity * 0.75)
                    print(f"Retrying with reduced quantity: {new_quantity}")
                    quantity = new_quantity
                
            elif execution['status'] == 'pending':
                print("Order pending... monitoring")
                # In real implementation, would monitor order status
                
        except ValueError as e:
            print(f"Validation error: {e}")
            execution_log.append({
                'attempt': attempt,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            break  # Don't retry validation errors
            
        except ConnectionError as e:
            print(f"Connection error: {e}")
            execution_log.append({
                'attempt': attempt,
                'error': 'Connection error',
                'timestamp': datetime.now().isoformat()
            })
            
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            execution_log.append({
                'attempt': attempt,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            
            if attempt < max_retries:
                time.sleep(5)
    
    # All attempts failed
    return {
        'success': False,
        'attempts': attempt,
        'log': execution_log,
        'error': 'Max retries exceeded'
    }

# Usage with error handling
result = robust_trade_execution("momentum_crossover", "TSLA", "buy", 100)

if result['success']:
    print(f"\nTrade executed successfully in {result['attempts']} attempt(s)")
    if 'warning' in result:
        print(f"Warning: {result['warning']}")
else:
    print(f"\nTrade execution failed after {result['attempts']} attempts")
    print("Execution log:")
    for log_entry in result['log']:
        print(f"  Attempt {log_entry['attempt']}: {log_entry.get('status', log_entry.get('error'))}")
```

## Performance Tips

1. **Pre-validation**: Validate orders before submission
2. **Smart Routing**: Use smart order routing for best execution
3. **Batch Orders**: Submit related orders together
4. **Async Execution**: Use async for multiple independent orders
5. **Connection Pooling**: Maintain persistent API connections
6. **Order Caching**: Cache order status to reduce API calls

### Performance Example:
```python
import asyncio
from collections import defaultdict
import threading

class HighPerformanceExecutor:
    def __init__(self):
        self.order_cache = {}
        self.execution_stats = defaultdict(list)
        self.lock = threading.Lock()
    
    async def execute_basket_trades(self, trades, parallel=True):
        """Execute multiple trades with optimal performance"""
        
        start_time = time.time()
        
        if parallel:
            # Execute trades in parallel
            tasks = []
            for trade in trades:
                task = asyncio.create_task(
                    self._execute_single_trade_async(trade)
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute trades sequentially (for dependent orders)
            results = []
            for trade in trades:
                result = await self._execute_single_trade_async(trade)
                results.append(result)
        
        execution_time = time.time() - start_time
        
        # Aggregate results
        successful = sum(1 for r in results if r and r.get('status') == 'filled')
        total_volume = sum(r.get('order_details', {}).get('quantity_filled', 0) 
                          for r in results if r)
        
        return {
            'executions': results,
            'summary': {
                'total_trades': len(trades),
                'successful': successful,
                'failed': len(trades) - successful,
                'total_volume': total_volume,
                'execution_time': execution_time,
                'trades_per_second': len(trades) / execution_time
            }
        }
    
    async def _execute_single_trade_async(self, trade):
        """Execute single trade asynchronously"""
        
        loop = asyncio.get_event_loop()
        
        # Check cache for recent execution
        cache_key = f"{trade['symbol']}_{trade['action']}_{trade['quantity']}"
        if cache_key in self.order_cache:
            cached_order, cache_time = self.order_cache[cache_key]
            if time.time() - cache_time < 5:  # 5 second cache
                return cached_order
        
        # Execute trade
        try:
            result = await loop.run_in_executor(
                None,
                lambda: mcp__ai_news_trader__execute_trade(**trade)
            )
            
            # Cache successful execution
            if result['status'] in ['filled', 'partial']:
                with self.lock:
                    self.order_cache[cache_key] = (result, time.time())
                    self.execution_stats[trade['symbol']].append({
                        'timestamp': time.time(),
                        'latency': result['execution_details']['execution_time_ms']
                    })
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'trade': trade,
                'status': 'failed'
            }
    
    def get_execution_metrics(self, symbol=None):
        """Get execution performance metrics"""
        
        if symbol:
            stats = self.execution_stats.get(symbol, [])
        else:
            stats = [s for symbol_stats in self.execution_stats.values() 
                    for s in symbol_stats]
        
        if not stats:
            return None
        
        latencies = [s['latency'] for s in stats]
        
        return {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'total_executions': len(stats),
            'cache_size': len(self.order_cache)
        }

# Usage
async def main():
    executor = HighPerformanceExecutor()
    
    # Define basket of trades
    trades = [
        {
            'strategy': 'momentum_crossover',
            'symbol': 'AAPL',
            'action': 'buy',
            'quantity': 100
        },
        {
            'strategy': 'mean_reversion_bands',
            'symbol': 'GOOGL',
            'action': 'sell',
            'quantity': 50
        },
        {
            'strategy': 'swing_trader',
            'symbol': 'MSFT',
            'action': 'buy',
            'quantity': 75
        }
    ]
    
    # Execute basket
    results = await executor.execute_basket_trades(trades, parallel=True)
    
    print(f"=== Basket Execution Summary ===")
    print(f"Total Orders: {results['summary']['total_trades']}")
    print(f"Successful: {results['summary']['successful']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Execution Time: {results['summary']['execution_time']:.2f}s")
    print(f"Throughput: {results['summary']['trades_per_second']:.2f} trades/second")
    
    # Get performance metrics
    metrics = executor.get_execution_metrics()
    if metrics:
        print(f"\n=== Performance Metrics ===")
        print(f"Average Latency: {metrics['avg_latency_ms']:.1f}ms")
        print(f"Min Latency: {metrics['min_latency_ms']}ms")
        print(f"Max Latency: {metrics['max_latency_ms']}ms")

# Run
asyncio.run(main())
```

## Related Tools
- `simulate_trade`: Test trades before execution
- `quick_analysis`: Analyze market before trading
- `get_portfolio_status`: Check portfolio before/after trades
- `risk_analysis`: Assess portfolio risk impact
- `performance_report`: Track execution quality over time