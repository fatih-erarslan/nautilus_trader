# Monitor Command - Real-Time Trading Dashboard

The monitor command provides a comprehensive real-time monitoring dashboard for trading strategies using Ink (React for CLI).

## Features

- **Real-time Updates**: Dashboard updates every second with live strategy data
- **Multiple Panels**: Strategy status, positions, P&L, trades, metrics, system resources, and alerts
- **Interactive**: Keyboard shortcuts for navigation and control
- **Color-Coded**: Visual indicators for profits/losses, alerts, and status
- **Mock Mode**: Test dashboard with simulated data when NAPI bindings unavailable
- **Alert System**: Configurable alerts for losses, drawdowns, system issues

## Commands

### Launch Monitoring Dashboard

```bash
# Monitor default demo strategy
neural-trader monitor

# Monitor specific strategy
neural-trader monitor strategy-id

# Use mock data mode
neural-trader monitor --mock
neural-trader monitor strategy-id -m
```

### List Running Strategies

```bash
neural-trader monitor list
```

Shows all running strategies with their status, uptime, and P&L.

### View Strategy Logs

```bash
# View logs for a strategy
neural-trader monitor logs strategy-id

# Follow logs in real-time
neural-trader monitor logs strategy-id --follow
neural-trader monitor logs strategy-id -f
```

### Show Performance Metrics

```bash
neural-trader monitor metrics strategy-id
```

Displays detailed performance metrics including:
- Profitability (total P&L, daily P&L, average)
- Trade statistics (win rate, average win/loss, largest win/loss)
- Risk metrics (Sharpe ratio, Sortino ratio, max drawdown, profit factor)
- Overall rating (1-5 stars)

### Show Alerts

```bash
# Show unacknowledged alerts
neural-trader monitor alerts

# Show all alerts (including acknowledged)
neural-trader monitor alerts --all

# Filter by severity
neural-trader monitor alerts --severity=error
neural-trader monitor alerts --severity=warning
```

## Dashboard Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  ⚡ Neural Trader - Real-Time Monitoring Dashboard               │
├──────────────────────────────────────────────────────────────────┤
│  ┌─ Strategy Status ───┐  ┌─ Profit & Loss ────┐               │
│  │ Name: Momentum      │  │ Today P&L: +$1,234  │               │
│  │ Type: momentum      │  │ Total P&L: +$5,432  │               │
│  │ Status: RUNNING     │  │ Total Trades: 147   │               │
│  │ Runtime: 2h 34m 15s │  │ Win Rate: 60.5%     │               │
│  └─────────────────────┘  └─────────────────────┘               │
│                                                                   │
│  ┌─ Current Positions ─────────────────────────────┐             │
│  │ Symbol  Qty  Entry    Current     P&L           │             │
│  │ AAPL    100  $150.25  $152.48    +$223.00       │             │
│  │ MSFT     50  $340.10  $338.90    -$60.00        │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                   │
│  ┌─ Recent Trades ─────────────────────────────────┐             │
│  │ Time        Symbol Side Qty  Price     P&L      │             │
│  │ 10:45:31    AAPL   SELL 100  $152.48  +$222.00  │             │
│  │ 10:31:24    AAPL   BUY  100  $150.26   $0.00    │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                   │
│  ┌─ Performance Metrics ┐  ┌─ System Resources ──┐               │
│  │ Sharpe Ratio: 1.85   │  │ CPU: 45.2%          │               │
│  │ Max Drawdown: -12.5% │  │ ████████████░░░░░░░ │               │
│  │ Current DD: -3.2%    │  │ Memory: 65.8%       │               │
│  │ Win Rate: 60.5%      │  │ ████████████████░░░ │               │
│  └──────────────────────┘  │ Network: 15.3ms     │               │
│                             └─────────────────────┘               │
│                                                                   │
│  ┌─ Alerts (2 unacknowledged) ────────────────────┐              │
│  │ [10:45:32] ⚠️  High drawdown detected: -12.5%   │              │
│  │ [10:30:15] ℹ️  Strategy started successfully    │              │
│  └──────────────────────────────────────────────────┘             │
│                                                                   │
│  Last updated: 10:46:15        Press 'h' for help, 'q' to quit  │
└──────────────────────────────────────────────────────────────────┘
```

## Keyboard Shortcuts

- **q** - Quit dashboard
- **h** - Toggle help screen
- **r** - Force refresh data
- **c** - Clear acknowledged alerts

## Dashboard Panels

### Strategy Status Panel
- Strategy name and type
- Current status (running/stopped/error)
- Runtime duration
- Error messages (if any)

### P&L Panel
- Today's P&L
- Total P&L
- Total trades count
- Win rate percentage

### Positions Panel
- Current open positions
- Symbol, quantity, entry price
- Current price and unrealized P&L
- Color-coded (green=profit, red=loss)

### Recent Trades Panel
- Last 5 executed trades
- Timestamp, symbol, side (BUY/SELL)
- Quantity, price, and P&L
- Color-coded by side and profit

### Performance Metrics Panel
- Sharpe ratio
- Maximum drawdown
- Current drawdown
- Win rate
- Color-coded based on thresholds

### System Resources Panel
- CPU usage with bar chart
- Memory usage with bar chart
- Network latency
- Color-coded (green/yellow/red)

### Alerts Panel
- Recent alerts and notifications
- Severity levels (error/warning/info)
- Timestamp and message
- Acknowledgment status

## Alert Rules

The dashboard includes built-in alert rules:

1. **High Loss**: Triggered when daily P&L drops below -$1,000
2. **High Drawdown**: Triggered when current drawdown exceeds -10%
3. **Low Win Rate**: Triggered when win rate drops below 40% (after 10+ trades)
4. **Strategy Error**: Triggered when strategy enters error state
5. **High CPU**: Triggered when CPU usage exceeds 80%
6. **High Memory**: Triggered when memory usage exceeds 85%

## Integration with NAPI

The dashboard integrates with Neural Trader's NAPI bindings for real-time data:

- `getPortfolio()` - Fetch portfolio data
- `getPositions()` - Fetch current positions
- `calculateMetrics()` - Calculate performance metrics
- System metrics from Node.js process

When NAPI bindings are unavailable, the dashboard automatically switches to mock mode with simulated data.

## Mock Mode

Mock mode generates realistic simulated data for testing:

- Simulated P&L with volatility
- Random position generation
- Periodic trade execution
- System metrics simulation

Perfect for:
- Testing dashboard functionality
- Development without live data
- Demonstrations and presentations

## Architecture

### Core Components

1. **StrategyMonitor** (`src/cli/lib/strategy-monitor.js`)
   - Monitors strategy execution
   - Collects real-time data
   - Emits update events

2. **MetricsCollector** (`src/cli/lib/metrics-collector.js`)
   - Collects and aggregates metrics
   - Maintains metric history
   - Calculates statistics

3. **AlertManager** (`src/cli/lib/alert-manager.js`)
   - Manages alert rules
   - Triggers alerts based on conditions
   - Tracks acknowledgments

4. **WebSocketClient** (`src/cli/lib/websocket-client.js`)
   - Manages WebSocket connections
   - Handles subscriptions
   - Real-time data streaming

### Dashboard Components

All components built with Ink (React for CLI):

- **Dashboard.jsx** - Main dashboard container
- **StrategyPanel.jsx** - Strategy status display
- **PositionsPanel.jsx** - Positions table
- **PnLPanel.jsx** - P&L metrics
- **TradesPanel.jsx** - Recent trades list
- **MetricsPanel.jsx** - Performance metrics
- **SystemPanel.jsx** - System resources
- **AlertsPanel.jsx** - Alerts and notifications

## Examples

### Monitor a Live Strategy

```bash
# Start monitoring a running strategy
neural-trader monitor my-momentum-strategy
```

### Test Dashboard with Mock Data

```bash
# Launch dashboard in mock mode
neural-trader monitor --mock

# This displays simulated data that updates every second
```

### Check Strategy Performance

```bash
# View detailed metrics
neural-trader monitor metrics my-momentum-strategy

# Check recent logs
neural-trader monitor logs my-momentum-strategy

# View active alerts
neural-trader monitor alerts
```

### Monitor Multiple Strategies

```bash
# List all running strategies
neural-trader monitor list

# Pick one to monitor in detail
neural-trader monitor strategy-id
```

## Troubleshooting

### Dashboard not updating

- Check that the strategy is running
- Verify NAPI bindings are loaded (`neural-trader doctor`)
- Try mock mode: `neural-trader monitor --mock`

### Missing data or panels

- Ensure all dependencies are installed: `npm install`
- Check that Ink and React are available
- Verify Node.js version >= 18

### Performance issues

- Reduce update interval in code (default: 1 second)
- Close other terminal applications
- Check system resources

## Development

### Adding Custom Panels

1. Create new component in `src/cli/commands/monitor/components/`
2. Import in `Dashboard.jsx`
3. Add to dashboard layout

### Adding Alert Rules

```javascript
alertManager.addRule('my_rule', strategy => {
  return strategy.metrics.someValue > threshold;
}, { severity: 'warning' });
```

### Customizing Update Interval

```javascript
const strategyMonitor = new StrategyMonitor({
  mockMode: false,
  updateInterval: 2000 // 2 seconds
});
```

## Future Enhancements

- [ ] Multiple strategy monitoring in single dashboard
- [ ] Historical charts with ascii-chart or blessed-contrib
- [ ] Export metrics to CSV/JSON
- [ ] Email/SMS alerts
- [ ] WebSocket streaming for sub-second updates
- [ ] Strategy control (pause/resume/stop)
- [ ] Backtesting results comparison
- [ ] Custom dashboard layouts

## See Also

- [Neural Trader Documentation](../README.md)
- [CLI Commands](../bin/cli.js)
- [NAPI Bindings](../index.js)
- [Ink Documentation](https://github.com/vadimdemedes/ink)
