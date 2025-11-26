# Monitor Command - Quick Start Guide

## Installation

Dependencies are already installed with Neural Trader:
- ink (React for CLI)
- react
- ink-table
- ink-spinner
- ink-text-input

## Quick Start

### 1. Launch Dashboard

```bash
# Start monitoring with mock data
neural-trader monitor --mock

# Monitor a specific strategy
neural-trader monitor my-strategy
```

### 2. View All Strategies

```bash
neural-trader monitor list
```

### 3. Check Performance

```bash
neural-trader monitor metrics strategy-id
```

### 4. View Logs

```bash
neural-trader monitor logs strategy-id
```

### 5. Check Alerts

```bash
neural-trader monitor alerts
```

## Dashboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit dashboard |
| `h` | Toggle help screen |
| `r` | Force refresh |
| `c` | Clear acknowledged alerts |

## Dashboard Panels

The dashboard displays 7 real-time panels:

1. **Strategy Status** - Name, type, status, runtime
2. **P&L** - Today's and total profit/loss, win rate
3. **Positions** - Open positions with P&L
4. **Recent Trades** - Last 5 executed trades
5. **Performance Metrics** - Sharpe ratio, drawdowns
6. **System Resources** - CPU, memory, network
7. **Alerts** - Warnings and notifications

## Color Coding

- 🟢 **Green** = Profit, good status, success
- 🔴 **Red** = Loss, errors, critical alerts
- 🟡 **Yellow** = Warnings, neutral states
- 🔵 **Cyan** = Information, headers
- ⚪ **Gray** = Inactive, secondary info

## Alert Types

Built-in alerts monitor:
- High losses (> $1,000)
- High drawdown (> 10%)
- Low win rate (< 40%)
- Strategy errors
- High CPU usage (> 80%)
- High memory usage (> 85%)

## Examples

### Monitor with Mock Data
```bash
# Perfect for testing and demonstrations
neural-trader monitor --mock
```

### Follow Strategy Logs
```bash
# Watch logs in real-time
neural-trader monitor logs my-strategy --follow
```

### Filter Alerts by Severity
```bash
# Show only errors
neural-trader monitor alerts --severity=error

# Show all alerts (including acknowledged)
neural-trader monitor alerts --all
```

## Tips

1. **Use Mock Mode for Testing**: Always test with `--mock` flag first
2. **Terminal Size**: Works best with 120+ character width
3. **Multiple Strategies**: Use `monitor list` to see all strategies
4. **Quick Metrics**: Use `monitor metrics` for snapshot without dashboard
5. **Background Monitoring**: Run dashboard in tmux/screen session

## Troubleshooting

### Dashboard not updating?
- Press `r` to force refresh
- Check strategy is running with `monitor list`
- Verify NAPI bindings with `neural-trader doctor`

### NAPI errors?
- Use `--mock` flag for testing
- Build bindings: `npm run build`

### Terminal display issues?
- Ensure terminal supports ANSI colors
- Try increasing terminal width
- Update terminal emulator

## Next Steps

1. Read full documentation: `docs/MONITOR_COMMAND.md`
2. Check implementation details: `docs/MONITOR_IMPLEMENTATION.md`
3. Explore example strategies in `packages/examples/`

## Architecture

```
Monitor Command
├── Dashboard (Ink/React)
│   ├── Strategy Panel
│   ├── P&L Panel
│   ├── Positions Panel
│   ├── Trades Panel
│   ├── Metrics Panel
│   ├── System Panel
│   └── Alerts Panel
├── Strategy Monitor (Real-time data)
├── Metrics Collector (Time-series)
├── Alert Manager (Rule engine)
└── WebSocket Client (Streaming)
```

## Support

- Documentation: `/docs/MONITOR_COMMAND.md`
- Issues: https://github.com/ruvnet/neural-trader/issues
- Examples: `/packages/examples/`

---

**Happy Monitoring!** 📊⚡
