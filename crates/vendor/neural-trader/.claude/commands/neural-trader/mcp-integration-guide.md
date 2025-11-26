# MCP Integration Guide for Claude Code

## Overview
This guide covers advanced integration topics for using MCP tools with Claude Code, including memory persistence, error handling, monitoring, and production deployment patterns.

## Memory Integration with MCP

### Storing MCP Results
```
"Run neural forecast for AAPL and store the results as 'aapl_forecast_daily'"

# Claude Code will:
1. Execute: ai-news-trader:neural_forecast
2. Store: ./claude-flow memory store "aapl_forecast_daily" <results>
```

### Using Stored Data in MCP Calls
```
"Use the stored momentum parameters to backtest on TSLA"

# Claude Code will:
1. Retrieve: ./claude-flow memory get "momentum_params"
2. Execute: ai-news-trader:run_backtest with retrieved parameters
```

### Building Trading Memory
```python
# Progressive memory building
"Every day at close:
1. Run portfolio analysis
2. Store as 'portfolio_[date]'
3. Compare to yesterday
4. Track changes over time"

# Result: Historical portfolio evolution
```

## Error Handling Patterns

### Graceful Fallbacks
```
"Try neural forecast, if it fails use technical analysis instead"

# Claude Code implements:
try:
    ai-news-trader:neural_forecast
except:
    ai-news-trader:quick_analysis
    notify: "Using technical analysis fallback"
```

### Retry Logic
```
"Get real-time data with retries"

# Automatic retry pattern:
- Attempt 1: Immediate
- Attempt 2: After 5 seconds
- Attempt 3: After 30 seconds
- Fallback: Use cached data
```

### Circuit Breakers
```
"Stop trading if:
- More than 3 errors in 5 minutes
- Connection lost for >60 seconds
- Risk limits breached"
```

## Production Deployment

### Health Monitoring
```
"Set up monitoring for MCP tools:
1. Ping server every 5 minutes
2. Check tool response times
3. Validate data freshness
4. Alert on anomalies"

# Monitoring dashboard:
- Server uptime: 99.9%
- Avg response time: 245ms
- Failed requests: 0.1%
- GPU utilization: 78%
```

### Performance Optimization
```python
# Batch operations
"Analyze these 20 stocks efficiently"
# Claude Code will batch into optimal groups

# Caching strategy
"Cache frequently used data:
- Static correlations: 24 hours
- Technical indicators: 5 minutes
- News sentiment: 30 minutes"

# GPU optimization
"Always use GPU for:
- Neural operations
- Correlation matrices >10 assets
- Backtests >1 year"
```

### Load Balancing
```
"Distribute requests optimally:
- Neural forecasts: GPU queue
- Quick analysis: CPU pool
- News analysis: Dedicated worker
- Risk calculations: Priority queue"
```

## Advanced Integration Patterns

### Event-Driven Architecture
```python
# WebSocket integration
"Connect to real-time feed and:
- On price spike: run quick_analysis
- On news alert: analyze_news
- On risk breach: emergency assessment"

# Event handlers
on_market_open: morning_routine()
on_volatility_spike: risk_check()
on_news_event: sentiment_analysis()
```

### Multi-System Integration
```
"Integrate MCP with:
1. Trading platform (execute orders)
2. Risk system (position limits)
3. Compliance (trade approval)
4. Accounting (P&L tracking)"

# Data flow:
MCP Analysis → Decision → Compliance Check → Execution → Booking
```

### API Gateway Pattern
```python
# Single entry point for all requests
"Route trading requests:
/analyze → quick_analysis
/forecast → neural_forecast
/risk → risk_analysis
/trade → execute_trade"

# Benefits:
- Rate limiting
- Authentication
- Logging
- Monitoring
```

## Security & Compliance

### Secure Credential Management
```
"Never expose credentials:
- API keys in environment variables
- Encrypted storage for sensitive data
- Audit trail for all operations
- Role-based access control"
```

### Compliance Integration
```
"Before each trade:
1. Check restricted list
2. Verify position limits
3. Log decision rationale
4. Store for audit"
```

## Debugging & Troubleshooting

### Verbose Mode
```
"Debug neural forecast for AAPL with full details"

# Returns:
- Input preprocessing steps
- Model selection logic
- Feature engineering
- Inference details
- Post-processing
```

### Trace Analysis
```
"Trace the full execution path for my morning routine"

# Shows:
1. Tools called in sequence
2. Data flow between tools
3. Time spent in each step
4. Any errors or warnings
```

### Performance Profiling
```
"Profile the portfolio optimization workflow"

# Results:
- Total time: 4.3s
- Neural forecast: 1.2s (28%)
- Correlation: 0.8s (19%)
- Risk analysis: 2.1s (49%)
- Other: 0.2s (4%)
```

## Integration Testing

### Test Scenarios
```python
# Happy path testing
test_successful_trade_flow()

# Edge cases
test_market_closed()
test_invalid_symbol()
test_extreme_parameters()

# Failure scenarios
test_network_timeout()
test_gpu_unavailable()
test_data_corruption()
```

### Mock Mode
```
"Run in test mode without real trades:
- Use historical data
- Simulate executions
- Track virtual P&L
- Validate logic"
```

## Scaling Considerations

### Concurrent Requests
```
"Handle multiple users:
- User A: Portfolio analysis
- User B: Neural training
- User C: Backtesting
All running simultaneously"

# Resource allocation:
- GPU: 80% neural, 20% other
- CPU: Load balanced
- Memory: 32GB per user max
```

### Data Management
```
"Optimize data storage:
- Hot data: Redis (1 day)
- Warm data: PostgreSQL (1 month)
- Cold data: S3 (archive)
- Purge policy: 2 years"
```

## Best Practices Summary

### DO:
1. **Use memory** for persistent state
2. **Handle errors** gracefully
3. **Monitor everything** in production
4. **Batch similar operations**
5. **Cache appropriately**
6. **Test thoroughly**
7. **Document workflows**

### DON'T:
1. **Don't hardcode** credentials
2. **Don't ignore** error responses
3. **Don't skip** validation
4. **Don't overload** the server
5. **Don't bypass** risk checks

## Quick Integration Checklist

- [ ] MCP server connected and responding
- [ ] Memory system initialized
- [ ] Error handlers implemented
- [ ] Monitoring active
- [ ] Security measures in place
- [ ] Test suite passing
- [ ] Documentation complete
- [ ] Backup procedures ready

## Support Resources

### Getting Help
```
"If you encounter issues:
1. Check server status: ping
2. Review recent logs
3. Test with simple request
4. Check documentation
5. Contact support"
```

### Common Solutions
- **Timeout errors**: Increase MCP_TIMEOUT
- **GPU errors**: Check CUDA installation
- **Memory errors**: Increase allocation
- **Connection errors**: Verify network settings

Remember: Claude Code handles most integration complexity automatically - focus on describing what you want to achieve!