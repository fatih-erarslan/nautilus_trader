# AI News Trading Platform - Troubleshooting Guide

## Table of Contents
1. [Common Issues](#common-issues)
2. [MCP Server Issues](#mcp-server-issues)
3. [News Aggregation Issues](#news-aggregation-issues)
4. [Trading & Strategy Issues](#trading--strategy-issues)
5. [Performance Issues](#performance-issues)
6. [Integration Issues](#integration-issues)
7. [Error Messages](#error-messages)
8. [Debugging Tools](#debugging-tools)

## Common Issues

### Issue: MCP Server Not Starting
**Symptoms:**
- Claude Code shows "MCP server not responding"
- Tools not available in Claude Code

**Solutions:**
1. Check Python installation:
   ```bash
   python --version  # Should be 3.8+
   ```

2. Verify dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Check MCP configuration:
   ```bash
   cat .roo/mcp.json
   ```

4. Test server manually:
   ```bash
   python src/mcp/mcp_server_integrated.py
   ```

### Issue: GPU Not Detected
**Symptoms:**
- Tools report `gpu_available: false`
- Slow neural forecasting

**Solutions:**
1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Verify PyTorch CUDA:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

3. Install GPU dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install cupy-cuda11x
   ```

## MCP Server Issues

### Issue: Tool Timeout Errors
**Symptoms:**
- "Tool execution timed out" errors
- Slow response times

**Solutions:**
1. Check system resources:
   ```python
   # Use the system metrics tool
   mcp__ai-news-trader__get_system_metrics
   - metrics: ["cpu", "memory", "latency"]
   ```

2. Reduce concurrent operations:
   - Decrease `max_concurrent_requests` in configuration
   - Use smaller batch sizes

3. Enable GPU acceleration where available:
   ```python
   # Always set use_gpu: true for supported tools
   mcp__ai-news-trader__neural_forecast
   - use_gpu: true
   ```

### Issue: Tool Not Found
**Symptoms:**
- "Unknown tool" errors
- Missing `mcp__ai-news-trader__` prefix

**Solutions:**
1. Verify server version:
   ```python
   mcp__ai-news-trader__ping
   ```

2. List available tools:
   ```python
   mcp__ai-news-trader__list_strategies
   ```

3. Check integration status:
   ```python
   # Get resource
   mcp.get_resource("integration://status")
   ```

## News Aggregation Issues

### Issue: No News Data
**Symptoms:**
- Empty news results
- "News aggregation not available" errors

**Solutions:**
1. Check API keys:
   ```bash
   # Verify environment variables
   echo $ALPHA_VANTAGE_API_KEY
   echo $NEWSAPI_API_KEY
   echo $FINNHUB_API_KEY
   ```

2. Test news providers:
   ```python
   mcp__ai-news-trader__get_news_provider_status
   ```

3. Start news collection:
   ```python
   mcp__ai-news-trader__control_news_collection
   - action: "start"
   - symbols: ["AAPL"]
   ```

### Issue: Redis Connection Failed
**Symptoms:**
- Cache errors
- Slow news fetching

**Solutions:**
1. Check Redis service:
   ```bash
   redis-cli ping  # Should return PONG
   ```

2. Start Redis:
   ```bash
   # Docker
   docker run -d -p 6379:6379 redis:alpine
   
   # Or system service
   sudo systemctl start redis
   ```

3. Configure Redis URL:
   ```python
   # In config.py
   REDIS_URL = "redis://localhost:6379"
   ```

## Trading & Strategy Issues

### Issue: Strategy Not Found
**Symptoms:**
- "Strategy 'X' not found" errors
- Empty strategy list

**Solutions:**
1. List available strategies:
   ```python
   mcp__ai-news-trader__list_strategies
   ```

2. Check strategy info:
   ```python
   mcp__ai-news-trader__get_strategy_info
   - strategy: "momentum_trading"
   ```

3. Verify model files:
   ```bash
   ls models/
   ls neural_models/
   ```

### Issue: Trade Execution Failed
**Symptoms:**
- "Failed to execute trade" errors
- Risk limit exceeded messages

**Solutions:**
1. Check risk limits:
   ```python
   mcp__ai-news-trader__execute_multi_asset_trade
   - risk_limit: 100000  # Increase if needed
   ```

2. Verify portfolio status:
   ```python
   mcp__ai-news-trader__get_portfolio_status
   - include_analytics: true
   ```

3. Monitor strategy health:
   ```python
   mcp__ai-news-trader__monitor_strategy_health
   - strategy: "your_strategy"
   ```

## Performance Issues

### Issue: High Latency
**Symptoms:**
- Slow tool responses
- P95 latency > 1000ms

**Solutions:**
1. Check system load:
   ```python
   mcp__ai-news-trader__get_system_metrics
   - metrics: ["cpu", "memory", "latency", "throughput"]
   - include_history: true
   ```

2. Enable parallel execution:
   ```python
   mcp__ai-news-trader__execute_multi_asset_trade
   - execute_parallel: true
   ```

3. Optimize batch sizes:
   - Reduce number of symbols per request
   - Use shorter forecast horizons

### Issue: Memory Errors
**Symptoms:**
- Out of memory errors
- Server crashes

**Solutions:**
1. Monitor memory usage:
   ```bash
   free -h
   top -p $(pgrep -f mcp_server)
   ```

2. Reduce model complexity:
   - Use smaller batch sizes
   - Limit concurrent neural operations

3. Enable swap (temporary fix):
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

## Integration Issues

### Issue: Module Import Errors
**Symptoms:**
- "ModuleNotFoundError" messages
- Components not available

**Solutions:**
1. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

2. Install missing dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

3. Verify directory structure:
   ```bash
   tree src/ -d -L 2
   ```

### Issue: Test Failures
**Symptoms:**
- Coverage tests failing
- Integration tests timeout

**Solutions:**
1. Run specific test suites:
   ```bash
   # Unit tests only
   pytest tests/unit -v
   
   # Integration tests
   pytest tests/integration -v
   
   # With timeout
   pytest tests/load --timeout=300
   ```

2. Check test fixtures:
   ```bash
   # Verify test data exists
   ls tests/fixtures/
   ```

3. Skip GPU tests if no GPU:
   ```bash
   pytest -m "not gpu_required"
   ```

## Error Messages

### Common Error Codes and Solutions

| Error Code | Message | Solution |
|------------|---------|----------|
| ERR_001 | "News aggregation not available" | Install news dependencies and set API keys |
| ERR_002 | "Strategy manager not available" | Check strategy_manager.py import |
| ERR_003 | "GPU not available" | Install CUDA and GPU libraries |
| ERR_004 | "Rate limit exceeded" | Wait or use different API keys |
| ERR_005 | "Invalid date format" | Use YYYY-MM-DD format |
| ERR_006 | "Risk limit exceeded" | Increase risk_limit parameter |
| ERR_007 | "Model not found" | Run model training or download models |
| ERR_008 | "Cache connection failed" | Start Redis service |
| ERR_009 | "Correlation matrix failed" | Reduce number of assets |
| ERR_010 | "Timeout error" | Increase timeout or use GPU |

## Debugging Tools

### 1. Enable Debug Logging
```python
# In mcp_server_integrated.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Test Individual Components
```python
# Test news aggregator
from src.integrations.news_aggregator import UnifiedNewsAggregator
aggregator = UnifiedNewsAggregator()
await aggregator.health_check()

# Test strategy manager
from src.mcp.trading.strategy_manager import StrategyManager
manager = StrategyManager()
await manager.initialize()
```

### 3. Monitor Real-time Metrics
```python
# Continuous monitoring script
import asyncio

async def monitor():
    while True:
        metrics = await mcp.call_tool("get_system_metrics", {
            "metrics": ["cpu", "memory", "latency", "throughput"]
        })
        print(f"CPU: {metrics['current_metrics']['cpu']['usage_percent']}%")
        print(f"Memory: {metrics['current_metrics']['memory']['usage_percent']}%")
        await asyncio.sleep(5)

asyncio.run(monitor())
```

### 4. Performance Profiling
```bash
# Profile MCP server
python -m cProfile -o profile.stats src/mcp/mcp_server_integrated.py

# Analyze profile
python -m pstats profile.stats
```

### 5. Network Debugging
```bash
# Check API connectivity
curl -X GET "https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=$ALPHA_VANTAGE_API_KEY"

# Test Redis
redis-cli ping

# Check port usage
netstat -tulpn | grep -E "(6379|8080)"
```

## Getting Help

### 1. Check Logs
```bash
# MCP server logs
tail -f ~/.mcp/logs/ai-news-trader.log

# System logs
journalctl -u mcp-server -f
```

### 2. Generate Diagnostic Report
```python
# Run diagnostic tool
python scripts/diagnose_system.py

# Generate coverage report
python scripts/generate_coverage.py
```

### 3. Community Resources
- GitHub Issues: Report bugs and feature requests
- Documentation: Check `/docs` folder for detailed guides
- API Reference: See CLAUDE.md for complete tool reference

### 4. Emergency Recovery
If the system is completely broken:
```bash
# 1. Stop all services
pkill -f mcp_server
docker-compose down

# 2. Clear cache
redis-cli FLUSHALL

# 3. Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# 4. Restart services
docker-compose up -d
python src/mcp/mcp_server_integrated.py
```

## Performance Tuning Tips

1. **Use GPU whenever possible**: 10-1000x speedup for neural operations
2. **Enable caching**: Set appropriate cache TTLs
3. **Batch operations**: Process multiple symbols together
4. **Optimize timeouts**: Balance between reliability and speed
5. **Monitor continuously**: Use system metrics tools regularly

Remember: Most issues can be resolved by checking dependencies, verifying configurations, and ensuring all services are running properly.