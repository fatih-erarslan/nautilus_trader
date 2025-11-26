# Polymarket API Setup Guide

This guide will help you set up the real Polymarket API integration for the AI News Trader system.

## Prerequisites

1. Polymarket account
2. API credentials (API Key, Secret, and Passphrase)
3. Private key for transaction signing

## Step 1: Get Polymarket API Credentials

1. Visit [Polymarket](https://polymarket.com) and sign in
2. Navigate to your account settings
3. Generate API credentials (if not already done)
4. Save your:
   - API Key
   - API Secret  
   - API Passphrase
   - Private Key (for order signing)

## Step 2: Set Environment Variables

Set the following environment variables before running the MCP server:

```bash
# Required for API authentication
export POLYMARKET_API_KEY="your-api-key"
export POLYMARKET_PRIVATE_KEY="your-private-key"

# Optional configuration
export POLYMARKET_ENVIRONMENT="production"  # or "staging" for testing
export POLYMARKET_CLOB_URL="https://clob.polymarket.com"
export POLYMARKET_GAMMA_URL="https://gamma-api.polymarket.com"
export POLYMARKET_WS_URL="wss://ws.polymarket.com"
export POLYMARKET_RATE_LIMIT="100"  # requests per minute
export POLYMARKET_DEBUG="false"
export POLYMARKET_LOG_LEVEL="INFO"
```

### For permanent setup (bash/zsh):

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Polymarket API Configuration
export POLYMARKET_API_KEY="your-api-key"
export POLYMARKET_PRIVATE_KEY="your-private-key"
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Step 3: Alternative Configuration File

You can also create a configuration file at one of these locations:
- `./polymarket.json` (current directory)
- `~/.polymarket/config.json` (home directory)
- `/etc/polymarket/config.json` (system-wide)

Example `polymarket.json`:
```json
{
  "clob_url": "https://clob.polymarket.com",
  "gamma_url": "https://gamma-api.polymarket.com",
  "ws_url": "wss://ws.polymarket.com",
  "rate_limit": 100,
  "timeout": 30,
  "max_retries": 3,
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "enable_websocket": true,
  "enable_gpu_acceleration": true,
  "enable_caching": true,
  "enable_metrics": true
}
```

**Note**: API keys should always be set via environment variables, not in config files.

## Step 4: Verify Setup

Test your configuration by running:

```bash
# Start the MCP server
python src/mcp/mcp_server_enhanced.py

# In another terminal, test the Polymarket tools
# The server will automatically use the real API if credentials are configured
```

## Step 5: Using the Real API

Once configured, all Polymarket MCP tools will automatically use the real API:

- `mcp__ai-news-trader__get_prediction_markets_tool` - Lists real markets
- `mcp__ai-news-trader__analyze_market_sentiment_tool` - Analyzes real market data
- `mcp__ai-news-trader__get_market_orderbook_tool` - Gets real orderbook data
- `mcp__ai-news-trader__place_prediction_order_tool` - Places real orders (be careful!)
- `mcp__ai-news-trader__get_prediction_positions_tool` - Shows your real positions
- `mcp__ai-news-trader__calculate_expected_value_tool` - Calculates EV on real markets

## Important Notes

1. **Demo Mode Safety**: By default, order placement operates in demo mode. To enable real trading, you must explicitly set `demo_mode=false` in the configuration.

2. **Rate Limits**: The Polymarket API has rate limits. The client automatically handles rate limiting, but be mindful of your usage.

3. **API Costs**: Some Polymarket API endpoints may have associated costs. Check Polymarket's documentation for details.

4. **Error Handling**: If API credentials are not set or invalid, the system will automatically fall back to mock data mode.

## Troubleshooting

### Issue: "API key is required for production environment"
**Solution**: Ensure `POLYMARKET_API_KEY` environment variable is set.

### Issue: "Failed to initialize Polymarket clients"
**Solution**: Check that all required environment variables are set correctly.

### Issue: Rate limit errors
**Solution**: Reduce request frequency or increase `POLYMARKET_RATE_LIMIT`.

### Issue: Authentication failures
**Solution**: Verify your API credentials are correct and not expired.

## API Documentation

For more details on the Polymarket API:
- [Polymarket Documentation](https://docs.polymarket.com/)
- [CLOB API Reference](https://docs.polymarket.com/developers/CLOB/)
- [Authentication Guide](https://docs.polymarket.com/developers/CLOB/authentication)

## Security Best Practices

1. **Never commit API keys** to version control
2. Use environment variables or secure secret management
3. Rotate API keys regularly
4. Monitor API usage for suspicious activity
5. Use read-only keys when possible for analysis tools
6. Enable 2FA on your Polymarket account

## Support

For issues with:
- **Polymarket API**: Contact Polymarket support
- **AI News Trader Integration**: Open an issue on the GitHub repository