# Neural Trader Configuration Guide

This directory contains configuration templates for different deployment environments.

## Configuration Files

- **production.toml** - Production environment (uses environment variables for secrets)
- **staging.toml** - Staging environment (paper trading enabled)
- **development.toml** - Local development (relaxed limits, verbose logging)

## Environment Variables

### Required for Production

```bash
# Database
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
export REDIS_URL="redis://host:6379/0"

# Market Data Providers
export ALPACA_API_KEY="your-api-key"
export ALPACA_API_SECRET="your-api-secret"
export BINANCE_API_KEY="your-api-key"
export BINANCE_API_SECRET="your-api-secret"

# AgentDB
export AGENTDB_PATH="/var/lib/neural-trader/agentdb"

# TLS (Production)
export TLS_CERT_PATH="/etc/neural-trader/tls/cert.pem"
export TLS_KEY_PATH="/etc/neural-trader/tls/key.pem"

# Monitoring
export JAEGER_ENDPOINT="http://jaeger:14268/api/traces"
```

### Optional Configuration

```bash
# Override config file location
export NEURAL_TRADER_CONFIG="/path/to/config.toml"

# Override specific settings
export NEURAL_TRADER_LOG_LEVEL="debug"
export NEURAL_TRADER_PORT="9000"
```

## Configuration Priority

Configuration is loaded in the following order (later sources override earlier):

1. Default values (in code)
2. Config file (`.config/*.toml`)
3. Environment variables
4. Command-line arguments

## Usage

### CLI

```bash
# Use specific config file
neural-trader --config .config/production.toml run

# Override via environment
NEURAL_TRADER_CONFIG=.config/staging.toml neural-trader run

# Override specific values
neural-trader --log-level debug --port 9000 run
```

### Docker

```bash
# Mount config file
docker run -v $(pwd)/.config/production.toml:/app/config.toml \
  -e ALPACA_API_KEY=$ALPACA_API_KEY \
  neural-trader --config /app/config.toml run

# Or use environment variables only
docker run \
  -e DATABASE_URL=$DATABASE_URL \
  -e REDIS_URL=$REDIS_URL \
  -e ALPACA_API_KEY=$ALPACA_API_KEY \
  neural-trader run
```

### Docker Compose

```yaml
services:
  neural-trader:
    image: neural-trader:latest
    volumes:
      - ./.config/production.toml:/app/config.toml
    env_file:
      - .env.production
    command: ["--config", "/app/config.toml", "run"]
```

## Security Best Practices

1. **Never commit secrets** - Use environment variables for sensitive data
2. **Use .env files** - Keep environment-specific secrets in `.env.{environment}`
3. **Add to .gitignore** - Ensure `.env*` files are gitignored
4. **Rotate credentials** - Regularly rotate API keys and secrets
5. **Least privilege** - Use read-only keys where possible
6. **Enable TLS** - Always use TLS in production
7. **Rate limiting** - Enable rate limiting to prevent abuse
8. **Audit logging** - Keep audit logs for compliance

## Configuration Sections

### [environment]
General environment settings (name, debug mode, log level)

### [server]
HTTP server configuration (host, port, workers, timeouts)

### [database]
PostgreSQL connection settings and pool configuration

### [redis]
Redis cache configuration for real-time data

### [market_data]
Market data provider settings (Alpaca, Binance, etc.)

### [execution]
Order execution limits and paper trading settings

### [risk]
Risk management limits (drawdown, position sizing, stops)

### [backtesting]
Backtesting parameters (capital, commissions, slippage)

### [neural]
Neural network configuration (models, GPU, caching)

### [agentdb]
AgentDB persistent memory settings (vector DB configuration)

### [monitoring]
Metrics, Prometheus, and Jaeger tracing configuration

### [security]
TLS, authentication, CORS, and rate limiting

### [logging]
Log format, output, rotation, and retention

### [features]
Feature flags for optional functionality

## Validation

Validate your configuration:

```bash
# Check configuration syntax
neural-trader --config .config/production.toml validate

# Dry-run to test without executing
neural-trader --config .config/production.toml --dry-run run
```

## Troubleshooting

### Connection Issues

```bash
# Test database connection
neural-trader --config .config/production.toml test-db

# Test Redis connection
neural-trader --config .config/production.toml test-redis

# Test market data provider
neural-trader --config .config/production.toml test-market-data
```

### Performance Tuning

1. **Database pool size** - Adjust based on worker count (2-4x workers)
2. **Redis pool** - Usually 5-10 connections sufficient
3. **Worker count** - Match CPU cores for CPU-bound tasks
4. **Rate limits** - Stay within provider limits to avoid throttling

### Common Errors

**Error: Database connection timeout**
- Check DATABASE_URL is correct
- Verify network connectivity
- Increase connection_timeout_s

**Error: Rate limit exceeded**
- Reduce max_orders_per_second
- Check rate_limit_per_minute matches provider
- Use multiple API keys with load balancing

**Error: TLS certificate not found**
- Verify TLS_CERT_PATH and TLS_KEY_PATH
- Check file permissions (readable by process)
- Use Let's Encrypt for free certificates

## Migration from Python

If migrating from the Python version:

1. Export your existing configuration
2. Map Python env vars to Rust config
3. Test in development environment first
4. Validate all connections work
5. Run backtests to verify behavior
6. Deploy to staging before production

## Support

For configuration help:
- Documentation: https://docs.neural-trader.io
- Issues: https://github.com/ruvnet/neural-trader/issues
- Discord: https://discord.gg/neural-trader
