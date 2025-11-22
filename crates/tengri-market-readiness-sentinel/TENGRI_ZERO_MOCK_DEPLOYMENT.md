# TENGRI Zero-Mock Sentinel Deployment Guide

## Overview

The TENGRI Zero-Mock Sentinel is a comprehensive real integration testing infrastructure that enforces zero tolerance for synthetic, mock, or fake data in integration testing environments. This system ensures that all testing uses actual systems, real data sources, and live integrations.

## Core Components

### 1. Zero-Mock Detection Engine
- **Purpose**: Detects and prevents any synthetic/mock data usage
- **Features**: 
  - Real-time code scanning
  - Pattern-based violation detection
  - Automated rollback on violations
  - Agent behavior modification

### 2. Real Database Integration Tester
- **Purpose**: Tests with live PostgreSQL/Redis instances
- **Features**:
  - Live database connectivity testing
  - Real transaction validation
  - Performance benchmarking
  - Data integrity verification

### 3. Live API Integration Tester
- **Purpose**: Tests with actual exchange APIs (testnet/sandbox)
- **Features**:
  - Real exchange API connectivity
  - Authentication testing
  - WebSocket connection validation
  - Rate limiting compliance

### 4. Real Network Validation Tester
- **Purpose**: Validates actual network communications
- **Features**:
  - Network connectivity testing
  - DNS resolution validation
  - SSL certificate verification
  - Latency and performance monitoring

## Installation

### Prerequisites

1. **Rust Environment**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update stable
   ```

2. **Required System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y build-essential pkg-config libssl-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install openssl-devel
   
   # macOS
   xcode-select --install
   brew install openssl
   ```

3. **Database Dependencies**
   ```bash
   # PostgreSQL
   sudo apt-get install postgresql postgresql-contrib libpq-dev
   
   # Redis
   sudo apt-get install redis-server
   ```

### Building the Sentinel

1. **Clone and Build**
   ```bash
   cd /path/to/tengri-market-readiness-sentinel
   cargo build --release
   ```

2. **Run Tests**
   ```bash
   cargo test
   ```

3. **Install Binary**
   ```bash
   cargo install --path .
   ```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database Configuration
POSTGRESQL_URL=postgresql://user:password@localhost:5432/trading
REDIS_URL=redis://localhost:6379/0

# Exchange API Keys (Testnet)
BINANCE_TESTNET_API_KEY=your_binance_testnet_api_key
BINANCE_TESTNET_API_SECRET=your_binance_testnet_api_secret
COINBASE_SANDBOX_API_KEY=your_coinbase_sandbox_api_key
COINBASE_SANDBOX_API_SECRET=your_coinbase_sandbox_api_secret
COINBASE_SANDBOX_PASSPHRASE=your_coinbase_sandbox_passphrase
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_API_SECRET=your_kraken_api_secret

# Monitoring Configuration
MONITORING_ENABLED=true
MONITORING_INTERVAL=60
ALERT_WEBHOOK_URL=https://your-webhook-url.com/alerts

# Zero-Mock Configuration
ZERO_MOCK_ENFORCEMENT=strict
VIOLATION_TOLERANCE=0
AUTO_ROLLBACK_ENABLED=true
```

### Configuration File

Create `config/production.toml`:

```toml
[zero_mock_detection]
enabled = true
real_time_scanning = true
violation_tolerance = 0
auto_rollback = true

[database_integration]
[database_integration.postgresql]
host = "localhost"
port = 5432
database = "trading"
username = "user"
password = "password"
ssl_mode = "require"
connection_pool_size = 10

[database_integration.redis]
host = "localhost"
port = 6379
database = 0
connection_pool_size = 10

[api_integration]
[api_integration.binance]
base_url = "https://testnet.binance.vision"
websocket_url = "wss://testnet.binance.vision/ws"
testnet = true
rate_limit = 10

[api_integration.coinbase]
base_url = "https://api-public.sandbox.pro.coinbase.com"
websocket_url = "wss://ws-feed-public.sandbox.pro.coinbase.com"
testnet = true
rate_limit = 10

[network_validation]
timeout_seconds = 30
max_retries = 3
parallel_tests = true

[monitoring]
enabled = true
interval_seconds = 60
metrics_retention_days = 30
alerting_enabled = true

[security]
ssl_verification = true
certificate_validation = true
auth_required = true
```

## Deployment

### Local Development

1. **Start Required Services**
   ```bash
   # Start PostgreSQL
   sudo systemctl start postgresql
   
   # Start Redis
   sudo systemctl start redis
   
   # Create database
   createdb trading
   ```

2. **Run Zero-Mock Sentinel**
   ```bash
   # Validate configuration
   ./target/release/tengri-sentinel validate-config
   
   # Run comprehensive validation
   ./target/release/tengri-sentinel validate --yes
   
   # Start monitoring mode
   ./target/release/tengri-sentinel monitor --ui --port 8080
   ```

### Production Deployment

#### Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM rust:1.70 as builder
   
   WORKDIR /app
   COPY . .
   RUN cargo build --release
   
   FROM debian:bullseye-slim
   
   RUN apt-get update && apt-get install -y \
       ca-certificates \
       libssl1.1 \
       libpq5 \
       && rm -rf /var/lib/apt/lists/*
   
   COPY --from=builder /app/target/release/tengri-sentinel /usr/local/bin/
   COPY config/ /app/config/
   
   WORKDIR /app
   EXPOSE 8080
   
   CMD ["tengri-sentinel", "monitor", "--ui", "--port", "8080"]
   ```

2. **Docker Compose Setup**
   ```yaml
   version: '3.8'
   
   services:
     tengri-sentinel:
       build: .
       ports:
         - "8080:8080"
       environment:
         - POSTGRESQL_URL=postgresql://postgres:password@postgres:5432/trading
         - REDIS_URL=redis://redis:6379/0
       depends_on:
         - postgres
         - redis
       volumes:
         - ./config:/app/config
         - ./logs:/app/logs
   
     postgres:
       image: postgres:15
       environment:
         POSTGRES_DB: trading
         POSTGRES_USER: postgres
         POSTGRES_PASSWORD: password
       ports:
         - "5432:5432"
       volumes:
         - postgres_data:/var/lib/postgresql/data
   
     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data
   
   volumes:
     postgres_data:
     redis_data:
   ```

#### Kubernetes Deployment

1. **ConfigMap**
   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: tengri-sentinel-config
   data:
     production.toml: |
       [zero_mock_detection]
       enabled = true
       real_time_scanning = true
       # ... rest of config
   ```

2. **Deployment**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: tengri-sentinel
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: tengri-sentinel
     template:
       metadata:
         labels:
           app: tengri-sentinel
       spec:
         containers:
         - name: tengri-sentinel
           image: tengri-sentinel:latest
           ports:
           - containerPort: 8080
           env:
           - name: POSTGRESQL_URL
             valueFrom:
               secretKeyRef:
                 name: database-secrets
                 key: postgresql-url
           - name: REDIS_URL
             valueFrom:
               secretKeyRef:
                 name: database-secrets
                 key: redis-url
           volumeMounts:
           - name: config
             mountPath: /app/config
           livenessProbe:
             httpGet:
               path: /health
               port: 8080
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8080
             initialDelaySeconds: 5
             periodSeconds: 5
         volumes:
         - name: config
           configMap:
             name: tengri-sentinel-config
   ```

3. **Service**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: tengri-sentinel-service
   spec:
     selector:
       app: tengri-sentinel
     ports:
     - protocol: TCP
       port: 8080
       targetPort: 8080
     type: LoadBalancer
   ```

## Usage

### Command Line Interface

```bash
# Run comprehensive validation
tengri-sentinel validate

# Run specific validation phase
tengri-sentinel validate --phase database_integration

# Start monitoring mode
tengri-sentinel monitor --ui --port 8080

# Test specific exchange
tengri-sentinel test-exchange --name binance --test-type all

# Test market data feeds
tengri-sentinel test-feeds --duration 300

# Generate diagnostics
tengri-sentinel diagnostics --output diagnostics.json --include-system

# Health check
tengri-sentinel health

# Start API server
tengri-sentinel serve --port 8080 --metrics
```

### Web UI

Access the web interface at `http://localhost:8080` when running in monitor mode.

**Features:**
- Real-time validation status
- Violation detection dashboard
- Performance metrics
- Historical reports
- Configuration management

### API Endpoints

```bash
# Health check
GET /health

# Validation status
GET /api/v1/validation/status

# Run validation
POST /api/v1/validation/run

# Get violations
GET /api/v1/violations

# Get metrics
GET /api/v1/metrics

# Get reports
GET /api/v1/reports
```

## Monitoring and Alerting

### Metrics Collection

The sentinel collects comprehensive metrics:

- **Violation Metrics**: Number and types of violations detected
- **Performance Metrics**: Response times, throughput, error rates
- **System Metrics**: CPU, memory, disk, network usage
- **Integration Metrics**: Database connectivity, API response times
- **Network Metrics**: Latency, packet loss, bandwidth usage

### Alerting

Configure alerts for:

- **Critical Violations**: Zero-tolerance violations detected
- **System Failures**: Component failures or connectivity issues
- **Performance Degradation**: Response times exceed thresholds
- **Security Issues**: Authentication failures or SSL issues

### Integration with Monitoring Systems

#### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tengri-sentinel'
    static_configs:
      - targets: ['tengri-sentinel:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Grafana Dashboard

Import the provided Grafana dashboard for comprehensive monitoring.

## Security Considerations

### Authentication

- **API Keys**: Store in secure secret management systems
- **Database Credentials**: Use environment variables or secrets
- **TLS/SSL**: Enable encryption for all connections
- **Access Control**: Implement role-based access control

### Network Security

- **Firewall Rules**: Restrict access to required ports only
- **VPN Access**: Use VPN for remote access
- **Network Segmentation**: Isolate testing environment

### Data Protection

- **Encryption at Rest**: Encrypt stored data and logs
- **Encryption in Transit**: Use TLS for all communications
- **Data Retention**: Implement data retention policies
- **Audit Logging**: Maintain comprehensive audit trails

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database connectivity
   psql -h localhost -U user -d trading -c "SELECT 1;"
   redis-cli ping
   ```

2. **API Authentication Failures**
   ```bash
   # Verify API credentials
   curl -H "X-MBX-APIKEY: $BINANCE_TESTNET_API_KEY" https://testnet.binance.vision/api/v3/time
   ```

3. **Network Connectivity Issues**
   ```bash
   # Test network connectivity
   ping api.binance.com
   nslookup api.coinbase.com
   ```

4. **SSL Certificate Issues**
   ```bash
   # Check SSL certificates
   openssl s_client -connect api.binance.com:443 -servername api.binance.com
   ```

### Logging

Enable detailed logging:

```bash
# Set log level
export RUST_LOG=tengri_market_readiness_sentinel=debug

# Run with verbose logging
tengri-sentinel validate --log-level debug
```

### Debug Mode

```bash
# Enable debug mode
tengri-sentinel validate --debug
```

## Performance Tuning

### Database Optimization

- **Connection Pooling**: Adjust pool sizes based on load
- **Query Optimization**: Monitor and optimize slow queries
- **Indexing**: Ensure proper indexing for performance

### Network Optimization

- **Timeout Configuration**: Adjust timeouts for network conditions
- **Parallel Testing**: Enable parallel tests for faster execution
- **Rate Limiting**: Respect API rate limits

### System Resources

- **Memory Allocation**: Monitor memory usage and adjust heap size
- **CPU Utilization**: Balance test parallelism with CPU cores
- **Disk I/O**: Use fast storage for better performance

## Maintenance

### Regular Tasks

1. **Update Dependencies**
   ```bash
   cargo update
   cargo audit
   ```

2. **Rotate Credentials**
   - Update API keys regularly
   - Rotate database passwords
   - Renew SSL certificates

3. **Backup Data**
   ```bash
   # Backup database
   pg_dump trading > backup.sql
   
   # Backup Redis
   redis-cli save
   ```

4. **Monitor Logs**
   ```bash
   # Check for errors
   tail -f /app/logs/tengri-sentinel.log | grep ERROR
   ```

### Updates and Upgrades

1. **Testing Updates**
   - Test updates in staging environment
   - Run comprehensive validation after updates
   - Monitor for regressions

2. **Rolling Deployments**
   - Use blue-green deployments for zero downtime
   - Gradually roll out updates
   - Have rollback plan ready

## Best Practices

### Development

1. **Code Quality**
   - Follow Rust best practices
   - Write comprehensive tests
   - Use static analysis tools
   - Maintain documentation

2. **Testing**
   - Test with real data sources
   - Use testnet/sandbox environments
   - Validate all integrations
   - Monitor test coverage

### Operations

1. **Monitoring**
   - Set up comprehensive monitoring
   - Configure meaningful alerts
   - Monitor all metrics
   - Regular health checks

2. **Security**
   - Follow security best practices
   - Regular security audits
   - Keep dependencies updated
   - Use secure configurations

3. **Documentation**
   - Maintain up-to-date documentation
   - Document all configurations
   - Keep runbooks current
   - Train team members

## Support and Contributing

### Getting Help

- **Documentation**: Check this guide and inline documentation
- **Logs**: Review application logs for error details
- **Community**: Join the TENGRI community for support

### Contributing

1. **Code Contributions**
   - Fork the repository
   - Create feature branches
   - Write tests for new features
   - Submit pull requests

2. **Documentation**
   - Improve documentation
   - Add examples
   - Fix typos and errors

3. **Bug Reports**
   - Use GitHub issues
   - Provide detailed reproduction steps
   - Include logs and configuration

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This deployment guide ensures that the TENGRI Zero-Mock Sentinel enforces real integration testing with zero tolerance for synthetic or mock data. All components must use authentic data sources and live integrations for production readiness validation.
