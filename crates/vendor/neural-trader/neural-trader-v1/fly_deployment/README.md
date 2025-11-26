# Fly.io GPU Deployment Guide for RuvTrade

This directory contains all necessary files for deploying the RuvTrade GPU-accelerated trading platform to Fly.io.

## 🚀 Quick Start

### 1. Prerequisites

- Fly.io account with GPU access enabled
- Docker installed locally (for testing)
- Sufficient credits for GPU instances (~$900/month for A100-40GB)

### 2. Authentication

First, authenticate with Fly.io:

```bash
/home/codespace/.fly/bin/flyctl auth login
```

### 3. Deploy

Run the automated deployment script:

```bash
cd /workspaces/ai-news-trader/fly_deployment
./scripts/deploy.sh
```

The script will:
- Create the `ruvtrade` app
- Configure multi-region deployment
- Set up persistent storage
- Configure secrets
- Build and deploy the Docker container

### 4. Access Your App

Once deployed, your app will be available at: **https://ruvtrade.fly.dev**

## 📁 File Structure

```
fly_deployment/
├── Dockerfile              # GPU-optimized container
├── fly.toml                # Fly.io configuration
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Local testing
├── .dockerignore          # Docker build optimization
├── scripts/
│   ├── deploy.sh          # Automated deployment
│   └── scale.sh           # Scaling management
└── README.md              # This file
```

## 🔧 Configuration Details

### GPU Configuration (fly.toml)

```toml
[vm]
  size = "a100-40gb"        # NVIDIA A100 40GB GPU
  memory = "32gb"           # 32GB RAM
  cpus = 8                  # 8 CPU cores

[gpu]
  type = "nvidia-a100-40gb"
  driver_version = "525"
  cuda_version = "11.8"
```

### Regions

- **Primary**: ORD (Chicago)
- **Backup**: FRA (Frankfurt), NRT (Tokyo), SYD (Sydney)

### Auto-scaling

- **Min instances**: 1
- **Max instances**: 3
- **Auto-stop**: After 10 minutes of inactivity
- **Auto-start**: On incoming requests

## 🔒 Environment Variables & Secrets

### Required Secrets

Set these via `flyctl secrets set`:

```bash
flyctl secrets set API_KEY="your_trading_api_key" --app ruvtrade
flyctl secrets set REDIS_URL="redis://your-redis-url:6379" --app ruvtrade
flyctl secrets set DATABASE_URL="postgresql://user:pass@host:5432/db" --app ruvtrade
```

### Environment Variables (in fly.toml)

```toml
[env]
  CUDA_VISIBLE_DEVICES = "0"
  TRADING_MODE = "production"
  MAX_CONCURRENT_TRADES = "10"
  RISK_MULTIPLIER = "0.8"
```

## 💰 Cost Management

### GPU Instance Costs (Estimated)

- **A10**: ~$250/month per instance
- **A100-40GB**: ~$900/month per instance  
- **A100-80GB**: ~$1,200/month per instance

### Cost Optimization Commands

```bash
# Scale down for development
./scripts/scale.sh down

# Scale up for production
./scripts/scale.sh up --instances 2

# Stop all instances
./scripts/scale.sh stop

# Show cost optimization tips
./scripts/scale.sh cost
```

## 🔄 Scaling Operations

### Manual Scaling

```bash
# Scale to 2 instances
flyctl scale count 2 --app ruvtrade

# Change GPU type (requires redeploy)
# Edit fly.toml, then:
flyctl deploy --app ruvtrade
```

### Using Scale Script

```bash
# Scale up to production (2 instances)
./scripts/scale.sh up --instances 2

# Scale down to minimum (1 instance)
./scripts/scale.sh down

# Check current status
./scripts/scale.sh status

# Enable auto-scaling
./scripts/scale.sh auto
```

## 📊 Monitoring & Logs

### Real-time Logs

```bash
flyctl logs --app ruvtrade --follow
```

### Application Status

```bash
flyctl status --app ruvtrade
```

### Health Checks

The application provides several health check endpoints:

- `/health` - General health status
- `/gpu-status` - GPU utilization and memory
- `/metrics` - Prometheus metrics
- `/trading/status` - Trading operations status

## 🔧 Local Development & Testing

### Using Docker Compose

```bash
# Start full stack locally (requires NVIDIA Docker)
docker-compose up --build

# Access application
curl http://localhost:8080/health
```

### Requirements

- NVIDIA Docker runtime
- CUDA-compatible GPU
- 16GB+ GPU memory recommended

## 🚀 API Endpoints

### Core Endpoints

- `GET /` - Platform information
- `GET /health` - Health check
- `GET /gpu-status` - GPU status
- `GET /metrics` - Prometheus metrics

### Trading Operations

- `POST /trading/start` - Start trading
- `POST /trading/stop` - Stop trading  
- `GET /trading/status` - Trading status

### Example Usage

```bash
# Check GPU status
curl https://ruvtrade.fly.dev/gpu-status

# Start trading
curl -X POST https://ruvtrade.fly.dev/trading/start

# Get trading status
curl https://ruvtrade.fly.dev/trading/status
```

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   flyctl auth login
   ```

2. **GPU Not Available**
   - Check if GPU quota is enabled in Fly.io account
   - Verify CUDA_VISIBLE_DEVICES environment variable

3. **Out of Memory**
   - Reduce batch sizes in trading strategies
   - Scale up to larger GPU instance

4. **Build Failures**
   - Check Docker build logs: `flyctl logs --app ruvtrade`
   - Verify all dependencies in requirements.txt

### Debug Commands

```bash
# SSH into running instance
flyctl ssh console --app ruvtrade

# Check machine status
flyctl machine list --app ruvtrade

# View detailed logs
flyctl logs --app ruvtrade --lines 100
```

## 📈 Performance Optimization

### GPU Optimization

1. **RAPIDS Configuration**
   - cuDF for DataFrame operations
   - cuML for machine learning
   - CuPy for array operations

2. **Memory Management**
   - Automatic memory pool limiting
   - Batch processing optimization
   - GPU cache management

3. **Concurrent Processing**
   - Multi-strategy execution
   - Asynchronous data processing
   - Parallel model training

### Production Recommendations

- Monitor GPU utilization to optimize costs
- Use auto-scaling during market hours
- Implement circuit breakers for risk management
- Set up alerting for trading anomalies

## 🛡️ Security Considerations

- All API keys stored as Fly.io secrets
- HTTPS enforced for all connections
- GPU isolation between processes
- Regular security updates in base image

## 📞 Support

For issues with:
- **Fly.io Platform**: Contact Fly.io support
- **GPU Access**: Request GPU quota increase
- **Trading Logic**: Check application logs and health endpoints

---

**Ready to deploy?** Run `./scripts/deploy.sh` and start GPU-accelerated trading! 🚀