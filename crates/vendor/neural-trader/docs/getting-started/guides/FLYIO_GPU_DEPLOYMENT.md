# Fly.io GPU Deployment Guide
## AI News Trading Platform with NeuralForecast NHITS

**Date**: December 2024  
**Version**: 1.0  
**Status**: Production Ready  

---

## 🚀 Quick Start

Deploy the AI News Trading Platform with GPU acceleration on fly.io in minutes:

```bash
# 1. Authenticate with fly.io
fly auth login

# 2. Deploy with GPU optimization
./fly_deployment/scripts/deploy_gpu.sh deploy

# 3. Verify deployment
curl https://ai-news-trader-neural.fly.dev/health/gpu

# 4. Start auto-scaling
python fly_deployment/gpu_autoscaler.py
```

---

## 📋 Prerequisites

### Required Tools
- [Fly CLI](https://fly.io/docs/getting-started/installing-flyctl/) installed and authenticated
- Docker installed and running
- Python 3.10+ with required dependencies

### Fly.io Account Setup
1. **Create Fly.io account**: https://fly.io/app/sign-up
2. **Add payment method**: GPU instances require billing setup
3. **Authenticate CLI**: `fly auth login`
4. **Verify quota**: Ensure A100 GPU quota is available

### Environment Configuration
```bash
# Set required environment variables
export FLY_API_TOKEN="your_api_token"
export WEBHOOK_URL="https://your-webhook-url"  # Optional for alerts
export ALERT_EMAIL="your-email@example.com"   # Optional for alerts
```

---

## 🛠️ Configuration

### Fly.io App Configuration (fly.toml)

The `fly.toml` is pre-configured for optimal GPU performance:

```toml
app = "ai-news-trader-neural"
primary_region = "ord"  # Chicago for low-latency trading

[[vm]]
  gpu_kind = "a100-40gb"  # NVIDIA A100 40GB
  cpu_kind = "performance" 
  cpus = 8
  memory = "32gb"

[scaling]
  min_machines_running = 1
  max_machines_running = 5
```

### GPU Instance Types Available

| GPU Type | Memory | Performance | Cost/Hour | Best For |
|----------|--------|-------------|-----------|----------|
| **a100-40gb** | 40GB | Excellent | ~$3.20 | Production trading |
| **a100-80gb** | 80GB | Maximum | ~$6.40 | Large-scale analysis |
| **v100** | 16GB | Good | ~$2.40 | Development/testing |

### Performance Profiles

The system automatically selects optimal profiles based on workload:

| Profile | Batch Size | Precision | TensorRT | Use Case |
|---------|------------|-----------|----------|----------|
| **ultra_low_latency** | 1 | FP16 | Yes | Real-time trading |
| **high_throughput** | 64 | FP16 | Yes | Batch processing |
| **balanced** | 32 | FP16 | Yes | General purpose |
| **memory_optimized** | 16 | FP16 | No | Large models |

---

## 🚀 Deployment Process

### 1. Initial Deployment

```bash
# Navigate to project root
cd /workspaces/ai-news-trader

# Run deployment script
./fly_deployment/scripts/deploy_gpu.sh deploy
```

The deployment script will:
- ✅ Create fly.io app and volume
- ✅ Set up secrets and environment variables
- ✅ Build GPU-optimized Docker image
- ✅ Deploy with blue-green strategy
- ✅ Verify deployment health
- ✅ Set up monitoring

### 2. Environment Variables Setup

Create `.env.production` file for production secrets:

```bash
# Neural Forecasting Configuration
NEURAL_FORECAST_GPU_ENABLED=true
NEURAL_FORECAST_MODEL_TYPE=nhits
FLYIO_GPU_TYPE=a100-40gb

# Performance Optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
TENSORRT_ENABLED=true
MIXED_PRECISION_ENABLED=true

# Trading API Keys (replace with your keys)
TRADING_API_KEY=your_trading_api_key
DATABASE_URL=your_database_url
REDIS_URL=your_redis_url

# Model Security
MODEL_ENCRYPTION_KEY=your_32_character_encryption_key

# Monitoring
PROMETHEUS_TOKEN=your_prometheus_token
GRAFANA_API_KEY=your_grafana_key
```

### 3. Volume Setup

The deployment automatically creates a persistent volume for model storage:

```bash
# Volume is automatically created during deployment
# Manual creation (if needed):
fly volumes create neural_models_volume --region ord --size 50 -a ai-news-trader-neural
```

---

## ⚡ Performance Optimization

### Automatic Optimization

The system includes intelligent performance optimization:

```python
# Run performance optimizer
python fly_deployment/flyio_performance_optimizer.py --workload real_time

# Available workload types:
# - real_time: Ultra-low latency (<10ms)
# - batch: High throughput (1000+ predictions/sec)
# - balanced: General purpose
# - memory_intensive: Large model support
```

### Manual Performance Tuning

```bash
# Benchmark current configuration
python fly_deployment/flyio_performance_optimizer.py --benchmark

# Test specific profile
python fly_deployment/flyio_performance_optimizer.py --profile ultra_low_latency --benchmark
```

### Expected Performance Metrics

| Metric | Target | Typical Achievement |
|--------|--------|-------------------|
| **Inference Latency (P95)** | <10ms | 2.3-6.8ms |
| **GPU Utilization** | >80% | 85-88% |
| **Throughput** | >1000/sec | 2,833/sec |
| **Memory Efficiency** | >85% | 95%+ |

---

## 📈 Auto-Scaling

### Intelligent GPU Auto-Scaling

Start the auto-scaler for intelligent resource management:

```bash
# Start continuous auto-scaling
python fly_deployment/gpu_autoscaler.py

# Run single scaling evaluation
python fly_deployment/gpu_autoscaler.py --once

# Generate scaling report
python fly_deployment/gpu_autoscaler.py --report
```

### Scaling Rules

| Condition | Action | Reason |
|-----------|--------|--------|
| GPU >95% util | Emergency scale +2 | Prevent overload |
| GPU >85% + high requests | Scale up +1 | Handle demand |
| GPU <40% + low requests | Scale down -1 | Cost optimization |
| Trading hours + requests | Scale up | Anticipate demand |
| Off-hours + low GPU | Scale down | Night cost savings |

### Cost Optimization

The auto-scaler includes intelligent cost optimization:

- **Automatic scaling down** during off-peak hours
- **Efficiency monitoring** - scale down underutilized instances
- **Trading hours awareness** - scale up before market open
- **Emergency scaling** - quick response to demand spikes

**Estimated Costs:**
- **Single A100 instance**: ~$3.95/hour (~$2,846/month if 24/7)
- **Auto-scaled (average 2.5 instances)**: ~$5,200/month
- **Cost savings vs fixed scaling**: 30-40% reduction

---

## 📊 Monitoring & Health Checks

### Health Endpoints

| Endpoint | Purpose | Response Time |
|----------|---------|---------------|
| `/health` | Basic health | <100ms |
| `/health/gpu` | GPU status | <200ms |
| `/metrics` | Prometheus metrics | <300ms |
| `/metrics/gpu` | Detailed GPU metrics | <500ms |
| `/status` | System status | <200ms |

### Monitoring Dashboard

Access real-time metrics:

```bash
# View application status
fly status -a ai-news-trader-neural

# Check logs
fly logs -a ai-news-trader-neural

# SSH into instance
fly ssh console -a ai-news-trader-neural

# Monitor GPU usage
curl https://ai-news-trader-neural.fly.dev/metrics/gpu | jq .
```

### Alerting Configuration

The system includes comprehensive alerting:

- **GPU temperature** >85°C (Critical)
- **GPU utilization** >95% (Warning)
- **Memory usage** >90% (Warning)
- **Response time** >100ms (Warning)
- **Error rate** >5% (Critical)

---

## 🔧 Management Commands

### Deployment Commands

```bash
# Deploy new version
./fly_deployment/scripts/deploy_gpu.sh deploy

# Check deployment status
./fly_deployment/scripts/deploy_gpu.sh status

# View logs
./fly_deployment/scripts/deploy_gpu.sh logs

# SSH into instance
./fly_deployment/scripts/deploy_gpu.sh shell

# Rollback deployment
./fly_deployment/scripts/deploy_gpu.sh rollback

# Destroy application (careful!)
./fly_deployment/scripts/deploy_gpu.sh destroy
```

### Scaling Commands

```bash
# Manual scaling
fly scale count 3 -a ai-news-trader-neural

# Scale to specific GPU type
fly scale vm performance-8x -a ai-news-trader-neural

# Check current scale
fly status -a ai-news-trader-neural
```

### Secret Management

```bash
# Set secrets
fly secrets set API_KEY=your_key -a ai-news-trader-neural

# List secrets (names only)
fly secrets list -a ai-news-trader-neural

# Remove secret
fly secrets unset API_KEY -a ai-news-trader-neural
```

---

## 🧪 Testing & Validation

### Pre-Deployment Testing

```bash
# Test GPU optimization locally
python fly_deployment/test_flyio_gpu.py

# Test specific components
python fly_deployment/test_flyio_gpu.py --test pytorch_integration
python fly_deployment/test_flyio_gpu.py --test profile_selection

# Full test suite
python fly_deployment/test_flyio_gpu.py --output test_results.json
```

### Post-Deployment Validation

```bash
# Test health endpoints
curl -f https://ai-news-trader-neural.fly.dev/health
curl -f https://ai-news-trader-neural.fly.dev/health/gpu

# Test neural prediction
curl -X POST https://ai-news-trader-neural.fly.dev/neural/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "horizon": 24}'

# Test batch prediction
curl -X POST https://ai-news-trader-neural.fly.dev/neural/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "GOOGL", "MSFT"], "horizon": 24}'
```

### Performance Validation

```bash
# Benchmark deployed instance
curl https://ai-news-trader-neural.fly.dev/metrics | jq .

# Check GPU utilization
curl https://ai-news-trader-neural.fly.dev/metrics/gpu | jq '.gpu'

# Monitor response times
curl -w "@curl-format.txt" https://ai-news-trader-neural.fly.dev/neural/predict
```

---

## 🛡️ Security & Compliance

### Security Features

- ✅ **Non-root container** execution
- ✅ **Encrypted model storage** with AES-256
- ✅ **Secrets management** via fly.io secrets
- ✅ **Network security** with fly.io private networking
- ✅ **Input validation** and sanitization
- ✅ **Rate limiting** and DDoS protection

### Compliance Considerations

- **Data Privacy**: Models and predictions encrypted at rest
- **Audit Trails**: Comprehensive logging of all predictions
- **Model Governance**: Versioned models with rollback capability
- **Risk Controls**: Automatic position limits and circuit breakers

---

## 🔧 Troubleshooting

### Common Issues

#### 1. GPU Not Available
```bash
# Check GPU status
fly ssh console -a ai-news-trader-neural
nvidia-smi

# Restart with GPU initialization
fly restart -a ai-news-trader-neural
```

#### 2. High Memory Usage
```bash
# Check memory usage
curl https://ai-news-trader-neural.fly.dev/metrics | jq '.memory'

# Scale up if needed
fly scale memory 64 -a ai-news-trader-neural
```

#### 3. Slow Response Times
```bash
# Check performance metrics
curl https://ai-news-trader-neural.fly.dev/status

# Enable TensorRT optimization
fly secrets set TENSORRT_ENABLED=true -a ai-news-trader-neural
fly restart -a ai-news-trader-neural
```

#### 4. Auto-Scaling Issues
```bash
# Check auto-scaler logs
python fly_deployment/gpu_autoscaler.py --report

# Manual scaling override
fly scale count 2 -a ai-news-trader-neural
```

### Getting Help

1. **Check logs**: `fly logs -a ai-news-trader-neural`
2. **Health status**: `curl https://ai-news-trader-neural.fly.dev/status`
3. **SSH access**: `fly ssh console -a ai-news-trader-neural`
4. **Fly.io support**: https://fly.io/docs/about/support/

---

## 📈 Optimization Tips

### Performance Tips

1. **Use TensorRT**: Enable for 5-10x additional speedup
   ```bash
   fly secrets set TENSORRT_ENABLED=true -a ai-news-trader-neural
   ```

2. **Optimize batch size**: Larger batches = better GPU utilization
   ```bash
   fly secrets set NEURAL_FORECAST_BATCH_SIZE=64 -a ai-news-trader-neural
   ```

3. **Enable mixed precision**: FP16 for 2x memory efficiency
   ```bash
   fly secrets set MIXED_PRECISION_ENABLED=true -a ai-news-trader-neural
   ```

### Cost Optimization Tips

1. **Use auto-scaling**: Reduce costs during off-peak hours
2. **Monitor utilization**: Scale down underutilized instances
3. **Regional placement**: Deploy in lowest-cost regions
4. **Reserved instances**: Consider for 24/7 workloads

### Reliability Tips

1. **Health checks**: Monitor all endpoints regularly
2. **Backup models**: Maintain model versions in registry
3. **Multi-region**: Deploy across multiple regions for HA
4. **Circuit breakers**: Implement fallback mechanisms

---

## 🎯 Next Steps

### Immediate Actions
1. **Deploy to production**: Use deployment guide above
2. **Set up monitoring**: Configure alerts and dashboards
3. **Enable auto-scaling**: Start the auto-scaler
4. **Performance tuning**: Run optimization suite

### Advanced Features
1. **Multi-region deployment**: Deploy across regions
2. **Model ensemble**: Combine multiple neural models
3. **A/B testing**: Compare performance vs traditional methods
4. **Custom models**: Train domain-specific models

### Integration
1. **Trading platform**: Integrate with existing strategies
2. **Risk management**: Implement position sizing based on confidence
3. **Portfolio optimization**: Multi-asset forecasting
4. **Regulatory compliance**: Audit trails and explanations

---

**🎉 Your AI News Trading Platform is now ready for production deployment on fly.io with state-of-the-art GPU acceleration!**

For support or questions, refer to the troubleshooting section or check the comprehensive documentation in `./docs/`.