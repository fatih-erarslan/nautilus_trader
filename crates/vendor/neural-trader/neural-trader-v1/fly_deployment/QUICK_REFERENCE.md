# Quick Reference - GPU Trading Platform Deployment

## Essential Commands

### Deployment
```bash
# Standard deployment
./scripts/deploy.sh

# Blue-green deployment  
./scripts/blue_green_deploy.sh deploy

# Check deployment status
./scripts/blue_green_deploy.sh status

# Rollback deployment
./scripts/blue_green_deploy.sh rollback "reason"
```

### Health & Monitoring
```bash
# Check health
curl https://ruvtrade.fly.dev/health
curl https://ruvtrade.fly.dev/health/detailed
curl https://ruvtrade.fly.dev/gpu-status

# Run health checks locally
python health_checks.py --check all --format pretty

# View metrics
curl https://ruvtrade.fly.dev/metrics
```

### Scaling
```bash
# Auto-scale
python scaling/auto_scaler.py run --app ruvtrade

# Manual scale up/down
./scripts/scale.sh up --instances 3
./scripts/scale.sh down
./scripts/scale.sh status

# Cost analysis
./scripts/scale.sh cost
```

### Backup & Recovery
```bash
# Create backups
python backup/backup_manager.py dr-backup --app ruvtrade

# List backups
python backup/backup_manager.py list

# Cleanup old backups
python backup/backup_manager.py cleanup

# Restore from backup
python backup/backup_manager.py restore --backup-id "backup_id"
```

### Troubleshooting
```bash
# View logs
flyctl logs --app ruvtrade --lines 100

# SSH into machine
flyctl ssh console --app ruvtrade

# Check machine status
flyctl machine list --app ruvtrade

# Check volumes
flyctl volumes list --app ruvtrade
```

## Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic health | `{"status": "healthy"}` |
| `/health/detailed` | Comprehensive health | JSON with all components |
| `/gpu-status` | GPU-specific health | GPU metrics and status |
| `/metrics` | Prometheus metrics | Metrics in Prometheus format |

## Instance Types & Costs

| Type | GPU | Memory | Cost/Hour | Use Case |
|------|-----|--------|-----------|----------|
| A10 | 24GB | 16GB | $0.50 | Development/Testing |
| A100-40GB | 40GB | 32GB | $2.40 | Production Trading |
| A100-80GB | 80GB | 64GB | $3.20 | Heavy ML Workloads |

## Scaling Thresholds

| Metric | Scale Up | Scale Down | Emergency |
|--------|----------|------------|-----------|
| GPU Utilization | >80% | <30% | >95% |
| GPU Temperature | >80°C | <70°C | >90°C |
| Response Time | >5s | <2s | >30s |
| Error Rate | >10% | <1% | >50% |

## Emergency Procedures

### Service Down
1. Check: `curl -I https://ruvtrade.fly.dev/health`
2. Logs: `flyctl logs --app ruvtrade`
3. Rollback: `./scripts/blue_green_deploy.sh rollback "service down"`

### High GPU Temperature
1. Check: `curl https://ruvtrade.fly.dev/gpu-status`
2. Scale: `./scripts/scale.sh up --instances 2`
3. Monitor: `python health_checks.py --check gpu`

### Performance Issues
1. Scale: `./scripts/scale.sh up --instances 3`
2. Check: `python scaling/auto_scaler.py analyze --app ruvtrade`
3. Monitor: `watch -n 5 'curl -s https://ruvtrade.fly.dev/health/detailed'`

## File Structure

```
fly_deployment/
├── fly.toml                    # Main fly.io configuration
├── Dockerfile                 # GPU-enabled container
├── health_checks.py           # Health monitoring system
├── scripts/
│   ├── deploy.sh              # Standard deployment
│   ├── blue_green_deploy.sh   # Blue-green deployment
│   └── scale.sh               # Scaling operations
├── monitoring/
│   ├── logging_config.py      # Logging setup
│   └── dashboard_config.py    # Monitoring dashboards
├── scaling/
│   └── auto_scaler.py         # Auto-scaling engine
├── backup/
│   └── backup_manager.py      # Backup/recovery system
├── ci_cd/
│   └── github_actions.yml     # CI/CD pipeline
└── DEPLOYMENT_GUIDE.md        # Complete documentation
```

## Environment Variables

### Required
```bash
FLY_API_TOKEN=your_token
API_KEY=your_trading_api_key
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Scaling
```bash
MIN_INSTANCES=1
MAX_INSTANCES=5
SCALE_UP_THRESHOLD=80
SCALE_DOWN_THRESHOLD=30
```

### Backup
```bash
BACKUP_S3_BUCKET=your-bucket
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

## Common Issues & Solutions

### GPU Not Detected
- Check CUDA installation in container
- Verify GPU instance type in fly.toml
- Restart application: `flyctl machine restart`

### High Memory Usage
- Scale up: `./scripts/scale.sh up`
- Check for memory leaks in logs
- Consider larger instance type

### Slow Response Times
- Enable auto-scaling
- Check database performance
- Scale horizontally

### Failed Health Checks
- Check external service connectivity
- Verify secrets configuration
- Review application logs

## Monitoring Dashboards

### System Overview
- CPU, Memory, Disk usage
- Network metrics
- Instance health

### GPU Monitoring  
- GPU utilization
- Memory usage
- Temperature monitoring
- Power consumption

### Trading Performance
- Trade volume and success rate
- Portfolio value tracking
- Response time metrics
- Error rate monitoring

## Cost Optimization Tips

1. **Use Auto-Stop**: Scale down during off-peak hours
2. **Right-Size Instances**: Match GPU type to workload
3. **Monitor Utilization**: Keep GPU usage >60%
4. **Optimize Images**: Reduce container size
5. **Use Spot Instances**: For non-critical workloads

## Security Checklist

- [ ] All secrets stored in fly.io secrets
- [ ] HTTPS enforced for all endpoints
- [ ] Regular security updates applied
- [ ] Backup encryption enabled
- [ ] Access logs monitored
- [ ] API keys rotated regularly

## Support Resources

- **Fly.io Docs**: https://fly.io/docs/
- **Community**: https://community.fly.io/
- **Status Page**: https://status.fly.io/
- **GPU Docs**: https://fly.io/docs/gpus/

---

*Keep this reference handy for quick operations and troubleshooting.*