# Bayesian VaR Production Deployment Guide

## Zero-Downtime Deployment with Constitutional Prime Directive Compliance

This directory contains the complete production deployment configuration for the Bayesian VaR system, implementing zero-downtime blue-green deployment strategy with Constitutional Prime Directive compliance.

## 🚀 Quick Deployment

### Prerequisites

1. **Kubernetes cluster** with sufficient resources
2. **kubectl** configured for target cluster
3. **Required secrets** configured (see `kubernetes/secrets.yaml`)
4. **Monitoring stack** deployed (Prometheus, Grafana, AlertManager)
5. **E2B API tokens** configured
6. **Binance API credentials** configured

### Deploy to Production

```bash
# 1. Execute zero-downtime deployment
./scripts/production_deployment.sh

# 2. Validate deployment
./scripts/deployment_validation.sh

# 3. Start E2B monitoring
./scripts/e2b_production_monitor.sh &
```

## 📁 Directory Structure

```
deploy/
├── kubernetes/                 # Kubernetes manifests
│   ├── bayesian-var-deployment.yaml    # Main blue-green deployment
│   └── secrets.yaml                    # Secret templates
├── observability/              # Monitoring and alerting
│   ├── grafana-dashboard.json          # Production dashboard
│   ├── prometheus-alerts.yaml          # Alert rules
│   └── opentelemetry-collector.yaml    # Observability stack
└── scripts/                    # Deployment automation
    ├── production_deployment.sh        # Main deployment script
    ├── deployment_validation.sh        # Validation suite
    ├── zero_downtime_rollback.sh       # Emergency rollback
    └── e2b_production_monitor.sh       # E2B monitoring
```

## 🔧 Configuration

### Environment Variables

Configure these environment variables before deployment:

```bash
# Required
export E2B_API_TOKEN="your_e2b_api_token"
export BINANCE_API_KEY="your_binance_api_key"
export BINANCE_SECRET_KEY="your_binance_secret_key"

# Optional - for notifications
export SLACK_WEBHOOK_URL="your_slack_webhook"
export PAGERDUTY_API_KEY="your_pagerduty_key"
export PAGERDUTY_SERVICE_ID="your_service_id"
```

### Secrets Management

1. **Create Kubernetes secrets**:
```bash
# Base64 encode your credentials first
echo -n "your_api_key" | base64

# Apply secrets (after updating secrets.yaml)
kubectl apply -f kubernetes/secrets.yaml
```

2. **Verify secrets**:
```bash
kubectl get secrets -n production
```

## 🚨 Zero-Downtime Deployment Process

### Blue-Green Deployment Flow

1. **Green Deployment**: New version deployed to "green" environment
2. **Health Validation**: Comprehensive health checks on green
3. **Traffic Switch**: Gradual traffic routing to green
4. **Blue Scale-Down**: Previous version (blue) scaled to zero
5. **Rollback Ready**: Blue kept ready for emergency rollback

### Constitutional Prime Directive Checks

- ✅ **Real Data Connectivity**: Binance WebSocket active
- ✅ **E2B Sandbox Health**: All training sandboxes operational  
- ✅ **Model Accuracy**: ≥95% accuracy maintained
- ✅ **Error Rate**: <1% calculation error rate
- ✅ **Performance**: <1s P99 response time
- ✅ **Security**: All pods run as non-root with read-only filesystem

## 📊 Monitoring and Observability

### Key Metrics Monitored

| Metric | Threshold | Action |
|--------|-----------|--------|
| System Availability | 99.99% | Constitutional violation if breached |
| VaR Calculation Error Rate | 1% | Critical alert |
| Model Accuracy | 95% | Warning alert, retraining required |
| Binance Connections | >0 | Critical - no real data |
| E2B Sandbox Health | 2/3 healthy | Constitutional compliance check |
| Response Time P99 | 1 second | Performance degradation alert |

### Dashboards

- **Production Dashboard**: `/deploy/observability/grafana-dashboard.json`
- **Constitutional Compliance**: Real-time compliance monitoring
- **E2B Sandbox Status**: Training pipeline health
- **Performance Metrics**: SLA/SLO tracking

### Alerting

- **PagerDuty**: Critical and constitutional violations
- **Slack**: All severity levels with appropriate channels
- **Executive Escalation**: Constitutional Prime Directive violations

## 🔄 Emergency Procedures

### Rollback Process

1. **Automatic Rollback Triggers**:
   - Error rate >1%
   - Constitutional Prime Directive violation
   - Health checks failing >2 minutes

2. **Manual Rollback**:
```bash
./scripts/zero_downtime_rollback.sh
```

3. **Rollback Validation**:
   - Traffic switched to blue (previous version)
   - Green deployment scaled to zero
   - System health verified
   - E2B sandboxes restored

### Emergency Contacts

- **On-Call Engineer**: Via PagerDuty
- **Executive Escalation**: Constitutional violations
- **DevOps Team**: Slack #bayesian-var-alerts

## 🧪 Testing and Validation

### Pre-Deployment Tests

1. **Security Scanning**: Container and dependency scanning
2. **Performance Testing**: Load testing with realistic data
3. **Integration Testing**: E2B and Binance connectivity
4. **Compliance Testing**: Constitutional Prime Directive validation

### Post-Deployment Validation

The deployment validation script performs:

- ✅ Health endpoint verification
- ✅ Real data connectivity testing
- ✅ E2B sandbox integration testing
- ✅ Model accuracy validation
- ✅ Performance SLA compliance
- ✅ Error rate monitoring
- ✅ Security configuration verification
- ✅ Resource usage validation
- ✅ Constitutional compliance verification

## 📋 Deployment Checklist

### Before Deployment

- [ ] Secrets configured and validated
- [ ] Monitoring stack operational
- [ ] Blue deployment healthy and serving traffic
- [ ] E2B sandboxes accessible
- [ ] Binance API connectivity verified
- [ ] Rollback procedure tested

### During Deployment

- [ ] Green deployment successful
- [ ] All health checks pass
- [ ] Performance tests pass
- [ ] Traffic switch successful
- [ ] Blue deployment scaled down
- [ ] Monitoring alerts nominal

### After Deployment

- [ ] System stability monitored for 30 minutes
- [ ] All integrations verified
- [ ] Deployment report generated
- [ ] Team notifications sent
- [ ] Documentation updated

## 🔒 Security Considerations

### Container Security

- **Non-root execution**: All containers run as non-root user (UID 1000)
- **Read-only filesystem**: Root filesystem is read-only
- **No privilege escalation**: allowPrivilegeEscalation: false
- **Dropped capabilities**: All Linux capabilities dropped
- **Seccomp profile**: Runtime security profiles enabled

### Network Security

- **Network policies**: Restrict ingress/egress traffic
- **TLS encryption**: All communication encrypted
- **Secret management**: Kubernetes secrets with proper RBAC
- **API authentication**: Bearer token authentication

### Constitutional Compliance

- **Audit logging**: All actions logged with correlation IDs
- **Access controls**: RBAC with least privilege principle  
- **Data protection**: Sensitive data encrypted at rest
- **Compliance monitoring**: Real-time constitutional violation detection

## 🎯 Performance Targets

### Service Level Objectives (SLOs)

| Service | Availability | Response Time | Error Rate |
|---------|-------------|---------------|------------|
| VaR Calculations | 99.99% | P99 < 1s | <1% |
| Real Data Feed | 99.95% | P95 < 100ms | <0.1% |
| E2B Training | 99.9% | P95 < 30s | <5% |
| Health Checks | 99.99% | P95 < 500ms | <0.01% |

### Resource Allocation

- **CPU**: 500m request, 1000m limit
- **Memory**: 1Gi request, 2Gi limit  
- **Storage**: 2Gi ephemeral, 4Gi limit
- **Replicas**: 5 minimum, 20 maximum (HPA)

## 📞 Support and Troubleshooting

### Common Issues

1. **Deployment Stuck**: Check resource availability and image pull
2. **Health Check Failures**: Verify dependencies (Binance, E2B, DB)
3. **High Error Rate**: Check model accuracy and data connectivity
4. **Performance Degradation**: Review resource usage and scaling

### Log Locations

- **Application logs**: `kubectl logs -n production -l app=bayesian-var`
- **Deployment logs**: `/var/log/bayesian-var-deployment.log`
- **E2B monitoring**: `/var/log/e2b-production-monitor.log`
- **Rollback logs**: `/var/log/bayesian-var-rollback.log`

### Metrics and Reports

- **Deployment reports**: `/var/reports/bayesian-var-deployment-*.json`
- **Validation reports**: `/var/reports/deployment-validation-*.json`
- **E2B monitoring**: `/var/reports/e2b-monitoring-*.json`
- **Rollback reports**: `/var/reports/bayesian-var-rollback-*.json`

## 🚀 Next Steps

After successful deployment:

1. **Monitor system stability** for initial 30 minutes
2. **Verify all metrics** in Grafana dashboards  
3. **Test rollback capability** in staging environment
4. **Update runbooks** with any deployment-specific notes
5. **Schedule next deployment window** for future releases

---

## Constitutional Prime Directive Compliance

This deployment implements full Constitutional Prime Directive compliance:

- ✅ **Zero-downtime requirement** met through blue-green deployment
- ✅ **Real data requirement** enforced with Binance WebSocket monitoring
- ✅ **E2B sandbox integration** validated and monitored continuously
- ✅ **Model accuracy enforcement** with automatic retraining triggers
- ✅ **Performance SLA compliance** monitored and alerted
- ✅ **Security hardening** implemented at all levels
- ✅ **Emergency procedures** tested and documented
- ✅ **Comprehensive monitoring** with executive escalation

**Ready for Production Traffic** ✅