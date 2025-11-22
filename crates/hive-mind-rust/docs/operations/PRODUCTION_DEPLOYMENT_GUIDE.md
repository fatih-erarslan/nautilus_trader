# Hive Mind Rust - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Hive Mind Rust financial trading system to production with 99.99% uptime requirements.

## 🚨 Critical Prerequisites

### System Requirements
- **Kubernetes Cluster**: v1.28+ with high availability
- **Node Specifications**: Minimum 32 CPU cores, 128GB RAM per node
- **Storage**: NVMe SSD with 3000+ IOPS for sub-millisecond latency
- **Network**: Dedicated low-latency network with <1ms between nodes
- **Security**: Isolated network with financial-grade compliance

### Infrastructure Dependencies
- **Container Registry**: Private registry with security scanning
- **Secret Management**: HashiCorp Vault or AWS Secrets Manager
- **Monitoring**: Prometheus + Grafana + AlertManager
- **Logging**: ELK stack with log aggregation
- **Backup**: Cross-region backup storage (S3/equivalent)

## Pre-Deployment Checklist

### 1. Security Validation
```bash
# Verify security scanning is enabled
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image hive-mind-rust:latest

# Check for vulnerabilities
kubectl security-benchmark run --config financial-grade
```

### 2. Performance Validation
```bash
# Run performance benchmarks
./scripts/production/performance-test.sh

# Validate latency requirements
./scripts/production/latency-test.sh --target-latency 100us
```

### 3. Configuration Validation
```bash
# Validate production configuration
hive-mind-server validate --config config/production.toml

# Check resource limits
kubectl apply --dry-run=client -f deployment/kubernetes/
```

## Deployment Process

### Phase 1: Infrastructure Preparation

#### 1.1 Create Namespace and Resources
```bash
# Apply namespace and resource quotas
kubectl apply -f deployment/kubernetes/namespace.yaml

# Verify namespace creation
kubectl get namespace hive-mind-production
```

#### 1.2 Set Up Persistent Storage
```bash
# Create storage classes for high-performance storage
kubectl apply -f deployment/kubernetes/pvc.yaml

# Verify storage availability
kubectl get pvc -n hive-mind-production
```

#### 1.3 Configure Secrets Management
```bash
# Set up external secrets (preferred)
kubectl apply -f deployment/kubernetes/secrets.yaml

# Verify secret creation (without exposing values)
kubectl get secrets -n hive-mind-production
```

### Phase 2: Application Deployment

#### 2.1 Build and Push Container
```bash
# Build production image
docker build \
  --target runtime \
  --build-arg RUST_VERSION=1.75 \
  --tag hive-mind-rust:$(git rev-parse --short HEAD) \
  .

# Security scan
trivy image --exit-code 1 --severity HIGH,CRITICAL hive-mind-rust:latest

# Push to registry
docker push your-registry/hive-mind-rust:$(git rev-parse --short HEAD)
```

#### 2.2 Deploy Application
```bash
# Apply RBAC configuration
kubectl apply -f deployment/kubernetes/rbac.yaml

# Apply network policies
kubectl apply -f deployment/kubernetes/network-policy.yaml

# Apply ConfigMaps
kubectl apply -f deployment/kubernetes/configmap.yaml

# Deploy application
kubectl apply -f deployment/kubernetes/deployment.yaml

# Verify deployment
kubectl rollout status deployment/hive-mind-rust -n hive-mind-production
```

#### 2.3 Configure Auto-Scaling
```bash
# Apply HPA and VPA
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify auto-scaling configuration
kubectl get hpa,vpa -n hive-mind-production
```

### Phase 3: Monitoring and Alerting

#### 3.1 Deploy Monitoring Stack
```bash
# Deploy Prometheus rules
kubectl apply -f monitoring/alerting/prometheus-rules.yaml

# Import Grafana dashboards
kubectl apply -f monitoring/grafana/dashboards/
```

#### 3.2 Configure Alerts
```bash
# Test critical alerts
curl -X POST http://alertmanager:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","severity":"critical"}}]'
```

### Phase 4: Production Validation

#### 4.1 Health Checks
```bash
# Verify all pods are ready
kubectl get pods -n hive-mind-production -l app=hive-mind-rust

# Check health endpoints
kubectl port-forward svc/hive-mind-service 8091:8091 &
curl -f http://localhost:8091/health/live
curl -f http://localhost:8091/health/ready
```

#### 4.2 Performance Testing
```bash
# Run comprehensive performance tests
./scripts/production/performance-test.sh --duration 300 --concurrency 1000

# Validate latency SLAs
./scripts/production/latency-validation.sh --p99-threshold 1ms
```

#### 4.3 Disaster Recovery Testing
```bash
# Test backup creation
./scripts/production/disaster-recovery.sh backup

# Test restore procedure (in staging)
./scripts/production/disaster-recovery.sh restore backup-20250821-143000
```

## Blue-Green Deployment

### Automated Deployment
```bash
# Use the production deployment script
./scripts/production/deploy.sh blue

# After validation, switch traffic
./scripts/production/deploy.sh switch-traffic blue
```

### Manual Blue-Green Process

#### Deploy to Blue Environment
```bash
# Create blue deployment
kubectl apply -f deployment/kubernetes/deployment-blue.yaml

# Wait for blue deployment to be ready
kubectl rollout status deployment/hive-mind-rust-blue -n hive-mind-production
```

#### Validate Blue Environment
```bash
# Run validation tests against blue environment
./scripts/production/validate-deployment.sh blue

# Check metrics and health
curl -f http://blue-service:8091/health
curl -f http://blue-service:9090/metrics
```

#### Switch Traffic
```bash
# Update service to point to blue deployment
kubectl patch service hive-mind-service -n hive-mind-production \
  -p '{"spec":{"selector":{"deployment":"blue"}}}'
```

## Monitoring and Observability

### Key Metrics to Monitor

#### Trading Performance
- **Latency**: P50, P95, P99, P99.9 trading operation latency
- **Throughput**: Operations per second, trades per second
- **Error Rate**: Failed trades, consensus failures
- **Business Metrics**: P&L, position risk, market exposure

#### System Health
- **Resource Usage**: CPU, memory, disk I/O, network
- **Consensus Health**: Node connectivity, leader elections
- **Database Performance**: Query latency, connection pool usage

#### Security Metrics
- **Authentication**: Failed login attempts, token expires
- **Network**: Suspicious connections, rate limit violations
- **Audit**: Compliance events, unauthorized access attempts

### Alert Thresholds

#### Critical Alerts (Immediate Response)
- Trading latency P99 > 1ms
- System availability < 99.99%
- Consensus failures > 5 in 5 minutes
- Security breach indicators
- Data inconsistency detected

#### Warning Alerts (Monitor Closely)
- Trading latency P95 > 500μs
- CPU usage > 80% for 5 minutes
- Memory usage > 85%
- Error rate > 1%
- Disk space < 15%

## Security Considerations

### Runtime Security
```bash
# Enable Pod Security Standards
kubectl label namespace hive-mind-production \
  pod-security.kubernetes.io/enforce=restricted

# Scan running containers
kubectl security-scan runtime -n hive-mind-production
```

### Network Security
```bash
# Verify network policies are active
kubectl get networkpolicy -n hive-mind-production

# Test network isolation
kubectl run test-pod --rm -i --tty --image=alpine -- sh
# Try to connect to restricted services (should fail)
```

### Compliance Auditing
```bash
# Generate compliance report
kubectl compliance-check --standard financial-grade \
  --namespace hive-mind-production

# Audit trail verification
kubectl audit-trail --from "24h ago" --namespace hive-mind-production
```

## Backup and Disaster Recovery

### Automated Backups
```bash
# Schedule regular backups
kubectl create cronjob hive-mind-backup \
  --schedule="0 */6 * * *" \
  --image=hive-mind-rust:latest \
  -- /scripts/production/disaster-recovery.sh backup
```

### Disaster Recovery Procedures

#### RTO/RPO Targets
- **Recovery Time Objective (RTO)**: < 1 minute
- **Recovery Point Objective (RPO)**: < 10 seconds
- **Data Loss Tolerance**: Zero for trading data

#### Failover Process
```bash
# Initiate emergency failover
./scripts/production/disaster-recovery.sh failover us-west-2 dr-cluster

# Verify DR system operational
./scripts/production/disaster-recovery.sh verify
```

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod events
kubectl describe pod -n hive-mind-production -l app=hive-mind-rust

# Check container logs
kubectl logs -n hive-mind-production -l app=hive-mind-rust --previous
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n hive-mind-production

# Analyze application metrics
curl http://hive-mind-service:9090/metrics | grep trading_latency
```

#### Network Connectivity
```bash
# Test inter-pod communication
kubectl exec -it pod-name -n hive-mind-production -- ping other-pod-ip

# Verify DNS resolution
kubectl exec -it pod-name -n hive-mind-production -- nslookup hive-mind-service
```

### Emergency Procedures

#### Emergency Rollback
```bash
# Immediate rollback to previous version
kubectl rollout undo deployment/hive-mind-rust -n hive-mind-production

# Verify rollback success
kubectl rollout status deployment/hive-mind-rust -n hive-mind-production
```

#### Emergency Scale Down
```bash
# Scale down to minimal viable replicas
kubectl scale deployment hive-mind-rust --replicas=1 -n hive-mind-production

# Monitor system stability
kubectl get pods -n hive-mind-production -w
```

## Maintenance Windows

### Planned Maintenance
1. **Schedule**: Off-market hours when possible
2. **Communication**: 48-hour advance notice to stakeholders
3. **Testing**: Full validation in staging environment
4. **Rollback Plan**: Always ready for immediate rollback

### Maintenance Checklist
- [ ] Backup current system state
- [ ] Verify rollback procedures
- [ ] Update monitoring thresholds
- [ ] Coordinate with trading desk
- [ ] Test all critical functionality
- [ ] Monitor for 2 hours post-deployment

## Compliance and Auditing

### Regulatory Requirements
- **MiFID II**: Transaction reporting and audit trails
- **SOX**: Change management and access controls  
- **GDPR**: Data protection and privacy controls
- **PCI DSS**: Payment card data security (if applicable)

### Audit Trail Requirements
```bash
# Enable audit logging
kubectl create configmap audit-policy \
  --from-file=audit-policy.yaml \
  -n hive-mind-production

# Verify audit logs
kubectl logs kube-apiserver-master | grep audit
```

## Contact Information

### Escalation Contacts
- **Primary Oncall**: +1-XXX-XXX-XXXX
- **Secondary Oncall**: +1-XXX-XXX-XXXX  
- **Engineering Manager**: engineering-manager@ximera.trading
- **CTO**: cto@ximera.trading

### Emergency Procedures
1. **System Down**: Page primary oncall immediately
2. **Security Incident**: Contact security team + CTO
3. **Regulatory Issue**: Contact compliance officer
4. **Data Loss**: Initiate disaster recovery protocol

---

**Document Version**: 1.0  
**Last Updated**: August 21, 2025  
**Review Schedule**: Monthly  
**Approval**: Production Engineering Team