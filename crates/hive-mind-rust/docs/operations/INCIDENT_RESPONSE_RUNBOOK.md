# Hive Mind Rust - Incident Response Runbook

## 🚨 Emergency Contacts

### Primary Response Team
- **On-Call Engineer**: +1-XXX-XXX-XXXX (Escalate immediately for P1 incidents)
- **Lead SRE**: +1-XXX-XXX-XXXX
- **Engineering Manager**: +1-XXX-XXX-XXXX
- **CTO**: +1-XXX-XXX-XXXX (P1 escalation after 15 minutes)

### Business Contacts
- **Trading Desk**: +1-XXX-XXX-XXXX (Notify for any trading impact)
- **Compliance Officer**: +1-XXX-XXX-XXXX (Regulatory issues)
- **Communications**: +1-XXX-XXX-XXXX (External communications)

## Incident Severity Levels

### P1 - Critical (Response: Immediate)
- **Trading system completely down**
- **Data corruption or loss**
- **Security breach**
- **Regulatory compliance violation**
- **Financial loss > $10,000/hour**

### P2 - High (Response: 15 minutes)
- **Significant performance degradation**
- **Partial system outage**
- **Failed deployments**
- **Monitoring/alerting failures**

### P3 - Medium (Response: 2 hours)
- **Minor performance issues**
- **Non-critical feature failures**
- **Delayed batch processes**

### P4 - Low (Response: Next business day)
- **Cosmetic issues**
- **Documentation updates**
- **Enhancement requests**

## Common Incident Scenarios

### 1. System Completely Down

#### Symptoms
- All pods in CrashLoopBackOff state
- Health check endpoints returning 503/404
- No trading operations processing
- Grafana dashboards showing 0 metrics

#### Immediate Actions (< 2 minutes)
```bash
# Check overall system status
kubectl get pods -n hive-mind-production -l app=hive-mind-rust

# Check recent deployments
kubectl rollout history deployment/hive-mind-rust -n hive-mind-production

# Check node health
kubectl get nodes
kubectl describe node <problematic-node>

# Check persistent volumes
kubectl get pv,pvc -n hive-mind-production
```

#### Diagnosis Steps
```bash
# Check pod logs
kubectl logs -n hive-mind-production -l app=hive-mind-rust --tail=100

# Check events
kubectl get events -n hive-mind-production --sort-by=.metadata.creationTimestamp

# Check resource usage
kubectl top nodes
kubectl top pods -n hive-mind-production

# Check cluster DNS
kubectl run test-pod --rm -i --image=busybox --restart=Never -- nslookup hive-mind-service.hive-mind-production.svc.cluster.local
```

#### Resolution Actions
```bash
# Option 1: Emergency rollback
kubectl rollout undo deployment/hive-mind-rust -n hive-mind-production
kubectl rollout status deployment/hive-mind-rust -n hive-mind-production --timeout=300s

# Option 2: Restart deployment
kubectl rollout restart deployment/hive-mind-rust -n hive-mind-production

# Option 3: Scale up resources
kubectl scale deployment hive-mind-rust --replicas=5 -n hive-mind-production

# Option 4: Disaster recovery
./scripts/production/disaster-recovery.sh restore <latest-backup-id>
```

### 2. High Latency (>1ms P99)

#### Symptoms
- Trading latency alerts firing
- P99 response time > 1000μs
- Timeout errors in logs
- Degraded trading performance

#### Diagnosis
```bash
# Check current latency metrics
kubectl exec -it <pod-name> -n hive-mind-production -- curl http://localhost:9090/metrics | grep trading_operation_duration

# Check system resources
kubectl top pods -n hive-mind-production
kubectl describe hpa hive-mind-hpa -n hive-mind-production

# Check network latency
kubectl run network-test --rm -i --image=nicolaka/netshoot --restart=Never -- \
  ping -c 10 hive-mind-service.hive-mind-production.svc.cluster.local

# Check database performance
kubectl logs -n hive-mind-production -l app=hive-mind-rust | grep -i "database\|sql\|query"
```

#### Resolution Actions
```bash
# Scale up replicas
kubectl scale deployment hive-mind-rust --replicas=6 -n hive-mind-production

# Increase resource limits
kubectl patch deployment hive-mind-rust -n hive-mind-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"hive-mind-rust","resources":{"limits":{"cpu":"6","memory":"12Gi"}}}]}}}}'

# Check and optimize database connections
kubectl exec -it <pod-name> -n hive-mind-production -- \
  /app/bin/hive-mind-server admin database optimize-connections

# Enable performance optimizations
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...enable_simd = true\nenable_zero_copy = true..."}}'
kubectl rollout restart deployment/hive-mind-rust -n hive-mind-production
```

### 3. Memory Leak

#### Symptoms
- Memory usage continuously increasing
- Pods being OOMKilled
- System becoming unresponsive
- GC pressure alerts

#### Diagnosis
```bash
# Check memory usage trends
kubectl top pods -n hive-mind-production --sort-by=memory

# Check pod restarts
kubectl get pods -n hive-mind-production -o wide

# Check memory limits and requests
kubectl describe deployment hive-mind-rust -n hive-mind-production

# Get memory profiling data
kubectl exec -it <pod-name> -n hive-mind-production -- \
  curl http://localhost:8091/admin/memory-profile > memory-profile.json
```

#### Resolution Actions
```bash
# Immediate: Restart affected pods
kubectl delete pod -n hive-mind-production -l app=hive-mind-rust

# Increase memory limits temporarily
kubectl patch deployment hive-mind-rust -n hive-mind-production -p '{"spec":{"template":{"spec":{"containers":[{"name":"hive-mind-rust","resources":{"limits":{"memory":"16Gi"}}}]}}}}'

# Enable memory monitoring
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...enable_memory_profiling = true..."}}'

# Schedule emergency memory optimization deployment
# (This requires code changes to fix the leak)
```

### 4. Consensus Failures

#### Symptoms
- Consensus failure alerts
- Inconsistent data between nodes
- Leader election failures
- Distributed operations failing

#### Diagnosis
```bash
# Check consensus health across all nodes
for pod in $(kubectl get pods -n hive-mind-production -l app=hive-mind-rust -o name); do
    echo "=== $pod ==="
    kubectl exec -n hive-mind-production $pod -- curl -s http://localhost:8091/health/consensus
done

# Check network connectivity between pods
kubectl run network-debug --rm -i --image=nicolaka/netshoot --restart=Never -- \
  nmap -p 8080 hive-mind-service.hive-mind-production.svc.cluster.local

# Check logs for consensus issues
kubectl logs -n hive-mind-production -l app=hive-mind-rust | grep -i "consensus\|raft\|election\|leader"

# Check cluster partitioning
kubectl get pods -n hive-mind-production -l app=hive-mind-rust -o wide
```

#### Resolution Actions
```bash
# Force leader re-election
kubectl exec -it <leader-pod> -n hive-mind-production -- \
  /app/bin/hive-mind-server admin consensus force-election

# Restart consensus subsystem
kubectl exec -it <pod-name> -n hive-mind-production -- \
  /app/bin/hive-mind-server admin consensus restart

# Scale down and back up to reset cluster state
kubectl scale deployment hive-mind-rust --replicas=1 -n hive-mind-production
sleep 30
kubectl scale deployment hive-mind-rust --replicas=3 -n hive-mind-production

# Emergency: Restore from backup if data is corrupted
./scripts/production/disaster-recovery.sh restore <trusted-backup-id>
```

### 5. Security Incident

#### Symptoms
- Security alerts firing
- Suspicious network activity
- Unauthorized access attempts
- Data access violations

#### Immediate Actions (< 1 minute)
```bash
# Isolate the system
kubectl patch networkpolicy deny-all-default -n hive-mind-production -p '{"spec":{"podSelector":{},"policyTypes":["Ingress","Egress"]}}'

# Check for compromised pods
kubectl get pods -n hive-mind-production -o wide
kubectl describe pods -n hive-mind-production -l app=hive-mind-rust

# Preserve evidence
kubectl logs -n hive-mind-production -l app=hive-mind-rust --since=1h > security-incident-logs.txt
kubectl get events -n hive-mind-production --since=1h > security-incident-events.txt
```

#### Investigation
```bash
# Check authentication logs
kubectl logs -n hive-mind-production -l app=hive-mind-rust | grep -i "auth\|login\|token\|unauthorized"

# Check network connections
kubectl exec -it <pod-name> -n hive-mind-production -- netstat -tulpn

# Check file system integrity
kubectl exec -it <pod-name> -n hive-mind-production -- find /app -type f -exec sha256sum {} \;

# Check running processes
kubectl exec -it <pod-name> -n hive-mind-production -- ps aux
```

#### Response Actions
```bash
# Force password/token rotation
kubectl delete secret hive-mind-api-keys -n hive-mind-production
# (External secret management will regenerate)

# Emergency deployment with security patches
# (Requires pre-built secure image)

# Enable audit logging
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...enable_audit_logging = true\naudit_level = \"verbose\"..."}}'

# Notify security team and compliance
curl -X POST "$SECURITY_WEBHOOK_URL" -d '{"incident":"security_breach","severity":"P1","system":"hive-mind-rust"}'
```

### 6. Database Connection Issues

#### Symptoms
- Database connection errors
- Query timeouts
- Connection pool exhaustion
- Data persistence failures

#### Diagnosis
```bash
# Check database connectivity
kubectl run db-test --rm -i --image=postgres:13 --restart=Never -- \
  psql "$DATABASE_URL" -c "SELECT 1"

# Check connection pool status
kubectl exec -it <pod-name> -n hive-mind-production -- \
  curl http://localhost:8091/admin/database/pool-status

# Check database logs
kubectl logs -n database -l app=postgresql

# Check network policies affecting database
kubectl get networkpolicy -n hive-mind-production
kubectl get networkpolicy -n database
```

#### Resolution Actions
```bash
# Increase connection pool size
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...max_connections = 100\nconnection_timeout = \"30s\"..."}}'
kubectl rollout restart deployment/hive-mind-rust -n hive-mind-production

# Reset connection pools
kubectl exec -it <pod-name> -n hive-mind-production -- \
  /app/bin/hive-mind-server admin database reset-pools

# Switch to read replica if available
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...use_read_replica = true..."}}'

# Emergency: Enable degraded mode without database
kubectl patch configmap hive-mind-config -n hive-mind-production -p '{"data":{"production.toml":"...enable_memory_only_mode = true..."}}'
```

## Post-Incident Procedures

### Immediate Post-Resolution (< 30 minutes)
```bash
# Verify system health
./scripts/production/validate-deployment.sh
./scripts/production/performance-test.sh --duration 60

# Create incident backup
./scripts/production/disaster-recovery.sh backup

# Update status page
curl -X POST "$STATUS_PAGE_API" -d '{"status":"operational","message":"System restored"}'

# Notify stakeholders
curl -X POST "$NOTIFICATION_WEBHOOK" -d '{"message":"Hive Mind incident resolved"}'
```

### Documentation (< 2 hours)
1. **Incident Timeline**
   - Incident start time
   - Detection time
   - Response time
   - Resolution time
   - Impact duration

2. **Root Cause Analysis**
   - What happened
   - Why it happened
   - Contributing factors
   - Detection gaps

3. **Impact Assessment**
   - Systems affected
   - Users impacted
   - Financial impact
   - Regulatory implications

4. **Response Evaluation**
   - What worked well
   - What could be improved
   - Communication effectiveness
   - Tool availability

### Follow-up Actions (< 1 week)
1. **Implement Preventive Measures**
   - Code fixes
   - Configuration changes
   - Monitoring improvements
   - Alert tuning

2. **Update Runbooks**
   - Add new scenarios
   - Improve procedures
   - Update contact information
   - Test procedures

3. **Post-Mortem Meeting**
   - Review incident response
   - Discuss improvements
   - Plan preventive actions
   - Update processes

## Tools and Resources

### Monitoring Dashboards
- **Grafana**: https://grafana.ximera.trading/d/hive-mind
- **Prometheus**: https://prometheus.ximera.trading/graph
- **AlertManager**: https://alerts.ximera.trading

### Command References
```bash
# Quick system status
kubectl get pods,svc,ing -n hive-mind-production

# Resource usage
kubectl top nodes && kubectl top pods -n hive-mind-production

# Recent events
kubectl get events -n hive-mind-production --sort-by=.metadata.creationTimestamp | tail -20

# Application logs
kubectl logs -n hive-mind-production -l app=hive-mind-rust --tail=50 -f

# Health checks
curl -f http://hive-mind-service:8091/health
```

### Emergency Scripts
- `./scripts/production/emergency-rollback.sh`
- `./scripts/production/disaster-recovery.sh`
- `./scripts/production/scale-emergency.sh`
- `./scripts/production/network-isolate.sh`

### External Resources
- **AWS Console**: https://console.aws.amazon.com
- **Kubernetes Dashboard**: https://k8s-dashboard.ximera.trading
- **Log Aggregation**: https://logs.ximera.trading
- **Status Page**: https://status.ximera.trading

---

**Document Version**: 1.0  
**Last Updated**: August 21, 2025  
**Review Schedule**: Monthly  
**Emergency Update Procedure**: Page on-call engineer immediately for critical changes