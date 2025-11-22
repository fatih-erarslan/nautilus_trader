# Deployment Guide

Complete guide for deploying NHITS models in production environments, from development to enterprise-scale deployments.

## Table of Contents

- [Deployment Architecture](#deployment-architecture)
- [Environment Setup](#environment-setup)
- [Container Deployment](#container-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Performance Optimization](#performance-optimization)
- [Monitoring & Observability](#monitoring--observability)
- [Security Considerations](#security-considerations)
- [Scaling Strategies](#scaling-strategies)
- [Backup & Recovery](#backup--recovery)
- [CI/CD Pipeline](#cicd-pipeline)

## Deployment Architecture

### Single-Node Deployment

For small to medium workloads, a single-node deployment provides simplicity and cost-effectiveness.

```
┌─────────────────────────────────────┐
│              Load Balancer          │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│           Application Server        │
│  ┌─────────────────────────────────┐│
│  │         NHITS Service           ││
│  │  ┌─────────────────────────────┐││
│  │  │    Forecasting Pipeline     │││
│  │  │  ┌─────┐ ┌─────┐ ┌─────────┐│││
│  │  │  │Model│ │Cache│ │Database ││││
│  │  │  └─────┘ └─────┘ └─────────┘│││
│  │  └─────────────────────────────┘││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
```

### Multi-Node Deployment

For high-availability and increased throughput, distribute components across multiple nodes.

```
┌─────────────────────────────────────┐
│              Load Balancer          │
└─────────────┬───────────┬───────────┘
              │           │
    ┌─────────▼─┐     ┌───▼─────────┐
    │   API     │     │   API       │
    │  Server   │     │  Server     │
    └─────┬─────┘     └─────┬───────┘
          │                 │
    ┌─────▼─────────────────▼─────┐
    │      Message Queue          │
    └─────┬───────────────────────┘
          │
    ┌─────▼─────────────────┐
    │   Inference Cluster   │
    │  ┌─────┐ ┌─────┐     │
    │  │Node1│ │Node2│ ... │
    │  └─────┘ └─────┘     │
    └───────────────────────┘
```

### Microservices Architecture

For enterprise deployments, break down into specialized microservices.

```
                ┌─────────────┐
                │   Gateway   │
                └──────┬──────┘
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │  Auth   │   │Forecast │   │Monitor  │
    │Service  │   │Service  │   │Service  │
    └─────────┘   └────┬────┘   └─────────┘
                       │
              ┌────────▼────────┐
              │  Model Service  │
              │ ┌─────────────┐ │
              │ │   NHITS     │ │
              │ │ Instances   │ │
              │ └─────────────┘ │
              └─────────────────┘
```

## Environment Setup

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.4 GHz
- RAM: 8 GB
- Storage: 50 GB SSD
- Network: 1 Gbps

**Recommended Production:**
- CPU: 16+ cores, 3.0+ GHz
- RAM: 32+ GB
- Storage: 500+ GB NVMe SSD
- Network: 10+ Gbps
- GPU: Optional, CUDA 11.0+ compatible

### Operating System Setup

#### Ubuntu/Debian

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    gfortran

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install additional tools
sudo apt install -y \
    htop \
    iotop \
    nethogs \
    prometheus-node-exporter
```

#### RHEL/CentOS

```bash
# Update system
sudo dnf update -y

# Install development tools
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    openssl-devel \
    postgresql-devel \
    blas-devel \
    lapack-devel \
    gcc-gfortran

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Environment Configuration

Create environment configuration file:

```bash
# /etc/nhits/production.env
RUST_LOG=info
NHITS_BIND_ADDRESS=0.0.0.0:8080
NHITS_DATABASE_URL=postgresql://user:pass@localhost/nhits
NHITS_REDIS_URL=redis://localhost:6379
NHITS_MAX_CONCURRENT_FORECASTS=100
NHITS_MODEL_CACHE_SIZE=1000
NHITS_CONSCIOUSNESS_ENABLED=true
NHITS_MONITORING_ENABLED=true
NHITS_METRICS_PORT=9090
```

## Container Deployment

### Dockerfile

```dockerfile
# Multi-stage build for production
FROM rust:1.75-slim as builder

WORKDIR /usr/src/app

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and build dependencies first (for caching)
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy source and build application
COPY src ./src
RUN touch src/main.rs && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -s /bin/false nhits

# Copy binary
COPY --from=builder /usr/src/app/target/release/nhits-server /usr/local/bin/

# Create directories
RUN mkdir -p /var/lib/nhits/models \
    && mkdir -p /var/log/nhits \
    && chown -R nhits:nhits /var/lib/nhits /var/log/nhits

USER nhits

EXPOSE 8080 9090

CMD ["nhits-server"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  nhits:
    build: .
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - RUST_LOG=info
      - NHITS_DATABASE_URL=postgresql://nhits:password@postgres:5432/nhits
      - NHITS_REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - nhits_models:/var/lib/nhits/models
      - nhits_logs:/var/log/nhits
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=nhits
      - POSTGRES_USER=nhits
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  nhits_models:
  nhits_logs:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Build and Deploy

```bash
# Build production image
docker build -t nhits:latest .

# Deploy with compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nhits

# Scale inference workers
docker-compose up -d --scale nhits=3
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nhits-system

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nhits-config
  namespace: nhits-system
data:
  config.toml: |
    [server]
    bind_address = "0.0.0.0:8080"
    max_concurrent_forecasts = 1000
    
    [database]
    url = "postgresql://nhits:password@postgres-service:5432/nhits"
    max_connections = 20
    
    [redis]
    url = "redis://redis-service:6379"
    
    [consciousness]
    enabled = true
    coherence_weight = 0.1
    
    [monitoring]
    enabled = true
    metrics_port = 9090
```

### Database Deployment

```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: nhits-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: nhits
        - name: POSTGRES_USER
          value: nhits
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: nhits-system
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### NHITS Application Deployment

```yaml
# k8s/nhits-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nhits-api
  namespace: nhits-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nhits-api
  template:
    metadata:
      labels:
        app: nhits-api
    spec:
      containers:
      - name: nhits
        image: nhits:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: NHITS_CONFIG_PATH
          value: "/etc/nhits/config.toml"
        volumeMounts:
        - name: config
          mountPath: /etc/nhits
        - name: models
          mountPath: /var/lib/nhits/models
        resources:
          limits:
            memory: "4Gi"
            cpu: "2000m"
          requests:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: config
        configMap:
          name: nhits-config
      - name: models
        persistentVolumeClaim:
          claimName: nhits-models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: nhits-service
  namespace: nhits-system
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    prometheus.io/path: "/metrics"
spec:
  selector:
    app: nhits-api
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nhits-ingress
  namespace: nhits-system
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.nhits.company.com
    secretName: nhits-tls
  rules:
  - host: api.nhits.company.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nhits-service
            port:
              number: 8080
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nhits-hpa
  namespace: nhits-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nhits-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/nhits-deployment.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check deployment status
kubectl get pods -n nhits-system
kubectl get services -n nhits-system
kubectl get ingress -n nhits-system

# Monitor logs
kubectl logs -f deployment/nhits-api -n nhits-system

# Scale deployment
kubectl scale deployment nhits-api --replicas=5 -n nhits-system
```

## Performance Optimization

### Resource Allocation

```yaml
# Optimal resource allocation for different workloads
resources:
  # Development
  dev:
    limits: { memory: "1Gi", cpu: "500m" }
    requests: { memory: "512Mi", cpu: "250m" }
  
  # Production - CPU intensive
  prod_cpu:
    limits: { memory: "4Gi", cpu: "4000m" }
    requests: { memory: "2Gi", cpu: "2000m" }
  
  # Production - Memory intensive
  prod_memory:
    limits: { memory: "8Gi", cpu: "2000m" }
    requests: { memory: "4Gi", cpu: "1000m" }
```

### JVM Tuning (if using JNI)

```bash
# Environment variables for JVM optimization
export JAVA_OPTS="-Xms2g -Xmx4g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Database Optimization

```sql
-- PostgreSQL configuration for NHITS
-- postgresql.conf optimizations

# Memory
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connections
max_connections = 100
max_prepared_transactions = 100

# WAL
wal_buffers = 16MB
checkpoint_completion_target = 0.9
wal_writer_delay = 200ms

# Query optimization
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "nhits_rules.yml"

scrape_configs:
  - job_name: 'nhits'
    static_configs:
      - targets: ['nhits-service:9090']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "NHITS Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(nhits_http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Forecast Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(nhits_forecast_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "nhits_model_accuracy",
            "legendFormat": "Current Accuracy"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

```yaml
# nhits_rules.yml
groups:
  - name: nhits_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(nhits_http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(nhits_forecast_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High forecast latency"
          description: "95th percentile latency is {{ $value }} seconds"

      - alert: ModelAccuracyDrop
        expr: nhits_model_accuracy < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Model accuracy dropped"
          description: "Model accuracy is {{ $value }}"

      - alert: ConsciousnessCoherenceLow
        expr: nhits_consciousness_coherence < 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Consciousness coherence low"
          description: "Coherence is {{ $value }}"
```

## Security Considerations

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nhits-network-policy
  namespace: nhits-system
spec:
  podSelector:
    matchLabels:
      app: nhits-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Authentication & Authorization

```rust
// API authentication middleware
use jwt::verify;

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    role: String,
}

async fn auth_middleware(
    req: Request<Body>,
    next: Next<Request<Body>>,
) -> Result<Response<Body>, StatusCode> {
    let auth_header = req.headers()
        .get("authorization")
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let token = auth_header
        .to_str()
        .map_err(|_| StatusCode::UNAUTHORIZED)?
        .strip_prefix("Bearer ")
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    let claims: Claims = verify_jwt(token)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;
    
    // Check permissions based on role
    match claims.role.as_str() {
        "admin" => { /* Full access */ },
        "user" => { /* Limited access */ },
        _ => return Err(StatusCode::FORBIDDEN),
    }
    
    Ok(next.run(req).await)
}
```

### Secrets Management

```yaml
# Kubernetes secrets
apiVersion: v1
kind: Secret
metadata:
  name: nhits-secrets
  namespace: nhits-system
type: Opaque
data:
  database-password: <base64-encoded-password>
  jwt-secret: <base64-encoded-jwt-secret>
  api-key: <base64-encoded-api-key>
```

## Scaling Strategies

### Horizontal Scaling

```bash
# Auto-scaling based on custom metrics
kubectl create hpa nhits-hpa \
  --cpu-percent=70 \
  --memory-percent=80 \
  --min=3 \
  --max=20 \
  -n nhits-system
```

### Vertical Scaling

```yaml
# Vertical Pod Autoscaler
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: nhits-vpa
  namespace: nhits-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nhits-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: nhits
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

### Load Balancing

```yaml
# Service configuration for load balancing
apiVersion: v1
kind: Service
metadata:
  name: nhits-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

## Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# backup-db.sh
set -e

BACKUP_DIR="/var/backups/nhits"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_HOST="postgres-service"
DB_NAME="nhits"
DB_USER="nhits"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME \
  --format=custom \
  --compress=9 \
  --file="$BACKUP_DIR/nhits_backup_$TIMESTAMP.dump"

# Cleanup old backups (keep last 30 days)
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete

echo "Backup completed: nhits_backup_$TIMESTAMP.dump"
```

### Model Backup

```bash
#!/bin/bash
# backup-models.sh
set -e

MODEL_DIR="/var/lib/nhits/models"
BACKUP_DIR="/var/backups/nhits/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
tar -czf "$BACKUP_DIR/models_backup_$TIMESTAMP.tar.gz" -C $MODEL_DIR .

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR/models_backup_$TIMESTAMP.tar.gz" \
  s3://nhits-backups/models/

echo "Model backup completed: models_backup_$TIMESTAMP.tar.gz"
```

### Disaster Recovery

```yaml
# CronJob for automated backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: nhits-backup
  namespace: nhits-system
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres-service -U nhits -d nhits \
                --format=custom --compress=9 \
                --file="/backup/nhits_$(date +%Y%m%d_%H%M%S).dump"
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          restartPolicy: OnFailure
          volumes:
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-pvc
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy NHITS

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cargo
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
        
    - name: Run tests
      run: cargo test --all-features
      
    - name: Run benchmarks
      run: cargo bench
      
    - name: Security audit
      run: cargo audit

  build:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
        
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        
    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/k8s-set-context@v1
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
        
    - name: Deploy to staging
      run: |
        helm upgrade --install nhits-staging ./helm \
          --namespace nhits-staging \
          --set image.tag=${{ github.sha }} \
          --wait
          
    - name: Run smoke tests
      run: |
        kubectl wait --for=condition=ready pod \
          -l app=nhits-api -n nhits-staging \
          --timeout=300s
        ./scripts/smoke-test.sh nhits-staging
        
    - name: Deploy to production
      if: success()
      run: |
        helm upgrade --install nhits-prod ./helm \
          --namespace nhits-production \
          --set image.tag=${{ github.sha }} \
          --wait
```

### Helm Chart

```yaml
# helm/values.yaml
replicaCount: 3

image:
  repository: ghcr.io/company/nhits
  pullPolicy: IfNotPresent
  tag: ""

service:
  type: ClusterIP
  port: 8080

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.nhits.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: nhits-tls
      hosts:
        - api.nhits.company.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}
```

## Best Practices

### Configuration Management
- Use environment-specific configurations
- Store secrets in secure vaults
- Implement configuration validation
- Use configuration templates

### Deployment Strategy
- Blue-green deployments for zero downtime
- Canary releases for gradual rollouts
- Health checks and readiness probes
- Graceful shutdown handling

### Monitoring
- Comprehensive metrics collection
- Alerting on key performance indicators
- Log aggregation and analysis
- Distributed tracing

### Security
- Regular security scans
- Network policies and isolation
- Authentication and authorization
- Secrets rotation

### Performance
- Resource limits and requests
- Auto-scaling configurations
- Load testing and optimization
- Database query optimization

This deployment guide provides a comprehensive foundation for deploying NHITS in production environments, from simple single-node setups to enterprise-scale Kubernetes deployments.