# Deployment Guide

## Overview

This guide covers the deployment of the Cerebellar-Norse neural network system across different environments, from development to production.

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 8 cores, 2.4GHz or higher
- **Memory**: 32GB RAM
- **Storage**: 100GB SSD
- **Network**: 1Gbps connection

#### Recommended Requirements
- **CPU**: 16+ cores, 3.0GHz or higher (Intel Xeon or AMD EPYC)
- **Memory**: 128GB RAM or higher
- **GPU**: NVIDIA RTX 4090 or Tesla V100 (for CUDA acceleration)
- **Storage**: 500GB NVMe SSD
- **Network**: 10Gbps connection with low latency

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 22.04 LTS, CentOS 8+, RHEL 8+
- **Kernel**: 5.4+ (required for optimal memory management)

#### Dependencies
```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
rustup target add x86_64-unknown-linux-gnu

# CUDA (for GPU acceleration)
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# System libraries
sudo apt update
sudo apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    clang \
    llvm-dev
```

## Deployment Methods

### 1. Container Deployment (Recommended)

#### Docker Setup
```dockerfile
# Dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install Rust and system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy source code
WORKDIR /app
COPY . .

# Build application
RUN cargo build --release --features lightning-gpu

# Runtime configuration
EXPOSE 8080
CMD ["./target/release/cerebellar-norse"]
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  cerebellar-norse:
    build: .
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - CEREBELLAR_CONFIG_PATH=/config/production.toml
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./config:/config:ro
      - ./data:/data
      - ./logs:/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro

volumes:
  grafana-storage:
```

### 2. Kubernetes Deployment

#### Deployment Manifest
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cerebellar-norse
  namespace: neural-systems
  labels:
    app: cerebellar-norse
    version: v0.1.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: cerebellar-norse
  template:
    metadata:
      labels:
        app: cerebellar-norse
        version: v0.1.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: cerebellar-norse
        image: cerebellar-norse:v0.1.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: CEREBELLAR_CONFIG_PATH
          value: "/config/production.toml"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        - name: data
          mountPath: /data
        - name: logs
          mountPath: /logs
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
      volumes:
      - name: config
        configMap:
          name: cerebellar-norse-config
      - name: data
        persistentVolumeClaim:
          claimName: cerebellar-norse-data
      - name: logs
        persistentVolumeClaim:
          claimName: cerebellar-norse-logs
      nodeSelector:
        gpu: nvidia
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

#### Service Configuration
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cerebellar-norse-service
  namespace: neural-systems
  labels:
    app: cerebellar-norse
spec:
  selector:
    app: cerebellar-norse
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: cerebellar-norse-ingress
  namespace: neural-systems
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.cerebellar-norse.ai
    secretName: cerebellar-norse-tls
  rules:
  - host: api.cerebellar-norse.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: cerebellar-norse-service
            port:
              number: 80
```

### 3. Bare Metal Deployment

#### SystemD Service
```ini
# /etc/systemd/system/cerebellar-norse.service
[Unit]
Description=Cerebellar-Norse Neural Network Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=cerebellar
Group=cerebellar
WorkingDirectory=/opt/cerebellar-norse
ExecStart=/opt/cerebellar-norse/target/release/cerebellar-norse
ExecReload=/bin/kill -HUP $MAINPID
KillMode=process
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=cerebellar-norse

# Environment variables
Environment=RUST_LOG=info
Environment=CEREBELLAR_CONFIG_PATH=/etc/cerebellar-norse/production.toml

# Resource limits
LimitNOFILE=65536
LimitNPROC=65536

# Security
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/var/lib/cerebellar-norse /var/log/cerebellar-norse

[Install]
WantedBy=multi-user.target
```

#### Installation Script
```bash
#!/bin/bash
# install.sh

set -euo pipefail

# Configuration
INSTALL_DIR="/opt/cerebellar-norse"
CONFIG_DIR="/etc/cerebellar-norse"
DATA_DIR="/var/lib/cerebellar-norse"
LOG_DIR="/var/log/cerebellar-norse"
USER="cerebellar"
GROUP="cerebellar"

# Create user and directories
sudo useradd -r -s /bin/false -d $INSTALL_DIR $USER || true
sudo mkdir -p $INSTALL_DIR $CONFIG_DIR $DATA_DIR $LOG_DIR
sudo chown -R $USER:$GROUP $INSTALL_DIR $DATA_DIR $LOG_DIR
sudo chmod 755 $CONFIG_DIR

# Build and install
cargo build --release --features lightning-gpu
sudo cp target/release/cerebellar-norse $INSTALL_DIR/
sudo cp -r config/* $CONFIG_DIR/
sudo chown -R $USER:$GROUP $INSTALL_DIR

# Install systemd service
sudo cp deploy/cerebellar-norse.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cerebellar-norse

echo "Installation completed. Start with: sudo systemctl start cerebellar-norse"
```

## Configuration Management

### Environment-Specific Configurations

#### Development Configuration
```toml
# config/development.toml
[neural]
granule_size = 1000
purkinje_size = 50
golgi_size = 25
dcn_size = 5
learning_rate = 0.01
sparsity = 0.1

[performance]
device = "cpu"
batch_size = 32
max_threads = 4

[logging]
level = "debug"
format = "human"

[monitoring]
enabled = true
metrics_port = 9090
tracing_enabled = true
```

#### Production Configuration
```toml
# config/production.toml
[neural]
granule_size = 4000000
purkinje_size = 15000
golgi_size = 400
dcn_size = 100
learning_rate = 0.001
sparsity = 0.02

[performance]
device = "cuda"
batch_size = 1024
max_threads = 16
memory_pool_size = "30GB"
cuda_streams = 4

[security]
tls_enabled = true
cert_path = "/etc/ssl/certs/cerebellar-norse.pem"
key_path = "/etc/ssl/private/cerebellar-norse.key"
api_key_required = true

[logging]
level = "info"
format = "json"
output = "file"
rotation = "daily"
max_files = 30

[monitoring]
enabled = true
metrics_port = 9090
prometheus_enabled = true
jaeger_enabled = true
health_check_interval = 30
```

### Configuration Validation

```bash
#!/bin/bash
# validate-config.sh

CONFIG_FILE=${1:-"config/production.toml"}

echo "Validating configuration: $CONFIG_FILE"

# Check file exists and is readable
if [[ ! -r "$CONFIG_FILE" ]]; then
    echo "ERROR: Configuration file not found or not readable: $CONFIG_FILE"
    exit 1
fi

# Validate TOML syntax
if ! toml-test --validate "$CONFIG_FILE" 2>/dev/null; then
    echo "ERROR: Invalid TOML syntax in $CONFIG_FILE"
    exit 1
fi

# Validate against schema
if ! cerebellar-norse --config "$CONFIG_FILE" --validate-only; then
    echo "ERROR: Configuration validation failed"
    exit 1
fi

echo "Configuration validation successful"
```

## Deployment Checklist

### Pre-Deployment
- [ ] Hardware requirements met
- [ ] All dependencies installed
- [ ] Configuration files validated
- [ ] SSL certificates generated/obtained
- [ ] Firewall rules configured
- [ ] Monitoring infrastructure ready
- [ ] Backup procedures tested

### Deployment
- [ ] Application binary deployed
- [ ] Configuration files in place
- [ ] Service started successfully
- [ ] Health checks passing
- [ ] Metrics collection working
- [ ] Load balancer configured
- [ ] DNS records updated

### Post-Deployment
- [ ] Smoke tests executed
- [ ] Performance benchmarks run
- [ ] Monitoring dashboards verified
- [ ] Alert rules tested
- [ ] Documentation updated
- [ ] Team notified of deployment

## Rollback Procedures

### Automated Rollback (Kubernetes)
```bash
# Rollback to previous version
kubectl rollout undo deployment/cerebellar-norse -n neural-systems

# Rollback to specific revision
kubectl rollout undo deployment/cerebellar-norse --to-revision=2 -n neural-systems

# Check rollback status
kubectl rollout status deployment/cerebellar-norse -n neural-systems
```

### Manual Rollback (SystemD)
```bash
# Stop current service
sudo systemctl stop cerebellar-norse

# Restore previous binary
sudo cp /opt/cerebellar-norse/backup/cerebellar-norse /opt/cerebellar-norse/

# Restore previous configuration
sudo cp /etc/cerebellar-norse/backup/production.toml /etc/cerebellar-norse/

# Start service
sudo systemctl start cerebellar-norse

# Verify rollback
sudo systemctl status cerebellar-norse
curl -f http://localhost:8080/health
```

## Performance Tuning

### Kernel Parameters
```bash
# /etc/sysctl.d/99-cerebellar-norse.conf

# Network optimization
net.core.rmem_max = 268435456
net.core.wmem_max = 268435456
net.ipv4.tcp_rmem = 4096 87380 268435456
net.ipv4.tcp_wmem = 4096 65536 268435456

# Memory optimization
vm.swappiness = 1
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# CPU optimization
kernel.sched_migration_cost_ns = 5000000
kernel.sched_autogroup_enabled = 0
```

### CUDA Optimization
```bash
# GPU performance mode
sudo nvidia-smi -pm 1

# Set GPU clock speeds
sudo nvidia-smi -ac 1215,1410

# Persistence mode
sudo nvidia-smi -pm 1
```

## Security Considerations

### Network Security
- Use TLS 1.3 for all communications
- Implement proper firewall rules
- Regular security scanning
- API rate limiting

### Application Security
- Input validation and sanitization
- Secure configuration management
- Regular dependency updates
- Security audit logging

### Infrastructure Security
- Regular OS updates
- Intrusion detection system
- Access control and authentication
- Encrypted storage

## Disaster Recovery

### Backup Strategy
- Daily automated backups of configuration and models
- Off-site backup storage
- Recovery time objective (RTO): 4 hours
- Recovery point objective (RPO): 1 hour

### Recovery Procedures
1. Assess the failure scope
2. Activate disaster recovery plan
3. Restore from backups
4. Validate system functionality
5. Update monitoring and alerting
6. Document lessons learned

---

*For support and questions, contact the DevOps team or create an issue in the project repository.*