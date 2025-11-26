# Deployment Guide

Comprehensive guide for deploying the AI News Trading Platform with Neural Forecasting in production environments.

## Deployment Overview

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │  Web Interface  │    │  Monitoring     │
│   (NGINX/HAProxy│    │  (Streamlit)    │    │  (Grafana)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Claude-Flow Orchestrator                       │
├─────────────────────────────────┼─────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ MCP Server  │  │ Neural      │  │ Trading     │  │ Risk     │ │
│  │ (Port 3000) │  │ Forecasting │  │ Strategies  │  │ Manager  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │                │                │               │
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ PostgreSQL  │  │ Redis Cache │  │ GPU Cluster │  │ Log Storage │
│ Database    │  │             │  │ (CUDA)      │  │ (ELK Stack) │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

### Deployment Strategies

1. **Single Server**: Development and small-scale production
2. **Multi-Server**: Medium-scale with load balancing
3. **Kubernetes**: Large-scale with orchestration
4. **Cloud Native**: Serverless and managed services

## Single Server Deployment

### Hardware Requirements

#### Production Server Specifications
- **CPU**: 16+ cores, 3.5+ GHz
- **RAM**: 64GB+ DDR4
- **Storage**: 1TB+ NVMe SSD
- **GPU**: NVIDIA A100, V100, or RTX 4090
- **Network**: 10 Gbps connection
- **OS**: Ubuntu 22.04 LTS Server

### System Preparation

#### 1. System Updates

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    curl wget git vim htop \
    build-essential software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release
```

#### 2. User Setup

```bash
# Create application user
sudo useradd -m -s /bin/bash ai-trader
sudo usermod -aG sudo ai-trader

# Setup SSH keys
sudo -u ai-trader ssh-keygen -t rsa -b 4096 -C "ai-trader@production"
```

#### 3. Security Configuration

```bash
# Configure firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 3000/tcp  # MCP Server
sudo ufw allow 8080/tcp  # Web Interface
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 80/tcp    # HTTP

# Install fail2ban
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
```

### Application Deployment

#### 1. Clone and Setup

```bash
# Switch to application user
sudo -u ai-trader -i

# Clone repository
git clone https://github.com/your-org/ai-news-trader.git /opt/ai-news-trader
cd /opt/ai-news-trader

# Checkout production tag
git checkout production

# Set permissions
sudo chown -R ai-trader:ai-trader /opt/ai-news-trader
```

#### 2. Environment Setup

```bash
# Create Python environment
python3 -m venv /opt/ai-news-trader/venv
source /opt/ai-news-trader/venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements-mcp.txt
pip install neuralforecast[gpu]
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Configuration

Create production configuration `/opt/ai-news-trader/.env.production`:

```bash
# Production Environment Variables
NODE_ENV=production
FLASK_ENV=production

# System Configuration
NEURAL_FORECAST_GPU=true
NEURAL_FORECAST_DEVICE=cuda
NEURAL_FORECAST_BATCH_SIZE=64
NEURAL_FORECAST_MAX_MEMORY=24

# MCP Server Configuration
MCP_SERVER_PORT=3000
MCP_SERVER_HOST=0.0.0.0
MCP_NEURAL_ENABLED=true
MCP_GPU_ACCELERATION=true
MCP_WORKERS=8

# Database Configuration (PostgreSQL)
DATABASE_URL=postgresql://ai_trader:secure_password@localhost:5432/ai_news_trader_prod

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CACHE_DEFAULT_TIMEOUT=3600

# API Keys (from secure store)
ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
FINNHUB_API_KEY=${FINNHUB_API_KEY}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
SECURITY_PASSWORD_SALT=${SECURITY_PASSWORD_SALT}

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/ai-news-trader/app.log
ACCESS_LOG=/var/log/ai-news-trader/access.log
ERROR_LOG=/var/log/ai-news-trader/error.log

# Performance
MAX_WORKERS=16
WORKER_TIMEOUT=300
KEEP_ALIVE=2
MAX_REQUESTS=10000
MAX_REQUESTS_JITTER=100

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_PATH=/health
```

#### 4. Database Setup

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database
sudo -u postgres psql << EOF
CREATE DATABASE ai_news_trader_prod;
CREATE USER ai_trader WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE ai_news_trader_prod TO ai_trader;
ALTER USER ai_trader CREATEDB;
\q
EOF

# Install Redis
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

#### 5. Service Configuration

Create systemd service `/etc/systemd/system/ai-news-trader.service`:

```ini
[Unit]
Description=AI News Trading Platform
After=network.target postgresql.service redis.service
Wants=postgresql.service redis.service

[Service]
Type=exec
User=ai-trader
Group=ai-trader
WorkingDirectory=/opt/ai-news-trader
Environment=PATH=/opt/ai-news-trader/venv/bin
EnvironmentFile=/opt/ai-news-trader/.env.production
ExecStart=/opt/ai-news-trader/venv/bin/python mcp_server_enhanced.py
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
PrivateTmp=true
Restart=always
RestartSec=10

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/ai-news-trader /var/log/ai-news-trader /tmp

# Resource limits
LimitNOFILE=65536
LimitNPROC=8192

[Install]
WantedBy=multi-user.target
```

Create claude-flow service `/etc/systemd/system/claude-flow.service`:

```ini
[Unit]
Description=Claude-Flow Orchestrator
After=network.target ai-news-trader.service
Wants=ai-news-trader.service

[Service]
Type=exec
User=ai-trader
Group=ai-trader
WorkingDirectory=/opt/ai-news-trader
Environment=PATH=/opt/ai-news-trader/venv/bin
EnvironmentFile=/opt/ai-news-trader/.env.production
ExecStart=/opt/ai-news-trader/claude-flow start --neural-forecast --gpu --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start services:

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable ai-news-trader
sudo systemctl enable claude-flow

# Start services
sudo systemctl start ai-news-trader
sudo systemctl start claude-flow

# Check status
sudo systemctl status ai-news-trader
sudo systemctl status claude-flow
```

### Reverse Proxy Setup (NGINX)

#### 1. Install NGINX

```bash
sudo apt install -y nginx
sudo systemctl enable nginx
```

#### 2. SSL Certificate

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com -d api.your-domain.com
```

#### 3. NGINX Configuration

Create `/etc/nginx/sites-available/ai-news-trader`:

```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=forecast:10m rate=5r/s;

# Upstream servers
upstream mcp_server {
    server 127.0.0.1:3000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream claude_flow {
    server 127.0.0.1:8080 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# Main server block
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # Logging
    access_log /var/log/nginx/ai-news-trader-access.log;
    error_log /var/log/nginx/ai-news-trader-error.log;
    
    # Claude-Flow Web Interface
    location / {
        proxy_pass http://claude_flow;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # MCP API endpoints
    location /mcp {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://mcp_server;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings for long-running operations
        proxy_read_timeout 600s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 600s;
    }
    
    # Neural forecasting endpoints (stricter rate limiting)
    location /mcp/neural {
        limit_req zone=forecast burst=10 nodelay;
        
        proxy_pass http://mcp_server;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Extended timeouts for neural processing
        proxy_read_timeout 900s;
        proxy_connect_timeout 75s;
        proxy_send_timeout 900s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://mcp_server/health;
        access_log off;
    }
    
    # Static files (if any)
    location /static {
        alias /opt/ai-news-trader/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# API subdomain
server {
    listen 443 ssl http2;
    server_name api.your-domain.com;
    
    # SSL configuration (same as above)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # API-specific configuration
    location / {
        limit_req zone=api burst=50 nodelay;
        
        proxy_pass http://mcp_server;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers for API access
        add_header Access-Control-Allow-Origin "*";
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization";
        
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin "*";
            add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
            add_header Access-Control-Allow-Headers "Origin, X-Requested-With, Content-Type, Accept, Authorization";
            add_header Content-Length 0;
            add_header Content-Type text/plain;
            return 200;
        }
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name your-domain.com api.your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

Enable the configuration:

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ai-news-trader /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx
```

## Multi-Server Deployment

### Architecture Overview

```
Internet
    │
    ▼
┌─────────────┐
│ Load        │
│ Balancer    │ (HAProxy/AWS ALB)
└─────────────┘
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ App Server  │ │ App Server  │ │ App Server  │
│ Node 1      │ │ Node 2      │ │ Node 3      │
└─────────────┘ └─────────────┘ └─────────────┘
    │                 │                 │
    └─────────────────┼─────────────────┘
                      ▼
            ┌─────────────────┐
            │ Shared Services │
            │ • PostgreSQL    │
            │ • Redis Cluster │
            │ • GPU Cluster   │
            └─────────────────┘
```

### Load Balancer Configuration (HAProxy)

Install HAProxy:

```bash
sudo apt install -y haproxy
```

Configure `/etc/haproxy/haproxy.cfg`:

```
global
    daemon
    maxconn 4096
    log stdout local0 info
    
defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms
    option httplog
    
# Frontend configuration
frontend ai_news_trader_frontend
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/your-domain.pem
    redirect scheme https if !{ ssl_fc }
    
    # ACLs for routing
    acl is_api hdr_beg(host) -i api.
    acl is_mcp path_beg /mcp
    acl is_neural path_beg /mcp/neural
    
    # Route to appropriate backend
    use_backend neural_backend if is_neural
    use_backend api_backend if is_api or is_mcp
    default_backend web_backend

# Backend configurations
backend web_backend
    balance roundrobin
    option httpchk GET /health
    server web1 10.0.1.10:8080 check
    server web2 10.0.1.11:8080 check
    server web3 10.0.1.12:8080 check

backend api_backend
    balance roundrobin
    option httpchk GET /health
    server api1 10.0.1.10:3000 check
    server api2 10.0.1.11:3000 check
    server api3 10.0.1.12:3000 check

backend neural_backend
    balance leastconn  # Use least connections for GPU-intensive tasks
    option httpchk GET /health
    timeout server 900s  # Extended timeout for neural processing
    server gpu1 10.0.1.20:3000 check
    server gpu2 10.0.1.21:3000 check
```

### Shared Database Setup

#### PostgreSQL Cluster

Set up PostgreSQL with streaming replication:

**Master Server (10.0.1.100):**

```bash
# Install PostgreSQL
sudo apt install -y postgresql-14 postgresql-contrib-14

# Configure postgresql.conf
sudo -u postgres psql << EOF
ALTER SYSTEM SET listen_addresses = '*';
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET max_wal_senders = 3;
ALTER SYSTEM SET wal_keep_size = '100MB';
ALTER SYSTEM SET synchronous_commit = 'on';
SELECT pg_reload_conf();
EOF

# Configure pg_hba.conf
echo "host replication replicator 10.0.1.0/24 md5" | sudo tee -a /etc/postgresql/14/main/pg_hba.conf

# Create replication user
sudo -u postgres psql << EOF
CREATE ROLE replicator WITH REPLICATION LOGIN PASSWORD 'replica_password';
EOF

# Restart PostgreSQL
sudo systemctl restart postgresql
```

**Replica Server (10.0.1.101):**

```bash
# Stop PostgreSQL
sudo systemctl stop postgresql

# Clear data directory
sudo rm -rf /var/lib/postgresql/14/main/*

# Create base backup
sudo -u postgres pg_basebackup -h 10.0.1.100 -D /var/lib/postgresql/14/main -U replicator -W

# Configure standby.signal
sudo -u postgres touch /var/lib/postgresql/14/main/standby.signal

# Configure postgresql.conf
sudo -u postgres tee /var/lib/postgresql/14/main/postgresql.auto.conf << EOF
primary_conninfo = 'host=10.0.1.100 port=5432 user=replicator password=replica_password'
promote_trigger_file = '/var/lib/postgresql/14/main/promote_trigger'
EOF

# Start PostgreSQL
sudo systemctl start postgresql
```

#### Redis Cluster

Set up Redis Cluster for caching:

```bash
# Install Redis on each node
sudo apt install -y redis-server

# Configure Redis cluster
redis-cli --cluster create \
  10.0.1.10:6379 10.0.1.11:6379 10.0.1.12:6379 \
  10.0.1.13:6379 10.0.1.14:6379 10.0.1.15:6379 \
  --cluster-replicas 1
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3.0+
- NVIDIA GPU Operator (for GPU support)

### GPU Node Setup

Install NVIDIA GPU Operator:

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
  nvidia/gpu-operator \
  --set driver.enabled=true
```

### Application Deployment

#### 1. Namespace and Secrets

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-news-trader
---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-news-trader-secrets
  namespace: ai-news-trader
type: Opaque
stringData:
  DATABASE_URL: "postgresql://ai_trader:password@postgres:5432/ai_news_trader"
  REDIS_URL: "redis://redis-cluster:6379/0"
  ALPHA_VANTAGE_API_KEY: "your-api-key"
  FINNHUB_API_KEY: "your-api-key"
  OPENAI_API_KEY: "your-api-key"
```

#### 2. ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-news-trader-config
  namespace: ai-news-trader
data:
  neural_forecast.yaml: |
    models:
      nhits:
        input_size: 168
        horizon: 30
        max_epochs: 100
        batch_size: 32
        accelerator: "gpu"
        devices: [0]
    gpu:
      enabled: true
      memory_fraction: 0.8
      allow_growth: true
```

#### 3. Database Deployment

```yaml
# postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ai-news-trader
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
        image: postgres:14
        env:
        - name: POSTGRES_DB
          value: ai_news_trader
        - name: POSTGRES_USER
          value: ai_trader
        - name: POSTGRES_PASSWORD
          value: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ai-news-trader
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
```

#### 4. Application Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-news-trader
  namespace: ai-news-trader
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-news-trader
  template:
    metadata:
      labels:
        app: ai-news-trader
    spec:
      containers:
      - name: mcp-server
        image: ai-news-trader:latest
        command: ["python", "mcp_server_enhanced.py"]
        ports:
        - containerPort: 3000
        env:
        - name: MCP_SERVER_PORT
          value: "3000"
        - name: MCP_SERVER_HOST
          value: "0.0.0.0"
        - name: NEURAL_FORECAST_GPU
          value: "true"
        envFrom:
        - secretRef:
            name: ai-news-trader-secrets
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
      - name: claude-flow
        image: ai-news-trader:latest
        command: ["./claude-flow", "start", "--neural-forecast", "--gpu", "--port", "8080"]
        ports:
        - containerPort: 8080
        envFrom:
        - secretRef:
            name: ai-news-trader-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
      volumes:
      - name: config-volume
        configMap:
          name: ai-news-trader-config
      nodeSelector:
        accelerator: nvidia-tesla-k80
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

#### 5. Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-news-trader-service
  namespace: ai-news-trader
spec:
  selector:
    app: ai-news-trader
  ports:
  - name: mcp-server
    port: 3000
    targetPort: 3000
  - name: claude-flow
    port: 8080
    targetPort: 8080
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-news-trader-ingress
  namespace: ai-news-trader
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - your-domain.com
    - api.your-domain.com
    secretName: ai-news-trader-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-news-trader-service
            port:
              number: 8080
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-news-trader-service
            port:
              number: 3000
```

#### Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f secrets.yaml
kubectl apply -f configmap.yaml
kubectl apply -f postgres.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n ai-news-trader
kubectl get services -n ai-news-trader
kubectl get ingress -n ai-news-trader
```

## Cloud Native Deployment

### AWS Deployment

#### Using EKS with GPU Nodes

```bash
# Create EKS cluster with GPU nodes
eksctl create cluster \
  --name ai-news-trader \
  --region us-west-2 \
  --node-type p3.2xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --ssh-access \
  --ssh-public-key ~/.ssh/id_rsa.pub

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
```

#### Using AWS Batch for Batch Processing

```yaml
# batch-job-definition.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: neural-forecast-batch
spec:
  template:
    spec:
      containers:
      - name: neural-forecaster
        image: ai-news-trader:latest
        command: ["python", "-m", "src.forecasting.cli", "batch-forecast"]
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"  
            cpu: "8"
        env:
        - name: AWS_BATCH_JOB_ID
          value: "$(AWS_BATCH_JOB_ID)"
        volumeMounts:
        - name: shared-storage
          mountPath: /data
      volumes:
      - name: shared-storage
        persistentVolumeClaim:
          claimName: efs-claim
      restartPolicy: Never
  backoffLimit: 3
```

### Google Cloud Platform (GCP)

#### Using GKE with GPU Nodes

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create ai-news-trader \
  --accelerator type=nvidia-tesla-v100,count=1 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --machine-type n1-standard-4 \
  --zone us-central1-a

# Install NVIDIA drivers
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Azure Deployment

#### Using AKS with GPU Nodes

```bash
# Create AKS cluster with GPU nodes
az aks create \
  --resource-group ai-news-trader-rg \
  --name ai-news-trader-aks \
  --node-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/Azure/aks-engine/master/examples/addons/nvidia-device-plugin/nvidia-device-plugin.yaml
```

## Monitoring and Observability

### Prometheus and Grafana

#### Install Prometheus Stack

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword='admin123'
```

#### Custom Metrics Collection

Create `metrics-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'ai-news-trader'
      static_configs:
      - targets: ['ai-news-trader-service.ai-news-trader:9090']
      metrics_path: '/metrics'
      scrape_interval: 10s
    - job_name: 'neural-forecasting'
      static_configs:
      - targets: ['ai-news-trader-service.ai-news-trader:9091']
      metrics_path: '/neural/metrics'
      scrape_interval: 30s
```

### Grafana Dashboards

#### Neural Forecasting Dashboard

```json
{
  "dashboard": {
    "title": "Neural Forecasting Performance",
    "panels": [
      {
        "title": "Forecast Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, neural_forecast_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_percent",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "graph",
        "targets": [
          {
            "expr": "neural_forecast_accuracy_mape",
            "legendFormat": "MAPE {{model}}"
          }
        ]
      }
    ]
  }
}
```

### Logging with ELK Stack

#### Elasticsearch Configuration

```yaml
# elasticsearch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: elasticsearch
  namespace: logging
spec:
  replicas: 1
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
    spec:
      containers:
      - name: elasticsearch
        image: docker.elastic.co/elasticsearch/elasticsearch:8.6.0
        env:
        - name: discovery.type
          value: single-node
        - name: ES_JAVA_OPTS
          value: "-Xms2g -Xmx2g"
        ports:
        - containerPort: 9200
        volumeMounts:
        - name: es-data
          mountPath: /usr/share/elasticsearch/data
      volumes:
      - name: es-data
        persistentVolumeClaim:
          claimName: es-pvc
```

#### Logstash Configuration

```yaml
# logstash-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: logstash-config
  namespace: logging
data:
  logstash.conf: |
    input {
      beats {
        port => 5044
      }
    }
    filter {
      if [fields][app] == "ai-news-trader" {
        grok {
          match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{WORD:level} - %{GREEDYDATA:msg}" }
        }
        if [msg] =~ /neural_forecast/ {
          grok {
            match => { "msg" => "neural_forecast symbol=%{WORD:symbol} latency=%{NUMBER:latency:float}ms accuracy=%{NUMBER:accuracy:float}" }
          }
        }
      }
    }
    output {
      elasticsearch {
        hosts => ["elasticsearch:9200"]
        index => "ai-news-trader-%{+YYYY.MM.dd}"
      }
    }
```

### Alerting

#### Prometheus Alerting Rules

```yaml
# alerts.yaml
groups:
- name: ai-news-trader
  rules:
  - alert: HighForecastLatency
    expr: histogram_quantile(0.95, neural_forecast_duration_seconds_bucket) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High neural forecast latency detected"
      description: "95th percentile forecast latency is {{ $value }}s"
      
  - alert: GPUMemoryHigh
    expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.9
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "GPU memory usage high"
      description: "GPU memory usage is {{ $value | humanizePercentage }}"
      
  - alert: ModelAccuracyDegraded
    expr: neural_forecast_accuracy_mape > 0.1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy degraded"
      description: "Model MAPE is {{ $value | humanizePercentage }}"
```

## Backup and Disaster Recovery

### Database Backup

#### Automated PostgreSQL Backup

```bash
#!/bin/bash
# backup-postgres.sh

BACKUP_DIR="/opt/backups/postgres"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATABASE="ai_news_trader_prod"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database dump
pg_dump -h localhost -U ai_trader -d $DATABASE | gzip > $BACKUP_DIR/backup_${TIMESTAMP}.sql.gz

# Upload to S3 (optional)
aws s3 cp $BACKUP_DIR/backup_${TIMESTAMP}.sql.gz s3://your-backup-bucket/postgres/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

Create cron job:

```bash
# Add to crontab
0 2 * * * /opt/ai-news-trader/scripts/backup-postgres.sh
```

### Model Backup

#### Neural Model Versioning

```python
# model_backup.py
import boto3
import os
from datetime import datetime

def backup_models():
    s3 = boto3.client('s3')
    bucket = 'your-model-bucket'
    
    model_dir = '/opt/ai-news-trader/models'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pth'):
            local_path = os.path.join(model_dir, model_file)
            s3_key = f'models/{timestamp}/{model_file}'
            
            s3.upload_file(local_path, bucket, s3_key)
            print(f'Uploaded {model_file} to s3://{bucket}/{s3_key}')

if __name__ == '__main__':
    backup_models()
```

### Disaster Recovery Plan

#### Recovery Procedures

1. **Database Recovery**:
   ```bash
   # Restore from backup
   gunzip -c backup_20241226_020000.sql.gz | psql -h localhost -U ai_trader -d ai_news_trader_prod
   ```

2. **Model Recovery**:
   ```bash
   # Download models from S3
   aws s3 sync s3://your-model-bucket/models/latest/ /opt/ai-news-trader/models/
   ```

3. **Service Recovery**:
   ```bash
   # Restart services
   sudo systemctl restart ai-news-trader
   sudo systemctl restart claude-flow
   ```

### High Availability Setup

#### Database High Availability

Configure PostgreSQL streaming replication with automatic failover using Patroni:

```yaml
# patroni.yml
scope: ai-news-trader-cluster
namespace: /pg_cluster/
name: postgresql-01

restapi:
  listen: 0.0.0.0:8008
  connect_address: 10.0.1.100:8008

etcd:
  hosts: 10.0.1.10:2379,10.0.1.11:2379,10.0.1.12:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    retry_timeout: 30
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        wal_level: replica
        hot_standby: "on"
        max_connections: 100
        max_worker_processes: 8
        wal_keep_segments: 8
        max_wal_senders: 10
        max_replication_slots: 10
        checkpoint_timeout: 30

postgresql:
  listen: 0.0.0.0:5432
  connect_address: 10.0.1.100:5432
  data_dir: /var/lib/postgresql/14/main
  pgpass: /tmp/pgpass
  authentication:
    replication:
      username: replicator
      password: replica_password
    superuser:
      username: postgres
      password: super_password

tags:
  nofailover: false
  noloadbalance: false
  clonefrom: false
  nosync: false
```

## Security

### SSL/TLS Configuration

#### Certificate Management

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain certificates
sudo certbot --nginx -d your-domain.com -d api.your-domain.com

# Set up auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### Firewall Configuration

```bash
# Configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow from 10.0.1.0/24 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.1.0/24 to any port 6379  # Redis

# Enable firewall
sudo ufw enable
```

### Application Security

#### Authentication and Authorization

```python
# security_config.py
SECURITY_CONFIG = {
    'jwt': {
        'secret_key': os.environ['JWT_SECRET_KEY'],
        'expiration_hours': 24,
        'refresh_expiration_days': 30
    },
    'rate_limiting': {
        'default': '100/hour',
        'neural_forecast': '10/minute',
        'heavy_computation': '5/minute'
    },
    'cors': {
        'origins': ['https://your-domain.com'],
        'methods': ['GET', 'POST'],
        'headers': ['Content-Type', 'Authorization']
    }
}
```

#### Input Validation

```python
# validation.py
from marshmallow import Schema, fields, validate

class ForecastRequestSchema(Schema):
    symbol = fields.Str(required=True, validate=validate.Regexp(r'^[A-Z]{1,5}$'))
    horizon = fields.Int(required=True, validate=validate.Range(min=1, max=365))
    model = fields.Str(validate=validate.OneOf(['nhits', 'nbeats', 'autoformer']))
    confidence = fields.List(fields.Int(validate=validate.Range(min=1, max=99)))
```

## Performance Optimization

### Database Optimization

#### PostgreSQL Configuration

```sql
-- postgresql.conf optimizations
shared_buffers = '8GB'
effective_cache_size = '24GB'
maintenance_work_mem = '2GB'
checkpoint_completion_target = 0.9
wal_buffers = '64MB'
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = '128MB'
min_wal_size = '2GB'
max_wal_size = '8GB'

-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_forecasts_symbol_date ON forecasts(symbol, created_at);
CREATE INDEX CONCURRENTLY idx_trades_strategy_symbol ON trades(strategy, symbol);
CREATE INDEX CONCURRENTLY idx_news_sentiment_symbol_date ON news_sentiment(symbol, published_at);
```

### Application Optimization

#### Connection Pooling

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

#### Caching Strategy

```python
# cache.py
import redis
from functools import wraps

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

def cache_forecast(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"forecast:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Compute and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

### GPU Optimization

#### Memory Management

```python
# gpu_optimization.py
import torch

class GPUMemoryManager:
    def __init__(self, memory_fraction=0.8):
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            torch.cuda.empty_cache()
    
    def __enter__(self):
        if torch.cuda.is_available():
            self.initial_memory = torch.cuda.memory_allocated()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            if final_memory > self.initial_memory * 1.5:
                print(f"Warning: GPU memory increased by {final_memory - self.initial_memory} bytes")
```

## Testing in Production

### Health Checks

```bash
#!/bin/bash
# health_check.sh

# Check MCP server
curl -f http://localhost:3000/health || exit 1

# Check Claude-flow
curl -f http://localhost:8080/health || exit 1

# Check database connectivity
pg_isready -h localhost -p 5432 -U ai_trader || exit 1

# Check Redis
redis-cli ping | grep -q PONG || exit 1

# Check GPU availability
nvidia-smi > /dev/null || exit 1

echo "All health checks passed"
```

### Load Testing

```python
# load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def test_forecast_endpoint(session, symbol):
    async with session.post('/mcp', json={
        "jsonrpc": "2.0",
        "method": "quick_analysis",
        "params": {"symbol": symbol, "use_gpu": True},
        "id": 1
    }) as response:
        return await response.json()

async def run_load_test(num_requests=100, concurrency=10):
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(
        connector=connector,
        base_url="http://localhost:3000"
    ) as session:
        
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
        tasks = []
        
        for i in range(num_requests):
            symbol = symbols[i % len(symbols)]
            task = test_forecast_endpoint(session, symbol)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = len([r for r in results if not isinstance(r, Exception)])
        failed = len(results) - successful
        
        print(f"Load test results:")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Duration: {end_time - start_time:.2f}s")
        print(f"RPS: {num_requests / (end_time - start_time):.2f}")

if __name__ == "__main__":
    asyncio.run(run_load_test())
```

### Canary Deployment

```yaml
# canary-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: ai-news-trader-rollout
  namespace: ai-news-trader
spec:
  replicas: 10
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 10m}
      - setWeight: 25
      - pause: {duration: 10m}
      - setWeight: 50
      - pause: {duration: 10m}
      - setWeight: 75
      - pause: {duration: 5m}
      analysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: ai-news-trader-service
  selector:
    matchLabels:
      app: ai-news-trader
  template:
    metadata:
      labels:
        app: ai-news-trader
    spec:
      containers:
      - name: ai-news-trader
        image: ai-news-trader:latest
        # ... rest of container spec
```

## Troubleshooting

### Common Deployment Issues

#### Port Conflicts

```bash
# Check port usage
sudo netstat -tulpn | grep :3000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:3000)
```

#### GPU Issues

```bash
# Check GPU status
nvidia-smi

# Reset GPU
sudo nvidia-smi --gpu-reset -i 0

# Check CUDA installation
nvcc --version
```

#### Memory Issues

```bash
# Check memory usage
free -h
sudo systemctl status ai-news-trader

# Check for memory leaks
ps aux --sort=-%mem | head -10
```

#### Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U ai_trader -d ai_news_trader_prod -c "SELECT 1;"

# Check logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

### Performance Issues

#### Slow Neural Forecasting

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Profile model performance
python -m cProfile -o profile.stats src/forecasting/neural_forecast_integration.py

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### High Latency

```bash
# Check network latency
ping localhost

# Check system load
htop

# Monitor system calls
strace -p $(pgrep -f mcp_server_enhanced.py)
```

### Log Analysis

#### Centralized Logging

```bash
# View application logs
sudo journalctl -u ai-news-trader -f

# Search for errors
sudo journalctl -u ai-news-trader --since "1 hour ago" | grep ERROR

# View Claude-flow logs
sudo journalctl -u claude-flow -f
```

#### Error Patterns

```bash
# Common error patterns
grep -E "(ERROR|CRITICAL|Exception)" /var/log/ai-news-trader/app.log | tail -20

# GPU errors
grep -i "cuda\|gpu\|out of memory" /var/log/ai-news-trader/app.log

# Database errors
grep -i "database\|connection\|timeout" /var/log/ai-news-trader/app.log
```

## Maintenance

### Regular Updates

```bash
#!/bin/bash
# update_system.sh

# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
source /opt/ai-news-trader/venv/bin/activate
pip install --upgrade -r requirements-mcp.txt

# Update neural forecasting
pip install --upgrade neuralforecast[gpu]

# Restart services
sudo systemctl restart ai-news-trader
sudo systemctl restart claude-flow

# Run health checks
./health_check.sh
```

### Database Maintenance

```sql
-- Vacuum and analyze
VACUUM ANALYZE;

-- Reindex
REINDEX DATABASE ai_news_trader_prod;

-- Update statistics
ANALYZE;

-- Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

### Model Retraining

```python
# retrain_models.py
import os
import schedule
import time
from datetime import datetime

def retrain_neural_models():
    """Retrain neural forecasting models with latest data"""
    os.system("python -m src.forecasting.cli models retrain --all")
    print(f"Model retraining completed at {datetime.now()}")

def backup_models():
    """Backup models before retraining"""
    os.system("python model_backup.py")
    print(f"Model backup completed at {datetime.now()}")

# Schedule retraining
schedule.every().sunday.at("02:00").do(backup_models)
schedule.every().sunday.at("02:30").do(retrain_neural_models)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour
```

---

This comprehensive deployment guide covers all aspects of deploying the AI News Trading Platform with Neural Forecasting in production environments. Follow the appropriate sections based on your deployment strategy and scale requirements.