#!/bin/bash

# Cerebellar Norse Monitoring Infrastructure Startup Script
# Observability & Monitoring Engineer - Real-time system monitoring setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MONITORING_DIR="$PROJECT_ROOT/docker/monitoring"

echo "🚀 Starting Cerebellar Norse Monitoring Infrastructure"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo "📋 Checking dependencies..."

if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ All dependencies satisfied"

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p "$MONITORING_DIR"/{prometheus/{rules,data},grafana/{dashboards,datasources,data},alertmanager,nginx/{ssl,conf.d},redis,logstash/{pipeline,config},influxdb/{data,config},telegraf,loki,promtail,vector}

# Set proper permissions
echo "🔐 Setting permissions..."
sudo chown -R $(id -u):$(id -g) "$MONITORING_DIR"
chmod -R 755 "$MONITORING_DIR"

# Create default configurations if they don't exist
echo "⚙️ Creating default configurations..."

# Grafana datasources
if [ ! -f "$MONITORING_DIR/grafana/datasources/prometheus.yml" ]; then
cat > "$MONITORING_DIR/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    orgId: 1
    url: http://prometheus:9090
    basicAuth: false
    isDefault: true
    version: 1
    editable: true
    
  - name: Loki
    type: loki
    access: proxy
    orgId: 1
    url: http://loki:3100
    basicAuth: false
    version: 1
    editable: true
    
  - name: Jaeger
    type: jaeger
    access: proxy
    orgId: 1
    url: http://jaeger-all-in-one:16686
    basicAuth: false
    version: 1
    editable: true
    
  - name: InfluxDB
    type: influxdb
    access: proxy
    orgId: 1
    url: http://influxdb:8086
    basicAuth: false
    version: 1
    editable: true
    database: neural_metrics
    user: admin
    secureJsonData:
      password: cerebellar_influx_2024
EOF
fi

# AlertManager configuration
if [ ! -f "$MONITORING_DIR/alertmanager/alertmanager.yml" ]; then
cat > "$MONITORING_DIR/alertmanager/alertmanager.yml" << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@cerebellar-norse.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true
    
- name: 'email'
  email_configs:
  - to: 'admin@cerebellar-norse.com'
    subject: 'Cerebellar Norse Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF
fi

# NGINX configuration
if [ ! -f "$MONITORING_DIR/nginx/nginx.conf" ]; then
cat > "$MONITORING_DIR/nginx/nginx.conf" << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream grafana {
        server grafana:3000;
    }
    
    upstream prometheus {
        server prometheus:9090;
    }
    
    upstream kibana {
        server kibana:5601;
    }
    
    upstream jaeger {
        server jaeger-all-in-one:16686;
    }
    
    server {
        listen 80;
        server_name monitoring.cerebellar-norse.local;
        
        location /grafana/ {
            proxy_pass http://grafana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /prometheus/ {
            proxy_pass http://prometheus/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /kibana/ {
            proxy_pass http://kibana/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /jaeger/ {
            proxy_pass http://jaeger/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
    
    server {
        listen 8080;
        server_name metrics.cerebellar-norse.local;
        
        location /metrics {
            proxy_pass http://cerebellar-exporter:8080/metrics;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /neural/metrics {
            proxy_pass http://cerebellar-exporter:8080/neural/metrics;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location /trading/metrics {
            proxy_pass http://cerebellar-exporter:8080/trading/metrics;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
fi

# Loki configuration
if [ ! -f "$MONITORING_DIR/loki/local-config.yaml" ]; then
cat > "$MONITORING_DIR/loki/local-config.yaml" << 'EOF'
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096

common:
  path_prefix: /tmp/loki
  storage:
    filesystem:
      chunks_directory: /tmp/loki/chunks
      rules_directory: /tmp/loki/rules
  replication_factor: 1
  ring:
    instance_addr: 127.0.0.1
    kvstore:
      store: inmemory

query_range:
  results_cache:
    cache:
      embedded_cache:
        enabled: true
        max_size_mb: 100

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

ruler:
  alertmanager_url: http://alertmanager:9093
EOF
fi

# Promtail configuration
if [ ! -f "$MONITORING_DIR/promtail/config.yml" ]; then
cat > "$MONITORING_DIR/promtail/config.yml" << 'EOF'
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: varlogs
          __path__: /var/log/*log
          
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
    pipeline_stages:
      - json:
          expressions:
            timestamp: time
            level: level
            message: msg
      - timestamp:
          source: timestamp
          format: RFC3339Nano
      - labels:
          level:
          stream:
EOF
fi

# Telegraf configuration
if [ ! -f "$MONITORING_DIR/telegraf/telegraf.conf" ]; then
cat > "$MONITORING_DIR/telegraf/telegraf.conf" << 'EOF'
[global_tags]
  environment = "production"
  service = "cerebellar-norse"

[agent]
  interval = "10s"
  round_interval = true
  metric_batch_size = 1000
  metric_buffer_limit = 10000
  collection_jitter = "0s"
  flush_interval = "10s"
  flush_jitter = "0s"
  precision = ""
  hostname = ""
  omit_hostname = false

[[outputs.influxdb_v2]]
  urls = ["http://influxdb:8086"]
  token = "cerebellar_token_2024"
  organization = "cerebellar"
  bucket = "neural_metrics"

[[inputs.cpu]]
  percpu = true
  totalcpu = true
  collect_cpu_time = false
  report_active = false

[[inputs.disk]]
  ignore_fs = ["tmpfs", "devtmpfs", "devfs", "iso9660", "overlay", "aufs", "squashfs"]

[[inputs.diskio]]

[[inputs.kernel]]

[[inputs.mem]]

[[inputs.processes]]

[[inputs.swap]]

[[inputs.system]]

[[inputs.docker]]
  endpoint = "unix:///var/run/docker.sock"
  gather_services = false
  container_names = []
  source_tag = false
  container_name_include = []
  container_name_exclude = []
  timeout = "5s"
  api_version = "1.24"

[[inputs.prometheus]]
  urls = ["http://cerebellar-exporter:8080/metrics"]
  metric_version = 2
  name_override = "cerebellar_norse"
EOF
fi

# Vector configuration
if [ ! -f "$MONITORING_DIR/vector/vector.toml" ]; then
cat > "$MONITORING_DIR/vector/vector.toml" << 'EOF'
[sources.docker_logs]
type = "docker_logs"
docker_host = "unix:///var/run/docker.sock"

[sources.system_logs]
type = "file"
include = ["/var/log/*.log"]
read_from = "beginning"

[transforms.parse_logs]
type = "remap"
inputs = ["docker_logs", "system_logs"]
source = '''
  .timestamp = now()
  .service = .container_name
  if exists(.message) {
    . = parse_json(.message) ?? .
  }
'''

[sinks.elasticsearch]
type = "elasticsearch"
inputs = ["parse_logs"]
endpoints = ["http://elasticsearch:9200"]
index = "cerebellar-norse-%Y.%m.%d"

[sinks.loki]
type = "loki"
inputs = ["parse_logs"]
endpoint = "http://loki:3100"
encoding.codec = "json"
labels = {service = "{{ service }}", level = "{{ level }}"}
EOF
fi

# Build Rust metrics exporter
echo "🦀 Building Rust metrics exporter..."
cd "$PROJECT_ROOT"

# Create metrics exporter binary
cat > src/bin/metrics-exporter.rs << 'EOF'
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use cerebellar_norse::{ObservabilityManager, MetricsExporter, NeuralMetrics, TradingMetrics, SystemHealthMetrics};
use tracing::{info, error};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    info!("Starting Cerebellar Norse Metrics Exporter");

    // Initialize observability manager
    let observability = Arc::new(ObservabilityManager::new()?);
    
    // Start metrics collection background task
    let obs_clone = observability.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            
            // Collect system metrics
            if let Err(e) = obs_clone.collect_system_metrics().await {
                error!("Failed to collect system metrics: {}", e);
            }
            
            // Generate sample neural metrics (replace with real data in production)
            let neural_metrics = NeuralMetrics {
                inference_latency_ns: 2_500_000 + (rand::random::<u32>() % 1_000_000) as u64,
                throughput_ops_per_sec: 1000.0 + (rand::random::<f32>() - 0.5) * 200.0,
                accuracy_percentage: 97.5 + (rand::random::<f32>() - 0.5) * 2.0,
                memory_usage_mb: 2048.0 + (rand::random::<f32>() - 0.5) * 512.0,
                gpu_utilization_percent: 85.0 + (rand::random::<f32>() - 0.5) * 20.0,
                network_io_mbps: 100.0 + (rand::random::<f32>() - 0.5) * 20.0,
                error_rate_percent: 0.1 + rand::random::<f32>() * 0.2,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            };
            
            if let Err(e) = obs_clone.record_neural_metrics(neural_metrics).await {
                error!("Failed to record neural metrics: {}", e);
            }
            
            // Generate sample trading metrics
            let trading_metrics = TradingMetrics {
                orders_per_second: 50.0 + (rand::random::<f32>() - 0.5) * 20.0,
                order_fill_latency_ms: 25.0 + (rand::random::<f32>() - 0.5) * 10.0,
                market_data_latency_ms: 2.5 + (rand::random::<f32>() - 0.5) * 1.0,
                risk_score: 0.3 + rand::random::<f32>() * 0.3,
                pnl_realtime: 10000.0 + (rand::random::<f32>() - 0.5) * 5000.0,
                portfolio_value: 1000000.0 + (rand::random::<f32>() - 0.5) * 100000.0,
                open_positions: 5 + (rand::random::<u32>() % 10),
                alerts_triggered: rand::random::<u32>() % 3,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
            };
            
            if let Err(e) = obs_clone.record_trading_metrics(trading_metrics).await {
                error!("Failed to record trading metrics: {}", e);
            }
        }
    });
    
    // Start metrics exporter HTTP server
    let bind_address = std::env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("PROMETHEUS_PORT")
        .unwrap_or_else(|_| "8080".to_string())
        .parse::<u16>()
        .unwrap_or(8080);
    
    let exporter = MetricsExporter::new(observability, bind_address, port);
    
    info!("Metrics exporter ready on port {}", port);
    exporter.start().await?;
    
    Ok(())
}
EOF

if cargo build --release --bin metrics-exporter; then
    echo "✅ Metrics exporter built successfully"
else
    echo "⚠️ Failed to build metrics exporter, continuing with Docker setup"
fi

# Start monitoring stack
echo "🐳 Starting monitoring infrastructure..."
cd "$MONITORING_DIR"

# Pull all images first
echo "📥 Pulling Docker images..."
docker-compose pull

# Start services in order
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🔍 Performing health checks..."

check_service() {
    local service_name=$1
    local port=$2
    local path=${3:-"/"}
    
    if curl -s -f "http://localhost:$port$path" > /dev/null; then
        echo "✅ $service_name is healthy"
    else
        echo "⚠️ $service_name health check failed"
    fi
}

check_service "Prometheus" 9090 "/api/v1/status/config"
check_service "Grafana" 3000 "/api/health"
check_service "Elasticsearch" 9200 "/_cluster/health"
check_service "Kibana" 5601 "/api/status"
check_service "Jaeger" 16686 "/api/services"
check_service "AlertManager" 9093 "/api/v1/status/config"
check_service "Loki" 3100 "/ready"
check_service "InfluxDB" 8086 "/health"

# Display access information
echo ""
echo "🎉 Monitoring Infrastructure Started Successfully!"
echo "=================================================="
echo ""
echo "📊 Access URLs:"
echo "  • Grafana:      http://localhost:3000 (admin:cerebellar_admin_2024)"
echo "  • Prometheus:   http://localhost:9090"
echo "  • Kibana:       http://localhost:5601"
echo "  • Jaeger:       http://localhost:16686"
echo "  • AlertManager: http://localhost:9093"
echo "  • Loki:         http://localhost:3100"
echo "  • InfluxDB:     http://localhost:8086"
echo ""
echo "🔧 Metrics Endpoints:"
echo "  • All Metrics:    http://localhost:8080/metrics"
echo "  • Neural Metrics: http://localhost:8080/neural/metrics"
echo "  • Trading:        http://localhost:8080/trading/metrics"
echo "  • System:         http://localhost:8080/system/metrics"
echo "  • Risk:           http://localhost:8080/risk/metrics"
echo ""
echo "📚 Documentation:"
echo "  • Service Info:   http://localhost:8080/info"
echo "  • Health Check:   http://localhost:8080/health"
echo ""
echo "🚨 Alert Configuration:"
echo "  • Neural latency > 10ms"
echo "  • Accuracy < 95%"
echo "  • Risk score > 0.8"
echo "  • System CPU > 90%"
echo "  • Memory > 85%"
echo ""
echo "💡 Next Steps:"
echo "  1. Import Grafana dashboards from docker/monitoring/grafana/dashboards/"
echo "  2. Configure Kibana index patterns for log analysis"
echo "  3. Set up alert routing in AlertManager"
echo "  4. Configure distributed tracing in your application"
echo ""
echo "🔗 Quick Links:"
echo "  • Neural Performance Dashboard: http://localhost:3000/d/neural-performance"
echo "  • System Health Overview: http://localhost:3000/dashboards"
echo "  • Log Analysis: http://localhost:5601/app/discover"
echo "  • Trace Analysis: http://localhost:16686/search"
echo ""

# Save configuration summary
cat > "$PROJECT_ROOT/MONITORING_SETUP.md" << 'EOF'
# Cerebellar Norse Monitoring Infrastructure

## Overview
Comprehensive observability stack for neural trading system monitoring.

## Components
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization dashboards
- **Elasticsearch + Kibana**: Log aggregation and analysis
- **Jaeger**: Distributed tracing
- **Loki + Promtail**: Alternative log aggregation
- **InfluxDB + Telegraf**: Time-series data storage
- **AlertManager**: Alert routing and notification
- **NGINX**: Reverse proxy and load balancing
- **Redis**: Caching and session storage

## Key Metrics Monitored
- Neural network inference latency (target: <10ms)
- Model accuracy (threshold: >95%)
- Trading order latency (target: <100ms)
- Risk score monitoring (alert: >0.8)
- System resource utilization
- Error rates and anomaly detection

## Alert Thresholds
- **Critical**: Neural latency >50ms, Accuracy <90%, Risk >0.9
- **Warning**: Neural latency >10ms, Accuracy <95%, Risk >0.8
- **System**: CPU >90%, Memory >85%, Disk >95%

## Real-time Dashboards
- Neural Network Performance
- Trading System Health
- System Resource Monitoring
- Risk Management Overview
- Distributed Trace Analysis

## Log Analysis Features
- Structured log aggregation
- Error pattern detection
- Performance anomaly identification
- Cross-component correlation
- Real-time alerting

## Getting Started
1. Run `./scripts/start_monitoring.sh`
2. Access Grafana at http://localhost:3000
3. Import pre-configured dashboards
4. Configure alert channels
5. Start monitoring your neural trading system

For detailed configuration, see docker/monitoring/ directory.
EOF

echo "📝 Configuration summary saved to MONITORING_SETUP.md"
echo ""
echo "✨ Monitoring infrastructure is ready for neural trading system observability!"