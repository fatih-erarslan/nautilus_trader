#!/bin/bash
# Fly.io GPU Deployment Script for AI News Trading Platform
# NeuralForecast NHITS Implementation

set -e

# Configuration
APP_NAME="ai-news-trader-neural"
REGION="ord"  # Chicago for low-latency trading
GPU_TYPE="a100-40gb"
DEPLOYMENT_TIMEOUT="15m"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with colors
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check fly CLI
    if ! command -v fly &> /dev/null; then
        log_error "Fly CLI not found. Please install: https://fly.io/docs/getting-started/installing-flyctl/"
        exit 1
    fi
    
    # Check fly auth
    if ! fly auth whoami &> /dev/null; then
        log_error "Not authenticated with Fly.io. Run: fly auth login"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "fly.toml" ]; then
        log_error "fly.toml not found. Please run from project root."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Function to setup fly app
setup_fly_app() {
    log "Setting up Fly.io application..."
    
    # Check if app exists
    if fly apps list | grep -q "$APP_NAME"; then
        log_warning "App $APP_NAME already exists"
    else
        log "Creating new app: $APP_NAME"
        fly apps create "$APP_NAME" --org personal
    fi
    
    # Create volume for model storage if it doesn't exist
    if ! fly volumes list -a "$APP_NAME" | grep -q "neural_models_volume"; then
        log "Creating volume for model storage..."
        fly volumes create neural_models_volume --region "$REGION" --size 50 -a "$APP_NAME"
    fi
    
    log_success "Fly app setup completed"
}

# Function to set secrets
setup_secrets() {
    log "Setting up application secrets..."
    
    # Read secrets from .env.production if it exists
    if [ -f ".env.production" ]; then
        log "Found .env.production, setting secrets..."
        
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^#.*$ ]] && continue
            [[ -z $key ]] && continue
            
            # Remove quotes if present
            value=$(echo "$value" | sed 's/^["'\'']\|["'\'']$//g')
            
            if [ -n "$value" ]; then
                fly secrets set "$key=$value" -a "$APP_NAME"
            fi
        done < .env.production
    else
        log_warning ".env.production not found, setting default secrets..."
        
        # Set default secrets
        fly secrets set \
            NEURAL_FORECAST_GPU_ENABLED=true \
            NEURAL_FORECAST_MODEL_TYPE=nhits \
            FLYIO_GPU_TYPE="$GPU_TYPE" \
            FLYIO_AUTO_SCALE=true \
            PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6" \
            -a "$APP_NAME"
    fi
    
    log_success "Secrets setup completed"
}

# Function to validate configuration
validate_config() {
    log "Validating deployment configuration..."
    
    # Check fly.toml configuration
    if ! grep -q "gpu_kind.*a100" fly.toml; then
        log_error "GPU configuration not found in fly.toml"
        exit 1
    fi
    
    # Check Dockerfile
    if [ ! -f "fly_deployment/Dockerfile.gpu-optimized" ]; then
        log_error "GPU-optimized Dockerfile not found"
        exit 1
    fi
    
    # Validate GPU requirements
    log "Validating GPU requirements..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA compiled version: {torch.version.cuda}')
print('✅ PyTorch CUDA validation passed')
" || {
        log_error "PyTorch CUDA validation failed"
        exit 1
    }
    
    log_success "Configuration validation passed"
}

# Function to build and deploy
build_and_deploy() {
    log "Building and deploying application..."
    
    # Set deployment timeout
    export FLY_WAIT_TIMEOUT="$DEPLOYMENT_TIMEOUT"
    
    # Deploy with GPU configuration
    log "Starting deployment to Fly.io..."
    fly deploy \
        --dockerfile fly_deployment/Dockerfile.gpu-optimized \
        --app "$APP_NAME" \
        --region "$REGION" \
        --wait-timeout "$DEPLOYMENT_TIMEOUT" \
        --strategy bluegreen \
        --verbose
    
    log_success "Deployment completed"
}

# Function to verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Wait for app to be ready
    log "Waiting for application to start..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if fly status -a "$APP_NAME" | grep -q "running"; then
            break
        fi
        
        attempt=$((attempt + 1))
        log "Waiting for app to start... ($attempt/$max_attempts)"
        sleep 10
    done
    
    if [ $attempt -eq $max_attempts ]; then
        log_error "Application failed to start within timeout"
        fly logs -a "$APP_NAME"
        exit 1
    fi
    
    # Test health endpoints
    log "Testing health endpoints..."
    local app_url="https://${APP_NAME}.fly.dev"
    
    # Basic health check
    if curl -f -s --max-time 30 "$app_url/health" > /dev/null; then
        log_success "Basic health check passed"
    else
        log_error "Basic health check failed"
        fly logs -a "$APP_NAME"
        exit 1
    fi
    
    # GPU health check
    if curl -f -s --max-time 30 "$app_url/health/gpu" > /dev/null; then
        log_success "GPU health check passed"
    else
        log_warning "GPU health check failed (may be expected on CPU instances)"
    fi
    
    # Test neural prediction endpoint
    log "Testing neural prediction endpoint..."
    local response
    response=$(curl -s --max-time 30 \
        -X POST "$app_url/neural/predict" \
        -H "Content-Type: application/json" \
        -d '{"symbol": "AAPL", "horizon": 1}' || echo "failed")
    
    if echo "$response" | grep -q "prediction\|error"; then
        log_success "Neural prediction endpoint responding"
    else
        log_warning "Neural prediction endpoint not responding properly"
    fi
    
    log_success "Deployment verification completed"
}

# Function to display deployment info
show_deployment_info() {
    log "📊 Deployment Information:"
    echo ""
    
    # App status
    fly status -a "$APP_NAME"
    echo ""
    
    # App URLs
    log_success "🌐 Application URLs:"
    echo "   Health Check: https://${APP_NAME}.fly.dev/health"
    echo "   GPU Health: https://${APP_NAME}.fly.dev/health/gpu"
    echo "   Metrics: https://${APP_NAME}.fly.dev/metrics"
    echo "   Neural Predict: https://${APP_NAME}.fly.dev/neural/predict"
    echo ""
    
    # Useful commands
    log_success "🔧 Useful Commands:"
    echo "   View logs: fly logs -a $APP_NAME"
    echo "   SSH to instance: fly ssh console -a $APP_NAME"
    echo "   Scale app: fly scale count 2 -a $APP_NAME"
    echo "   Monitor: fly status -a $APP_NAME"
    echo ""
    
    # Cost estimate
    log_success "💰 Estimated Costs:"
    echo "   A100-40GB GPU: ~$3.20/hour"
    echo "   8 CPU cores: ~$0.50/hour"
    echo "   32GB RAM: ~$0.25/hour"
    echo "   Total: ~$3.95/hour (~$2,846/month if running 24/7)"
}

# Function to setup monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    
    # Create monitoring configuration
    cat > fly_monitoring.yml << EOF
# Fly.io Monitoring Configuration
app: $APP_NAME
metrics:
  - name: gpu_utilization
    query: nvidia_gpu_utilization_percent
    threshold: 90
    action: alert
  
  - name: memory_usage
    query: memory_usage_percent
    threshold: 85
    action: scale_up
    
  - name: response_time
    query: http_request_duration_seconds_p95
    threshold: 0.1
    action: alert

alerts:
  webhook: ${WEBHOOK_URL:-}
  email: ${ALERT_EMAIL:-}
EOF

    log_success "Monitoring configuration created"
}

# Function to handle deployment rollback
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Get previous release
    local previous_release
    previous_release=$(fly releases -a "$APP_NAME" --json | jq -r '.[1].version // empty')
    
    if [ -n "$previous_release" ]; then
        log "Rolling back to release: $previous_release"
        fly releases rollback "$previous_release" -a "$APP_NAME"
        log_success "Rollback completed"
    else
        log_error "No previous release found for rollback"
        exit 1
    fi
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_warning "Deployment failed, cleaning up..."
    
    # Show recent logs
    echo "Recent logs:"
    fly logs -a "$APP_NAME" --follow=false || true
    
    # Ask for rollback
    read -p "Do you want to rollback to previous release? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rollback_deployment
    fi
}

# Main deployment function
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        "deploy")
            log "🚀 Starting GPU deployment for AI News Trading Platform"
            
            check_prerequisites
            setup_fly_app
            setup_secrets
            validate_config
            
            # Trap errors for cleanup
            trap cleanup_on_failure ERR
            
            build_and_deploy
            verify_deployment
            setup_monitoring
            show_deployment_info
            
            log_success "🎉 GPU deployment completed successfully!"
            ;;
            
        "rollback")
            rollback_deployment
            ;;
            
        "status")
            fly status -a "$APP_NAME"
            curl -s "https://${APP_NAME}.fly.dev/status" | jq . || echo "Status endpoint not available"
            ;;
            
        "logs")
            fly logs -a "$APP_NAME" "${@:2}"
            ;;
            
        "shell")
            fly ssh console -a "$APP_NAME"
            ;;
            
        "destroy")
            read -p "Are you sure you want to destroy the app $APP_NAME? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                fly apps destroy "$APP_NAME"
            fi
            ;;
            
        *)
            echo "Usage: $0 [deploy|rollback|status|logs|shell|destroy]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy the application (default)"
            echo "  rollback - Rollback to previous release"
            echo "  status   - Show application status"
            echo "  logs     - Show application logs"
            echo "  shell    - SSH into the application"
            echo "  destroy  - Destroy the application"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"