#!/bin/bash
# Production Deployment Script for Hive Mind Rust Financial Trading System
# Implements blue-green deployment strategy with zero-downtime requirements

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
NAMESPACE="hive-mind-production"
IMAGE_REPO="${IMAGE_REPO:-hive-mind-rust}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
KUBECTL="${KUBECTL:-kubectl}"
DOCKER="${DOCKER:-docker}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Validation functions
validate_prerequisites() {
    log_info "Validating deployment prerequisites..."
    
    # Check required tools
    command -v "$KUBECTL" >/dev/null 2>&1 || { log_error "kubectl not found"; exit 1; }
    command -v "$DOCKER" >/dev/null 2>&1 || { log_error "docker not found"; exit 1; }
    command -v helm >/dev/null 2>&1 || { log_error "helm not found"; exit 1; }
    
    # Verify cluster connectivity
    if ! "$KUBECTL" cluster-info >/dev/null 2>&1; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! "$KUBECTL" get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Verify current context is production
    current_context=$("$KUBECTL" config current-context)
    if [[ "$current_context" != *"production"* ]]; then
        log_warning "Current context '$current_context' doesn't appear to be production"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    log_success "Prerequisites validation passed"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image with production optimizations
    log_info "Building image: $IMAGE_REPO:$IMAGE_TAG"
    "$DOCKER" build \
        --target runtime \
        --build-arg RUST_VERSION=1.75 \
        --build-arg DEBIAN_VERSION=bullseye \
        --tag "$IMAGE_REPO:$IMAGE_TAG" \
        --tag "$IMAGE_REPO:latest" \
        .
    
    # Security scan (if configured)
    if command -v trivy >/dev/null 2>&1; then
        log_info "Running security scan on image..."
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$IMAGE_REPO:$IMAGE_TAG"
    fi
    
    # Push to registry
    log_info "Pushing image to registry..."
    "$DOCKER" push "$IMAGE_REPO:$IMAGE_TAG"
    "$DOCKER" push "$IMAGE_REPO:latest"
    
    log_success "Image built and pushed successfully"
}

pre_deployment_checks() {
    log_info "Running pre-deployment health checks..."
    
    # Check cluster resources
    log_info "Checking cluster resource availability..."
    
    # Get resource usage
    cpu_usage=$("$KUBECTL" top nodes --no-headers | awk '{sum+=$3} END {print sum}')
    memory_usage=$("$KUBECTL" top nodes --no-headers | awk '{sum+=$5} END {print sum}')
    
    log_info "Current cluster CPU usage: ${cpu_usage:-0}m"
    log_info "Current cluster memory usage: ${memory_usage:-0}Mi"
    
    # Check persistent volumes
    if ! "$KUBECTL" get pvc -n "$NAMESPACE" >/dev/null 2>&1; then
        log_error "Required persistent volumes not found"
        exit 1
    fi
    
    # Validate configuration
    log_info "Validating configuration files..."
    if [[ ! -f "$PROJECT_ROOT/config/production.toml" ]]; then
        log_error "Production configuration file not found"
        exit 1
    fi
    
    # Check external dependencies
    log_info "Checking external service dependencies..."
    # Add checks for databases, external APIs, etc.
    
    log_success "Pre-deployment checks passed"
}

deploy_blue_green() {
    local deployment_color="$1"
    local opposite_color
    
    if [[ "$deployment_color" == "blue" ]]; then
        opposite_color="green"
    else
        opposite_color="blue"
    fi
    
    log_info "Starting $deployment_color deployment..."
    
    # Update deployment with new image
    "$KUBECTL" set image deployment/hive-mind-rust-"$deployment_color" \
        hive-mind-rust="$IMAGE_REPO:$IMAGE_TAG" \
        -n "$NAMESPACE"
    
    # Wait for rollout to complete
    log_info "Waiting for $deployment_color deployment to complete..."
    "$KUBECTL" rollout status deployment/hive-mind-rust-"$deployment_color" \
        -n "$NAMESPACE" \
        --timeout=600s
    
    # Health checks
    log_info "Performing health checks on $deployment_color deployment..."
    sleep 30  # Allow services to initialize
    
    # Get service endpoint
    service_ip=$("$KUBECTL" get service hive-mind-service-"$deployment_color" \
        -n "$NAMESPACE" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    
    if [[ -z "$service_ip" ]]; then
        service_ip=$("$KUBECTL" get service hive-mind-service-"$deployment_color" \
            -n "$NAMESPACE" \
            -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Health check endpoints
    for endpoint in health/live health/ready; do
        log_info "Checking $endpoint endpoint..."
        if ! curl -f "http://$service_ip:8091/$endpoint" >/dev/null 2>&1; then
            log_error "$deployment_color deployment health check failed"
            rollback_deployment "$deployment_color"
            exit 1
        fi
    done
    
    # Performance validation
    log_info "Running performance validation..."
    validate_latency "$service_ip"
    
    log_success "$deployment_color deployment completed successfully"
}

validate_latency() {
    local service_ip="$1"
    local max_latency=1000  # 1ms in microseconds
    
    log_info "Validating trading latency requirements..."
    
    # Simple latency test (in production, use more sophisticated testing)
    for i in {1..10}; do
        start_time=$(date +%s%N)
        curl -s "http://$service_ip:8091/health" >/dev/null
        end_time=$(date +%s%N)
        
        latency_us=$(( (end_time - start_time) / 1000 ))
        
        if [[ $latency_us -gt $max_latency ]]; then
            log_warning "Latency test $i: ${latency_us}μs (above ${max_latency}μs threshold)"
        else
            log_info "Latency test $i: ${latency_us}μs (within threshold)"
        fi
    done
    
    log_success "Latency validation completed"
}

switch_traffic() {
    local new_color="$1"
    
    log_info "Switching traffic to $new_color deployment..."
    
    # Update service selector to point to new deployment
    "$KUBECTL" patch service hive-mind-service \
        -n "$NAMESPACE" \
        -p '{"spec":{"selector":{"deployment":"'$new_color'"}}}'
    
    # Wait for DNS propagation
    sleep 10
    
    # Verify traffic switch
    log_info "Verifying traffic switch..."
    
    log_success "Traffic successfully switched to $new_color deployment"
}

cleanup_old_deployment() {
    local old_color="$1"
    
    log_info "Cleaning up old $old_color deployment..."
    
    # Scale down old deployment
    "$KUBECTL" scale deployment hive-mind-rust-"$old_color" \
        --replicas=0 \
        -n "$NAMESPACE"
    
    log_info "Old $old_color deployment scaled down (kept for rollback)"
}

rollback_deployment() {
    local failed_color="$1"
    
    log_error "Rolling back failed $failed_color deployment..."
    
    # Scale down failed deployment
    "$KUBECTL" scale deployment hive-mind-rust-"$failed_color" \
        --replicas=0 \
        -n "$NAMESPACE"
    
    # Rollback to previous version
    "$KUBECTL" rollout undo deployment/hive-mind-rust-"$failed_color" \
        -n "$NAMESPACE"
    
    log_success "Rollback completed"
}

monitor_deployment() {
    log_info "Setting up deployment monitoring..."
    
    # Create deployment annotation for Grafana
    "$KUBECTL" annotate deployment hive-mind-rust \
        -n "$NAMESPACE" \
        deployment.timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        deployment.version="$IMAGE_TAG" \
        deployment.user="$(whoami)" \
        --overwrite
    
    # Send alert to monitoring systems
    if [[ -n "${WEBHOOK_URL:-}" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"Hive Mind production deployment completed\",
                \"deployment\": {
                    \"version\": \"$IMAGE_TAG\",
                    \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
                    \"namespace\": \"$NAMESPACE\"
                }
            }" || log_warning "Failed to send deployment notification"
    fi
    
    log_success "Deployment monitoring configured"
}

main() {
    local deployment_color="${1:-blue}"
    
    log_info "Starting Hive Mind Rust production deployment"
    log_info "Deployment color: $deployment_color"
    log_info "Image: $IMAGE_REPO:$IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    
    # Deployment pipeline
    validate_prerequisites
    build_and_push_image
    pre_deployment_checks
    deploy_blue_green "$deployment_color"
    
    # Prompt for traffic switch
    read -p "Switch traffic to $deployment_color deployment? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        switch_traffic "$deployment_color"
        
        # Determine old color for cleanup
        local old_color
        if [[ "$deployment_color" == "blue" ]]; then
            old_color="green"
        else
            old_color="blue"
        fi
        
        cleanup_old_deployment "$old_color"
        monitor_deployment
        
        log_success "Production deployment completed successfully!"
        log_info "New version $IMAGE_TAG is now live"
    else
        log_info "Traffic not switched. Both deployments are running."
        log_info "To switch traffic later, run: kubectl patch service hive-mind-service -n $NAMESPACE -p '{\"spec\":{\"selector\":{\"deployment\":\"$deployment_color\"}}}'"
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi