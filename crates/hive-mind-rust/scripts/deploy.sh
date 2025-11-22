#!/bin/bash

# Hive Mind Rust Backend Deployment Script
set -euo pipefail

# Configuration
ENVIRONMENT=${1:-production}
CONFIG_FILE="config/${ENVIRONMENT}.toml"
LOG_LEVEL=${LOG_LEVEL:-info}
DATA_DIR=${DATA_DIR:-./data}
LOG_DIR=${LOG_DIR:-./logs}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Validate environment
validate_environment() {
    log "Validating deployment environment: $ENVIRONMENT"
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null; then
        error "docker-compose is not installed or not in PATH"
        exit 1
    fi
    
    log "Environment validation passed"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories"
    
    mkdir -p "$DATA_DIR" "$LOG_DIR"
    chmod 755 "$DATA_DIR" "$LOG_DIR"
    
    log "Directories created successfully"
}

# Build the application
build_application() {
    log "Building Hive Mind application"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        cargo build
    else
        cargo build --release
    fi
    
    log "Application built successfully"
}

# Run configuration validation
validate_configuration() {
    log "Validating configuration file: $CONFIG_FILE"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        ./target/debug/hive-mind validate -c "$CONFIG_FILE"
    else
        ./target/release/hive-mind validate -c "$CONFIG_FILE"
    fi
    
    log "Configuration validation passed"
}

# Deploy with Docker Compose
deploy_docker() {
    log "Deploying with Docker Compose"
    
    # Set environment variables for docker-compose
    export HIVE_MIND_CONFIG="/app/$CONFIG_FILE"
    export RUST_LOG="$LOG_LEVEL"
    
    # Pull latest images
    docker-compose pull
    
    # Build and start services
    docker-compose up -d --build
    
    log "Docker deployment completed"
}

# Deploy standalone
deploy_standalone() {
    log "Deploying standalone instance"
    
    # Kill existing process if running
    pkill -f "hive-mind" || true
    sleep 2
    
    # Start the application
    if [[ "$ENVIRONMENT" == "development" ]]; then
        nohup ./target/debug/hive-mind start -c "$CONFIG_FILE" --daemon -p "$DATA_DIR/hive-mind.pid" > "$LOG_DIR/hive-mind.log" 2>&1 &
    else
        nohup ./target/release/hive-mind start -c "$CONFIG_FILE" --daemon -p "$DATA_DIR/hive-mind.pid" > "$LOG_DIR/hive-mind.log" 2>&1 &
    fi
    
    log "Standalone deployment completed"
}

# Health check
health_check() {
    log "Performing health check"
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8091/health &> /dev/null; then
            log "Health check passed"
            return 0
        fi
        
        warn "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
    return 1
}

# Main deployment function
main() {
    log "Starting Hive Mind deployment for environment: $ENVIRONMENT"
    
    validate_environment
    setup_directories
    build_application
    validate_configuration
    
    # Choose deployment method
    if [[ "${DOCKER_DEPLOY:-true}" == "true" ]]; then
        deploy_docker
    else
        deploy_standalone
    fi
    
    # Wait a moment for services to start
    sleep 10
    
    # Run health check
    if health_check; then
        log "Deployment completed successfully!"
        
        # Show status
        if [[ "${DOCKER_DEPLOY:-true}" == "true" ]]; then
            docker-compose ps
        else
            if [[ "$ENVIRONMENT" == "development" ]]; then
                ./target/debug/hive-mind status
            else
                ./target/release/hive-mind status
            fi
        fi
    else
        error "Deployment failed health check"
        exit 1
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [ENVIRONMENT]"
    echo ""
    echo "Arguments:"
    echo "  ENVIRONMENT    Deployment environment (production|development|minimal)"
    echo ""
    echo "Environment Variables:"
    echo "  LOG_LEVEL      Log level (debug|info|warn|error) [default: info]"
    echo "  DATA_DIR       Data directory [default: ./data]"
    echo "  LOG_DIR        Log directory [default: ./logs]"
    echo "  DOCKER_DEPLOY  Use Docker deployment [default: true]"
    echo ""
    echo "Examples:"
    echo "  $0 production                    # Deploy production with Docker"
    echo "  DOCKER_DEPLOY=false $0 development  # Deploy development standalone"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    production|development|minimal)
        main "$@"
        ;;
    "")
        main "production"
        ;;
    *)
        error "Invalid environment: $1"
        usage
        exit 1
        ;;
esac