#!/bin/bash

# Blue-Green Deployment Script for GPU Trading Platform
# Implements zero-downtime deployments with automatic rollback

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="ruvtrade"
BLUE_APP="${APP_NAME}-blue"
GREEN_APP="${APP_NAME}-green"
FLYCTL_PATH="/home/codespace/.fly/bin/flyctl"
DEPLOYMENT_DIR="/workspaces/ai-news-trader/fly_deployment"
HEALTH_CHECK_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_INTERVAL=10  # 10 seconds
ROLLBACK_TIMEOUT=180      # 3 minutes

# State file to track current deployment
STATE_FILE="/tmp/blue_green_state.json"

# Function to log messages
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

# Function to check if flyctl is available
check_flyctl() {
    if [ ! -f "$FLYCTL_PATH" ]; then
        log_error "flyctl not found at $FLYCTL_PATH"
        exit 1
    fi
    
    if ! $FLYCTL_PATH auth whoami > /dev/null 2>&1; then
        log_error "Not authenticated with Fly.io"
        echo "Please run: $FLYCTL_PATH auth login"
        exit 1
    fi
}

# Function to save deployment state
save_state() {
    local active_app="$1"
    local inactive_app="$2"
    local deployment_id="$3"
    local timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    
    cat > "$STATE_FILE" << EOF
{
  "active_app": "$active_app",
  "inactive_app": "$inactive_app", 
  "deployment_id": "$deployment_id",
  "timestamp": "$timestamp",
  "status": "deployed"
}
EOF
    
    log "Deployment state saved to $STATE_FILE"
}

# Function to load deployment state
load_state() {
    if [ -f "$STATE_FILE" ]; then
        cat "$STATE_FILE"
    else
        echo '{"active_app": null, "inactive_app": null, "deployment_id": null, "timestamp": null, "status": "new"}'
    fi
}

# Function to get current active app
get_active_app() {
    local state=$(load_state)
    echo "$state" | jq -r '.active_app // "none"'
}

# Function to get current inactive app
get_inactive_app() {
    local state=$(load_state)
    echo "$state" | jq -r '.inactive_app // "none"'
}

# Function to determine target app for deployment
determine_target_app() {
    local active_app=$(get_active_app)
    
    if [ "$active_app" = "$BLUE_APP" ]; then
        echo "$GREEN_APP"
    elif [ "$active_app" = "$GREEN_APP" ]; then
        echo "$BLUE_APP"
    else
        # No active app, use blue as default
        echo "$BLUE_APP"
    fi
}

# Function to check if app exists
app_exists() {
    local app_name="$1"
    $FLYCTL_PATH apps list | grep -q "^$app_name\s"
}

# Function to create app if it doesn't exist
ensure_app_exists() {
    local app_name="$1"
    
    if ! app_exists "$app_name"; then
        log "Creating app: $app_name"
        $FLYCTL_PATH apps create "$app_name"
        
        # Set regions
        $FLYCTL_PATH regions set ord fra nrt syd --app "$app_name"
        
        # Create volume
        local volume_name="${app_name}_data"
        if ! $FLYCTL_PATH volumes list --app "$app_name" | grep -q "$volume_name"; then
            log "Creating volume: $volume_name"
            $FLYCTL_PATH volumes create "$volume_name" --region ord --size 10 --app "$app_name"
        fi
        
        log_success "App $app_name created successfully"
    else
        log "App $app_name already exists"
    fi
}

# Function to deploy to target app
deploy_to_app() {
    local app_name="$1"
    local config_file="$2"
    
    log "Deploying to $app_name..."
    
    cd "$DEPLOYMENT_DIR"
    
    # Update fly.toml with app name
    local temp_config="/tmp/fly_${app_name}.toml"
    sed "s/app = \"$APP_NAME\"/app = \"$app_name\"/" "$config_file" > "$temp_config"
    
    # Also update volume mount name
    local volume_name="${app_name}_data"
    sed -i "s/source = \"ruvtrade_data\"/source = \"$volume_name\"/" "$temp_config"
    
    # Deploy
    $FLYCTL_PATH deploy --app "$app_name" --config "$temp_config"
    
    # Clean up temp config
    rm -f "$temp_config"
    
    log_success "Deployment to $app_name completed"
}

# Function to wait for app to be healthy
wait_for_health() {
    local app_name="$1"
    local timeout="$2"
    local start_time=$(date +%s)
    
    log "Waiting for $app_name to become healthy (timeout: ${timeout}s)..."
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            log_error "Health check timeout for $app_name after ${timeout}s"
            return 1
        fi
        
        # Check app status
        local status=$($FLYCTL_PATH status --app "$app_name" --json | jq -r '.status // "unknown"')
        
        if [ "$status" = "running" ]; then
            # Check health endpoint
            local app_url="https://${app_name}.fly.dev"
            
            if curl -f -s --max-time 10 "$app_url/health" > /dev/null 2>&1; then
                log_success "$app_name is healthy"
                return 0
            fi
        fi
        
        log "Health check failed for $app_name (status: $status), retrying in ${HEALTH_CHECK_INTERVAL}s..."
        sleep $HEALTH_CHECK_INTERVAL
    done
}

# Function to run deployment validation
validate_deployment() {
    local app_name="$1"
    local app_url="https://${app_name}.fly.dev"
    
    log "Validating deployment for $app_name..."
    
    # Test basic health check
    if ! curl -f -s --max-time 10 "$app_url/health" > /dev/null; then
        log_error "Basic health check failed"
        return 1
    fi
    
    # Test detailed health check
    local health_response=$(curl -f -s --max-time 30 "$app_url/health/detailed")
    local overall_status=$(echo "$health_response" | jq -r '.overall_status // "unknown"')
    
    if [ "$overall_status" != "healthy" ] && [ "$overall_status" != "degraded" ]; then
        log_error "Detailed health check failed: $overall_status"
        echo "$health_response" | jq '.'
        return 1
    fi
    
    # Test GPU status
    if ! curl -f -s --max-time 15 "$app_url/gpu-status" > /dev/null; then
        log_error "GPU status check failed"
        return 1
    fi
    
    # Test metrics endpoint
    if ! curl -f -s --max-time 10 "$app_url/metrics" > /dev/null; then
        log_warning "Metrics endpoint not responding (non-critical)"
    fi
    
    log_success "Deployment validation passed for $app_name"
    return 0
}

# Function to switch traffic to new app
switch_traffic() {
    local new_app="$1"
    local old_app="$2"
    
    log "Switching traffic from $old_app to $new_app..."
    
    # Create/update CNAME record to point to new app
    # This would typically be done through your DNS provider's API
    # For now, we'll use fly.io's hostname feature
    
    # Remove old hostname if it exists
    if [ "$old_app" != "none" ]; then
        $FLYCTL_PATH certs list --app "$old_app" | grep -q "$APP_NAME.fly.dev" && \
            $FLYCTL_PATH certs remove "$APP_NAME.fly.dev" --app "$old_app" 2>/dev/null || true
    fi
    
    # Add hostname to new app
    $FLYCTL_PATH certs create "$APP_NAME.fly.dev" --app "$new_app" 2>/dev/null || true
    
    log_success "Traffic switched to $new_app"
}

# Function to scale down old app
scale_down_old_app() {
    local app_name="$1"
    
    if [ "$app_name" = "none" ]; then
        return 0
    fi
    
    log "Scaling down old app: $app_name"
    
    # Scale to 0 instances
    $FLYCTL_PATH scale count 0 --app "$app_name"
    
    log_success "Scaled down $app_name"
}

# Function to rollback deployment
rollback() {
    local rollback_reason="$1"
    log_warning "Initiating rollback: $rollback_reason"
    
    local state=$(load_state)
    local active_app=$(echo "$state" | jq -r '.active_app // "none"')
    local inactive_app=$(echo "$state" | jq -r '.inactive_app // "none"')
    
    if [ "$inactive_app" = "none" ]; then
        log_error "No previous deployment found for rollback"
        return 1
    fi
    
    # Scale up the previous app
    log "Scaling up previous app: $inactive_app"
    $FLYCTL_PATH scale count 1 --app "$inactive_app"
    
    # Wait for it to be healthy
    if ! wait_for_health "$inactive_app" "$ROLLBACK_TIMEOUT"; then
        log_error "Rollback failed: $inactive_app did not become healthy"
        return 1
    fi
    
    # Switch traffic back
    switch_traffic "$inactive_app" "$active_app"
    
    # Scale down the failed app
    scale_down_old_app "$active_app"
    
    # Update state
    local deployment_id="rollback_$(date +%s)"
    save_state "$inactive_app" "$active_app" "$deployment_id"
    
    log_success "Rollback completed successfully"
    return 0
}

# Function to perform blue-green deployment
deploy() {
    local config_file="${1:-fly.toml}"
    
    log "Starting blue-green deployment..."
    
    # Determine target app
    local target_app=$(determine_target_app)
    local current_app=$(get_active_app)
    
    log "Target app: $target_app"
    log "Current active app: $current_app"
    
    # Ensure target app exists
    ensure_app_exists "$target_app"
    
    # Deploy to target app
    if ! deploy_to_app "$target_app" "$config_file"; then
        log_error "Deployment failed"
        return 1
    fi
    
    # Wait for target app to be healthy
    if ! wait_for_health "$target_app" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Target app failed health checks"
        rollback "Health check failure"
        return 1
    fi
    
    # Validate deployment
    if ! validate_deployment "$target_app"; then
        log_error "Deployment validation failed"
        rollback "Validation failure"
        return 1
    fi
    
    # Switch traffic to new app
    switch_traffic "$target_app" "$current_app"
    
    # Give it a moment to settle
    sleep 30
    
    # Final validation after traffic switch
    if ! validate_deployment "$target_app"; then
        log_error "Post-traffic-switch validation failed"
        rollback "Post-switch validation failure"
        return 1
    fi
    
    # Scale down old app
    scale_down_old_app "$current_app"
    
    # Save deployment state
    local deployment_id="deploy_$(date +%s)"
    save_state "$target_app" "$current_app" "$deployment_id"
    
    log_success "Blue-green deployment completed successfully!"
    log "Active app: $target_app"
    log "Previous app: $current_app (scaled down)"
    
    return 0
}

# Function to show current status
show_status() {
    log "Current Deployment Status:"
    echo
    
    local state=$(load_state)
    local active_app=$(echo "$state" | jq -r '.active_app // "none"')
    local inactive_app=$(echo "$state" | jq -r '.inactive_app // "none"')
    local deployment_id=$(echo "$state" | jq -r '.deployment_id // "none"')
    local timestamp=$(echo "$state" | jq -r '.timestamp // "none"')
    
    echo "Active App: $active_app"
    echo "Inactive App: $inactive_app"
    echo "Deployment ID: $deployment_id"
    echo "Last Deployment: $timestamp"
    echo
    
    # Show app statuses
    if [ "$active_app" != "none" ]; then
        echo "Active App Status:"
        $FLYCTL_PATH status --app "$active_app" || echo "  Failed to get status"
        echo
    fi
    
    if [ "$inactive_app" != "none" ]; then
        echo "Inactive App Status:"
        $FLYCTL_PATH status --app "$inactive_app" || echo "  Failed to get status"
        echo
    fi
}

# Function to clean up old deployments
cleanup() {
    log "Cleaning up old deployments..."
    
    local state=$(load_state)
    local active_app=$(echo "$state" | jq -r '.active_app // "none"')
    local inactive_app=$(echo "$state" | jq -r '.inactive_app // "none"')
    
    # Scale down inactive app
    if [ "$inactive_app" != "none" ]; then
        scale_down_old_app "$inactive_app"
    fi
    
    log_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo -e "${BLUE}Blue-Green Deployment Script${NC}"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  deploy [config]   Deploy using blue-green strategy"
    echo "  rollback [reason] Rollback to previous deployment"
    echo "  status            Show current deployment status"
    echo "  cleanup           Clean up old deployments"
    echo "  switch [app]      Manually switch traffic to app"
    echo
    echo "Options:"
    echo "  --timeout N       Health check timeout in seconds (default: 300)"
    echo "  --interval N      Health check interval in seconds (default: 10)"
    echo
    echo "Examples:"
    echo "  $0 deploy"
    echo "  $0 deploy fly-production.toml"
    echo "  $0 rollback 'Performance issue'"
    echo "  $0 status"
    echo "  $0 cleanup"
}

# Parse command line arguments
COMMAND=""
CONFIG_FILE="fly.toml"
ROLLBACK_REASON="Manual rollback"

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|rollback|status|cleanup|switch)
            COMMAND="$1"
            shift
            ;;
        --timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --interval)
            HEALTH_CHECK_INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [ -z "$COMMAND" ]; then
                echo -e "${RED}Unknown command: $1${NC}"
                show_help
                exit 1
            elif [ "$COMMAND" = "deploy" ] && [ -f "$1" ]; then
                CONFIG_FILE="$1"
            elif [ "$COMMAND" = "rollback" ]; then
                ROLLBACK_REASON="$1"
            elif [ "$COMMAND" = "switch" ]; then
                TARGET_APP="$1"
            fi
            shift
            ;;
    esac
done

# Check prerequisites
check_flyctl

# Create deployment directory if it doesn't exist
mkdir -p "$DEPLOYMENT_DIR"

# Execute command
case $COMMAND in
    deploy)
        if [ ! -f "$DEPLOYMENT_DIR/$CONFIG_FILE" ]; then
            log_error "Config file not found: $DEPLOYMENT_DIR/$CONFIG_FILE"
            exit 1
        fi
        deploy "$CONFIG_FILE"
        ;;
    rollback)
        rollback "$ROLLBACK_REASON"
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    switch)
        if [ -z "$TARGET_APP" ]; then
            log_error "Target app not specified for switch command"
            exit 1
        fi
        current_app=$(get_active_app)
        switch_traffic "$TARGET_APP" "$current_app"
        deployment_id="manual_switch_$(date +%s)"
        save_state "$TARGET_APP" "$current_app" "$deployment_id"
        ;;
    "")
        show_help
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

log_success "Operation completed successfully"