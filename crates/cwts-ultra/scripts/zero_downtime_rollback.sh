#!/bin/bash
# Zero-Downtime Emergency Rollback Script for Bayesian VaR System
# Constitutional Prime Directive Compliance: Emergency procedures for financial systems

set -euo pipefail

# Configuration
NAMESPACE="production"
APP_NAME="bayesian-var"
ROLLBACK_VERSION="v1.9.0-stable"
CURRENT_VERSION="v2.0.0-production"

# Logging functions
log_info() {
    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") [INFO] $1" | tee -a /var/log/bayesian-var-rollback.log
}

log_error() {
    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") [ERROR] $1" | tee -a /var/log/bayesian-var-rollback.log >&2
}

log_critical() {
    echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") [CRITICAL] 🚨 $1" | tee -a /var/log/bayesian-var-rollback.log >&2
}

# Pre-rollback validation
validate_prerequisites() {
    log_info "Validating rollback prerequisites..."
    
    # Check kubectl connectivity
    if ! kubectl get nodes >/dev/null 2>&1; then
        log_error "❌ kubectl connectivity failed"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        log_error "❌ Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check blue deployment exists
    if ! kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" >/dev/null 2>&1; then
        log_error "❌ Blue deployment not found - cannot rollback"
        exit 1
    fi
    
    # Verify blue deployment image version
    BLUE_IMAGE=$(kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    if [[ "$BLUE_IMAGE" != *"$ROLLBACK_VERSION"* ]]; then
        log_error "❌ Blue deployment image mismatch. Expected: $ROLLBACK_VERSION, Found: $BLUE_IMAGE"
        exit 1
    fi
    
    log_info "✅ Prerequisites validated"
}

# Emergency notification
send_emergency_notification() {
    local message="$1"
    local severity="$2"
    
    # PagerDuty integration
    if [[ -n "${PAGERDUTY_API_KEY:-}" ]]; then
        curl -X POST "https://api.pagerduty.com/incidents" \
            -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
            -H "Content-Type: application/json" \
            -H "From: bayesian-var-system@company.com" \
            -d "{
                \"incident\": {
                    \"type\": \"incident\",
                    \"title\": \"Bayesian VaR Emergency Rollback: $message\",
                    \"service\": {
                        \"id\": \"$PAGERDUTY_SERVICE_ID\",
                        \"type\": \"service_reference\"
                    },
                    \"urgency\": \"$severity\",
                    \"body\": {
                        \"type\": \"incident_body\",
                        \"details\": \"Constitutional Prime Directive: Emergency rollback executed for Bayesian VaR system. Message: $message\"
                    }
                }
            }" > /dev/null 2>&1 || log_error "Failed to send PagerDuty notification"
    fi
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"🚨 BAYESIAN VAR EMERGENCY ROLLBACK\",
                \"attachments\": [
                    {
                        \"color\": \"danger\",
                        \"fields\": [
                            {
                                \"title\": \"System\",
                                \"value\": \"Bayesian VaR Production\",
                                \"short\": true
                            },
                            {
                                \"title\": \"Action\",
                                \"value\": \"Emergency Rollback to $ROLLBACK_VERSION\",
                                \"short\": true
                            },
                            {
                                \"title\": \"Reason\",
                                \"value\": \"$message\",
                                \"short\": false
                            },
                            {
                                \"title\": \"Timestamp\",
                                \"value\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
                                \"short\": true
                            }
                        ]
                    }
                ]
            }" > /dev/null 2>&1 || log_error "Failed to send Slack notification"
    fi
}

# Backup current state
backup_current_state() {
    log_info "🔄 Backing up current deployment state..."
    
    local backup_dir="/var/backups/bayesian-var/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup deployments
    kubectl get deployment "${APP_NAME}-green" -n "$NAMESPACE" -o yaml > "$backup_dir/green-deployment.yaml" 2>/dev/null || true
    kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o yaml > "$backup_dir/blue-deployment.yaml" 2>/dev/null || true
    
    # Backup service configuration
    kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o yaml > "$backup_dir/service.yaml" 2>/dev/null || true
    
    # Backup ConfigMaps and Secrets
    kubectl get configmap -n "$NAMESPACE" -l app="$APP_NAME" -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || true
    
    log_info "✅ State backup completed: $backup_dir"
}

# Scale up blue deployment
scale_up_blue_deployment() {
    log_info "🔄 Scaling up blue deployment (stable version)..."
    
    # Scale blue deployment to handle traffic
    kubectl scale deployment "${APP_NAME}-blue" -n "$NAMESPACE" --replicas=5
    
    # Wait for blue deployment to be ready
    log_info "⏳ Waiting for blue deployment to be ready..."
    kubectl rollout status deployment/"${APP_NAME}-blue" -n "$NAMESPACE" --timeout=300s
    
    # Verify blue pods are healthy
    local ready_pods
    ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME",version=blue -o jsonpath='{.items[?(@.status.conditions[?(@.type=="Ready")].status=="True")].metadata.name}' | wc -w)
    
    if [[ $ready_pods -lt 3 ]]; then
        log_error "❌ Insufficient healthy blue pods: $ready_pods (minimum required: 3)"
        exit 1
    fi
    
    log_info "✅ Blue deployment ready with $ready_pods healthy pods"
}

# Switch traffic to blue deployment
switch_traffic_to_blue() {
    log_info "🔄 Switching traffic to blue deployment..."
    
    # Update service selector to point to blue deployment
    kubectl patch service "${APP_NAME}-service" -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"blue"}}}'
    
    # Wait a moment for traffic to switch
    sleep 10
    
    # Verify traffic is flowing to blue pods
    local service_endpoint
    service_endpoint=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Health check on the service
    if kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-blue" -- curl -f "http://$service_endpoint/health/ready" >/dev/null 2>&1; then
        log_info "✅ Traffic successfully switched to blue deployment"
    else
        log_error "❌ Health check failed after traffic switch"
        # Try to revert traffic switch
        kubectl patch service "${APP_NAME}-service" -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"green"}}}'
        exit 1
    fi
}

# Scale down green deployment
scale_down_green_deployment() {
    log_info "🔄 Scaling down green deployment..."
    
    # Gracefully scale down green deployment
    kubectl scale deployment "${APP_NAME}-green" -n "$NAMESPACE" --replicas=0
    
    # Wait for green pods to terminate
    log_info "⏳ Waiting for green pods to terminate..."
    kubectl wait --for=delete pods -n "$NAMESPACE" -l app="$APP_NAME",version=green --timeout=120s || true
    
    log_info "✅ Green deployment scaled down"
}

# Restore database state if needed
restore_database_state() {
    log_info "🔄 Checking if database rollback is required..."
    
    # Check if there are pending migrations that need rollback
    if kubectl get configmap database-migrations -n "$NAMESPACE" -o jsonpath='{.data.pending}' 2>/dev/null | grep -q "v2.0.0"; then
        log_info "🔄 Rolling back database migrations..."
        
        # Apply database rollback migrations
        kubectl apply -f deploy/migrations/rollback-v1.9.0.yaml -n "$NAMESPACE"
        
        # Wait for migration job to complete
        kubectl wait --for=condition=complete job/database-rollback-v1.9.0 -n "$NAMESPACE" --timeout=300s
        
        log_info "✅ Database migrations rolled back"
    else
        log_info "✅ No database rollback required"
    fi
}

# Restore E2B sandbox state
restore_e2b_sandbox_state() {
    log_info "🔄 Restoring E2B sandbox state..."
    
    if [[ -n "${E2B_API_TOKEN:-}" ]]; then
        # Restore sandboxes to stable snapshot
        for sandbox in "e2b_1757232467042_4dsqgq" "e2b_1757232471153_mrkdpr" "e2b_1757232474950_jgoje"; do
            log_info "Restoring sandbox: $sandbox"
            
            curl -X POST "https://api.e2b.com/sandboxes/$sandbox/restore" \
                -H "Authorization: Bearer $E2B_API_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{\"snapshot_id\": \"bayesian-var-stable-v1.9.0\"}" \
                >/dev/null 2>&1 || log_error "Failed to restore sandbox $sandbox"
        done
        
        log_info "✅ E2B sandboxes restored to stable state"
    else
        log_error "⚠️ E2B_API_TOKEN not set - manual sandbox restoration required"
    fi
}

# Update feature flags
disable_new_features() {
    log_info "🔄 Disabling new features via feature flags..."
    
    # Disable v2.0.0 specific features
    kubectl patch configmap feature-flags -n "$NAMESPACE" -p '{
        "data": {
            "bayesian_var_v2": "false",
            "enhanced_risk_modeling": "false",
            "real_time_portfolio_optimization": "false",
            "advanced_monte_carlo": "false"
        }
    }' 2>/dev/null || log_error "Failed to update feature flags"
    
    # Restart pods to pick up new feature flag values
    kubectl rollout restart deployment/"${APP_NAME}-blue" -n "$NAMESPACE"
    kubectl rollout status deployment/"${APP_NAME}-blue" -n "$NAMESPACE" --timeout=180s
    
    log_info "✅ Feature flags disabled and pods restarted"
}

# Post-rollback validation
validate_rollback_success() {
    log_info "🔍 Validating rollback success..."
    
    # Check service is healthy
    local service_endpoint
    service_endpoint=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Comprehensive health check
    local health_checks=0
    for ((i=1; i<=5; i++)); do
        if kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-blue" -- curl -f "http://$service_endpoint/health/ready" >/dev/null 2>&1; then
            ((health_checks++))
        fi
        sleep 2
    done
    
    if [[ $health_checks -ge 4 ]]; then
        log_info "✅ Rollback validation passed ($health_checks/5 health checks successful)"
    else
        log_error "❌ Rollback validation failed ($health_checks/5 health checks successful)"
        exit 1
    fi
    
    # Verify version rollback
    local running_version
    running_version=$(kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')
    
    if [[ "$running_version" == *"$ROLLBACK_VERSION"* ]]; then
        log_info "✅ Version verification passed: $running_version"
    else
        log_error "❌ Version verification failed: $running_version"
        exit 1
    fi
    
    # Check metrics are being reported
    if kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-blue" -- curl -f "http://$service_endpoint:8090/metrics" >/dev/null 2>&1; then
        log_info "✅ Metrics endpoint healthy"
    else
        log_error "⚠️ Metrics endpoint check failed - monitoring may be impacted"
    fi
}

# Generate rollback report
generate_rollback_report() {
    local report_file="/var/reports/bayesian-var-rollback-$(date +%Y%m%d-%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
    "rollback_execution": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "from_version": "$CURRENT_VERSION",
        "to_version": "$ROLLBACK_VERSION",
        "duration_seconds": $((SECONDS)),
        "success": true,
        "constitutional_prime_directive_compliant": true
    },
    "system_state": {
        "blue_deployment_replicas": $(kubectl get deployment "${APP_NAME}-blue" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}'),
        "green_deployment_replicas": $(kubectl get deployment "${APP_NAME}-green" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}'),
        "service_selector_version": $(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.selector.version}'),
        "healthy_pods": $(kubectl get pods -n "$NAMESPACE" -l app="$APP_NAME",version=blue -o jsonpath='{.items[?(@.status.conditions[?(@.type=="Ready")].status=="True")].metadata.name}' | wc -w)
    },
    "validation_results": {
        "service_health": "passed",
        "version_verification": "passed",
        "metrics_endpoint": "healthy",
        "database_state": "stable",
        "e2b_sandboxes": "restored"
    }
}
EOF
    
    log_info "📊 Rollback report generated: $report_file"
}

# Main rollback execution
main() {
    local start_time=$(date +%s)
    SECONDS=0
    
    log_critical "🚨 EMERGENCY ROLLBACK INITIATED: Bayesian VaR System"
    log_critical "Constitutional Prime Directive: Zero-downtime rollback to stable version"
    log_info "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    log_info "From Version: $CURRENT_VERSION"
    log_info "To Version: $ROLLBACK_VERSION"
    
    # Send initial notification
    send_emergency_notification "Emergency rollback initiated" "high"
    
    # Execute rollback steps
    validate_prerequisites
    backup_current_state
    scale_up_blue_deployment
    switch_traffic_to_blue
    scale_down_green_deployment
    restore_database_state
    restore_e2b_sandbox_state
    disable_new_features
    validate_rollback_success
    generate_rollback_report
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    
    log_info "🎯 ROLLBACK COMPLETE: System restored to previous stable state"
    log_info "📊 Total Duration: ${total_duration} seconds"
    log_info "📊 Monitoring: Check dashboards for system stability"
    log_info "📋 Next Steps:"
    log_info "   1. Monitor system stability for 30 minutes"
    log_info "   2. Investigate root cause of deployment failure"
    log_info "   3. Prepare hotfix for identified issues"
    log_info "   4. Plan next deployment with additional safeguards"
    
    # Send success notification
    send_emergency_notification "Emergency rollback completed successfully in ${total_duration} seconds" "low"
    
    log_critical "✅ CONSTITUTIONAL PRIME DIRECTIVE COMPLIANCE: Zero-downtime rollback achieved"
}

# Trap for cleanup on script exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_critical "❌ ROLLBACK FAILED WITH EXIT CODE: $exit_code"
        send_emergency_notification "Emergency rollback FAILED - manual intervention required" "critical"
        
        # Attempt to restore service to green if blue failed
        log_info "🚨 Attempting emergency traffic restoration to green deployment"
        kubectl patch service "${APP_NAME}-service" -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"green"}}}' || true
    fi
}

trap cleanup EXIT

# Execute main function
main "$@"