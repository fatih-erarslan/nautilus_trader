#!/bin/bash
# E2B Sandbox Production Monitoring Script
# Constitutional Prime Directive Compliance for E2B Integration

set -euo pipefail

# Configuration
E2B_API_TOKEN="${E2B_API_TOKEN:-}"
E2B_API_BASE_URL="https://api.e2b.com"
NAMESPACE="production"
APP_NAME="bayesian-var"
MONITORING_INTERVAL="30"  # seconds
LOG_FILE="/var/log/e2b-production-monitor.log"

# E2B Sandbox IDs for Bayesian VaR system
BAYESIAN_TRAINING_SANDBOX="e2b_1757232467042_4dsqgq"
MONTE_CARLO_SANDBOX="e2b_1757232471153_mrkdpr"
REALTIME_SANDBOX="e2b_1757232474950_jgoje"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_critical() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${RED}[CRITICAL]${NC} 🚨 $1" | tee -a "$LOG_FILE"
}

# Health tracking
declare -A SANDBOX_HEALTH
declare -A SANDBOX_METRICS
declare -A ALERT_COOLDOWN

# Initialize health tracking
init_monitoring() {
    log_info "🚀 Initializing E2B production monitoring"
    
    # Verify E2B API token
    if [[ -z "$E2B_API_TOKEN" ]]; then
        log_critical "E2B_API_TOKEN not set - cannot monitor sandboxes"
        exit 1
    fi
    
    # Initialize sandbox health status
    SANDBOX_HEALTH["$BAYESIAN_TRAINING_SANDBOX"]="unknown"
    SANDBOX_HEALTH["$MONTE_CARLO_SANDBOX"]="unknown"
    SANDBOX_HEALTH["$REALTIME_SANDBOX"]="unknown"
    
    # Initialize alert cooldown (prevent spam)
    ALERT_COOLDOWN["$BAYESIAN_TRAINING_SANDBOX"]=0
    ALERT_COOLDOWN["$MONTE_CARLO_SANDBOX"]=0
    ALERT_COOLDOWN["$REALTIME_SANDBOX"]=0
    
    log_success "✅ E2B monitoring initialized"
}

# Check individual sandbox health
check_sandbox_health() {
    local sandbox_id="$1"
    local sandbox_name="$2"
    
    log_info "🔍 Checking health for $sandbox_name ($sandbox_id)"
    
    # Get sandbox status from E2B API
    local response
    response=$(curl -s -w "%{http_code}" \
        -H "Authorization: Bearer $E2B_API_TOKEN" \
        -H "Content-Type: application/json" \
        "$E2B_API_BASE_URL/sandboxes/$sandbox_id" 2>/dev/null || echo "000")
    
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        # Parse sandbox status
        local status
        status=$(echo "$body" | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
        
        local cpu_usage
        cpu_usage=$(echo "$body" | jq -r '.resources.cpu_usage_percent // 0' 2>/dev/null || echo "0")
        
        local memory_usage
        memory_usage=$(echo "$body" | jq -r '.resources.memory_usage_mb // 0' 2>/dev/null || echo "0")
        
        local last_activity
        last_activity=$(echo "$body" | jq -r '.last_activity // null' 2>/dev/null || echo "null")
        
        # Store metrics
        SANDBOX_METRICS["${sandbox_id}_status"]="$status"
        SANDBOX_METRICS["${sandbox_id}_cpu"]="$cpu_usage"
        SANDBOX_METRICS["${sandbox_id}_memory"]="$memory_usage"
        SANDBOX_METRICS["${sandbox_id}_last_activity"]="$last_activity"
        
        # Determine health status
        if [[ "$status" == "running" ]]; then
            # Check resource usage
            if (( $(echo "$cpu_usage > 95" | bc -l) )) || (( $(echo "$memory_usage > 1800" | bc -l) )); then
                SANDBOX_HEALTH["$sandbox_id"]="degraded"
                log_warning "⚠️ $sandbox_name resources high: CPU ${cpu_usage}%, Memory ${memory_usage}MB"
            else
                SANDBOX_HEALTH["$sandbox_id"]="healthy"
                log_success "✅ $sandbox_name healthy"
            fi
        elif [[ "$status" == "starting" ]]; then
            SANDBOX_HEALTH["$sandbox_id"]="starting"
            log_info "🔄 $sandbox_name starting up"
        else
            SANDBOX_HEALTH["$sandbox_id"]="unhealthy"
            log_error "❌ $sandbox_name status: $status"
        fi
        
    elif [[ "$http_code" == "404" ]]; then
        SANDBOX_HEALTH["$sandbox_id"]="missing"
        log_error "❌ $sandbox_name not found (404)"
        
    else
        SANDBOX_HEALTH["$sandbox_id"]="unreachable"
        log_error "❌ $sandbox_name unreachable (HTTP $http_code)"
    fi
}

# Check E2B API health
check_e2b_api_health() {
    log_info "🌐 Checking E2B API health"
    
    local response
    response=$(curl -s -w "%{http_code}" \
        -H "Authorization: Bearer $E2B_API_TOKEN" \
        "$E2B_API_BASE_URL/health" 2>/dev/null || echo "000")
    
    local http_code="${response: -3}"
    
    if [[ "$http_code" == "200" ]]; then
        log_success "✅ E2B API healthy"
        return 0
    else
        log_error "❌ E2B API unhealthy (HTTP $http_code)"
        return 1
    fi
}

# Test sandbox training capability
test_sandbox_training() {
    local sandbox_id="$1"
    local sandbox_name="$2"
    
    log_info "🧪 Testing training capability for $sandbox_name"
    
    # Submit a quick test training job
    local training_response
    training_response=$(curl -s -w "%{http_code}" \
        -X POST \
        -H "Authorization: Bearer $E2B_API_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"code\": \"print('Bayesian VaR training test')\",
            \"timeout\": 10,
            \"test_run\": true
        }" \
        "$E2B_API_BASE_URL/sandboxes/$sandbox_id/run" 2>/dev/null || echo "000")
    
    local http_code="${training_response: -3}"
    local response_body="${training_response%???}"
    
    if [[ "$http_code" == "200" ]]; then
        local success
        success=$(echo "$response_body" | jq -r '.success // false' 2>/dev/null || echo "false")
        
        if [[ "$success" == "true" ]]; then
            log_success "✅ $sandbox_name training test passed"
            return 0
        else
            log_error "❌ $sandbox_name training test failed"
            return 1
        fi
    else
        log_error "❌ $sandbox_name training test error (HTTP $http_code)"
        return 1
    fi
}

# Update Kubernetes metrics
update_k8s_metrics() {
    log_info "📊 Updating Kubernetes metrics"
    
    # Calculate overall E2B health
    local healthy_sandboxes=0
    local total_sandboxes=3
    
    for sandbox_id in "$BAYESIAN_TRAINING_SANDBOX" "$MONTE_CARLO_SANDBOX" "$REALTIME_SANDBOX"; do
        if [[ "${SANDBOX_HEALTH[$sandbox_id]}" == "healthy" ]]; then
            ((healthy_sandboxes++))
        fi
    done
    
    local health_percentage
    health_percentage=$(echo "scale=2; $healthy_sandboxes * 100 / $total_sandboxes" | bc -l)
    
    # Update metrics via Prometheus pushgateway or direct pod annotation
    kubectl annotate pods -n "$NAMESPACE" -l app="$APP_NAME",version=green \
        e2b.health.percentage="$health_percentage" \
        e2b.healthy.sandboxes="$healthy_sandboxes" \
        e2b.total.sandboxes="$total_sandboxes" \
        e2b.last.check="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --overwrite 2>/dev/null || log_warning "Failed to update pod annotations"
    
    # Export metrics to file for Prometheus file discovery
    local metrics_file="/var/lib/prometheus/e2b-metrics.prom"
    mkdir -p "$(dirname "$metrics_file")"
    
    cat > "$metrics_file" << EOF
# E2B Sandbox Production Metrics
e2b_sandbox_health_percentage $health_percentage
e2b_healthy_sandboxes $healthy_sandboxes
e2b_total_sandboxes $total_sandboxes
EOF
    
    # Add individual sandbox metrics
    for sandbox_id in "$BAYESIAN_TRAINING_SANDBOX" "$MONTE_CARLO_SANDBOX" "$REALTIME_SANDBOX"; do
        local sandbox_name
        case "$sandbox_id" in
            "$BAYESIAN_TRAINING_SANDBOX") sandbox_name="bayesian_training" ;;
            "$MONTE_CARLO_SANDBOX") sandbox_name="monte_carlo" ;;
            "$REALTIME_SANDBOX") sandbox_name="realtime" ;;
        esac
        
        local health_value=0
        case "${SANDBOX_HEALTH[$sandbox_id]}" in
            "healthy") health_value=1 ;;
            "degraded") health_value=0.5 ;;
            "starting") health_value=0.3 ;;
            *) health_value=0 ;;
        esac
        
        cat >> "$metrics_file" << EOF
e2b_sandbox_health{sandbox="$sandbox_name"} $health_value
e2b_sandbox_cpu_usage{sandbox="$sandbox_name"} ${SANDBOX_METRICS["${sandbox_id}_cpu"]:-0}
e2b_sandbox_memory_usage{sandbox="$sandbox_name"} ${SANDBOX_METRICS["${sandbox_id}_memory"]:-0}
EOF
    done
    
    log_info "📊 Metrics updated: $health_percentage% sandboxes healthy"
}

# Send alerts for critical issues
send_alert() {
    local severity="$1"
    local message="$2"
    local sandbox_id="$3"
    
    # Check alert cooldown to prevent spam
    local current_time=$(date +%s)
    local last_alert=${ALERT_COOLDOWN["$sandbox_id"]:-0}
    local cooldown_period=300  # 5 minutes
    
    if (( current_time - last_alert < cooldown_period )); then
        return 0  # Skip alert due to cooldown
    fi
    
    ALERT_COOLDOWN["$sandbox_id"]=$current_time
    
    # Send Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        local color="warning"
        local emoji="⚠️"
        
        if [[ "$severity" == "critical" ]]; then
            color="danger"
            emoji="🚨"
        fi
        
        curl -X POST "$SLACK_WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"text\": \"$emoji E2B Sandbox Alert\",
                \"attachments\": [
                    {
                        \"color\": \"$color\",
                        \"fields\": [
                            {
                                \"title\": \"Severity\",
                                \"value\": \"$severity\",
                                \"short\": true
                            },
                            {
                                \"title\": \"Sandbox\",
                                \"value\": \"$sandbox_id\",
                                \"short\": true
                            },
                            {
                                \"title\": \"Message\",
                                \"value\": \"$message\",
                                \"short\": false
                            }
                        ]
                    }
                ]
            }" > /dev/null 2>&1 || log_warning "Failed to send Slack alert"
    fi
    
    # Send PagerDuty alert for critical issues
    if [[ "$severity" == "critical" && -n "${PAGERDUTY_API_KEY:-}" ]]; then
        curl -X POST "https://api.pagerduty.com/incidents" \
            -H "Authorization: Token token=$PAGERDUTY_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{
                \"incident\": {
                    \"type\": \"incident\",
                    \"title\": \"E2B Sandbox Critical Issue: $sandbox_id\",
                    \"service\": {
                        \"id\": \"$PAGERDUTY_SERVICE_ID\",
                        \"type\": \"service_reference\"
                    },
                    \"urgency\": \"high\",
                    \"body\": {
                        \"type\": \"incident_body\",
                        \"details\": \"$message\"
                    }
                }
            }" > /dev/null 2>&1 || log_warning "Failed to send PagerDuty alert"
    fi
    
    log_info "🚨 Alert sent: $severity - $message"
}

# Analyze sandbox health and send alerts
analyze_and_alert() {
    local critical_issues=0
    local degraded_issues=0
    
    for sandbox_id in "$BAYESIAN_TRAINING_SANDBOX" "$MONTE_CARLO_SANDBOX" "$REALTIME_SANDBOX"; do
        local sandbox_name
        case "$sandbox_id" in
            "$BAYESIAN_TRAINING_SANDBOX") sandbox_name="Bayesian Training" ;;
            "$MONTE_CARLO_SANDBOX") sandbox_name="Monte Carlo" ;;
            "$REALTIME_SANDBOX") sandbox_name="Realtime" ;;
        esac
        
        local health="${SANDBOX_HEALTH[$sandbox_id]}"
        
        case "$health" in
            "unhealthy"|"missing"|"unreachable")
                ((critical_issues++))
                send_alert "critical" "$sandbox_name sandbox is $health - training capabilities compromised" "$sandbox_id"
                ;;
            "degraded")
                ((degraded_issues++))
                send_alert "warning" "$sandbox_name sandbox performance degraded" "$sandbox_id"
                ;;
            "starting")
                log_info "🔄 $sandbox_name sandbox starting - monitoring..."
                ;;
            "healthy")
                log_success "✅ $sandbox_name sandbox healthy"
                ;;
        esac
    done
    
    # Constitutional Prime Directive compliance check
    if [[ $critical_issues -gt 1 ]]; then
        log_critical "🚨 CONSTITUTIONAL PRIME DIRECTIVE VIOLATION: Multiple E2B sandboxes critical ($critical_issues/3)"
        send_alert "critical" "CONSTITUTIONAL VIOLATION: $critical_issues of 3 E2B sandboxes are critical - model training severely impacted" "constitutional_violation"
    elif [[ $critical_issues -gt 0 ]]; then
        log_error "❌ $critical_issues E2B sandbox(es) critical - reduced training capacity"
    fi
    
    if [[ $degraded_issues -gt 0 ]]; then
        log_warning "⚠️ $degraded_issues E2B sandbox(es) degraded - performance impact"
    fi
}

# Attempt to restart unhealthy sandboxes
restart_unhealthy_sandboxes() {
    log_info "🔄 Checking for sandboxes needing restart"
    
    for sandbox_id in "$BAYESIAN_TRAINING_SANDBOX" "$MONTE_CARLO_SANDBOX" "$REALTIME_SANDBOX"; do
        local health="${SANDBOX_HEALTH[$sandbox_id]}"
        
        if [[ "$health" == "unhealthy" ]]; then
            log_info "🔄 Attempting to restart sandbox: $sandbox_id"
            
            # Attempt restart via E2B API
            local restart_response
            restart_response=$(curl -s -w "%{http_code}" \
                -X POST \
                -H "Authorization: Bearer $E2B_API_TOKEN" \
                "$E2B_API_BASE_URL/sandboxes/$sandbox_id/restart" 2>/dev/null || echo "000")
            
            local http_code="${restart_response: -3}"
            
            if [[ "$http_code" == "200" ]]; then
                log_success "✅ Restart initiated for sandbox: $sandbox_id"
                SANDBOX_HEALTH["$sandbox_id"]="starting"
            else
                log_error "❌ Failed to restart sandbox: $sandbox_id (HTTP $http_code)"
            fi
        fi
    done
}

# Generate monitoring report
generate_monitoring_report() {
    local report_file="/var/reports/e2b-monitoring-$(date +%Y%m%d-%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    cat > "$report_file" << EOF
{
    "e2b_monitoring_report": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "api_health": "$(check_e2b_api_health >/dev/null 2>&1 && echo "healthy" || echo "unhealthy")",
        "sandbox_status": {
            "bayesian_training": {
                "id": "$BAYESIAN_TRAINING_SANDBOX",
                "health": "${SANDBOX_HEALTH[$BAYESIAN_TRAINING_SANDBOX]}",
                "status": "${SANDBOX_METRICS["${BAYESIAN_TRAINING_SANDBOX}_status"]:-unknown}",
                "cpu_usage": "${SANDBOX_METRICS["${BAYESIAN_TRAINING_SANDBOX}_cpu"]:-0}",
                "memory_usage": "${SANDBOX_METRICS["${BAYESIAN_TRAINING_SANDBOX}_memory"]:-0}"
            },
            "monte_carlo": {
                "id": "$MONTE_CARLO_SANDBOX",
                "health": "${SANDBOX_HEALTH[$MONTE_CARLO_SANDBOX]}",
                "status": "${SANDBOX_METRICS["${MONTE_CARLO_SANDBOX}_status"]:-unknown}",
                "cpu_usage": "${SANDBOX_METRICS["${MONTE_CARLO_SANDBOX}_cpu"]:-0}",
                "memory_usage": "${SANDBOX_METRICS["${MONTE_CARLO_SANDBOX}_memory"]:-0}"
            },
            "realtime": {
                "id": "$REALTIME_SANDBOX",
                "health": "${SANDBOX_HEALTH[$REALTIME_SANDBOX]}",
                "status": "${SANDBOX_METRICS["${REALTIME_SANDBOX}_status"]:-unknown}",
                "cpu_usage": "${SANDBOX_METRICS["${REALTIME_SANDBOX}_cpu"]:-0}",
                "memory_usage": "${SANDBOX_METRICS["${REALTIME_SANDBOX}_memory"]:-0}"
            }
        },
        "summary": {
            "total_sandboxes": 3,
            "healthy_sandboxes": $(for k in "${!SANDBOX_HEALTH[@]}"; do [[ "${SANDBOX_HEALTH[$k]}" == "healthy" ]] && echo "1"; done | wc -l),
            "degraded_sandboxes": $(for k in "${!SANDBOX_HEALTH[@]}"; do [[ "${SANDBOX_HEALTH[$k]}" == "degraded" ]] && echo "1"; done | wc -l),
            "critical_sandboxes": $(for k in "${!SANDBOX_HEALTH[@]}"; do [[ "${SANDBOX_HEALTH[$k]}" =~ ^(unhealthy|missing|unreachable)$ ]] && echo "1"; done | wc -l),
            "constitutional_compliance": $([ $(for k in "${!SANDBOX_HEALTH[@]}"; do [[ "${SANDBOX_HEALTH[$k]}" =~ ^(unhealthy|missing|unreachable)$ ]] && echo "1"; done | wc -l) -lt 2 ] && echo "true" || echo "false")
        }
    }
}
EOF
    
    log_info "📊 Monitoring report generated: $report_file"
}

# Main monitoring loop
monitoring_loop() {
    log_info "🔄 Starting E2B monitoring loop (interval: ${MONITORING_INTERVAL}s)"
    
    while true; do
        log_info "🔍 E2B monitoring cycle started"
        
        # Check E2B API health first
        if ! check_e2b_api_health; then
            log_critical "🚨 E2B API unreachable - skipping sandbox checks"
            sleep "$MONITORING_INTERVAL"
            continue
        fi
        
        # Check each sandbox
        check_sandbox_health "$BAYESIAN_TRAINING_SANDBOX" "Bayesian Training"
        check_sandbox_health "$MONTE_CARLO_SANDBOX" "Monte Carlo"
        check_sandbox_health "$REALTIME_SANDBOX" "Realtime"
        
        # Test training capabilities periodically (every 10th cycle)
        if (( $(date +%s) % 600 == 0 )); then
            test_sandbox_training "$BAYESIAN_TRAINING_SANDBOX" "Bayesian Training"
        fi
        
        # Update metrics and analyze health
        update_k8s_metrics
        analyze_and_alert
        
        # Attempt to restart unhealthy sandboxes
        restart_unhealthy_sandboxes
        
        # Generate report every 30 minutes
        if (( $(date +%s) % 1800 == 0 )); then
            generate_monitoring_report
        fi
        
        log_info "✅ E2B monitoring cycle completed"
        sleep "$MONITORING_INTERVAL"
    done
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "🔄 Shutting down E2B monitoring"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log_info "🚀 Starting E2B Production Monitoring"
    log_info "📋 Configuration:"
    log_info "   Monitoring Interval: ${MONITORING_INTERVAL}s"
    log_info "   Bayesian Training: $BAYESIAN_TRAINING_SANDBOX"
    log_info "   Monte Carlo: $MONTE_CARLO_SANDBOX"
    log_info "   Realtime: $REALTIME_SANDBOX"
    
    init_monitoring
    monitoring_loop
}

# Execute main function
main "$@"