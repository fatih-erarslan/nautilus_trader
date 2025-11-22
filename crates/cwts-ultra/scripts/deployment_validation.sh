#!/bin/bash
# Production Deployment Validation Script for Bayesian VaR System
# Constitutional Prime Directive Compliance Validation

set -euo pipefail

# Configuration
NAMESPACE="production"
APP_NAME="bayesian-var"
VERSION="v2.0.0-production"
VALIDATION_TIMEOUT="600s"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${RED}[ERROR]${NC} $1"
}

log_critical() {
    echo -e "$(date -u +"%Y-%m-%dT%H:%M:%SZ") ${RED}[CRITICAL]${NC} 🚨 $1"
}

# Validation results tracking
VALIDATION_RESULTS=()
FAILED_VALIDATIONS=()

add_validation_result() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    VALIDATION_RESULTS+=("$test_name:$status:$message")
    
    if [[ "$status" == "FAIL" ]]; then
        FAILED_VALIDATIONS+=("$test_name: $message")
        log_error "❌ $test_name: $message"
    else
        log_success "✅ $test_name: $message"
    fi
}

# 1. Health Check Validation
validate_health_endpoints() {
    log_info "🩺 Validating health endpoints..."
    
    local service_ip
    service_ip=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test readiness endpoint
    if kubectl run health-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -f -s "http://$service_ip:8080/health/ready" --max-time 10 >/dev/null 2>&1; then
        add_validation_result "readiness_endpoint" "PASS" "Readiness endpoint responding correctly"
    else
        add_validation_result "readiness_endpoint" "FAIL" "Readiness endpoint not responding"
        return 1
    fi
    
    # Test liveness endpoint
    if kubectl run liveness-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -f -s "http://$service_ip:8080/health/live" --max-time 10 >/dev/null 2>&1; then
        add_validation_result "liveness_endpoint" "PASS" "Liveness endpoint responding correctly"
    else
        add_validation_result "liveness_endpoint" "FAIL" "Liveness endpoint not responding"
        return 1
    fi
    
    # Test startup endpoint
    if kubectl run startup-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -f -s "http://$service_ip:8080/health/startup" --max-time 10 >/dev/null 2>&1; then
        add_validation_result "startup_endpoint" "PASS" "Startup endpoint responding correctly"
    else
        add_validation_result "startup_endpoint" "FAIL" "Startup endpoint not responding"
        return 1
    fi
}

# 2. Real Data Connectivity Validation
validate_real_data_connectivity() {
    log_info "🔗 Validating real data connectivity..."
    
    local service_ip
    service_ip=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test Binance WebSocket connectivity
    local binance_status
    binance_status=$(kubectl run binance-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -s "http://$service_ip:8080/health/binance" --max-time 10 | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "error")
    
    if [[ "$binance_status" == "healthy" ]]; then
        add_validation_result "binance_connectivity" "PASS" "Binance WebSocket connection active"
    else
        add_validation_result "binance_connectivity" "FAIL" "Binance connectivity failed: $binance_status"
        return 1
    fi
    
    # Validate active connections count
    local active_connections
    active_connections=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "bayesian_var_active_connections" | awk '{print $2}' 2>/dev/null || echo "0")
    
    if [[ "$active_connections" -gt 0 ]]; then
        add_validation_result "active_connections" "PASS" "$active_connections active Binance connections"
    else
        add_validation_result "active_connections" "FAIL" "No active Binance connections"
        return 1
    fi
}

# 3. E2B Sandbox Integration Validation
validate_e2b_integration() {
    log_info "🧪 Validating E2B sandbox integration..."
    
    local service_ip
    service_ip=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Test E2B connectivity
    local e2b_status
    e2b_status=$(kubectl run e2b-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -s "http://$service_ip:8080/health/e2b" --max-time 15 | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "error")
    
    if [[ "$e2b_status" == "healthy" ]]; then
        add_validation_result "e2b_connectivity" "PASS" "E2B sandbox connectivity active"
    else
        add_validation_result "e2b_connectivity" "FAIL" "E2B connectivity failed: $e2b_status"
        return 1
    fi
    
    # Test sandbox training capability
    local training_response
    training_response=$(kubectl run e2b-training-test --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
        curl -s -X POST "http://$service_ip:8080/api/train-model" \
        -H "Content-Type: application/json" \
        -d '{"training_type":"validation_test","duration":5}' \
        --max-time 30 2>/dev/null | grep -o '"success":[^,}]*' | cut -d':' -f2 || echo "false")
    
    if [[ "$training_response" == "true" ]]; then
        add_validation_result "e2b_training" "PASS" "E2B sandbox training operational"
    else
        add_validation_result "e2b_training" "FAIL" "E2B sandbox training failed"
        return 1
    fi
}

# 4. Model Accuracy Validation
validate_model_accuracy() {
    log_info "🎯 Validating model accuracy..."
    
    # Get current model accuracy from metrics
    local model_accuracy
    model_accuracy=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "bayesian_model_accuracy_score" | awk '{print $2}' 2>/dev/null || echo "0")
    
    if (( $(echo "$model_accuracy >= 0.95" | bc -l) )); then
        add_validation_result "model_accuracy" "PASS" "Model accuracy: $(printf "%.2f%%" $(echo "$model_accuracy * 100" | bc -l))"
    else
        add_validation_result "model_accuracy" "FAIL" "Model accuracy below threshold: $(printf "%.2f%%" $(echo "$model_accuracy * 100" | bc -l))"
        return 1
    fi
}

# 5. Performance Validation (SLA Compliance)
validate_performance_sla() {
    log_info "⚡ Validating performance SLA compliance..."
    
    # Test VaR calculation response time
    local service_ip
    service_ip=$(kubectl get service "${APP_NAME}-service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Perform 10 VaR calculations and measure response times
    local total_time=0
    local successful_calculations=0
    
    for i in {1..10}; do
        local start_time=$(date +%s%N)
        
        local response
        response=$(kubectl run perf-test-$i --image=curlimages/curl:latest --rm -i --tty=false --restart=Never -n "$NAMESPACE" -- \
            curl -s -X POST "http://$service_ip:8080/api/calculate-var" \
            -H "Content-Type: application/json" \
            -d "{\"portfolio\":{\"BTC\":0.1},\"confidence_level\":0.95,\"use_real_data\":true}" \
            --max-time 5 --write-out "%{time_total}" 2>/dev/null || echo "error")
        
        local end_time=$(date +%s%N)
        local duration=$(echo "scale=3; ($end_time - $start_time) / 1000000000" | bc -l)
        
        if [[ "$response" != "error" ]] && echo "$response" | grep -q "var_estimate"; then
            total_time=$(echo "$total_time + $duration" | bc -l)
            ((successful_calculations++))
        fi
    done
    
    if [[ $successful_calculations -ge 8 ]]; then
        local avg_response_time
        avg_response_time=$(echo "scale=3; $total_time / $successful_calculations" | bc -l)
        
        if (( $(echo "$avg_response_time <= 1.0" | bc -l) )); then
            add_validation_result "performance_sla" "PASS" "Average response time: ${avg_response_time}s (${successful_calculations}/10 successful)"
        else
            add_validation_result "performance_sla" "FAIL" "Average response time ${avg_response_time}s exceeds 1s SLA"
            return 1
        fi
    else
        add_validation_result "performance_sla" "FAIL" "Only ${successful_calculations}/10 calculations successful"
        return 1
    fi
}

# 6. Error Rate Validation
validate_error_rates() {
    log_info "📊 Validating error rates..."
    
    # Get error rate from metrics
    local failed_calculations
    failed_calculations=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "bayesian_var_calculations_failed_total" | awk '{print $2}' 2>/dev/null || echo "0")
    
    local total_calculations
    total_calculations=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "bayesian_var_total_calculations" | awk '{print $2}' 2>/dev/null || echo "1")
    
    local error_rate
    error_rate=$(echo "scale=6; $failed_calculations / $total_calculations" | bc -l)
    
    if (( $(echo "$error_rate <= 0.01" | bc -l) )); then
        add_validation_result "error_rate" "PASS" "Error rate: $(printf "%.4f%%" $(echo "$error_rate * 100" | bc -l))"
    else
        add_validation_result "error_rate" "FAIL" "Error rate $(printf "%.4f%%" $(echo "$error_rate * 100" | bc -l)) exceeds 1% threshold"
        return 1
    fi
}

# 7. Security Validation
validate_security() {
    log_info "🔒 Validating security configuration..."
    
    # Check pod security context
    local security_context
    security_context=$(kubectl get pod -n "$NAMESPACE" -l app="$APP_NAME",version=green -o jsonpath='{.items[0].spec.securityContext.runAsNonRoot}' 2>/dev/null || echo "false")
    
    if [[ "$security_context" == "true" ]]; then
        add_validation_result "pod_security" "PASS" "Pod running as non-root user"
    else
        add_validation_result "pod_security" "FAIL" "Pod not configured to run as non-root"
        return 1
    fi
    
    # Check container security context
    local container_security
    container_security=$(kubectl get pod -n "$NAMESPACE" -l app="$APP_NAME",version=green -o jsonpath='{.items[0].spec.containers[0].securityContext.readOnlyRootFilesystem}' 2>/dev/null || echo "false")
    
    if [[ "$container_security" == "true" ]]; then
        add_validation_result "container_security" "PASS" "Container using read-only root filesystem"
    else
        add_validation_result "container_security" "FAIL" "Container not using read-only root filesystem"
        return 1
    fi
    
    # Check for required secrets
    local required_secrets=("binance-api-credentials" "e2b-api-credentials")
    for secret in "${required_secrets[@]}"; do
        if kubectl get secret "$secret" -n "$NAMESPACE" >/dev/null 2>&1; then
            add_validation_result "secret_$secret" "PASS" "Required secret '$secret' present"
        else
            add_validation_result "secret_$secret" "FAIL" "Required secret '$secret' missing"
            return 1
        fi
    done
}

# 8. Resource Usage Validation
validate_resource_usage() {
    log_info "💾 Validating resource usage..."
    
    # Check memory usage
    local memory_usage
    memory_usage=$(kubectl top pod -n "$NAMESPACE" -l app="$APP_NAME",version=green --no-headers 2>/dev/null | awk '{print $3}' | sed 's/Mi//' | head -1 || echo "0")
    
    if [[ $memory_usage -lt 1800 ]]; then  # Under 1.8GB (90% of 2GB limit)
        add_validation_result "memory_usage" "PASS" "Memory usage: ${memory_usage}Mi"
    else
        add_validation_result "memory_usage" "FAIL" "Memory usage ${memory_usage}Mi approaching limit"
        return 1
    fi
    
    # Check CPU usage
    local cpu_usage
    cpu_usage=$(kubectl top pod -n "$NAMESPACE" -l app="$APP_NAME",version=green --no-headers 2>/dev/null | awk '{print $2}' | sed 's/m//' | head -1 || echo "0")
    
    if [[ $cpu_usage -lt 800 ]]; then  # Under 800m (80% of 1000m limit)
        add_validation_result "cpu_usage" "PASS" "CPU usage: ${cpu_usage}m"
    else
        add_validation_result "cpu_usage" "FAIL" "CPU usage ${cpu_usage}m approaching limit"
        return 1
    fi
}

# 9. Constitutional Prime Directive Compliance
validate_constitutional_compliance() {
    log_info "🏛️ Validating Constitutional Prime Directive compliance..."
    
    # Check for any constitutional violations
    local violations
    violations=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "constitutional_prime_directive_violations_total" | awk '{print $2}' 2>/dev/null || echo "0")
    
    if [[ "$violations" == "0" ]]; then
        add_validation_result "constitutional_compliance" "PASS" "No Constitutional Prime Directive violations"
    else
        add_validation_result "constitutional_compliance" "FAIL" "$violations Constitutional Prime Directive violations detected"
        return 1
    fi
    
    # Check emergency stops
    local emergency_stops
    emergency_stops=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "emergency_stops_total" | awk '{print $2}' 2>/dev/null || echo "0")
    
    if [[ "$emergency_stops" == "0" ]]; then
        add_validation_result "emergency_stops" "PASS" "No emergency stops activated"
    else
        add_validation_result "emergency_stops" "FAIL" "$emergency_stops emergency stops detected"
        return 1
    fi
}

# 10. Monitoring and Observability
validate_monitoring() {
    log_info "📊 Validating monitoring and observability..."
    
    # Check metrics endpoint
    local metrics_count
    metrics_count=$(kubectl exec -n "$NAMESPACE" deployment/"${APP_NAME}-green" -- \
        curl -s http://localhost:8090/metrics | grep "bayesian_var" | wc -l 2>/dev/null || echo "0")
    
    if [[ $metrics_count -gt 10 ]]; then
        add_validation_result "metrics_endpoint" "PASS" "$metrics_count Bayesian VaR metrics available"
    else
        add_validation_result "metrics_endpoint" "FAIL" "Insufficient metrics available: $metrics_count"
        return 1
    fi
    
    # Check OpenTelemetry traces
    if kubectl get pod -n "$NAMESPACE" -l app=opentelemetry-collector | grep Running >/dev/null 2>&1; then
        add_validation_result "opentelemetry" "PASS" "OpenTelemetry collector running"
    else
        add_validation_result "opentelemetry" "FAIL" "OpenTelemetry collector not running"
        return 1
    fi
    
    # Check Prometheus ServiceMonitor
    if kubectl get servicemonitor "${APP_NAME}-monitor" -n "$NAMESPACE" >/dev/null 2>&1; then
        add_validation_result "prometheus_monitor" "PASS" "Prometheus ServiceMonitor configured"
    else
        add_validation_result "prometheus_monitor" "FAIL" "Prometheus ServiceMonitor missing"
        return 1
    fi
}

# Generate validation report
generate_validation_report() {
    local report_file="/var/reports/deployment-validation-$(date +%Y%m%d-%H%M%S).json"
    mkdir -p "$(dirname "$report_file")"
    
    local total_tests=${#VALIDATION_RESULTS[@]}
    local failed_tests=${#FAILED_VALIDATIONS[@]}
    local passed_tests=$((total_tests - failed_tests))
    local success_rate=$(echo "scale=2; $passed_tests * 100 / $total_tests" | bc -l)
    
    cat > "$report_file" << EOF
{
    "deployment_validation": {
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "version": "$VERSION",
        "namespace": "$NAMESPACE",
        "total_tests": $total_tests,
        "passed_tests": $passed_tests,
        "failed_tests": $failed_tests,
        "success_rate": "$success_rate%",
        "overall_status": "$([ $failed_tests -eq 0 ] && echo "PASS" || echo "FAIL")",
        "constitutional_prime_directive_compliant": $([ $failed_tests -eq 0 ] && echo "true" || echo "false")
    },
    "test_results": [
EOF
    
    local first=true
    for result in "${VALIDATION_RESULTS[@]}"; do
        IFS=':' read -r test_name status message <<< "$result"
        
        if [[ "$first" == "false" ]]; then
            echo "," >> "$report_file"
        fi
        first=false
        
        cat >> "$report_file" << EOF
        {
            "test_name": "$test_name",
            "status": "$status",
            "message": "$message"
        }EOF
    done
    
    cat >> "$report_file" << EOF
    ],
    "failed_validations": [
EOF
    
    if [[ ${#FAILED_VALIDATIONS[@]} -gt 0 ]]; then
        first=true
        for failure in "${FAILED_VALIDATIONS[@]}"; do
            if [[ "$first" == "false" ]]; then
                echo "," >> "$report_file"
            fi
            first=false
            echo "        \"$failure\"" >> "$report_file"
        done
    fi
    
    cat >> "$report_file" << EOF
    ]
}
EOF
    
    log_info "📊 Validation report generated: $report_file"
    echo "$report_file"
}

# Main validation function
main() {
    log_info "🚀 Starting production deployment validation"
    log_info "📋 Validation Details:"
    log_info "   Application: $APP_NAME"
    log_info "   Namespace: $NAMESPACE"
    log_info "   Version: $VERSION"
    log_info "   Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    
    # Execute all validation steps
    validate_health_endpoints || true
    validate_real_data_connectivity || true
    validate_e2b_integration || true
    validate_model_accuracy || true
    validate_performance_sla || true
    validate_error_rates || true
    validate_security || true
    validate_resource_usage || true
    validate_constitutional_compliance || true
    validate_monitoring || true
    
    # Generate comprehensive report
    local report_file
    report_file=$(generate_validation_report)
    
    # Summary
    local total_tests=${#VALIDATION_RESULTS[@]}
    local failed_tests=${#FAILED_VALIDATIONS[@]}
    local passed_tests=$((total_tests - failed_tests))
    
    log_info "📊 VALIDATION SUMMARY:"
    log_info "   Total Tests: $total_tests"
    log_success "   Passed: $passed_tests"
    
    if [[ $failed_tests -gt 0 ]]; then
        log_error "   Failed: $failed_tests"
        log_critical "❌ DEPLOYMENT VALIDATION FAILED"
        log_critical "Failed validations:"
        for failure in "${FAILED_VALIDATIONS[@]}"; do
            log_critical "   - $failure"
        done
        log_critical "🏛️ Constitutional Prime Directive: Deployment does not meet production requirements"
        exit 1
    else
        log_success "   Failed: $failed_tests"
        log_success "🎯 DEPLOYMENT VALIDATION PASSED"
        log_success "✅ All validation criteria met"
        log_success "🏛️ Constitutional Prime Directive: Deployment ready for production traffic"
        log_success "📋 Validation Report: $report_file"
    fi
}

# Execute main validation
main "$@"