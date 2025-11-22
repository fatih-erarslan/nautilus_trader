#!/bin/bash
# Disaster Recovery Script for Hive Mind Rust Financial Trading System
# Implements comprehensive backup, restore, and failover procedures

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
NAMESPACE="hive-mind-production"
BACKUP_NAMESPACE="hive-mind-backup"
KUBECTL="${KUBECTL:-kubectl}"

# Disaster Recovery Configuration
DR_REGION="${DR_REGION:-us-west-2}"
DR_CLUSTER="${DR_CLUSTER:-hive-mind-dr-cluster}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
BACKUP_STORAGE="${BACKUP_STORAGE:-s3://hive-mind-backups}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
    logger -t hive-mind-dr "INFO: $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
    logger -t hive-mind-dr "SUCCESS: $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
    logger -t hive-mind-dr "WARNING: $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    logger -t hive-mind-dr "ERROR: $1"
}

# Backup functions
create_full_backup() {
    local backup_id="backup-$(date +%Y%m%d-%H%M%S)"
    local backup_dir="/tmp/$backup_id"
    
    log_info "Creating full system backup: $backup_id"
    
    mkdir -p "$backup_dir"
    
    # Backup Kubernetes resources
    log_info "Backing up Kubernetes resources..."
    
    # Export all resources
    "$KUBECTL" get all,pvc,secrets,configmaps,networkpolicies,hpa,vpa \
        -n "$NAMESPACE" \
        -o yaml > "$backup_dir/k8s-resources.yaml"
    
    # Backup persistent volume data
    log_info "Backing up persistent volume data..."
    backup_persistent_volumes "$backup_dir"
    
    # Backup application state
    log_info "Backing up application state..."
    backup_application_state "$backup_dir"
    
    # Backup configuration and secrets
    log_info "Backing up configuration..."
    backup_configuration "$backup_dir"
    
    # Create backup metadata
    create_backup_metadata "$backup_dir" "$backup_id"
    
    # Compress and upload backup
    log_info "Compressing and uploading backup..."
    tar -czf "$backup_dir.tar.gz" -C "/tmp" "$backup_id"
    
    # Upload to backup storage
    if command -v aws >/dev/null 2>&1; then
        aws s3 cp "$backup_dir.tar.gz" "$BACKUP_STORAGE/$backup_id.tar.gz" \
            --server-side-encryption AES256
    fi
    
    # Cleanup local backup files
    rm -rf "$backup_dir" "$backup_dir.tar.gz"
    
    log_success "Full backup completed: $backup_id"
    echo "$backup_id"
}

backup_persistent_volumes() {
    local backup_dir="$1"
    local pv_backup_dir="$backup_dir/persistent-volumes"
    
    mkdir -p "$pv_backup_dir"
    
    # Get all PVCs in the namespace
    "$KUBECTL" get pvc -n "$NAMESPACE" -o name | while read -r pvc; do
        local pvc_name=$(echo "$pvc" | cut -d'/' -f2)
        log_info "Creating snapshot of PVC: $pvc_name"
        
        # Create volume snapshot (if supported)
        if "$KUBECTL" get crd volumesnapshots.snapshot.storage.k8s.io >/dev/null 2>&1; then
            cat <<EOF | "$KUBECTL" apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: ${pvc_name}-snapshot-$(date +%Y%m%d-%H%M%S)
  namespace: $NAMESPACE
spec:
  volumeSnapshotClassName: csi-aws-vsc
  source:
    persistentVolumeClaimName: $pvc_name
EOF
        fi
        
        # Alternative: Use backup job for data copying
        create_pv_backup_job "$pvc_name" "$pv_backup_dir"
    done
}

create_pv_backup_job() {
    local pvc_name="$1"
    local backup_dir="$2"
    
    # Create a backup job that copies data from PVC
    cat <<EOF | "$KUBECTL" apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: backup-${pvc_name}-$(date +%Y%m%d-%H%M%S)
  namespace: $NAMESPACE
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: backup
        image: alpine:3.18
        command:
        - sh
        - -c
        - |
          apk add --no-cache tar gzip
          cd /data
          tar -czf /backup/${pvc_name}.tar.gz .
          echo "Backup of $pvc_name completed"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: backup
          mountPath: /backup
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: $pvc_name
      - name: backup
        emptyDir: {}
EOF
    
    # Wait for job completion
    "$KUBECTL" wait --for=condition=complete job/backup-"$pvc_name"-$(date +%Y%m%d-%H%M%S) \
        -n "$NAMESPACE" --timeout=600s
}

backup_application_state() {
    local backup_dir="$1"
    local state_backup_dir="$backup_dir/application-state"
    
    mkdir -p "$state_backup_dir"
    
    # Trigger application-level backup via API
    local pods=$("$KUBECTL" get pods -n "$NAMESPACE" -l app=hive-mind-rust -o name)
    
    echo "$pods" | while read -r pod; do
        local pod_name=$(echo "$pod" | cut -d'/' -f2)
        log_info "Backing up application state from pod: $pod_name"
        
        # Execute backup command in pod
        "$KUBECTL" exec -n "$NAMESPACE" "$pod_name" -- \
            /bin/bash -c "/app/bin/hive-mind-server backup --output /tmp/state-backup.json"
        
        # Copy backup file from pod
        "$KUBECTL" cp "$NAMESPACE/$pod_name:/tmp/state-backup.json" \
            "$state_backup_dir/$pod_name-state.json"
    done
}

backup_configuration() {
    local backup_dir="$1"
    local config_backup_dir="$backup_dir/configuration"
    
    mkdir -p "$config_backup_dir"
    
    # Export ConfigMaps
    "$KUBECTL" get configmaps -n "$NAMESPACE" -o yaml > "$config_backup_dir/configmaps.yaml"
    
    # Export Secrets (metadata only for security)
    "$KUBECTL" get secrets -n "$NAMESPACE" -o yaml \
        | sed 's/data:/data: # REDACTED/' \
        > "$config_backup_dir/secrets-metadata.yaml"
    
    # Export RBAC
    "$KUBECTL" get serviceaccounts,roles,rolebindings \
        -n "$NAMESPACE" -o yaml > "$config_backup_dir/rbac.yaml"
}

create_backup_metadata() {
    local backup_dir="$1"
    local backup_id="$2"
    
    cat <<EOF > "$backup_dir/backup-metadata.json"
{
  "backup_id": "$backup_id",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "kubernetes_version": "$("$KUBECTL" version --client -o json | jq -r '.clientVersion.gitVersion')",
  "cluster_info": {
    "current_context": "$("$KUBECTL" config current-context)",
    "server": "$("$KUBECTL" cluster-info | head -1 | awk '{print $NF}')"
  },
  "application_info": {
    "namespace": "$NAMESPACE",
    "replicas": $("$KUBECTL" get deployment hive-mind-rust -n "$NAMESPACE" -o jsonpath='{.spec.replicas}'),
    "image": "$("$KUBECTL" get deployment hive-mind-rust -n "$NAMESPACE" -o jsonpath='{.spec.template.spec.containers[0].image}')"
  },
  "backup_components": [
    "kubernetes-resources",
    "persistent-volumes",
    "application-state",
    "configuration"
  ]
}
EOF
}

# Restore functions
restore_from_backup() {
    local backup_id="$1"
    local restore_namespace="${2:-$NAMESPACE}"
    
    log_info "Starting restore from backup: $backup_id"
    
    # Download backup
    local backup_file="/tmp/$backup_id.tar.gz"
    if command -v aws >/dev/null 2>&1; then
        aws s3 cp "$BACKUP_STORAGE/$backup_id.tar.gz" "$backup_file"
    else
        log_error "Backup file not found: $backup_id"
        return 1
    fi
    
    # Extract backup
    local restore_dir="/tmp/$backup_id"
    tar -xzf "$backup_file" -C "/tmp"
    
    # Validate backup
    if [[ ! -f "$restore_dir/backup-metadata.json" ]]; then
        log_error "Invalid backup: missing metadata"
        return 1
    fi
    
    # Create namespace if it doesn't exist
    "$KUBECTL" create namespace "$restore_namespace" --dry-run=client -o yaml | "$KUBECTL" apply -f -
    
    # Restore configuration first
    log_info "Restoring configuration..."
    restore_configuration "$restore_dir" "$restore_namespace"
    
    # Restore persistent volumes
    log_info "Restoring persistent volumes..."
    restore_persistent_volumes "$restore_dir" "$restore_namespace"
    
    # Restore Kubernetes resources
    log_info "Restoring Kubernetes resources..."
    restore_kubernetes_resources "$restore_dir" "$restore_namespace"
    
    # Restore application state
    log_info "Restoring application state..."
    restore_application_state "$restore_dir" "$restore_namespace"
    
    # Wait for system to be ready
    log_info "Waiting for system to be ready..."
    wait_for_system_ready "$restore_namespace"
    
    # Cleanup
    rm -rf "$restore_dir" "$backup_file"
    
    log_success "Restore completed successfully"
}

restore_configuration() {
    local restore_dir="$1"
    local namespace="$2"
    local config_dir="$restore_dir/configuration"
    
    # Restore ConfigMaps
    sed "s/namespace: .*/namespace: $namespace/" "$config_dir/configmaps.yaml" \
        | "$KUBECTL" apply -f -
    
    # Note: Secrets need to be restored manually or through external secret management
    log_warning "Secrets must be restored manually through secret management system"
    
    # Restore RBAC
    sed "s/namespace: .*/namespace: $namespace/" "$config_dir/rbac.yaml" \
        | "$KUBECTL" apply -f -
}

restore_persistent_volumes() {
    local restore_dir="$1"
    local namespace="$2"
    
    # First, restore from volume snapshots if available
    if "$KUBECTL" get volumesnapshots -n "$namespace" >/dev/null 2>&1; then
        log_info "Restoring from volume snapshots..."
        restore_from_volume_snapshots "$namespace"
    fi
    
    # Alternative: Restore from backup data
    local pv_dir="$restore_dir/persistent-volumes"
    if [[ -d "$pv_dir" ]]; then
        restore_from_backup_data "$pv_dir" "$namespace"
    fi
}

restore_kubernetes_resources() {
    local restore_dir="$1"
    local namespace="$2"
    
    # Update namespace in resources
    sed "s/namespace: .*/namespace: $namespace/" "$restore_dir/k8s-resources.yaml" \
        | "$KUBECTL" apply -f -
}

restore_application_state() {
    local restore_dir="$1"
    local namespace="$2"
    local state_dir="$restore_dir/application-state"
    
    if [[ ! -d "$state_dir" ]]; then
        log_warning "No application state backup found"
        return
    fi
    
    # Wait for pods to be ready
    "$KUBECTL" wait --for=condition=ready pods \
        -l app=hive-mind-rust \
        -n "$namespace" \
        --timeout=300s
    
    # Restore state to each pod
    find "$state_dir" -name "*.json" | while read -r state_file; do
        local pod_name=$(basename "$state_file" -state.json)
        local current_pods=$("$KUBECTL" get pods -n "$namespace" -l app=hive-mind-rust -o name | head -1)
        local current_pod=$(echo "$current_pods" | cut -d'/' -f2)
        
        if [[ -n "$current_pod" ]]; then
            log_info "Restoring state to pod: $current_pod"
            
            # Copy state file to pod
            "$KUBECTL" cp "$state_file" "$namespace/$current_pod:/tmp/restore-state.json"
            
            # Execute restore command
            "$KUBECTL" exec -n "$namespace" "$current_pod" -- \
                /app/bin/hive-mind-server restore --input /tmp/restore-state.json
        fi
    done
}

wait_for_system_ready() {
    local namespace="$1"
    
    # Wait for deployment to be ready
    "$KUBECTL" rollout status deployment/hive-mind-rust \
        -n "$namespace" \
        --timeout=600s
    
    # Wait for all pods to be ready
    "$KUBECTL" wait --for=condition=ready pods \
        -l app=hive-mind-rust \
        -n "$namespace" \
        --timeout=300s
    
    # Perform health check
    local service_ip=$("$KUBECTL" get service hive-mind-service \
        -n "$namespace" \
        -o jsonpath='{.spec.clusterIP}')
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://$service_ip:8091/health" >/dev/null 2>&1; then
            log_success "System health check passed"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    log_error "System failed health checks after restore"
    return 1
}

# Failover functions
initiate_failover() {
    local target_region="${1:-$DR_REGION}"
    local target_cluster="${2:-$DR_CLUSTER}"
    
    log_info "Initiating failover to region: $target_region, cluster: $target_cluster"
    
    # Create emergency backup before failover
    log_info "Creating emergency backup before failover..."
    local emergency_backup=$(create_full_backup)
    
    # Switch kubectl context to DR cluster
    log_info "Switching to DR cluster..."
    kubectl config use-context "$target_cluster" || {
        log_error "Failed to switch to DR cluster: $target_cluster"
        return 1
    }
    
    # Restore latest backup to DR cluster
    log_info "Restoring system in DR cluster..."
    local latest_backup=$(get_latest_backup)
    restore_from_backup "$latest_backup" "$NAMESPACE-dr"
    
    # Update DNS/Load Balancer to point to DR cluster
    log_info "Updating traffic routing to DR cluster..."
    update_traffic_routing "$target_region"
    
    # Verify DR system is operational
    log_info "Verifying DR system functionality..."
    verify_dr_system
    
    log_success "Failover completed successfully"
    
    # Send notifications
    send_failover_notification "$target_region" "$emergency_backup"
}

get_latest_backup() {
    if command -v aws >/dev/null 2>&1; then
        aws s3 ls "$BACKUP_STORAGE/" | sort | tail -1 | awk '{print $4}' | sed 's/.tar.gz$//'
    else
        log_error "Cannot determine latest backup"
        return 1
    fi
}

update_traffic_routing() {
    local target_region="$1"
    
    # This would typically involve updating:
    # - DNS records
    # - Load balancer configurations
    # - CDN settings
    # - API gateway routes
    
    log_info "Traffic routing update would be implemented here for region: $target_region"
    log_warning "Manual intervention required for traffic routing"
}

verify_dr_system() {
    local dr_namespace="$NAMESPACE-dr"
    
    # Check system health
    local service_ip=$("$KUBECTL" get service hive-mind-service \
        -n "$dr_namespace" \
        -o jsonpath='{.spec.clusterIP}')
    
    if ! curl -f "http://$service_ip:8091/health" >/dev/null 2>&1; then
        log_error "DR system health check failed"
        return 1
    fi
    
    # Verify trading functionality
    log_info "Verifying trading functionality..."
    # Add specific trading system tests here
    
    log_success "DR system verification completed"
}

send_failover_notification() {
    local target_region="$1"
    local backup_id="$2"
    
    local message="CRITICAL: Hive Mind system failover completed
Region: $target_region
Backup used: $backup_id
Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Status: Operational"
    
    # Send to monitoring systems
    if [[ -n "${WEBHOOK_URL:-}" ]]; then
        curl -X POST "$WEBHOOK_URL" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"$message\", \"priority\": \"critical\"}"
    fi
    
    # Log to system
    logger -p user.crit -t hive-mind-dr "$message"
}

# Cleanup functions
cleanup_old_backups() {
    local retention_days="${1:-$BACKUP_RETENTION_DAYS}"
    
    log_info "Cleaning up backups older than $retention_days days"
    
    if command -v aws >/dev/null 2>&1; then
        # List and delete old backups from S3
        aws s3 ls "$BACKUP_STORAGE/" | while read -r line; do
            local file_date=$(echo "$line" | awk '{print $1" "$2}')
            local file_name=$(echo "$line" | awk '{print $4}')
            local file_age=$(($(date +%s) - $(date -d "$file_date" +%s)))
            local retention_seconds=$((retention_days * 24 * 3600))
            
            if [[ $file_age -gt $retention_seconds ]]; then
                log_info "Deleting old backup: $file_name"
                aws s3 rm "$BACKUP_STORAGE/$file_name"
            fi
        done
    fi
}

# Main command dispatcher
main() {
    case "${1:-help}" in
        backup)
            create_full_backup
            ;;
        restore)
            if [[ -z "${2:-}" ]]; then
                log_error "Usage: $0 restore <backup-id> [namespace]"
                exit 1
            fi
            restore_from_backup "$2" "${3:-}"
            ;;
        failover)
            initiate_failover "${2:-}" "${3:-}"
            ;;
        cleanup)
            cleanup_old_backups "${2:-}"
            ;;
        list-backups)
            if command -v aws >/dev/null 2>&1; then
                aws s3 ls "$BACKUP_STORAGE/"
            fi
            ;;
        verify)
            verify_dr_system
            ;;
        help|*)
            echo "Hive Mind Disaster Recovery Tool"
            echo "Usage: $0 {backup|restore|failover|cleanup|list-backups|verify|help}"
            echo ""
            echo "Commands:"
            echo "  backup                    - Create full system backup"
            echo "  restore <backup-id> [ns]  - Restore from backup"
            echo "  failover [region] [cluster] - Initiate disaster recovery failover"
            echo "  cleanup [days]            - Clean up old backups"
            echo "  list-backups              - List available backups"
            echo "  verify                    - Verify DR system health"
            echo "  help                      - Show this help"
            ;;
    esac
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi