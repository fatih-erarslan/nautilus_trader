#!/bin/bash

# AI News Trading Benchmark - Cleanup Script
# This script cleans up temporary files, logs, and system resources

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Global variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DRY_RUN=false
FORCE=false
DEEP_CLEAN=false
KEEP_DAYS=7

# Function to show usage
show_usage() {
    cat << EOF
AI News Trading Benchmark - Cleanup Script

Usage: $0 [OPTIONS]

Options:
  --dry-run           Show what would be cleaned without actually doing it
  --force             Force cleanup without confirmation prompts
  --deep              Deep clean including virtual environment and Docker
  --keep-days DAYS    Keep files newer than N days (default: 7)
  --help, -h          Show this help

Cleanup Operations:
  - Temporary files and directories
  - Old log files
  - Cache files
  - Old result files
  - Python bytecode files
  - Test artifacts
  - Docker containers and images (with --deep)
  - Virtual environment (with --deep)

Examples:
  $0 --dry-run                # Show what would be cleaned
  $0 --keep-days 3            # Keep files from last 3 days
  $0 --deep --force           # Deep clean without prompts

EOF
}

# Function to confirm action
confirm_action() {
    local message="$1"
    
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    echo -n "$message (y/N): "
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to safely remove files/directories
safe_remove() {
    local target="$1"
    local description="$2"
    
    if [[ ! -e "$target" ]]; then
        return 0
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would remove: $target ($description)"
        return 0
    fi
    
    log_info "Removing: $target ($description)"
    
    if [[ -d "$target" ]]; then
        rm -rf "$target"
    else
        rm -f "$target"
    fi
}

# Function to clean temporary files
clean_temp_files() {
    log_info "Cleaning temporary files..."
    
    cd "$PROJECT_ROOT"
    
    # Python cache files
    find . -type d -name "__pycache__" | while read -r dir; do
        safe_remove "$dir" "Python cache directory"
    done
    
    find . -name "*.pyc" -type f | while read -r file; do
        safe_remove "$file" "Python bytecode file"
    done
    
    find . -name "*.pyo" -type f | while read -r file; do
        safe_remove "$file" "Python optimized bytecode file"
    done
    
    # Temporary directories
    local temp_dirs=(
        "/tmp/benchmark_*"
        "tmp"
        ".tmp"
        "temp"
        ".pytest_cache"
        ".coverage"
        "htmlcov"
        ".mypy_cache"
        ".tox"
        "dist"
        "build"
        "*.egg-info"
    )
    
    for pattern in "${temp_dirs[@]}"; do
        for item in $pattern; do
            if [[ -e "$item" ]]; then
                safe_remove "$item" "Temporary directory/file"
            fi
        done
    done
    
    # System temporary files
    if [[ -d "/tmp" ]]; then
        find /tmp -name "benchmark_*" -type d -user "$(whoami)" 2>/dev/null | while read -r dir; do
            safe_remove "$dir" "System temporary directory"
        done
    fi
}

# Function to clean log files
clean_log_files() {
    log_info "Cleaning old log files..."
    
    cd "$PROJECT_ROOT"
    
    # Clean logs directory
    if [[ -d "logs" ]]; then
        find logs -name "*.log" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old log file (>$KEEP_DAYS days)"
        done
        
        find logs -name "*.log.*" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old rotated log file (>$KEEP_DAYS days)"
        done
    fi
    
    # Clean system logs (if writable)
    local log_patterns=(
        "benchmark*.log"
        "*.benchmark.log"
    )
    
    for pattern in "${log_patterns[@]}"; do
        for file in $pattern; do
            if [[ -f "$file" && -w "$file" ]]; then
                if [[ $(find "$file" -mtime +$KEEP_DAYS 2>/dev/null) ]]; then
                    safe_remove "$file" "Old log file (>$KEEP_DAYS days)"
                fi
            fi
        done
    done
}

# Function to clean cache files
clean_cache_files() {
    log_info "Cleaning cache files..."
    
    cd "$PROJECT_ROOT"
    
    # Data cache
    if [[ -d "data/.cache" ]]; then
        find data/.cache -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old cache file (>$KEEP_DAYS days)"
        done
    fi
    
    # Monitoring data cache
    if [[ -d "monitoring_data" ]]; then
        find monitoring_data -name "*_cache*" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old monitoring cache (>$KEEP_DAYS days)"
        done
    fi
    
    # Pipeline data cache
    if [[ -d "pipeline_data" ]]; then
        find pipeline_data -name "packet_*" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old pipeline packet (>$KEEP_DAYS days)"
        done
    fi
    
    # User cache directories
    if [[ -d "$HOME/.cache/benchmark" ]]; then
        safe_remove "$HOME/.cache/benchmark" "User cache directory"
    fi
}

# Function to clean old result files
clean_result_files() {
    log_info "Cleaning old result files..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -d "results" ]]; then
        # Keep recent results but clean old ones
        find results -name "*.json" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old result file (>$KEEP_DAYS days)"
        done
        
        find results -name "*.csv" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old result file (>$KEEP_DAYS days)"
        done
        
        find results -name "*.html" -type f -mtime +$KEEP_DAYS | while read -r file; do
            safe_remove "$file" "Old result file (>$KEEP_DAYS days)"
        done
        
        # Clean empty session directories
        find results -type d -name "session_*" -empty | while read -r dir; do
            safe_remove "$dir" "Empty session directory"
        done
    fi
}

# Function to clean test artifacts
clean_test_artifacts() {
    log_info "Cleaning test artifacts..."
    
    cd "$PROJECT_ROOT"
    
    # Test result files
    local test_files=(
        "test-results"
        "test_results"
        "junit.xml"
        "coverage.xml"
        "validation_report.txt"
        ".coverage"
        "htmlcov"
    )
    
    for item in "${test_files[@]}"; do
        if [[ -e "$item" ]]; then
            safe_remove "$item" "Test artifact"
        fi
    done
    
    # Temporary test files
    find . -name "test_*.tmp" -type f | while read -r file; do
        safe_remove "$file" "Temporary test file"
    done
    
    find . -name "*_test.tmp" -type f | while read -r file; do
        safe_remove "$file" "Temporary test file"
    done
}

# Function to clean Docker resources
clean_docker_resources() {
    if ! command -v docker >/dev/null 2>&1; then
        log_warn "Docker not installed, skipping Docker cleanup"
        return
    fi
    
    log_info "Cleaning Docker resources..."
    
    cd "$PROJECT_ROOT"
    
    # Stop containers
    if docker ps -q --filter "name=benchmark" | grep -q .; then
        if confirm_action "Stop running benchmark containers?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                docker stop $(docker ps -q --filter "name=benchmark")
                log_info "Stopped benchmark containers"
            else
                log_info "Would stop benchmark containers"
            fi
        fi
    fi
    
    # Remove containers
    if docker ps -aq --filter "name=benchmark" | grep -q .; then
        if confirm_action "Remove benchmark containers?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                docker rm $(docker ps -aq --filter "name=benchmark")
                log_info "Removed benchmark containers"
            else
                log_info "Would remove benchmark containers"
            fi
        fi
    fi
    
    # Remove images
    if docker images -q --filter "reference=*benchmark*" | grep -q .; then
        if confirm_action "Remove benchmark Docker images?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                docker rmi $(docker images -q --filter "reference=*benchmark*")
                log_info "Removed benchmark images"
            else
                log_info "Would remove benchmark images"
            fi
        fi
    fi
    
    # Clean Docker build cache
    if confirm_action "Clean Docker build cache?"; then
        if [[ "$DRY_RUN" != "true" ]]; then
            docker builder prune -f
            log_info "Cleaned Docker build cache"
        else
            log_info "Would clean Docker build cache"
        fi
    fi
    
    # Clean unused volumes
    if docker volume ls -q --filter "name=benchmark" | grep -q .; then
        if confirm_action "Remove benchmark Docker volumes?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                docker volume rm $(docker volume ls -q --filter "name=benchmark")
                log_info "Removed benchmark volumes"
            else
                log_info "Would remove benchmark volumes"
            fi
        fi
    fi
}

# Function to clean virtual environment
clean_virtual_environment() {
    log_info "Cleaning virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    local venv_dirs=(
        "benchmark-venv"
        "venv"
        ".venv"
        "env"
    )
    
    for venv_dir in "${venv_dirs[@]}"; do
        if [[ -d "$venv_dir" ]]; then
            if confirm_action "Remove virtual environment: $venv_dir?"; then
                safe_remove "$venv_dir" "Virtual environment"
            fi
        fi
    done
    
    # Clean pip cache
    if command -v pip >/dev/null 2>&1; then
        if confirm_action "Clear pip cache?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                pip cache purge 2>/dev/null || true
                log_info "Cleared pip cache"
            else
                log_info "Would clear pip cache"
            fi
        fi
    fi
}

# Function to clean system resources
clean_system_resources() {
    log_info "Cleaning system resources..."
    
    # Kill any running benchmark processes
    local benchmark_processes=$(pgrep -f "benchmark" || true)
    if [[ -n "$benchmark_processes" ]]; then
        if confirm_action "Kill running benchmark processes?"; then
            if [[ "$DRY_RUN" != "true" ]]; then
                pkill -f "benchmark" || true
                log_info "Killed benchmark processes"
            else
                log_info "Would kill benchmark processes"
            fi
        fi
    fi
    
    # Clean shared memory
    if [[ -d "/dev/shm" ]]; then
        find /dev/shm -name "*benchmark*" -user "$(whoami)" 2>/dev/null | while read -r file; do
            safe_remove "$file" "Shared memory file"
        done
    fi
}

# Function to show cleanup summary
show_cleanup_summary() {
    log_info "Cleanup Summary"
    echo "==============="
    
    cd "$PROJECT_ROOT"
    
    # Calculate disk usage
    local current_size=$(du -sh . 2>/dev/null | cut -f1)
    echo "Current project size: $current_size"
    
    # Count files by type
    local python_files=$(find . -name "*.py" -type f | wc -l)
    local log_files=$(find . -name "*.log*" -type f | wc -l)
    local cache_dirs=$(find . -type d -name "__pycache__" | wc -l)
    local result_files=$(find results -name "*.json" -type f 2>/dev/null | wc -l || echo "0")
    
    echo "Python files: $python_files"
    echo "Log files: $log_files"
    echo "Cache directories: $cache_dirs"
    echo "Result files: $result_files"
    
    # Docker info
    if command -v docker >/dev/null 2>&1; then
        local docker_containers=$(docker ps -aq --filter "name=benchmark" | wc -l)
        local docker_images=$(docker images -q --filter "reference=*benchmark*" | wc -l)
        echo "Docker containers: $docker_containers"
        echo "Docker images: $docker_images"
    fi
    
    echo "==============="
}

# Main function
main() {
    log_info "AI News Trading Benchmark - Cleanup Script"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --deep)
                DEEP_CLEAN=true
                shift
                ;;
            --keep-days)
                KEEP_DAYS="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN MODE - No files will actually be removed"
    fi
    
    log_info "Keep files newer than: $KEEP_DAYS days"
    log_info "Deep clean: $DEEP_CLEAN"
    echo ""
    
    # Show current state
    show_cleanup_summary
    echo ""
    
    # Confirm operation
    if [[ "$DRY_RUN" != "true" ]]; then
        if ! confirm_action "Proceed with cleanup?"; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
    
    # Perform cleanup operations
    clean_temp_files
    clean_log_files
    clean_cache_files
    clean_result_files
    clean_test_artifacts
    
    if [[ "$DEEP_CLEAN" == "true" ]]; then
        clean_system_resources
        clean_docker_resources
        clean_virtual_environment
    fi
    
    echo ""
    log_info "Cleanup completed!"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo ""
        show_cleanup_summary
    fi
    
    echo ""
    log_info "Cleanup operations completed successfully"
    
    if [[ "$DEEP_CLEAN" == "true" && "$DRY_RUN" != "true" ]]; then
        echo ""
        log_warn "Deep clean performed. You may need to run setup_environment.sh again."
    fi
}

# Run main function
main "$@"