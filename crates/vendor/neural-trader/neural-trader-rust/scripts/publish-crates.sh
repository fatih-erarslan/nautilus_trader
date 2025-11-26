#!/bin/bash
set -e

# Cargo Publish Script - Publishes all crates in dependency order
# Usage: ./publish-crates.sh [--dry-run] [--skip-tests]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=false
SKIP_TESTS=false
PUBLISH_LOG="$PROJECT_ROOT/publish-crates.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        *)
            ;;
    esac
done

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$PUBLISH_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$PUBLISH_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$PUBLISH_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$PUBLISH_LOG"
}

# Initialize log file
echo "Publishing started at $(date)" > "$PUBLISH_LOG"

# Check if we're in a git repository
if [ ! -d "$PROJECT_ROOT/.git" ]; then
    log_error "Not in a git repository. Aborting."
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    log_warning "There are uncommitted changes in the repository."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_error "Aborting due to uncommitted changes."
        exit 1
    fi
fi

# Function to get crate version
get_crate_version() {
    local crate_path=$1
    grep -m1 '^version' "$crate_path/Cargo.toml" | sed 's/version = "\(.*\)"/\1/' | tr -d ' '
}

# Function to check if crate is already published
is_crate_published() {
    local crate_name=$1
    local version=$2

    if cargo search "$crate_name" --limit 1 | grep -q "^$crate_name = \"$version\""; then
        return 0  # Already published
    else
        return 1  # Not published
    fi
}

# Function to publish a crate
publish_crate() {
    local crate_path=$1
    local crate_name=$2
    local wait_time=${3:-30}

    log_info "Publishing crate: $crate_name from $crate_path"

    cd "$crate_path"

    # Get version
    local version=$(get_crate_version "$crate_path")
    log_info "Version: $version"

    # Check if already published
    if is_crate_published "$crate_name" "$version"; then
        log_warning "Crate $crate_name@$version is already published. Skipping."
        return 0
    fi

    # Run tests unless skipped
    if [ "$SKIP_TESTS" = false ]; then
        log_info "Running tests for $crate_name..."
        if ! cargo test --release; then
            log_error "Tests failed for $crate_name. Aborting."
            return 1
        fi
        log_success "Tests passed for $crate_name"
    fi

    # Check package contents
    log_info "Checking package contents for $crate_name..."
    if ! cargo package --list; then
        log_error "Package check failed for $crate_name. Aborting."
        return 1
    fi

    # Publish (or dry-run)
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would publish $crate_name@$version"
        cargo publish --dry-run
    else
        log_info "Publishing $crate_name@$version to crates.io..."
        if cargo publish --no-verify; then
            log_success "Published $crate_name@$version"

            # Wait for crates.io index update
            log_info "Waiting ${wait_time}s for crates.io index to update..."
            sleep "$wait_time"
        else
            log_error "Failed to publish $crate_name@$version"
            return 1
        fi
    fi

    cd "$PROJECT_ROOT"
    return 0
}

# Main publishing flow
main() {
    log_info "Starting Rust crates publishing process"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Dry run: $DRY_RUN"
    log_info "Skip tests: $SKIP_TESTS"

    # Define crates in dependency order
    # Base crates first, then dependent crates
    declare -a CRATES=(
        "crates/nt-core:nt-core:45"
        "crates/nt-strategies:nt-strategies:45"
        "crates/nt-neural:nt-neural:45"
        "crates/nt-risk:nt-risk:45"
        "crates/nt-portfolio:nt-portfolio:45"
        "crates/nt-backtest:nt-backtest:45"
        "crates/nt-execution:nt-execution:45"
        "crates/nt-data:nt-data:45"
        "crates/nt-sentiment:nt-sentiment:45"
        "crates/nt-prediction:nt-prediction:45"
        "crates/nt-sports:nt-sports:45"
        "crates/nt-syndicate:nt-syndicate:45"
        "crates/nt-e2b:nt-e2b:45"
        "crates/nt-napi:nt-napi:60"
    )

    local failed_crates=()
    local published_crates=()

    # Publish each crate
    for crate_info in "${CRATES[@]}"; do
        IFS=':' read -r crate_path crate_name wait_time <<< "$crate_info"

        log_info "=========================================="
        log_info "Processing: $crate_name"
        log_info "=========================================="

        if publish_crate "$PROJECT_ROOT/$crate_path" "$crate_name" "$wait_time"; then
            published_crates+=("$crate_name")
        else
            failed_crates+=("$crate_name")
            log_error "Failed to publish $crate_name"

            # Ask if we should continue
            if [ "$DRY_RUN" = false ]; then
                read -p "Continue with remaining crates? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_error "Aborting remaining publications."
                    break
                fi
            fi
        fi
    done

    # Summary
    log_info "=========================================="
    log_info "Publishing Summary"
    log_info "=========================================="
    log_info "Published crates: ${#published_crates[@]}"
    for crate in "${published_crates[@]}"; do
        log_success "  ✓ $crate"
    done

    if [ ${#failed_crates[@]} -gt 0 ]; then
        log_error "Failed crates: ${#failed_crates[@]}"
        for crate in "${failed_crates[@]}"; do
            log_error "  ✗ $crate"
        done
        exit 1
    else
        log_success "All crates published successfully!"
        log_info "Log file: $PUBLISH_LOG"
    fi
}

# Run main function
main
