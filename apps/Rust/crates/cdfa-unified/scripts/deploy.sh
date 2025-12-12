#!/bin/bash

# CDFA Unified Deployment Script
# Automated deployment for various environments and platforms

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/target"
DIST_DIR="$PROJECT_ROOT/dist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
ENVIRONMENT="local"
VERSION=""
REGISTRY=""
BUILD_FIRST=true
FORCE=false
DRY_RUN=false
VERBOSE=""
BACKUP=true
HEALTH_CHECK=true

# Deployment environments
ENVIRONMENTS=(
    "local"          # Local development
    "staging"        # Staging environment
    "production"     # Production deployment
    "docker"         # Docker container deployment
    "kubernetes"     # Kubernetes cluster deployment
    "pypi"          # Python package registry
    "crates"        # Crates.io registry
    "github"        # GitHub releases
    "npm"           # NPM registry (if applicable)
)

# Help function
show_help() {
    cat << EOF
CDFA Unified Deployment Script

Usage: $0 [OPTIONS] ENVIRONMENT

ENVIRONMENTS:
    local           Deploy to local development environment
    staging         Deploy to staging environment  
    production      Deploy to production environment
    docker          Build and deploy Docker image
    kubernetes      Deploy to Kubernetes cluster
    pypi           Publish Python package to PyPI
    crates         Publish to crates.io
    github         Create GitHub release
    npm            Publish to NPM (if applicable)

OPTIONS:
    -v, --version VERSION     Version for release/deployment [required for releases]
    -r, --registry REGISTRY   Custom registry URL
    -B, --no-build           Skip building before deployment
    -f, --force              Force deployment (skip confirmations)
    -d, --dry-run            Show what would be deployed without executing
    -V, --verbose            Verbose output
    -b, --no-backup          Skip creating backup
    -H, --no-health-check    Skip health checks after deployment
    -h, --help               Show this help

EXAMPLES:
    $0 local                                    # Deploy to local environment
    $0 staging -v 0.1.0                       # Deploy version 0.1.0 to staging
    $0 production -v 1.0.0 -f                 # Force deploy to production
    $0 docker -v latest                       # Build and deploy Docker image
    $0 crates -v 0.1.0                        # Publish to crates.io
    $0 pypi -v 0.1.0                          # Publish Python package
    $0 github -v 1.0.0                        # Create GitHub release
    $0 kubernetes -v 1.0.0 --dry-run          # Dry run Kubernetes deployment

EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -B|--no-build)
            BUILD_FIRST=false
            shift
            ;;
        -f|--force)
            FORCE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -V|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -b|--no-backup)
            BACKUP=false
            shift
            ;;
        -H|--no-health-check)
            HEALTH_CHECK=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [[ -z "$ENVIRONMENT" || "$ENVIRONMENT" == "local" ]]; then
                ENVIRONMENT="$1"
            else
                log_error "Multiple environments specified"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate environment argument
if [[ ! " ${ENVIRONMENTS[*]} " =~ " ${ENVIRONMENT} " ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    echo "Valid environments: ${ENVIRONMENTS[*]}"
    exit 1
fi

# Validate version for release environments
case "$ENVIRONMENT" in
    "crates"|"pypi"|"github"|"npm")
        if [[ -z "$VERSION" ]]; then
            log_error "Version is required for $ENVIRONMENT deployment"
            exit 1
        fi
        ;;
esac

# Validate environment and dependencies
check_environment() {
    log_info "Checking deployment environment for $ENVIRONMENT..."
    
    # Common dependencies
    if ! command -v git &> /dev/null; then
        log_error "Git not found. Required for deployment."
        exit 1
    fi
    
    case "$ENVIRONMENT" in
        "docker"|"kubernetes")
            if ! command -v docker &> /dev/null; then
                log_error "Docker not found. Required for container deployment."
                exit 1
            fi
            ;;
        "kubernetes")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl not found. Required for Kubernetes deployment."
                exit 1
            fi
            ;;
        "pypi")
            if ! command -v python3 &> /dev/null; then
                log_error "Python3 not found. Required for PyPI deployment."
                exit 1
            fi
            if ! python3 -c "import twine" &> /dev/null; then
                log_error "Twine not found. Install with: pip install twine"
                exit 1
            fi
            ;;
        "crates")
            if ! command -v cargo &> /dev/null; then
                log_error "Cargo not found. Required for crates.io deployment."
                exit 1
            fi
            ;;
        "github")
            if ! command -v gh &> /dev/null; then
                log_error "GitHub CLI not found. Required for GitHub releases."
                exit 1
            fi
            ;;
        "npm")
            if ! command -v npm &> /dev/null; then
                log_error "NPM not found. Required for NPM deployment."
                exit 1
            fi
            ;;
    esac
    
    log_success "Environment validation passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check for uncommitted changes (except for local deployment)
    if [[ "$ENVIRONMENT" != "local" ]]; then
        if ! git diff-index --quiet HEAD --; then
            if [[ "$FORCE" != true ]]; then
                log_error "Uncommitted changes detected. Use --force to override."
                exit 1
            else
                log_warning "Proceeding with uncommitted changes (--force specified)"
            fi
        fi
    fi
    
    # Check current branch for production deployment
    if [[ "$ENVIRONMENT" == "production" ]]; then
        local current_branch
        current_branch=$(git branch --show-current)
        if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
            if [[ "$FORCE" != true ]]; then
                log_error "Production deployment must be from main/master branch. Current: $current_branch"
                exit 1
            else
                log_warning "Deploying from non-main branch: $current_branch"
            fi
        fi
    fi
    
    # Version validation for releases
    if [[ -n "$VERSION" ]]; then
        if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$ ]]; then
            log_error "Invalid version format: $VERSION (expected: X.Y.Z or X.Y.Z-suffix)"
            exit 1
        fi
        
        # Check if version tag already exists
        if git tag -l | grep -q "^v$VERSION$"; then
            if [[ "$FORCE" != true ]]; then
                log_error "Version tag v$VERSION already exists. Use --force to override."
                exit 1
            else
                log_warning "Version tag v$VERSION already exists (--force specified)"
            fi
        fi
    fi
    
    log_success "Pre-deployment checks passed"
}

# Build project if needed
build_project() {
    if [[ "$BUILD_FIRST" == true ]]; then
        log_info "Building project before deployment..."
        
        local build_script="$SCRIPT_DIR/build.sh"
        local build_args=()
        
        case "$ENVIRONMENT" in
            "local"|"staging")
                build_args+=("--profile" "debug")
                ;;
            "production"|"crates"|"github")
                build_args+=("--profile" "release")
                build_args+=("--features" "full-performance")
                ;;
            "docker"|"kubernetes")
                build_args+=("--profile" "release")
                build_args+=("--features" "full-performance")
                ;;
            "pypi")
                build_args+=("--python")
                build_args+=("--profile" "release")
                build_args+=("--features" "python")
                ;;
        esac
        
        if [[ -n "$VERBOSE" ]]; then
            build_args+=("$VERBOSE")
        fi
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would run: $build_script ${build_args[*]}"
        else
            "$build_script" "${build_args[@]}"
        fi
        
        log_success "Build completed"
    fi
}

# Create backup
create_backup() {
    if [[ "$BACKUP" == true && "$ENVIRONMENT" != "local" ]]; then
        log_info "Creating deployment backup..."
        
        local timestamp
        timestamp=$(date +"%Y%m%d_%H%M%S")
        local backup_dir="$PROJECT_ROOT/backups/$ENVIRONMENT/$timestamp"
        
        if [[ "$DRY_RUN" == true ]]; then
            log_info "Would create backup at: $backup_dir"
        else
            mkdir -p "$backup_dir"
            
            # Backup current deployment state
            case "$ENVIRONMENT" in
                "staging"|"production")
                    # Backup configuration files, current version info, etc.
                    echo "$(date): Deployment to $ENVIRONMENT v$VERSION" > "$backup_dir/deployment.log"
                    git log -1 --oneline > "$backup_dir/commit.txt"
                    ;;
                "docker"|"kubernetes")
                    # Backup current container/pod configurations
                    if command -v docker &> /dev/null; then
                        docker images cdfa-unified --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}" > "$backup_dir/docker_images.txt" 2>/dev/null || true
                    fi
                    ;;
            esac
            
            log_success "Backup created at: $backup_dir"
        fi
    fi
}

# Deploy to local environment
deploy_local() {
    log_info "Deploying to local environment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would install locally built binaries"
        return
    fi
    
    # Install locally built binaries
    cargo install --path . --force
    
    log_success "Local deployment completed"
}

# Deploy to staging environment
deploy_staging() {
    log_info "Deploying to staging environment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would deploy to staging servers"
        return
    fi
    
    # Example staging deployment (customize based on your infrastructure)
    log_warning "Staging deployment not fully implemented. Customize for your infrastructure."
    
    log_success "Staging deployment completed"
}

# Deploy to production environment
deploy_production() {
    log_info "Deploying to production environment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would deploy to production servers"
        return
    fi
    
    # Example production deployment (customize based on your infrastructure)
    log_warning "Production deployment not fully implemented. Customize for your infrastructure."
    
    log_success "Production deployment completed"
}

# Deploy Docker image
deploy_docker() {
    log_info "Building and deploying Docker image..."
    
    local image_tag="cdfa-unified:${VERSION:-latest}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would build Docker image: $image_tag"
        if [[ -n "$REGISTRY" ]]; then
            log_info "Would push to registry: $REGISTRY/$image_tag"
        fi
        return
    fi
    
    # Build Docker image
    docker build -t "$image_tag" "$PROJECT_ROOT"
    
    # Tag for registry if specified
    if [[ -n "$REGISTRY" ]]; then
        docker tag "$image_tag" "$REGISTRY/$image_tag"
        docker push "$REGISTRY/$image_tag"
        log_success "Docker image pushed to $REGISTRY/$image_tag"
    else
        log_success "Docker image built: $image_tag"
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would deploy to Kubernetes cluster"
        return
    fi
    
    # Check if Kubernetes manifests exist
    local k8s_dir="$PROJECT_ROOT/k8s"
    if [[ ! -d "$k8s_dir" ]]; then
        log_error "Kubernetes manifests directory not found: $k8s_dir"
        exit 1
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f "$k8s_dir/"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/cdfa-unified
    
    log_success "Kubernetes deployment completed"
}

# Publish to crates.io
deploy_crates() {
    log_info "Publishing to crates.io..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would publish version $VERSION to crates.io"
        return
    fi
    
    # Update version in Cargo.toml
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$PROJECT_ROOT/Cargo.toml"
    
    # Publish to crates.io
    cargo publish --allow-dirty
    
    # Restore Cargo.toml backup
    mv "$PROJECT_ROOT/Cargo.toml.bak" "$PROJECT_ROOT/Cargo.toml"
    
    log_success "Published version $VERSION to crates.io"
}

# Publish to PyPI
deploy_pypi() {
    log_info "Publishing to PyPI..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would publish version $VERSION to PyPI"
        return
    fi
    
    # Build Python package
    cd "$PROJECT_ROOT"
    python3 setup.py sdist bdist_wheel
    
    # Upload to PyPI
    twine upload dist/*
    
    log_success "Published version $VERSION to PyPI"
}

# Create GitHub release
deploy_github() {
    log_info "Creating GitHub release..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would create GitHub release v$VERSION"
        return
    fi
    
    # Create git tag
    git tag -a "v$VERSION" -m "Release version $VERSION"
    git push origin "v$VERSION"
    
    # Create GitHub release
    gh release create "v$VERSION" \
        --title "CDFA Unified v$VERSION" \
        --notes "Release version $VERSION" \
        --target "$(git rev-parse HEAD)"
    
    # Upload release artifacts if they exist
    local artifacts_dir="$DIST_DIR"
    if [[ -d "$artifacts_dir" ]]; then
        gh release upload "v$VERSION" "$artifacts_dir"/*
    fi
    
    log_success "GitHub release v$VERSION created"
}

# Publish to NPM
deploy_npm() {
    log_info "Publishing to NPM..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Would publish version $VERSION to NPM"
        return
    fi
    
    # Check if package.json exists
    if [[ ! -f "$PROJECT_ROOT/package.json" ]]; then
        log_error "package.json not found. NPM deployment requires Node.js package configuration."
        exit 1
    fi
    
    # Update version in package.json
    npm version "$VERSION" --no-git-tag-version
    
    # Publish to NPM
    npm publish
    
    log_success "Published version $VERSION to NPM"
}

# Run health checks
run_health_checks() {
    if [[ "$HEALTH_CHECK" == true && "$ENVIRONMENT" != "local" && "$DRY_RUN" != true ]]; then
        log_info "Running post-deployment health checks..."
        
        case "$ENVIRONMENT" in
            "staging"|"production")
                # Custom health check implementation
                log_warning "Health checks not implemented for $ENVIRONMENT"
                ;;
            "docker")
                # Check if container is running
                if docker ps | grep -q "cdfa-unified"; then
                    log_success "Docker container is running"
                else
                    log_warning "Docker container not found in running state"
                fi
                ;;
            "kubernetes")
                # Check deployment status
                if kubectl get deployment cdfa-unified -o jsonpath='{.status.readyReplicas}' | grep -q "[1-9]"; then
                    log_success "Kubernetes deployment is healthy"
                else
                    log_warning "Kubernetes deployment may not be ready"
                fi
                ;;
            "crates"|"pypi"|"github"|"npm")
                log_info "Release published successfully - no additional health checks needed"
                ;;
        esac
    fi
}

# Print deployment summary
print_summary() {
    echo
    log_success "=== DEPLOYMENT SUMMARY ==="
    log_info "Environment: $ENVIRONMENT"
    
    if [[ -n "$VERSION" ]]; then
        log_info "Version: $VERSION"
    fi
    
    if [[ -n "$REGISTRY" ]]; then
        log_info "Registry: $REGISTRY"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "Mode: Dry run (no changes made)"
    fi
    
    case "$ENVIRONMENT" in
        "docker")
            log_info "Docker image: cdfa-unified:${VERSION:-latest}"
            ;;
        "github")
            log_info "GitHub release: v$VERSION"
            ;;
        "crates"|"pypi"|"npm")
            log_info "Package version: $VERSION"
            ;;
    esac
    
    echo
    log_success "Deployment completed successfully! 🚀"
}

# Main deployment function
main() {
    log_info "Starting CDFA Unified deployment to $ENVIRONMENT..."
    
    cd "$PROJECT_ROOT"
    
    check_environment
    pre_deployment_checks
    create_backup
    build_project
    
    # Execute environment-specific deployment
    case "$ENVIRONMENT" in
        "local")
            deploy_local
            ;;
        "staging")
            deploy_staging
            ;;
        "production")
            deploy_production
            ;;
        "docker")
            deploy_docker
            ;;
        "kubernetes")
            deploy_kubernetes
            ;;
        "crates")
            deploy_crates
            ;;
        "pypi")
            deploy_pypi
            ;;
        "github")
            deploy_github
            ;;
        "npm")
            deploy_npm
            ;;
        *)
            log_error "Deployment not implemented for environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    run_health_checks
    print_summary
}

# Execute main function
main "$@"