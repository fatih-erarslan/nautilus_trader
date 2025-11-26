#!/bin/bash
set -e

# NPM Publish Script - Publishes all NPM packages
# Usage: ./publish-npm.sh [--dry-run] [--skip-build]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=false
SKIP_BUILD=false
PUBLISH_LOG="$PROJECT_ROOT/publish-npm.log"

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
        --skip-build)
            SKIP_BUILD=true
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
echo "NPM Publishing started at $(date)" > "$PUBLISH_LOG"

# Check npm authentication
check_npm_auth() {
    log_info "Checking npm authentication..."
    if ! npm whoami &>/dev/null; then
        log_error "Not logged in to npm. Run 'npm login' first."
        exit 1
    fi
    local npm_user=$(npm whoami)
    log_success "Logged in as: $npm_user"
}

# Function to get package version
get_package_version() {
    local package_json=$1
    node -p "require('$package_json').version"
}

# Function to check if package version is already published
is_package_published() {
    local package_name=$1
    local version=$2

    if npm view "$package_name@$version" version &>/dev/null; then
        return 0  # Already published
    else
        return 1  # Not published
    fi
}

# Function to build NAPI binaries
build_napi_binaries() {
    log_info "Building NAPI binaries for all platforms..."

    cd "$PROJECT_ROOT"

    if [ "$SKIP_BUILD" = true ]; then
        log_warning "Skipping NAPI build (--skip-build flag)"
        return 0
    fi

    # Build for current platform first
    log_info "Building for current platform..."
    if ! npm run build; then
        log_error "Failed to build for current platform"
        return 1
    fi

    # Note: Cross-platform builds are handled by GitHub Actions
    # This script builds for the current platform only
    log_success "NAPI binaries built successfully"

    return 0
}

# Function to publish a package
publish_package() {
    local package_path=$1
    local package_name=$2

    log_info "Publishing package: $package_name from $package_path"

    cd "$package_path"

    # Get version
    local version=$(get_package_version "$package_path/package.json")
    log_info "Version: $version"

    # Check if already published
    if is_package_published "$package_name" "$version"; then
        log_warning "Package $package_name@$version is already published. Skipping."
        return 0
    fi

    # Run prepublishOnly if it exists
    if grep -q "prepublishOnly" package.json; then
        log_info "Running prepublishOnly script..."
        if ! npm run prepublishOnly; then
            log_error "prepublishOnly script failed for $package_name"
            return 1
        fi
    fi

    # Verify package contents
    log_info "Verifying package contents..."
    if ! npm pack --dry-run; then
        log_error "Package verification failed for $package_name"
        return 1
    fi

    # Publish (or dry-run)
    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would publish $package_name@$version"
        npm publish --dry-run --access public
    else
        log_info "Publishing $package_name@$version to npm..."
        if npm publish --access public; then
            log_success "Published $package_name@$version"
        else
            log_error "Failed to publish $package_name@$version"
            return 1
        fi
    fi

    cd "$PROJECT_ROOT"
    return 0
}

# Function to create GitHub release
create_github_release() {
    local version=$1
    local tag="v$version"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would create GitHub release $tag"
        return 0
    fi

    log_info "Creating GitHub release $tag..."

    # Check if gh CLI is installed
    if ! command -v gh &>/dev/null; then
        log_warning "GitHub CLI (gh) not installed. Skipping release creation."
        log_info "Install gh CLI: https://cli.github.com/"
        return 0
    fi

    # Check if tag exists
    if ! git rev-parse "$tag" >/dev/null 2>&1; then
        log_error "Git tag $tag does not exist. Create it first with bump-version.sh"
        return 1
    fi

    # Extract changelog for this version
    local changelog_entry=""
    if [ -f "$PROJECT_ROOT/CHANGELOG.md" ]; then
        changelog_entry=$(awk "/^## \[$version\]/,/^## \[/" "$PROJECT_ROOT/CHANGELOG.md" | sed '$ d')
    fi

    # Create release
    if gh release create "$tag" \
        --title "Release $version" \
        --notes "$changelog_entry" \
        --draft=false \
        --prerelease=false; then
        log_success "Created GitHub release $tag"
    else
        log_error "Failed to create GitHub release"
        return 1
    fi

    return 0
}

# Main publishing flow
main() {
    log_info "Starting NPM packages publishing process"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Dry run: $DRY_RUN"
    log_info "Skip build: $SKIP_BUILD"

    # Check npm authentication
    check_npm_auth

    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_warning "There are uncommitted changes in the repository."
        if [ "$DRY_RUN" = false ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_error "Aborting due to uncommitted changes."
                exit 1
            fi
        fi
    fi

    # Build NAPI binaries
    if ! build_napi_binaries; then
        log_error "Failed to build NAPI binaries"
        exit 1
    fi

    # Define packages in dependency order
    declare -a PACKAGES=(
        "packages/mcp-protocol:@neural-trader/mcp-protocol"
        "packages/mcp:@neural-trader/mcp"
        "packages/neural-trader:neural-trader"
    )

    local failed_packages=()
    local published_packages=()

    # Publish each package
    for package_info in "${PACKAGES[@]}"; do
        IFS=':' read -r package_path package_name <<< "$package_info"

        log_info "=========================================="
        log_info "Processing: $package_name"
        log_info "=========================================="

        if publish_package "$PROJECT_ROOT/$package_path" "$package_name"; then
            published_packages+=("$package_name")
        else
            failed_packages+=("$package_name")
            log_error "Failed to publish $package_name"

            # Ask if we should continue
            if [ "$DRY_RUN" = false ]; then
                read -p "Continue with remaining packages? (y/N) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_error "Aborting remaining publications."
                    break
                fi
            fi
        fi
    done

    # Create GitHub release
    if [ ${#failed_packages[@]} -eq 0 ]; then
        local version=$(get_package_version "$PROJECT_ROOT/package.json")
        create_github_release "$version"
    fi

    # Summary
    log_info "=========================================="
    log_info "Publishing Summary"
    log_info "=========================================="
    log_info "Published packages: ${#published_packages[@]}"
    for package in "${published_packages[@]}"; do
        log_success "  ✓ $package"
    done

    if [ ${#failed_packages[@]} -gt 0 ]; then
        log_error "Failed packages: ${#failed_packages[@]}"
        for package in "${failed_packages[@]}"; do
            log_error "  ✗ $package"
        done
        exit 1
    else
        log_success "All packages published successfully!"
        log_info "Log file: $PUBLISH_LOG"
    fi
}

# Run main function
main
