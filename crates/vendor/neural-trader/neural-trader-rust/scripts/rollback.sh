#!/bin/bash
set -e

# Rollback Script - Yank published versions and revert changes
# Usage: ./rollback.sh <version> [--confirm]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIRM=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
VERSION=$1
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: $0 <version> [--confirm]"
    echo "Example: $0 1.0.7"
    exit 1
fi

if [ "$2" = "--confirm" ]; then
    CONFIRM=true
fi

# Remove 'v' prefix if present
VERSION=${VERSION#v}

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

# Function to yank a crate from crates.io
yank_crate() {
    local crate_name=$1
    local version=$2

    log_info "Yanking crate: $crate_name@$version"

    if cargo yank --vers "$version" "$crate_name"; then
        log_success "Yanked $crate_name@$version from crates.io"
        return 0
    else
        log_error "Failed to yank $crate_name@$version"
        return 1
    fi
}

# Function to deprecate an npm package version
deprecate_npm_package() {
    local package_name=$1
    local version=$2

    log_info "Deprecating npm package: $package_name@$version"

    if npm deprecate "$package_name@$version" "This version has been rolled back due to issues. Please use a newer version."; then
        log_success "Deprecated $package_name@$version on npm"
        return 0
    else
        log_error "Failed to deprecate $package_name@$version"
        return 1
    fi
}

# Function to delete GitHub release
delete_github_release() {
    local tag="v$1"

    log_info "Deleting GitHub release: $tag"

    # Check if gh CLI is installed
    if ! command -v gh &>/dev/null; then
        log_warning "GitHub CLI (gh) not installed. Skipping release deletion."
        log_info "Install gh CLI: https://cli.github.com/"
        return 0
    fi

    if gh release delete "$tag" --yes; then
        log_success "Deleted GitHub release $tag"
    else
        log_warning "Failed to delete GitHub release $tag (may not exist)"
    fi

    # Delete the git tag
    if git rev-parse "$tag" >/dev/null 2>&1; then
        git tag -d "$tag"
        log_info "Deleted local git tag $tag"

        if git push origin ":refs/tags/$tag" 2>/dev/null; then
            log_success "Deleted remote git tag $tag"
        else
            log_warning "Failed to delete remote tag $tag"
        fi
    fi
}

# Function to revert git commits
revert_git_commits() {
    local tag="v$1"

    log_info "Reverting git commits to before $tag"

    # Find the commit before the tag
    local tag_commit=$(git rev-list -n 1 "$tag" 2>/dev/null || echo "")

    if [ -z "$tag_commit" ]; then
        log_warning "Tag $tag not found. Cannot revert commits."
        return 0
    fi

    local previous_commit=$(git rev-list -n 1 "$tag_commit^" 2>/dev/null || echo "")

    if [ -z "$previous_commit" ]; then
        log_error "Cannot find commit before $tag"
        return 1
    fi

    log_info "Reverting to commit: $previous_commit"

    if [ "$CONFIRM" = true ]; then
        git reset --hard "$previous_commit"
        log_success "Reverted to commit $previous_commit"
    else
        log_info "Would revert to commit $previous_commit (use --confirm to execute)"
    fi
}

# Main rollback flow
main() {
    log_warning "=========================================="
    log_warning "ROLLBACK PROCEDURE FOR VERSION $VERSION"
    log_warning "=========================================="

    if [ "$CONFIRM" = false ]; then
        log_warning "DRY RUN MODE - No changes will be made"
        log_info "Use --confirm flag to execute rollback"
    fi

    # Confirm with user
    if [ "$CONFIRM" = true ]; then
        echo
        log_error "This will:"
        echo "  1. Yank all Rust crates version $VERSION from crates.io"
        echo "  2. Deprecate all NPM packages version $VERSION"
        echo "  3. Delete GitHub release v$VERSION"
        echo "  4. Delete git tag v$VERSION"
        echo "  5. Revert git commits to before the release"
        echo
        read -p "Are you absolutely sure? Type 'ROLLBACK' to confirm: " -r
        echo
        if [ "$REPLY" != "ROLLBACK" ]; then
            log_error "Rollback cancelled."
            exit 1
        fi
    fi

    # Define crates to yank
    declare -a CRATES=(
        "nt-core"
        "nt-strategies"
        "nt-neural"
        "nt-risk"
        "nt-portfolio"
        "nt-backtest"
        "nt-execution"
        "nt-data"
        "nt-sentiment"
        "nt-prediction"
        "nt-sports"
        "nt-syndicate"
        "nt-e2b"
        "nt-napi"
    )

    # Define npm packages to deprecate
    declare -a NPM_PACKAGES=(
        "@neural-trader/mcp-protocol"
        "@neural-trader/mcp"
        "neural-trader"
    )

    local failed_operations=()

    # Yank Rust crates
    log_info "=========================================="
    log_info "Yanking Rust Crates"
    log_info "=========================================="

    if [ "$CONFIRM" = true ]; then
        for crate in "${CRATES[@]}"; do
            if ! yank_crate "$crate" "$VERSION"; then
                failed_operations+=("crate: $crate")
            fi
        done
    else
        for crate in "${CRATES[@]}"; do
            log_info "Would yank: $crate@$VERSION"
        done
    fi

    # Deprecate NPM packages
    log_info "=========================================="
    log_info "Deprecating NPM Packages"
    log_info "=========================================="

    if [ "$CONFIRM" = true ]; then
        for package in "${NPM_PACKAGES[@]}"; do
            if ! deprecate_npm_package "$package" "$VERSION"; then
                failed_operations+=("npm: $package")
            fi
        done
    else
        for package in "${NPM_PACKAGES[@]}"; do
            log_info "Would deprecate: $package@$VERSION"
        done
    fi

    # Delete GitHub release
    log_info "=========================================="
    log_info "Deleting GitHub Release"
    log_info "=========================================="

    if [ "$CONFIRM" = true ]; then
        delete_github_release "$VERSION"
    else
        log_info "Would delete GitHub release v$VERSION"
    fi

    # Revert git commits
    log_info "=========================================="
    log_info "Reverting Git Commits"
    log_info "=========================================="

    revert_git_commits "$VERSION"

    # Summary
    log_info "=========================================="
    log_info "Rollback Summary"
    log_info "=========================================="

    if [ ${#failed_operations[@]} -gt 0 ]; then
        log_error "Failed operations: ${#failed_operations[@]}"
        for op in "${failed_operations[@]}"; do
            log_error "  ✗ $op"
        done
        exit 1
    else
        if [ "$CONFIRM" = true ]; then
            log_success "Rollback completed successfully!"
            log_info ""
            log_info "Next steps:"
            log_info "1. Fix the issues in the codebase"
            log_info "2. Bump to a new version"
            log_info "3. Re-publish with the fixes"
        else
            log_info "Dry run completed. Use --confirm to execute."
        fi
    fi
}

# Run main function
main
