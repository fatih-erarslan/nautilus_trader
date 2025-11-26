#!/bin/bash
set -e

# Version Management Script - Bump versions across all packages
# Usage: ./bump-version.sh <new-version> [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
NEW_VERSION=$1
if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo "Usage: $0 <new-version> [--dry-run]"
    echo "Example: $0 1.0.7"
    exit 1
fi

if [ "$2" = "--dry-run" ]; then
    DRY_RUN=true
fi

# Validate version format (semver)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.-]+)?$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version must follow semver format: MAJOR.MINOR.PATCH[-PRERELEASE]"
    exit 1
fi

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

# Function to update Cargo.toml version
update_cargo_toml() {
    local file=$1
    local version=$2

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would update $file to version $version"
        return 0
    fi

    if [ -f "$file" ]; then
        # Update version field
        sed -i.bak "s/^version = \".*\"/version = \"$version\"/" "$file"

        # Update internal dependencies (nt-* crates)
        sed -i.bak "s/\(nt-[a-z-]*\).*version = \"[^\"]*\"/\1\", version = \"$version\"/" "$file"

        rm -f "$file.bak"
        log_success "Updated $file"
    else
        log_warning "File not found: $file"
    fi
}

# Function to update package.json version
update_package_json() {
    local file=$1
    local version=$2

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would update $file to version $version"
        return 0
    fi

    if [ -f "$file" ]; then
        # Use node to update JSON properly
        node -e "
            const fs = require('fs');
            const pkg = JSON.parse(fs.readFileSync('$file', 'utf8'));
            pkg.version = '$version';
            fs.writeFileSync('$file', JSON.stringify(pkg, null, 2) + '\n');
        "
        log_success "Updated $file"
    else
        log_warning "File not found: $file"
    fi
}

# Function to generate CHANGELOG entry
generate_changelog() {
    local version=$1
    local date=$(date +%Y-%m-%d)
    local changelog_file="$PROJECT_ROOT/CHANGELOG.md"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would update CHANGELOG.md"
        return 0
    fi

    # Get commits since last tag
    local last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    local commits=""

    if [ -n "$last_tag" ]; then
        commits=$(git log --pretty=format:"- %s" "$last_tag"..HEAD)
    else
        commits=$(git log --pretty=format:"- %s" HEAD)
    fi

    # Create temporary file with new entry
    cat > /tmp/changelog_entry.md << EOF
## [$version] - $date

### Changes
$commits

EOF

    # Prepend to existing CHANGELOG or create new one
    if [ -f "$changelog_file" ]; then
        cat /tmp/changelog_entry.md "$changelog_file" > /tmp/changelog_new.md
        mv /tmp/changelog_new.md "$changelog_file"
    else
        cat > "$changelog_file" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

EOF
        cat /tmp/changelog_entry.md >> "$changelog_file"
    fi

    rm -f /tmp/changelog_entry.md
    log_success "Updated CHANGELOG.md"
}

# Function to create git tag
create_git_tag() {
    local version=$1
    local tag="v$version"

    if [ "$DRY_RUN" = true ]; then
        log_info "DRY RUN: Would create git tag $tag"
        return 0
    fi

    # Check if tag already exists
    if git rev-parse "$tag" >/dev/null 2>&1; then
        log_warning "Tag $tag already exists"
        read -p "Replace existing tag? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git tag -d "$tag"
            log_info "Deleted existing tag $tag"
        else
            return 0
        fi
    fi

    # Create annotated tag
    git tag -a "$tag" -m "Release $version"
    log_success "Created git tag $tag"
}

# Main version bump flow
main() {
    log_info "Starting version bump to $NEW_VERSION"
    log_info "Project root: $PROJECT_ROOT"
    log_info "Dry run: $DRY_RUN"

    # Check for uncommitted changes
    if [ "$DRY_RUN" = false ] && ! git diff-index --quiet HEAD --; then
        log_warning "There are uncommitted changes in the repository."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_error "Aborting due to uncommitted changes."
            exit 1
        fi
    fi

    # Update all Cargo.toml files
    log_info "Updating Cargo.toml files..."
    find "$PROJECT_ROOT/crates" -name "Cargo.toml" -type f | while read -r cargo_file; do
        update_cargo_toml "$cargo_file" "$NEW_VERSION"
    done

    # Update workspace Cargo.toml
    update_cargo_toml "$PROJECT_ROOT/Cargo.toml" "$NEW_VERSION"

    # Update all package.json files
    log_info "Updating package.json files..."
    update_package_json "$PROJECT_ROOT/packages/neural-trader/package.json" "$NEW_VERSION"
    update_package_json "$PROJECT_ROOT/packages/mcp/package.json" "$NEW_VERSION"
    update_package_json "$PROJECT_ROOT/packages/mcp-protocol/package.json" "$NEW_VERSION"
    update_package_json "$PROJECT_ROOT/package.json" "$NEW_VERSION"

    # Update root package-lock.json
    if [ "$DRY_RUN" = false ]; then
        log_info "Updating package-lock.json..."
        cd "$PROJECT_ROOT"
        npm install --package-lock-only
    fi

    # Generate CHANGELOG
    log_info "Updating CHANGELOG.md..."
    generate_changelog "$NEW_VERSION"

    # Create git tag
    log_info "Creating git tag..."
    create_git_tag "$NEW_VERSION"

    # Summary
    log_info "=========================================="
    log_info "Version Bump Summary"
    log_info "=========================================="
    log_info "New version: $NEW_VERSION"

    if [ "$DRY_RUN" = false ]; then
        log_success "All files updated successfully!"
        log_info ""
        log_info "Next steps:"
        log_info "1. Review changes: git diff"
        log_info "2. Commit changes: git add -A && git commit -m 'chore: bump version to $NEW_VERSION'"
        log_info "3. Push tag: git push origin v$NEW_VERSION"
        log_info "4. Run publish scripts"
    else
        log_info "Dry run completed. No changes were made."
    fi
}

# Run main function
main
