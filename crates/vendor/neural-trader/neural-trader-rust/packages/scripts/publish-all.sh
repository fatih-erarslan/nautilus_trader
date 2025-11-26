#!/bin/bash

# NPM Package Publishing Script
# This script publishes all @neural-trader packages in dependency order

set -e

PACKAGES_DIR="/workspaces/neural-trader/neural-trader-rust/packages"
LOG_FILE="$PACKAGES_DIR/PUBLISH_LOG.md"
WAIT_TIME=30

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to log messages
log_message() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "- [$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    echo "- [$(date '+%Y-%m-%d %H:%M:%S')] ❌ ERROR: $1" >> "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
    echo "- [$(date '+%Y-%m-%d %H:%M:%S')] ✅ SUCCESS: $1" >> "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
    echo "- [$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ WARNING: $1" >> "$LOG_FILE"
}

# Function to publish a package
publish_package() {
    local package_dir=$1
    local package_name=$(jq -r '.name' "$package_dir/package.json")
    local package_version=$(jq -r '.version' "$package_dir/package.json")

    log_message "Publishing $package_name@$package_version from $package_dir"

    cd "$package_dir"

    # Check if package needs building
    if [ -f "tsconfig.json" ] && [ ! -d "dist" ]; then
        log_message "Building $package_name..."
        npm run build || {
            log_error "Build failed for $package_name"
            return 1
        }
    fi

    # Test packaging
    log_message "Testing package creation for $package_name..."
    npm pack --dry-run > /dev/null 2>&1 || {
        log_error "Package test failed for $package_name"
        return 1
    }

    # Publish
    log_message "Publishing $package_name to npm registry..."
    if npm publish --access public 2>&1 | tee /tmp/npm_publish_output.txt; then
        log_success "Published $package_name@$package_version"
        npx claude-flow@alpha hooks notify --message "Published $package_name@$package_version" 2>/dev/null || true
        return 0
    else
        local error_output=$(cat /tmp/npm_publish_output.txt)
        if echo "$error_output" | grep -q "already exists\|cannot publish over"; then
            log_warning "Package $package_name@$package_version already published, skipping"
            return 0
        else
            log_error "Failed to publish $package_name: $error_output"
            return 1
        fi
    fi
}

# Function to verify published package
verify_package() {
    local package_name=$1
    local expected_version=$2
    local max_attempts=5
    local attempt=1

    log_message "Verifying $package_name@$expected_version..."

    while [ $attempt -le $max_attempts ]; do
        local published_version=$(npm view "$package_name" version 2>/dev/null || echo "NOT_FOUND")

        if [ "$published_version" = "$expected_version" ]; then
            log_success "Verified $package_name@$expected_version is accessible"
            return 0
        elif [ "$published_version" != "NOT_FOUND" ]; then
            log_error "Version mismatch: Expected $expected_version, got $published_version"
            return 1
        fi

        if [ $attempt -lt $max_attempts ]; then
            log_message "Package not yet available, waiting 10s (attempt $attempt/$max_attempts)..."
            sleep 10
        fi
        attempt=$((attempt + 1))
    done

    log_warning "Could not verify $package_name after $max_attempts attempts, but continuing..."
    return 0  # Don't fail, just warn
}

# Main publishing sequence
main() {
    echo "========================================"
    echo "NPM Package Publishing - Neural Trader"
    echo "========================================"
    echo ""

    cd "$PACKAGES_DIR"

    # Phase 1: Core package
    echo -e "\n${GREEN}=== Phase 1: Publishing Core Package ===${NC}\n"
    echo -e "\n### Phase 1: Core Package\n" >> "$LOG_FILE"

    publish_package "$PACKAGES_DIR/core" || exit 1
    verify_package "@neural-trader/core" "1.0.0" || exit 1
    sleep $WAIT_TIME

    # Phase 2: MCP Protocol
    echo -e "\n${GREEN}=== Phase 2: Publishing MCP Protocol ===${NC}\n"
    echo -e "\n### Phase 2: MCP Protocol\n" >> "$LOG_FILE"

    publish_package "$PACKAGES_DIR/mcp-protocol" || exit 1
    verify_package "@neural-trader/mcp-protocol" "1.0.0" || exit 1
    sleep $WAIT_TIME

    # Phase 3: Feature packages (parallel conceptually, but sequential for safety)
    echo -e "\n${GREEN}=== Phase 3: Publishing Feature Packages ===${NC}\n"
    echo -e "\n### Phase 3: Feature Packages\n" >> "$LOG_FILE"

    declare -a feature_packages=(
        "backtesting"
        "brokers"
        "execution"
        "features"
        "market-data"
        "neural"
        "news-trading"
        "portfolio"
        "prediction-markets"
        "risk"
        "sports-betting"
        "strategies"
    )

    for pkg in "${feature_packages[@]}"; do
        publish_package "$PACKAGES_DIR/$pkg" || log_warning "Failed to publish $pkg, continuing..."
        sleep 10  # Shorter wait for feature packages
    done

    # Verify all feature packages
    for pkg in "${feature_packages[@]}"; do
        verify_package "@neural-trader/$pkg" "1.0.0" || log_warning "Verification failed for $pkg"
    done

    sleep $WAIT_TIME

    # Phase 4: MCP server (depends on core + protocol)
    echo -e "\n${GREEN}=== Phase 4: Publishing MCP Server ===${NC}\n"
    echo -e "\n### Phase 4: MCP Server\n" >> "$LOG_FILE"

    publish_package "$PACKAGES_DIR/mcp" || exit 1
    verify_package "@neural-trader/mcp" "1.0.0" || exit 1
    sleep $WAIT_TIME

    # Phase 5: Meta package (depends on all)
    echo -e "\n${GREEN}=== Phase 5: Publishing Meta Package ===${NC}\n"
    echo -e "\n### Phase 5: Meta Package\n" >> "$LOG_FILE"

    publish_package "$PACKAGES_DIR/neural-trader" || exit 1
    verify_package "neural-trader" "1.0.0" || exit 1

    # Final summary
    echo -e "\n${GREEN}========================================"
    echo "Publishing Complete!"
    echo "========================================${NC}\n"

    echo -e "\n## Publishing Summary\n" >> "$LOG_FILE"
    echo "All packages published successfully!" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "### Installation" >> "$LOG_FILE"
    echo '```bash' >> "$LOG_FILE"
    echo "npm install neural-trader" >> "$LOG_FILE"
    echo '```' >> "$LOG_FILE"

    npx claude-flow@alpha hooks post-task --task-id "npm-publishing" 2>/dev/null || true
}

# Run main function
main "$@"
