#!/bin/bash

set -e

echo "Applying quick fixes for compilation errors..."

cd "$(dirname "$0")/.."

# Fix 1: Comment out analytics routes temporarily
echo "Commenting out analytics routes..."
sed -i.bak 's|^\s*\.route("/api/analytics/dashboard"|        // .route("/api/analytics/dashboard"|' crates/api/src/main.rs
sed -i.bak 's|^\s*\.route("/api/analytics/usage"|        // .route("/api/analytics/usage"|' crates/api/src/main.rs
sed -i.bak 's|^\s*\.route("/api/analytics/performance"|        // .route("/api/analytics/performance"|' crates/api/src/main.rs
sed -i.bak 's|^\s*\.route("/api/activity/feed"|        // .route("/api/activity/feed"|' crates/api/src/main.rs
sed -i.bak 's|^\s*\.route("/api/activity/log"|        // .route("/api/activity/log"|' crates/api/src/main.rs

echo "✅ Analytics routes temporarily disabled"

# Try to build
echo ""
echo "Attempting build..."
if cargo build --release 2>&1 | tee /tmp/build-output.log; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "You can now run:"
    echo "  cargo run --release"
    echo "  ./scripts/run-integration-tests.sh"
    exit 0
else
    echo ""
    echo "❌ Build failed. Check /tmp/build-output.log for details"
    echo ""
    echo "You may need to fix additional errors manually."
    echo "See docs/validation/fix-compilation-errors-guide.md for help"
    exit 1
fi
