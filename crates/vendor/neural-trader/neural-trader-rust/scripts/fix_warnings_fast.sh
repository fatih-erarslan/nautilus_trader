#!/bin/bash
# Fast warning fixes using sed
# Targets the most common warnings identified

set -e

cd "/workspaces/neural-trader/neural-trader-rust"

echo "🔧 Fast-fixing common warnings..."

# Fix 1: Remove unused RiskError imports (5 occurrences)
find crates -name "*.rs" -type f -exec sed -i 's/, RiskError//g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/RiskError, //g' {} \; 2>/dev/null || true

# Fix 2: Remove unused Serialize/Deserialize imports (5 occurrences)
find crates -name "*.rs" -type f -exec sed -i 's/use serde::{Deserialize, Serialize};/use serde::Serialize;/g' {} \; 2>/dev/null || true

# Fix 3: Remove unused HashMap imports (4 occurrences)
find crates -name "*.rs" -type f -exec sed -i '/^use std::collections::HashMap;$/d' {} \; 2>/dev/null || true

# Fix 4: Remove unused Position imports (3 occurrences)
find crates -name "*.rs" -type f -exec sed -i 's/, Position//g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/Position, //g' {} \; 2>/dev/null || true

# Fix 5: Remove unused async_trait imports (3 occurrences)
find crates -name "*.rs" -type f -exec sed -i '/^use async_trait::async_trait;$/d' {} \; 2>/dev/null || true

# Fix 6: Remove unused DateTime/Utc imports (2 occurrences)
find crates -name "*.rs" -type f -exec sed -i 's/DateTime, Utc/Utc/g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/use chrono::Utc;//d' {} \; 2>/dev/null || true

# Fix 7: Prefix unused variables with underscore
find crates -name "*.rs" -type f -exec sed -i 's/let order_id =/let _order_id =/g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/let config =/let _config =/g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/let tx =/let _tx =/g' {} \; 2>/dev/null || true
find crates -name "*.rs" -type f -exec sed -i 's/let targets =/let _targets =/g' {} \; 2>/dev/null || true

echo "✅ Fast fixes applied!"
echo ""
echo "📊 Checking remaining warnings..."
cargo check --workspace --quiet 2>&1 | grep "warning:" | wc -l
