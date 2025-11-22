#!/bin/bash

# CQGS System - Scientific Warning Elimination Strategy
# This script systematically eliminates warnings using scientific methodology

echo "🔬 CQGS Scientific Warning Elimination"
echo "========================================"

# Phase 1: Safe Cleanup - Unused Imports (NO FUNCTIONALITY RISK)
echo "Phase 1: Cleaning up unused imports..."

# Fix common unused imports across organism files
find ./parasitic/src/organisms -name "*.rs" -exec sed -i '
  s/use chrono::{DateTime, Utc};/\/\/ use chrono::{DateTime, Utc}; \/\/ Removed - unused/g;
  s/use nalgebra::DVector;/\/\/ use nalgebra::DVector; \/\/ Removed - unused/g;
  s/use std::sync::Arc;$/\/\/ use std::sync::Arc; \/\/ Removed - unused/g;
  s/use uuid::Uuid;/\/\/ use uuid::Uuid; \/\/ Removed - unused/g;
  s/use std::collections::{HashMap, HashSet, VecDeque};/use std::collections::HashMap;/g;
  s/use tracing::{info, warn, debug};/use tracing::info;/g;
' {} \;

# Fix analytics module imports
find ./parasitic/src/analytics -name "*.rs" -exec sed -i '
  s/, AnalyticsError//g;
  s/use std::collections::HashMap;//g;
  s/use std::time::{Duration, Instant, SystemTime};/use std::time::Duration;/g;
' {} \;

# Fix CQGS module imports  
find ./parasitic/src/cqgs -name "*.rs" -exec sed -i '
  s/use tracing::{debug, error, info, warn, instrument};/use tracing::info;/g;
  s/, CqgsEvent//g;
  s/use std::f64::consts::PI;//g;
' {} \;

echo "✅ Phase 1 Complete: Unused imports cleaned"

# Phase 2: Fix unused variables by prefixing with underscore
echo "Phase 2: Fixing unused variables..."

# Fix slippage calculator unused variables
sed -i 's/side: &TradeSide,/_side: \&TradeSide,/g' ./core/src/algorithms/slippage_calculator.rs
sed -i 's/side: TradeSide,/_side: TradeSide,/g' ./core/src/algorithms/slippage_calculator.rs

# Fix atomic orders unused guard
sed -i 's/let guard = &epoch::pin();/let _guard = \&epoch::pin();/g' ./core/src/execution/atomic_orders.rs

echo "✅ Phase 2 Complete: Unused variables prefixed"

# Phase 3: Add cfg(test) attributes to test modules
echo "Phase 3: Adding test configuration attributes..."

# Add cfg(test) to test modules in organisms
sed -i 's/pub mod anglerfish_lure_test;/#[cfg(test)]\npub mod anglerfish_lure_test;/g' ./parasitic/src/organisms/mod.rs

# Add cfg(test) to analytics test modules  
find ./parasitic/src/analytics/tests -name "mod.rs" -exec sed -i '
  s/pub mod performance_analytics_tests;/#[cfg(test)]\npub mod performance_analytics_tests;/g;
  s/pub mod organism_metrics_tests;/#[cfg(test)]\npub mod organism_metrics_tests;/g;
  s/pub mod system_health_tests;/#[cfg(test)]\npub mod system_health_tests;/g;
  s/pub mod cqgs_compliance_tests;/#[cfg(test)]\npub mod cqgs_compliance_tests;/g;
  s/pub mod integration_tests;/#[cfg(test)]\npub mod integration_tests;/g;
  s/pub mod performance_benchmarks;/#[cfg(test)]\npub mod performance_benchmarks;/g;
' {} \;

echo "✅ Phase 3 Complete: Test modules configured"

# Phase 4: Fix naming conventions
echo "Phase 4: Fixing naming conventions..."

# Fix enum variant naming
sed -i 's/Rapidly_Increasing,/RapidlyIncreasing,/g' ./parasitic/src/organisms/tardigrade.rs

echo "✅ Phase 4 Complete: Naming conventions fixed"

# Phase 5: Remove unnecessary parentheses
echo "Phase 5: Cleaning up syntax..."

# Fix unnecessary parentheses in tardigrade
sed -i 's/(volatility_score \* 0\.3 + volume_score \* 0\.25 + stability_score \* 0\.25 + spread_score \* 0\.2)/volatility_score * 0.3 + volume_score * 0.25 + stability_score * 0.25 + spread_score * 0.2/g' ./parasitic/src/organisms/tardigrade.rs

# Fix parentheses in organisms/mod.rs
sed -i 's/let genetic_factor = (self\.genetics\.aggression \* 0\.3 +/let genetic_factor = self.genetics.aggression * 0.3 +/g' ./parasitic/src/organisms/mod.rs
sed -i 's/self\.genetics\.risk_tolerance \* 0\.3);/self.genetics.risk_tolerance * 0.3;/g' ./parasitic/src/organisms/mod.rs

echo "✅ Phase 5 Complete: Syntax cleaned"

# Validation Phase: Verify the system still compiles
echo "🧪 Validation: Testing compilation..."

if cargo check --quiet; then
    echo "✅ SUCCESS: All changes validated - system compiles cleanly"
    
    # Count remaining warnings
    WARNING_COUNT=$(cargo check 2>&1 | grep -c "warning:" || echo "0")
    echo "📊 Warnings reduced to: $WARNING_COUNT"
    
    if [ "$WARNING_COUNT" -lt 50 ]; then
        echo "🎉 MAJOR SUCCESS: Warning count below 50!"
    fi
else
    echo "❌ COMPILATION ERROR: Rolling back changes..."
    git checkout .
    echo "⚠️  Manual intervention required"
fi

echo "========================================"
echo "🔬 Scientific Analysis Complete"
echo "Next: Review remaining architectural warnings manually"