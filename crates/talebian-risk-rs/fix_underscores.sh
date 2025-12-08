#!/bin/bash

echo "=== FIXING ALL UNDERSCORE PARAMETERS ==="
echo "Production Financial System - Removing ALL underscore prefixes from parameters"

# Fix all _market_data parameters
find src -name "*.rs" -type f -exec sed -i 's/_market_data: &MarketData/market_data: \&MarketData/g' {} \;
find src -name "*.rs" -type f -exec sed -i 's/_market_data: &\[f64\]/market_data: \&\[f64\]/g' {} \;

# Fix all _whale_detection parameters  
find src -name "*.rs" -type f -exec sed -i 's/_whale_detection: &WhaleDetection/whale_detection: \&WhaleDetection/g' {} \;
find src -name "*.rs" -type f -exec sed -i 's/_whale_detection: Option/whale_detection: Option/g' {} \;
find src -name "*.rs" -type f -exec sed -i 's/_whale_detection: &crate::WhaleDetection/whale_detection: \&crate::WhaleDetection/g' {} \;

# Fix all _assets parameters
find src -name "*.rs" -type f -exec sed -i 's/_assets: &\[String\]/assets: \&\[String\]/g' {} \;

# Fix all _strikes parameters  
find src -name "*.rs" -type f -exec sed -i 's/_strikes: &\[f64\]/strikes: \&\[f64\]/g' {} \;

# Fix any remaining underscore parameters (generic)
find src -name "*.rs" -type f -exec sed -i 's/fn \([^(]*\)([^)]*, _\([a-z_]*\):/fn \1([^)]*, \2:/g' {} \;

echo "✅ Underscore parameters fixed"

# Run build to check
echo "Testing build..."
cargo build --release --features python-bindings 2>&1 | tee build_after_fix.log

ERRORS=$(grep "^error" build_after_fix.log | wc -l)
echo "Errors after fix: $ERRORS"