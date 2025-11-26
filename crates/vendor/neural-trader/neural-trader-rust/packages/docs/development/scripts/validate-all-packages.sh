#!/bin/bash

# Validate All Neural Trader Packages
# This script validates all 17 packages and generates a comprehensive report

set -e

PACKAGES_DIR="/workspaces/neural-trader/neural-trader-rust/packages"
BENCHOPTIMIZER="$PACKAGES_DIR/benchoptimizer/bin/benchoptimizer.js"
OUTPUT_DIR="$PACKAGES_DIR/docs/validation"
mkdir -p "$OUTPUT_DIR"

echo "🔍 Neural Trader Package Validation"
echo "===================================="
echo ""

# List of all packages
PACKAGES=(
  "core"
  "mcp-protocol"
  "mcp"
  "backtesting"
  "neural"
  "risk"
  "strategies"
  "portfolio"
  "execution"
  "brokers"
  "market-data"
  "features"
  "sports-betting"
  "prediction-markets"
  "news-trading"
  "neural-trader"
  "benchoptimizer"
)

echo "Packages to validate: ${#PACKAGES[@]}"
echo ""

# Validate each package
for package in "${PACKAGES[@]}"; do
  echo "Validating: @neural-trader/$package"
  $BENCHOPTIMIZER validate "$PACKAGES_DIR/$package" --format json --output "$OUTPUT_DIR/$package-validation.json" 2>&1 | grep -v "Native binding" || true
done

echo ""
echo "✅ Validation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Generate summary
echo "📊 Generating Summary Report..."
node -e "
const fs = require('fs');
const path = require('path');

const outputDir = '$OUTPUT_DIR';
const packages = ${PACKAGES[@]/#/"'}.${PACKAGES[@]/%/"'};
const results = [];

packages.forEach(pkg => {
  try {
    const file = path.join(outputDir, \`\${pkg}-validation.json\`);
    if (fs.existsSync(file)) {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      results.push({
        package: pkg,
        valid: data.valid || false,
        errors: data.errors || [],
        warnings: data.warnings || []
      });
    }
  } catch (err) {
    console.error(\`Error reading \${pkg}: \${err.message}\`);
  }
});

// Generate summary
console.log('\n📋 Validation Summary\n');
console.log(\`Total Packages: \${results.length}\`);
console.log(\`Valid: \${results.filter(r => r.valid).length}\`);
console.log(\`Invalid: \${results.filter(r => !r.valid).length}\`);
console.log(\`Errors: \${results.reduce((sum, r) => sum + r.errors.length, 0)}\`);
console.log(\`Warnings: \${results.reduce((sum, r) => sum + r.warnings.length, 0)}\`);

// Save summary
fs.writeFileSync(
  path.join(outputDir, 'summary.json'),
  JSON.stringify(results, null, 2)
);

console.log(\`\nSummary saved to: \${path.join(outputDir, 'summary.json')}\`);
"

echo "✅ Complete!"
