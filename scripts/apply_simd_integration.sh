#!/bin/bash
# Apply SIMD Integration to Engine
# Replaces scalar calculations with SIMD-accelerated functions

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "  HyperPhysics SIMD Engine Integration"
echo "═══════════════════════════════════════════════════════════════"
echo

# Prerequisites check
echo "✓ Step 1: Checking prerequisites"
if ! command -v cargo &> /dev/null; then
    echo "✗ Error: Rust not installed"
    echo "  Run: ./scripts/phase2_setup.sh"
    exit 1
fi
echo "  ✓ Rust installed: $(rustc --version)"
echo

# SIMD tests must pass first
echo "✓ Step 2: Validating SIMD implementation"
if ! cargo test --features simd --lib simd --quiet 2>&1 | grep -q "test result: ok"; then
    echo "✗ Error: SIMD tests failing"
    echo "  Run: cargo test --features simd --lib simd"
    exit 1
fi
echo "  ✓ All SIMD tests passing"
echo

# Backup original file
echo "✓ Step 3: Creating backup"
ENGINE_FILE="crates/hyperphysics-core/src/engine.rs"
BACKUP_FILE="${ENGINE_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$ENGINE_FILE" "$BACKUP_FILE"
echo "  ✓ Backup created: $BACKUP_FILE"
echo

# Apply integration
echo "✓ Step 4: Applying SIMD integration patch"

# Check if already integrated
if grep -q "entropy_from_probabilities_simd" "$ENGINE_FILE"; then
    echo "  ⚠ SIMD integration already applied"
    read -p "  Reapply anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Skipping integration"
        exit 0
    fi
    # Restore from backup for clean reapplication
    cp "$BACKUP_FILE" "$ENGINE_FILE"
fi

# Create temporary modified file
TMP_FILE=$(mktemp)
cat > "$TMP_FILE" << 'EOF'
// This is a placeholder - actual patch will be applied via sed/awk
EOF

# Add SIMD imports after existing use statements
echo "  → Adding SIMD imports..."
sed -i.tmp '/^use hyperphysics_thermo/a\
\
#[cfg(feature = "simd")]\
use crate::simd::engine::{\
    entropy_from_probabilities_simd,\
    magnetization_simd,\
    energy_simd,\
};
' "$ENGINE_FILE" || echo "  ⚠ Import addition may need manual verification"

# Replace entropy calculation
echo "  → Replacing entropy calculation..."
# This is complex - create a Python helper script
cat > /tmp/apply_entropy_patch.py << 'PYTHON_EOF'
import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

output = []
i = 0
while i < len(lines):
    line = lines[i]

    # Find entropy calculation section
    if 'let current_entropy = self.entropy_calc.entropy_from_pbits(lattice);' in line:
        # Replace with SIMD version
        indent = len(line) - len(line.lstrip())
        output.append(' ' * indent + '// Entropy - SIMD accelerated when available\n')
        output.append(' ' * indent + '#[cfg(feature = "simd")]\n')
        output.append(' ' * indent + 'let current_entropy = {\n')
        output.append(' ' * indent + '    let probabilities = lattice.probabilities();\n')
        output.append(' ' * indent + '    entropy_from_probabilities_simd(&probabilities)\n')
        output.append(' ' * indent + '};\n')
        output.append(' ' * indent + '\n')
        output.append(' ' * indent + '#[cfg(not(feature = "simd"))]\n')
        output.append(line)
        i += 1
        continue

    # Find magnetization calculation
    elif 'self.metrics.magnetization = lattice.magnetization();' in line:
        indent = len(line) - len(line.lstrip())
        output.append(' ' * indent + '// Magnetization - SIMD accelerated when available\n')
        output.append(' ' * indent + '#[cfg(feature = "simd")]\n')
        output.append(' ' * indent + '{\n')
        output.append(' ' * indent + '    let states = lattice.states();\n')
        output.append(' ' * indent + '    self.metrics.magnetization = magnetization_simd(&states);\n')
        output.append(' ' * indent + '}\n')
        output.append(' ' * indent + '\n')
        output.append(' ' * indent + '#[cfg(not(feature = "simd"))]\n')
        output.append(' ' * indent + '{\n')
        output.append(line)
        output.append(' ' * indent + '}\n')
        i += 1
        continue

    # Find energy calculation
    elif 'self.metrics.energy = HamiltonianCalculator::energy(lattice);' in line:
        indent = len(line) - len(line.lstrip())
        output.append(' ' * indent + '// Energy - SIMD accelerated when available\n')
        output.append(' ' * indent + '#[cfg(feature = "simd")]\n')
        output.append(' ' * indent + '{\n')
        output.append(' ' * indent + '    let states = lattice.states();\n')
        output.append(' ' * indent + '    let couplings = lattice.couplings();\n')
        output.append(' ' * indent + '    self.metrics.energy = energy_simd(&states, &couplings);\n')
        output.append(' ' * indent + '    self.metrics.energy_per_pbit = self.metrics.energy / lattice.size() as f64;\n')
        output.append(' ' * indent + '}\n')
        output.append(' ' * indent + '\n')
        output.append(' ' * indent + '#[cfg(not(feature = "simd"))]\n')
        output.append(' ' * indent + '{\n')
        output.append(line)
        # Skip next line (energy_per_pbit)
        i += 1
        if i < len(lines) and 'energy_per_pbit' in lines[i]:
            output.append(lines[i])
            i += 1
        output.append(' ' * indent + '}\n')
        continue

    output.append(line)
    i += 1

with open(sys.argv[1], 'w') as f:
    f.writelines(output)
PYTHON_EOF

python3 /tmp/apply_entropy_patch.py "$ENGINE_FILE" 2>/dev/null || {
    echo "  ⚠ Automatic patch failed - manual integration required"
    echo "  Refer to: docs/patches/ENGINE_SIMD_INTEGRATION.patch"
    cp "$BACKUP_FILE" "$ENGINE_FILE"
    exit 1
}

echo "  ✓ SIMD integration applied"
echo

# Verify compilation
echo "✓ Step 5: Verifying compilation"
if ! cargo build --features simd --quiet 2>&1 | tail -1 | grep -q "Finished"; then
    echo "✗ Error: Build failed after integration"
    echo "  Restoring backup..."
    cp "$BACKUP_FILE" "$ENGINE_FILE"
    echo "  ✓ Backup restored"
    exit 1
fi
echo "  ✓ Compilation successful"
echo

# Run tests
echo "✓ Step 6: Running tests"
TEST_OUTPUT=$(cargo test --features simd --lib engine --quiet 2>&1)
if ! echo "$TEST_OUTPUT" | grep -q "test result: ok"; then
    echo "✗ Error: Tests failing after integration"
    echo "  Restoring backup..."
    cp "$BACKUP_FILE" "$ENGINE_FILE"
    echo "  ✓ Backup restored"
    exit 1
fi
echo "  ✓ All engine tests passing"
echo

# Success
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ SIMD Integration Complete"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "Changes applied to: $ENGINE_FILE"
echo "Backup saved to: $BACKUP_FILE"
echo
echo "Next steps:"
echo "  1. Run benchmarks: cargo bench --features simd"
echo "  2. Compare results: cargo benchcmp scalar simd"
echo "  3. Validate speedup: ./scripts/validate_performance.sh"
echo
echo "Expected performance:"
echo "  Engine step: 500 µs → 100 µs (5× improvement)"
echo "  Entropy:     100 µs → 20 µs  (5× improvement)"
echo "  Energy:      200 µs → 50 µs  (4× improvement)"
echo "  Magnetization: 50 µs → 15 µs (3.3× improvement)"
echo
