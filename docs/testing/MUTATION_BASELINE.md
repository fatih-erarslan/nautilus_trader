# Mutation Testing Baseline - HyperPhysics

## Executive Summary

This document establishes the baseline for mutation testing in the HyperPhysics project. Mutation testing is a powerful technique that evaluates the quality of our test suite by introducing artificial bugs (mutations) and checking if our tests catch them.

## Methodology

### Mutation Testing Tool
- **Tool**: `cargo-mutants`
- **Version**: Latest stable
- **Coverage**: All workspace crates

### Test Infrastructure
1. **Property-based testing** with `proptest`
2. **Fuzz testing** with `cargo-fuzz`
3. **Unit tests** per module
4. **Integration tests** for system behavior

## Initial Results (To Be Populated)

```bash
# Run mutation tests
cd /Users/ashina/Desktop/Kurultay/HyperPhysics
cargo mutants --workspace

# Expected output:
# - Total mutants generated: TBD
# - Caught by tests: TBD
# - Missed mutants: TBD
# - Mutation score: TBD%
```

### Baseline Metrics

| Crate | Total Mutants | Caught | Missed | Score |
|-------|---------------|--------|--------|-------|
| hyperphysics-core | TBD | TBD | TBD | TBD% |
| hyperphysics-geometry | TBD | TBD | TBD | TBD% |
| hyperphysics-pbit | TBD | TBD | TBD | TBD% |
| hyperphysics-thermo | TBD | TBD | TBD | TBD% |
| hyperphysics-consciousness | TBD | TBD | TBD | TBD% |
| **TOTAL** | **TBD** | **TBD** | **TBD** | **TBD%** |

## Quality Goals

### Target Metrics
- **Mutation Score**: ≥ 95%
- **Test Coverage**: 100% (lines, branches, functions)
- **Property Test Coverage**: All public APIs
- **Fuzz Stability**: 1M iterations without crashes

### Critical Invariants to Test

#### 1. Gillespie Algorithm
- ✅ Never negative transition rates
- ✅ Time monotonically increases
- ✅ Particle number conservation (where applicable)
- ✅ Finite time steps
- ✅ Proper equilibration behavior

#### 2. Metropolis Algorithm
- ✅ Acceptance probability ∈ [0,1]
- ✅ Downhill moves always accepted
- ✅ Temperature scaling correct
- ✅ Detailed balance satisfied

#### 3. Lattice Geometry
- ✅ Size consistency
- ✅ Neighbor symmetry
- ✅ ROI-48 structure validation
- ✅ Boundary condition handling

#### 4. Energy Calculations
- ✅ Flip reversibility
- ✅ Extensivity property
- ✅ Coupling symmetry
- ✅ Field linearity

## Testing Strategy

### Phase 1: Property-Based Testing (Current)
```rust
// Mathematical invariants that MUST hold
proptest! {
    #[test]
    fn prop_never_violates_physics(inputs) {
        // Generate random valid inputs
        // Verify physical laws hold
    }
}
```

### Phase 2: Mutation Testing
```bash
# Introduce mutations
cargo mutants --workspace

# Analyze failures
cargo mutants --output mutations.json
```

### Phase 3: Fuzz Testing
```bash
# Run fuzzing campaign
cargo fuzz run fuzz_gillespie -- -max_total_time=3600
cargo fuzz run fuzz_metropolis -- -max_total_time=3600
cargo fuzz run fuzz_lattice -- -max_total_time=3600
```

### Phase 4: Coverage Analysis
```bash
# Generate coverage report
cargo tarpaulin --workspace --out Html
```

## Known Gaps (To Be Addressed)

### Unimplemented Tests
1. Detailed balance verification for Metropolis
2. Long-time correlation function tests
3. Phase transition behavior validation
4. Multi-temperature replica exchange testing
5. Boundary condition edge cases

### Missing Property Tests
1. Energy landscape continuity
2. Ergodicity verification
3. Convergence rate properties
4. Numerical stability under extreme parameters

### Fuzzing Gaps
1. Multi-threaded simulation fuzzing
2. Serialization/deserialization roundtrip
3. Complex coupling pattern generation

## Iteration Protocol

### Failure Resolution Process
1. **Identify**: Mutation survives → test gap detected
2. **Analyze**: Determine what property was not tested
3. **Research**: Find peer-reviewed algorithm specification
4. **Implement**: Add property test or unit test
5. **Validate**: Re-run mutation testing
6. **Document**: Update this baseline

### Success Criteria Per Iteration
- Mutation score increases by ≥ 5%
- No new mutations introduced in fixed code
- All property tests pass
- Fuzzing finds no new crashes

## Scientific Validation

### Peer-Reviewed References
- [ ] Gillespie, D.T. (1977) - Exact stochastic simulation
- [ ] Metropolis et al. (1953) - Equation of state calculations
- [ ] Wolff, U. (1989) - Collective Monte Carlo updating
- [ ] Newman, M.E.J. & Barkema, G.T. (1999) - Monte Carlo Methods

### Implementation Verification
- [ ] Compare with reference implementations (e.g., LAMMPS)
- [ ] Reproduce known phase transition results
- [ ] Validate against analytical solutions where available

## Continuous Integration

### Pre-Commit Checks
```yaml
- name: Mutation Testing
  run: cargo mutants --no-shuffle --timeout 300

- name: Property Tests
  run: cargo test --workspace --all-features

- name: Fuzz Regression
  run: |
    cargo fuzz run fuzz_gillespie -- -runs=10000
    cargo fuzz run fuzz_metropolis -- -runs=10000
    cargo fuzz run fuzz_lattice -- -runs=10000
```

### Weekly Full Scan
```yaml
- name: Deep Mutation Analysis
  run: cargo mutants --workspace --output mutations.json

- name: Coverage Report
  run: cargo tarpaulin --workspace --out Html --out Lcov

- name: Extended Fuzzing
  run: cargo fuzz run --all -- -max_total_time=86400  # 24 hours
```

## Dashboard Metrics

### Real-Time Tracking
- Mutation score trend
- Test coverage percentage
- Average time per mutation test
- Fuzz crash rate
- Property test pass rate

### Quality Gates
```yaml
PASS_GATES:
  mutation_score: 0.95
  line_coverage: 1.00
  branch_coverage: 1.00
  fuzz_stability: 1000000  # iterations

BLOCK_MERGE_IF:
  - mutation_score < 0.90
  - any test fails
  - fuzz crashes found
  - coverage decreases
```

## Appendix: Test Examples

### Property Test Template
```rust
proptest! {
    #[test]
    fn prop_name(param in strategy) {
        // Setup
        let system = create_system(param);

        // Execute
        let result = system.operation();

        // Verify invariant
        prop_assert!(invariant_holds(result));
    }
}
```

### Fuzz Target Template
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Parse input
    if let Some(input) = parse(data) {
        // Test operation
        let _ = risky_operation(input);

        // Verify invariants
        assert!(invariants_hold());
    }
});
```

---

**Last Updated**: 2025-11-12
**Next Review**: After first mutation test run
**Owner**: QA-Lead Agent (Queen Seraphina's Hive)
