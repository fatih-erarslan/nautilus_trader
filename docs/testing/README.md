# HyperPhysics Testing Documentation

This directory contains comprehensive test coverage analysis and action plans for the HyperPhysics project.

## Quick Links

- **[Coverage Summary](./COVERAGE_SUMMARY.md)** - Executive brief with key findings
- **[Coverage Baseline Report](./coverage-baseline-report.md)** - Detailed technical analysis
- **[Action Plan](./coverage-action-plan.md)** - 8-week roadmap to 70% coverage

## Quick Start

### Check Coverage Status
```bash
# Quick estimate (fast)
./scripts/coverage-check.sh

# Full analysis (slow, accurate)
./scripts/coverage-check.sh --full
```

### Run Coverage Analysis
```bash
# Install tools
cargo install cargo-llvm-cov

# Generate HTML report
cargo llvm-cov --workspace --html --output-dir coverage/ \
    --ignore-filename-regex "vendor/.*|.*/examples/.*|.*/benches/.*"

# Open report
open coverage/html/index.html
```

### MVP Crate Status

| Crate | Est. Coverage | Target | Priority |
|-------|--------------|--------|----------|
| rapier-hyperphysics | 0% | 70% | 🔴 P0 |
| gpu-marl | 17% | 70% | 🔴 P0 |
| hyperphysics-reasoning-router | 22% | 70% | 🔴 P0 |
| active-inference-agent | 26% | 70% | 🟡 P1 |
| hyperphysics-core | 47% | 70% | 🟡 P1 |
| hyperphysics-market | 62% | 70% | 🟡 P2 |
| hyperphysics-risk | 98% | 70% | ✅ DONE |

## Key Findings

### Critical Gaps
1. **rapier-hyperphysics**: 0% coverage - Physics simulation foundation
2. **gpu-marl**: 17% coverage - Multi-agent RL coordination
3. **hyperphysics-reasoning-router**: 22% coverage - Backend routing logic

### Excellent Examples
- **hyperphysics-risk**: 98% coverage - Risk management
- **cwts-ultra**: 11,098 tests - Extensive property testing
- **quantum-circuit**: 1,865 tests - Formal verification

## Action Items

### Week 1 (Current)
- [x] Generate coverage baseline
- [ ] Fix compilation errors (Z3 linking, private fields)
- [ ] Run full coverage analysis
- [ ] Document test patterns

### Week 2-4
- [ ] rapier-hyperphysics: Add 53 tests (0% → 70%)
- [ ] gpu-marl: Add 18 tests (17% → 70%)
- [ ] hyperphysics-core: Add 34 tests (47% → 70%)

### Week 5-8
- [ ] All MVP crates ≥ 70%
- [ ] CI coverage gates active
- [ ] Testing best practices documented

## Test Templates

### Unit Test
```rust
#[test]
fn test_function_name() {
    // Arrange
    let input = setup_input();

    // Act
    let result = function_under_test(input);

    // Assert
    assert_eq!(result, expected);
}
```

### Property Test
```rust
#[quickcheck]
fn prop_reversible(input: Vec<f64>) -> bool {
    let processed = process(input.clone());
    let recovered = unprocess(processed);
    approx_eq(&input, &recovered, 1e-10)
}
```

### Integration Test
```rust
#[tokio::test]
async fn test_workflow() {
    let system = create_system().await;
    let result = system.execute(input).await;
    assert!(result.is_ok());
}
```

## Coverage Goals

- **Current**: ~30% overall, 1/7 MVP crates ≥ 70%
- **Target**: 70% overall, 7/7 MVP crates ≥ 70%
- **Timeline**: 8 weeks
- **Tests Needed**: 248 new tests

## Tools

### Installed
- ✅ cargo-llvm-cov
- ✅ cargo-tarpaulin
- ✅ quickcheck (in dependencies)

### To Install
- [ ] proptest (property testing)
- [ ] cargo-mutants (mutation testing)

## Resources

- [Rust Testing Book](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Property Testing Guide](https://github.com/BurntSushi/quickcheck)
- [Coverage Tools Comparison](https://github.com/mozilla/grcov)

## Metrics Tracking

Run weekly:
```bash
./scripts/coverage-check.sh > docs/testing/weekly-status-$(date +%Y-%m-%d).txt
```

## Contributing

When adding tests:
1. Follow test templates in this directory
2. Aim for 70%+ coverage on new code
3. Include property tests for mathematical functions
4. Add integration tests for workflows
5. Run coverage check before PR

## Questions?

See detailed reports or contact:
- Test Lead: TBD
- Physics SME: TBD
- ML/RL SME: TBD

---

**Last Updated**: 2025-11-24
**Next Review**: 2025-12-01
