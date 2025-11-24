# Test Coverage Action Plan - Path to 70%

**Target**: Achieve 70% test coverage across all MVP crates
**Timeline**: 8 weeks
**Priority**: HIGH (Required for production readiness)

## Quick Status

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Overall Coverage | ~30% | 70% | 🔴 |
| MVP Crates @ 70%+ | 1/7 | 7/7 | 🔴 |
| Tests Compiling | ~85% | 100% | 🟡 |
| Critical Gaps | 3 | 0 | 🔴 |

## Week 1: Foundation & Fixes

### Goals
- [x] Generate accurate baseline report
- [ ] Fix all compilation errors
- [ ] Run complete coverage analysis
- [ ] Document test patterns

### Tasks

**Day 1-2: Fix Compilation Errors**
```bash
# Priority 1: Fix Z3 linking in hyperphysics-verification
# Check Z3 installation
brew install z3
export Z3_SYS_Z3_HEADER=/opt/homebrew/include/z3.h

# Priority 2: Fix RiskManager test
# Make returns field public or add getter
cd crates/hyperphysics-risk/src
# Edit risk_manager.rs to add: pub returns: Vec<f64>
```

**Day 3-4: Run Coverage Analysis**
```bash
# Generate HTML report
cargo llvm-cov --workspace --html --output-dir coverage/ \
    --ignore-filename-regex "vendor/.*|.*/examples/.*|.*/benches/.*"

# Generate JSON for tracking
cargo llvm-cov --workspace --json --output-path coverage/coverage.json

# Extract per-crate metrics
cargo llvm-cov --package hyperphysics-core --text
cargo llvm-cov --package hyperphysics-market --text
cargo llvm-cov --package rapier-hyperphysics --text
```

**Day 5: Document & Plan**
- [ ] Review HTML coverage report
- [ ] Identify top 100 uncovered functions
- [ ] Create test templates for each crate
- [ ] Set up coverage CI pipeline

### Deliverables
- ✅ Coverage baseline report
- [ ] All tests compiling successfully
- [ ] HTML coverage report
- [ ] Weekly test plan

## Week 2-3: Critical Crate - rapier-hyperphysics (0% → 70%)

**Current**: 0 tests
**Target**: 53 tests
**Impact**: CRITICAL - Physics simulation foundation

### Test Breakdown

**Integration Tests (28 tests)**:
```rust
// tests/physics_integration.rs
#[test] fn test_rigid_body_falling() { }
#[test] fn test_collision_detection_sphere_sphere() { }
#[test] fn test_collision_detection_sphere_plane() { }
#[test] fn test_joint_constraints_revolute() { }
#[test] fn test_joint_constraints_prismatic() { }
#[test] fn test_force_application() { }
#[test] fn test_impulse_application() { }
#[test] fn test_gravity_simulation() { }
#[test] fn test_friction_sliding() { }
#[test] fn test_friction_rolling() { }
#[test] fn test_restitution_bounce() { }
#[test] fn test_damping_effects() { }
#[test] fn test_multi_body_system() { }
#[test] fn test_stacking_stability() { }
#[test] fn test_chain_constraints() { }
#[test] fn test_rope_simulation() { }
#[test] fn test_soft_body_deformation() { }
#[test] fn test_fluid_coupling() { }
#[test] fn test_contact_manifold_generation() { }
#[test] fn test_continuous_collision_detection() { }
#[test] fn test_island_detection() { }
#[test] fn test_sleeping_bodies() { }
#[test] fn test_wakeup_propagation() { }
#[test] fn test_deterministic_simulation() { }
#[test] fn test_substep_accuracy() { }
#[test] fn test_energy_conservation() { }
#[test] fn test_momentum_conservation() { }
#[test] fn test_angular_momentum_conservation() { }
```

**Unit Tests (15 tests)**:
```rust
// src/physics_wrapper.rs
#[cfg(test)]
mod tests {
    #[test] fn test_create_rigid_body() { }
    #[test] fn test_create_collider() { }
    #[test] fn test_set_position() { }
    #[test] fn test_set_velocity() { }
    #[test] fn test_apply_force_at_point() { }
    #[test] fn test_compute_aabb() { }
    #[test] fn test_shape_casting() { }
    #[test] fn test_ray_casting() { }
    #[test] fn test_collision_groups() { }
    #[test] fn test_collision_filtering() { }
    #[test] fn test_sensor_detection() { }
    #[test] fn test_kinematic_bodies() { }
    #[test] fn test_fixed_bodies() { }
    #[test] fn test_dynamic_bodies() { }
    #[test] fn test_mass_properties() { }
}
```

**Property Tests (10 tests)**:
```rust
// tests/property_tests.rs
#[quickcheck]
fn prop_position_velocity_consistency(dt: f64, v: Vec3) -> bool { }

#[quickcheck]
fn prop_collision_symmetry(obj1: Object, obj2: Object) -> bool { }

#[quickcheck]
fn prop_energy_bounded(initial_energy: f64, steps: u32) -> bool { }

#[quickcheck]
fn prop_numerical_stability(high_velocity: Vec3) -> bool { }

#[quickcheck]
fn prop_constraint_satisfaction(bodies: Vec<RigidBody>) -> bool { }

#[quickcheck]
fn prop_contact_normal_validity(contact: Contact) -> bool { }

#[quickcheck]
fn prop_rotation_normalization(quat: Quaternion) -> bool { }

#[quickcheck]
fn prop_aabb_containment(shape: Shape, transform: Transform) -> bool { }

#[quickcheck]
fn prop_impulse_validity(impulse: f64, mass: f64) -> bool { }

#[quickcheck]
fn prop_determinism(seed: u64, steps: u32) -> bool { }
```

## Week 3-4: GPU MARL & Core

### gpu-marl (17% → 70%)

**Current**: 2 tests
**Target**: 18 tests (+16)

```rust
// Add 8 unit tests
#[test] fn test_env_reset() { }
#[test] fn test_env_step() { }
#[test] fn test_observation_space() { }
#[test] fn test_action_space() { }
#[test] fn test_reward_calculation() { }
#[test] fn test_done_condition() { }
#[test] fn test_info_metadata() { }
#[test] fn test_parallel_envs() { }

// Add 6 integration tests
#[test] fn test_multi_agent_coordination() { }
#[test] fn test_communication_protocol() { }
#[test] fn test_reward_shaping() { }
#[test] fn test_policy_gradient_update() { }
#[test] fn test_value_network_training() { }
#[test] fn test_episode_rollout() { }

// Add 4 performance tests
#[bench] fn bench_env_throughput() { }
#[bench] fn bench_policy_inference() { }
#[bench] fn bench_batch_processing() { }
#[bench] fn bench_gpu_transfer() { }
```

### hyperphysics-core (47% → 70%)

**Current**: 70 tests
**Target**: 104 tests (+34)

**Focus Areas**:
1. Conformal field theory (8 tests)
2. Phase space dynamics (6 tests)
3. Ergodic theory (6 tests)
4. Hamiltonian mechanics (7 tests)
5. Lagrangian formalism (7 tests)

## Week 5-6: Reasoning & Inference

### hyperphysics-reasoning-router (22% → 70%)

**Current**: 33 tests
**Target**: 104 tests (+71)

```rust
// Router logic (20 tests)
#[test] fn test_backend_selection_round_robin() { }
#[test] fn test_backend_selection_least_loaded() { }
#[test] fn test_backend_selection_weighted() { }
#[test] fn test_fallback_primary_to_secondary() { }
#[test] fn test_fallback_cascade() { }
// ... 15 more

// Health checks (15 tests)
#[test] fn test_health_check_timeout() { }
#[test] fn test_health_check_retry() { }
#[test] fn test_health_check_interval() { }
// ... 12 more

// Load balancing (18 tests)
#[test] fn test_load_distribution_uniform() { }
#[test] fn test_load_distribution_weighted() { }
#[test] fn test_load_shedding() { }
// ... 15 more

// Circuit breaker (18 tests)
#[test] fn test_circuit_breaker_open() { }
#[test] fn test_circuit_breaker_half_open() { }
#[test] fn test_circuit_breaker_closed() { }
// ... 15 more
```

### active-inference-agent (26% → 70%)

**Current**: 42 tests
**Target**: 114 tests (+72)

```rust
// Belief updating (25 tests)
#[test] fn test_bayesian_update() { }
#[test] fn test_prior_initialization() { }
#[test] fn test_likelihood_computation() { }
// ... 22 more

// Free energy minimization (20 tests)
#[test] fn test_variational_free_energy() { }
#[test] fn test_expected_free_energy() { }
#[test] fn test_kl_divergence() { }
// ... 17 more

// Policy selection (15 tests)
#[test] fn test_policy_prior() { }
#[test] fn test_policy_posterior() { }
#[test] fn test_action_sampling() { }
// ... 12 more

// Learning (12 tests)
#[test] fn test_learning_rate_decay() { }
#[test] fn test_precision_learning() { }
#[test] fn test_generative_model_update() { }
// ... 9 more
```

## Week 7-8: Polish & Maintain

### Maintain High Coverage
- hyperphysics-market: 62% → 75%
- hyperphysics-risk: 98% (maintain)

### Set Up CI/CD
```yaml
# .github/workflows/coverage.yml
name: Coverage
on: [push, pull_request]
jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions-rs/toolchain@v1
      - name: Install coverage tools
        run: cargo install cargo-llvm-cov
      - name: Run coverage
        run: cargo llvm-cov --workspace --json --output-path coverage.json
      - name: Check coverage threshold
        run: |
          COVERAGE=$(jq '.data[0].totals.lines.percent' coverage.json)
          if (( $(echo "$COVERAGE < 70" | bc -l) )); then
            echo "Coverage $COVERAGE% is below 70%"
            exit 1
          fi
```

## Test Templates

### Unit Test Template
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name_normal_case() {
        // Arrange
        let input = create_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected_output);
    }

    #[test]
    fn test_function_name_edge_case() {
        // Test boundary conditions
    }

    #[test]
    #[should_panic(expected = "error message")]
    fn test_function_name_error_case() {
        // Test error handling
    }
}
```

### Property Test Template
```rust
use quickcheck::{Arbitrary, quickcheck};

#[quickcheck]
fn prop_reversible(input: Vec<f64>) -> bool {
    let processed = process(input.clone());
    let recovered = unprocess(processed);
    approx_eq(&input, &recovered, 1e-10)
}

#[quickcheck]
fn prop_bounds_respected(input: f64) -> bool {
    let result = bounded_function(input);
    result >= MIN && result <= MAX
}
```

### Integration Test Template
```rust
#[tokio::test]
async fn test_end_to_end_workflow() {
    // Setup
    let system = create_test_system().await;

    // Execute
    let result = system.run_workflow(test_input).await;

    // Verify
    assert!(result.is_ok());
    assert_eq!(result.unwrap().status, Status::Success);

    // Cleanup
    system.shutdown().await;
}
```

## Tracking Progress

### Weekly Metrics
```bash
# Run every Friday
./scripts/coverage-report.sh

# Output format:
# Crate                          | Coverage | Change | Status
# hyperphysics-core              | 52.3%    | +5.3%  | 🟡
# hyperphysics-market            | 67.8%    | +5.8%  | 🟡
# rapier-hyperphysics            | 0.0%     | +0.0%  | 🔴
# ...
```

### Success Criteria

**Week 1**: ✅ Foundation
- [ ] All tests compile
- [ ] Coverage report generated
- [ ] Action plan approved

**Week 4**: 🎯 Critical Crates
- [ ] rapier-hyperphysics ≥ 70%
- [ ] gpu-marl ≥ 70%
- [ ] hyperphysics-core ≥ 70%

**Week 8**: 🏆 MVP Complete
- [ ] All MVP crates ≥ 70%
- [ ] CI coverage gates active
- [ ] Documentation complete
- [ ] Team trained on testing practices

## Resources

### Documentation
- [Coverage Baseline Report](./coverage-baseline-report.md)
- [Rust Testing Best Practices](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [Property Testing Guide](https://github.com/BurntSushi/quickcheck)

### Tools
- cargo-llvm-cov: `cargo install cargo-llvm-cov`
- cargo-tarpaulin: `cargo install cargo-tarpaulin`
- quickcheck: Already in dependencies
- proptest: `cargo add --dev proptest`

### Team Contacts
- **Test Lead**: TBD
- **Physics SME**: TBD (rapier-hyperphysics)
- **ML/RL SME**: TBD (gpu-marl, active-inference)
- **Market SME**: TBD (hyperphysics-market)

---

**Last Updated**: 2025-11-24
**Next Review**: 2025-12-01
**Status**: 🔴 IN PROGRESS
