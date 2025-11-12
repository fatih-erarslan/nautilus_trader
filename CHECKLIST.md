# Phase 2 Week 3 Execution Checklist

**Target**: Validate 5× SIMD speedup
**Duration**: 4-6 hours
**Prerequisites**: Terminal access, internet connection

---

## ✅ Pre-Flight (5 minutes)

- [ ] Navigate to project: `cd /Users/ashina/Desktop/Kurultay/HyperPhysics`
- [ ] Check STATUS.md for current state
- [ ] Verify scripts are executable: `ls -la scripts/*.sh`
- [ ] Read QUICK_START.md

---

## ✅ Phase 1: Environment (30 minutes)

### Step 1: Install Rust + Lean 4
- [ ] Run: `./scripts/phase2_setup.sh`
- [ ] Wait for completion (15-20 minutes)
- [ ] Verify Rust: `rustc --version`
- [ ] Verify Cargo: `cargo --version`
- [ ] Verify Lean 4: `lean --version`
- [ ] Expected: "All systems ready for Phase 2"

### Step 2: Verify SIMD Integration
- [ ] Run: `./scripts/verify_simd_integration.sh`
- [ ] Verify: "SIMD Integration Status: ✅ COMPLETE"
- [ ] Check: All 6 validation checks PASS
- [ ] Confirm: 15 SIMD unit tests found

---

## ✅ Phase 2: SIMD Testing (30 minutes)

### Step 3: Build with SIMD
- [ ] Run: `cargo build --features simd`
- [ ] Verify: "Finished dev [unoptimized + debuginfo]"
- [ ] Check for warnings: Should be zero
- [ ] Estimate: 1-2 minutes

### Step 4: Run SIMD Tests
- [ ] Run: `cargo test --features simd --lib simd`
- [ ] Verify: "test result: ok. 15 passed"
- [ ] Check: All tests green
- [ ] Estimate: 10 seconds

### Step 5: Full Test Suite
- [ ] Run: `cargo test --workspace --features simd`
- [ ] Verify: "106 passed" (91 + 15)
- [ ] Check: No test failures
- [ ] Estimate: 30-60 seconds

---

## ✅ Phase 3: Baselines (30 minutes)

### Step 6: Establish Scalar Baseline
- [ ] Run: `./scripts/benchmark_baseline.sh`
- [ ] Wait for completion (10-15 minutes)
- [ ] Verify: Baseline saved to docs/performance/baselines/
- [ ] Note scalar timing: `engine_step: _____ ns`

### Step 7: Detailed Scalar Benchmarks
- [ ] Run: `cargo bench --no-default-features -- --save-baseline scalar`
- [ ] Wait for completion (10-15 minutes)
- [ ] Verify: "Baseline saved: scalar"
- [ ] Note component timings

---

## ✅ Phase 4: Integration (15 minutes)

### Step 8: Apply SIMD Integration
- [ ] Run: `./scripts/apply_simd_integration.sh`
- [ ] Confirm when prompted (if reapplying)
- [ ] Wait for completion (2-3 minutes)
- [ ] Verify: "✅ SIMD Integration Complete"
- [ ] Check: Backup created successfully
- [ ] Verify: Compilation successful
- [ ] Verify: All engine tests passing

**If fails**: Refer to manual patch in `/docs/patches/ENGINE_SIMD_INTEGRATION.patch`

---

## ✅ Phase 5: SIMD Benchmarks (30 minutes)

### Step 9: Run SIMD Benchmarks
- [ ] Run: `cargo bench --features simd -- --save-baseline simd`
- [ ] Wait for completion (10-15 minutes)
- [ ] Verify: "Baseline saved: simd"
- [ ] Note SIMD timing: `engine_step: _____ ns`

### Step 10: Compare Baselines
- [ ] Run: `cargo benchcmp scalar simd > docs/performance/SIMD_COMPARISON.txt`
- [ ] View: `cat docs/performance/SIMD_COMPARISON.txt`
- [ ] Check speedup values:
  - [ ] engine_step: ___× (target: 5×)
  - [ ] entropy: ___× (target: 5×)
  - [ ] energy: ___× (target: 4×)
  - [ ] magnetization: ___× (target: 3.3×)

---

## ✅ Phase 6: Validation (15 minutes)

### Step 11: Comprehensive Validation
- [ ] Run: `./scripts/validate_performance.sh`
- [ ] Wait for completion (5-10 minutes)
- [ ] Check: "Status: ✅ PASS (Week 3 Gate Cleared)"
- [ ] Verify: Minimum 3× speedup achieved
- [ ] Verify: "🎯 TARGET ACHIEVED: 5× speedup goal met!" (if 5×)
- [ ] Review: Full report in docs/performance/validation/

### Step 12: Final System Validation
- [ ] Run: `./scripts/validate_system.sh`
- [ ] Verify: "✅ All 10 validation checks PASSED"
- [ ] Check: Zero warnings
- [ ] Check: No forbidden patterns
- [ ] Check: All feature flags work

---

## ✅ Success Criteria

### Minimum Gate (Required):
- [ ] SIMD feature compiles
- [ ] All 106 tests passing
- [ ] 3× speedup achieved (500 µs → 167 µs)
- [ ] Zero compiler warnings
- [ ] Backend detection works

### Target Goal (Week 3):
- [ ] 5× speedup achieved (500 µs → 100 µs)
- [ ] <1% numerical error
- [ ] Cross-platform validated
- [ ] Documentation complete

### Stretch Goal (Bonus):
- [ ] 8× speedup with AVX-512
- [ ] WASM support validated
- [ ] Benchmarks published

---

## ✅ Deliverables

After completion, you should have:

### Performance Data:
- [ ] Scalar baseline in docs/performance/baselines/
- [ ] SIMD comparison in docs/performance/SIMD_COMPARISON.txt
- [ ] Validation report in docs/performance/validation/

### Code:
- [ ] engine.rs with SIMD integration
- [ ] Backup of original engine.rs
- [ ] All SIMD modules in crates/hyperphysics-core/src/simd/

### Verification:
- [ ] All tests passing
- [ ] Performance validated
- [ ] System checks complete

---

## 📊 Record Your Results

| Metric | Scalar | SIMD | Speedup | Target | Status |
|--------|--------|------|---------|--------|--------|
| Engine step | _____ ns | _____ ns | ___× | 5× | [ ] |
| Entropy | _____ ns | _____ ns | ___× | 5× | [ ] |
| Energy | _____ ns | _____ ns | ___× | 4× | [ ] |
| Magnetization | _____ ns | _____ ns | ___× | 3.3× | [ ] |

**Overall Assessment**: [ ] PASS / [ ] FAIL

---

## 🐛 Troubleshooting

If any step fails:
1. Check error message carefully
2. Review relevant documentation in /docs/
3. Check scripts/*.sh for automation issues
4. Refer to PHASE2_WEEK3_EXECUTION_GUIDE.md for detailed help

Common issues:
- **Rust install fails**: Check internet, run again
- **Tests fail**: Run with `--verbose` flag
- **Low speedup**: Check backend detection, verify AVX2/NEON enabled
- **Integration fails**: Use manual patch from /docs/patches/

---

## ✅ Completion

When all checkboxes are complete:

- [ ] Phase 2 Week 3 COMPLETE
- [ ] 5× speedup validated
- [ ] Documentation updated
- [ ] Ready for Week 4 (ARM NEON port)

**Next Step**: Review Week 4 planning in /docs/PHASE2_KICKOFF.md

---

**Estimated Total Time**: 4-6 hours
**Critical Path**: Each phase depends on previous completion
**Success Metric**: Green validation from `validate_performance.sh`

**Start Now**: `cd /Users/ashina/Desktop/Kurultay/HyperPhysics && ./scripts/phase2_setup.sh`
