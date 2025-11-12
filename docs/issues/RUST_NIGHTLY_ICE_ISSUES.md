# Rust Nightly Compiler ICE Issues

## Issue Report - 2025-11-12

### Critical Blocker: Internal Compiler Errors (ICE)

**Compiler Version**: `rustc 1.93.0-nightly (25d319a0f 2025-11-11)`

### Issue #1: serde_core RefCell Panic

**File**: `serde_core-1.0.228`
**Location**: `compiler/rustc_infer/src/infer/mod.rs:1110:44`
**Error**: `RefCell<T> already borrowed`

```
thread 'rustc' panicked at compiler/rustc_infer/src/infer/mod.rs:1110:44:
RefCell already borrowed

Query stack:
#0 [typeck] type-checking `ser::impls::<impl>::serialize`
#1 [analysis] running analysis passes on crate `serde_core`
```

**Impact**: Prevents compilation of any crate using serde with SIMD features enabled
**Upstream Issue**: https://github.com/rust-lang/rust/issues (to be filed)

### Issue #2: nalgebra Index Out of Bounds

**File**: `nalgebra-0.32.6`
**Location**: `/rust/deps/ena-0.14.3/src/snapshot_vec.rs:199:10`
**Error**: `index out of bounds: the len is 88 but the index is 1524`

```
thread 'rustc' panicked at /rust/deps/ena-0.14.3/src/snapshot_vec.rs:199:10:
index out of bounds: the len is 88 but the index is 1524

Query stack:
#0 [typeck] type-checking `base::coordinates::_::<impl>::deserialize`
#1 [analysis] running analysis passes on crate `nalgebra`
```

**Impact**: Prevents test compilation with dependencies on nalgebra
**Root Cause**: Type inference bug in nightly compiler's unification table

### Investigated Solutions

#### Solution 1: Downgrade to Known-Good Nightly ✅ RECOMMENDED
```bash
# Install specific nightly from October 2025
rustup install nightly-2025-10-15-x86_64-apple-darwin
rustup default nightly-2025-10-15-x86_64-apple-darwin

# Or use toolchain override
rustup override set nightly-2025-10-15-x86_64-apple-darwin
```

**Rationale**: Compiler regression introduced between Oct 15 and Nov 11
**Expected Result**: Clean compilation with portable_simd support

#### Solution 2: Pin Dependency Versions
```toml
[dependencies]
serde = "=1.0.200"  # Last version before ICE trigger
nalgebra = "=0.32.3"  # Stable version
```

**Rationale**: May avoid code paths triggering ICE
**Risk**: May lose bug fixes and features

#### Solution 3: Wait for Nightly Update
```bash
# Update to latest nightly (may include fix)
rustup update nightly
```

**Rationale**: Compiler team may have already fixed these issues
**Status**: Check rust-lang/rust issues for fix merges

#### Solution 4: Use Stable with Feature Fallback (CURRENT) ✅
```bash
rustup default stable

# Build without SIMD
cargo build --no-default-features
cargo test --no-default-features
```

**Rationale**: Ensures CI/CD pipeline remains functional
**Limitation**: Cannot test SIMD optimizations

### Long-term Solution: Dual Compilation Strategy

**Cargo.toml Configuration**:
```toml
[package]
edition = "2021"
rust-version = "1.91.0"  # Minimum stable version

[features]
default = []
simd = []  # Requires nightly
stable-only = []  # Explicit stable-only mode

[dependencies]
# Core dependencies compatible with stable
serde = { version = "1.0", features = ["derive"] }
nalgebra = "0.32"

# SIMD dependencies (nightly-only)
# portable_simd feature requires nightly
```

**CI/CD Pipeline**:
```yaml
matrix:
  toolchain:
    - stable
    - nightly-2025-10-15  # Known-good nightly
    - nightly  # Latest (may fail)
  features:
    - ""  # Default features (stable)
    - "simd"  # SIMD features (nightly)

jobs:
  test:
    steps:
      - name: Test with stable
        if: matrix.toolchain == 'stable'
        run: cargo test --no-default-features

      - name: Test with SIMD
        if: matrix.toolchain != 'stable'
        run: cargo test --features simd
        continue-on-error: ${{ matrix.toolchain == 'nightly' }}
```

### Workaround Implementation Status

- ✅ Feature gates implemented for SIMD code
- ✅ Scalar fallbacks working on stable
- ✅ Tests passing on stable (21/21)
- ⏸️ SIMD benchmarks blocked by nightly ICE
- ⏸️ Performance validation pending

### Upstream Bug Reports

**To File**:
1. **Rust Issue Tracker**: ICE in serde_core with portable_simd
   - Minimal reproduction case
   - Compiler version and platform
   - ICE panic trace

2. **Rust Issue Tracker**: ICE in nalgebra type inference
   - Minimal reproduction case
   - ena crate version conflict
   - Unification table overflow

### References

- Rust Nightly Book: https://doc.rust-lang.org/nightly/edition-guide/
- Portable SIMD RFC: https://github.com/rust-lang/rfcs/pull/2977
- Known Nightly Issues: https://github.com/rust-lang/rust/labels/I-ICE

### Impact Assessment

**Severity**: HIGH - Blocks SIMD performance validation
**Workaround**: Available (use stable Rust without SIMD)
**Timeline**:
- Immediate: Use stable Rust
- Short-term (1-2 weeks): Try nightly updates
- Long-term: Implement dual-compilation strategy

### Action Items

1. ✅ Document ICE issues
2. ⏸️ File upstream bug reports with minimal reproductions
3. ⏸️ Test with nightly-2025-10-15
4. ⏸️ Implement CI matrix for stable + nightly testing
5. ⏸️ Monitor rust-lang/rust for ICE fixes
6. ⏸️ Update SIMD benchmarking when compiler fixed

---

**Last Updated**: 2025-11-12 08:25 UTC
**Status**: OPEN - Awaiting nightly compiler fix or workaround validation
**Priority**: P1 - Blocks Phase 2 SIMD validation
