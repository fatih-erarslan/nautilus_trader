# Dilithium Crate - Known Compilation Issues

**Status**: Excluded from workspace builds (requires comprehensive refactoring)
**Estimated Effort**: 6 weeks (per Institutional Remediation Plan)
**Priority**: CRITICAL (security foundation)

## Issues Resolved

✅ **curve25519-dalek version conflict** (61 errors → 20 errors)
- Changed from `curve25519-dalek 4.1` to `curve25519-dalek-ng 4.1`
- Aligned with bulletproofs 4.0 dependency
- Fixed import statements in zk_proofs.rs

## Remaining Issues (20 errors)

### 1. Duplicate Definitions
- `BARRETT_MULTIPLIER` defined multiple times
- `NTT` struct defined multiple times
- `barrett_reduce` function defined multiple times
- `montgomery_reduce` function defined multiple times

**Root Cause**: NTT implementation duplicated across files

**Fix**: Consolidate into single `lattice/ntt.rs` module

### 2. Missing Implementations
- `ModuleLWE` type not defined
- Need full FIPS 204 Dilithium implementation

**Fix**: Implement complete lattice-based cryptography module per FIPS 204 spec

### 3. Visibility Issues
- `barrett_reduce` and `montgomery_reduce` marked private but used externally
- Methods need to be `pub(crate)` or `pub`

**Fix**: Adjust visibility modifiers

### 4. Zeroize Trait Bounds
- `SecurityLevel` doesn't satisfy trait bounds for zeroize
- Missing `#[derive(Zeroize)]` or manual implementation

**Fix**: Add proper zeroize implementation for security-critical types

## Remediation Plan (6 Weeks)

### Week 1-2: Core NTT Implementation
- [ ] Implement FIPS 204 compliant Number Theoretic Transform
- [ ] Barrett and Montgomery reduction (constant-time)
- [ ] Precomputed roots of unity (zetas table)
- [ ] Forward/inverse NTT with bit-reversal

### Week 3-4: Dilithium Algorithm
- [ ] Key generation (ML-DSA.KeyGen)
- [ ] Signing (ML-DSA.Sign)
- [ ] Verification (ML-DSA.Verify)
- [ ] Module-LWE structures
- [ ] High/Low bits decomposition

### Week 5: Security Hardening
- [ ] Constant-time operations (no timing leaks)
- [ ] Side-channel resistance
- [ ] Proper zeroization of secrets
- [ ] FIPS 204 test vectors validation

### Week 6: Integration & Testing
- [ ] Zero-knowledge proof integration
- [ ] Signed consciousness states
- [ ] External cryptography audit
- [ ] Performance benchmarking

## Current Workaround

The dilithium crate is excluded from workspace builds:

```toml
# Cargo.toml (workspace root)
[workspace]
members = [
    "crates/hyperphysics-core",
    # ... other crates ...
    # "crates/hyperphysics-dilithium",  # EXCLUDED - compilation errors
]
```

## Dependencies Until Fix

External packages can use:
- `pqcrypto-dilithium` (interim solution)
- `pqcrypto-kyber` (for key exchange)

## Reference

- FIPS 204: Module-Lattice-Based Digital Signature Standard (2024)
- NIST PQC: https://csrc.nist.gov/Projects/post-quantum-cryptography
- Institutional Remediation Plan: Section 3.1 (Complete CRYSTALS-Dilithium Implementation)

## Next Steps

1. Allocate 1 cryptography engineer for 6 weeks
2. Implement NTT per FIPS 204 specification
3. External security audit upon completion
4. Re-enable in workspace builds

---

**Last Updated**: 2025-11-13
**Tracking Issue**: #TBD
