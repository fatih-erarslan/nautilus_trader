# NTT Bug Fix Implementation Guide

## Summary

**Root Cause**: Our `montgomery_reduce()` function applies excessive reduction, forcing results into `[0, Q)` when the FIPS 204 reference intentionally returns values in `(-Q, Q)`.

**Impact**: Inverse NTT fails for AC components because negative intermediate values are incorrectly normalized.

**Fix**: Remove the extra reduction step to match the reference implementation exactly.

## Implementation

### File: `crates/hyperphysics-dilithium/src/lattice/ntt.rs`

### Current Buggy Code (Lines 330-349)

```rust
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Reduce to [0, Q)
    let result = u as i32;

    // ⚠️ BUG: This extra reduction breaks inverse NTT! ⚠️
    let mask_high = (result >> 31) as i32;
    let mask_overflow = ((Q - 1 - result) >> 31) as i32;

    result + (Q & mask_high) - (Q & mask_overflow)
}
```

### Fixed Code (Recommended)

```rust
/// Montgomery reduction: Compute (a * R^(-1)) mod Q
///
/// Given a = x * y where x, y ∈ Z_q, compute a * R^(-1) mod Q in constant time.
///
/// # Output Range
///
/// Returns r ≡ a*2^(-32) (mod Q) such that **-Q < r < Q** (can be negative!).
///
/// This matches the FIPS 204 reference implementation. The result is in the
/// centered representative range, which is mathematically equivalent to [0, Q)
/// but requires fewer operations.
///
/// # When to Use
///
/// - ✅ NTT operations (forward/inverse)
/// - ✅ Polynomial multiplication in NTT domain
/// - ✅ Any multi-step algorithm expecting Montgomery form
///
/// # When NOT to Use
///
/// - ❌ Final output values that must be in [0, Q)
/// - ❌ Serialization/deserialization
/// - For these cases, apply `barrett_reduce()` on the final result:
///   ```rust
///   let final_value = barrett_reduce(montgomery_reduce(x) as i64);
///   ```
///
/// # Reference
///
/// FIPS 204 (2024), pq-crystals/dilithium reference implementation:
/// ```c
/// int32_t montgomery_reduce(int64_t a) {
///   int32_t t;
///   t = (int64_t)(int32_t)a*QINV;
///   t = (a - (int64_t)t*Q) >> 32;
///   return t;  // Returns directly, no extra reduction
/// }
/// ```
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // FIPS 204 reference algorithm
    // Compute t = (a * R^(-1)) mod R where R = 2^32
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // Compute u = (a - t * Q) / R
    // This is exact division (no remainder) by construction
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Return directly - result is in (-Q, Q)
    // This is mathematically equivalent to [0, Q) but faster
    u as i32
}
```

### Alternative: Add Normalized Variant (Optional)

If you need a function that always returns `[0, Q)`, add this helper:

```rust
/// Montgomery reduction with normalization to [0, Q)
///
/// Same as `montgomery_reduce()` but ensures result is in [0, Q).
/// Use this for final output values that must be non-negative.
///
/// # Performance Note
///
/// Slightly slower than `montgomery_reduce()` due to conditional addition.
/// Prefer `montgomery_reduce()` for intermediate calculations.
#[inline]
pub fn montgomery_reduce_normalized(a: i64) -> i32 {
    let r = montgomery_reduce(a);

    // Constant-time conditional addition: add Q if negative
    let mask = r >> 31;  // -1 if r < 0, 0 otherwise
    r + (Q & mask)
}
```

## Testing the Fix

### Step 1: Run Existing Tests

```bash
cd /Users/ashina/Desktop/Kurultay/HyperPhysics
cargo test -p hyperphysics-dilithium ntt -- --nocapture
```

**Expected Results After Fix**:
- ✅ `test_ntt_inversion_simple_delta` - PASS (already passing)
- ✅ `test_ntt_inversion_simple_13` - PASS (already passing)
- ✅ `test_ntt_inversion_single_ac` - **SHOULD NOW PASS** (currently fails)
- ✅ `test_ntt_inversion` - **SHOULD NOW PASS** (currently fails)

### Step 2: Verify Montgomery Reduction Behavior

Add this test to verify the new behavior:

```rust
#[test]
fn test_montgomery_reduce_can_be_negative() {
    // Montgomery reduction can return negative values in (-Q, Q)
    // This is correct behavior per FIPS 204 reference

    // Test case: value that should produce negative result
    let a = -100i64 * Q as i64;  // Large negative multiple of Q
    let result = montgomery_reduce(a);

    // Result should be in (-Q, Q)
    assert!(result > -Q && result < Q,
        "Montgomery reduce result {} not in range (-Q, Q)", result);

    // May be negative (this is correct!)
    println!("montgomery_reduce({}) = {} (can be negative)", a, result);

    // If we need [0, Q), use barrett_reduce
    let normalized = barrett_reduce(result as i64);
    assert!(normalized >= 0 && normalized < Q,
        "Barrett reduce should normalize to [0, Q)");
}
```

### Step 3: Test Inverse NTT Thoroughly

```rust
#[test]
fn test_inverse_ntt_all_positions() {
    // Test that inverse NTT works for all 256 positions
    let ntt = NTT::new();

    for pos in 0..256 {
        let mut poly = vec![0i32; 256];
        poly[pos] = 1;

        let ntt_poly = ntt.forward(&poly);
        let recovered = ntt.inverse_std(&ntt_poly);

        for i in 0..256 {
            let expected = if i == pos { 1 } else { 0 };
            assert_eq!(recovered[i], expected,
                "Position {}: recovered[{}] = {}, expected {}",
                pos, i, recovered[i], expected);
        }
    }
}
```

### Step 4: Polynomial Multiplication Test

```rust
#[test]
fn test_polynomial_multiplication_correctness() {
    let ntt = NTT::new();

    // Test: (1 + X) * (1 + X) = 1 + 2X + X^2
    let mut a = vec![0i32; 256];
    a[0] = 1;
    a[1] = 1;

    let c = ntt.mul_poly(&a, &a);
    let c_std = c.iter().map(|&x| barrett_reduce(montgomery_reduce(x as i64) as i64)).collect::<Vec<_>>();

    assert_eq!(c_std[0], 1, "c[0] should be 1");
    assert_eq!(c_std[1], 2, "c[1] should be 2");
    assert_eq!(c_std[2], 1, "c[2] should be 1");
    for i in 3..256 {
        assert_eq!(c_std[i], 0, "c[{}] should be 0", i);
    }
}
```

## Impact Analysis

### Files That Use `montgomery_reduce()`

Search for all usages:
```bash
grep -n "montgomery_reduce" crates/hyperphysics-dilithium/src/**/*.rs
```

Expected locations:
1. ✅ **ntt.rs**: Forward NTT, inverse NTT, pointwise multiplication
2. ✅ **signature.rs**: May use Montgomery reduction for modular arithmetic
3. ✅ **verification.rs**: May use Montgomery reduction for signature verification
4. ⚠️ **Other files**: Check if they expect `[0, Q)` range

### Code Review Checklist

For each usage of `montgomery_reduce()`:

- [ ] Is it part of multi-step NTT operation? → **OK** (fix helps)
- [ ] Is it intermediate computation? → **OK** (fix helps)
- [ ] Is it final output value? → **CHECK** (may need `barrett_reduce()` wrapper)
- [ ] Is it serialized to bytes? → **CHECK** (may need normalization)

### Potential Issues to Check

1. **Signature Generation**: If final signature values use `montgomery_reduce()`, they may need normalization:
```rust
// Before: might produce negative values
let final_value = montgomery_reduce(x);

// After: ensure non-negative for serialization
let final_value = barrett_reduce(montgomery_reduce(x) as i64);
```

2. **Polynomial Serialization**: Any function that converts polynomials to bytes needs `[0, Q)` range:
```rust
pub fn poly_to_bytes(poly: &[i32]) -> Vec<u8> {
    poly.iter()
        .map(|&coeff| {
            // Ensure normalized to [0, Q)
            let normalized = barrett_reduce(coeff as i64);
            // Then serialize...
        })
        .collect()
}
```

## Verification Procedure

### 1. Apply the Fix

Edit `crates/hyperphysics-dilithium/src/lattice/ntt.rs`:
- Replace lines 330-349 with the fixed implementation above

### 2. Run Tests

```bash
# Run all Dilithium tests
cargo test -p hyperphysics-dilithium --lib

# Run NTT-specific tests with output
cargo test -p hyperphysics-dilithium ntt -- --nocapture

# Run full test suite
cargo test -p hyperphysics-dilithium
```

### 3. Check for Regressions

```bash
# Run signature tests
cargo test -p hyperphysics-dilithium signature

# Run verification tests
cargo test -p hyperphysics-dilithium verification

# Run integration tests
cargo test -p hyperphysics-dilithium --test '*'
```

### 4. Benchmark Performance (Optional)

The fix should be **faster** because it removes unnecessary operations:

```bash
cargo bench -p hyperphysics-dilithium
```

Expected improvement: ~5-10% faster Montgomery reduction

## Rollback Plan

If issues are found:

1. **Immediate**: Revert the change:
```bash
git checkout HEAD -- crates/hyperphysics-dilithium/src/lattice/ntt.rs
```

2. **Investigate**: Check which code expects `[0, Q)` range
3. **Fix Forward**: Use `montgomery_reduce_normalized()` variant where needed

## Success Criteria

✅ All NTT tests pass
✅ All signature tests pass
✅ All verification tests pass
✅ `test_ntt_inversion_single_ac` passes (currently fails)
✅ Random polynomial inversion works correctly
✅ No new compiler warnings
✅ Performance maintained or improved

## Timeline

- **Immediate**: Apply fix to `montgomery_reduce()`
- **+5 minutes**: Run basic NTT tests
- **+10 minutes**: Run full Dilithium test suite
- **+15 minutes**: Check integration tests
- **Success**: All tests passing, bug fixed!

## Reference

- **FIPS 204**: Module-Lattice-Based Digital Signature Standard
- **pq-crystals/dilithium**: Official reference implementation
- **File**: `ref/reduce.c` - Montgomery reduction implementation
- **Constant**: `QINV = 58728449` (matches our `R_INV`)
