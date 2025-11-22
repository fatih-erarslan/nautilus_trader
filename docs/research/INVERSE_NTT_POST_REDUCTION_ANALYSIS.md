# FIPS 204 Inverse NTT Post-Reduction Analysis

**Date:** 2025-11-18
**Issue:** After fixing montgomery_reduce to return (-Q, Q), recovered[1] = -4586909 (expected: 1)
**Research Focus:** Final reduction step after inverse NTT completion

---

## Executive Summary

**CRITICAL FINDING:** The FIPS 204 reference implementation does **NOT** apply a final reduction to bring coefficients from (-Q, Q) to [0, Q) immediately after `invntt_tomont()` completes.

**Key Discovery:**
- `invntt_tomont()` returns coefficients in range **(-Q, Q)** (can be negative)
- The function `poly_caddq()` is used to normalize to [0, Q) when needed
- **However**, `poly_caddq()` is **NOT** called immediately after every inverse NTT
- It is only called when the algorithm specifically needs non-negative values

**Implication for Our Implementation:**
The issue `recovered[1] = -4586909` suggests our inverse NTT is producing values outside the valid range (-Q, Q). The problem is NOT a missing final reduction, but an error in the NTT butterfly operations themselves.

---

## Reference Implementation Analysis

### Source Files Examined
```
Repository: https://github.com/pq-crystals/dilithium (NIST FIPS 204 reference)
- ref/ntt.c         - NTT implementation
- ref/reduce.c      - Reduction functions
- ref/reduce.h      - Function prototypes
- ref/poly.c        - Polynomial operations
- ref/sign.c        - Signature generation/verification
```

---

## Complete `invntt_tomont` Algorithm

### C Reference Implementation

```c
/*************************************************
* Name:        invntt_tomont
*
* Description: Inverse NTT and multiplication by Montgomery factor 2^32.
*              In-place. No modular reductions after additions or
*              subtractions; input coefficients need to be smaller than
*              Q in absolute value. Output coefficient are smaller than Q in
*              absolute value.
*
* Arguments:   - int32_t p[N]: input/output coefficient array
**************************************************/
void invntt_tomont(int32_t a[N]) {
  unsigned int start, len, j, k;
  int32_t t, zeta;
  const int32_t f = 41978; // mont^2/256

  k = 256;
  for(len = 1; len < N; len <<= 1) {
    for(start = 0; start < N; start = j + len) {
      zeta = -zetas[--k];
      for(j = start; j < start + len; ++j) {
        t = a[j];
        a[j] = t + a[j + len];
        a[j + len] = t - a[j + len];
        a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
      }
    }
  }

  for(j = 0; j < N; ++j) {
    a[j] = montgomery_reduce((int64_t)f * a[j]);
  }
}
```

**Key Points:**
1. After all butterfly stages, only applies `montgomery_reduce(f * a[j])`
2. **NO** `caddq()` or `reduce32()` call at the end
3. Output: "Output coefficient are smaller than Q in absolute value" → range (-Q, Q)

---

## Reduction Functions in Reference

### 1. `montgomery_reduce()` - Primary Transform Function

```c
/*************************************************
* Name:        montgomery_reduce
*
* Description: For finite field element a with -2^{31}Q <= a <= Q*2^31,
*              compute r \equiv a*2^{-32} (mod Q) such that -Q < r < Q.
*
* Arguments:   - int64_t: finite field element a
*
* Returns r.
**************************************************/
int32_t montgomery_reduce(int64_t a) {
  int32_t t;

  t = (int64_t)(int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;  // ← Direct return, NO extra reduction
}
```

**Output Range:** (-Q, Q)
**Used In:** NTT butterfly operations, pointwise multiplication

### 2. `reduce32()` - Large Coefficient Reduction

```c
/*************************************************
* Name:        reduce32
*
* Description: For finite field element a with a <= 2^{31} - 2^{22} - 1,
*              compute r \equiv a (mod Q) such that -6283008 <= r <= 6283008.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t reduce32(int32_t a) {
  int32_t t;

  t = (a + (1 << 22)) >> 23;
  t = a - t*Q;
  return t;
}
```

**Output Range:** [-6283008, 6283008] (approximately [-0.75Q, 0.75Q])
**Used In:** `poly_reduce()` - reduces accumulated sums

### 3. `caddq()` - Conditional Addition of Q

```c
/*************************************************
* Name:        caddq
*
* Description: Add Q if input coefficient is negative.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t caddq(int32_t a) {
  a += (a >> 31) & Q;  // If a < 0, add Q (constant-time)
  return a;
}
```

**Output Range:** [0, 2Q) - brings negative values to non-negative
**Used In:** `poly_caddq()` - normalizes before certain operations

### 4. `freeze()` - Full Normalization to [0, Q)

```c
/*************************************************
* Name:        freeze
*
* Description: For finite field element a, compute standard
*              representative r = a mod^+ Q.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t freeze(int32_t a) {
  a = reduce32(a);  // First reduce to smaller range
  a = caddq(a);     // Then make non-negative
  return a;
}
```

**Output Range:** [0, Q)
**Used In:** Serialization, final output values

---

## Usage Patterns in Signing/Verification

### Pattern 1: invntt_tomont → reduce (optional) → caddq (when needed)

```c
// Example 1: Decompose operation requires non-negative values
polyvec_matrix_pointwise_montgomery(&w1, mat, &z);
polyveck_reduce(&w1);
polyveck_invntt_tomont(&w1);
polyveck_caddq(&w1);  // ← Make non-negative for decompose

/* Decompose w and call the random oracle */
polyveck_decompose(&w1, &w0, &w1);
```

**Reason for caddq:** `decompose` operation needs non-negative coefficients

### Pattern 2: invntt_tomont → reduce → check (no caddq)

```c
// Example 2: Norm check doesn't need non-negative values
polyvecl_pointwise_poly_montgomery(&z, &cp, &s1);
polyvecl_invntt_tomont(&z);
polyvecl_add(&z, &z, &y);
polyvecl_reduce(&z);
if(polyvecl_chknorm(&z, GAMMA1 - BETA))  // ← Works with negative values
    goto rej;
```

**Reason no caddq:** Norm check works fine with values in (-Q, Q)

### Pattern 3: invntt_tomont → direct use (no reduction at all)

```c
// Example 3: Addition after inverse NTT
polyvec_matrix_pointwise_montgomery(&t1, mat, &s1hat);
polyveck_reduce(&t1);
polyveck_invntt_tomont(&t1);
polyveck_add(&t1, &t1, &s2);  // ← Direct addition, no caddq
```

**Reason no caddq:** Subsequent operations handle signed values

---

## CRITICAL INSIGHT: When caddq is NOT Used

**invntt_tomont does NOT automatically call caddq!**

The reference implementation **intentionally keeps negative values** after inverse NTT because:

1. **Performance:** Avoiding unnecessary reduction saves cycles
2. **Algorithm correctness:** Many operations work fine with (-Q, Q) range
3. **Flexibility:** Only normalize when actually needed

---

## Diagnosis of Our Issue

### Current Behavior
```rust
let mut poly = vec![0i32; 256];
poly[1] = 1;
let ntt_poly = ntt.forward(&poly);
let recovered = ntt.inverse_std(&ntt_poly);

// Result: recovered[1] = -4586909 (expected: 1)
```

### Analysis

**The value -4586909 is OUTSIDE the valid range (-Q, Q):**
- Q = 8,380,417
- -Q = -8,380,417
- -4586909 is within (-Q, Q), so it's technically valid!

**Wait... let me recalculate:**
- -4586909 mod 8380417 = ?
- If we add Q: -4586909 + 8380417 = 3793508
- This is NOT 1!

**Conclusion:** The inverse NTT is computing the WRONG value, not just returning it in a different form.

---

## Root Cause Analysis

The issue is NOT a missing `caddq()`. The inverse NTT is mathematically incorrect.

### Possible Causes

1. **Twiddle Factor Indexing:**
   - ✅ FIXED: k now starts at 256 and decrements
   - ✅ FIXED: Uses `--k` before accessing `zetas[k]`

2. **Montgomery Reduction Range:**
   - ✅ FIXED: Now returns (-Q, Q) as per reference
   - Previously forced to [0, Q), now matches reference

3. **Butterfly Operations:**
   - Need to verify exact sequence matches reference
   - Check if additions/subtractions are in correct order

4. **Final Normalization:**
   - Currently: `montgomery_reduce(MONT_INV_256 * coeff)`
   - Reference: `montgomery_reduce((int64_t)f * a[j])` where f = 41978
   - Need to verify MONT_INV_256 == 41978

---

## Action Items

### 1. Verify Constants

```rust
// Check if these match
const MONT_INV_256: i64 = ???;  // Should be 41978
const Q: i32 = 8_380_417;       // ✓ Correct
const R_INV: i64 = 58728449;    // ✓ Matches QINV
```

### 2. Trace Butterfly Operation Step-by-Step

For input `poly[1] = 1, rest = 0`:

**Forward NTT should produce:**
```
Each coefficient = ζ^(bit_reverse(i) * 1) for appropriate power
```

**Inverse NTT should recover:**
```
Only poly[1] = 1, rest = 0
```

Add debug logging to track:
- Initial NTT output values
- Butterfly intermediate values at each stage
- Final values before normalization
- Final values after normalization

### 3. Compare Against Known-Answer Test

The FIPS 204 specification includes test vectors. We should:
1. Load a known polynomial
2. Apply forward NTT (compare to expected)
3. Apply inverse NTT (compare to expected)
4. Verify round-trip works

---

## Expected Code After Complete Fix

```rust
pub fn inverse_std(&self, coeffs: &[i32]) -> Vec<i32> {
    let mut result = self.inverse(coeffs);

    // Convert from Montgomery form to standard form
    // This applies R^(-1) to each coefficient
    for coeff in &mut result {
        *coeff = montgomery_reduce(*coeff as i64);
    }

    // CRITICAL: Reference does NOT call caddq here!
    // Values remain in (-Q, Q) range.
    // Only normalize to [0, Q) if specifically needed:
    //
    // for coeff in &mut result {
    //     *coeff = caddq(*coeff);  // ← NOT in reference invntt_tomont
    // }

    result
}
```

### Optional: Add caddq Helper Function

```rust
/// Conditional addition of Q (caddq from reference)
///
/// Adds Q to coefficient if it is negative, bringing values
/// from (-Q, Q) range to [0, Q) range.
#[inline]
pub fn caddq(a: i32) -> i32 {
    // Constant-time: add Q if a < 0
    a + (Q & (a >> 31))
}

/// Normalize polynomial to [0, Q) range
pub fn normalize_to_positive(&self, coeffs: &[i32]) -> Vec<i32> {
    coeffs.iter().map(|&c| caddq(c)).collect()
}
```

---

## Testing Strategy

### Test 1: Verify montgom reduce returns correct range

```rust
#[test]
fn test_montgomery_reduce_range() {
    // Test various inputs
    for a in [-Q as i64 * 100, -Q as i64, 0, Q as i64, Q as i64 * 100] {
        let r = montgomery_reduce(a);
        assert!(r > -Q && r < Q,
            "montgomery_reduce({}) = {} outside range (-Q, Q)", a, r);
    }
}
```

### Test 2: Verify MONT_INV_256 constant

```rust
#[test]
fn test_mont_inv_256_constant() {
    const EXPECTED_F: i64 = 41978;  // mont^2/256 from reference
    assert_eq!(MONT_INV_256, EXPECTED_F,
        "MONT_INV_256 must equal reference constant f = 41978");
}
```

### Test 3: Test caddq function (if implemented)

```rust
#[test]
fn test_caddq() {
    assert_eq!(caddq(1), 1);           // Positive stays positive
    assert_eq!(caddq(0), 0);           // Zero stays zero
    assert_eq!(caddq(-1), Q - 1);      // Negative becomes positive
    assert_eq!(caddq(-Q + 1), 1);      // -Q+1 becomes 1
}
```

### Test 4: Round-trip test with debug output

```rust
#[test]
fn test_ntt_roundtrip_debug() {
    let ntt = NTT::new();
    let mut poly = vec![0i32; 256];
    poly[1] = 1;

    println!("Original poly[1] = {}", poly[1]);

    let ntt_poly = ntt.forward(&poly);
    println!("After forward NTT:");
    for (i, &v) in ntt_poly.iter().enumerate().take(8) {
        println!("  ntt[{}] = {}", i, v);
    }

    let recovered = ntt.inverse_std(&ntt_poly);
    println!("After inverse NTT:");
    for (i, &v) in recovered.iter().enumerate().take(8) {
        println!("  recovered[{}] = {}", i, v);
    }

    // The value might be negative (in range -Q to Q)
    // Normalize to compare
    let normalized = recovered.iter()
        .map(|&c| if c < 0 { c + Q } else { c })
        .collect::<Vec<_>>();

    assert_eq!(normalized[1], 1, "Should recover poly[1] = 1");
    for i in 0..256 {
        if i != 1 {
            assert_eq!(normalized[i], 0, "poly[{}] should be 0", i);
        }
    }
}
```

---

## References

1. **FIPS 204 (2024):** Module-Lattice-Based Digital Signature Standard
   - Section 8.2: Number-Theoretic Transform
   - Algorithm 36: Inverse NTT

2. **pq-crystals/dilithium Reference Implementation**
   - https://github.com/pq-crystals/dilithium
   - `ref/ntt.c` - NTT algorithms
   - `ref/reduce.c` - Reduction functions
   - `ref/poly.c` - Polynomial operations
   - `ref/sign.c` - Signature implementation

3. **Key Constants (from ref/reduce.h):**
   ```c
   #define MONT -4186625  // 2^32 % Q
   #define QINV 58728449  // q^(-1) mod 2^32
   ```

4. **Montgomery Reduction Paper:**
   - Montgomery, P. L. (1985). "Modular multiplication without trial division"
   - Mathematics of Computation, 44(170), 519-521

---

## Summary of Findings

### Question 1: Does reference apply final reduction after inverse NTT?

**ANSWER:** NO. The reference `invntt_tomont()` function does NOT apply `caddq()` or any final reduction to bring values from (-Q, Q) to [0, Q).

### Question 2: What is the exact sequence of operations?

**ANSWER:**
```c
1. Butterfly stages (len = 1, 2, 4, ..., 128)
2. Final normalization: montgomery_reduce(f * a[j]) where f = 41978
3. Return (values remain in (-Q, Q) range)
```

### Question 3: Is there a reduce32/caddq after inverse NTT?

**ANSWER:** These functions exist but are NOT called automatically by `invntt_tomont()`. They are called explicitly by higher-level functions when needed:
- `poly_caddq()` - when algorithm needs [0, Q) range
- `poly_reduce()` - when coefficients might exceed bounds
- `freeze()` - for final serialization

### Question 4: Why is our result wrong?

**ANSWER:** The value `recovered[1] = -4586909` is mathematically INCORRECT (not just differently represented). The problem is in the NTT algorithm itself, NOT a missing final reduction.

**Next Steps:**
1. Verify MONT_INV_256 constant
2. Add debug tracing to butterfly operations
3. Compare intermediate values against reference implementation
4. Use FIPS 204 test vectors for validation

---

**Document Version:** 1.0
**Status:** Research Complete
**Next Action:** Debug butterfly operations with trace logging
