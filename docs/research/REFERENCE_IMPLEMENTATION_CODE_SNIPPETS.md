# FIPS 204 Reference Implementation - Complete Code Snippets

**Source:** https://github.com/pq-crystals/dilithium (Official NIST Reference)
**Date Extracted:** 2025-11-18
**Purpose:** Exact reference code for debugging NTT implementation

---

## Complete inverse_ntt Function

### File: `ref/ntt.c`

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
* Arguments:   - uint32_t p[N]: input/output coefficient array
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

**KEY OBSERVATIONS:**
1. ✅ `k = 256` - starts at array end
2. ✅ `zeta = -zetas[--k]` - pre-decrement and negate
3. ✅ Butterfly: `t = a[j]; a[j] = t + a[j+len]; a[j+len] = t - a[j+len]`
4. ✅ Final normalization: `montgomery_reduce((int64_t)f * a[j])`
5. ❌ **NO caddq() or reduce32() call at the end**

---

## Complete montgomery_reduce Function

### File: `ref/reduce.c`

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
  return t;
}
```

**KEY OBSERVATIONS:**
1. ✅ Returns `t` directly without conditional reduction
2. ✅ Output range: **-Q < r < Q** (can be negative!)
3. ✅ No branches (constant-time)
4. ❌ **NO normalization to [0, Q)**

---

## Additional Reduction Functions

### File: `ref/reduce.c`

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
  a += (a >> 31) & Q;
  return a;
}

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
  a = reduce32(a);
  a = caddq(a);
  return a;
}
```

**USAGE NOTE:** These are NOT called by `invntt_tomont()`. They are used by higher-level polynomial operations when specifically needed.

---

## Constants

### File: `ref/reduce.h`

```c
#define MONT -4186625  // 2^32 % Q
#define QINV 58728449  // q^(-1) mod 2^32
```

### File: `ref/params.h`

```c
#define N 256
#define Q 8380417
```

### Verification

Our Rust constants:
```rust
const Q: i32 = 8_380_417;           // ✓ Matches
const R_INV: i64 = 58728449;        // ✓ Matches QINV
const MONT_INV_256: i32 = 41978;    // ✓ Matches f in invntt_tomont
```

---

## Polynomial Wrapper Functions

### File: `ref/poly.c`

```c
/*************************************************
* Name:        poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_invntt_tomont(poly *a) {
  invntt_tomont(a->coeffs);
}

/*************************************************
* Name:        poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283008,6283008].
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_reduce(poly *a) {
  unsigned int i;

  for(i = 0; i < N; ++i)
    a->coeffs[i] = reduce32(a->coeffs[i]);
}

/*************************************************
* Name:        poly_caddq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_caddq(poly *a) {
  unsigned int i;

  for(i = 0; i < N; ++i)
    a->coeffs[i] = caddq(a->coeffs[i]);
}
```

**KEY OBSERVATION:** `poly_invntt_tomont()` does **NOT** call `poly_caddq()` or `poly_reduce()`. It only calls `invntt_tomont()`.

---

## Usage Patterns in Signature Generation

### File: `ref/sign.c`

### Pattern 1: invntt → caddq (when decompose needs non-negative)

```c
polyvec_matrix_pointwise_montgomery(&w1, mat, &z);
polyveck_reduce(&w1);
polyveck_invntt_tomont(&w1);

/* Decompose w and call the random oracle */
polyveck_caddq(&w1);  // ← EXPLICIT call, NOT automatic
polyveck_decompose(&w1, &w0, &w1);
```

### Pattern 2: invntt → reduce → use (no caddq)

```c
polyvecl_pointwise_poly_montgomery(&z, &cp, &s1);
polyvecl_invntt_tomont(&z);
polyvecl_add(&z, &z, &y);
polyvecl_reduce(&z);
if(polyvecl_chknorm(&z, GAMMA1 - BETA))
    goto rej;
```

### Pattern 3: invntt → direct use (no reduction at all)

```c
polyvec_matrix_pointwise_montgomery(&t1, mat, &s1hat);
polyveck_reduce(&t1);
polyveck_invntt_tomont(&t1);

/* Add error vector s2 */
polyveck_add(&t1, &t1, &s2);  // ← Direct addition
```

---

## Critical Insights

### 1. Montgomery Reduce Output Range

From the comment in `montgomery_reduce()`:
> "compute r ≡ a*2^{-32} (mod Q) such that **-Q < r < Q**"

**Implication:** Negative values are intentional and correct!

### 2. invntt_tomont Output Range

From the comment in `invntt_tomont()`:
> "Output coefficient are smaller than Q in **absolute value**"

**Meaning:** Range is (-Q, Q), not [0, Q)

### 3. When to Normalize

- `poly_caddq()` is called **explicitly** when an algorithm needs [0, Q)
- Most operations work fine with (-Q, Q) range
- Only normalize when necessary (performance optimization)

---

## Detailed Butterfly Operation Breakdown

### Inverse NTT Butterfly (from reference)

```c
for(j = start; j < start + len; ++j) {
    t = a[j];                     // Step 1: Save original a[j]
    a[j] = t + a[j + len];        // Step 2: Sum (may overflow Q)
    a[j + len] = t - a[j + len];  // Step 3: Difference (may be negative)
    a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);  // Step 4: Scale
}
```

**No modular reduction between steps 2 and 3!**

The reference implementation comment states:
> "No modular reductions after additions or subtractions"

This means:
- `a[j] = t + a[j + len]` can produce values up to 2Q-2
- `a[j + len] = t - a[j + len]` can produce values down to -(Q-1)
- Only the multiplication applies Montgomery reduction

---

## Our Implementation Comparison

### Current Rust Code (from our ntt.rs)

```rust
for j in start..(start + len) {
    let t = result[j];

    // NO barrett_reduce here (matches reference)
    result[j] = t + result[j + len];
    result[j + len] = t - result[j + len];
    result[j + len] = montgomery_reduce(zeta as i64 * result[j + len] as i64);
}
```

**Matches reference!** ✓

### Current montgomery_reduce (from our ntt.rs)

```rust
pub fn montgomery_reduce(a: i64) -> i32 {
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;
    u as i32  // Return directly - FIPS 204 allows (-Q, Q)
}
```

**Matches reference!** ✓

### Current Final Normalization (from our ntt.rs)

```rust
for coeff in &mut result {
    *coeff = montgomery_reduce(MONT_INV_256 as i64 * (*coeff) as i64);
}
```

**Matches reference!** ✓

---

## Debugging Next Steps

Since our implementation matches the reference exactly, but still produces wrong results, we need to check:

### 1. Forward NTT Correctness

```rust
// Test: Does forward NTT produce correct output?
let poly = vec![0, 1, 0, 0, ...];  // Delta at position 1
let ntt_result = ntt.forward(&poly);

// Expected: Each coefficient should be ζ^(bit_reverse(i))
// Compare against reference implementation output
```

### 2. Twiddle Factor Values

```rust
// Verify zetas array matches reference exactly
let zetas = precompute_zetas();
// Compare zetas[0..256] against reference ntt.c zetas array
```

### 3. Step-by-Step Trace

Add debug output at each butterfly stage:

```rust
println!("Stage len={}, k={}", len, k);
for stage in 0..8 {
    println!("After stage {}: result[0..8] = {:?}", stage, &result[0..8]);
}
```

### 4. Known-Answer Test

Use FIPS 204 official test vectors:
- Load known polynomial
- Apply NTT (compare intermediate values)
- Apply inverse NTT (compare intermediate values)
- Verify output matches expected

---

## Summary: What We Know

### ✅ CORRECT in Our Implementation

1. Twiddle factor indexing: `k = 256`, decrement with `--k`
2. Butterfly operation order: matches reference
3. Montgomery reduction: returns (-Q, Q) as per reference
4. Final normalization constant: `f = 41978`
5. No extra caddq() after inverse NTT

### ❓ UNKNOWN - Need to Verify

1. Are zetas[] values exactly correct?
2. Is forward NTT producing correct output?
3. Are intermediate butterfly values correct?
4. Is there an overflow/underflow in i32 arithmetic?

### 🎯 Recommended Action

**Add comprehensive debug tracing** to compare our intermediate values against reference implementation:

```rust
#[cfg(test)]
mod debug_trace {
    #[test]
    fn trace_inverse_ntt_single_ac() {
        // Detailed trace comparing every intermediate value
        // against known-correct reference output
    }
}
```

---

## Reference Implementation Files

For complete verification, examine these files:

```
dilithium/ref/
├── ntt.c          ← Core NTT algorithms
├── ntt.h          ← Function prototypes
├── reduce.c       ← Reduction functions
├── reduce.h       ← Constants and prototypes
├── poly.c         ← Polynomial operations
├── sign.c         ← Usage examples
└── params.h       ← Parameter definitions
```

**Repository:** https://github.com/pq-crystals/dilithium
**Standard:** NIST FIPS 204 (2024)

---

**Document Status:** Complete
**Next Action:** Debug trace comparison with reference implementation
