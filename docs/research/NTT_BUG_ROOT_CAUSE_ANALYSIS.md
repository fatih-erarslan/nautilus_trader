# Root Cause Analysis: Inverse NTT Bug

## Executive Summary

**BUG FOUND**: Our `montgomery_reduce()` implementation applies **excessive reduction** that forces all intermediate values into the range `[0, Q)`, while the FIPS 204 reference implementation intentionally allows values in the range `(-Q, Q)`.

This extra reduction changes intermediate values during the inverse NTT butterfly operations, causing accumulated errors that manifest as incorrect recovery of AC components (coefficients other than index 0).

## Constants Verification

### Reference Implementation (pq-crystals/dilithium)
```c
#define QINV 58728449  // q^(-1) mod 2^32
#define MONT -4186625  // 2^32 % Q
```

### Our Implementation
```rust
const R_INV: i64 = 58728449;  // ✅ CORRECT (matches QINV)
const Q: i32 = 8_380_417;     // ✅ CORRECT
```

Constants are identical. ✅

## Montgomery Reduction Comparison

### Reference Implementation (C)
```c
int32_t montgomery_reduce(int64_t a) {
  int32_t t;
  t = (int64_t)(int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;  // ← Returns directly, no extra reduction!
}
```

**Output Range**: `-Q < result < Q` (can be negative)

**Comment from reference**:
> "For finite field element a with -2^{31}Q <= a <= Q*2^31,
> compute r ≡ a*2^{-32} (mod Q) such that **-Q < r < Q**."

### Our Implementation (Rust)
```rust
pub fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Reduce to [0, Q)
    let result = u as i32;

    // ⚠️ EXTRA REDUCTION - THIS IS THE BUG! ⚠️
    let mask_high = (result >> 31) as i32;
    let mask_overflow = ((Q - 1 - result) >> 31) as i32;

    result + (Q & mask_high) - (Q & mask_overflow)  // ← Forces to [0, Q)
}
```

**Output Range**: `[0, Q)` (always non-negative)

## Why This Breaks Inverse NTT

### Expected Behavior (Reference)

The inverse NTT butterfly allows negative intermediate values:

```c
// Stage len=1, processing indices 0,1 with zeta=-zetas[255]
t = a[0];
a[0] = t + a[1];        // Could be > Q (e.g., 1 + 8380416 = 8380417)
a[1] = t - a[1];        // Could be < 0 (e.g., 1 - 8380416 = -8380415)
a[1] = montgomery_reduce(zeta * a[1]);  // Returns value in (-Q, Q)
```

**Key Point**: The intermediate values `a[0]` and `a[1]` can exceed `Q` or be negative. The Montgomery reduction on the multiplied result returns a value in `(-Q, Q)`, which is **mathematically correct** for the next iteration.

### Actual Behavior (Our Bug)

Our extra reduction forces everything to `[0, Q)`:

```rust
// Stage len=1, processing indices 0,1
let t = result[0];
result[0] = t + result[1];     // Could be > Q
result[1] = t - result[1];     // Could be < 0
result[1] = montgomery_reduce(zeta * result[1]); // ← FORCED to [0, Q)
```

**Problem**: When `result[1]` would be negative (e.g., `-8380415`), our reduction converts it to `Q + (-8380415) = 2`. This changes the mathematical value!

### Error Propagation Example

Test case: `poly[1] = 1, rest = 0`

**Forward NTT** (works correctly):
```
Input:  [0, 1, 0, 0, ...]
Output: [zeta_values...]  // Each coefficient multiplied by appropriate zeta
```

**Inverse NTT** (reference behavior):
```
Stage 1 (len=1):
  Butterfly on [0,1]: zeta = -zetas[255]
    t = ntt[0]
    ntt[0] = t + ntt[1]
    ntt[1] = montgomery_reduce(zeta * (t - ntt[1]))  // Can be negative!

Stage 2 (len=2):
  Uses results from Stage 1 (including negative values)

... continues with possibly negative intermediate values ...

Final normalization:
  result[i] = montgomery_reduce(41978 * result[i])
  This brings everything to [0, Q)
```

**Inverse NTT** (our buggy behavior):
```
Stage 1 (len=1):
  Butterfly on [0,1]: zeta = -zetas[255]
    t = result[0]
    result[0] = t + result[1]
    result[1] = montgomery_reduce(zeta * (t - result[1]))
    ↓
    result[1] is FORCED to [0, Q) even if it should be negative!

Stage 2 (len=2):
  Uses WRONG values from Stage 1
  Error propagates and amplifies!

... error accumulates through all stages ...

Final normalization:
  Still wrong because input is wrong!
```

## Mathematical Proof of Bug

### Modular Arithmetic Property

For values in `(-Q, Q)`:
- If `x ∈ (-Q, 0)`: `x mod Q = x + Q`
- If `x ∈ [0, Q)`: `x mod Q = x`

Both represent the **same equivalence class** modulo Q.

### The Reference's Intentional Choice

The reference implementation uses the **centered representation** `(-Q, Q)` because:

1. **Fewer operations**: No conditional reduction needed
2. **Faster**: Direct return after shift
3. **Mathematically equivalent**: Final normalization handles it

### Our Bug's Impact

By forcing to `[0, Q)` **at every step**:

1. **Changes intermediate values** that should be negative
2. **Breaks mathematical equivalence** for multi-stage algorithms
3. **Accumulates errors** through butterfly network

## The Fix

### Option 1: Minimal Change (Match Reference Exactly)

```rust
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // FIPS 204 reference algorithm
    // Returns r ≡ a*2^{-32} (mod Q) such that -Q < r < Q

    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    u as i32  // Return directly, no extra reduction
}
```

**Pros**:
- Exact match to reference
- Fastest implementation
- Provably correct

**Cons**:
- Returns values that can be negative
- Requires final normalization (already done in our code)

### Option 2: Lazy Reduction with Comments

```rust
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // FIPS 204 reference algorithm
    // Returns r ≡ a*2^{-32} (mod Q) such that -Q < r < Q
    //
    // NOTE: Result can be negative! This is intentional for performance.
    // Use barrett_reduce() if you need [0, Q) range.

    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    u as i32
}
```

**Pros**:
- Clear documentation
- Matches reference behavior
- Guides users to correct usage

**Cons**:
- Same as Option 1

### Option 3: Add Separate Function for Full Reduction

Keep `montgomery_reduce()` matching reference, add new function:

```rust
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // Standard Montgomery reduction: returns value in (-Q, Q)
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;
    u as i32
}

#[inline]
pub fn montgomery_reduce_normalized(a: i64) -> i32 {
    // Montgomery reduction with normalization to [0, Q)
    let r = montgomery_reduce(a);

    // Constant-time conditional addition
    let mask = r >> 31;  // -1 if negative, 0 otherwise
    r + (Q & mask)
}
```

**Pros**:
- Provides both options
- Clear naming convention
- Can optimize different use cases

**Cons**:
- More code to maintain
- Need to choose correct function

## Recommended Fix: Option 2

I recommend **Option 2** because it:

1. **Exactly matches the reference** implementation
2. **Provides clear documentation** about behavior
3. **Maintains performance** of the reference
4. **Guides developers** to use the right function

## Testing After Fix

### Test Cases

1. **DC Component (Already Passes)**:
```rust
let mut poly = vec![0i32; 256];
poly[0] = 1;
let ntt_poly = ntt.forward(&poly);
let recovered = ntt.inverse_std(&ntt_poly);
assert_eq!(recovered[0], 1);
```

2. **Single AC Component (Currently Fails)**:
```rust
let mut poly = vec![0i32; 256];
poly[1] = 1;
let ntt_poly = ntt.forward(&poly);
let recovered = ntt.inverse_std(&ntt_poly);
assert_eq!(recovered[1], 1);  // ← Should pass after fix
```

3. **Multiple Coefficients**:
```rust
let mut poly = vec![0i32; 256];
poly[0] = 13;
poly[1] = 7;
poly[5] = 42;
let ntt_poly = ntt.forward(&poly);
let recovered = ntt.inverse_std(&ntt_poly);
assert_eq!(recovered[0], 13);
assert_eq!(recovered[1], 7);
assert_eq!(recovered[5], 42);
```

4. **Random Polynomial**:
```rust
let poly: Vec<i32> = (0..256).map(|i| (i * 13) % Q).collect();
let ntt_poly = ntt.forward(&poly);
let recovered = ntt.inverse_std(&ntt_poly);
for (i, (&orig, &rec)) in poly.iter().zip(recovered.iter()).enumerate() {
    assert_eq!(orig, rec, "Failed at index {}", i);
}
```

## Additional Verification

After implementing the fix, verify:

1. **Forward NTT still works** (should be unchanged)
2. **Inverse NTT recovers all inputs** (should now pass)
3. **Polynomial multiplication is correct**:
```rust
let a = vec![1, 2, 3, 4, ...];  // Test polynomial
let b = vec![5, 6, 7, 8, ...];  // Test polynomial
let c = ntt.mul_poly(&a, &b);
// Verify c[i] matches (a * b mod (X^256 + 1))[i]
```

## Impact on Other Code

Search for all uses of `montgomery_reduce()`:

```bash
grep -r "montgomery_reduce" crates/hyperphysics-dilithium/src/
```

Verify that:
- ✅ NTT operations: Already expect `(-Q, Q)` range (fix will help)
- ✅ Final normalization: Already uses `montgomery_reduce(MONT_INV_256 * x)`
- ⚠️ Other uses: May need to check if they expect `[0, Q)` range

If any code **requires** `[0, Q)` range, use:
```rust
barrett_reduce(montgomery_reduce(x) as i64)
```

Or implement `montgomery_reduce_normalized()` helper.

## Conclusion

The bug is in `montgomery_reduce()` performing excessive reduction. The fix is simple: return the value directly after the shift operation, matching the FIPS 204 reference implementation exactly.

This will allow negative intermediate values during inverse NTT, which is mathematically correct and necessary for the algorithm to work properly.
