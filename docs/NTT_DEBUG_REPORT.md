# NTT Inversion Debugging Report

**Date:** 2025-11-18
**Status:** Partial Fix Applied, Still Failing
**Issue:** NTT inversion test fails with 255/256 coefficients incorrect

---

## Fixes Applied

### 1. Forward NTT - Fixed k Increment Pattern ✅
**Issue:** Was using post-increment (k++) instead of pre-increment (++k)

**Before:**
```rust
let zeta = zetas[k];
k += 1;
// Accessed: zetas[0], zetas[1], ..., zetas[254]
```

**After:**
```rust
k += 1;
let zeta = zetas[k];
// Accesses: zetas[1], zetas[2], ..., zetas[255]
```

### 2. Forward NTT - Removed Unnecessary Barrett Reduction ✅
**Issue:** Reference implementation does NOT reduce intermediate addition/subtraction values

**Before:**
```rust
result[j + len] = barrett_reduce(result[j] as i64 - t as i64);
result[j] = barrett_reduce(result[j] as i64 + t as i64);
```

**After:**
```rust
result[j + len] = result[j] - t;
result[j] = result[j] + t;
```

### 3. Inverse NTT - Fixed k Decrement Pattern ✅
**Issue:** Was already partially fixed (k starts at 256, pre-decrements)

**Current:**
```rust
let mut k = 256_usize;
...
k -= 1;
let zeta = barrett_reduce(-(zetas[k] as i64));
// Accesses: zetas[255], zetas[254], ..., zetas[1]
```

### 4. Inverse NTT - Removed Unnecessary Barrett Reduction ✅
**Issue:** Same as forward NTT

**Before:**
```rust
result[j] = barrett_reduce(t as i64 + result[j + len] as i64);
result[j + len] = barrett_reduce(t as i64 - result[j + len] as i64);
```

**After:**
```rust
result[j] = t + result[j + len];
result[j + len] = t - result[j + len];
```

---

## Current Test Results

### Passing Tests ✅
- `test_ntt_inversion_simple_delta` - poly[0]=1, rest=0
- `test_ntt_inversion_simple_13` - poly[0]=13, rest=0

### Failing Tests ❌
- `test_ntt_inversion_single_ac` - poly[1]=1, rest=0
  - Expected: recovered[1] = 1
  - Actual: recovered[1] = 3793508

- `test_ntt_inversion` - poly[i] = (i * 13) % 1000
  - 255/256 coefficients incorrect

---

## Observed Behavior

### DC Component (Works) ✅
```
Input:  [13, 0, 0, ..., 0]
Forward NTT: [13, 13, 13, ..., 13]  (all same value)
Inverse NTT: [13, 0, 0, ..., 0]  (correctly recovered)
```

### Single AC Component (Fails) ❌
```
Input:  [0, 1, 0, ..., 0]
Forward NTT: [2766376, -2766376, 6332482, -6332482, ...]
Inverse NTT: [0, 3793508, 0, 2836891, 0, ...]  (WRONG!)
Expected:    [0, 1, 0, 0, 0, ...]
```

**Key Observation:** Forward NTT produces alternating positive/negative pairs, which is mathematically expected for AC components.

---

## Twiddle Factor Access Patterns (Verified Correct)

###  Forward NTT
- Starts: k=0
- Each iteration: k+=1, then use zetas[k]
- Accesses: zetas[1], zetas[2], ..., zetas[255]
- Total: 255 accesses

### Inverse NTT
- Starts: k=256
- Each iteration: k-=1, then use -zetas[k]
- Accesses: -zetas[255], -zetas[254], ..., -zetas[1]
- Total: 255 accesses

Perfect symmetry! ✅

---

## Potential Remaining Issues

### 1. Montgomery Form Handling ⚠
The recovered value 3793508 doesn't match simple Montgomery form predictions:
```python
expected = 1
recovered = 3793508
R = 2^32

# Check: recovered = expected * R mod Q?
(1 * 2^32) % 8380417 = 4193792 ≠ 3793508

# Check: recovered * R^(-1) mod Q = expected?
3793508 * 8265825 % 8380417 = 5973112 ≠ 1
```

This suggests the issue is more subtle than simple Montgomery conversion.

### 2. Negation of Zetas in Inverse NTT ⚠
Current code:
```rust
k -= 1;
let zeta = barrett_reduce(-(zetas[k] as i64));
```

Reference C code:
```c
zeta = -zetas[--k];
```

The reference uses direct negation, but we wrap it in barrett_reduce. This might be introducing subtle differences!

### 3. Possible Signed Arithmetic Issues ⚠
The forward NTT produces negative values (e.g., -2766376), which is mathematically correct. However, these negative values flow through the inverse NTT. The reference C code relies on specific signed overflow behavior that might differ from Rust.

---

## Next Steps

### Investigation Priority 1: Remove Barrett Reduction from Zeta Negation
Try changing:
```rust
let zeta = barrett_reduce(-(zetas[k] as i64));
```

To:
```rust
let zeta = -zetas[k];
```

Match the reference exactly!

### Investigation Priority 2: Verify Montgomery Constants
Check that MONT_INV_256 and R_INV are correct:
```rust
const MONT_INV_256: i64 = 41978;  // mont^2/256
const R_INV: i64 = 8265825;        // R^(-1) mod Q
```

### Investigation Priority 3: Compare Against Reference Test Vectors
Download NIST FIPS 204 test vectors and compare our NTT output against known-good values.

### Investigation Priority 4: Check Intermediate Values
Add extensive logging to trace exactly where values diverge from expected.

---

## References

1. FIPS 204 (2024) - Module-Lattice-Based Digital Signature Standard
2. pq-crystals/dilithium reference implementation
   https://github.com/pq-crystals/dilithium
3. Previous research: `/docs/research/dilithium_ntt_bug_analysis.md`
4. Visual diagrams: `/docs/research/ntt_indexing_diagrams.md`

---

**Author:** Code Implementation Agent
**Review Status:** Awaiting next debugging cycle
