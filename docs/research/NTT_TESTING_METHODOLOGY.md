# NTT Testing Methodology - FIPS 204 Reference Implementation

## Critical Discovery: Modular Comparison is the Answer

### The Question
How does the FIPS 204 reference implementation test NTT inversion when `invntt_tomont` returns values in the range (-Q, Q)?

### The Answer: **Option B - Modular Comparison**

Found in `/tmp/dilithium-check/ref/test/test_mul.c`, lines 40-44:

```c
poly_invntt_tomont(&c);
for(j = 0; j < N; ++j) {
    if((c.coeffs[j] - a.coeffs[j]) % Q)  // ← MODULAR COMPARISON!
        fprintf(stderr, "ERROR in ntt/invntt: c[%d] = %d != %d\n",
                j, c.coeffs[j]%Q, a.coeffs[j]);
}
```

**They use: `if((recovered - expected) % Q)`**

This means:
- If `(recovered - expected) % Q == 0`, the test **passes** ✓
- They do **NOT** normalize to [0, Q) before testing
- They do **NOT** use exact equality
- They accept any value that is congruent modulo Q

## Why This Works

### Our Current Results
```
recovered[1] = -8380416
expected[1]  = 1

(-8380416 - 1) % 8380417 = -8380417 % 8380417 = 0 ✓ PASS
```

### Mathematical Equivalence
```
-8380416 ≡ 1 (mod 8380417) because:
-8380416 = -1 × 8380417 + 1
```

## Implementation Fix Required

Our current test uses **exact equality**:
```rust
// WRONG - This fails for negative values
assert_eq!(recovered[i], expected[i]);
```

We need **modular equality**:
```rust
// CORRECT - Matches reference implementation
let diff = (recovered[i] - expected[i]).rem_euclid(Q as i32);
assert_eq!(diff, 0, "Coefficient {} mismatch: {} != {} (mod Q)", i, recovered[i], expected[i]);
```

## Reference Implementation Functions

### caddq - Conditional Add Q
```c
int32_t caddq(int32_t a) {
  a += (a >> 31) & Q;  // Add Q if negative
  return a;
}
```
Purpose: Normalize negative values to [0, Q)

### freeze - Full Normalization
```c
int32_t freeze(int32_t a) {
  a = reduce32(a);
  a = caddq(a);
  return a;
}
```
Purpose: Reduce to canonical [0, Q) range

### montgomery_reduce - Montgomery Reduction
```c
int32_t montgomery_reduce(int64_t a) {
  int32_t t;
  t = (int64_t)(int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;  // Returns value in (-Q, Q)
}
```
Purpose: Efficient modular reduction, **intentionally** returns (-Q, Q) range

## Key Insight

**The reference implementation NEVER normalizes after `invntt_tomont`**

They designed the entire system to work with values in (-Q, Q) range:
- Storage efficiency: signed i32 instead of unsigned
- Performance: avoid unnecessary normalization steps
- Testing: use modular arithmetic for comparisons

## Application to HyperPhysics

Our NTT implementation is **mathematically correct**!

The "bug" is in the **test**, not the implementation:
1. `invntt_tomont` correctly returns values in (-Q, Q) ✓
2. These values are mathematically equivalent modulo Q ✓
3. We just need to test using modular comparison ✓

## Next Steps

1. Update test to use modular comparison
2. Verify all test cases pass
3. Document that our NTT matches FIPS 204 behavior exactly
4. Consider adding `caddq()` helper for cases where [0, Q) range is needed

## Performance Impact

**No normalization needed** = **Better performance**
- Saves N × (caddq operations) per inverse NTT
- Matches reference implementation design philosophy
- Maintains mathematical correctness

---

## Conclusion

The FIPS 204 reference implementation proves that:
- ✅ Values in (-Q, Q) range are **intentional and correct**
- ✅ Testing should use **modular comparison**, not exact equality
- ✅ Our NTT implementation is **working correctly**

**Fix**: Change test from exact equality to modular equality. That's it!
