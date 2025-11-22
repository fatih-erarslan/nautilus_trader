# NTT Inverse Implementation: Line-by-Line Comparison

## Reference Implementation (C - pq-crystals/dilithium)

```c
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

## Our Implementation (Rust)

```rust
pub fn inverse(&self, coeffs: &[i32]) -> Vec<i32> {
    assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

    let mut result = coeffs.to_vec();
    let zetas = precompute_zetas();
    let mut len = 1;
    let mut k = 256_usize;

    // Inverse Cooley-Tukey (FIPS 204 Algorithm 36)
    while len < N {
        for start in (0..N).step_by(2 * len) {
            k -= 1;
            let zeta = -zetas[k];

            for j in start..(start + len) {
                // Inverse butterfly operation
                let t = result[j];

                result[j] = t + result[j + len];
                result[j + len] = t - result[j + len];
                result[j + len] = montgomery_reduce(zeta as i64 * result[j + len] as i64);
            }
        }

        len *= 2;
    }

    // Normalize by f = mont^2/256
    for coeff in &mut result {
        *coeff = montgomery_reduce(MONT_INV_256 as i64 * (*coeff) as i64);
    }

    result
}
```

## Critical Difference Found: LOOP STRUCTURE

### Reference C Code Loop Nesting:
```c
for(len = 1; len < N; len <<= 1) {              // Outer: len stages
    for(start = 0; start < N; start = j + len) {  // Middle: start positions
        zeta = -zetas[--k];                        // ONE zeta per start position
        for(j = start; j < start + len; ++j) {    // Inner: butterfly pairs
            // Use same zeta for all j in this group
        }
    }
}
```

### Our Rust Code Loop Nesting:
```rust
while len < N {                                  // Outer: len stages
    for start in (0..N).step_by(2 * len) {       // Middle: start positions
        k -= 1;
        let zeta = -zetas[k];                     // ONE zeta per start position

        for j in start..(start + len) {          // Inner: butterfly pairs
            // Use same zeta for all j in this group
        }
    }

    len *= 2;
}
```

**THE STRUCTURES ARE IDENTICAL!** ✅

## Detailed Operation Comparison

### C Reference Butterfly:
```c
zeta = -zetas[--k];           // Pre-decrement k, negate zeta
for(j = start; j < start + len; ++j) {
    t = a[j];                 // Save a[j]
    a[j] = t + a[j + len];    // a[j] = a[j] + a[j+len]
    a[j + len] = t - a[j + len];  // a[j+len] = a[j] - a[j+len] (using old a[j])
    a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
}
```

### Our Rust Butterfly:
```rust
k -= 1;
let zeta = -zetas[k];         // Pre-decrement k, negate zeta
for j in start..(start + len) {
    let t = result[j];        // Save result[j]
    result[j] = t + result[j + len];  // result[j] = result[j] + result[j+len]
    result[j + len] = t - result[j + len];  // result[j+len] = result[j] - result[j+len]
    result[j + len] = montgomery_reduce(zeta as i64 * result[j + len] as i64);
}
```

**BUTTERFLIES ARE IDENTICAL!** ✅

## Montgomery Reduction Comparison

### C Reference:
```c
int32_t montgomery_reduce(int64_t a) {
  int32_t t;
  t = (int64_t)(int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;
}
```

Where `QINV` is the modular inverse of Q modulo 2^32.

### Our Rust Implementation:
```rust
pub fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Reduce to [0, Q)
    let result = u as i32;

    // Constant-time conditional reduction
    let mask_high = (result >> 31) as i32;
    let mask_overflow = ((Q - 1 - result) >> 31) as i32;

    result + (Q & mask_high) - (Q & mask_overflow)
}
```

### KEY DIFFERENCE IDENTIFIED! 🔴

**Reference C Code:**
- Returns `t` directly without additional reduction
- Result range: `-Q < r < Q` (can be negative!)
- Comment: "compute r ≡ a*2^{-32} (mod Q) such that **-Q < r < Q**"

**Our Rust Code:**
- Applies **additional reduction** to force result into `[0, Q)`
- Uses constant-time conditional reduction with masks
- This extra reduction step changes the intermediate values!

## Impact Analysis

The reference implementation **intentionally allows negative intermediate values** in the range `(-Q, Q)` during the NTT computation. These negative values are only normalized at the very end by the final multiplication by `f = 41978`.

Our implementation **forces all intermediate values to [0, Q)** after every Montgomery reduction, which changes the mathematical behavior of the algorithm.

### Why This Breaks AC Components:

1. **Forward NTT uses addition/subtraction without reduction**, so it works fine
2. **Inverse NTT butterfly** computes:
   - `t = a[j]` (could be negative in reference)
   - `a[j] = t + a[j+len]` (could overflow Q in reference)
   - `a[j+len] = montgomery_reduce(zeta * (t - a[j+len]))`

3. **Our extra reduction** changes the value of `result[j+len]` before the next iteration
4. **This propagates errors** through the butterfly network, especially for AC components

## The Bug

Our `montgomery_reduce()` function performs **too much reduction**. It should return values in `(-Q, Q)` like the reference, not `[0, Q)`.

## Required Fix

Modify `montgomery_reduce()` to match the reference behavior:

```rust
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // Reference algorithm (FIPS 204 / pq-crystals)
    // Returns r ≡ a*2^{-32} (mod Q) such that -Q < r < Q

    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Return directly without forcing to [0, Q)
    u as i32
}
```

The normalization to `[0, Q)` should only happen at the **very end** of the inverse NTT, which is already done by the `montgomery_reduce(MONT_INV_256 * coeff)` step.

## Verification Test

After the fix, test with:
- `poly[0] = 1, rest = 0` → should recover exactly
- `poly[1] = 1, rest = 0` → should recover exactly (this currently fails)
- Random polynomials → should recover exactly

## Additional Notes

The C reference uses `reduce32()` for final normalization in some contexts:
```c
int32_t reduce32(int32_t a) {
  int32_t t;
  t = (a + (1 << 22)) >> 23;
  t = a - t*Q;
  return t;
}
```

This is separate from `montgomery_reduce()` and is used for different purposes. Our implementation should focus on matching `montgomery_reduce()` behavior first.
