# NTT Twiddle Factor Indexing - Visual Diagrams

## Forward NTT Indexing (CORRECT in current implementation)

```
Length Progression: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1

Iteration │ len │ k index │ zeta used    │ Operations
──────────┼─────┼─────────┼──────────────┼────────────
Stage 1   │ 128 │ 0→1     │ zetas[1]     │ 1 butterfly group
Stage 2   │  64 │ 1→3     │ zetas[2,3]   │ 2 butterfly groups
Stage 3   │  32 │ 3→7     │ zetas[4..7]  │ 4 butterfly groups
Stage 4   │  16 │ 7→15    │ zetas[8..15] │ 8 butterfly groups
Stage 5   │   8 │ 15→31   │ zetas[16..31]│ 16 butterfly groups
Stage 6   │   4 │ 31→63   │ zetas[32..63]│ 32 butterfly groups
Stage 7   │   2 │ 63→127  │ zetas[64..127]│ 64 butterfly groups
Stage 8   │   1 │ 127→255 │ zetas[128..255]│ 128 butterfly groups

k increments: 0 → 1 → 2 → 3 → ... → 127 (forward)
```

## Inverse NTT Indexing (CURRENTLY BROKEN)

### Current Implementation (WRONG ❌)

```
Length Progression: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128

Iteration │ len │ k index │ zeta used      │ SHOULD BE
──────────┼─────┼─────────┼────────────────┼─────────────
Stage 1   │   1 │ 0→127   │ -zetas[0..127] │ -zetas[255..128] ❌
Stage 2   │   2 │ 128→191 │ OVERFLOW! ❌   │ -zetas[127..64]
Stage 3   │   4 │ ...     │ CRASH ❌       │ -zetas[63..32]
...

k increments: 0 → 1 → 2 → ... (WRONG DIRECTION!)
```

### Reference Implementation (CORRECT ✅)

```
Length Progression: 1 → 2 → 4 → 8 → 16 → 32 → 64 → 128

Iteration │ len │ k index  │ zeta used        │ Butterfly groups
──────────┼─────┼──────────┼──────────────────┼─────────────────
Stage 1   │   1 │ 256→255  │ -zetas[255]      │ 128 groups
Stage 2   │   2 │ 255→191  │ -zetas[254,253]  │ 64 groups
Stage 3   │   4 │ 191→127  │ -zetas[252..249] │ 32 groups
Stage 4   │   8 │ 127→63   │ -zetas[248..241] │ 16 groups
Stage 5   │  16 │ 63→31    │ -zetas[240..225] │ 8 groups
Stage 6   │  32 │ 31→15    │ -zetas[224..193] │ 4 groups
Stage 7   │  64 │ 15→7     │ -zetas[192..129] │ 2 groups
Stage 8   │ 128 │ 7→3      │ -zetas[128]      │ 1 group

k decrements: 256 → 255 → 254 → ... → 129 (REVERSE!)
```

## Butterfly Operation Flow

### Forward NTT Butterfly

```
Input:  a[j], a[j+len]
       │      │
       │      ├─── × ζ ──→ t
       │      │
       ├──────┴──→ + ──→ a[j]
       │
       └────────→ - ──→ a[j+len]

Operations:
  t = ζ * a[j+len]
  a[j] = a[j] + t
  a[j+len] = a[j] - t
```

### Inverse NTT Butterfly

```
Input:  a[j], a[j+len]
       │      │
       ├──────┼──→ + ──→ a[j]
       │      │
  t ←──┘      │
       │      │
       ├──────┴──→ - ──→ temp
              │
              └─── × ζ ──→ a[j+len]

Operations:
  t = a[j]
  a[j] = t + a[j+len]
  a[j+len] = (t - a[j+len]) * ζ
```

## Twiddle Factor Array Structure

```
zetas[0..255] Precomputed Values:

Index Range │ Usage                    │ Access Pattern
────────────┼──────────────────────────┼───────────────
0           │ Unused (ζ^0 = 1)         │ Never accessed
1..127      │ Forward NTT              │ k++ (0→127)
128..255    │ Inverse NTT (negated)    │ --k (256→129)

Visual:
[0] [1→127: Forward →] [128→255: ← Inverse (negated)]
     ↑ k++                         ↑ --k
```

## Why Reverse Order?

### Mathematical Reason

Forward NTT computes:
```
X[k] = Σ x[i] * ω^(i*k)
       i=0 to N-1
```

Inverse NTT computes:
```
x[i] = N^(-1) * Σ X[k] * ω^(-i*k)
                 k=0 to N-1
```

Note the **negative exponent**: ω^(-i*k)

### Bit-Reversal Property

```
zetas[k] = ζ^bit_reverse(k)

For forward NTT stages, we need:
  ζ^1, ζ^2, ζ^4, ζ^8, ...

For inverse NTT stages, we need:
  ζ^(-1), ζ^(-2), ζ^(-4), ζ^(-8), ...

Due to bit-reversal:
  ζ^(-1) ≡ -ζ^(N-1) (in NTT domain)

So inverse uses -zetas[255], -zetas[254], ... in reverse!
```

## The Bug Visualization

### Correct Inverse NTT Indexing

```
Zetas Array: [0] [1...127] [128...255]
                              ↑
Inverse NTT:          k=256 ─┘
                      k=255 ─┐
                      k=254 ─┤
                      k=253 ─┤
                        ...  │
                      k=129 ─┤
                      k=128 ─┘ (last used)

Access: 255, 254, 253, ..., 129, 128
Pattern: PRE-decrement (--k)
```

### Current Broken Indexing

```
Zetas Array: [0] [1...127] [128...255]
                  ↑
Inverse NTT:      k=0
                  k=1
                  k=2
                  k=3
                    ...
                  k=127
                  k=128 ← OVERFLOW into unused region!

Access: 0, 1, 2, 3, ..., 127 (WRONG!)
Pattern: POST-increment (k++)
```

## Stage-by-Stage Example (len=1 stage)

### Correct Implementation

```
len = 1, 128 butterfly operations needed

Butterfly 0:   k=255, zeta=-zetas[255], process a[0] and a[1]
Butterfly 1:   k=254, zeta=-zetas[254], process a[2] and a[3]
Butterfly 2:   k=253, zeta=-zetas[253], process a[4] and a[5]
...
Butterfly 127: k=128, zeta=-zetas[128], process a[254] and a[255]

Each butterfly uses DIFFERENT twiddle factor from upper half of array
```

### Broken Implementation

```
len = 1, 128 butterfly operations

Butterfly 0:   k=0, zeta=-zetas[0], process a[0] and a[1] ❌
Butterfly 1:   k=1, zeta=-zetas[1], process a[2] and a[3] ❌
Butterfly 2:   k=2, zeta=-zetas[2], process a[4] and a[5] ❌
...
Butterfly 127: k=127, zeta=-zetas[127], process a[254] and a[255] ❌

Uses LOWER half of array instead of UPPER half!
Completely wrong twiddle factors!
```

## Impact on Frequency Components

### DC Component (poly[0]=1, rest=0)

```
Forward NTT:
  poly[0]=1 → [1, 1, 1, 1, ..., 1]  (all frequency bins = 1)

Inverse NTT (even with bug):
  [1, 1, 1, ..., 1] → Only uses zeta[0] effectively
  Since all inputs are same, butterfly operations collapse
  Result: [1, 0, 0, ..., 0] ✓ CORRECT (by accident!)
```

### AC Components (poly[1]=1, rest=0)

```
Forward NTT:
  poly[1]=1 → [complex spectrum across all frequencies]
  Example: [6973429, 3312029, 2405575, ...]

Inverse NTT with bug:
  Uses wrong twiddle factors from zetas[0..127]
  Should use -zetas[255..128]

  Result: [0, 4667042, 1106441, ...] ✗ GARBAGE!
  255/256 coefficients wrong!
```

## Visual Summary

```
FORWARD NTT: Correct ✅
┌────────────────────────────────┐
│ Polynomial (time domain)       │
│ [x₀, x₁, x₂, ..., x₂₅₅]       │
└────────────────────────────────┘
         │
         │ NTT with zetas[1→127]
         ↓
┌────────────────────────────────┐
│ NTT domain (frequency domain)  │
│ [X₀, X₁, X₂, ..., X₂₅₅]       │
└────────────────────────────────┘

INVERSE NTT: Broken ❌
┌────────────────────────────────┐
│ NTT domain                     │
│ [X₀, X₁, X₂, ..., X₂₅₅]       │
└────────────────────────────────┘
         │
         │ INTT with -zetas[0→127] ❌ WRONG!
         │ (should be -zetas[255→128])
         ↓
┌────────────────────────────────┐
│ Corrupted polynomial ❌        │
│ [garbage, garbage, ...]        │
└────────────────────────────────┘

INVERSE NTT: Fixed ✅
┌────────────────────────────────┐
│ NTT domain                     │
│ [X₀, X₁, X₂, ..., X₂₅₅]       │
└────────────────────────────────┘
         │
         │ INTT with -zetas[255→128] ✅ CORRECT!
         ↓
┌────────────────────────────────┐
│ Recovered polynomial ✅        │
│ [x₀, x₁, x₂, ..., x₂₅₅]       │
└────────────────────────────────┘
```

## Code Fix Diagram

```rust
// BEFORE (BROKEN ❌)
let mut k = 0_usize;  // ← Starts at wrong end
while len < N {
    for start in (0..N).step_by(2 * len) {
        let zeta = barrett_reduce(-(zetas[k] as i64));  // ← Wrong index
        k += 1;  // ← Increments forward (wrong direction!)
        ...
    }
    len *= 2;
}

// AFTER (FIXED ✅)
let mut k = 256_usize;  // ← Starts at correct end
while len < N {
    for start in (0..N).step_by(2 * len) {
        k -= 1;  // ← Decrements backward (correct direction!)
        let zeta = barrett_reduce(-(zetas[k] as i64));  // ← Correct index
        ...
    }
    len *= 2;
}
```

---

**Note:** These diagrams illustrate the fundamental bug: using twiddle factors in forward order (0→127) when inverse NTT requires reverse order (255→128) to compute the inverse transform correctly.
