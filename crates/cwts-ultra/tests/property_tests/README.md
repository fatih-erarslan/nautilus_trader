# Property-Based Tests for CWTS-Ultra

This directory contains comprehensive property-based tests using the `proptest` framework.

## Overview

Property-based testing verifies that code satisfies mathematical properties and invariants
across thousands of randomly generated test cases, providing much stronger guarantees than
traditional example-based unit tests.

## Test Suites

### 1. Liquidation Engine Properties (`liquidation_properties.rs`)
Tests financial calculation correctness:
- Initial margin is always non-negative and bounded
- Maintenance margin < initial margin (invariant)
- Liquidation price calculations (isolated vs cross margin)
- Position values are overflow-safe
- Margin ratio calculations are sound
- Leverage constraints are enforced
- Funding payments are proportional
- No arithmetic overflow in position values
- Deterministic calculations (same input = same output)
- Edge cases (zero positions, extreme values)

### 2. Quantum LSH Properties (`quantum_lsh_properties.rs`)
Tests quantum-inspired locality-sensitive hashing:
- Hash consistency (same input = same hash)
- Different tables produce different hashes
- Similarity metric symmetry: sim(a,b) = sim(b,a)
- Self-similarity is maximal
- Triangle inequality for distance metric
- No NaN propagation in computations
- No infinity propagation in similarity
- Insert and query consistency
- Jensen-Shannon divergence symmetry
- Wasserstein distance properties
- Multi-probe query effectiveness
- Batch insert consistency
- Reasonable collision rates
- Zero vector handling

### 3. Byzantine Consensus Properties (`consensus_properties.rs`)
Tests distributed consensus correctness:
- Byzantine fault tolerance: 3f+1 ≤ n
- Quorum size is always 2f+1
- Sequence numbers are monotonically increasing
- View changes increment view number
- Quantum signatures have all components
- Message timestamps are ordered
- Consensus phases transition correctly
- No duplicate validators in vote sets
- Fault tolerance guarantees
- State consistency across reads
- Invalid signature rejection
- Old message timeout handling
- Minimum validators for BFT
- Prepared state requires quorum

## Running the Tests

Run all property tests:
```bash
cargo test --test property_tests -- --test-threads=1
```

Run specific test suite:
```bash
cargo test --test liquidation_properties
cargo test --test quantum_lsh_properties
cargo test --test consensus_properties
```

Run with verbose output:
```bash
cargo test --test property_tests -- --nocapture
```

## Test Configuration

Each test suite runs **1000 test cases** by default (configurable via `PROP_TEST_CASES` constant).

Test cases are generated with:
- Random but constrained inputs (realistic financial values)
- Edge cases automatically explored
- Shrinking on failure to find minimal failing case

## Adding New Property Tests

1. Create a new `prop_*` test function
2. Use `proptest!` macro with appropriate strategies
3. Assert properties using `prop_assert!` and `prop_assert_eq!`
4. Document the property being tested

Example:
```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    
    #[test]
    fn prop_my_invariant(
        input in my_strategy(),
    ) {
        let result = calculate(input);
        prop_assert!(result >= 0.0, "Result must be non-negative");
    }
}
```

## Benefits

- **Comprehensive Coverage**: Tests thousands of edge cases automatically
- **Bug Finding**: Discovers issues traditional tests miss
- **Regression Prevention**: Properties serve as executable specifications
- **Documentation**: Properties document mathematical invariants
- **Confidence**: Strong guarantees about code correctness

## Mathematical Properties Verified

### Financial Correctness
- No negative margins
- No arithmetic overflow
- Proportional calculations
- Correct leverage constraints
- Deterministic results

### Data Structure Correctness
- Hash consistency
- Metric properties (symmetry, triangle inequality)
- No NaN/infinity propagation
- Collision rate bounds

### Distributed Systems Correctness
- Byzantine fault tolerance
- Quorum requirements
- Message ordering
- State consistency
- Safety properties

## References

- [Proptest Documentation](https://docs.rs/proptest/)
- [Byzantine Fault Tolerance](https://pmg.csail.mit.edu/papers/osdi99.pdf)
- [Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
