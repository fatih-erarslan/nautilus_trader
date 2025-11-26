# ADR 004: Use Lean4 for Formal Verification

**Status**: Accepted

**Date**: 2025-11-16

**Deciders**: System Architect, Verification Specialist

## Context

Financial accounting systems must guarantee correctness for regulatory compliance. Traditional testing (unit, integration, E2E) can catch bugs but cannot prove absence of errors. We need mathematical guarantees for:

- **Balance Consistency**: Assets = Liabilities + Equity
- **Non-Negative Holdings**: No negative position quantities
- **Cost Basis Accuracy**: Disposal cost basis equals sum of lot costs
- **Wash Sale Compliance**: No violations of 30-day rule
- **Double-Entry Integrity**: Every debit has corresponding credit

A single accounting error can lead to:
- Failed IRS audits → $10k+ fines
- Incorrect tax liabilities → legal liability
- Loss of user trust → business failure

## Decision

We will use **Lean4** theorem prover to formally verify accounting invariants and generate mathematical proofs of correctness.

## Rationale

### Lean4 Advantages:
1. **Mathematical Proofs**: Proves theorems with 100% certainty
2. **Modern Syntax**: More approachable than Coq/Isabelle
3. **Tactics**: Powerful automation for routine proofs
4. **VSCode Integration**: Excellent IDE support with live feedback
5. **Growing Ecosystem**: Active community, improving libraries
6. **Real-World Use**: Used in mathematics (Liquid Tensor Experiment)

### Comparison with Alternatives:

| Approach | Correctness | Complexity | Integration | Maturity |
|----------|-------------|------------|-------------|----------|
| **Lean4** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Traditional Testing | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Coq | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Isabelle/HOL | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| TLA+ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Static Analysis | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### Why Not Testing Alone?
```typescript
// Testing can verify examples
test('FIFO calculation', () => {
  expect(calculateFifo(lot1, lot2, sale)).toEqual(expectedResult);
});

// But cannot prove general correctness:
// "For ALL possible lot combinations and sale transactions,
//  the cost basis equals the sum of selected lot costs"
```

### Lean4 Proof Example:
```lean
-- Prove cost basis accuracy for FIFO method
theorem fifo_cost_basis_correct (lots : List TaxLot) (disposal : Disposal) :
  disposal.costBasis = (lots.filter (λ lot => lot.id ∈ disposal.selectedLotIds))
    .map (λ lot => lot.unitCostBasis * disposal.quantityFromLot lot.id)
    .sum := by
  -- Proof by induction on lot list
  induction lots with
  | nil => simp [Disposal.costBasis]
  | cons head tail ih =>
    simp only [List.filter, List.map, List.sum]
    cases (head.id ∈ disposal.selectedLotIds) with
    | true =>
      rw [Disposal.costBasis_cons_selected]
      rw [ih]
      ring
    | false =>
      rw [Disposal.costBasis_cons_not_selected]
      exact ih
```

## Consequences

### Positive:
- **100% Correctness**: Mathematical guarantee of invariants
- **Audit Confidence**: Can present proofs to regulators
- **Regression Prevention**: Proofs catch breaking changes
- **Documentation**: Theorems serve as formal specification
- **Competitive Advantage**: Very few financial systems have formal verification

### Negative:
- **Learning Curve**: Team must learn Lean4 (3-6 months)
- **Development Time**: Proofs take 2-5x longer than tests
- **Proof Maintenance**: Code changes require proof updates
- **Limited Scope**: Can't verify external APIs or hardware
- **Tooling**: IDE support less mature than mainstream languages

### Mitigation:
- Start with critical invariants only (5-10 theorems)
- Hire/train verification specialist on team
- Use automated tactics for routine proofs
- Document proof patterns for common scenarios
- Combine with traditional testing (not replace)

## Implementation

### Architecture:
```
┌─────────────────────────────────────────────────────────────┐
│                    TypeScript Application                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Tax Calculation (Rust)                            │    │
│  │  - Implements algorithms                           │    │
│  └────────────────────┬───────────────────────────────┘    │
│                       │                                      │
│                       │ Runtime Checks                       │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Verification Layer (TypeScript)                   │    │
│  │  - Extract computation context                     │    │
│  │  - Call Lean4 verifier                             │    │
│  │  - Log proof certificates                          │    │
│  └────────────────────┬───────────────────────────────┘    │
│                       │                                      │
│                       │ Async Verification                   │
│                       ▼                                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Lean4 Verifier (Subprocess)                       │    │
│  │  - Load theorem and proof                          │    │
│  │  - Check proof validity                            │    │
│  │  - Return verification result                      │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Verified Invariants:
1. **Balance Consistency**
   ```lean
   theorem balance_equation (ledger : Ledger) :
     ledger.assets = ledger.liabilities + ledger.equity
   ```

2. **Non-Negative Holdings**
   ```lean
   theorem positions_non_negative (position : Position) :
     position.quantity ≥ 0
   ```

3. **Cost Basis Accuracy**
   ```lean
   theorem disposal_cost_basis_correct (lots : List TaxLot) (disposal : Disposal) :
     disposal.costBasis = lots.selectedSum
   ```

4. **Wash Sale Compliance**
   ```lean
   theorem no_wash_sale_violations (sales : List Disposal) (purchases : List Transaction) :
     ∀ sale ∈ sales, ∀ purchase ∈ purchases,
       (purchase.timestamp - sale.timestamp).days ≥ 30 ∨
       purchase.asset ≠ sale.asset
   ```

5. **Lot Quantity Conservation**
   ```lean
   theorem lot_quantity_conserved (lot : TaxLot) (disposals : List Disposal) :
     lot.quantity + disposals.sumQuantityFromLot lot.id = lot.originalQuantity
   ```

### Integration Workflow:
1. TypeScript calls Rust tax calculation
2. Result stored in database
3. Async job extracts context and generates Lean4 code
4. Lean4 subprocess verifies theorem
5. Proof certificate stored in `verification_proofs` table
6. Alert triggered if verification fails

### Performance:
- Verification runs asynchronously (doesn't block user)
- Simple proofs: <100ms
- Complex proofs: <5 seconds
- Cached results for identical inputs

## Adoption Strategy

### Phase 1: Foundation (Weeks 1-4)
- Set up Lean4 development environment
- Implement 5 core theorem proofs
- Build verification infrastructure

### Phase 2: Critical Path (Weeks 5-8)
- Verify all FIFO/LIFO/HIFO calculations
- Prove wash sale detection correctness
- Add automated proof generation

### Phase 3: Expansion (Weeks 9-12)
- Extend to compliance rules
- Verify double-entry accounting
- Generate proof certificates for audits

### Phase 4: Maintenance (Ongoing)
- Update proofs for new features
- Refactor proofs for maintainability
- Train team on Lean4

## References

- [Lean4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Theorem Proving in Lean](https://leanprover.github.io/theorem_proving_in_lean4/)
- [Liquid Tensor Experiment](https://leanprover-community.github.io/blog/posts/lte-final/)
- [Formal Verification in Finance](https://www.fstar-lang.org/tutorial/)
