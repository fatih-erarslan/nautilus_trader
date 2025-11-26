# Tax Calculation Methods Implementation

## Overview

Implemented two additional tax calculation methods for the agentic accounting system:
- **Specific Identification**: Manual lot selection by user
- **Average Cost**: Weighted average cost basis calculation

## Implementation Status

### ✅ Completed

1. **Specific Identification (`specific_id.rs`)**
   - User manually selects lots in desired order
   - Validates lot existence and availability
   - Processes lots in user-specified order
   - Most flexible method requiring detailed records

2. **Average Cost (`average_cost.rs`)**
   - Calculates weighted average cost across all lots
   - Used primarily for mutual funds and cryptocurrency
   - Proportionally reduces all lots
   - Single disposal record with average cost

3. **Module Integration**
   - Updated `calculator.rs` with high-level wrapper functions
   - Updated `mod.rs` for proper module exports
   - Added NAPI exports in `lib.rs` for TypeScript/JavaScript

4. **Comprehensive Testing**
   - `tests/specific_id_test.rs`: 13 integration tests
   - `tests/average_cost_test.rs`: 16 integration tests
   - Unit tests embedded in each module

## API Documentation

### Specific Identification

```rust
pub fn calculate_specific_id(
    sale: &Transaction,
    selected_lot_ids: &[String],
    all_lots: &[TaxLot],
) -> Result<(Vec<Disposal>, Vec<TaxLot>)>
```

**Parameters:**
- `sale`: The sale transaction to process
- `selected_lot_ids`: User-selected lot IDs in processing order
- `all_lots`: All available tax lots for the asset

**Returns:**
- `Ok((disposals, updated_lots))`: Disposal records and updated lot states
- `Err`: Validation error or insufficient quantity

**JavaScript/NAPI:**
```typescript
function calculate_specific_identification(
  sale: JsTransaction,
  selected_lot_ids: string[],
  available_lots: JsTaxLot[]
): JsDisposalResult
```

### Average Cost

```rust
pub fn calculate_average_cost(
    sale: &Transaction,
    all_lots: &[TaxLot],
) -> Result<(Vec<Disposal>, Vec<TaxLot>)>
```

**Parameters:**
- `sale`: The sale transaction to process
- `all_lots`: All available tax lots for the asset

**Returns:**
- `Ok((disposals, updated_lots))`: Single disposal with average cost, proportionally updated lots
- `Err`: Validation error or insufficient quantity

**JavaScript/NAPI:**
```typescript
function calculate_average_cost_method(
  sale: JsTransaction,
  available_lots: JsTaxLot[]
): JsDisposalResult
```

## Usage Examples

### Specific Identification

```rust
use agentic_accounting_rust_core::*;

// Create sale transaction
let sale = Transaction {
    id: "sale_1".to_string(),
    transaction_type: TransactionType::Sell,
    asset: "BTC".to_string(),
    quantity: Decimal::from_str("1.5").unwrap(),
    price: Decimal::from_str("60000").unwrap(),
    timestamp: Utc::now(),
    source: "Coinbase".to_string(),
    fees: Decimal::ZERO,
};

// Available lots
let lots = vec![
    lot1, // High cost basis
    lot2, // Medium cost basis
    lot3, // Low cost basis
];

// Manually select lots for tax optimization
let selected_ids = vec![
    "lot1".to_string(), // Use highest cost first
    "lot3".to_string(), // Then lowest
];

// Calculate disposals
let (disposals, updated_lots) = calculate_specific_id(
    &sale,
    &selected_ids,
    &lots
)?;

// Process results
for disposal in disposals {
    println!("Lot: {}, Gain/Loss: {}",
        disposal.lot_id, disposal.gain_loss);
}
```

### Average Cost

```rust
use agentic_accounting_rust_core::*;

// Create sale transaction
let sale = Transaction {
    id: "sale_1".to_string(),
    transaction_type: TransactionType::Sell,
    asset: "BTC".to_string(),
    quantity: Decimal::from_str("1.0").unwrap(),
    price: Decimal::from_str("60000").unwrap(),
    timestamp: Utc::now(),
    source: "Coinbase".to_string(),
    fees: Decimal::ZERO,
};

// Available lots (e.g., from dollar-cost averaging)
let lots = vec![
    lot1, // Bought at $40k
    lot2, // Bought at $50k
    lot3, // Bought at $60k
];

// Calculate with average cost
let (disposals, updated_lots) = calculate_average_cost(&sale, &lots)?;

// Single disposal with weighted average
println!("Average cost basis: {}", disposals[0].cost_basis);
println!("Gain/Loss: {}", disposals[0].gain_loss);

// All lots proportionally reduced
for lot in updated_lots {
    println!("Lot {} remaining: {}", lot.id, lot.remaining_quantity);
}
```

## When to Use Each Method

### FIFO (First-In, First-Out)
- **Default method**: Simplest, most common
- **Tax impact**: Moderate
- **Use when**: No specific tax strategy needed

### LIFO (Last-In, First-Out)
- **Best for**: Rising markets
- **Tax impact**: Lower short-term gains in bull markets
- **Use when**: Minimizing current year taxes

### HIFO (Highest-In, First-Out)
- **Best for**: Tax minimization
- **Tax impact**: Lowest (uses highest cost lots first)
- **Use when**: Maximizing cost basis to reduce gains

### Specific Identification
- **Best for**: Maximum control
- **Tax impact**: Customizable by user selection
- **Use when**:
  - Complex tax situations
  - Strategic lot selection needed
  - Detailed record-keeping available
  - Tax-loss harvesting strategies

### Average Cost
- **Best for**: Mutual funds, cryptocurrency
- **Tax impact**: Smoothed across all purchases
- **Use when**:
  - IRS requires it (crypto, mutual funds)
  - Dollar-cost averaging strategy
  - Simplifying record-keeping
  - Equal weighting across purchases

## Edge Cases Handled

### Specific Identification
- ✅ Invalid lot IDs → Error
- ✅ Duplicate lot IDs → Error
- ✅ Insufficient quantity in selected lots → Error
- ✅ Zero quantity lots → Error
- ✅ Asset mismatch → Error
- ✅ Partial lot quantities → Correct calculation
- ✅ Mixed holding periods → Separate disposals

### Average Cost
- ✅ No available lots → Error
- ✅ Insufficient total quantity → Error
- ✅ Wrong asset → Error
- ✅ Zero quantity lots → Skipped
- ✅ Partially disposed lots → Correct averaging on remaining
- ✅ Rounding precision → Handled with epsilon
- ✅ Complete disposal → All lots reduced to ~zero

## Testing

### Run All Tests
```bash
cd packages/agentic-accounting-rust-core

# Run library tests
cargo test --lib

# Run specific ID tests
cargo test --test specific_id_test

# Run average cost tests
cargo test --test average_cost_test

# Run all tests
cargo test
```

### Test Coverage

**Specific Identification:**
- Single lot disposal (complete and partial)
- Multiple lot disposal (ordered)
- Tax optimization (HIFO-like manual selection)
- Error cases (invalid lot, insufficient qty, duplicates)
- Asset mismatch
- Mixed term gains (short vs long)
- Fee handling
- Decimal precision

**Average Cost:**
- Single lot
- Multiple lots (equal and different weights)
- Complete disposal
- Holding period (uses oldest lot)
- Partially disposed lots
- Error cases (insufficient qty, no lots, wrong asset)
- Fee handling
- Loss scenarios
- Many lots (5+)
- Rounding precision
- Dollar-cost averaging simulation

## Performance

Both methods are optimized for performance:
- **Specific ID**: O(n*m) where n=selected lots, m=all lots
- **Average Cost**: O(n) where n=number of lots
- **Memory**: Minimal allocations, in-place updates where possible
- **Precision**: Rust Decimal for financial accuracy

## Next Steps

### Remaining Methods (TODO)
1. **LIFO (Last-In, First-Out)**
   - Sort lots by acquisition date DESC
   - Process newest lots first
   - Similar structure to FIFO

2. **HIFO (Highest-In, First-Out)**
   - Sort lots by unit cost basis DESC
   - Process highest cost lots first
   - Tax optimization method

### Future Enhancements
- Wash sale detection integration
- Tax-loss harvesting recommendations
- Multi-year disposal tracking
- IRS Form 8949 generation
- Cost basis adjustment tracking

## References

- IRS Publication 550: Investment Income and Expenses
- IRS Notice 2014-21: Virtual Currency Guidance
- Pseudocode: `/plans/agentic-accounting/pseudocode/01-tax-calculation-algorithms.md`
- Architecture: `/plans/agentic-accounting/architecture/01-system-overview.md`
