# Agentic Accounting Rust Core

High-performance Rust addon for crypto tax calculations using NAPI-RS.

## Features

- **Precise Decimal Arithmetic**: Uses `rust_decimal` to avoid floating-point errors
- **Date/Time Utilities**: Handles ISO 8601 dates and wash sale period calculations
- **Type Safety**: Strongly-typed Transaction, TaxLot, and Disposal structs
- **Performance**: Compiled Rust code for CPU-intensive operations
- **Node.js Bindings**: Seamless integration with TypeScript via NAPI-RS

## Installation

```bash
npm install @neural-trader/agentic-accounting-rust-core
```

## Building from Source

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Node.js 16+ and npm

### Build Steps

```bash
# Install dependencies
npm install

# Build the Rust addon
npm run build

# Run Rust tests
npm test

# Run benchmarks
npm run bench
```

## Usage

```typescript
import {
  addDecimals,
  subtractDecimals,
  multiplyDecimals,
  divideDecimals,
  calculateGainLoss,
  daysBetwee,
  isWithinWashSalePeriod,
  createTransaction,
  createTaxLot,
  createDisposal,
} from '@neural-trader/agentic-accounting-rust-core';

// Decimal math (no floating-point errors!)
const sum = addDecimals('10.50', '5.25'); // "15.75"
const product = multiplyDecimals('0.00000001', '1000000'); // "0.01"

// Gain/loss calculation
const gainLoss = calculateGainLoss(
  '50000', // sale price
  '1.5',   // sale quantity
  '40000', // cost basis
  '1.5'    // cost quantity
); // "15000" (gain)

// Date utilities
const days = daysBetween('2024-01-01T00:00:00Z', '2024-01-31T00:00:00Z'); // 30
const isWash = isWithinWashSalePeriod('2024-01-15T00:00:00Z', '2024-01-20T00:00:00Z'); // true

// Type conversions
const transaction = createTransaction({
  id: 'tx1',
  transaction_type: 'BUY',
  asset: 'BTC',
  quantity: '1.5',
  price: '50000',
  timestamp: '2024-01-15T10:30:00Z',
  source: 'Coinbase',
  fees: '10.50',
});
```

## API Reference

### Decimal Math

- `addDecimals(a: string, b: string): string`
- `subtractDecimals(a: string, b: string): string`
- `multiplyDecimals(a: string, b: string): string`
- `divideDecimals(a: string, b: string): string`
- `calculateGainLoss(salePrice: string, saleQty: string, costBasis: string, costQty: string): string`

### Date/Time

- `parseDateTime(dateStr: string): string`
- `formatDateTime(dateStr: string): string`
- `daysBetween(date1: string, date2: string): number`
- `isWithinWashSalePeriod(saleDate: string, purchaseDate: string): boolean`
- `addDays(dateStr: string, days: number): string`
- `subtractDays(dateStr: string, days: number): string`
- `getTaxYearStart(dateStr: string): string`
- `getTaxYearEnd(dateStr: string): string`

### Types

- `createTransaction(tx: JsTransaction): JsTransaction`
- `createTaxLot(lot: JsTaxLot): JsTaxLot`
- `createDisposal(disposal: JsDisposal): JsDisposal`

## Performance

Rust provides significant performance advantages for:

- High-volume transaction processing
- Complex tax lot matching algorithms
- Precise decimal arithmetic (no rounding errors)
- Date/time calculations for wash sale detection
- PDF generation and cryptographic operations

Benchmarks show 10-100x speedup vs pure JavaScript for number-crunching tasks.

## Architecture

```
src/
├── lib.rs          # NAPI entry point
├── error.rs        # Error types
├── math.rs         # Decimal arithmetic
├── datetime.rs     # Date/time utilities
├── types.rs        # Core data types
└── tax/
    ├── mod.rs
    └── calculator.rs  # Tax calculations (to be implemented)
```

## Development

### Running Tests

```bash
cargo test --all-features
```

### Running Benchmarks

```bash
cargo bench
```

### Building for Production

```bash
npm run build
```

This compiles with full optimizations (LTO, single codegen unit, opt-level=3).

## License

MIT

## Contributing

This package is part of the Neural Trader agentic accounting system. Tax calculation algorithms are being implemented by specialized agents in Phase 2 of development.

## Roadmap

- [x] Basic types and decimal math
- [x] Date/time utilities
- [x] Wash sale period detection
- [ ] FIFO implementation (Phase 2 - Tax Algorithm Agent 1)
- [ ] LIFO implementation (Phase 2 - Tax Algorithm Agent 2)
- [ ] HIFO implementation (Phase 2 - Tax Algorithm Agent 3)
- [ ] Specific ID implementation (Phase 2 - Tax Algorithm Agent 4)
- [ ] Average cost implementation (Phase 2 - Tax Algorithm Agent 5)
- [ ] Wash sale detection (Phase 2 - Tax Algorithm Agent 6)
- [ ] PDF generation (Phase 3)
- [ ] Cryptographic operations (Phase 3)
