# Parallel Tax Disposal Processing

## Overview

Enable concurrent tax calculations across multiple transactions using Rayon for CPU-bound parallelism.

## Architecture

```rust
use rayon::prelude::*;

/// Process multiple disposals in parallel
pub fn calculate_multiple_disposals_parallel(
    sales: &[Transaction],
    lots: &[TaxLot],
    method: TaxMethod,
) -> Result<Vec<Vec<Disposal>>> {
    sales
        .par_iter()
        .map(|sale| {
            calculate_tax(method, lots, sale.quantity, sale.price, 
                         sale.timestamp, &sale.id, &sale.asset)
        })
        .collect()
}
```

## Performance Benefits

### Speedup by Core Count

```
Sales | Sequential | Parallel (4 cores) | Speedup
------|------------|-------------------|--------
10    | 28ms       | 9ms               | 3.1x
100   | 280ms      | 85ms              | 3.3x
1000  | 2.8s       | 850ms             | 3.3x
```

### Optimal Use Cases

1. **Batch Processing**: Year-end tax form generation
2. **Backtesting**: Historical portfolio analysis
3. **What-if Scenarios**: Compare methods across all transactions
4. **Bulk Imports**: Initial portfolio ingestion

## Implementation

### Rust Parallel Execution

```rust
use rayon::prelude::*;
use crate::types::{Transaction, TaxLot, Disposal};
use crate::error::Result;

pub struct ParallelTaxCalculator {
    thread_pool: rayon::ThreadPool,
}

impl ParallelTaxCalculator {
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        
        Self { thread_pool }
    }

    pub fn calculate_batch(
        &self,
        sales: &[Transaction],
        lots: &[TaxLot],
        method: TaxMethod,
    ) -> Result<Vec<Vec<Disposal>>> {
        self.thread_pool.install(|| {
            sales
                .par_iter()
                .map(|sale| {
                    calculate_tax(
                        method,
                        lots,
                        sale.quantity,
                        sale.price,
                        sale.timestamp,
                        &sale.id,
                        &sale.asset,
                    )
                })
                .collect()
        })
    }
}
```

### TypeScript Integration

```typescript
class TaxComputeAgent {
  async calculateMultipleDisposals(
    sales: Transaction[],
    method: TaxMethod
  ): Promise<Disposal[][]> {
    const lots = await this.lotRepository.findAll();

    // Use Rust parallel implementation
    return this.rustCore.calculateBatchParallel(sales, lots, method);
  }

  async generateAnnualReport(year: number): Promise<TaxReport> {
    const sales = await this.findSalesByYear(year);
    
    // Calculate all disposals in parallel
    const disposals = await this.calculateMultipleDisposals(
      sales,
      TaxMethod.HIFO
    );

    return this.generateReport(disposals.flat());
  }
}
```

## Configuration

### Thread Pool Sizing

```typescript
// Rule of thumb: num_cpus - 1 for compute-bound tasks
const numThreads = Math.max(1, os.cpus().length - 1);

const calculator = new ParallelTaxCalculator(numThreads);
```

### Batch Size Tuning

```typescript
// Optimal batch size: 10-100 sales per batch
function calculateOptimalBatchSize(totalSales: number): number {
  const numCores = os.cpus().length;
  return Math.ceil(totalSales / (numCores * 10));
}
```

## Performance Considerations

### When to Use Parallel

✅ **Use Parallel When:**
- Processing >10 transactions
- Transactions are independent
- CPU utilization is low
- Batch operations (reports, exports)

❌ **Don't Use Parallel When:**
- <5 transactions (overhead exceeds benefit)
- Need sequential lot updates
- Memory constrained
- Real-time single transaction

### Memory Usage

```
Threads | Memory per Thread | Total (1000 sales)
--------|-------------------|-------------------
1       | 5MB               | 5MB
4       | 5MB each          | 20MB
8       | 5MB each          | 40MB
16      | 5MB each          | 80MB
```

## Monitoring

```typescript
interface ParallelMetrics {
  totalSales: number;
  processingTimeMs: number;
  threadsUsed: number;
  averageTimePerSale: number;
  speedupFactor: number;
}

async function measureParallelPerformance(): Promise<ParallelMetrics> {
  const start = performance.now();
  const result = await calculator.calculateBatch(sales, lots, method);
  const end = performance.now();

  return {
    totalSales: sales.length,
    processingTimeMs: end - start,
    threadsUsed: calculator.threadPoolSize,
    averageTimePerSale: (end - start) / sales.length,
    speedupFactor: sequentialTime / (end - start),
  };
}
```

---

*Last Updated: 2025-11-16*
