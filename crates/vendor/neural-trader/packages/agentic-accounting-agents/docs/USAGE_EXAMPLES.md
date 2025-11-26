# TaxComputeAgent Usage Examples

## Quick Start

```typescript
import { TaxComputeAgent } from '@neural-trader/agentic-accounting-agents';

// Initialize agent
const agent = new TaxComputeAgent('my-tax-agent');
await agent.start();

// Run calculation
const result = await agent.execute({
  taskId: 'calc-1',
  description: 'Calculate BTC sale',
  priority: 'high',
  data: {
    sale: { /* transaction */ },
    lots: [ /* tax lots */ ],
  },
});

console.log(result.data?.calculation.netGainLoss);
```

## Example 1: Simple FIFO Calculation

```typescript
const agent = new TaxComputeAgent();
await agent.start();

const result = await agent.execute({
  taskId: 'example-1',
  description: 'Simple FIFO',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-001',
      transactionType: 'SELL',
      asset: 'BTC',
      quantity: '0.5',
      price: '50000.00',
      timestamp: '2024-01-15T12:00:00Z',
      source: 'coinbase',
      fees: '25.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '30000.00',
        acquisitionDate: '2023-06-01T12:00:00Z',
      },
    ],
    method: 'FIFO', // Explicitly specify method
    enableCache: true,
  },
});

// Expected: $10,000 long-term capital gain
console.log('Net Gain/Loss:', result.data?.calculation.netGainLoss);
console.log('Long-term?', result.data?.calculation.disposals[0].isLongTerm);
```

## Example 2: Intelligent Method Selection

```typescript
// Let the agent choose the best method
const result = await agent.execute({
  taskId: 'example-2',
  description: 'Auto-select method',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-002',
      transactionType: 'SELL',
      asset: 'ETH',
      quantity: '10.0',
      price: '3000.00',
      timestamp: '2024-01-20T12:00:00Z',
      source: 'binance',
      fees: '5.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'ETH',
        quantity: '5.0',
        remainingQuantity: '5.0',
        costBasis: '10000.00', // $2000 per ETH
        acquisitionDate: '2023-01-01T12:00:00Z',
      },
      {
        id: 'lot-002',
        transactionId: 'buy-002',
        asset: 'ETH',
        quantity: '5.0',
        remainingQuantity: '5.0',
        costBasis: '17500.00', // $3500 per ETH
        acquisitionDate: '2023-06-01T12:00:00Z',
      },
    ],
    profile: {
      jurisdiction: 'US',
      taxBracket: 'high', // 37% short-term, 20% long-term
      optimizationGoal: 'minimize_current_tax',
    },
  },
});

// Agent will likely choose HIFO to minimize gain
console.log('Selected Method:', result.data?.recommendation?.method);
console.log('Rationale:', result.data?.recommendation?.rationale);
console.log('Score:', result.data?.recommendation?.score);
console.log('Net Gain/Loss:', result.data?.calculation.netGainLoss);
```

## Example 3: Multi-Method Comparison

```typescript
// Compare all methods to see savings
const result = await agent.execute({
  taskId: 'example-3',
  description: 'Compare all methods',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-003',
      transactionType: 'SELL',
      asset: 'BTC',
      quantity: '2.0',
      price: '45000.00',
      timestamp: '2024-02-01T12:00:00Z',
      source: 'kraken',
      fees: '50.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '25000.00',
        acquisitionDate: '2022-01-01T12:00:00Z',
      },
      {
        id: 'lot-002',
        transactionId: 'buy-002',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '35000.00',
        acquisitionDate: '2022-06-01T12:00:00Z',
      },
      {
        id: 'lot-003',
        transactionId: 'buy-003',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '40000.00',
        acquisitionDate: '2023-01-01T12:00:00Z',
      },
    ],
    method: 'FIFO', // Will use FIFO but also compare others
    compareAll: true, // Enable comparison
  },
});

console.log('Using Method:', result.data?.calculation.method);
console.log('Net Gain/Loss:', result.data?.calculation.netGainLoss);
console.log('\nComparison:');
console.log('Best Method:', result.data?.comparison.best);
console.log('Savings vs Worst:', result.data?.comparison.savings);
console.log('\nAll Methods:');
result.data?.comparison.comparison.forEach(c => {
  console.log(`${c.rank}. ${c.method}: Gain=${c.gain}, Tax=${c.tax}`);
});
```

## Example 4: Wash Sale Detection

```typescript
// Detect potential wash sales
const result = await agent.execute({
  taskId: 'example-4',
  description: 'Detect wash sales',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-004',
      transactionType: 'SELL',
      asset: 'BTC',
      quantity: '1.0',
      price: '25000.00', // Selling at loss
      timestamp: '2024-01-15T12:00:00Z',
      source: 'coinbase',
      fees: '10.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '40000.00', // Loss of $15,000
        acquisitionDate: '2024-01-01T12:00:00Z', // Within 30 days!
      },
    ],
    method: 'FIFO',
    detectWashSales: true, // Enable detection
  },
});

console.log('Net Gain/Loss:', result.data?.calculation.netGainLoss);
console.log('Wash Sales Detected:', result.data?.washSales?.length || 0);

if (result.data?.washSales && result.data.washSales.length > 0) {
  result.data.washSales.forEach(ws => {
    console.log(`Warning: ${ws.warning}`);
    console.log(`  Asset: ${ws.asset}`);
    console.log(`  Loss: ${ws.loss}`);
    console.log(`  Disposal: ${ws.disposalDate}`);
    console.log(`  Acquisition: ${ws.acquisitionDate}`);
  });
}
```

## Example 5: Specific ID Method

```typescript
// Use Specific Identification
const result = await agent.execute({
  taskId: 'example-5',
  description: 'Specific ID calculation',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-005',
      transactionType: 'SELL',
      asset: 'ETH',
      quantity: '5.0',
      price: '3200.00',
      timestamp: '2024-01-25T12:00:00Z',
      source: 'gemini',
      fees: '8.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'ETH',
        quantity: '2.0',
        remainingQuantity: '2.0',
        costBasis: '4000.00', // $2000 per ETH
        acquisitionDate: '2023-01-01T12:00:00Z',
      },
      {
        id: 'lot-002',
        transactionId: 'buy-002',
        asset: 'ETH',
        quantity: '3.0',
        remainingQuantity: '3.0',
        costBasis: '9000.00', // $3000 per ETH
        acquisitionDate: '2023-06-01T12:00:00Z',
      },
      {
        id: 'lot-003',
        transactionId: 'buy-003',
        asset: 'ETH',
        quantity: '5.0',
        remainingQuantity: '5.0',
        costBasis: '15000.00', // $3000 per ETH
        acquisitionDate: '2023-08-01T12:00:00Z',
      },
    ],
    method: 'SPECIFIC_ID',
    // Note: In production, you'd specify which lots to use
    // The wrapper will use first N lots by default
  },
});

console.log('Method:', result.data?.calculation.method);
console.log('Disposals:', result.data?.calculation.disposals.length);
result.data?.calculation.disposals.forEach(d => {
  console.log(`  Lot ${d.lotId}: ${d.quantity} @ ${d.costBasis} = ${d.gainLoss}`);
});
```

## Example 6: Average Cost Method

```typescript
// Use Average Cost basis
const result = await agent.execute({
  taskId: 'example-6',
  description: 'Average cost calculation',
  priority: 'high',
  data: {
    sale: {
      id: 'sale-006',
      transactionType: 'SELL',
      asset: 'BTC',
      quantity: '1.5',
      price: '48000.00',
      timestamp: '2024-02-10T12:00:00Z',
      source: 'bitstamp',
      fees: '30.00',
    },
    lots: [
      {
        id: 'lot-001',
        transactionId: 'buy-001',
        asset: 'BTC',
        quantity: '0.5',
        remainingQuantity: '0.5',
        costBasis: '15000.00', // $30k per BTC
        acquisitionDate: '2023-01-01T12:00:00Z',
      },
      {
        id: 'lot-002',
        transactionId: 'buy-002',
        asset: 'BTC',
        quantity: '1.0',
        remainingQuantity: '1.0',
        costBasis: '35000.00', // $35k per BTC
        acquisitionDate: '2023-06-01T12:00:00Z',
      },
      {
        id: 'lot-003',
        transactionId: 'buy-003',
        asset: 'BTC',
        quantity: '0.5',
        remainingQuantity: '0.5',
        costBasis: '20000.00', // $40k per BTC
        acquisitionDate: '2023-09-01T12:00:00Z',
      },
    ],
    method: 'AVERAGE_COST',
  },
});

// Average cost = (15000 + 35000 + 20000) / 2.0 = $35,000 per BTC
console.log('Method:', result.data?.calculation.method);
console.log('Net Gain/Loss:', result.data?.calculation.netGainLoss);
console.log('Total Proceeds:', result.data?.calculation.disposals[0].proceeds);
console.log('Total Cost Basis:', result.data?.calculation.disposals[0].costBasis);
```

## Example 7: Caching Performance

```typescript
// First call - will calculate
console.time('First Call');
const result1 = await agent.execute({
  taskId: 'cache-test-1',
  description: 'First calculation',
  priority: 'high',
  data: {
    sale: { /* ... */ },
    lots: [ /* ... */ ],
    method: 'FIFO',
    enableCache: true, // Enable caching
  },
});
console.timeEnd('First Call');
console.log('Cache Hit:', result1.data?.cacheHit); // false
console.log('Calculation Time:', result1.data?.performance.calculationTime);

// Second call - will use cache
console.time('Second Call');
const result2 = await agent.execute({
  taskId: 'cache-test-2',
  description: 'Cached calculation',
  priority: 'high',
  data: {
    sale: { /* same as above */ },
    lots: [ /* same as above */ ],
    method: 'FIFO',
    enableCache: true,
  },
});
console.timeEnd('Second Call');
console.log('Cache Hit:', result2.data?.cacheHit); // true
console.log('Calculation Time:', result2.data?.performance.calculationTime); // 0

// Get cache statistics
const stats = agent.getCacheStats();
console.log('Cache Stats:', {
  size: stats.size,
  hits: stats.hits,
  misses: stats.misses,
  hitRate: (stats.hitRate * 100).toFixed(1) + '%',
});
```

## Example 8: Error Handling

```typescript
try {
  const result = await agent.execute({
    taskId: 'error-test',
    description: 'Test error handling',
    priority: 'high',
    data: {
      sale: {
        id: 'sale-999',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '10.0', // Selling more than available!
        price: '50000.00',
        timestamp: '2024-01-15T12:00:00Z',
        source: 'test',
        fees: '0.00',
      },
      lots: [
        {
          id: 'lot-001',
          transactionId: 'buy-001',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T12:00:00Z',
        },
      ],
      method: 'FIFO',
    },
  });
} catch (error) {
  console.error('Calculation failed:', error.message);
  // Expected: "Insufficient quantity: need 10.0, have 1.0"
}
```

## Example 9: Market Condition Analysis

```typescript
import { StrategySelector } from '@neural-trader/agentic-accounting-agents';

const selector = new StrategySelector();

// Get recommendation with market analysis
const recommendation = await selector.selectOptimalMethod(
  {
    id: 'sale-007',
    transactionType: 'SELL',
    asset: 'BTC',
    quantity: '1.0',
    price: '55000.00',
    timestamp: '2024-03-01T12:00:00Z',
    source: 'test',
    fees: '0.00',
  },
  [
    { /* lot 1 - $30k basis */ },
    { /* lot 2 - $40k basis */ },
    { /* lot 3 - $50k basis */ },
  ],
  {
    jurisdiction: 'US',
    taxBracket: 'high',
    optimizationGoal: 'minimize_current_tax',
  }
);

console.log('Recommended Method:', recommendation.method);
console.log('Score:', recommendation.score);
console.log('Rationale:', recommendation.rationale);
console.log('Estimated Savings:', recommendation.estimatedSavings);
console.log('Alternatives:', recommendation.alternatives);
```

## Example 10: Full Production Workflow

```typescript
import { TaxComputeAgent } from '@neural-trader/agentic-accounting-agents';

async function calculateTaxes(userId: string, saleTransaction: any) {
  // Initialize agent
  const agent = new TaxComputeAgent(`tax-agent-${userId}`);
  await agent.start();

  try {
    // Load user's tax profile
    const profile = await loadUserTaxProfile(userId);

    // Load available tax lots for the asset
    const lots = await loadUserTaxLots(userId, saleTransaction.asset);

    // Execute calculation with all features
    const result = await agent.execute({
      taskId: `calc-${saleTransaction.id}`,
      description: `Calculate taxes for ${saleTransaction.asset} sale`,
      priority: 'high',
      data: {
        sale: saleTransaction,
        lots,
        profile,
        compareAll: true, // Compare all methods
        enableCache: true, // Use caching
        detectWashSales: true, // Detect wash sales
      },
    });

    if (!result.success) {
      throw new Error(`Calculation failed: ${result.error?.message}`);
    }

    // Log results
    console.log('Calculation complete:');
    console.log('  Method:', result.data?.calculation.method);
    console.log('  Net Gain/Loss:', result.data?.calculation.netGainLoss);
    console.log('  Short-term:', result.data?.calculation.shortTermGain);
    console.log('  Long-term:', result.data?.calculation.longTermGain);
    console.log('  Performance:', result.data?.performance.totalTime + 'ms');

    if (result.data?.washSales && result.data.washSales.length > 0) {
      console.warn('  ⚠️  Wash sales detected:', result.data.washSales.length);
    }

    if (result.data?.comparison) {
      console.log('  💡 Best method:', result.data.comparison.best);
      console.log('  💰 Savings vs worst:', result.data.comparison.savings);
    }

    // Store results
    await saveTaxCalculation(userId, saleTransaction.id, result.data);

    return result.data;
  } finally {
    await agent.stop();
  }
}

// Helper functions (implement based on your data layer)
async function loadUserTaxProfile(userId: string) {
  // Load from database
  return {
    jurisdiction: 'US',
    taxBracket: 'high',
    optimizationGoal: 'minimize_current_tax',
  };
}

async function loadUserTaxLots(userId: string, asset: string) {
  // Load from database
  return [];
}

async function saveTaxCalculation(userId: string, saleId: string, data: any) {
  // Save to database
}
```

## Performance Tips

1. **Enable Caching**: Always use `enableCache: true` for repeated calculations
2. **Batch Operations**: Process multiple sales in parallel using Promise.all()
3. **Cleanup Cache**: Periodically run `agent.invalidateCache()` to free memory
4. **Monitor Stats**: Check `agent.getCacheStats()` to optimize cache settings
5. **Reuse Agents**: Keep agent instances alive for multiple calculations

## Error Handling Best Practices

1. **Always check `result.success`** before accessing `result.data`
2. **Log errors** to ReasoningBank for learning
3. **Validate inputs** before calling the agent (use ValidationError)
4. **Handle network errors** when loading data
5. **Set appropriate timeouts** for long-running calculations

---

**Next**: See `TAX_COMPUTE_AGENT.md` for detailed implementation documentation.
