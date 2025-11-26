/**
 * Test Data Factories
 *
 * Provides factory functions for creating test data objects
 * with sensible defaults and easy overrides.
 */

import { randomUUID } from 'crypto';
import Decimal from 'decimal.js';

// ============================================================================
// Transaction Factory
// ============================================================================

export interface TransactionData {
  id?: string;
  type: 'BUY' | 'SELL' | 'TRADE' | 'INCOME' | 'TRANSFER';
  asset: string;
  quantity: number | string;
  price: number | string;
  timestamp: Date | string;
  source?: string;
  fees?: number | string;
  notes?: string;
}

export function createTransaction(overrides: Partial<TransactionData> = {}): TransactionData {
  return {
    id: randomUUID(),
    type: 'BUY',
    asset: 'BTC',
    quantity: new Decimal(1.0).toString(),
    price: new Decimal(50000).toString(),
    timestamp: new Date('2023-01-01T00:00:00Z'),
    source: 'coinbase',
    fees: new Decimal(10).toString(),
    notes: '',
    ...overrides,
  };
}

export function createBuyTransaction(overrides: Partial<TransactionData> = {}): TransactionData {
  return createTransaction({ type: 'BUY', ...overrides });
}

export function createSellTransaction(overrides: Partial<TransactionData> = {}): TransactionData {
  return createTransaction({ type: 'SELL', ...overrides });
}

export function generateTransactions(count: number, overrides: Partial<TransactionData> = {}): TransactionData[] {
  return Array.from({ length: count }, (_, i) =>
    createTransaction({
      timestamp: new Date(Date.UTC(2023, 0, 1 + i)),
      ...overrides,
    })
  );
}

// ============================================================================
// TaxLot Factory
// ============================================================================

export interface TaxLotData {
  id?: string;
  asset: string;
  quantity: number | string;
  costBasis: number | string;
  acquiredDate: Date | string;
  transactionId?: string;
  disposed?: boolean;
  disposedDate?: Date | string;
}

export function createLot(overrides: Partial<TaxLotData> = {}): TaxLotData {
  return {
    id: randomUUID(),
    asset: 'BTC',
    quantity: new Decimal(1.0).toString(),
    costBasis: new Decimal(50000).toString(),
    acquiredDate: new Date('2023-01-01T00:00:00Z'),
    transactionId: randomUUID(),
    disposed: false,
    ...overrides,
  };
}

export function generateLots(count: number, overrides: Partial<TaxLotData> = {}): TaxLotData[] {
  return Array.from({ length: count }, (_, i) =>
    createLot({
      acquiredDate: new Date(Date.UTC(2023, 0, 1 + i)),
      costBasis: new Decimal(50000 + i * 100).toString(),
      ...overrides,
    })
  );
}

// ============================================================================
// Disposal Factory
// ============================================================================

export interface DisposalData {
  id?: string;
  asset: string;
  quantity: number | string;
  proceeds: number | string;
  costBasis: number | string;
  gain: number | string;
  disposalDate: Date | string;
  acquiredDate: Date | string;
  term: 'SHORT' | 'LONG';
  washSale?: boolean;
  disallowedLoss?: number | string;
}

export function createDisposal(overrides: Partial<DisposalData> = {}): DisposalData {
  const acquiredDate = new Date('2022-01-01T00:00:00Z');
  const disposalDate = new Date('2023-06-01T00:00:00Z');
  const costBasis = new Decimal(50000);
  const proceeds = new Decimal(55000);
  const gain = proceeds.minus(costBasis);

  const daysDiff = Math.floor((disposalDate.getTime() - acquiredDate.getTime()) / (1000 * 60 * 60 * 24));
  const term = daysDiff > 365 ? 'LONG' : 'SHORT';

  return {
    id: randomUUID(),
    asset: 'BTC',
    quantity: new Decimal(1.0).toString(),
    proceeds: proceeds.toString(),
    costBasis: costBasis.toString(),
    gain: gain.toString(),
    disposalDate,
    acquiredDate,
    term,
    washSale: false,
    ...overrides,
  };
}

export function createWashSaleDisposal(overrides: Partial<DisposalData> = {}): DisposalData {
  return createDisposal({
    gain: new Decimal(-1000).toString(), // Loss
    washSale: true,
    disallowedLoss: new Decimal(1000).toString(),
    ...overrides,
  });
}

// ============================================================================
// Position Factory
// ============================================================================

export interface PositionData {
  id?: string;
  asset: string;
  totalQuantity: number | string;
  averageCostBasis: number | string;
  totalCostBasis: number | string;
  currentValue?: number | string;
  unrealizedGain?: number | string;
  lots: TaxLotData[];
}

export function createPosition(overrides: Partial<PositionData> = {}): PositionData {
  const lots = overrides.lots || generateLots(3);
  const totalQuantity = lots.reduce((sum, lot) => sum.plus(new Decimal(lot.quantity)), new Decimal(0));
  const totalCostBasis = lots.reduce((sum, lot) => sum.plus(new Decimal(lot.costBasis)), new Decimal(0));
  const averageCostBasis = totalCostBasis.dividedBy(totalQuantity);

  return {
    id: randomUUID(),
    asset: 'BTC',
    totalQuantity: totalQuantity.toString(),
    averageCostBasis: averageCostBasis.toString(),
    totalCostBasis: totalCostBasis.toString(),
    lots,
    ...overrides,
  };
}

// ============================================================================
// Compliance Rule Factory
// ============================================================================

export interface ComplianceRuleData {
  id?: string;
  name: string;
  description: string;
  ruleType: 'WASH_SALE' | 'LIMIT' | 'RESTRICTION' | 'REPORTING';
  enabled: boolean;
  severity: 'INFO' | 'WARNING' | 'ERROR';
  conditions: Record<string, any>;
}

export function createComplianceRule(overrides: Partial<ComplianceRuleData> = {}): ComplianceRuleData {
  return {
    id: randomUUID(),
    name: 'Wash Sale Detection',
    description: 'Detects wash sale violations within 30-day window',
    ruleType: 'WASH_SALE',
    enabled: true,
    severity: 'WARNING',
    conditions: {
      windowDays: 30,
      checkReplacements: true,
    },
    ...overrides,
  };
}

// ============================================================================
// Audit Entry Factory
// ============================================================================

export interface AuditEntryData {
  id?: string;
  timestamp: Date | string;
  action: string;
  entity: string;
  entityId: string;
  userId?: string;
  changes?: Record<string, any>;
  hash?: string;
  previousHash?: string;
}

export function createAuditEntry(overrides: Partial<AuditEntryData> = {}): AuditEntryData {
  return {
    id: randomUUID(),
    timestamp: new Date(),
    action: 'CREATE',
    entity: 'Transaction',
    entityId: randomUUID(),
    userId: 'test-user',
    changes: {},
    ...overrides,
  };
}

export function generateAuditEntries(count: number): AuditEntryData[] {
  const entries: AuditEntryData[] = [];
  let previousHash = '0000000000000000';

  for (let i = 0; i < count; i++) {
    const entry = createAuditEntry({
      timestamp: new Date(Date.now() + i * 1000),
      previousHash,
    });

    // Simple hash simulation (replace with actual hash in implementation)
    entry.hash = `hash-${i}-${entry.id}`;
    previousHash = entry.hash;

    entries.push(entry);
  }

  return entries;
}

// ============================================================================
// Statistical Test Data Generators
// ============================================================================

export function generateNormalTransactions(
  count: number,
  params: { mean: number; stdDev: number }
): TransactionData[] {
  // Box-Muller transform for normal distribution
  const normalRandom = () => {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * params.stdDev + params.mean;
  };

  return Array.from({ length: count }, () =>
    createTransaction({
      price: Math.abs(normalRandom()).toString(),
    })
  );
}

export function generateTestLots(count: number): TaxLotData[] {
  return generateLots(count, {
    costBasis: new Decimal(Math.random() * 100000).toString(),
  });
}

export function generateTestSale(overrides: Partial<TransactionData> = {}): TransactionData {
  return createSellTransaction({
    quantity: new Decimal(Math.random() * 10).toString(),
    ...overrides,
  });
}

// ============================================================================
// Mock Data Generators
// ============================================================================

export function generateRandomVector(dimensions: number): number[] {
  return Array.from({ length: dimensions }, () => Math.random());
}

export function createMockFraudSignature() {
  return {
    id: randomUUID(),
    name: 'Suspicious Large Transfer',
    embedding: generateRandomVector(768),
    severity: 'HIGH',
    description: 'Large transfer to unverified destination',
  };
}

export function loadFraudSignatures() {
  return Promise.resolve([
    createMockFraudSignature(),
    createMockFraudSignature(),
    createMockFraudSignature(),
  ]);
}

// ============================================================================
// Complex Tax Scenario Factories (Phase 2)
// ============================================================================

export interface ComplexTaxScenarioData {
  transactions: TransactionData[];
  expectedDisposals: DisposalData[];
  description: string;
}

export function createComplexTaxScenario(): ComplexTaxScenarioData {
  // Multi-lot, multi-year scenario with various cost bases
  const baseDate = new Date('2020-01-01');

  const transactions = [
    // Year 1: Accumulation phase
    createBuyTransaction({
      asset: 'BTC',
      quantity: '5',
      price: '8000',
      timestamp: new Date(baseDate.getTime() + 30 * 24 * 60 * 60 * 1000)
    }),
    createBuyTransaction({
      asset: 'BTC',
      quantity: '3',
      price: '9000',
      timestamp: new Date(baseDate.getTime() + 90 * 24 * 60 * 60 * 1000)
    }),

    // Year 2: More accumulation
    createBuyTransaction({
      asset: 'BTC',
      quantity: '2',
      price: '35000',
      timestamp: new Date(baseDate.getTime() + 400 * 24 * 60 * 60 * 1000)
    }),

    // Year 3: Strategic sales
    createSellTransaction({
      asset: 'BTC',
      quantity: '4',
      price: '45000',
      timestamp: new Date(baseDate.getTime() + 800 * 24 * 60 * 60 * 1000)
    }),

    // More buys
    createBuyTransaction({
      asset: 'BTC',
      quantity: '1',
      price: '42000',
      timestamp: new Date(baseDate.getTime() + 820 * 24 * 60 * 60 * 1000)
    }),

    // Final sale
    createSellTransaction({
      asset: 'BTC',
      quantity: '3',
      price: '50000',
      timestamp: new Date(baseDate.getTime() + 1000 * 24 * 60 * 60 * 1000)
    }),
  ];

  return {
    transactions,
    expectedDisposals: [], // To be calculated
    description: 'Multi-year Bitcoin trading with varying cost bases',
  };
}

export function createWashSaleScenario(): ComplexTaxScenarioData {
  // Classic wash sale: sell at loss, rebuy within 30 days
  const transactions = [
    createBuyTransaction({
      asset: 'ETH',
      quantity: '100',
      price: '3000',
      timestamp: new Date('2023-05-01'),
    }),
    createSellTransaction({
      asset: 'ETH',
      quantity: '100',
      price: '2000', // $100k loss
      timestamp: new Date('2023-06-15'),
    }),
    createBuyTransaction({
      asset: 'ETH',
      quantity: '100',
      price: '1900',
      timestamp: new Date('2023-06-25'), // 10 days later - wash sale!
    }),
  ];

  return {
    transactions,
    expectedDisposals: [
      createWashSaleDisposal({
        asset: 'ETH',
        quantity: '100',
        costBasis: '300000',
        proceeds: '200000',
        gain: '-100000',
        disposalDate: new Date('2023-06-15'),
        acquiredDate: new Date('2023-05-01'),
        washSale: true,
        disallowedLoss: '100000',
      }),
    ],
    description: 'Wash sale scenario with loss disallowance',
  };
}

export function createIRSExampleData(exampleNumber: number): ComplexTaxScenarioData {
  switch (exampleNumber) {
    case 1:
      // IRS Pub 550 Example 1: Basic FIFO
      return {
        transactions: [
          createBuyTransaction({
            asset: 'XYZ',
            quantity: '100',
            price: '20',
            timestamp: new Date('2023-01-03'),
          }),
          createBuyTransaction({
            asset: 'XYZ',
            quantity: '100',
            price: '30',
            timestamp: new Date('2023-02-01'),
          }),
          createSellTransaction({
            asset: 'XYZ',
            quantity: '150',
            price: '40',
            timestamp: new Date('2023-06-30'),
          }),
        ],
        expectedDisposals: [
          createDisposal({
            asset: 'XYZ',
            quantity: '100',
            costBasis: '2000',
            proceeds: '4000',
            gain: '2000',
            term: 'SHORT',
          }),
          createDisposal({
            asset: 'XYZ',
            quantity: '50',
            costBasis: '1500',
            proceeds: '2000',
            gain: '500',
            term: 'SHORT',
          }),
        ],
        description: 'IRS Publication 550 Example 1',
      };

    default:
      return createComplexTaxScenario();
  }
}

export function createPerformanceTestData(lotCount: number): TaxLotData[] {
  // Generate large dataset for performance testing
  const baseDate = new Date('2020-01-01');
  const lots: TaxLotData[] = [];

  for (let i = 0; i < lotCount; i++) {
    const daysOffset = Math.floor(i / 3); // ~3 lots per day
    const priceVariation = (i % 100) * 100; // Price varies

    lots.push(createLot({
      asset: 'BTC',
      quantity: new Decimal(Math.random() * 0.1).toString(),
      costBasis: new Decimal(30000 + priceVariation).toString(),
      acquiredDate: new Date(baseDate.getTime() + daysOffset * 24 * 60 * 60 * 1000),
    }));
  }

  return lots;
}

export function createMultiAssetPortfolio(): {
  assets: string[];
  transactionsByAsset: Map<string, TransactionData[]>;
} {
  const assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC'];
  const transactionsByAsset = new Map<string, TransactionData[]>();

  assets.forEach((asset, index) => {
    const basePrice = [40000, 2000, 100, 20, 0.5][index];
    const transactions = generateTransactions(20, { asset });

    // Mix of buys and sells
    transactions.forEach((tx, i) => {
      tx.type = i % 4 === 0 ? 'SELL' : 'BUY';
      tx.price = new Decimal(basePrice * (1 + (Math.random() - 0.5) * 0.2)).toString();
    });

    transactionsByAsset.set(asset, transactions);
  });

  return {
    assets,
    transactionsByAsset,
  };
}

export function createLongTermVsShortTermScenario(): ComplexTaxScenarioData {
  // Scenario demonstrating tax optimization between long/short term
  const transactions = [
    // Old lot (will be long-term)
    createBuyTransaction({
      asset: 'BTC',
      quantity: '5',
      price: '30000',
      timestamp: new Date('2022-01-01'),
    }),

    // Recent lot (will be short-term)
    createBuyTransaction({
      asset: 'BTC',
      quantity: '5',
      price: '55000',
      timestamp: new Date('2023-10-01'),
    }),

    // Sale after one year from first lot
    createSellTransaction({
      asset: 'BTC',
      quantity: '3',
      price: '50000',
      timestamp: new Date('2023-12-01'),
    }),
  ];

  return {
    transactions,
    expectedDisposals: [
      // FIFO uses old lot (long-term gain)
      createDisposal({
        asset: 'BTC',
        quantity: '3',
        costBasis: '90000',
        proceeds: '150000',
        gain: '60000',
        term: 'LONG',
        acquiredDate: new Date('2022-01-01'),
        disposalDate: new Date('2023-12-01'),
      }),
    ],
    description: 'Long-term vs short-term capital gains',
  };
}

export function createFractionalCryptoScenario(): ComplexTaxScenarioData {
  // High-precision fractional quantities (realistic crypto)
  const transactions = [
    createBuyTransaction({
      asset: 'BTC',
      quantity: '0.12345678',
      price: '42000',
      timestamp: new Date('2023-01-15'),
    }),
    createBuyTransaction({
      asset: 'BTC',
      quantity: '0.98765432',
      price: '48000',
      timestamp: new Date('2023-03-20'),
    }),
    createSellTransaction({
      asset: 'BTC',
      quantity: '0.55555555',
      price: '50000',
      timestamp: new Date('2023-09-10'),
    }),
  ];

  return {
    transactions,
    expectedDisposals: [],
    description: 'Fractional cryptocurrency with 8 decimal precision',
  };
}

// ============================================================================
// Export all factories
// ============================================================================

export const factories = {
  transaction: createTransaction,
  buyTransaction: createBuyTransaction,
  sellTransaction: createSellTransaction,
  lot: createLot,
  disposal: createDisposal,
  washSaleDisposal: createWashSaleDisposal,
  position: createPosition,
  complianceRule: createComplianceRule,
  auditEntry: createAuditEntry,

  // Phase 2: Complex scenarios
  complexTaxScenario: createComplexTaxScenario,
  washSaleScenario: createWashSaleScenario,
  irsExampleData: createIRSExampleData,
  performanceTestData: createPerformanceTestData,
  multiAssetPortfolio: createMultiAssetPortfolio,
  longTermVsShortTerm: createLongTermVsShortTermScenario,
  fractionalCrypto: createFractionalCryptoScenario,
};
