/**
 * Database Operations Performance Benchmarks
 *
 * Tests transaction insertion, tax lot queries, position tracking, and compliance evaluation.
 * Target: <20ms per query
 */

import { performance } from 'perf_hooks';

// Mock Database Types
interface Transaction {
  id: string;
  type: 'buy' | 'sell';
  asset: string;
  quantity: number;
  price: number;
  timestamp: Date;
  userId: string;
}

interface TaxLot {
  id: string;
  transactionId: string;
  asset: string;
  quantity: number;
  remainingQuantity: number;
  costBasis: number;
  acquisitionDate: Date;
  userId: string;
}

interface Position {
  userId: string;
  asset: string;
  totalQuantity: number;
  averageCost: number;
  realizedGainLoss: number;
  unrealizedGainLoss: number;
}

// Mock Database
class MockDatabase {
  private transactions: Map<string, Transaction> = new Map();
  private taxLots: Map<string, TaxLot> = new Map();
  private positions: Map<string, Position> = new Map();
  private indexes: {
    transactionsByUser: Map<string, string[]>;
    taxLotsByAsset: Map<string, string[]>;
    positionsByUser: Map<string, string[]>;
  };

  constructor() {
    this.indexes = {
      transactionsByUser: new Map(),
      taxLotsByAsset: new Map(),
      positionsByUser: new Map(),
    };
  }

  // Transaction operations
  insertTransaction(tx: Transaction): void {
    this.transactions.set(tx.id, tx);

    // Update index
    if (!this.indexes.transactionsByUser.has(tx.userId)) {
      this.indexes.transactionsByUser.set(tx.userId, []);
    }
    this.indexes.transactionsByUser.get(tx.userId)!.push(tx.id);
  }

  batchInsertTransactions(txs: Transaction[]): void {
    for (const tx of txs) {
      this.insertTransaction(tx);
    }
  }

  queryTransactionsByUser(userId: string): Transaction[] {
    const txIds = this.indexes.transactionsByUser.get(userId) || [];
    return txIds.map(id => this.transactions.get(id)!).filter(Boolean);
  }

  queryTransactionsByAsset(asset: string): Transaction[] {
    return Array.from(this.transactions.values()).filter(tx => tx.asset === asset);
  }

  // Tax lot operations
  insertTaxLot(lot: TaxLot): void {
    this.taxLots.set(lot.id, lot);

    // Update index
    if (!this.indexes.taxLotsByAsset.has(lot.asset)) {
      this.indexes.taxLotsByAsset.set(lot.asset, []);
    }
    this.indexes.taxLotsByAsset.get(lot.asset)!.push(lot.id);
  }

  queryTaxLotsByAsset(asset: string): TaxLot[] {
    const lotIds = this.indexes.taxLotsByAsset.get(asset) || [];
    return lotIds.map(id => this.taxLots.get(id)!).filter(Boolean);
  }

  queryAvailableTaxLots(userId: string, asset: string): TaxLot[] {
    return Array.from(this.taxLots.values())
      .filter(lot => lot.userId === userId && lot.asset === asset && lot.remainingQuantity > 0)
      .sort((a, b) => a.acquisitionDate.getTime() - b.acquisitionDate.getTime());
  }

  // Position operations
  upsertPosition(position: Position): void {
    const key = `${position.userId}_${position.asset}`;
    this.positions.set(key, position);

    // Update index
    if (!this.indexes.positionsByUser.has(position.userId)) {
      this.indexes.positionsByUser.set(position.userId, []);
    }
    if (!this.indexes.positionsByUser.get(position.userId)!.includes(key)) {
      this.indexes.positionsByUser.get(position.userId)!.push(key);
    }
  }

  queryPositionsByUser(userId: string): Position[] {
    const positionKeys = this.indexes.positionsByUser.get(userId) || [];
    return positionKeys.map(key => this.positions.get(key)!).filter(Boolean);
  }

  // Complex queries
  queryYearEndReport(userId: string, year: number): any {
    const transactions = this.queryTransactionsByUser(userId);
    const yearStart = new Date(year, 0, 1);
    const yearEnd = new Date(year + 1, 0, 1);

    return transactions.filter(tx =>
      tx.timestamp >= yearStart && tx.timestamp < yearEnd
    );
  }

  queryComplianceCheck(userId: string): boolean {
    // Simulate complex compliance query
    const positions = this.queryPositionsByUser(userId);
    const transactions = this.queryTransactionsByUser(userId);

    // Dummy compliance check
    return transactions.length > 0 && positions.length > 0;
  }

  clear(): void {
    this.transactions.clear();
    this.taxLots.clear();
    this.positions.clear();
    this.indexes.transactionsByUser.clear();
    this.indexes.taxLotsByAsset.clear();
    this.indexes.positionsByUser.clear();
  }

  stats(): any {
    return {
      transactions: this.transactions.size,
      taxLots: this.taxLots.size,
      positions: this.positions.size,
    };
  }
}

// Helper functions
function generateTransaction(id: number, userId: string): Transaction {
  return {
    id: `tx_${id}`,
    type: id % 2 === 0 ? 'buy' : 'sell',
    asset: ['BTC', 'ETH', 'SOL'][id % 3],
    quantity: Math.random() * 10,
    price: Math.random() * 50000,
    timestamp: new Date(Date.now() - id * 3600000),
    userId,
  };
}

function generateTaxLot(id: number, transactionId: string, userId: string): TaxLot {
  const qty = Math.random() * 10;
  return {
    id: `lot_${id}`,
    transactionId,
    asset: ['BTC', 'ETH', 'SOL'][id % 3],
    quantity: qty,
    remainingQuantity: qty * Math.random(),
    costBasis: Math.random() * 50000,
    acquisitionDate: new Date(Date.now() - id * 3600000),
    userId,
  };
}

function generatePosition(userId: string, asset: string): Position {
  return {
    userId,
    asset,
    totalQuantity: Math.random() * 100,
    averageCost: Math.random() * 50000,
    realizedGainLoss: (Math.random() - 0.5) * 10000,
    unrealizedGainLoss: (Math.random() - 0.5) * 10000,
  };
}

// Benchmark Functions
function benchmarkTransactionInsertion(): void {
  console.log('📊 Benchmark 1: Transaction Insertion');

  for (const batchSize of [100, 1000, 10000]) {
    const db = new MockDatabase();
    const transactions = Array.from({ length: batchSize }, (_, i) =>
      generateTransaction(i, `user_${i % 100}`)
    );

    const start = performance.now();
    db.batchInsertTransactions(transactions);
    const elapsed = performance.now() - start;

    console.log(`  Batch insert ${batchSize} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Average: ${(elapsed / batchSize).toFixed(3)}ms per transaction`);
    console.log(`    Throughput: ${(batchSize / (elapsed / 1000)).toFixed(0)} tx/second\n`);
  }
}

function benchmarkTaxLotQueries(): void {
  console.log('📊 Benchmark 2: Tax Lot Queries');

  const db = new MockDatabase();

  // Populate database
  console.log('  Populating database with 10,000 tax lots...');
  for (let i = 0; i < 10000; i++) {
    db.insertTaxLot(generateTaxLot(i, `tx_${i}`, `user_${i % 100}`));
  }

  // Query by asset
  const iterations = 1000;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    db.queryTaxLotsByAsset('BTC');
  }
  const elapsed = performance.now() - start;

  console.log(`  ${iterations} queries by asset: ${elapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(elapsed / iterations).toFixed(2)}ms per query`);

  const targetMs = 20;
  const avgTime = elapsed / iterations;
  const status = avgTime < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${status} Target: <${targetMs}ms per query\n`);

  // Query available lots
  const availStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    db.queryAvailableTaxLots(`user_${i % 100}`, 'BTC');
  }
  const availElapsed = performance.now() - availStart;

  console.log(`  ${iterations} queries for available lots: ${availElapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(availElapsed / iterations).toFixed(2)}ms per query`);

  const avgAvailTime = availElapsed / iterations;
  const statusAvail = avgAvailTime < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${statusAvail} Target: <${targetMs}ms per query\n`);
}

function benchmarkPositionTracking(): void {
  console.log('📊 Benchmark 3: Position Tracking Queries');

  const db = new MockDatabase();

  // Populate database
  console.log('  Populating database with positions...');
  for (let i = 0; i < 100; i++) {
    const userId = `user_${i}`;
    for (const asset of ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC']) {
      db.upsertPosition(generatePosition(userId, asset));
    }
  }

  const iterations = 1000;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    db.queryPositionsByUser(`user_${i % 100}`);
  }
  const elapsed = performance.now() - start;

  console.log(`  ${iterations} position queries: ${elapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(elapsed / iterations).toFixed(2)}ms per query`);

  const targetMs = 20;
  const avgTime = elapsed / iterations;
  const status = avgTime < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${status} Target: <${targetMs}ms per query\n`);
}

function benchmarkComplexQueries(): void {
  console.log('📊 Benchmark 4: Complex Compliance Queries');

  const db = new MockDatabase();

  // Populate database
  console.log('  Populating database...');
  for (let i = 0; i < 5000; i++) {
    const userId = `user_${i % 100}`;
    db.insertTransaction(generateTransaction(i, userId));
    db.insertTaxLot(generateTaxLot(i, `tx_${i}`, userId));
  }

  for (let i = 0; i < 100; i++) {
    const userId = `user_${i}`;
    for (const asset of ['BTC', 'ETH', 'SOL']) {
      db.upsertPosition(generatePosition(userId, asset));
    }
  }

  // Year-end report queries
  const iterations = 100;
  const reportStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    db.queryYearEndReport(`user_${i % 100}`, 2024);
  }
  const reportElapsed = performance.now() - reportStart;

  console.log(`  ${iterations} year-end reports: ${reportElapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(reportElapsed / iterations).toFixed(2)}ms per report`);

  const targetMs = 20;
  const avgReportTime = reportElapsed / iterations;
  const statusReport = avgReportTime < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${statusReport} Target: <${targetMs}ms per query\n`);

  // Compliance checks
  const complianceStart = performance.now();
  for (let i = 0; i < iterations; i++) {
    db.queryComplianceCheck(`user_${i % 100}`);
  }
  const complianceElapsed = performance.now() - complianceStart;

  console.log(`  ${iterations} compliance checks: ${complianceElapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(complianceElapsed / iterations).toFixed(2)}ms per check`);

  const avgComplianceTime = complianceElapsed / iterations;
  const statusCompliance = avgComplianceTime < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${statusCompliance} Target: <${targetMs}ms per query\n`);
}

function benchmarkIndexPerformance(): void {
  console.log('📊 Benchmark 5: Index Performance');

  const dbWithIndex = new MockDatabase();

  // Populate and measure with indexes
  console.log('  Testing indexed queries...');
  for (let i = 0; i < 10000; i++) {
    dbWithIndex.insertTransaction(generateTransaction(i, `user_${i % 100}`));
  }

  const iterations = 1000;
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    dbWithIndex.queryTransactionsByUser(`user_${i % 100}`);
  }
  const elapsed = performance.now() - start;

  console.log(`  ${iterations} indexed queries: ${elapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(elapsed / iterations).toFixed(3)}ms per query`);
  console.log(`    Throughput: ${(iterations / (elapsed / 1000)).toFixed(0)} queries/second\n`);
}

function benchmarkConcurrentOperations(): void {
  console.log('📊 Benchmark 6: Mixed Operation Throughput');

  const db = new MockDatabase();
  const operations = 10000;

  const start = performance.now();
  for (let i = 0; i < operations; i++) {
    const op = i % 4;
    const userId = `user_${i % 100}`;

    switch (op) {
      case 0:
        db.insertTransaction(generateTransaction(i, userId));
        break;
      case 1:
        db.insertTaxLot(generateTaxLot(i, `tx_${i}`, userId));
        break;
      case 2:
        db.queryTransactionsByUser(userId);
        break;
      case 3:
        db.queryPositionsByUser(userId);
        break;
    }
  }
  const elapsed = performance.now() - start;

  console.log(`  ${operations} mixed operations: ${elapsed.toFixed(2)}ms`);
  console.log(`    Average: ${(elapsed / operations).toFixed(3)}ms per operation`);
  console.log(`    Throughput: ${(operations / (elapsed / 1000)).toFixed(0)} ops/second`);

  const stats = db.stats();
  console.log(`  Final stats:`, stats);
  console.log();
}

// Main execution
async function runBenchmarks(): Promise<void> {
  console.log('\n🚀 Starting Database Operations Performance Benchmarks\n');

  benchmarkTransactionInsertion();
  benchmarkTaxLotQueries();
  benchmarkPositionTracking();
  benchmarkComplexQueries();
  benchmarkIndexPerformance();
  benchmarkConcurrentOperations();

  console.log('✅ All database benchmarks completed!\n');
}

// Run benchmarks
runBenchmarks().catch(console.error);
