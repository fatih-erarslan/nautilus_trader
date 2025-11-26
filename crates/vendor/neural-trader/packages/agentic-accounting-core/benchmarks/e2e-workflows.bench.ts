/**
 * End-to-End Workflows Performance Benchmarks
 *
 * Tests complete workflows: import → calculate taxes, compliance checks, Schedule D generation,
 * fraud detection, and tax-loss harvesting.
 * Target: <500ms for complete workflows
 */

import { performance } from 'perf_hooks';

// Mock comprehensive workflow components
interface Transaction {
  id: string;
  type: 'buy' | 'sell' | 'transfer';
  asset: string;
  quantity: number;
  price: number;
  timestamp: Date;
  source: string;
  fees: number;
}

interface TaxCalculationResult {
  shortTermGains: number;
  longTermGains: number;
  totalGains: number;
  disposals: any[];
  washSales: any[];
}

interface ComplianceResult {
  passed: boolean;
  violations: string[];
  warnings: string[];
}

interface ScheduleD {
  shortTermTransactions: any[];
  longTermTransactions: any[];
  totalShortTerm: number;
  totalLongTerm: number;
  netGainLoss: number;
}

interface FraudDetectionResult {
  suspicious: boolean;
  score: number;
  patterns: string[];
}

interface TaxLossHarvestingResult {
  opportunities: any[];
  potentialSavings: number;
}

// Mock workflow implementations
class TaxCalculationWorkflow {
  async importAndCalculate(transactions: Transaction[]): Promise<TaxCalculationResult> {
    // Simulate import + normalization + validation
    await this.simulateWork(50);

    // Simulate tax lot creation
    await this.simulateWork(30);

    // Simulate FIFO calculation
    await this.simulateWork(100);

    // Simulate wash sale detection
    await this.simulateWork(80);

    return {
      shortTermGains: Math.random() * 10000,
      longTermGains: Math.random() * 20000,
      totalGains: Math.random() * 30000,
      disposals: Array.from({ length: transactions.length / 2 }, (_, i) => ({ id: `disp_${i}` })),
      washSales: Array.from({ length: Math.floor(transactions.length * 0.1) }, (_, i) => ({ id: `wash_${i}` })),
    };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class ComplianceWorkflow {
  async checkCompliance(transactions: Transaction[]): Promise<ComplianceResult> {
    // Simulate rule evaluation
    await this.simulateWork(100);

    // Simulate pattern matching
    await this.simulateWork(50);

    const violations: string[] = [];
    const warnings: string[] = [];

    // Random compliance issues
    if (Math.random() > 0.8) {
      violations.push('Missing cost basis for some transactions');
    }
    if (Math.random() > 0.5) {
      warnings.push('Unusual trading pattern detected');
    }

    return {
      passed: violations.length === 0,
      violations,
      warnings,
    };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class ReportingWorkflow {
  async generateScheduleD(taxResult: TaxCalculationResult): Promise<ScheduleD> {
    // Simulate report generation
    await this.simulateWork(80);

    // Simulate PDF rendering (if applicable)
    await this.simulateWork(50);

    return {
      shortTermTransactions: taxResult.disposals.slice(0, Math.floor(taxResult.disposals.length / 2)),
      longTermTransactions: taxResult.disposals.slice(Math.floor(taxResult.disposals.length / 2)),
      totalShortTerm: taxResult.shortTermGains,
      totalLongTerm: taxResult.longTermGains,
      netGainLoss: taxResult.totalGains,
    };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class FraudDetectionWorkflow {
  async detectFraudPatterns(transactions: Transaction[]): Promise<FraudDetectionResult> {
    // Simulate vector embedding generation
    await this.simulateWork(100);

    // Simulate HNSW similarity search
    await this.simulateWork(50);

    // Simulate pattern analysis
    await this.simulateWork(80);

    const score = Math.random();
    const patterns: string[] = [];

    if (score > 0.7) {
      patterns.push('Circular trading detected');
    }
    if (score > 0.8) {
      patterns.push('Wash trading pattern');
    }
    if (score > 0.9) {
      patterns.push('Suspicious timing');
    }

    return {
      suspicious: score > 0.7,
      score,
      patterns,
    };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class TaxLossHarvestingWorkflow {
  async scanOpportunities(transactions: Transaction[]): Promise<TaxLossHarvestingResult> {
    // Simulate portfolio analysis
    await this.simulateWork(80);

    // Simulate loss identification
    await this.simulateWork(60);

    // Simulate wash sale rule checking
    await this.simulateWork(70);

    const opportunityCount = Math.floor(transactions.length * 0.15);
    const opportunities = Array.from({ length: opportunityCount }, (_, i) => ({
      id: `opportunity_${i}`,
      asset: ['BTC', 'ETH', 'SOL'][i % 3],
      potentialLoss: Math.random() * 5000,
      taxSavings: Math.random() * 1500,
    }));

    const potentialSavings = opportunities.reduce((sum, opp) => sum + opp.taxSavings, 0);

    return {
      opportunities,
      potentialSavings,
    };
  }

  private async simulateWork(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Helper functions
function generateTransactions(count: number): Transaction[] {
  const assets = ['BTC', 'ETH', 'SOL', 'ADA', 'MATIC'];
  return Array.from({ length: count }, (_, i) => ({
    id: `tx_${i}`,
    type: i % 3 === 0 ? 'sell' : 'buy' as 'buy' | 'sell',
    asset: assets[i % assets.length],
    quantity: Math.random() * 10,
    price: Math.random() * 50000,
    timestamp: new Date(Date.now() - i * 3600000),
    source: 'exchange',
    fees: Math.random() * 100,
  }));
}

// Benchmark Functions
async function benchmarkFullTaxCalculationWorkflow(): Promise<void> {
  console.log('📊 Benchmark 1: Full Tax Calculation Workflow');
  console.log('  (Import → Calculate Taxes → Wash Sales)');

  const workflow = new TaxCalculationWorkflow();

  for (const txCount of [100, 500, 1000]) {
    const transactions = generateTransactions(txCount);

    const start = performance.now();
    const result = await workflow.importAndCalculate(transactions);
    const elapsed = performance.now() - start;

    console.log(`\n  ${txCount} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Short-term gains: $${result.shortTermGains.toFixed(2)}`);
    console.log(`    Long-term gains: $${result.longTermGains.toFixed(2)}`);
    console.log(`    Disposals: ${result.disposals.length}`);
    console.log(`    Wash sales: ${result.washSales.length}`);

    const targetMs = 500;
    const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMs}ms`);
  }
  console.log();
}

async function benchmarkComplianceCheckWorkflow(): Promise<void> {
  console.log('📊 Benchmark 2: Compliance Check Workflow');

  const workflow = new ComplianceWorkflow();

  for (const txCount of [1000, 5000, 10000]) {
    const transactions = generateTransactions(txCount);

    const start = performance.now();
    const result = await workflow.checkCompliance(transactions);
    const elapsed = performance.now() - start;

    console.log(`\n  ${txCount} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Passed: ${result.passed}`);
    console.log(`    Violations: ${result.violations.length}`);
    console.log(`    Warnings: ${result.warnings.length}`);

    const targetMs = 500;
    const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMs}ms`);
  }
  console.log();
}

async function benchmarkScheduleDGeneration(): Promise<void> {
  console.log('📊 Benchmark 3: Schedule D Generation Workflow');

  const taxWorkflow = new TaxCalculationWorkflow();
  const reportWorkflow = new ReportingWorkflow();

  for (const txCount of [100, 500, 1000]) {
    const transactions = generateTransactions(txCount);

    const start = performance.now();
    const taxResult = await taxWorkflow.importAndCalculate(transactions);
    const scheduleD = await reportWorkflow.generateScheduleD(taxResult);
    const elapsed = performance.now() - start;

    console.log(`\n  ${txCount} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Short-term txs: ${scheduleD.shortTermTransactions.length}`);
    console.log(`    Long-term txs: ${scheduleD.longTermTransactions.length}`);
    console.log(`    Net gain/loss: $${scheduleD.netGainLoss.toFixed(2)}`);

    const targetMs = 500;
    const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMs}ms`);
  }
  console.log();
}

async function benchmarkFraudDetectionWorkflow(): Promise<void> {
  console.log('📊 Benchmark 4: Fraud Detection Workflow');

  const workflow = new FraudDetectionWorkflow();

  for (const txCount of [1000, 5000, 10000]) {
    const transactions = generateTransactions(txCount);

    const start = performance.now();
    const result = await workflow.detectFraudPatterns(transactions);
    const elapsed = performance.now() - start;

    console.log(`\n  ${txCount} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Suspicious: ${result.suspicious}`);
    console.log(`    Risk score: ${result.score.toFixed(3)}`);
    console.log(`    Patterns found: ${result.patterns.length}`);

    const targetMs = 500;
    const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMs}ms`);
  }
  console.log();
}

async function benchmarkTaxLossHarvestingWorkflow(): Promise<void> {
  console.log('📊 Benchmark 5: Tax-Loss Harvesting Workflow');

  const workflow = new TaxLossHarvestingWorkflow();

  for (const txCount of [500, 1000, 2000]) {
    const transactions = generateTransactions(txCount);

    const start = performance.now();
    const result = await workflow.scanOpportunities(transactions);
    const elapsed = performance.now() - start;

    console.log(`\n  ${txCount} transactions: ${elapsed.toFixed(2)}ms`);
    console.log(`    Opportunities: ${result.opportunities.length}`);
    console.log(`    Potential savings: $${result.potentialSavings.toFixed(2)}`);

    const targetMs = 500;
    const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
    console.log(`    ${status} Target: <${targetMs}ms`);
  }
  console.log();
}

async function benchmarkCompleteYearEndWorkflow(): Promise<void> {
  console.log('📊 Benchmark 6: Complete Year-End Workflow');
  console.log('  (Import → Calculate → Compliance → Schedule D → Fraud Check → Harvest)');

  const transactions = generateTransactions(1000);
  const taxWorkflow = new TaxCalculationWorkflow();
  const complianceWorkflow = new ComplianceWorkflow();
  const reportWorkflow = new ReportingWorkflow();
  const fraudWorkflow = new FraudDetectionWorkflow();
  const harvestWorkflow = new TaxLossHarvestingWorkflow();

  const start = performance.now();

  // Step 1: Import and calculate taxes
  const stepStart1 = performance.now();
  const taxResult = await taxWorkflow.importAndCalculate(transactions);
  const step1Time = performance.now() - stepStart1;

  // Step 2: Compliance check
  const stepStart2 = performance.now();
  const complianceResult = await complianceWorkflow.checkCompliance(transactions);
  const step2Time = performance.now() - stepStart2;

  // Step 3: Generate Schedule D
  const stepStart3 = performance.now();
  const scheduleD = await reportWorkflow.generateScheduleD(taxResult);
  const step3Time = performance.now() - stepStart3;

  // Step 4: Fraud detection
  const stepStart4 = performance.now();
  const fraudResult = await fraudWorkflow.detectFraudPatterns(transactions);
  const step4Time = performance.now() - stepStart4;

  // Step 5: Tax-loss harvesting
  const stepStart5 = performance.now();
  const harvestResult = await harvestWorkflow.scanOpportunities(transactions);
  const step5Time = performance.now() - stepStart5;

  const totalElapsed = performance.now() - start;

  console.log(`\n  1000 transactions - Complete workflow`);
  console.log(`    Step 1 (Import + Calculate): ${step1Time.toFixed(2)}ms`);
  console.log(`    Step 2 (Compliance): ${step2Time.toFixed(2)}ms`);
  console.log(`    Step 3 (Schedule D): ${step3Time.toFixed(2)}ms`);
  console.log(`    Step 4 (Fraud Detection): ${step4Time.toFixed(2)}ms`);
  console.log(`    Step 5 (Harvesting): ${step5Time.toFixed(2)}ms`);
  console.log(`    ─────────────────────────────────`);
  console.log(`    Total: ${totalElapsed.toFixed(2)}ms`);

  console.log(`\n  Results Summary:`);
  console.log(`    Tax gain/loss: $${taxResult.totalGains.toFixed(2)}`);
  console.log(`    Compliance: ${complianceResult.passed ? 'PASSED' : 'FAILED'}`);
  console.log(`    Schedule D entries: ${scheduleD.shortTermTransactions.length + scheduleD.longTermTransactions.length}`);
  console.log(`    Fraud risk: ${fraudResult.suspicious ? 'HIGH' : 'LOW'}`);
  console.log(`    Harvest opportunities: ${harvestResult.opportunities.length}`);

  const targetMs = 1000; // More lenient for complete workflow
  const status = totalElapsed < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`\n    ${status} Target: <${targetMs}ms for complete workflow`);
  console.log();
}

async function benchmarkParallelWorkflows(): Promise<void> {
  console.log('📊 Benchmark 7: Parallel Workflow Execution');
  console.log('  (Running all workflows in parallel)');

  const transactions = generateTransactions(1000);
  const taxWorkflow = new TaxCalculationWorkflow();
  const complianceWorkflow = new ComplianceWorkflow();
  const fraudWorkflow = new FraudDetectionWorkflow();
  const harvestWorkflow = new TaxLossHarvestingWorkflow();

  const start = performance.now();

  // Execute all workflows in parallel
  const [taxResult, complianceResult, fraudResult, harvestResult] = await Promise.all([
    taxWorkflow.importAndCalculate(transactions),
    complianceWorkflow.checkCompliance(transactions),
    fraudWorkflow.detectFraudPatterns(transactions),
    harvestWorkflow.scanOpportunities(transactions),
  ]);

  const elapsed = performance.now() - start;

  console.log(`\n  1000 transactions - Parallel execution: ${elapsed.toFixed(2)}ms`);
  console.log(`    Tax calculation: Done`);
  console.log(`    Compliance check: ${complianceResult.passed ? 'PASSED' : 'FAILED'}`);
  console.log(`    Fraud detection: ${fraudResult.suspicious ? 'FLAGGED' : 'CLEAR'}`);
  console.log(`    Harvest opportunities: ${harvestResult.opportunities.length}`);

  const targetMs = 500; // Should be faster than sequential
  const status = elapsed < targetMs ? '✅ PASS' : '❌ FAIL';
  console.log(`    ${status} Target: <${targetMs}ms (parallel speedup)`);
  console.log();
}

async function benchmarkThroughput(): Promise<void> {
  console.log('📊 Benchmark 8: Workflow Throughput');

  const workflow = new TaxCalculationWorkflow();
  const batchCount = 10;
  const txPerBatch = 100;

  const start = performance.now();
  for (let i = 0; i < batchCount; i++) {
    const transactions = generateTransactions(txPerBatch);
    await workflow.importAndCalculate(transactions);
  }
  const elapsed = performance.now() - start;

  const totalTransactions = batchCount * txPerBatch;
  const throughput = totalTransactions / (elapsed / 1000);

  console.log(`\n  ${batchCount} batches × ${txPerBatch} transactions = ${totalTransactions} total`);
  console.log(`    Total time: ${elapsed.toFixed(2)}ms`);
  console.log(`    Throughput: ${throughput.toFixed(0)} transactions/second`);
  console.log(`    Average per batch: ${(elapsed / batchCount).toFixed(2)}ms`);
  console.log();
}

// Main execution
async function runBenchmarks(): Promise<void> {
  console.log('\n🚀 Starting End-to-End Workflow Performance Benchmarks\n');

  await benchmarkFullTaxCalculationWorkflow();
  await benchmarkComplianceCheckWorkflow();
  await benchmarkScheduleDGeneration();
  await benchmarkFraudDetectionWorkflow();
  await benchmarkTaxLossHarvestingWorkflow();
  await benchmarkCompleteYearEndWorkflow();
  await benchmarkParallelWorkflows();
  await benchmarkThroughput();

  console.log('✅ All end-to-end workflow benchmarks completed!\n');
}

// Run benchmarks
runBenchmarks().catch(console.error);
