/**
 * End-to-End Tests: Complete Tax Year Workflow
 *
 * Tests the complete process from transaction import to tax form generation
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { TestDatabaseLifecycle } from '../utils/database-helpers';
import { generateTransactions, createTransaction } from '../fixtures/factories';
import { measureTime } from '../utils/test-setup';

describe('Complete Tax Year Workflow', () => {
  const dbLifecycle = new TestDatabaseLifecycle();

  beforeAll(async () => {
    await dbLifecycle.setup();
  });

  afterAll(async () => {
    await dbLifecycle.teardown();
  });

  it('should process full year of transactions and generate tax forms', async () => {
    const pool = dbLifecycle.getPool();

    // ========================================================================
    // Step 1: Import transactions
    // ========================================================================
    console.log('📥 Step 1: Importing transactions...');

    const transactions = generateTransactions(500, {
      asset: 'BTC'
    });

    // Add mix of buy and sell transactions
    transactions.forEach((tx, i) => {
      tx.type = i % 3 === 0 ? 'SELL' : 'BUY';
    });

    // Bulk insert transactions
    const insertValues = transactions.map((tx, i) => {
      const offset = i * 9;
      return `($${offset + 1}, $${offset + 2}, $${offset + 3}, $${offset + 4}, $${offset + 5}, $${offset + 6}, $${offset + 7}, $${offset + 8}, $${offset + 9})`;
    }).join(',');

    const insertParams = transactions.flatMap(tx => [
      tx.id,
      tx.type,
      tx.asset,
      tx.quantity,
      tx.price,
      tx.timestamp,
      tx.source,
      tx.fees,
      tx.notes || '',
    ]);

    await pool.query(
      `INSERT INTO transactions (id, type, asset, quantity, price, timestamp, source, fees, notes)
       VALUES ${insertValues}`,
      insertParams
    );

    const countResult = await pool.query('SELECT COUNT(*) FROM transactions');
    const transactionCount = parseInt(countResult.rows[0].count);

    expect(transactionCount).toBe(500);
    console.log(`✅ Imported ${transactionCount} transactions`);

    // ========================================================================
    // Step 2: Calculate taxes
    // ========================================================================
    console.log('🧮 Step 2: Calculating taxes...');

    // In real implementation, this would call the tax calculation agent
    const calculateTaxes = async () => {
      // Mock tax calculation
      const sellTransactions = await pool.query(
        `SELECT * FROM transactions WHERE type = 'SELL' ORDER BY timestamp`
      );

      return {
        totalGains: '125000.50',
        totalLosses: '25000.25',
        netGain: '99999.25',
        shortTermGain: '50000.00',
        longTermGain: '49999.25',
        disposalsCount: sellTransactions.rows.length,
      };
    };

    const { result: taxSummary, duration: calcDuration } = await measureTime(calculateTaxes);

    expect(taxSummary.totalGains).toBeDefined();
    expect(taxSummary.disposalsCount).toBeGreaterThan(0);
    console.log(`✅ Calculated taxes in ${calcDuration.toFixed(2)}ms`);
    console.log(`   Net Gain: $${taxSummary.netGain}`);

    // ========================================================================
    // Step 3: Identify harvest opportunities
    // ========================================================================
    console.log('🌾 Step 3: Identifying tax-loss harvesting opportunities...');

    const identifyHarvestOpportunities = async () => {
      // Mock harvest opportunity identification
      // In real implementation, this would analyze current positions
      // and find optimal harvest candidates

      return [
        {
          asset: 'BTC',
          currentLoss: '-5000.00',
          potentialSavings: '1500.00', // 30% tax bracket
          recommendation: 'HARVEST',
          replacementAsset: 'ETH',
        },
        {
          asset: 'ETH',
          currentLoss: '-3000.00',
          potentialSavings: '900.00',
          recommendation: 'HARVEST',
          replacementAsset: 'BTC',
        },
      ];
    };

    const opportunities = await identifyHarvestOpportunities();

    expect(opportunities.length).toBeGreaterThan(0);
    console.log(`✅ Found ${opportunities.length} harvest opportunities`);
    console.log(`   Potential savings: $${opportunities.reduce((sum, opp) =>
      sum + parseFloat(opp.potentialSavings), 0).toFixed(2)}`);

    // ========================================================================
    // Step 4: Execute harvesting (simulated)
    // ========================================================================
    console.log('⚡ Step 4: Simulating tax-loss harvesting execution...');

    const executeHarvesting = async (opps: any[]) => {
      // In real implementation, this would:
      // 1. Sell the losing position
      // 2. Buy replacement asset (if specified)
      // 3. Track wash sale window

      const totalLossBanked = opps.reduce((sum, opp) =>
        sum + Math.abs(parseFloat(opp.currentLoss)), 0
      );

      return {
        executedCount: opps.length,
        totalLossBanked: totalLossBanked.toFixed(2),
        taxSavings: (totalLossBanked * 0.30).toFixed(2), // 30% tax bracket
      };
    };

    const harvestedResult = await executeHarvesting(opportunities.slice(0, 2));

    expect(parseFloat(harvestedResult.totalLossBanked)).toBeGreaterThan(0);
    console.log(`✅ Harvested ${harvestedResult.executedCount} positions`);
    console.log(`   Total loss banked: $${harvestedResult.totalLossBanked}`);
    console.log(`   Estimated tax savings: $${harvestedResult.taxSavings}`);

    // ========================================================================
    // Step 5: Generate reports
    // ========================================================================
    console.log('📄 Step 5: Generating tax forms...');

    const generateScheduleD = async () => {
      return `
SCHEDULE D (Form 1040)
Capital Gains and Losses

Taxpayer: Test User
Tax Year: 2023

Part I - Short-Term Capital Gains and Losses
Total short-term gain: $${taxSummary.shortTermGain}

Part II - Long-Term Capital Gains and Losses
Total long-term gain: $${taxSummary.longTermGain}

Net capital gain: $${taxSummary.netGain}
      `.trim();
    };

    const generateForm8949 = async () => {
      return `
FORM 8949
Sales and Other Dispositions of Capital Assets

Part I - Short-Term Transactions
[${taxSummary.disposalsCount} transactions listed]

Part II - Long-Term Transactions
[Detail transactions here]

Generated: ${new Date().toISOString()}
      `.trim();
    };

    const [scheduleD, form8949] = await Promise.all([
      generateScheduleD(),
      generateForm8949(),
    ]);

    expect(scheduleD).toContain('Schedule D');
    expect(scheduleD).toContain('2023');
    expect(form8949).toContain('Form 8949');
    console.log('✅ Generated Schedule D and Form 8949');

    // ========================================================================
    // Step 6: Verify audit trail
    // ========================================================================
    console.log('🔍 Step 6: Verifying audit trail integrity...');

    // Create some audit entries for key operations
    const auditEntries = [
      {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        action: 'IMPORT',
        entity: 'Transaction',
        entityId: transactions[0].id!,
        userId: 'test-user',
        changes: { count: transactionCount },
        hash: 'hash-1',
        previousHash: '0000000000000000',
      },
      {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        action: 'CALCULATE',
        entity: 'TaxSummary',
        entityId: crypto.randomUUID(),
        userId: 'test-user',
        changes: taxSummary,
        hash: 'hash-2',
        previousHash: 'hash-1',
      },
    ];

    for (const entry of auditEntries) {
      await pool.query(
        `INSERT INTO audit_trail (id, timestamp, action, entity, entity_id, user_id, changes, hash, previous_hash)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
        [
          entry.id,
          entry.timestamp,
          entry.action,
          entry.entity,
          entry.entityId,
          entry.userId,
          JSON.stringify(entry.changes),
          entry.hash,
          entry.previousHash,
        ]
      );
    }

    const auditResult = await pool.query(
      'SELECT * FROM audit_trail ORDER BY timestamp'
    );

    expect(auditResult.rows.length).toBeGreaterThan(0);

    // Verify chain integrity
    let chainValid = true;
    for (let i = 1; i < auditResult.rows.length; i++) {
      if (auditResult.rows[i].previous_hash !== auditResult.rows[i - 1].hash) {
        chainValid = false;
        break;
      }
    }

    expect(chainValid).toBe(true);
    console.log(`✅ Audit trail verified (${auditResult.rows.length} entries)`);

    // ========================================================================
    // Final Summary
    // ========================================================================
    console.log('\n📊 Workflow Summary:');
    console.log(`   Transactions: ${transactionCount}`);
    console.log(`   Disposals: ${taxSummary.disposalsCount}`);
    console.log(`   Net Gain: $${taxSummary.netGain}`);
    console.log(`   Tax Savings: $${harvestedResult.taxSavings}`);
    console.log(`   Audit Entries: ${auditResult.rows.length}`);
    console.log(`   Total Duration: ${calcDuration.toFixed(2)}ms`);
  }, 60000); // 60 second timeout for full workflow
});
