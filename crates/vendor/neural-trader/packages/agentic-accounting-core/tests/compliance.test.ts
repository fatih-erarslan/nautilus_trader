/**
 * Compliance Rule Engine Tests
 * Coverage Target: 90%+
 */

import { ComplianceRuleEngine } from '../src/compliance/rules';
import { Transaction } from '@neural-trader/agentic-accounting-types';

describe('ComplianceRuleEngine', () => {
  let engine: ComplianceRuleEngine;

  beforeEach(() => {
    engine = new ComplianceRuleEngine();
  });

  describe('Default Rules', () => {
    it('should initialize with default rules', () => {
      const rules = engine.getRules();
      expect(rules.length).toBeGreaterThan(0);
    });

    it('should have transaction-limit rule', () => {
      const rule = engine.getRule('transaction-limit');
      expect(rule).toBeDefined();
      expect(rule?.name).toBe('Transaction Amount Limit');
    });

    it('should have wash-sale rule', () => {
      const rule = engine.getRule('wash-sale');
      expect(rule).toBeDefined();
      expect(rule?.name).toBe('Wash Sale Rule');
    });

    it('should have suspicious-pattern rule', () => {
      const rule = engine.getRule('suspicious-pattern');
      expect(rule).toBeDefined();
      expect(rule?.name).toBe('Suspicious Activity Pattern');
    });

    it('should have jurisdiction-limit rule', () => {
      const rule = engine.getRule('jurisdiction-limit');
      expect(rule).toBeDefined();
      expect(rule?.name).toBe('Jurisdiction Compliance');
    });
  });

  describe('Transaction Limit Rule', () => {
    it('should pass for transaction under limit', async () => {
      const transaction: Transaction = {
        id: 'txn-001',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000, // $50k < $1M default limit
        source: 'test',
      };

      const violations = await engine.validateTransaction(transaction);
      expect(violations).toHaveLength(0);
    });

    it('should fail for transaction over limit', async () => {
      const transaction: Transaction = {
        id: 'txn-002',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 100,
        price: 50000, // $5M > $1M default limit
        source: 'test',
      };

      const violations = await engine.validateTransaction(transaction);
      expect(violations.length).toBeGreaterThan(0);
      expect(violations[0].ruleId).toBe('transaction-limit');
    });

    it('should respect custom limit in context', async () => {
      const transaction: Transaction = {
        id: 'txn-003',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 10,
        price: 50000, // $500k
        source: 'test',
      };

      const context = { limit: 100000 }; // $100k limit
      const violations = await engine.validateTransaction(transaction, context);

      expect(violations.length).toBeGreaterThan(0);
      expect(violations[0].message).toContain('exceeds limit');
    });
  });

  describe('Wash Sale Rule', () => {
    it('should not trigger for sell without recent buys', async () => {
      const transaction: Transaction = {
        id: 'txn-004',
        timestamp: new Date(),
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const context = { recentBuys: [] };
      const violations = await engine.validateTransaction(transaction, context);

      const washSaleViolation = violations.find(v => v.ruleId === 'wash-sale');
      expect(washSaleViolation).toBeUndefined();
    });

    it('should trigger for sell with recent buy of same asset', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const transaction: Transaction = {
        id: 'txn-005',
        timestamp: now,
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 45000,
        source: 'test',
      };

      const recentBuy: Transaction = {
        id: 'txn-buy',
        timestamp: fifteenDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const context = { recentBuys: [recentBuy] };
      const violations = await engine.validateTransaction(transaction, context);

      const washSaleViolation = violations.find(v => v.ruleId === 'wash-sale');
      expect(washSaleViolation).toBeDefined();
      expect(washSaleViolation?.message).toContain('wash sale');
    });

    it('should not trigger for different asset', async () => {
      const now = new Date();
      const fifteenDaysAgo = new Date(now.getTime() - 15 * 24 * 60 * 60 * 1000);

      const transaction: Transaction = {
        id: 'txn-006',
        timestamp: now,
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 45000,
        source: 'test',
      };

      const recentBuy: Transaction = {
        id: 'txn-buy',
        timestamp: fifteenDaysAgo,
        type: 'BUY',
        asset: 'ETH', // Different asset
        quantity: 10,
        price: 2500,
        source: 'test',
      };

      const context = { recentBuys: [recentBuy] };
      const violations = await engine.validateTransaction(transaction, context);

      const washSaleViolation = violations.find(v => v.ruleId === 'wash-sale');
      expect(washSaleViolation).toBeUndefined();
    });

    it('should not trigger for buy outside 30-day window', async () => {
      const now = new Date();
      const fortyDaysAgo = new Date(now.getTime() - 40 * 24 * 60 * 60 * 1000);

      const transaction: Transaction = {
        id: 'txn-007',
        timestamp: now,
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 45000,
        source: 'test',
      };

      const oldBuy: Transaction = {
        id: 'txn-buy',
        timestamp: fortyDaysAgo,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const context = { recentBuys: [oldBuy] };
      const violations = await engine.validateTransaction(transaction, context);

      const washSaleViolation = violations.find(v => v.ruleId === 'wash-sale');
      expect(washSaleViolation).toBeUndefined();
    });
  });

  describe('Suspicious Pattern Rule', () => {
    it('should trigger for round number sell', async () => {
      const transaction: Transaction = {
        id: 'txn-008',
        timestamp: new Date(),
        type: 'SELL',
        asset: 'BTC',
        quantity: 1000, // Round number
        price: 50000,
        source: 'test',
      };

      const violations = await engine.validateTransaction(transaction);

      const suspiciousViolation = violations.find(v => v.ruleId === 'suspicious-pattern');
      expect(suspiciousViolation).toBeDefined();
      expect(suspiciousViolation?.message).toContain('Round number');
    });

    it('should not trigger for non-round number', async () => {
      const transaction: Transaction = {
        id: 'txn-009',
        timestamp: new Date(),
        type: 'SELL',
        asset: 'BTC',
        quantity: 1.234,
        price: 50000,
        source: 'test',
      };

      const violations = await engine.validateTransaction(transaction);

      const suspiciousViolation = violations.find(v => v.ruleId === 'suspicious-pattern');
      expect(suspiciousViolation).toBeUndefined();
    });

    it('should not trigger for buy transactions', async () => {
      const transaction: Transaction = {
        id: 'txn-010',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1000,
        price: 50000,
        source: 'test',
      };

      const violations = await engine.validateTransaction(transaction);

      const suspiciousViolation = violations.find(v => v.ruleId === 'suspicious-pattern');
      expect(suspiciousViolation).toBeUndefined();
    });
  });

  describe('Jurisdiction Compliance Rule', () => {
    it('should pass for transaction under US reporting threshold', async () => {
      const transaction: Transaction = {
        id: 'txn-011',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 0.1,
        price: 50000, // $5k < $10k threshold
        source: 'test',
      };

      const context = { jurisdiction: 'US', reportingFiled: false };
      const violations = await engine.validateTransaction(transaction, context);

      const jurisdictionViolation = violations.find(v => v.ruleId === 'jurisdiction-limit');
      expect(jurisdictionViolation).toBeUndefined();
    });

    it('should fail for transaction over US threshold without report', async () => {
      const transaction: Transaction = {
        id: 'txn-012',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000, // $50k > $10k threshold
        source: 'test',
      };

      const context = { jurisdiction: 'US', reportingFiled: false };
      const violations = await engine.validateTransaction(transaction, context);

      const jurisdictionViolation = violations.find(v => v.ruleId === 'jurisdiction-limit');
      expect(jurisdictionViolation).toBeDefined();
      expect(jurisdictionViolation?.message).toContain('reporting threshold');
    });

    it('should pass for transaction over threshold with report', async () => {
      const transaction: Transaction = {
        id: 'txn-013',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const context = { jurisdiction: 'US', reportingFiled: true };
      const violations = await engine.validateTransaction(transaction, context);

      const jurisdictionViolation = violations.find(v => v.ruleId === 'jurisdiction-limit');
      expect(jurisdictionViolation).toBeUndefined();
    });
  });

  describe('Rule Management', () => {
    it('should add custom rule', () => {
      const customRule = {
        id: 'custom-001',
        name: 'Custom Rule',
        description: 'Test custom rule',
        severity: 'warning' as const,
        enabled: true,
        validate: async () => null,
      };

      engine.addRule(customRule);
      const rule = engine.getRule('custom-001');

      expect(rule).toBeDefined();
      expect(rule?.name).toBe('Custom Rule');
    });

    it('should remove rule', () => {
      engine.removeRule('transaction-limit');
      const rule = engine.getRule('transaction-limit');
      expect(rule).toBeUndefined();
    });

    it('should enable/disable rule', () => {
      engine.setRuleEnabled('wash-sale', false);
      const rule = engine.getRule('wash-sale');
      expect(rule?.enabled).toBe(false);

      engine.setRuleEnabled('wash-sale', true);
      expect(rule?.enabled).toBe(true);
    });

    it('should skip disabled rules during validation', async () => {
      engine.setRuleEnabled('wash-sale', false);

      const transaction: Transaction = {
        id: 'txn-014',
        timestamp: new Date(),
        type: 'SELL',
        asset: 'BTC',
        quantity: 1,
        price: 45000,
        source: 'test',
      };

      const recentBuy: Transaction = {
        id: 'txn-buy',
        timestamp: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const context = { recentBuys: [recentBuy] };
      const violations = await engine.validateTransaction(transaction, context);

      const washSaleViolation = violations.find(v => v.ruleId === 'wash-sale');
      expect(washSaleViolation).toBeUndefined();
    });
  });

  describe('Batch Validation', () => {
    it('should validate multiple transactions', async () => {
      const transactions: Transaction[] = [
        {
          id: 'batch-001',
          timestamp: new Date(),
          type: 'BUY',
          asset: 'BTC',
          quantity: 1,
          price: 50000,
          source: 'test',
        },
        {
          id: 'batch-002',
          timestamp: new Date(),
          type: 'SELL',
          asset: 'ETH',
          quantity: 10,
          price: 2500,
          source: 'test',
        },
      ];

      const results = await engine.validateBatch(transactions);

      expect(results.size).toBe(2);
      expect(results.get('batch-001')).toBeDefined();
      expect(results.get('batch-002')).toBeDefined();
    });

    it('should handle empty batch', async () => {
      const results = await engine.validateBatch([]);
      expect(results.size).toBe(0);
    });

    it('should validate large batch efficiently', async () => {
      const transactions = Array.from({ length: 100 }, (_, i) => ({
        id: `perf-${i}`,
        timestamp: new Date(),
        type: 'BUY' as const,
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      }));

      const startTime = Date.now();
      const results = await engine.validateBatch(transactions);
      const duration = Date.now() - startTime;

      expect(results.size).toBe(100);
      expect(duration).toBeLessThan(2000); // <2s for 100 transactions
    });
  });

  describe('Performance', () => {
    it('should validate transaction in <500ms', async () => {
      const transaction: Transaction = {
        id: 'perf-001',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const startTime = Date.now();
      await engine.validateTransaction(transaction);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(500);
    });
  });
});
