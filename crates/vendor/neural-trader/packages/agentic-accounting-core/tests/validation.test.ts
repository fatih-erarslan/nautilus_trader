/**
 * Transaction Validation Tests
 * Coverage Target: 95%+
 */

import { ValidationService } from '../src/transactions/validation';
import { Transaction } from '@neural-trader/agentic-accounting-types';

describe('ValidationService', () => {
  let validationService: ValidationService;

  beforeEach(() => {
    validationService = new ValidationService();
  });

  describe('validate', () => {
    it('should validate complete valid transaction', async () => {
      const transaction: Transaction = {
        id: 'txn-001',
        timestamp: new Date('2024-01-15'),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1.5,
        price: 45000,
        fees: 50,
        source: 'coinbase',
        metadata: { note: 'test' },
      };

      const result = await validationService.validate(transaction);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should reject transaction with missing required fields', async () => {
      const invalid = {
        id: 'txn-002',
        // missing timestamp, type, asset, quantity, price
      };

      const result = await validationService.validate(invalid);

      expect(result.isValid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    it('should reject transaction with invalid type', async () => {
      const invalid = {
        id: 'txn-003',
        timestamp: new Date(),
        type: 'INVALID_TYPE',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const result = await validationService.validate(invalid);

      expect(result.isValid).toBe(false);
      expect(result.errors.some(e => e.includes('type'))).toBe(true);
    });

    it('should reject transaction with negative quantity', async () => {
      const invalid = {
        id: 'txn-004',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'ETH',
        quantity: -5,
        price: 2500,
        source: 'test',
      };

      const result = await validationService.validate(invalid);

      expect(result.isValid).toBe(false);
    });

    it('should reject transaction with zero price for buy', async () => {
      const invalid: Transaction = {
        id: 'txn-005',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'SOL',
        quantity: 100,
        price: 0,
        source: 'test',
      };

      const result = await validationService.validate(invalid);

      expect(result.isValid).toBe(false);
      expect(result.errors.some(e => e.includes('Price'))).toBe(true);
    });

    it('should warn about future timestamp', async () => {
      const future = new Date(Date.now() + 86400000); // +1 day
      const transaction: Transaction = {
        id: 'txn-006',
        timestamp: future,
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      };

      const result = await validationService.validate(transaction);

      expect(result.warnings.some(w => w.includes('future'))).toBe(true);
    });

    it('should warn about invalid asset symbol format', async () => {
      const transaction: Transaction = {
        id: 'txn-007',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'invalid-symbol', // lowercase and hyphen
        quantity: 1,
        price: 100,
        source: 'test',
      };

      const result = await validationService.validate(transaction);

      expect(result.warnings.some(w => w.includes('Asset symbol'))).toBe(true);
    });

    it('should warn about unusually high price', async () => {
      const transaction: Transaction = {
        id: 'txn-008',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1,
        price: 1e12, // 1 trillion
        source: 'test',
      };

      const result = await validationService.validate(transaction);

      expect(result.warnings.some(w => w.includes('high price'))).toBe(true);
    });

    it('should warn about unusually high quantity', async () => {
      const transaction: Transaction = {
        id: 'txn-009',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'SHIB',
        quantity: 1e16, // 10 quadrillion
        price: 0.00001,
        source: 'test',
      };

      const result = await validationService.validate(transaction);

      expect(result.warnings.some(w => w.includes('high quantity'))).toBe(true);
    });

    it('should warn when fees exceed transaction value', async () => {
      const transaction: Transaction = {
        id: 'txn-010',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 0.001,
        price: 50000,
        fees: 100, // Fees > quantity * price
        source: 'test',
      };

      const result = await validationService.validate(transaction);

      expect(result.warnings.some(w => w.includes('Fees exceed'))).toBe(true);
    });

    it('should validate all transaction types', async () => {
      const types: Transaction['type'][] = [
        'BUY',
        'SELL',
        'TRADE',
        'CONVERT',
        'INCOME',
        'DIVIDEND',
        'FEE',
        'TRANSFER',
      ];

      for (const type of types) {
        const transaction: Transaction = {
          id: `txn-${type}`,
          timestamp: new Date(),
          type,
          asset: 'BTC',
          quantity: 1,
          price: 50000,
          source: 'test',
        };

        const result = await validationService.validate(transaction);
        expect(result.isValid).toBe(true);
      }
    });

    it('should handle optional fields correctly', async () => {
      const minimal: Transaction = {
        id: 'txn-minimal',
        timestamp: new Date(),
        type: 'SELL',
        asset: 'ETH',
        quantity: 10,
        price: 2500,
        source: 'test',
      };

      const result = await validationService.validate(minimal);
      expect(result.isValid).toBe(true);
    });
  });

  describe('validateBatch', () => {
    it('should validate multiple transactions in parallel', async () => {
      const transactions = [
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
        {
          id: 'batch-003',
          timestamp: new Date(),
          type: 'TRADE',
          asset: 'SOL',
          quantity: 100,
          price: 75,
          source: 'test',
        },
      ];

      const results = await validationService.validateBatch(transactions);

      expect(results.size).toBe(3);
      expect(results.get('batch-001')?.isValid).toBe(true);
      expect(results.get('batch-002')?.isValid).toBe(true);
      expect(results.get('batch-003')?.isValid).toBe(true);
    });

    it('should handle mix of valid and invalid transactions', async () => {
      const transactions = [
        {
          id: 'valid-001',
          timestamp: new Date(),
          type: 'BUY',
          asset: 'BTC',
          quantity: 1,
          price: 50000,
          source: 'test',
        },
        {
          id: 'invalid-001',
          // Missing required fields
        },
      ];

      const results = await validationService.validateBatch(transactions);

      expect(results.size).toBe(2);
      expect(results.get('valid-001')?.isValid).toBe(true);
      expect(results.get('invalid-001')?.isValid).toBe(false);
    });

    it('should handle empty batch', async () => {
      const results = await validationService.validateBatch([]);
      expect(results.size).toBe(0);
    });

    it('should validate large batch efficiently', async () => {
      const transactions = Array.from({ length: 1000 }, (_, i) => ({
        id: `perf-${i}`,
        timestamp: new Date(),
        type: 'BUY' as const,
        asset: 'BTC',
        quantity: 1,
        price: 50000,
        source: 'test',
      }));

      const startTime = Date.now();
      const results = await validationService.validateBatch(transactions);
      const duration = Date.now() - startTime;

      expect(results.size).toBe(1000);
      expect(duration).toBeLessThan(5000); // Should complete in <5s
    });
  });

  describe('Edge Cases', () => {
    it('should handle transaction with all optional fields', async () => {
      const complete: Transaction = {
        id: 'complete-001',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 1.5,
        price: 45000,
        fees: 50,
        exchange: 'coinbase',
        walletAddress: '0x123',
        source: 'api',
        metadata: {
          note: 'Test transaction',
          orderId: 'order-123',
          custom: { field: 'value' },
        },
      };

      const result = await validationService.validate(complete);
      expect(result.isValid).toBe(true);
    });

    it('should handle very small quantities', async () => {
      const transaction: Transaction = {
        id: 'small-qty',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'BTC',
        quantity: 0.00000001, // 1 satoshi
        price: 50000,
        source: 'test',
      };

      const result = await validationService.validate(transaction);
      expect(result.isValid).toBe(true);
    });

    it('should handle very large numbers', async () => {
      const transaction: Transaction = {
        id: 'large-nums',
        timestamp: new Date(),
        type: 'BUY',
        asset: 'SHIB',
        quantity: 1e9, // 1 billion
        price: 0.00001,
        source: 'test',
      };

      const result = await validationService.validate(transaction);
      expect(result.isValid).toBe(true);
    });
  });
});
