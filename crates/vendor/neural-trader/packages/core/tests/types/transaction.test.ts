import { describe, it, expect } from 'vitest';
import { TransactionType, IncomeType, type Transaction, type Income } from '../../src/types/transaction.js';
import { Decimal } from '../../src/utils/decimal.js';
import { v4 as uuidv4 } from 'uuid';

describe('Transaction Types', () => {
  describe('TransactionType enum', () => {
    it('should have all required transaction types', () => {
      expect(TransactionType.BUY).toBe('BUY');
      expect(TransactionType.SELL).toBe('SELL');
      expect(TransactionType.TRADE).toBe('TRADE');
      expect(TransactionType.INCOME).toBe('INCOME');
      expect(TransactionType.EXPENSE).toBe('EXPENSE');
      expect(TransactionType.TRANSFER).toBe('TRANSFER');
    });

    it('should have exactly 6 transaction types', () => {
      const types = Object.values(TransactionType);
      expect(types).toHaveLength(6);
    });
  });

  describe('IncomeType enum', () => {
    it('should have all required income types', () => {
      expect(IncomeType.INTEREST).toBe('INTEREST');
      expect(IncomeType.DIVIDEND).toBe('DIVIDEND');
      expect(IncomeType.STAKING).toBe('STAKING');
      expect(IncomeType.MINING).toBe('MINING');
      expect(IncomeType.AIRDROP).toBe('AIRDROP');
    });

    it('should have exactly 5 income types', () => {
      const types = Object.values(IncomeType);
      expect(types).toHaveLength(5);
    });
  });

  describe('Transaction interface', () => {
    it('should create a valid transaction object', () => {
      const transaction: Transaction = {
        id: uuidv4(),
        timestamp: new Date('2024-01-15T10:00:00Z'),
        type: TransactionType.BUY,
        asset: 'BTC',
        quantity: new Decimal('0.5'),
        price: new Decimal('45000'),
        fees: new Decimal('25'),
        currency: 'USD',
        source: 'Coinbase',
        sourceId: 'cb-123456',
        taxable: true,
        metadata: {
          orderId: 'order-789',
          notes: 'Test purchase',
        },
      };

      expect(transaction.id).toBeDefined();
      expect(transaction.type).toBe(TransactionType.BUY);
      expect(transaction.asset).toBe('BTC');
      expect(transaction.quantity.toString()).toBe('0.5');
      expect(transaction.price.toString()).toBe('45000');
      expect(transaction.fees.toString()).toBe('25');
      expect(transaction.taxable).toBe(true);
    });

    it('should support optional embedding field', () => {
      const transaction: Transaction = {
        id: uuidv4(),
        timestamp: new Date(),
        type: TransactionType.SELL,
        asset: 'ETH',
        quantity: new Decimal('10'),
        price: new Decimal('3000'),
        fees: new Decimal('15'),
        currency: 'USD',
        source: 'Binance',
        sourceId: 'bn-456789',
        taxable: true,
        metadata: {},
        embedding: new Float32Array([0.1, 0.2, 0.3]),
      };

      expect(transaction.embedding).toBeDefined();
      expect(transaction.embedding).toBeInstanceOf(Float32Array);
      expect(transaction.embedding?.length).toBe(3);
    });

    it('should handle metadata as flexible object', () => {
      const transaction: Transaction = {
        id: uuidv4(),
        timestamp: new Date(),
        type: TransactionType.INCOME,
        asset: 'BTC',
        quantity: new Decimal('0.001'),
        price: new Decimal('45000'),
        fees: new Decimal('0'),
        currency: 'USD',
        source: 'Mining Pool',
        sourceId: 'mp-001',
        taxable: true,
        metadata: {
          poolName: 'Slush Pool',
          blockHeight: 800000,
          hashRate: '100 TH/s',
          customField: { nested: 'value' },
        },
      };

      expect(transaction.metadata.poolName).toBe('Slush Pool');
      expect(transaction.metadata.blockHeight).toBe(800000);
      expect(transaction.metadata.hashRate).toBe('100 TH/s');
    });
  });

  describe('Income interface', () => {
    it('should create a valid income object', () => {
      const income: Income = {
        id: uuidv4(),
        type: IncomeType.STAKING,
        amount: new Decimal('100'),
        asset: 'ETH',
        date: new Date('2024-01-15'),
        source: 'Lido',
      };

      expect(income.id).toBeDefined();
      expect(income.type).toBe(IncomeType.STAKING);
      expect(income.amount.toString()).toBe('100');
      expect(income.asset).toBe('ETH');
      expect(income.source).toBe('Lido');
    });

    it('should support all income types', () => {
      const incomeTypes = [
        IncomeType.INTEREST,
        IncomeType.DIVIDEND,
        IncomeType.STAKING,
        IncomeType.MINING,
        IncomeType.AIRDROP,
      ];

      incomeTypes.forEach((type) => {
        const income: Income = {
          id: uuidv4(),
          type,
          amount: new Decimal('50'),
          asset: 'BTC',
          date: new Date(),
          source: 'Test Source',
        };

        expect(income.type).toBe(type);
      });
    });
  });
});
