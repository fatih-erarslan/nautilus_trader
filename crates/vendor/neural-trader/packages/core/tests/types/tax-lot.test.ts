import { describe, it, expect } from 'vitest';
import { AccountingMethod, LotStatus, type TaxLot } from '../../src/types/tax-lot.js';
import { Decimal } from '../../src/utils/decimal.js';
import { v4 as uuidv4 } from 'uuid';

describe('Tax Lot Types', () => {
  describe('AccountingMethod enum', () => {
    it('should have all required accounting methods', () => {
      expect(AccountingMethod.FIFO).toBe('FIFO');
      expect(AccountingMethod.LIFO).toBe('LIFO');
      expect(AccountingMethod.HIFO).toBe('HIFO');
      expect(AccountingMethod.SPECIFIC_ID).toBe('SPECIFIC_ID');
      expect(AccountingMethod.AVERAGE_COST).toBe('AVERAGE_COST');
    });

    it('should have exactly 5 accounting methods', () => {
      const methods = Object.values(AccountingMethod);
      expect(methods).toHaveLength(5);
    });
  });

  describe('LotStatus enum', () => {
    it('should have all required statuses', () => {
      expect(LotStatus.OPEN).toBe('OPEN');
      expect(LotStatus.PARTIAL).toBe('PARTIAL');
      expect(LotStatus.CLOSED).toBe('CLOSED');
    });

    it('should have exactly 3 statuses', () => {
      const statuses = Object.values(LotStatus);
      expect(statuses).toHaveLength(3);
    });
  });

  describe('TaxLot interface', () => {
    it('should create a valid tax lot object', () => {
      const lot: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date('2024-01-01'),
        quantity: new Decimal('1.5'),
        originalQuantity: new Decimal('1.5'),
        costBasis: new Decimal('67500'),
        unitCostBasis: new Decimal('45000'),
        currency: 'USD',
        source: 'Coinbase',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.OPEN,
      };

      expect(lot.id).toBeDefined();
      expect(lot.asset).toBe('BTC');
      expect(lot.quantity.toString()).toBe('1.5');
      expect(lot.originalQuantity.toString()).toBe('1.5');
      expect(lot.costBasis.toString()).toBe('67500');
      expect(lot.unitCostBasis.toString()).toBe('45000');
      expect(lot.method).toBe(AccountingMethod.FIFO);
      expect(lot.status).toBe(LotStatus.OPEN);
      expect(lot.disposals).toHaveLength(0);
    });

    it('should maintain quantity invariants', () => {
      const lot: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'ETH',
        acquiredDate: new Date('2024-01-01'),
        quantity: new Decimal('5'),
        originalQuantity: new Decimal('10'),
        costBasis: new Decimal('15000'),
        unitCostBasis: new Decimal('3000'),
        currency: 'USD',
        source: 'Binance',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.PARTIAL,
      };

      // Quantity should be <= originalQuantity
      expect(lot.quantity.lessThanOrEqualTo(lot.originalQuantity)).toBe(true);

      // Status should be PARTIAL when quantity < originalQuantity
      expect(lot.status).toBe(LotStatus.PARTIAL);
    });

    it('should calculate unit cost basis correctly', () => {
      const quantity = new Decimal('2.5');
      const costBasis = new Decimal('100000');
      const unitCostBasis = costBasis.dividedBy(quantity);

      const lot: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date(),
        quantity,
        originalQuantity: quantity,
        costBasis,
        unitCostBasis,
        currency: 'USD',
        source: 'Kraken',
        method: AccountingMethod.AVERAGE_COST,
        disposals: [],
        status: LotStatus.OPEN,
      };

      expect(lot.unitCostBasis.toString()).toBe('40000');
      expect(lot.quantity.times(lot.unitCostBasis).equals(lot.costBasis)).toBe(true);
    });

    it('should support all accounting methods', () => {
      const methods = Object.values(AccountingMethod);

      methods.forEach((method) => {
        const lot: TaxLot = {
          id: uuidv4(),
          transactionId: uuidv4(),
          asset: 'BTC',
          acquiredDate: new Date(),
          quantity: new Decimal('1'),
          originalQuantity: new Decimal('1'),
          costBasis: new Decimal('45000'),
          unitCostBasis: new Decimal('45000'),
          currency: 'USD',
          source: 'Exchange',
          method,
          disposals: [],
          status: LotStatus.OPEN,
        };

        expect(lot.method).toBe(method);
      });
    });

    it('should handle closed lot status', () => {
      const lot: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date('2024-01-01'),
        quantity: new Decimal('0'),
        originalQuantity: new Decimal('1'),
        costBasis: new Decimal('0'),
        unitCostBasis: new Decimal('45000'),
        currency: 'USD',
        source: 'Coinbase',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.CLOSED,
      };

      expect(lot.quantity.isZero()).toBe(true);
      expect(lot.status).toBe(LotStatus.CLOSED);
    });
  });
});
