import { describe, it, expect } from 'vitest';
import { type Position } from '../../src/types/position.js';
import { type TaxLot, LotStatus, AccountingMethod } from '../../src/types/tax-lot.js';
import { Decimal } from '../../src/utils/decimal.js';
import { v4 as uuidv4 } from 'uuid';

describe('Position Types', () => {
  describe('Position interface', () => {
    it('should create a valid position object', () => {
      const lot: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date('2024-01-01'),
        quantity: new Decimal('2'),
        originalQuantity: new Decimal('2'),
        costBasis: new Decimal('90000'),
        unitCostBasis: new Decimal('45000'),
        currency: 'USD',
        source: 'Coinbase',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.OPEN,
      };

      const position: Position = {
        asset: 'BTC',
        totalQuantity: new Decimal('2'),
        totalCostBasis: new Decimal('90000'),
        averageCostBasis: new Decimal('45000'),
        currentPrice: new Decimal('50000'),
        marketValue: new Decimal('100000'),
        unrealizedGain: new Decimal('10000'),
        unrealizedGainPercent: 11.11,
        lots: [lot],
        lastUpdated: new Date(),
      };

      expect(position.asset).toBe('BTC');
      expect(position.totalQuantity.toString()).toBe('2');
      expect(position.totalCostBasis.toString()).toBe('90000');
      expect(position.averageCostBasis.toString()).toBe('45000');
      expect(position.lots).toHaveLength(1);
    });

    it('should calculate market value correctly', () => {
      const totalQuantity = new Decimal('3.5');
      const currentPrice = new Decimal('48000');
      const marketValue = totalQuantity.times(currentPrice);

      const position: Position = {
        asset: 'BTC',
        totalQuantity,
        totalCostBasis: new Decimal('157500'),
        averageCostBasis: new Decimal('45000'),
        currentPrice,
        marketValue,
        unrealizedGain: new Decimal('10500'),
        unrealizedGainPercent: 6.67,
        lots: [],
        lastUpdated: new Date(),
      };

      expect(position.marketValue.toString()).toBe('168000');
      expect(position.totalQuantity.times(position.currentPrice).equals(position.marketValue)).toBe(true);
    });

    it('should calculate unrealized gain correctly', () => {
      const marketValue = new Decimal('100000');
      const totalCostBasis = new Decimal('90000');
      const unrealizedGain = marketValue.minus(totalCostBasis);

      const position: Position = {
        asset: 'BTC',
        totalQuantity: new Decimal('2'),
        totalCostBasis,
        averageCostBasis: new Decimal('45000'),
        currentPrice: new Decimal('50000'),
        marketValue,
        unrealizedGain,
        unrealizedGainPercent: 11.11,
        lots: [],
        lastUpdated: new Date(),
      };

      expect(position.unrealizedGain.toString()).toBe('10000');
      expect(position.marketValue.minus(position.totalCostBasis).equals(position.unrealizedGain)).toBe(true);
    });

    it('should calculate unrealized gain percentage correctly', () => {
      const totalCostBasis = new Decimal('90000');
      const unrealizedGain = new Decimal('10000');
      const unrealizedGainPercent = unrealizedGain.dividedBy(totalCostBasis).times(100).toNumber();

      const position: Position = {
        asset: 'BTC',
        totalQuantity: new Decimal('2'),
        totalCostBasis,
        averageCostBasis: new Decimal('45000'),
        currentPrice: new Decimal('50000'),
        marketValue: new Decimal('100000'),
        unrealizedGain,
        unrealizedGainPercent,
        lots: [],
        lastUpdated: new Date(),
      };

      expect(position.unrealizedGainPercent).toBeCloseTo(11.11, 2);
    });

    it('should handle unrealized losses (negative gains)', () => {
      const marketValue = new Decimal('80000');
      const totalCostBasis = new Decimal('90000');
      const unrealizedGain = marketValue.minus(totalCostBasis);

      const position: Position = {
        asset: 'BTC',
        totalQuantity: new Decimal('2'),
        totalCostBasis,
        averageCostBasis: new Decimal('45000'),
        currentPrice: new Decimal('40000'),
        marketValue,
        unrealizedGain,
        unrealizedGainPercent: -11.11,
        lots: [],
        lastUpdated: new Date(),
      };

      expect(position.unrealizedGain.toString()).toBe('-10000');
      expect(position.unrealizedGain.isNegative()).toBe(true);
      expect(position.unrealizedGainPercent).toBeLessThan(0);
    });

    it('should aggregate multiple lots', () => {
      const lot1: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date('2024-01-01'),
        quantity: new Decimal('1'),
        originalQuantity: new Decimal('1'),
        costBasis: new Decimal('45000'),
        unitCostBasis: new Decimal('45000'),
        currency: 'USD',
        source: 'Coinbase',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.OPEN,
      };

      const lot2: TaxLot = {
        id: uuidv4(),
        transactionId: uuidv4(),
        asset: 'BTC',
        acquiredDate: new Date('2024-02-01'),
        quantity: new Decimal('1.5'),
        originalQuantity: new Decimal('1.5'),
        costBasis: new Decimal('67500'),
        unitCostBasis: new Decimal('45000'),
        currency: 'USD',
        source: 'Binance',
        method: AccountingMethod.FIFO,
        disposals: [],
        status: LotStatus.OPEN,
      };

      const totalQuantity = lot1.quantity.plus(lot2.quantity);
      const totalCostBasis = lot1.costBasis.plus(lot2.costBasis);
      const averageCostBasis = totalCostBasis.dividedBy(totalQuantity);

      const position: Position = {
        asset: 'BTC',
        totalQuantity,
        totalCostBasis,
        averageCostBasis,
        currentPrice: new Decimal('50000'),
        marketValue: totalQuantity.times(new Decimal('50000')),
        unrealizedGain: totalQuantity.times(new Decimal('50000')).minus(totalCostBasis),
        unrealizedGainPercent: 11.11,
        lots: [lot1, lot2],
        lastUpdated: new Date(),
      };

      expect(position.lots).toHaveLength(2);
      expect(position.totalQuantity.toString()).toBe('2.5');
      expect(position.totalCostBasis.toString()).toBe('112500');
      expect(position.averageCostBasis.toString()).toBe('45000');
    });

    it('should track last update timestamp', () => {
      const lastUpdated = new Date('2024-11-16T10:00:00Z');

      const position: Position = {
        asset: 'BTC',
        totalQuantity: new Decimal('1'),
        totalCostBasis: new Decimal('45000'),
        averageCostBasis: new Decimal('45000'),
        currentPrice: new Decimal('50000'),
        marketValue: new Decimal('50000'),
        unrealizedGain: new Decimal('5000'),
        unrealizedGainPercent: 11.11,
        lots: [],
        lastUpdated,
      };

      expect(position.lastUpdated).toEqual(lastUpdated);
    });
  });
});
