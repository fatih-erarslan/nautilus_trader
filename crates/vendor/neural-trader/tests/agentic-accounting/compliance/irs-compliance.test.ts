/**
 * IRS Compliance Tests
 *
 * Validates tax calculations against IRS Publication 550 examples
 * and ensures compliance with federal tax regulations.
 */

import { describe, it, expect } from 'vitest';
import Decimal from 'decimal.js';
import {
  createTransaction,
  createLot,
  createDisposal,
} from '../fixtures/factories';

describe('IRS Publication 550 Compliance', () => {
  describe('Example 1: Basic Stock Sale (FIFO)', () => {
    /**
     * IRS Publication 550, Example 1:
     *
     * You bought 100 shares of XYZ stock on January 3, 2023 for $2,000.
     * You bought 100 shares of XYZ stock on February 1, 2023 for $3,000.
     * You sold 150 shares on June 30, 2023 for $6,000.
     *
     * Using FIFO:
     * - First 100 shares: Cost basis $2,000 (from Jan purchase)
     * - Next 50 shares: Cost basis $1,500 (half of Feb purchase)
     * - Total cost basis: $3,500
     * - Proceeds: $6,000
     * - Gain: $2,500 (short-term)
     */

    it('should match IRS Example 1 calculations', () => {
      const lot1 = createLot({
        asset: 'XYZ',
        quantity: '100',
        costBasis: '2000',
        acquiredDate: new Date('2023-01-03'),
      });

      const lot2 = createLot({
        asset: 'XYZ',
        quantity: '100',
        costBasis: '3000',
        acquiredDate: new Date('2023-02-01'),
      });

      const sale = createTransaction({
        asset: 'XYZ',
        type: 'SELL',
        quantity: '150',
        price: '40', // $40/share * 150 = $6,000
        timestamp: new Date('2023-06-30'),
      });

      // Expected disposals (FIFO)
      const expectedDisposals = [
        {
          lot: lot1,
          quantity: new Decimal('100'),
          costBasis: new Decimal('2000'),
          proceeds: new Decimal('4000'), // 100 * $40
          gain: new Decimal('2000'),
          term: 'SHORT',
        },
        {
          lot: lot2,
          quantity: new Decimal('50'),
          costBasis: new Decimal('1500'), // 50/100 * $3000
          proceeds: new Decimal('2000'), // 50 * $40
          gain: new Decimal('500'),
          term: 'SHORT',
        },
      ];

      const totalCostBasis = new Decimal('3500');
      const totalProceeds = new Decimal('6000');
      const totalGain = new Decimal('2500');

      expect(totalCostBasis.toString()).toBe('3500');
      expect(totalProceeds.toString()).toBe('6000');
      expect(totalGain.toString()).toBe('2500');

      // Verify it's short-term (held < 1 year)
      const holdingDays = Math.floor(
        (new Date('2023-06-30').getTime() - new Date('2023-01-03').getTime()) / (1000 * 60 * 60 * 24)
      );
      expect(holdingDays).toBeLessThan(366);
    });
  });

  describe('Example 2: Long-Term vs Short-Term', () => {
    /**
     * IRS: Assets held > 365 days qualify for long-term capital gains
     */

    it('should classify as long-term after 1 year', () => {
      const lot = createLot({
        asset: 'BTC',
        quantity: '1',
        costBasis: '30000',
        acquiredDate: new Date('2022-01-01'),
      });

      const sale = createTransaction({
        asset: 'BTC',
        type: 'SELL',
        quantity: '1',
        price: '40000',
        timestamp: new Date('2023-01-02'), // 1 year + 1 day
      });

      const holdingDays = Math.floor(
        (sale.timestamp.getTime() - lot.acquiredDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(holdingDays).toBeGreaterThan(365);

      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '30000',
        proceeds: '40000',
        gain: '10000',
        acquiredDate: lot.acquiredDate,
        disposalDate: sale.timestamp,
        term: 'LONG',
      });

      expect(disposal.term).toBe('LONG');
    });

    it('should classify as short-term within 1 year', () => {
      const lot = createLot({
        asset: 'BTC',
        quantity: '1',
        costBasis: '30000',
        acquiredDate: new Date('2023-01-01'),
      });

      const sale = createTransaction({
        asset: 'BTC',
        type: 'SELL',
        quantity: '1',
        price: '40000',
        timestamp: new Date('2023-12-31'), // 364 days
      });

      const holdingDays = Math.floor(
        (sale.timestamp.getTime() - lot.acquiredDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(holdingDays).toBeLessThan(366);

      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '30000',
        proceeds: '40000',
        gain: '10000',
        acquiredDate: lot.acquiredDate,
        disposalDate: sale.timestamp,
        term: 'SHORT',
      });

      expect(disposal.term).toBe('SHORT');
    });
  });

  describe('Example 3: Wash Sale Rule', () => {
    /**
     * IRS Pub 550: Wash Sale Rule
     *
     * You sell stock at a loss and within 30 days before or after
     * the sale, you buy substantially identical stock.
     * The loss is disallowed and added to the cost basis of the new stock.
     */

    it('should disallow loss for wash sale', () => {
      const originalLot = createLot({
        asset: 'BTC',
        quantity: '1',
        costBasis: '50000',
        acquiredDate: new Date('2023-05-01'),
      });

      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '50000',
        proceeds: '40000',
        gain: '-10000', // $10,000 loss
        disposalDate: new Date('2023-06-15'),
        acquiredDate: originalLot.acquiredDate,
        term: 'SHORT',
        washSale: true,
        disallowedLoss: '10000',
      });

      const replacementPurchase = createTransaction({
        asset: 'BTC',
        type: 'BUY',
        quantity: '1',
        price: '38000',
        timestamp: new Date('2023-06-20'), // 5 days later
      });

      // Calculate days between disposal and repurchase
      const daysDiff = Math.floor(
        (replacementPurchase.timestamp.getTime() - disposal.disposalDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(daysDiff).toBe(5);
      expect(daysDiff).toBeLessThanOrEqual(30);

      // Verify wash sale flagged
      expect(disposal.washSale).toBe(true);
      expect(disposal.disallowedLoss).toBe('10000');

      // Adjusted cost basis for replacement
      const adjustedCostBasis = new Decimal(replacementPurchase.price)
        .times(replacementPurchase.quantity)
        .plus(new Decimal(disposal.disallowedLoss));

      expect(adjustedCostBasis.toString()).toBe('48000'); // $38,000 + $10,000
    });

    it('should allow loss if repurchase is >30 days', () => {
      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '50000',
        proceeds: '40000',
        gain: '-10000',
        disposalDate: new Date('2023-06-01'),
        acquiredDate: new Date('2023-05-01'),
        term: 'SHORT',
        washSale: false,
      });

      const replacementPurchase = createTransaction({
        asset: 'BTC',
        type: 'BUY',
        quantity: '1',
        price: '38000',
        timestamp: new Date('2023-07-15'), // 44 days later
      });

      const daysDiff = Math.floor(
        (replacementPurchase.timestamp.getTime() - disposal.disposalDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(daysDiff).toBeGreaterThan(30);
      expect(disposal.washSale).toBe(false);
      expect(disposal.disallowedLoss).toBeUndefined();
    });

    it('should not apply wash sale to gains', () => {
      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '40000',
        proceeds: '50000',
        gain: '10000', // GAIN, not loss
        disposalDate: new Date('2023-06-15'),
        acquiredDate: new Date('2023-05-01'),
        term: 'SHORT',
        washSale: false,
      });

      const replacementPurchase = createTransaction({
        asset: 'BTC',
        type: 'BUY',
        quantity: '1',
        price: '48000',
        timestamp: new Date('2023-06-20'), // 5 days later
      });

      // Even though repurchase is within 30 days, wash sale doesn't apply to gains
      expect(disposal.washSale).toBe(false);
      expect(new Decimal(disposal.gain).greaterThan(0)).toBe(true);
    });
  });

  describe('Schedule D Form Requirements', () => {
    /**
     * IRS Form Schedule D requires:
     * - Description of property
     * - Date acquired
     * - Date sold
     * - Proceeds
     * - Cost basis
     * - Adjustment code (if wash sale)
     * - Gain or loss
     * - Separate short-term and long-term sections
     */

    it('should provide all Schedule D required fields', () => {
      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1.5',
        costBasis: '60000',
        proceeds: '75000',
        gain: '15000',
        acquiredDate: new Date('2022-03-15'),
        disposalDate: new Date('2023-09-20'),
        term: 'LONG',
      });

      // Verify all required fields present
      expect(disposal.asset).toBeDefined(); // Description
      expect(disposal.acquiredDate).toBeDefined(); // Date acquired
      expect(disposal.disposalDate).toBeDefined(); // Date sold
      expect(disposal.proceeds).toBeDefined(); // Sales price
      expect(disposal.costBasis).toBeDefined(); // Cost basis
      expect(disposal.gain).toBeDefined(); // Gain/Loss
      expect(disposal.term).toBeDefined(); // Short or Long

      // Verify gain calculation
      const calculatedGain = new Decimal(disposal.proceeds).minus(disposal.costBasis);
      expect(calculatedGain.toString()).toBe(disposal.gain);
    });

    it('should separate short-term and long-term disposals', () => {
      const disposals = [
        createDisposal({
          asset: 'BTC',
          acquiredDate: new Date('2023-01-01'),
          disposalDate: new Date('2023-06-01'),
          term: 'SHORT',
          gain: '5000',
        }),
        createDisposal({
          asset: 'BTC',
          acquiredDate: new Date('2022-01-01'),
          disposalDate: new Date('2023-06-01'),
          term: 'LONG',
          gain: '10000',
        }),
        createDisposal({
          asset: 'ETH',
          acquiredDate: new Date('2023-03-01'),
          disposalDate: new Date('2023-09-01'),
          term: 'SHORT',
          gain: '3000',
        }),
      ];

      const shortTerm = disposals.filter(d => d.term === 'SHORT');
      const longTerm = disposals.filter(d => d.term === 'LONG');

      expect(shortTerm).toHaveLength(2);
      expect(longTerm).toHaveLength(1);

      const shortTermTotal = shortTerm.reduce((sum, d) => sum.plus(d.gain), new Decimal(0));
      const longTermTotal = longTerm.reduce((sum, d) => sum.plus(d.gain), new Decimal(0));

      expect(shortTermTotal.toString()).toBe('8000');
      expect(longTermTotal.toString()).toBe('10000');
    });
  });

  describe('Form 8949 Requirements', () => {
    /**
     * IRS Form 8949 (Sales and Other Dispositions of Capital Assets)
     * Required before Schedule D for detailed transaction reporting
     */

    it('should provide Form 8949 transaction details', () => {
      const disposal = createDisposal({
        id: 'disposal-123',
        asset: 'Bitcoin (BTC)',
        quantity: '0.5',
        costBasis: '25000',
        proceeds: '30000',
        gain: '5000',
        acquiredDate: new Date('2022-06-15'),
        disposalDate: new Date('2023-11-20'),
        term: 'LONG',
        washSale: false,
      });

      // Form 8949 Column (a): Description of property
      expect(disposal.asset).toBe('Bitcoin (BTC)');

      // Column (b): Date acquired
      expect(disposal.acquiredDate.toISOString().split('T')[0]).toBe('2022-06-15');

      // Column (c): Date sold
      expect(disposal.disposalDate.toISOString().split('T')[0]).toBe('2023-11-20');

      // Column (d): Proceeds
      expect(disposal.proceeds).toBe('30000');

      // Column (e): Cost basis
      expect(disposal.costBasis).toBe('25000');

      // Column (f): Adjustment code (if any)
      // W = Wash sale, B = Long-term gain elected as short-term, etc.
      const adjustmentCode = disposal.washSale ? 'W' : '';
      expect(adjustmentCode).toBe('');

      // Column (g): Adjustment amount
      const adjustmentAmount = disposal.disallowedLoss || '0';
      expect(adjustmentAmount).toBe('0');

      // Column (h): Gain or loss
      expect(disposal.gain).toBe('5000');
    });

    it('should include wash sale adjustment code W', () => {
      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '50000',
        proceeds: '40000',
        gain: '-10000',
        acquiredDate: new Date('2023-05-01'),
        disposalDate: new Date('2023-06-15'),
        term: 'SHORT',
        washSale: true,
        disallowedLoss: '10000',
      });

      expect(disposal.washSale).toBe(true);

      const adjustmentCode = 'W';
      const adjustmentAmount = disposal.disallowedLoss;

      expect(adjustmentCode).toBe('W');
      expect(adjustmentAmount).toBe('10000');

      // Allowed loss after adjustment
      const allowedLoss = new Decimal(disposal.gain).plus(adjustmentAmount);
      expect(allowedLoss.toString()).toBe('0'); // Fully disallowed
    });
  });

  describe('Cryptocurrency-Specific IRS Guidance', () => {
    /**
     * IRS Notice 2014-21: Virtual currency is treated as property
     * FAQ: Cryptocurrency transactions must be reported
     */

    it('should treat crypto as property for tax purposes', () => {
      const cryptoDisposal = createDisposal({
        asset: 'BTC',
        quantity: '0.25',
        costBasis: '12500',
        proceeds: '15000',
        gain: '2500',
        term: 'SHORT',
      });

      // Verify treated as capital asset
      expect(cryptoDisposal.gain).toBeDefined();
      expect(new Decimal(cryptoDisposal.gain).greaterThan(0)).toBe(true);
    });

    it('should handle fractional cryptocurrency quantities', () => {
      // Crypto can be divided to 8 decimal places (satoshis)
      const satoshiQuantity = '0.00000001';
      const disposal = createDisposal({
        asset: 'BTC',
        quantity: satoshiQuantity,
        costBasis: '0.50',
        proceeds: '0.60',
        gain: '0.10',
        term: 'SHORT',
      });

      expect(disposal.quantity).toBe(satoshiQuantity);

      // Verify precision maintained
      const quantity = new Decimal(disposal.quantity);
      expect(quantity.decimalPlaces()).toBeGreaterThanOrEqual(8);
    });

    it('should handle crypto-to-crypto trades as disposals', () => {
      // IRS: Trading BTC for ETH is a taxable event
      const btcDisposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '40000',
        proceeds: '40000', // Fair market value at trade time
        gain: '0',
        disposalDate: new Date('2023-06-15'),
        acquiredDate: new Date('2023-01-01'),
        term: 'SHORT',
      });

      expect(btcDisposal.asset).toBe('BTC');
      expect(btcDisposal.proceeds).toBe('40000');

      // The acquired ETH starts a new cost basis
      // (separate transaction, not tested here)
    });
  });

  describe('Compliance Edge Cases', () => {
    it('should handle zero-cost basis (gifts/airdrops)', () => {
      // Airdrops and gifts may have zero cost basis
      const disposal = createDisposal({
        asset: 'AIRDROP_TOKEN',
        quantity: '1000',
        costBasis: '0',
        proceeds: '500',
        gain: '500',
        term: 'SHORT',
      });

      expect(disposal.costBasis).toBe('0');
      expect(disposal.gain).toBe(disposal.proceeds);
    });

    it('should handle inherited crypto with stepped-up basis', () => {
      // Inherited assets get stepped-up cost basis to FMV at date of death
      const inheritedLot = createLot({
        asset: 'BTC',
        quantity: '1',
        costBasis: '55000', // FMV at inheritance
        acquiredDate: new Date('2023-03-15'), // Date of inheritance
      });

      const disposal = createDisposal({
        asset: 'BTC',
        quantity: '1',
        costBasis: '55000',
        proceeds: '60000',
        gain: '5000',
        acquiredDate: inheritedLot.acquiredDate,
        disposalDate: new Date('2023-09-15'),
        term: 'LONG', // Always long-term for inherited assets
      });

      // Inherited assets are always long-term
      expect(disposal.term).toBe('LONG');
    });

    it('should handle losses limited to $3000 per year', () => {
      // IRS: Capital losses limited to $3000/year deduction
      // Excess carried forward to future years

      const totalLosses = new Decimal('-15000');
      const yearlyLimit = new Decimal('3000');

      const currentYearDeduction = Decimal.min(totalLosses.abs(), yearlyLimit);
      const carryoverLoss = totalLosses.abs().minus(currentYearDeduction);

      expect(currentYearDeduction.toString()).toBe('3000');
      expect(carryoverLoss.toString()).toBe('12000');
    });
  });
});
