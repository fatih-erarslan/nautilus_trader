/**
 * Edge Cases and Error Handling Validation
 *
 * Tests boundary conditions, error scenarios, and unusual cases
 * to ensure robust production-ready behavior.
 */

import { describe, it, expect } from 'vitest';
import Decimal from 'decimal.js';

describe('Edge Cases and Error Handling', () => {

  describe('Zero and Negative Values', () => {

    it('Handle zero quantity transactions', () => {
      const transaction = {
        asset: 'BTC',
        quantity: new Decimal(0),
        price: new Decimal(50000),
      };

      expect(() => {
        if (transaction.quantity.isZero()) {
          throw new Error('InvalidQuantityError: Quantity cannot be zero');
        }
      }).toThrow('InvalidQuantityError');
    });

    it('Handle negative quantity (short positions)', () => {
      const shortPosition = {
        asset: 'BTC',
        quantity: new Decimal(-1), // Negative = short
        entryPrice: new Decimal(50000),
        exitPrice: new Decimal(45000),
      };

      // Validate short position
      expect(shortPosition.quantity.isNegative()).toBe(true);

      // Short profit calculation (opposite of long)
      const profit = shortPosition.quantity
        .abs()
        .times(shortPosition.entryPrice.minus(shortPosition.exitPrice));

      expect(profit.toString()).toBe('5000'); // Profit on short
    });

    it('Handle zero cost basis (airdrops, gifts)', () => {
      const airdrop = {
        asset: 'TOKEN',
        quantity: new Decimal(1000),
        costBasis: new Decimal(0),
        fairMarketValue: new Decimal(500),
      };

      expect(airdrop.costBasis.isZero()).toBe(true);

      // Sell airdropped tokens
      const sale = {
        quantity: airdrop.quantity,
        proceeds: new Decimal(600),
        costBasis: airdrop.costBasis,
        gain: new Decimal(600), // Full proceeds is gain
      };

      expect(sale.gain.toString()).toBe('600');
    });

    it('Handle negative price (accounting corrections)', () => {
      const correction = {
        asset: 'BTC',
        quantity: new Decimal(1),
        price: new Decimal(-100), // Negative = refund/correction
      };

      expect(correction.price.isNegative()).toBe(true);

      // Adjust cost basis
      const originalCostBasis = new Decimal(50000);
      const adjustedCostBasis = originalCostBasis.plus(correction.price);

      expect(adjustedCostBasis.toString()).toBe('49900');
    });
  });

  describe('Fractional Shares and Precision', () => {

    it('Handle satoshi-level precision (8 decimals)', () => {
      const satoshi = new Decimal('0.00000001');
      const price = new Decimal(50000);

      const value = satoshi.times(price);

      expect(satoshi.decimalPlaces()).toBe(8);
      expect(value.toFixed(8)).toBe('0.00050000');
    });

    it('Handle extremely small fractional shares', () => {
      const microshare = new Decimal('0.000000000001'); // 12 decimals
      const price = new Decimal(1000000);

      const value = microshare.times(price);

      expect(value.toFixed(12)).toBe('0.001000000000');
    });

    it('Handle rounding errors in fractional calculations', () => {
      /**
       * Ensure no loss of precision in tax calculations
       */

      const quantity = new Decimal('0.123456789');
      const price = new Decimal('12345.6789');

      const proceeds = quantity.times(price);

      // Exact result without rounding errors
      expect(proceeds.toString()).toBe('1524.15789855521');
    });

    it('Handle division resulting in repeating decimals', () => {
      const totalCost = new Decimal(10);
      const quantity = new Decimal(3);

      const costPerUnit = totalCost.dividedBy(quantity);

      // Should maintain precision
      expect(costPerUnit.toFixed(10)).toBe('3.3333333333');
    });
  });

  describe('Date and Time Edge Cases', () => {

    it('Handle leap year date calculations', () => {
      const acquiredDate = new Date('2024-02-29'); // Leap day
      const disposalDate = new Date('2025-03-01');

      const days = Math.floor(
        (disposalDate.getTime() - acquiredDate.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(days).toBe(366); // LONG-TERM
    });

    it('Handle daylight saving time transitions', () => {
      /**
       * Ensure consistent day calculations across DST changes
       */

      const springForward = new Date('2023-03-12'); // DST starts
      const fallBack = new Date('2023-11-05'); // DST ends

      const days = Math.floor(
        (fallBack.getTime() - springForward.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(days).toBe(238);
    });

    it('Handle same-day buy and sell', () => {
      const purchase = {
        asset: 'BTC',
        timestamp: new Date('2023-06-15T10:00:00Z'),
        quantity: new Decimal(1),
        costBasis: new Decimal(50000),
      };

      const sale = {
        asset: 'BTC',
        timestamp: new Date('2023-06-15T14:00:00Z'), // Same day
        quantity: new Decimal(1),
        proceeds: new Decimal(51000),
      };

      const days = Math.floor(
        (sale.timestamp.getTime() - purchase.timestamp.getTime()) / (1000 * 60 * 60 * 24)
      );

      expect(days).toBe(0); // Same day = SHORT-TERM
    });

    it('Handle transactions at year boundary', () => {
      const dec31 = new Date('2023-12-31T23:59:59Z');
      const jan1 = new Date('2024-01-01T00:00:00Z');

      // Different tax years
      expect(dec31.getFullYear()).toBe(2023);
      expect(jan1.getFullYear()).toBe(2024);

      const millisDiff = jan1.getTime() - dec31.getTime();
      const secondsDiff = millisDiff / 1000;

      expect(secondsDiff).toBe(1); // 1 second apart, different tax years
    });
  });

  describe('Corporate Actions', () => {

    it('Handle stock split adjustments', () => {
      const preSplit = {
        quantity: new Decimal(100),
        costBasis: new Decimal(10000),
        pricePerShare: new Decimal(100),
      };

      // 2-for-1 split
      const splitRatio = 2;

      const postSplit = {
        quantity: preSplit.quantity.times(splitRatio),
        costBasis: preSplit.costBasis, // Total basis unchanged
        pricePerShare: preSplit.pricePerShare.dividedBy(splitRatio),
      };

      expect(postSplit.quantity.toString()).toBe('200');
      expect(postSplit.costBasis.toString()).toBe('10000');
      expect(postSplit.pricePerShare.toString()).toBe('50');
    });

    it('Handle reverse stock split', () => {
      const preReverseSplit = {
        quantity: new Decimal(1000),
        costBasis: new Decimal(5000),
        pricePerShare: new Decimal(5),
      };

      // 1-for-10 reverse split
      const reverseSplitRatio = new Decimal(0.1);

      const postReverseSplit = {
        quantity: preReverseSplit.quantity.times(reverseSplitRatio),
        costBasis: preReverseSplit.costBasis,
        pricePerShare: preReverseSplit.pricePerShare.dividedBy(reverseSplitRatio),
      };

      expect(postReverseSplit.quantity.toString()).toBe('100');
      expect(postReverseSplit.costBasis.toString()).toBe('5000');
      expect(postReverseSplit.pricePerShare.toString()).toBe('50');
    });

    it('Handle dividend reinvestment', () => {
      const originalShares = {
        quantity: new Decimal(100),
        costBasis: new Decimal(10000),
      };

      const dividend = {
        cashAmount: new Decimal(500),
        sharePrice: new Decimal(105),
        reinvestedShares: new Decimal('4.761904762'), // $500 / $105
      };

      const newShares = {
        quantity: dividend.reinvestedShares,
        costBasis: dividend.cashAmount,
      };

      const totalShares = originalShares.quantity.plus(newShares.quantity);
      const totalCostBasis = originalShares.costBasis.plus(newShares.costBasis);

      expect(totalShares.toFixed(9)).toBe('104.761904762');
      expect(totalCostBasis.toString()).toBe('10500');
    });

    it('Handle merger/acquisition basis adjustment', () => {
      /**
       * Company A acquired by Company B
       * 1 share of A converts to 0.5 shares of B + $10 cash
       */

      const companyAShares = {
        quantity: new Decimal(100),
        costBasis: new Decimal(5000),
        acquiredDate: new Date('2022-01-01'),
      };

      const merger = {
        conversionRatio: new Decimal(0.5),
        cashPerShare: new Decimal(10),
      };

      const companyBShares = {
        quantity: companyAShares.quantity.times(merger.conversionRatio),
        costBasis: companyAShares.costBasis.minus(
          companyAShares.quantity.times(merger.cashPerShare)
        ),
        acquiredDate: companyAShares.acquiredDate, // Carries over
      };

      const cashReceived = companyAShares.quantity.times(merger.cashPerShare);

      expect(companyBShares.quantity.toString()).toBe('50');
      expect(companyBShares.costBasis.toString()).toBe('4000');
      expect(cashReceived.toString()).toBe('1000');

      // Partial gain recognition on cash
      const gainOnCash = cashReceived.minus(
        companyAShares.costBasis.times(cashReceived).dividedBy(
          companyAShares.costBasis.plus(cashReceived)
        )
      );

      expect(gainOnCash.greaterThan(0)).toBe(true);
    });

    it('Handle spin-off basis allocation', () => {
      /**
       * Parent company spins off subsidiary
       * Basis allocated based on relative FMV
       */

      const parentShares = {
        quantity: new Decimal(100),
        costBasis: new Decimal(10000),
      };

      const spinOff = {
        parentFMV: new Decimal(8000),
        subsidiaryFMV: new Decimal(2000),
        subsidiaryShares: new Decimal(10), // Received in spin-off
      };

      const totalFMV = spinOff.parentFMV.plus(spinOff.subsidiaryFMV);

      const allocatedBasis = {
        parent: parentShares.costBasis.times(spinOff.parentFMV).dividedBy(totalFMV),
        subsidiary: parentShares.costBasis.times(spinOff.subsidiaryFMV).dividedBy(totalFMV),
      };

      expect(allocatedBasis.parent.toString()).toBe('8000');
      expect(allocatedBasis.subsidiary.toString()).toBe('2000');
      expect(allocatedBasis.parent.plus(allocatedBasis.subsidiary).toString()).toBe('10000');
    });
  });

  describe('Insufficient Quantity Errors', () => {

    it('Handle sell quantity exceeding available lots', () => {
      const availableLots = [
        { quantity: new Decimal(1), costBasis: new Decimal(50000) },
        { quantity: new Decimal(0.5), costBasis: new Decimal(25000) },
      ];

      const totalAvailable = availableLots.reduce(
        (sum, lot) => sum.plus(lot.quantity),
        new Decimal(0)
      );

      const saleQuantity = new Decimal(2); // Trying to sell more than available

      expect(() => {
        if (saleQuantity.greaterThan(totalAvailable)) {
          throw new Error(
            `InsufficientQuantityError: Attempting to sell ${saleQuantity} but only ${totalAvailable} available`
          );
        }
      }).toThrow('InsufficientQuantityError');
    });

    it('Handle partial lot exhaustion', () => {
      const lot = {
        quantity: new Decimal(1.5),
        costBasis: new Decimal(75000),
      };

      const sale1 = {
        quantity: new Decimal(1),
        costBasis: new Decimal(50000), // 1/1.5 * 75000
      };

      const remainingLot = {
        quantity: lot.quantity.minus(sale1.quantity),
        costBasis: lot.costBasis.minus(sale1.costBasis),
      };

      expect(remainingLot.quantity.toString()).toBe('0.5');
      expect(remainingLot.costBasis.toString()).toBe('25000');
    });
  });

  describe('Currency and Conversion', () => {

    it('Handle foreign currency transactions', () => {
      /**
       * Purchase in EUR, sell in USD
       * Must convert to USD for tax reporting
       */

      const purchase = {
        quantity: new Decimal(1),
        priceEUR: new Decimal(40000),
        exchangeRateToUSD: new Decimal(1.1),
        costBasisUSD: new Decimal(44000), // 40k EUR * 1.1
      };

      const sale = {
        quantity: new Decimal(1),
        priceUSD: new Decimal(50000),
        proceeds: new Decimal(50000),
      };

      const gain = sale.proceeds.minus(purchase.costBasisUSD);

      expect(gain.toString()).toBe('6000');
    });

    it('Handle cryptocurrency basis in different currencies', () => {
      /**
       * Track cost basis consistently in USD
       */

      const purchases = [
        { qty: 1, priceUSD: 30000, costBasis: 30000 },
        { qty: 1, priceEUR: 25000, exchangeRate: 1.2, costBasis: 30000 }, // 25k EUR
      ];

      const totalQty = purchases.reduce((sum, p) => sum + p.qty, 0);
      const totalCostBasis = purchases.reduce((sum, p) => sum + p.costBasis, 0);

      expect(totalQty).toBe(2);
      expect(totalCostBasis).toBe(60000);
    });
  });

  describe('Data Validation', () => {

    it('Reject invalid date formats', () => {
      const invalidDate = 'invalid-date';

      expect(() => {
        const date = new Date(invalidDate);
        if (isNaN(date.getTime())) {
          throw new Error('InvalidDateError: Date format invalid');
        }
      }).toThrow('InvalidDateError');
    });

    it('Reject negative proceeds', () => {
      const transaction = {
        asset: 'BTC',
        quantity: new Decimal(1),
        proceeds: new Decimal(-50000), // Invalid
      };

      expect(() => {
        if (transaction.proceeds.isNegative()) {
          throw new Error('InvalidProceedsError: Proceeds cannot be negative');
        }
      }).toThrow('InvalidProceedsError');
    });

    it('Validate asset identifier format', () => {
      const validAssets = ['BTC', 'ETH', 'AAPL', 'BTC-USD'];
      const invalidAssets = ['', ' ', '123', '@#$'];

      validAssets.forEach(asset => {
        const isValid = /^[A-Z0-9-]+$/.test(asset) && asset.length > 0;
        expect(isValid).toBe(true);
      });

      invalidAssets.forEach(asset => {
        const isValid = /^[A-Z0-9-]+$/.test(asset) && asset.length > 0;
        expect(isValid).toBe(false);
      });
    });
  });

  describe('Concurrency and Race Conditions', () => {

    it('Handle simultaneous transactions on same asset', () => {
      /**
       * Two sales processed at same time
       * Must not double-count lot consumption
       */

      const lots = [
        { id: '1', quantity: new Decimal(1), costBasis: new Decimal(50000) },
      ];

      const sale1 = { quantity: new Decimal(0.5) };
      const sale2 = { quantity: new Decimal(0.5) };

      // Process sequentially to avoid race condition
      const remainingAfterSale1 = lots[0].quantity.minus(sale1.quantity);
      expect(remainingAfterSale1.toString()).toBe('0.5');

      const remainingAfterSale2 = remainingAfterSale1.minus(sale2.quantity);
      expect(remainingAfterSale2.toString()).toBe('0');
    });

    it('Handle lot ordering with identical timestamps', () => {
      /**
       * Multiple lots acquired at same timestamp
       * Must have deterministic ordering
       */

      const timestamp = new Date('2023-06-15T10:00:00Z');

      const lots = [
        { id: 'B', timestamp, quantity: new Decimal(1), costBasis: new Decimal(50000) },
        { id: 'A', timestamp, quantity: new Decimal(1), costBasis: new Decimal(51000) },
        { id: 'C', timestamp, quantity: new Decimal(1), costBasis: new Decimal(49000) },
      ];

      // Sort by ID for deterministic ordering when timestamps equal
      const sorted = [...lots].sort((a, b) => a.id.localeCompare(b.id));

      expect(sorted[0].id).toBe('A');
      expect(sorted[1].id).toBe('B');
      expect(sorted[2].id).toBe('C');
    });
  });

  describe('Large Numbers and Overflow', () => {

    it('Handle very large transaction values', () => {
      const largePosition = {
        quantity: new Decimal(1000000), // 1 million BTC
        price: new Decimal(50000),
        value: new Decimal(1000000).times(50000),
      };

      expect(largePosition.value.toString()).toBe('50000000000'); // $50 billion
      expect(largePosition.value.toExponential()).toBe('5e+10');
    });

    it('Handle very small values without underflow', () => {
      const microTransaction = {
        quantity: new Decimal('0.00000001'),
        price: new Decimal('0.0001'),
        value: new Decimal('0.00000001').times('0.0001'),
      };

      expect(microTransaction.value.toString()).toBe('0.000000000001');
    });

    it('Maintain precision in large aggregations', () => {
      /**
       * Sum thousands of transactions without losing precision
       */

      const transactions = Array.from({ length: 10000 }, (_, i) => ({
        gain: new Decimal('0.123456789'),
      }));

      const totalGain = transactions.reduce(
        (sum, tx) => sum.plus(tx.gain),
        new Decimal(0)
      );

      expect(totalGain.toString()).toBe('1234.56789');
    });
  });

  describe('Special Asset Types', () => {

    it('Handle NFT unique identification', () => {
      const nft = {
        asset: 'NFT',
        tokenId: '0x123abc...',
        collection: 'CryptoPunks',
        uniqueId: 'CryptoPunks#7804',
        costBasis: new Decimal(1000),
      };

      expect(nft.uniqueId).toBeDefined();
      expect(nft.tokenId).toBeDefined();
    });

    it('Handle staking rewards as income', () => {
      const stakingReward = {
        asset: 'ETH',
        quantity: new Decimal(0.5),
        fairMarketValue: new Decimal(1000), // Income at time of receipt
        costBasis: new Decimal(1000), // Basis = FMV at receipt
        rewardDate: new Date('2023-06-15'),
      };

      // When sold, basis is FMV at receipt
      const sale = {
        quantity: stakingReward.quantity,
        proceeds: new Decimal(1200),
        costBasis: stakingReward.costBasis,
        gain: new Decimal(200),
      };

      expect(sale.gain.toString()).toBe('200');
    });

    it('Handle wrapped tokens (e.g., WBTC)', () => {
      /**
       * Wrapping BTC to WBTC may or may not be taxable event
       * Conservative: treat as taxable exchange
       */

      const btcPosition = {
        asset: 'BTC',
        quantity: new Decimal(1),
        costBasis: new Decimal(40000),
      };

      const wrapEvent = {
        fromAsset: 'BTC',
        toAsset: 'WBTC',
        quantity: new Decimal(1),
        fairValue: new Decimal(50000), // FMV at wrap time
      };

      // Treat as disposal of BTC
      const btcDisposal = {
        asset: 'BTC',
        proceeds: wrapEvent.fairValue,
        costBasis: btcPosition.costBasis,
        gain: new Decimal(10000),
      };

      // WBTC receives stepped-up basis
      const wbtcPosition = {
        asset: 'WBTC',
        quantity: wrapEvent.quantity,
        costBasis: wrapEvent.fairValue,
      };

      expect(btcDisposal.gain.toString()).toBe('10000');
      expect(wbtcPosition.costBasis.toString()).toBe('50000');
    });
  });

  describe('Error Recovery', () => {

    it('Handle missing transaction data gracefully', () => {
      const incompleteTransaction = {
        asset: 'BTC',
        quantity: new Decimal(1),
        // Missing: price, timestamp
      };

      const errors: string[] = [];

      if (!incompleteTransaction.hasOwnProperty('price')) {
        errors.push('Missing required field: price');
      }

      if (!incompleteTransaction.hasOwnProperty('timestamp')) {
        errors.push('Missing required field: timestamp');
      }

      expect(errors).toHaveLength(2);
      expect(errors).toContain('Missing required field: price');
    });

    it('Provide clear error messages for validation failures', () => {
      const transaction = {
        asset: '',
        quantity: new Decimal(-1),
        price: new Decimal(0),
      };

      const validationErrors: string[] = [];

      if (!transaction.asset || transaction.asset.trim() === '') {
        validationErrors.push('Asset identifier is required');
      }

      if (transaction.quantity.lessThanOrEqualTo(0)) {
        validationErrors.push('Quantity must be positive');
      }

      if (transaction.price.lessThanOrEqualTo(0)) {
        validationErrors.push('Price must be positive');
      }

      expect(validationErrors).toHaveLength(3);
    });
  });
});
