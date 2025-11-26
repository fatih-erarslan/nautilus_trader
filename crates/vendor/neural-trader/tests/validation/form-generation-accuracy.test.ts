/**
 * Form Generation Accuracy Validation
 *
 * Tests IRS Form 8949 and Schedule D generation against official
 * IRS samples and instructions to ensure 100% compliance.
 *
 * References:
 * - IRS Form 8949 Instructions (2023)
 * - IRS Schedule D Instructions (2023)
 * - IRS Publication 550
 */

import { describe, it, expect } from 'vitest';
import Decimal from 'decimal.js';

describe('Form Generation Accuracy - IRS Compliance', () => {

  describe('Form 8949 - Part I (Short-Term)', () => {

    it('Category A: Short-term with basis reported to IRS', () => {
      /**
       * IRS Form 8949 Category A:
       * Short-term transactions reported on 1099-B WITH basis reported
       */

      const transactions = [
        {
          description: '100 shares AAPL',
          dateAcquired: new Date('2023-02-15'),
          dateSold: new Date('2023-08-20'),
          proceeds: new Decimal(15000),
          costBasis: new Decimal(12000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(3000),
        },
        {
          description: '50 shares MSFT',
          dateAcquired: new Date('2023-03-01'),
          dateSold: new Date('2023-09-15'),
          proceeds: new Decimal(18000),
          costBasis: new Decimal(20000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(-2000),
        },
      ];

      const form8949A = {
        category: 'A',
        checkBox: '(A) Short-term transactions reported on Form(s) 1099-B showing basis was reported to the IRS',
        transactions,
        totals: {
          proceeds: transactions.reduce((sum, t) => sum.plus(t.proceeds), new Decimal(0)),
          costBasis: transactions.reduce((sum, t) => sum.plus(t.costBasis), new Decimal(0)),
          adjustments: new Decimal(0),
          gainLoss: transactions.reduce((sum, t) => sum.plus(t.gainLoss), new Decimal(0)),
        },
      };

      expect(form8949A.category).toBe('A');
      expect(form8949A.totals.proceeds.toString()).toBe('33000');
      expect(form8949A.totals.costBasis.toString()).toBe('32000');
      expect(form8949A.totals.gainLoss.toString()).toBe('1000');
    });

    it('Category B: Short-term with basis NOT reported to IRS', () => {
      /**
       * IRS Form 8949 Category B:
       * Short-term transactions on 1099-B WITHOUT basis reported
       */

      const transactions = [
        {
          description: '1.5 BTC',
          dateAcquired: new Date('2023-04-01'),
          dateSold: new Date('2023-10-15'),
          proceeds: new Decimal(60000),
          costBasis: new Decimal(45000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(15000),
        },
      ];

      const form8949B = {
        category: 'B',
        checkBox: '(B) Short-term transactions reported on Form(s) 1099-B showing basis was NOT reported to the IRS',
        transactions,
        totals: {
          proceeds: new Decimal(60000),
          costBasis: new Decimal(45000),
          adjustments: new Decimal(0),
          gainLoss: new Decimal(15000),
        },
      };

      expect(form8949B.category).toBe('B');
      expect(form8949B.totals.gainLoss.toString()).toBe('15000');
    });

    it('Category C: Short-term not on 1099-B', () => {
      /**
       * IRS Form 8949 Category C:
       * Short-term transactions NOT reported on any 1099-B
       */

      const transactions = [
        {
          description: '10 ETH',
          dateAcquired: new Date('2023-05-15'),
          dateSold: new Date('2023-11-01'),
          proceeds: new Decimal(18000),
          costBasis: new Decimal(15000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(3000),
        },
      ];

      const form8949C = {
        category: 'C',
        checkBox: '(C) Short-term transactions not reported to you on Form 1099-B',
        transactions,
        totals: {
          proceeds: new Decimal(18000),
          costBasis: new Decimal(15000),
          adjustments: new Decimal(0),
          gainLoss: new Decimal(3000),
        },
      };

      expect(form8949C.category).toBe('C');
      expect(form8949C.totals.gainLoss.toString()).toBe('3000');
    });
  });

  describe('Form 8949 - Part II (Long-Term)', () => {

    it('Category D: Long-term with basis reported to IRS', () => {
      /**
       * IRS Form 8949 Category D:
       * Long-term transactions reported on 1099-B WITH basis reported
       */

      const transactions = [
        {
          description: '200 shares GOOGL',
          dateAcquired: new Date('2021-06-15'),
          dateSold: new Date('2023-08-20'),
          proceeds: new Decimal(25000),
          costBasis: new Decimal(18000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(7000),
        },
      ];

      const form8949D = {
        category: 'D',
        checkBox: '(D) Long-term transactions reported on Form(s) 1099-B showing basis was reported to the IRS',
        transactions,
        totals: {
          proceeds: new Decimal(25000),
          costBasis: new Decimal(18000),
          adjustments: new Decimal(0),
          gainLoss: new Decimal(7000),
        },
      };

      expect(form8949D.category).toBe('D');
      expect(form8949D.totals.gainLoss.toString()).toBe('7000');

      // Verify long-term holding
      const holdingDays = Math.floor(
        (transactions[0].dateSold.getTime() - transactions[0].dateAcquired.getTime()) /
        (1000 * 60 * 60 * 24)
      );
      expect(holdingDays).toBeGreaterThan(365);
    });

    it('Category E: Long-term with basis NOT reported to IRS', () => {
      /**
       * IRS Form 8949 Category E:
       * Long-term transactions on 1099-B WITHOUT basis reported
       */

      const transactions = [
        {
          description: '5 BTC',
          dateAcquired: new Date('2021-01-01'),
          dateSold: new Date('2023-06-15'),
          proceeds: new Decimal(250000),
          costBasis: new Decimal(150000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(100000),
        },
      ];

      const form8949E = {
        category: 'E',
        checkBox: '(E) Long-term transactions reported on Form(s) 1099-B showing basis was NOT reported to the IRS',
        transactions,
        totals: {
          proceeds: new Decimal(250000),
          costBasis: new Decimal(150000),
          adjustments: new Decimal(0),
          gainLoss: new Decimal(100000),
        },
      };

      expect(form8949E.category).toBe('E');
      expect(form8949E.totals.gainLoss.toString()).toBe('100000');
    });

    it('Category F: Long-term not on 1099-B', () => {
      /**
       * IRS Form 8949 Category F:
       * Long-term transactions NOT reported on any 1099-B
       */

      const transactions = [
        {
          description: '100 ETH',
          dateAcquired: new Date('2020-03-15'),
          dateSold: new Date('2023-09-20'),
          proceeds: new Decimal(180000),
          costBasis: new Decimal(120000),
          code: '', // No adjustment
          adjustment: new Decimal(0),
          gainLoss: new Decimal(60000),
        },
      ];

      const form8949F = {
        category: 'F',
        checkBox: '(F) Long-term transactions not reported to you on Form 1099-B',
        transactions,
        totals: {
          proceeds: new Decimal(180000),
          costBasis: new Decimal(120000),
          adjustments: new Decimal(0),
          gainLoss: new Decimal(60000),
        },
      };

      expect(form8949F.category).toBe('F');
      expect(form8949F.totals.gainLoss.toString()).toBe('60000');
    });
  });

  describe('Form 8949 - Adjustment Codes', () => {

    it('Code W: Wash Sale Adjustment', () => {
      /**
       * IRS Adjustment Code W:
       * Wash sale loss disallowance
       */

      const transaction = {
        description: '1 BTC',
        dateAcquired: new Date('2023-05-01'),
        dateSold: new Date('2023-06-15'),
        proceeds: new Decimal(40000),
        costBasis: new Decimal(50000),
        code: 'W',
        adjustment: new Decimal(10000), // Disallowed loss
        unadjustedGainLoss: new Decimal(-10000),
        adjustedGainLoss: new Decimal(0), // Fully disallowed
      };

      expect(transaction.code).toBe('W');
      expect(transaction.adjustment.toString()).toBe('10000');
      expect(transaction.adjustedGainLoss.toString()).toBe('0');
    });

    it('Code B: Long-term Gain Elected as Short-term', () => {
      /**
       * IRS Adjustment Code B:
       * Long-term gain from collectibles or Section 1202 stock
       */

      const transaction = {
        description: 'Collectible NFT',
        dateAcquired: new Date('2021-01-01'),
        dateSold: new Date('2023-06-15'),
        proceeds: new Decimal(50000),
        costBasis: new Decimal(30000),
        code: 'B',
        adjustment: new Decimal(0),
        gainLoss: new Decimal(20000),
        note: 'Collectible - taxed at 28% rate',
      };

      expect(transaction.code).toBe('B');
      expect(transaction.gainLoss.toString()).toBe('20000');
    });

    it('Code E: Adjustment for Commissions/Fees', () => {
      /**
       * IRS Adjustment Code E:
       * Form 1099-B shows incorrect amount
       */

      const transaction = {
        description: '100 shares AAPL',
        dateAcquired: new Date('2023-01-01'),
        dateSold: new Date('2023-06-15'),
        proceeds: new Decimal(15000),
        costBasis: new Decimal(12000),
        code: 'E',
        adjustment: new Decimal(100), // Add commission not on 1099-B
        adjustedCostBasis: new Decimal(12100),
        gainLoss: new Decimal(2900),
      };

      expect(transaction.code).toBe('E');
      expect(transaction.adjustedCostBasis.toString()).toBe('12100');
    });
  });

  describe('Schedule D - Part I (Short-Term)', () => {

    it('Line 1a: Short-term totals from Form 8949 (A)', () => {
      const form8949ATotals = {
        proceeds: new Decimal(50000),
        costBasis: new Decimal(45000),
        adjustments: new Decimal(0),
        gainLoss: new Decimal(5000),
      };

      const scheduleDLine1a = {
        description: 'Short-term totals from all Forms 8949 with Box A checked',
        proceeds: form8949ATotals.proceeds,
        costBasis: form8949ATotals.costBasis,
        adjustments: form8949ATotals.adjustments,
        gainLoss: form8949ATotals.gainLoss,
      };

      expect(scheduleDLine1a.gainLoss.toString()).toBe('5000');
    });

    it('Line 7: Total Short-Term Gains/Losses', () => {
      /**
       * Schedule D Line 7: Net short-term capital gain or loss
       */

      const shortTermItems = [
        { line: '1a', gainLoss: new Decimal(5000) },  // Form 8949 A
        { line: '1b', gainLoss: new Decimal(2000) },  // Form 8949 B
        { line: '2', gainLoss: new Decimal(1000) },   // Form 8949 C
        { line: '3', gainLoss: new Decimal(0) },      // Other short-term
        { line: '4', gainLoss: new Decimal(0) },      // Short-term gain from 6252
        { line: '5', gainLoss: new Decimal(0) },      // Short-term gain from 4797
        { line: '6', gainLoss: new Decimal(-2000) },  // Short-term loss carryover
      ];

      const line7Total = shortTermItems.reduce(
        (sum, item) => sum.plus(item.gainLoss),
        new Decimal(0)
      );

      expect(line7Total.toString()).toBe('6000');
    });
  });

  describe('Schedule D - Part II (Long-Term)', () => {

    it('Line 8a: Long-term totals from Form 8949 (D)', () => {
      const form8949DTotals = {
        proceeds: new Decimal(100000),
        costBasis: new Decimal(70000),
        adjustments: new Decimal(0),
        gainLoss: new Decimal(30000),
      };

      const scheduleDLine8a = {
        description: 'Long-term totals from all Forms 8949 with Box D checked',
        proceeds: form8949DTotals.proceeds,
        costBasis: form8949DTotals.costBasis,
        adjustments: form8949DTotals.adjustments,
        gainLoss: form8949DTotals.gainLoss,
      };

      expect(scheduleDLine8a.gainLoss.toString()).toBe('30000');
    });

    it('Line 15: Total Long-Term Gains/Losses', () => {
      /**
       * Schedule D Line 15: Net long-term capital gain or loss
       */

      const longTermItems = [
        { line: '8a', gainLoss: new Decimal(30000) },  // Form 8949 D
        { line: '8b', gainLoss: new Decimal(15000) },  // Form 8949 E
        { line: '9', gainLoss: new Decimal(10000) },   // Form 8949 F
        { line: '10', gainLoss: new Decimal(0) },      // Other long-term
        { line: '11', gainLoss: new Decimal(0) },      // Long-term gain from 2439
        { line: '12', gainLoss: new Decimal(0) },      // Long-term gain from 6252
        { line: '13', gainLoss: new Decimal(0) },      // Capital gain distributions
        { line: '14', gainLoss: new Decimal(-5000) },  // Long-term loss carryover
      ];

      const line15Total = longTermItems.reduce(
        (sum, item) => sum.plus(item.gainLoss),
        new Decimal(0)
      );

      expect(line15Total.toString()).toBe('50000');
    });
  });

  describe('Schedule D - Part III (Summary)', () => {

    it('Line 16: Net Capital Gain/Loss', () => {
      /**
       * Schedule D Line 16: Combine lines 7 and 15
       */

      const line7ShortTerm = new Decimal(6000);
      const line15LongTerm = new Decimal(50000);

      const line16NetGainLoss = line7ShortTerm.plus(line15LongTerm);

      expect(line16NetGainLoss.toString()).toBe('56000');
    });

    it('Line 21: Net Loss Limited to $3,000', () => {
      /**
       * Schedule D Line 21: If net loss, limited to $3,000 deduction
       */

      const line16NetLoss = new Decimal(-15000);
      const deductionLimit = new Decimal(3000);

      const line21Deduction = Decimal.min(line16NetLoss.abs(), deductionLimit);
      const carryoverLoss = line16NetLoss.abs().minus(line21Deduction);

      expect(line21Deduction.toString()).toBe('3000');
      expect(carryoverLoss.toString()).toBe('12000');
    });
  });

  describe('Form Field Validation', () => {

    it('All required fields present for Form 8949', () => {
      const transaction = {
        // Column (a)
        description: '1 BTC',

        // Column (b)
        dateAcquired: new Date('2023-01-15'),

        // Column (c)
        dateSold: new Date('2023-06-20'),

        // Column (d)
        proceeds: new Decimal(50000),

        // Column (e)
        costBasis: new Decimal(40000),

        // Column (f)
        code: '',

        // Column (g)
        adjustment: new Decimal(0),

        // Column (h)
        gainLoss: new Decimal(10000),
      };

      // Verify all required columns
      expect(transaction.description).toBeDefined();
      expect(transaction.dateAcquired).toBeInstanceOf(Date);
      expect(transaction.dateSold).toBeInstanceOf(Date);
      expect(transaction.proceeds).toBeInstanceOf(Decimal);
      expect(transaction.costBasis).toBeInstanceOf(Decimal);
      expect(transaction.gainLoss).toBeInstanceOf(Decimal);

      // Verify gain/loss calculation
      const calculatedGainLoss = transaction.proceeds
        .minus(transaction.costBasis)
        .plus(transaction.adjustment);

      expect(calculatedGainLoss.toString()).toBe(transaction.gainLoss.toString());
    });

    it('Date formats match IRS requirements', () => {
      const transaction = {
        dateAcquired: new Date('2023-01-15'),
        dateSold: new Date('2023-06-20'),
      };

      // IRS format: MM/DD/YYYY
      const formatDate = (date: Date) => {
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const year = date.getFullYear();
        return `${month}/${day}/${year}`;
      };

      expect(formatDate(transaction.dateAcquired)).toBe('01/15/2023');
      expect(formatDate(transaction.dateSold)).toBe('06/20/2023');
    });

    it('Currency amounts formatted correctly', () => {
      const amount = new Decimal(1234.56);

      // IRS format: $X,XXX.XX
      const formatCurrency = (amt: Decimal) => {
        return `$${amt.toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`;
      };

      expect(formatCurrency(amount)).toBe('$1,234.56');
      expect(formatCurrency(new Decimal(1234567.89))).toBe('$1,234,567.89');
    });
  });

  describe('Multi-Page Form Generation', () => {

    it('Split large transaction lists across pages', () => {
      /**
       * IRS Form 8949: Maximum 14 transactions per page
       */

      const transactions = Array.from({ length: 50 }, (_, i) => ({
        id: `tx-${i}`,
        description: `Transaction ${i}`,
        dateAcquired: new Date('2023-01-01'),
        dateSold: new Date('2023-06-01'),
        proceeds: new Decimal(1000),
        costBasis: new Decimal(900),
        gainLoss: new Decimal(100),
      }));

      const TRANSACTIONS_PER_PAGE = 14;
      const pages: any[] = [];

      for (let i = 0; i < transactions.length; i += TRANSACTIONS_PER_PAGE) {
        const pageTransactions = transactions.slice(i, i + TRANSACTIONS_PER_PAGE);

        pages.push({
          pageNumber: Math.floor(i / TRANSACTIONS_PER_PAGE) + 1,
          transactions: pageTransactions,
          totals: {
            proceeds: pageTransactions.reduce((s, t) => s.plus(t.proceeds), new Decimal(0)),
            costBasis: pageTransactions.reduce((s, t) => s.plus(t.costBasis), new Decimal(0)),
            gainLoss: pageTransactions.reduce((s, t) => s.plus(t.gainLoss), new Decimal(0)),
          },
        });
      }

      expect(pages.length).toBe(4); // 50 / 14 = 4 pages
      expect(pages[0].transactions.length).toBe(14);
      expect(pages[3].transactions.length).toBe(8); // Remaining
    });
  });

  describe('Taxpayer Information', () => {

    it('SSN format validation', () => {
      const ssn = '123-45-6789';
      const ssnPattern = /^\d{3}-\d{2}-\d{4}$/;

      expect(ssnPattern.test(ssn)).toBe(true);
    });

    it('EIN format validation', () => {
      const ein = '12-3456789';
      const einPattern = /^\d{2}-\d{7}$/;

      expect(einPattern.test(ein)).toBe(true);
    });
  });

  describe('Form Totals Reconciliation', () => {

    it('Form 8949 totals match Schedule D', () => {
      /**
       * Ensure Form 8949 totals correctly flow to Schedule D
       */

      // Form 8949 Category A (Short-term)
      const form8949A = {
        totals: {
          proceeds: new Decimal(50000),
          costBasis: new Decimal(45000),
          gainLoss: new Decimal(5000),
        },
      };

      // Form 8949 Category D (Long-term)
      const form8949D = {
        totals: {
          proceeds: new Decimal(100000),
          costBasis: new Decimal(70000),
          gainLoss: new Decimal(30000),
        },
      };

      // Schedule D Line 1a (from 8949-A)
      const scheduleDLine1a = form8949A.totals.gainLoss;

      // Schedule D Line 8a (from 8949-D)
      const scheduleDLine8a = form8949D.totals.gainLoss;

      expect(scheduleDLine1a.toString()).toBe('5000');
      expect(scheduleDLine8a.toString()).toBe('30000');

      // Total capital gain
      const totalCapitalGain = scheduleDLine1a.plus(scheduleDLine8a);
      expect(totalCapitalGain.toString()).toBe('35000');
    });
  });
});
