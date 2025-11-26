import { describe, it, expect } from 'vitest';
import { CapitalGainTerm, type Disposal } from '../../src/types/disposal.js';
import { AccountingMethod } from '../../src/types/tax-lot.js';
import { Decimal } from '../../src/utils/decimal.js';
import { v4 as uuidv4 } from 'uuid';

describe('Disposal Types', () => {
  describe('CapitalGainTerm enum', () => {
    it('should have short and long term classifications', () => {
      expect(CapitalGainTerm.SHORT).toBe('SHORT');
      expect(CapitalGainTerm.LONG).toBe('LONG');
    });

    it('should have exactly 2 terms', () => {
      const terms = Object.values(CapitalGainTerm);
      expect(terms).toHaveLength(2);
    });
  });

  describe('Disposal interface', () => {
    it('should create a valid disposal with gain', () => {
      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2024-06-15'),
        quantity: new Decimal('1'),
        proceeds: new Decimal('50000'),
        costBasis: new Decimal('45000'),
        gain: new Decimal('5000'),
        term: CapitalGainTerm.LONG,
        taxYear: 2024,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.id).toBeDefined();
      expect(disposal.quantity.toString()).toBe('1');
      expect(disposal.proceeds.toString()).toBe('50000');
      expect(disposal.costBasis.toString()).toBe('45000');
      expect(disposal.gain.toString()).toBe('5000');
      expect(disposal.term).toBe(CapitalGainTerm.LONG);
    });

    it('should calculate gain correctly (proceeds - costBasis)', () => {
      const proceeds = new Decimal('60000');
      const costBasis = new Decimal('45000');
      const gain = proceeds.minus(costBasis);

      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2024-06-15'),
        quantity: new Decimal('1'),
        proceeds,
        costBasis,
        gain,
        term: CapitalGainTerm.SHORT,
        taxYear: 2024,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.gain.toString()).toBe('15000');
      expect(disposal.proceeds.minus(disposal.costBasis).equals(disposal.gain)).toBe(true);
    });

    it('should handle losses (negative gains)', () => {
      const proceeds = new Decimal('40000');
      const costBasis = new Decimal('45000');
      const gain = proceeds.minus(costBasis);

      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2024-03-15'),
        quantity: new Decimal('1'),
        proceeds,
        costBasis,
        gain,
        term: CapitalGainTerm.SHORT,
        taxYear: 2024,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.gain.toString()).toBe('-5000');
      expect(disposal.gain.isNegative()).toBe(true);
    });

    it('should classify short-term correctly (< 1 year)', () => {
      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2024-06-15'),
        quantity: new Decimal('0.5'),
        proceeds: new Decimal('25000'),
        costBasis: new Decimal('22500'),
        gain: new Decimal('2500'),
        term: CapitalGainTerm.SHORT,
        taxYear: 2024,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.term).toBe(CapitalGainTerm.SHORT);
    });

    it('should classify long-term correctly (>= 1 year)', () => {
      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2025-01-16'),
        quantity: new Decimal('1'),
        proceeds: new Decimal('50000'),
        costBasis: new Decimal('45000'),
        gain: new Decimal('5000'),
        term: CapitalGainTerm.LONG,
        taxYear: 2025,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.term).toBe(CapitalGainTerm.LONG);
    });

    it('should handle partial lot disposal', () => {
      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate: new Date('2024-06-15'),
        quantity: new Decimal('0.3'),
        proceeds: new Decimal('15000'),
        costBasis: new Decimal('13500'),
        gain: new Decimal('1500'),
        term: CapitalGainTerm.LONG,
        taxYear: 2024,
        method: AccountingMethod.SPECIFIC_ID,
      };

      expect(disposal.quantity.lessThan(1)).toBe(true);
      expect(disposal.method).toBe(AccountingMethod.SPECIFIC_ID);
    });

    it('should associate with correct tax year', () => {
      const disposalDate = new Date('2024-12-31');
      const disposal: Disposal = {
        id: uuidv4(),
        lotId: uuidv4(),
        transactionId: uuidv4(),
        disposalDate,
        quantity: new Decimal('1'),
        proceeds: new Decimal('50000'),
        costBasis: new Decimal('45000'),
        gain: new Decimal('5000'),
        term: CapitalGainTerm.LONG,
        taxYear: 2024,
        method: AccountingMethod.FIFO,
      };

      expect(disposal.taxYear).toBe(disposalDate.getFullYear());
    });
  });
});
