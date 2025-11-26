import { describe, it, expect } from 'vitest';
import { DecimalMath, Decimal } from '../../src/utils/decimal.js';

describe('DecimalMath Utilities', () => {
  describe('from()', () => {
    it('should create Decimal from number', () => {
      const result = DecimalMath.from(42.5);
      expect(result.toString()).toBe('42.5');
    });

    it('should create Decimal from string', () => {
      const result = DecimalMath.from('123.456');
      expect(result.toString()).toBe('123.456');
    });

    it('should create Decimal from Decimal', () => {
      const original = new Decimal('99.99');
      const result = DecimalMath.from(original);
      expect(result.toString()).toBe('99.99');
    });
  });

  describe('Basic arithmetic', () => {
    it('should add two decimals', () => {
      const a = new Decimal('100.50');
      const b = new Decimal('50.25');
      const result = DecimalMath.add(a, b);
      expect(result.toString()).toBe('150.75');
    });

    it('should subtract two decimals', () => {
      const a = new Decimal('100');
      const b = new Decimal('30.5');
      const result = DecimalMath.subtract(a, b);
      expect(result.toString()).toBe('69.5');
    });

    it('should multiply two decimals', () => {
      const a = new Decimal('12.5');
      const b = new Decimal('4');
      const result = DecimalMath.multiply(a, b);
      expect(result.toString()).toBe('50');
    });

    it('should divide two decimals', () => {
      const a = new Decimal('100');
      const b = new Decimal('4');
      const result = DecimalMath.divide(a, b);
      expect(result.toString()).toBe('25');
    });

    it('should throw error on division by zero', () => {
      const a = new Decimal('100');
      const b = new Decimal('0');
      expect(() => DecimalMath.divide(a, b)).toThrow('Division by zero');
    });
  });

  describe('percentage()', () => {
    it('should calculate percentage', () => {
      const value = new Decimal('25');
      const total = new Decimal('200');
      const result = DecimalMath.percentage(value, total);
      expect(result.toString()).toBe('12.5');
    });

    it('should return 0 for percentage of zero total', () => {
      const value = new Decimal('50');
      const total = new Decimal('0');
      const result = DecimalMath.percentage(value, total);
      expect(result.toString()).toBe('0');
    });
  });

  describe('round()', () => {
    it('should round to 2 decimal places by default', () => {
      const value = new Decimal('12.3456');
      const result = DecimalMath.round(value);
      expect(result.toString()).toBe('12.35');
    });

    it('should round to specified decimal places', () => {
      const value = new Decimal('12.3456');
      const result = DecimalMath.round(value, 3);
      expect(result.toString()).toBe('12.346');
    });
  });

  describe('Comparison operations', () => {
    it('should check if value is zero', () => {
      expect(DecimalMath.isZero(new Decimal('0'))).toBe(true);
      expect(DecimalMath.isZero(new Decimal('0.0'))).toBe(true);
      expect(DecimalMath.isZero(new Decimal('0.1'))).toBe(false);
    });

    it('should check if value is positive', () => {
      expect(DecimalMath.isPositive(new Decimal('1'))).toBe(true);
      expect(DecimalMath.isPositive(new Decimal('0.001'))).toBe(true);
      expect(DecimalMath.isPositive(new Decimal('0'))).toBe(false);
      expect(DecimalMath.isPositive(new Decimal('-1'))).toBe(false);
    });

    it('should check if value is negative', () => {
      expect(DecimalMath.isNegative(new Decimal('-1'))).toBe(true);
      expect(DecimalMath.isNegative(new Decimal('-0.001'))).toBe(true);
      expect(DecimalMath.isNegative(new Decimal('0'))).toBe(false);
      expect(DecimalMath.isNegative(new Decimal('1'))).toBe(false);
    });

    it('should get absolute value', () => {
      expect(DecimalMath.abs(new Decimal('-10')).toString()).toBe('10');
      expect(DecimalMath.abs(new Decimal('10')).toString()).toBe('10');
      expect(DecimalMath.abs(new Decimal('0')).toString()).toBe('0');
    });

    it('should find minimum of two values', () => {
      const a = new Decimal('10');
      const b = new Decimal('20');
      expect(DecimalMath.min(a, b).toString()).toBe('10');
      expect(DecimalMath.min(b, a).toString()).toBe('10');
    });

    it('should find maximum of two values', () => {
      const a = new Decimal('10');
      const b = new Decimal('20');
      expect(DecimalMath.max(a, b).toString()).toBe('20');
      expect(DecimalMath.max(b, a).toString()).toBe('20');
    });

    it('should check equality', () => {
      const a = new Decimal('10.5');
      const b = new Decimal('10.5');
      const c = new Decimal('10.6');
      expect(DecimalMath.equals(a, b)).toBe(true);
      expect(DecimalMath.equals(a, c)).toBe(false);
    });

    it('should compare two decimals', () => {
      const a = new Decimal('10');
      const b = new Decimal('20');
      const c = new Decimal('10');
      expect(DecimalMath.compare(a, b)).toBe(-1);
      expect(DecimalMath.compare(b, a)).toBe(1);
      expect(DecimalMath.compare(a, c)).toBe(0);
    });
  });

  describe('Aggregate operations', () => {
    it('should sum array of decimals', () => {
      const values = [
        new Decimal('10'),
        new Decimal('20'),
        new Decimal('30'),
      ];
      const result = DecimalMath.sum(values);
      expect(result.toString()).toBe('60');
    });

    it('should return 0 for empty sum', () => {
      const result = DecimalMath.sum([]);
      expect(result.toString()).toBe('0');
    });

    it('should calculate average', () => {
      const values = [
        new Decimal('10'),
        new Decimal('20'),
        new Decimal('30'),
      ];
      const result = DecimalMath.average(values);
      expect(result.toString()).toBe('20');
    });

    it('should return 0 for empty average', () => {
      const result = DecimalMath.average([]);
      expect(result.toString()).toBe('0');
    });

    it('should calculate weighted average', () => {
      const values: Array<[Decimal, Decimal]> = [
        [new Decimal('10'), new Decimal('2')],  // value: 10, weight: 2
        [new Decimal('20'), new Decimal('3')],  // value: 20, weight: 3
        [new Decimal('30'), new Decimal('5')],  // value: 30, weight: 5
      ];
      // (10*2 + 20*3 + 30*5) / (2+3+5) = (20 + 60 + 150) / 10 = 230 / 10 = 23
      const result = DecimalMath.weightedAverage(values);
      expect(result.toString()).toBe('23');
    });

    it('should return 0 for empty weighted average', () => {
      const result = DecimalMath.weightedAverage([]);
      expect(result.toString()).toBe('0');
    });

    it('should return 0 for weighted average with zero weights', () => {
      const values: Array<[Decimal, Decimal]> = [
        [new Decimal('10'), new Decimal('0')],
        [new Decimal('20'), new Decimal('0')],
      ];
      const result = DecimalMath.weightedAverage(values);
      expect(result.toString()).toBe('0');
    });
  });

  describe('Conversion operations', () => {
    it('should convert to fixed-point string', () => {
      const value = new Decimal('123.456789');
      expect(DecimalMath.toFixed(value, 2)).toBe('123.46');
      expect(DecimalMath.toFixed(value, 4)).toBe('123.4568');
    });

    it('should convert to number', () => {
      const value = new Decimal('42.5');
      const result = DecimalMath.toNumber(value);
      expect(result).toBe(42.5);
      expect(typeof result).toBe('number');
    });
  });

  describe('Precision handling', () => {
    it('should maintain precision for large numbers', () => {
      const a = new Decimal('999999999999999999.999999999999999999');
      const b = new Decimal('0.000000000000000001');
      const result = DecimalMath.add(a, b);
      expect(result.toString()).toBe('1000000000000000000');
    });

    it('should maintain precision for small numbers', () => {
      const a = new Decimal('0.000000000000000001');
      const b = new Decimal('0.000000000000000002');
      const result = DecimalMath.add(a, b);
      // Decimal.js returns full precision, not scientific notation for this range
      expect(result.toString()).toBe('0.000000000000000003');
    });

    it('should handle financial calculations accurately', () => {
      // Test case: 0.1 + 0.2 (notoriously inaccurate with floats)
      const a = new Decimal('0.1');
      const b = new Decimal('0.2');
      const result = DecimalMath.add(a, b);
      expect(result.toString()).toBe('0.3');

      // Compare with native float (would be 0.30000000000000004)
      expect(0.1 + 0.2).not.toBe(0.3);
    });
  });
});
