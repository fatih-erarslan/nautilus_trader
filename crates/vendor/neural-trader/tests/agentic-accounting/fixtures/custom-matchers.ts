/**
 * Custom Jest/Vitest Matchers for Tax Calculations
 *
 * Provides decimal-aware matchers for precise financial calculations.
 */

import { expect } from 'vitest';
import Decimal from 'decimal.js';

declare global {
  namespace Vi {
    interface Matchers<R = any> {
      toBeDecimal(expected: string | number | Decimal): R;
      toBeDecimalCloseTo(expected: string | number | Decimal, precision?: number): R;
      toBePositiveDecimal(): R;
      toBeNegativeDecimal(): R;
      toBeLongTerm(acquisitionDate: Date, disposalDate: Date): R;
      toBeShortTerm(acquisitionDate: Date, disposalDate: Date): R;
    }
  }
}

/**
 * Exact decimal equality matcher
 */
expect.extend({
  toBeDecimal(received: any, expected: string | number | Decimal) {
    const receivedDecimal = new Decimal(received);
    const expectedDecimal = new Decimal(expected);

    const pass = receivedDecimal.equals(expectedDecimal);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to equal ${expected}`
          : `Expected ${received} to equal ${expected}\nDifference: ${receivedDecimal.minus(expectedDecimal).toString()}`,
    };
  },

  /**
   * Decimal equality within precision
   * @param precision Number of decimal places (default: 2 for currency)
   */
  toBeDecimalCloseTo(
    received: any,
    expected: string | number | Decimal,
    precision: number = 2
  ) {
    const receivedDecimal = new Decimal(received);
    const expectedDecimal = new Decimal(expected);

    const diff = receivedDecimal.minus(expectedDecimal).abs();
    const epsilon = new Decimal(10).pow(-precision);

    const pass = diff.lessThanOrEqualTo(epsilon);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be close to ${expected} (precision: ${precision})`
          : `Expected ${received} to be close to ${expected} (precision: ${precision})\nDifference: ${diff.toString()}`,
    };
  },

  toBePositiveDecimal(received: any) {
    const receivedDecimal = new Decimal(received);
    const pass = receivedDecimal.greaterThan(0);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be positive`
          : `Expected ${received} to be positive`,
    };
  },

  toBeNegativeDecimal(received: any) {
    const receivedDecimal = new Decimal(received);
    const pass = receivedDecimal.lessThan(0);

    return {
      pass,
      message: () =>
        pass
          ? `Expected ${received} not to be negative`
          : `Expected ${received} to be negative`,
    };
  },

  toBeLongTerm(received: any, acquisitionDate: Date, disposalDate: Date) {
    const holdingDays = Math.floor(
      (disposalDate.getTime() - acquisitionDate.getTime()) / (1000 * 60 * 60 * 24)
    );

    const pass = holdingDays > 365 && received === 'LONG';

    return {
      pass,
      message: () =>
        pass
          ? `Expected not to be long-term (${holdingDays} days)`
          : `Expected to be long-term (${holdingDays} days, need >365)`,
    };
  },

  toBeShortTerm(received: any, acquisitionDate: Date, disposalDate: Date) {
    const holdingDays = Math.floor(
      (disposalDate.getTime() - acquisitionDate.getTime()) / (1000 * 60 * 60 * 24)
    );

    const pass = holdingDays <= 365 && received === 'SHORT';

    return {
      pass,
      message: () =>
        pass
          ? `Expected not to be short-term (${holdingDays} days)`
          : `Expected to be short-term (${holdingDays} days, must be ≤365)`,
    };
  },
});

/**
 * Helper functions for test data validation
 */

export function assertValidDisposal(disposal: any) {
  expect(disposal).toHaveProperty('id');
  expect(disposal).toHaveProperty('asset');
  expect(disposal).toHaveProperty('quantity');
  expect(disposal).toHaveProperty('proceeds');
  expect(disposal).toHaveProperty('cost_basis');
  expect(disposal).toHaveProperty('gain_loss');
  expect(disposal).toHaveProperty('acquisition_date');
  expect(disposal).toHaveProperty('disposal_date');
  expect(disposal).toHaveProperty('is_long_term');

  // Verify gain calculation
  const proceeds = new Decimal(disposal.proceeds);
  const costBasis = new Decimal(disposal.cost_basis);
  const expectedGain = proceeds.minus(costBasis);

  expect(disposal.gain_loss).toBeDecimalCloseTo(expectedGain, 2);
}

export function assertValidTaxLot(lot: any) {
  expect(lot).toHaveProperty('id');
  expect(lot).toHaveProperty('asset');
  expect(lot).toHaveProperty('quantity');
  expect(lot).toHaveProperty('cost_basis');
  expect(lot).toHaveProperty('acquisition_date');

  // Quantities and costs should be positive
  expect(lot.quantity).toBePositiveDecimal();
  expect(lot.cost_basis).toBePositiveDecimal();
}

export function assertValidTransaction(transaction: any) {
  expect(transaction).toHaveProperty('id');
  expect(transaction).toHaveProperty('type');
  expect(transaction).toHaveProperty('asset');
  expect(transaction).toHaveProperty('quantity');
  expect(transaction).toHaveProperty('price');
  expect(transaction).toHaveProperty('timestamp');

  // Type should be valid
  expect(['BUY', 'SELL', 'TRADE', 'INCOME', 'TRANSFER']).toContain(transaction.type);

  // Quantities and prices should be positive (or zero for some types)
  const quantity = new Decimal(transaction.quantity);
  const price = new Decimal(transaction.price);

  expect(quantity.greaterThanOrEqualTo(0)).toBe(true);
  expect(price.greaterThanOrEqualTo(0)).toBe(true);
}

export function calculateExpectedGain(
  quantity: string | Decimal,
  salePrice: string | Decimal,
  costBasis: string | Decimal
): Decimal {
  const qty = new Decimal(quantity);
  const price = new Decimal(salePrice);
  const cost = new Decimal(costBasis);

  const proceeds = qty.times(price);
  return proceeds.minus(cost);
}

export function isLongTerm(acquisitionDate: Date, disposalDate: Date): boolean {
  const holdingDays = Math.floor(
    (disposalDate.getTime() - acquisitionDate.getTime()) / (1000 * 60 * 60 * 24)
  );
  return holdingDays > 365;
}

export function isWithinWashSaleWindow(
  disposalDate: Date,
  replacementDate: Date
): boolean {
  const daysDiff = Math.abs(
    Math.floor((replacementDate.getTime() - disposalDate.getTime()) / (1000 * 60 * 60 * 24))
  );
  return daysDiff <= 30;
}
