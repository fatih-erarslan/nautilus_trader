/**
 * Input Validation
 *
 * Validates transaction data and tax lots before calculation
 * Ensures data integrity and prevents calculation errors
 */

import { Transaction, TaxLot } from './calculator-wrapper';

export class ValidationError extends Error {
  constructor(message: string, public field?: string, public code?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class TaxInputValidator {
  /**
   * Validate a transaction
   */
  validateTransaction(tx: Transaction): void {
    // Required fields
    if (!tx.id || tx.id.trim() === '') {
      throw new ValidationError('Transaction ID is required', 'id', 'REQUIRED');
    }

    if (!tx.asset || tx.asset.trim() === '') {
      throw new ValidationError('Asset is required', 'asset', 'REQUIRED');
    }

    // Transaction type
    if (!['BUY', 'SELL'].includes(tx.transactionType)) {
      throw new ValidationError(
        'Transaction type must be BUY or SELL',
        'transactionType',
        'INVALID_TYPE'
      );
    }

    // Numeric fields
    this.validateDecimal(tx.quantity, 'quantity');
    this.validateDecimal(tx.price, 'price');
    this.validateDecimal(tx.fees, 'fees');

    // Positive values
    if (parseFloat(tx.quantity) <= 0) {
      throw new ValidationError('Quantity must be positive', 'quantity', 'INVALID_VALUE');
    }

    if (parseFloat(tx.price) < 0) {
      throw new ValidationError('Price cannot be negative', 'price', 'INVALID_VALUE');
    }

    if (parseFloat(tx.fees) < 0) {
      throw new ValidationError('Fees cannot be negative', 'fees', 'INVALID_VALUE');
    }

    // Date validation
    this.validateDate(tx.timestamp, 'timestamp');

    // Source
    if (!tx.source || tx.source.trim() === '') {
      throw new ValidationError('Source is required', 'source', 'REQUIRED');
    }
  }

  /**
   * Validate a tax lot
   */
  validateTaxLot(lot: TaxLot): void {
    // Required fields
    if (!lot.id || lot.id.trim() === '') {
      throw new ValidationError('Tax lot ID is required', 'id', 'REQUIRED');
    }

    if (!lot.transactionId || lot.transactionId.trim() === '') {
      throw new ValidationError('Transaction ID is required', 'transactionId', 'REQUIRED');
    }

    if (!lot.asset || lot.asset.trim() === '') {
      throw new ValidationError('Asset is required', 'asset', 'REQUIRED');
    }

    // Numeric fields
    this.validateDecimal(lot.quantity, 'quantity');
    this.validateDecimal(lot.remainingQuantity, 'remainingQuantity');
    this.validateDecimal(lot.costBasis, 'costBasis');

    // Positive values
    if (parseFloat(lot.quantity) <= 0) {
      throw new ValidationError('Quantity must be positive', 'quantity', 'INVALID_VALUE');
    }

    if (parseFloat(lot.remainingQuantity) < 0) {
      throw new ValidationError(
        'Remaining quantity cannot be negative',
        'remainingQuantity',
        'INVALID_VALUE'
      );
    }

    if (parseFloat(lot.remainingQuantity) > parseFloat(lot.quantity)) {
      throw new ValidationError(
        'Remaining quantity cannot exceed total quantity',
        'remainingQuantity',
        'INVALID_VALUE'
      );
    }

    if (parseFloat(lot.costBasis) < 0) {
      throw new ValidationError('Cost basis cannot be negative', 'costBasis', 'INVALID_VALUE');
    }

    // Date validation
    this.validateDate(lot.acquisitionDate, 'acquisitionDate');
  }

  /**
   * Validate sale transaction and lots are compatible
   */
  validateSaleAndLots(sale: Transaction, lots: TaxLot[]): void {
    // Must be a SELL transaction
    if (sale.transactionType !== 'SELL') {
      throw new ValidationError(
        'Transaction must be a SELL type',
        'transactionType',
        'INVALID_TYPE'
      );
    }

    // Must have lots
    if (lots.length === 0) {
      throw new ValidationError('At least one tax lot is required', 'lots', 'REQUIRED');
    }

    // All lots must match asset
    const mismatchedLots = lots.filter(lot => lot.asset !== sale.asset);
    if (mismatchedLots.length > 0) {
      throw new ValidationError(
        `All lots must be for asset ${sale.asset}`,
        'asset',
        'ASSET_MISMATCH'
      );
    }

    // Check sufficient quantity
    let totalAvailable = '0';
    for (const lot of lots) {
      // Add using string addition to avoid floating point issues
      const [intPart1, decPart1 = '0'] = totalAvailable.split('.');
      const [intPart2, decPart2 = '0'] = lot.remainingQuantity.split('.');

      const maxDecLen = Math.max(decPart1.length, decPart2.length);
      const dec1 = decPart1.padEnd(maxDecLen, '0');
      const dec2 = decPart2.padEnd(maxDecLen, '0');

      const decSum = parseInt(dec1) + parseInt(dec2);
      const decCarry = Math.floor(decSum / Math.pow(10, maxDecLen));
      const decResult = (decSum % Math.pow(10, maxDecLen)).toString().padStart(maxDecLen, '0');

      const intSum = parseInt(intPart1) + parseInt(intPart2) + decCarry;
      totalAvailable = `${intSum}.${decResult}`;
    }

    if (parseFloat(totalAvailable) < parseFloat(sale.quantity)) {
      throw new ValidationError(
        `Insufficient quantity: need ${sale.quantity}, have ${totalAvailable}`,
        'quantity',
        'INSUFFICIENT_QUANTITY'
      );
    }

    // All lots must be acquired before sale
    for (const lot of lots) {
      const lotDate = new Date(lot.acquisitionDate);
      const saleDate = new Date(sale.timestamp);

      if (lotDate > saleDate) {
        throw new ValidationError(
          `Lot ${lot.id} acquired after sale date`,
          'acquisitionDate',
          'INVALID_DATE'
        );
      }
    }
  }

  /**
   * Validate decimal string
   */
  private validateDecimal(value: string, field: string): void {
    if (typeof value !== 'string') {
      throw new ValidationError(
        `${field} must be a string`,
        field,
        'INVALID_TYPE'
      );
    }

    const decimalRegex = /^-?\d+(\.\d+)?$/;
    if (!decimalRegex.test(value)) {
      throw new ValidationError(
        `${field} must be a valid decimal string`,
        field,
        'INVALID_FORMAT'
      );
    }
  }

  /**
   * Validate ISO 8601 date string
   */
  private validateDate(value: string, field: string): void {
    if (typeof value !== 'string') {
      throw new ValidationError(
        `${field} must be a string`,
        field,
        'INVALID_TYPE'
      );
    }

    const date = new Date(value);
    if (isNaN(date.getTime())) {
      throw new ValidationError(
        `${field} must be a valid ISO 8601 date`,
        field,
        'INVALID_DATE'
      );
    }

    // Check if date is reasonable (not before 2009 for crypto, not in future)
    const minDate = new Date('2009-01-01');
    const maxDate = new Date(Date.now() + 86400000); // Allow 1 day in future

    if (date < minDate) {
      throw new ValidationError(
        `${field} is before minimum date (2009-01-01)`,
        field,
        'DATE_TOO_OLD'
      );
    }

    if (date > maxDate) {
      throw new ValidationError(
        `${field} is in the future`,
        field,
        'DATE_IN_FUTURE'
      );
    }
  }
}
