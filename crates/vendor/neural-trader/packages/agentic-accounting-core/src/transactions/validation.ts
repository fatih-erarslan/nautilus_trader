/**
 * Transaction Validation Service
 * Validates transaction data integrity and completeness
 * Performance target: <100ms per transaction
 */

import { Transaction } from '@neural-trader/agentic-accounting-types';
import { z } from 'zod';

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

// Zod schema for transaction validation
const TransactionSchema = z.object({
  id: z.string().min(1),
  timestamp: z.date(),
  type: z.enum(['BUY', 'SELL', 'TRADE', 'CONVERT', 'INCOME', 'DIVIDEND', 'FEE', 'TRANSFER']),
  asset: z.string().min(1),
  quantity: z.number().positive(),
  price: z.number().nonnegative(),
  fees: z.number().nonnegative().optional(),
  source: z.string().min(1),
  metadata: z.record(z.any()).optional()
});

export class ValidationService {
  /**
   * Validate transaction data
   */
  async validate(transaction: any): Promise<ValidationResult> {
    const result: ValidationResult = {
      isValid: true,
      errors: [],
      warnings: []
    };

    try {
      // Schema validation
      TransactionSchema.parse(transaction);

      // Business rule validation
      this.validateBusinessRules(transaction, result);

      // Data consistency checks
      this.validateConsistency(transaction, result);

      result.isValid = result.errors.length === 0;
    } catch (error) {
      if (error instanceof z.ZodError) {
        result.errors = error.errors.map(e => `${e.path.join('.')}: ${e.message}`);
        result.isValid = false;
      } else {
        result.errors.push('Unknown validation error');
        result.isValid = false;
      }
    }

    return result;
  }

  private validateBusinessRules(transaction: Transaction, result: ValidationResult): void {
    // Quantity must be positive for buys
    if (transaction.type === 'BUY' && transaction.quantity <= 0) {
      result.errors.push('Buy quantity must be positive');
    }

    // Price should be positive for most transaction types
    if (['BUY', 'SELL', 'TRADE'].includes(transaction.type) && transaction.price <= 0) {
      result.errors.push('Price must be positive for buy/sell/trade transactions');
    }

    // Timestamp should not be in the future
    if (transaction.timestamp > new Date()) {
      result.warnings.push('Transaction timestamp is in the future');
    }

    // Asset symbol validation
    if (!/^[A-Z0-9]{2,10}$/.test(transaction.asset)) {
      result.warnings.push('Asset symbol format may be invalid');
    }
  }

  private validateConsistency(transaction: Transaction, result: ValidationResult): void {
    // Check for reasonable price ranges (basic sanity check)
    if (transaction.price > 1e10) {
      result.warnings.push('Unusually high price detected');
    }

    // Check for reasonable quantities
    if (transaction.quantity > 1e15) {
      result.warnings.push('Unusually high quantity detected');
    }

    // Validate fees are reasonable relative to transaction value
    if (transaction.fees) {
      const txValue = transaction.quantity * transaction.price;
      if (transaction.fees > txValue) {
        result.warnings.push('Fees exceed transaction value');
      }
    }
  }

  /**
   * Batch validation for performance
   */
  async validateBatch(transactions: any[]): Promise<Map<string, ValidationResult>> {
    const results = new Map<string, ValidationResult>();

    await Promise.all(
      transactions.map(async (tx) => {
        const result = await this.validate(tx);
        results.set(tx.id, result);
      })
    );

    return results;
  }
}
