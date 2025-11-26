/**
 * Compliance Validator
 * Real-time transaction compliance validation
 * Performance target: <500ms per validation
 */

import { Transaction } from '@neural-trader/agentic-accounting-types';
import { ComplianceRuleEngine, RuleViolation } from './rules';
import { logger } from '../utils/logger';

export interface ValidationContext {
  jurisdiction?: string;
  recentTransactions?: Transaction[];
  userProfile?: any;
  limits?: {
    daily?: number;
    weekly?: number;
    monthly?: number;
  };
  [key: string]: any;
}

export interface ComplianceValidationResult {
  isCompliant: boolean;
  violations: RuleViolation[];
  warnings: RuleViolation[];
  info: RuleViolation[];
  timestamp: Date;
  duration: number;
}

export class ComplianceValidator {
  private ruleEngine: ComplianceRuleEngine;
  private validationCache: Map<string, ComplianceValidationResult> = new Map();

  constructor() {
    this.ruleEngine = new ComplianceRuleEngine();
  }

  /**
   * Validate transaction for compliance
   */
  async validate(
    transaction: Transaction,
    context?: ValidationContext
  ): Promise<ComplianceValidationResult> {
    const startTime = Date.now();
    logger.info(`Validating transaction ${transaction.id} for compliance`);

    try {
      // Check cache first
      const cacheKey = this.getCacheKey(transaction, context);
      const cached = this.validationCache.get(cacheKey);
      if (cached && this.isCacheValid(cached)) {
        logger.debug('Using cached validation result');
        return cached;
      }

      // Prepare validation context
      const validationContext = await this.prepareContext(transaction, context);

      // Run all compliance rules
      const violations = await this.ruleEngine.validateTransaction(
        transaction,
        validationContext
      );

      // Categorize violations by severity
      const result: ComplianceValidationResult = {
        isCompliant: !violations.some(v => v.severity === 'error' || v.severity === 'critical'),
        violations: violations.filter(v => v.severity === 'error' || v.severity === 'critical'),
        warnings: violations.filter(v => v.severity === 'warning'),
        info: violations.filter(v => v.severity === 'info'),
        timestamp: new Date(),
        duration: Date.now() - startTime
      };

      // Cache result
      this.validationCache.set(cacheKey, result);

      logger.info(`Validation completed in ${result.duration}ms`, {
        isCompliant: result.isCompliant,
        violations: result.violations.length,
        warnings: result.warnings.length
      });

      return result;
    } catch (error) {
      logger.error('Validation failed', { error, transaction });
      throw new Error(`Compliance validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Prepare validation context with additional data
   */
  private async prepareContext(
    transaction: Transaction,
    context?: ValidationContext
  ): Promise<any> {
    const enrichedContext = {
      ...context,
      transaction,
      timestamp: Date.now()
    };

    // Add recent transactions for wash sale detection
    if (transaction.type === 'SELL' && !context?.recentTransactions) {
      // In production, fetch from database
      enrichedContext.recentTransactions = [];
    }

    return enrichedContext;
  }

  /**
   * Pre-validate transaction before execution
   */
  async preValidate(
    transaction: Transaction,
    context?: ValidationContext
  ): Promise<boolean> {
    const result = await this.validate(transaction, context);
    return result.isCompliant;
  }

  /**
   * Post-validate transaction after execution
   */
  async postValidate(
    transaction: Transaction,
    context?: ValidationContext
  ): Promise<ComplianceValidationResult> {
    return await this.validate(transaction, context);
  }

  /**
   * Batch validate multiple transactions
   */
  async validateBatch(
    transactions: Transaction[],
    context?: ValidationContext
  ): Promise<Map<string, ComplianceValidationResult>> {
    const results = new Map<string, ComplianceValidationResult>();

    await Promise.all(
      transactions.map(async tx => {
        const result = await this.validate(tx, context);
        results.set(tx.id, result);
      })
    );

    return results;
  }

  /**
   * Generate compliance report
   */
  async generateReport(
    transactions: Transaction[],
    context?: ValidationContext
  ): Promise<{
    totalTransactions: number;
    compliant: number;
    violations: number;
    warnings: number;
    details: ComplianceValidationResult[];
  }> {
    const results = await this.validateBatch(transactions, context);
    const details = Array.from(results.values());

    return {
      totalTransactions: transactions.length,
      compliant: details.filter(r => r.isCompliant).length,
      violations: details.reduce((sum, r) => sum + r.violations.length, 0),
      warnings: details.reduce((sum, r) => sum + r.warnings.length, 0),
      details
    };
  }

  /**
   * Clear validation cache
   */
  clearCache(): void {
    this.validationCache.clear();
    logger.info('Validation cache cleared');
  }

  private getCacheKey(transaction: Transaction, context?: ValidationContext): string {
    return `${transaction.id}-${JSON.stringify(context || {})}`;
  }

  private isCacheValid(result: ComplianceValidationResult): boolean {
    // Cache valid for 5 minutes
    const fiveMinutes = 5 * 60 * 1000;
    return Date.now() - result.timestamp.getTime() < fiveMinutes;
  }
}
