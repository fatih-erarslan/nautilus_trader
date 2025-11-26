/**
 * Compliance Rule Engine
 * Configurable rules for transaction validation
 */

import { Transaction } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';

export interface ComplianceRule {
  id: string;
  name: string;
  description: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  enabled: boolean;
  validate: (transaction: Transaction, context?: any) => Promise<RuleViolation | null>;
}

export interface RuleViolation {
  ruleId: string;
  severity: string;
  message: string;
  transaction: Transaction;
  timestamp: Date;
  metadata?: any;
}

export class ComplianceRuleEngine {
  private rules: Map<string, ComplianceRule> = new Map();

  constructor() {
    this.initializeDefaultRules();
  }

  /**
   * Initialize default compliance rules
   */
  private initializeDefaultRules(): void {
    // Rule: Transaction amount limits
    this.addRule({
      id: 'transaction-limit',
      name: 'Transaction Amount Limit',
      description: 'Validates transaction does not exceed configured limits',
      severity: 'error',
      enabled: true,
      validate: async (tx: Transaction, context?: any) => {
        const limit = context?.limit || 1000000;
        const value = tx.quantity * tx.price;

        if (value > limit) {
          return {
            ruleId: 'transaction-limit',
            severity: 'error',
            message: `Transaction value $${value} exceeds limit of $${limit}`,
            transaction: tx,
            timestamp: new Date(),
            metadata: { value, limit }
          };
        }
        return null;
      }
    });

    // Rule: Wash sale detection
    this.addRule({
      id: 'wash-sale',
      name: 'Wash Sale Rule',
      description: 'Detects potential wash sale violations (30-day rule)',
      severity: 'warning',
      enabled: true,
      validate: async (tx: Transaction, context?: any) => {
        // Simplified wash sale detection
        // In production, this would check against all related transactions
        if (tx.type === 'SELL' && context?.recentBuys) {
          const thirtyDaysAgo = new Date(tx.timestamp.getTime() - 30 * 24 * 60 * 60 * 1000);
          const hasRecentBuy = context.recentBuys.some(
            (buy: Transaction) => buy.timestamp >= thirtyDaysAgo && buy.asset === tx.asset
          );

          if (hasRecentBuy) {
            return {
              ruleId: 'wash-sale',
              severity: 'warning',
              message: 'Potential wash sale violation detected',
              transaction: tx,
              timestamp: new Date(),
              metadata: { thirtyDaysAgo }
            };
          }
        }
        return null;
      }
    });

    // Rule: Suspicious pattern detection
    this.addRule({
      id: 'suspicious-pattern',
      name: 'Suspicious Activity Pattern',
      description: 'Detects unusual transaction patterns',
      severity: 'warning',
      enabled: true,
      validate: async (tx: Transaction, context?: any) => {
        // Check for round numbers (potential structuring)
        if (tx.type === 'SELL' && tx.quantity % 1000 === 0) {
          return {
            ruleId: 'suspicious-pattern',
            severity: 'warning',
            message: 'Round number transaction detected',
            transaction: tx,
            timestamp: new Date(),
            metadata: { pattern: 'round-number' }
          };
        }
        return null;
      }
    });

    // Rule: Jurisdiction-specific limits
    this.addRule({
      id: 'jurisdiction-limit',
      name: 'Jurisdiction Compliance',
      description: 'Validates transaction against jurisdiction-specific rules',
      severity: 'error',
      enabled: true,
      validate: async (tx: Transaction, context?: any) => {
        const jurisdiction = context?.jurisdiction || 'US';

        // Example: US reporting threshold
        if (jurisdiction === 'US') {
          const value = tx.quantity * tx.price;
          if (value > 10000 && !context?.reportingFiled) {
            return {
              ruleId: 'jurisdiction-limit',
              severity: 'error',
              message: 'Transaction exceeds reporting threshold without filed report',
              transaction: tx,
              timestamp: new Date(),
              metadata: { jurisdiction, threshold: 10000 }
            };
          }
        }
        return null;
      }
    });
  }

  /**
   * Add custom compliance rule
   */
  addRule(rule: ComplianceRule): void {
    this.rules.set(rule.id, rule);
    logger.info(`Added compliance rule: ${rule.id}`, { rule: rule.name });
  }

  /**
   * Remove compliance rule
   */
  removeRule(ruleId: string): void {
    this.rules.delete(ruleId);
    logger.info(`Removed compliance rule: ${ruleId}`);
  }

  /**
   * Enable/disable rule
   */
  setRuleEnabled(ruleId: string, enabled: boolean): void {
    const rule = this.rules.get(ruleId);
    if (rule) {
      rule.enabled = enabled;
      logger.info(`Rule ${ruleId} ${enabled ? 'enabled' : 'disabled'}`);
    }
  }

  /**
   * Validate transaction against all enabled rules
   * Performance target: <500ms
   */
  async validateTransaction(
    transaction: Transaction,
    context?: any
  ): Promise<RuleViolation[]> {
    const violations: RuleViolation[] = [];
    const startTime = Date.now();

    // Run all enabled rules in parallel
    const rulePromises = Array.from(this.rules.values())
      .filter(rule => rule.enabled)
      .map(async rule => {
        try {
          const violation = await rule.validate(transaction, context);
          if (violation) {
            violations.push(violation);
          }
        } catch (error) {
          logger.error(`Rule ${rule.id} failed`, { error, transaction });
        }
      });

    await Promise.all(rulePromises);

    const duration = Date.now() - startTime;
    logger.debug(`Validation completed in ${duration}ms`, {
      violations: violations.length,
      rulesChecked: this.rules.size
    });

    return violations;
  }

  /**
   * Batch validate multiple transactions
   */
  async validateBatch(
    transactions: Transaction[],
    context?: any
  ): Promise<Map<string, RuleViolation[]>> {
    const results = new Map<string, RuleViolation[]>();

    await Promise.all(
      transactions.map(async tx => {
        const violations = await this.validateTransaction(tx, context);
        results.set(tx.id, violations);
      })
    );

    return results;
  }

  /**
   * Get all rules
   */
  getRules(): ComplianceRule[] {
    return Array.from(this.rules.values());
  }

  /**
   * Get rule by ID
   */
  getRule(ruleId: string): ComplianceRule | undefined {
    return this.rules.get(ruleId);
  }
}
