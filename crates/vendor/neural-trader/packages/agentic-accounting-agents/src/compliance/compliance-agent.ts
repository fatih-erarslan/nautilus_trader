/**
 * Compliance Agent
 * Autonomous agent for regulatory compliance
 * Performance target: <500ms trade validation
 */

import { BaseAgent } from '../base/agent';
import { Transaction } from '@neural-trader/agentic-accounting-types';
import { ComplianceValidator, ValidationContext } from '@neural-trader/agentic-accounting-core';
import { ComplianceRuleEngine } from '@neural-trader/agentic-accounting-core';

export interface ComplianceAgentConfig {
  strictMode?: boolean;
  jurisdiction?: string;
  autoAlert?: boolean;
  alertThreshold?: number;
}

export class ComplianceAgent extends BaseAgent {
  private validator: ComplianceValidator;
  private ruleEngine: ComplianceRuleEngine;
  private config: ComplianceAgentConfig;

  constructor(config: ComplianceAgentConfig = {}) {
    super('compliance-agent', 'ComplianceAgent');
    this.validator = new ComplianceValidator();
    this.ruleEngine = new ComplianceRuleEngine();
    this.config = {
      strictMode: config.strictMode ?? true,
      jurisdiction: config.jurisdiction ?? 'US',
      autoAlert: config.autoAlert ?? true,
      alertThreshold: config.alertThreshold ?? 0.7
    };
  }

  /**
   * Execute compliance check
   */
  async execute(task: {
    action: 'validate' | 'check_batch' | 'generate_report';
    transaction?: Transaction;
    transactions?: Transaction[];
    context?: ValidationContext;
  }): Promise<any> {
    this.logger.info(`Executing compliance task: ${task.action}`);

    try {
      switch (task.action) {
        case 'validate':
          return await this.validateTransaction(task.transaction!, task.context);
        case 'check_batch':
          return await this.validateBatch(task.transactions!, task.context);
        case 'generate_report':
          return await this.generateComplianceReport(task.transactions!, task.context);
        default:
          throw new Error(`Unknown action: ${task.action}`);
      }
    } catch (error) {
      this.logger.error('Compliance task failed', { error, task });
      throw error;
    }
  }

  /**
   * Validate single transaction
   */
  private async validateTransaction(
    transaction: Transaction,
    context?: ValidationContext
  ): Promise<any> {
    const startTime = Date.now();

    // Enrich context with jurisdiction and config
    const enrichedContext: ValidationContext = {
      ...context,
      jurisdiction: this.config.jurisdiction,
      strictMode: this.config.strictMode
    };

    // Run validation
    const result = await this.validator.validate(transaction, enrichedContext);

    // Auto-alert if configured
    if (this.config.autoAlert && result.violations.length > 0) {
      await this.sendAlert(transaction, result);
    }

    // Log learning data
    await this.learn({
      action: 'validate_transaction',
      transactionId: transaction.id,
      isCompliant: result.isCompliant,
      violations: result.violations.length,
      warnings: result.warnings.length,
      duration: Date.now() - startTime,
      strictMode: this.config.strictMode
    });

    return result;
  }

  /**
   * Validate batch of transactions
   */
  private async validateBatch(
    transactions: Transaction[],
    context?: ValidationContext
  ): Promise<any> {
    this.logger.info(`Validating batch of ${transactions.length} transactions`);

    const enrichedContext: ValidationContext = {
      ...context,
      jurisdiction: this.config.jurisdiction
    };

    const results = await this.validator.validateBatch(transactions, enrichedContext);

    // Count violations
    let totalViolations = 0;
    let totalWarnings = 0;

    for (const result of results.values()) {
      totalViolations += result.violations.length;
      totalWarnings += result.warnings.length;

      // Auto-alert for critical violations
      if (this.config.autoAlert && result.violations.length > 0) {
        const tx = transactions.find(t => t.id === result.violations[0].transaction.id);
        if (tx) {
          await this.sendAlert(tx, result);
        }
      }
    }

    this.logger.info('Batch validation completed', {
      total: transactions.length,
      violations: totalViolations,
      warnings: totalWarnings
    });

    return {
      results,
      summary: {
        total: transactions.length,
        violations: totalViolations,
        warnings: totalWarnings
      }
    };
  }

  /**
   * Generate compliance report
   */
  private async generateComplianceReport(
    transactions: Transaction[],
    context?: ValidationContext
  ): Promise<any> {
    this.logger.info('Generating compliance report');

    const enrichedContext: ValidationContext = {
      ...context,
      jurisdiction: this.config.jurisdiction
    };

    const report = await this.validator.generateReport(transactions, enrichedContext);

    // Add metadata
    return {
      ...report,
      metadata: {
        jurisdiction: this.config.jurisdiction,
        strictMode: this.config.strictMode,
        generatedAt: new Date(),
        agent: this.name
      }
    };
  }

  /**
   * Send compliance alert
   */
  private async sendAlert(transaction: Transaction, result: any): Promise<void> {
    this.logger.warn('Compliance violation detected', {
      transactionId: transaction.id,
      violations: result.violations.length,
      severity: result.violations[0]?.severity
    });

    // In production, this would send to monitoring system
    // For now, just log
  }

  /**
   * Check if transaction meets compliance threshold
   */
  async isCompliant(transaction: Transaction, context?: ValidationContext): Promise<boolean> {
    const result = await this.validateTransaction(transaction, context);
    return result.isCompliant;
  }

  /**
   * Get all active compliance rules
   */
  async getRules(): Promise<any[]> {
    return this.ruleEngine.getRules();
  }

  /**
   * Add custom compliance rule
   */
  async addRule(rule: any): Promise<void> {
    this.ruleEngine.addRule(rule);
    this.logger.info(`Added custom rule: ${rule.id}`);
  }

  /**
   * Enable/disable rule
   */
  async setRuleEnabled(ruleId: string, enabled: boolean): Promise<void> {
    this.ruleEngine.setRuleEnabled(ruleId, enabled);
    this.logger.info(`Rule ${ruleId} ${enabled ? 'enabled' : 'disabled'}`);
  }
}
