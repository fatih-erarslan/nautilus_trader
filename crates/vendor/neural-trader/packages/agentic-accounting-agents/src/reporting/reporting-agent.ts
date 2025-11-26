/**
 * Reporting Agent
 * Autonomous agent for financial report generation
 * Performance target: <5 seconds for annual reports
 */

import { BaseAgent } from '../base/agent';
import { Transaction, TaxResult } from '@neural-trader/agentic-accounting-types';
import { ReportGenerator, ReportOptions } from '@neural-trader/agentic-accounting-core';
import { ScheduleDGenerator } from '@neural-trader/agentic-accounting-core';
import { Form8949Generator } from '@neural-trader/agentic-accounting-core';

export interface ReportingAgentConfig {
  defaultFormat?: 'json' | 'pdf' | 'csv';
  includeDetails?: boolean;
}

export class ReportingAgent extends BaseAgent {
  private reportGenerator: ReportGenerator;
  private scheduleDGenerator: ScheduleDGenerator;
  private form8949Generator: Form8949Generator;
  private config: ReportingAgentConfig;

  constructor(config: ReportingAgentConfig = {}) {
    super('reporting-agent', 'ReportingAgent');
    this.reportGenerator = new ReportGenerator();
    this.scheduleDGenerator = new ScheduleDGenerator();
    this.form8949Generator = new Form8949Generator();
    this.config = {
      defaultFormat: config.defaultFormat || 'json',
      includeDetails: config.includeDetails ?? true
    };
  }

  /**
   * Execute reporting task
   */
  async execute(task: {
    action: 'generate_pnl' | 'generate_tax_forms' | 'generate_audit' | 'generate_custom';
    transactions?: Transaction[];
    taxResults?: TaxResult[];
    options?: ReportOptions;
    taxpayerInfo?: any;
    year?: number;
  }): Promise<any> {
    this.logger.info(`Executing reporting task: ${task.action}`);

    try {
      switch (task.action) {
        case 'generate_pnl':
          return await this.generatePnLReport(task.transactions!, task.options!);
        case 'generate_tax_forms':
          return await this.generateTaxForms(task.taxResults!, task.year!, task.taxpayerInfo!);
        case 'generate_audit':
          return await this.generateAuditReport(task.transactions!, task.options!);
        case 'generate_custom':
          return await this.generateCustomReport(task);
        default:
          throw new Error(`Unknown action: ${task.action}`);
      }
    } catch (error) {
      this.logger.error('Reporting task failed', { error, task });
      throw error;
    }
  }

  /**
   * Generate P&L report
   */
  private async generatePnLReport(
    transactions: Transaction[],
    options: ReportOptions
  ): Promise<any> {
    const startTime = Date.now();
    this.logger.info('Generating P&L report');

    const report = await this.reportGenerator.generatePnL(transactions, options);

    // Export to requested format
    let exportedReport: any = report;
    if (options.format === 'csv') {
      exportedReport = await this.reportGenerator.exportToCSV(report);
    }

    const duration = Date.now() - startTime;

    // Log learning data
    await this.learn({
      action: 'generate_pnl',
      transactions: transactions.length,
      format: options.format,
      duration,
      performance: transactions.length / (duration / 1000) // transactions per second
    });

    this.logger.info(`P&L report generated in ${duration}ms`);

    return exportedReport;
  }

  /**
   * Generate tax forms (Schedule D and Form 8949)
   */
  private async generateTaxForms(
    taxResults: TaxResult[],
    year: number,
    taxpayerInfo: { name: string; ssn: string }
  ): Promise<any> {
    this.logger.info(`Generating tax forms for year ${year}`);

    // Generate Schedule D
    const scheduleD = await this.scheduleDGenerator.generate(taxResults, year, taxpayerInfo);

    // Generate Form 8949 for each category
    const form8949A = await this.form8949Generator.generate(taxResults, year, taxpayerInfo, 'A');
    const form8949D = await this.form8949Generator.generate(taxResults, year, taxpayerInfo, 'D');

    // Validate forms
    const scheduleDValidation = await this.scheduleDGenerator.validate(scheduleD);
    if (!scheduleDValidation.isValid) {
      this.logger.warn('Schedule D validation failed', { errors: scheduleDValidation.errors });
    }

    const result = {
      scheduleD,
      form8949: {
        shortTerm: form8949A,
        longTerm: form8949D
      },
      validation: {
        scheduleD: scheduleDValidation
      },
      generatedAt: new Date()
    };

    // Log learning data
    await this.learn({
      action: 'generate_tax_forms',
      year,
      transactions: taxResults.length,
      isValid: scheduleDValidation.isValid
    });

    return result;
  }

  /**
   * Generate audit report
   */
  private async generateAuditReport(
    transactions: Transaction[],
    options: ReportOptions
  ): Promise<any> {
    this.logger.info('Generating audit report');

    const report = await this.reportGenerator.generateAuditReport(transactions, options);

    // Add agent metadata
    return {
      ...report,
      agent: {
        name: this.name,
        generatedAt: new Date()
      }
    };
  }

  /**
   * Generate custom report based on user specifications
   */
  private async generateCustomReport(task: any): Promise<any> {
    this.logger.info('Generating custom report');

    // Placeholder for custom report logic
    return {
      type: 'custom',
      data: task,
      generatedAt: new Date()
    };
  }

  /**
   * Format report for PDF export
   */
  async formatForPDF(report: any, reportType: string): Promise<any> {
    this.logger.info(`Formatting ${reportType} for PDF export`);

    switch (reportType) {
      case 'schedule-d':
        return await this.scheduleDGenerator.formatForPDF(report);
      case 'form-8949':
        return await this.form8949Generator.formatForPDF(report);
      default:
        return report;
    }
  }

  /**
   * Batch generate multiple reports
   */
  async generateBatch(reportRequests: any[]): Promise<Map<string, any>> {
    const results = new Map<string, any>();

    await Promise.all(
      reportRequests.map(async request => {
        try {
          const report = await this.execute(request);
          results.set(request.id || request.action, report);
        } catch (error) {
          this.logger.error('Batch report failed', { error, request });
          results.set(request.id || request.action, { error });
        }
      })
    );

    return results;
  }
}
