/**
 * Report Generator
 * Generates financial and tax reports
 * Performance target: <5 seconds for annual reports
 */

import { Transaction, TaxResult } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';
import Decimal from 'decimal.js';

export interface ReportOptions {
  startDate: Date;
  endDate: Date;
  format?: 'json' | 'pdf' | 'csv';
  includeDetails?: boolean;
  groupBy?: 'asset' | 'type' | 'month';
}

export interface PnLReport {
  period: {
    start: Date;
    end: Date;
  };
  summary: {
    totalRevenue: Decimal;
    totalCost: Decimal;
    netProfit: Decimal;
    realizedGains: Decimal;
    realizedLosses: Decimal;
    unrealizedGains: Decimal;
    unrealizedLosses: Decimal;
  };
  byAsset: Map<string, AssetPnL>;
  transactions: Transaction[];
  generatedAt: Date;
}

export interface AssetPnL {
  asset: string;
  revenue: Decimal;
  cost: Decimal;
  netProfit: Decimal;
  transactions: number;
}

export class ReportGenerator {
  /**
   * Generate Profit & Loss report
   */
  async generatePnL(transactions: Transaction[], options: ReportOptions): Promise<PnLReport> {
    logger.info('Generating P&L report', {
      transactions: transactions.length,
      start: options.startDate,
      end: options.endDate
    });

    const startTime = Date.now();

    // Filter transactions by date range
    const filteredTxs = transactions.filter(
      tx => tx.timestamp >= options.startDate && tx.timestamp <= options.endDate
    );

    // Calculate summary metrics
    const summary = this.calculateSummary(filteredTxs);

    // Group by asset
    const byAsset = this.groupByAsset(filteredTxs);

    const report: PnLReport = {
      period: {
        start: options.startDate,
        end: options.endDate
      },
      summary,
      byAsset,
      transactions: options.includeDetails ? filteredTxs : [],
      generatedAt: new Date()
    };

    const duration = Date.now() - startTime;
    logger.info(`P&L report generated in ${duration}ms`);

    return report;
  }

  /**
   * Calculate summary metrics
   */
  private calculateSummary(transactions: Transaction[]): PnLReport['summary'] {
    let totalRevenue = new Decimal(0);
    let totalCost = new Decimal(0);
    let realizedGains = new Decimal(0);
    let realizedLosses = new Decimal(0);

    for (const tx of transactions) {
      const value = new Decimal(tx.quantity).mul(tx.price);

      if (tx.type === 'SELL') {
        totalRevenue = totalRevenue.add(value);
        // In production, would calculate actual gain/loss with cost basis
      } else if (tx.type === 'BUY') {
        totalCost = totalCost.add(value);
      } else if (tx.type === 'INCOME' || tx.type === 'DIVIDEND') {
        totalRevenue = totalRevenue.add(value);
        realizedGains = realizedGains.add(value);
      }

      // Add fees to cost
      if (tx.fees) {
        totalCost = totalCost.add(tx.fees);
      }
    }

    const netProfit = totalRevenue.sub(totalCost);

    return {
      totalRevenue,
      totalCost,
      netProfit,
      realizedGains,
      realizedLosses,
      unrealizedGains: new Decimal(0), // Would be calculated from open positions
      unrealizedLosses: new Decimal(0)
    };
  }

  /**
   * Group transactions by asset
   */
  private groupByAsset(transactions: Transaction[]): Map<string, AssetPnL> {
    const byAsset = new Map<string, AssetPnL>();

    for (const tx of transactions) {
      let assetPnL = byAsset.get(tx.asset);
      if (!assetPnL) {
        assetPnL = {
          asset: tx.asset,
          revenue: new Decimal(0),
          cost: new Decimal(0),
          netProfit: new Decimal(0),
          transactions: 0
        };
        byAsset.set(tx.asset, assetPnL);
      }

      const value = new Decimal(tx.quantity).mul(tx.price);

      if (tx.type === 'SELL' || tx.type === 'INCOME' || tx.type === 'DIVIDEND') {
        assetPnL.revenue = assetPnL.revenue.add(value);
      } else if (tx.type === 'BUY') {
        assetPnL.cost = assetPnL.cost.add(value);
      }

      assetPnL.transactions++;
    }

    // Calculate net profit for each asset
    for (const assetPnL of byAsset.values()) {
      assetPnL.netProfit = assetPnL.revenue.sub(assetPnL.cost);
    }

    return byAsset;
  }

  /**
   * Generate tax summary report
   */
  async generateTaxSummary(
    taxResults: TaxResult[],
    year: number
  ): Promise<any> {
    logger.info(`Generating tax summary for ${year}`);

    // Group by short-term vs long-term
    const shortTerm = taxResults.filter(r => !r.isLongTerm);
    const longTerm = taxResults.filter(r => r.isLongTerm);

    const summary = {
      year,
      shortTerm: {
        count: shortTerm.length,
        totalGain: shortTerm.reduce((sum, r) => sum.add(r.gainLoss), new Decimal(0)),
        totalProceeds: shortTerm.reduce((sum, r) => sum.add(r.proceeds), new Decimal(0)),
        totalCostBasis: shortTerm.reduce((sum, r) => sum.add(r.costBasis), new Decimal(0))
      },
      longTerm: {
        count: longTerm.length,
        totalGain: longTerm.reduce((sum, r) => sum.add(r.gainLoss), new Decimal(0)),
        totalProceeds: longTerm.reduce((sum, r) => sum.add(r.proceeds), new Decimal(0)),
        totalCostBasis: longTerm.reduce((sum, r) => sum.add(r.costBasis), new Decimal(0))
      },
      generatedAt: new Date()
    };

    return summary;
  }

  /**
   * Generate audit report
   */
  async generateAuditReport(transactions: Transaction[], options: ReportOptions): Promise<any> {
    logger.info('Generating audit report');

    const filteredTxs = transactions.filter(
      tx => tx.timestamp >= options.startDate && tx.timestamp <= options.endDate
    );

    return {
      period: {
        start: options.startDate,
        end: options.endDate
      },
      totalTransactions: filteredTxs.length,
      transactionsByType: this.groupByType(filteredTxs),
      transactionsByAsset: this.groupByAsset(filteredTxs),
      timeline: this.createTimeline(filteredTxs),
      generatedAt: new Date(),
      auditTrail: {
        generatedBy: 'ReportGenerator',
        format: options.format || 'json'
      }
    };
  }

  /**
   * Group transactions by type
   */
  private groupByType(transactions: Transaction[]): Map<string, number> {
    const byType = new Map<string, number>();

    for (const tx of transactions) {
      byType.set(tx.type, (byType.get(tx.type) || 0) + 1);
    }

    return byType;
  }

  /**
   * Create timeline of transactions
   */
  private createTimeline(transactions: Transaction[]): any[] {
    const timeline: any[] = [];

    // Group by month
    const byMonth = new Map<string, Transaction[]>();

    for (const tx of transactions) {
      const monthKey = `${tx.timestamp.getFullYear()}-${String(tx.timestamp.getMonth() + 1).padStart(2, '0')}`;
      if (!byMonth.has(monthKey)) {
        byMonth.set(monthKey, []);
      }
      byMonth.get(monthKey)!.push(tx);
    }

    // Create timeline entries
    for (const [month, txs] of byMonth.entries()) {
      timeline.push({
        month,
        transactions: txs.length,
        volume: txs.reduce((sum, tx) => sum + (tx.quantity * tx.price), 0)
      });
    }

    return timeline.sort((a, b) => a.month.localeCompare(b.month));
  }

  /**
   * Export report to CSV format
   */
  async exportToCSV(report: PnLReport): Promise<string> {
    const lines: string[] = [];

    // Header
    lines.push('Asset,Revenue,Cost,Net Profit,Transactions');

    // Data rows
    for (const [asset, pnl] of report.byAsset.entries()) {
      lines.push(
        `${asset},${pnl.revenue.toString()},${pnl.cost.toString()},${pnl.netProfit.toString()},${pnl.transactions}`
      );
    }

    return lines.join('\n');
  }
}
