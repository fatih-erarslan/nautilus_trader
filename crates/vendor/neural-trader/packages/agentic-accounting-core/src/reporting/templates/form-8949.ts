/**
 * IRS Form 8949 Generator
 * Sales and Other Dispositions of Capital Assets
 */

import { TaxResult } from '@neural-trader/agentic-accounting-types';
import { logger } from '../../utils/logger';
import Decimal from 'decimal.js';

export interface Form8949 {
  taxYear: number;
  taxpayerInfo: {
    name: string;
    ssn: string;
  };
  category: 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
  transactions: Form8949Transaction[];
  totals: {
    proceeds: Decimal;
    costBasis: Decimal;
    adjustments: Decimal;
    gainLoss: Decimal;
  };
  generatedAt: Date;
}

export interface Form8949Transaction {
  description: string;
  dateAcquired: Date;
  dateSold: Date;
  proceeds: Decimal;
  costBasis: Decimal;
  adjustmentCode?: string;
  adjustmentAmount?: Decimal;
  gainLoss: Decimal;
  metadata?: any;
}

/**
 * Form 8949 Categories:
 * A - Short-term with basis reported to IRS
 * B - Short-term with basis NOT reported to IRS
 * C - Short-term from transactions where you cannot check boxes A or B
 * D - Long-term with basis reported to IRS
 * E - Long-term with basis NOT reported to IRS
 * F - Long-term from transactions where you cannot check boxes D or E
 */

export class Form8949Generator {
  /**
   * Generate IRS Form 8949
   */
  async generate(
    taxResults: TaxResult[],
    taxYear: number,
    taxpayerInfo: { name: string; ssn: string },
    category: Form8949['category']
  ): Promise<Form8949> {
    logger.info(`Generating Form 8949 category ${category} for tax year ${taxYear}`);

    // Filter transactions by category
    const filteredResults = this.filterByCategory(taxResults, category);

    // Convert to Form 8949 transactions
    const transactions = filteredResults.map(r => this.convertToForm8949Transaction(r));

    // Calculate totals
    const totals = {
      proceeds: transactions.reduce((sum, t) => sum.add(t.proceeds), new Decimal(0)),
      costBasis: transactions.reduce((sum, t) => sum.add(t.costBasis), new Decimal(0)),
      adjustments: transactions.reduce(
        (sum, t) => sum.add(t.adjustmentAmount || 0),
        new Decimal(0)
      ),
      gainLoss: transactions.reduce((sum, t) => sum.add(t.gainLoss), new Decimal(0))
    };

    const form: Form8949 = {
      taxYear,
      taxpayerInfo,
      category,
      transactions,
      totals,
      generatedAt: new Date()
    };

    logger.info(`Form 8949 category ${category} generated`, {
      transactions: transactions.length,
      totalGainLoss: totals.gainLoss.toString()
    });

    return form;
  }

  /**
   * Filter transactions by Form 8949 category
   */
  private filterByCategory(
    taxResults: TaxResult[],
    category: Form8949['category']
  ): TaxResult[] {
    const isShortTerm = ['A', 'B', 'C'].includes(category);

    // Filter by short-term vs long-term
    return taxResults.filter(r => {
      if (isShortTerm && r.isLongTerm) return false;
      if (!isShortTerm && !r.isLongTerm) return false;

      // In production, would also filter by whether basis was reported to IRS
      // For now, include all matching term type
      return true;
    });
  }

  /**
   * Convert TaxResult to Form 8949 transaction
   */
  private convertToForm8949Transaction(taxResult: TaxResult): Form8949Transaction {
    const transaction: Form8949Transaction = {
      description: `${taxResult.asset} - ${taxResult.quantity} units`,
      dateAcquired: taxResult.acquisitionDate,
      dateSold: taxResult.disposalDate,
      proceeds: taxResult.proceeds,
      costBasis: taxResult.costBasis,
      gainLoss: taxResult.gainLoss,
      metadata: taxResult.metadata
    };

    // Add wash sale adjustment if applicable
    if (taxResult.washSaleAdjustment && !taxResult.washSaleAdjustment.isZero()) {
      transaction.adjustmentCode = 'W';
      transaction.adjustmentAmount = taxResult.washSaleAdjustment;
    }

    return transaction;
  }

  /**
   * Format Form 8949 for PDF generation
   */
  async formatForPDF(form: Form8949): Promise<any> {
    return {
      title: `Form 8949 - ${form.taxYear}`,
      subtitle: 'Sales and Other Dispositions of Capital Assets',
      taxpayer: {
        name: form.taxpayerInfo.name,
        ssn: this.formatSSN(form.taxpayerInfo.ssn)
      },
      category: {
        box: form.category,
        description: this.getCategoryDescription(form.category)
      },
      columns: [
        'Description of Property',
        'Date Acquired',
        'Date Sold',
        'Proceeds',
        'Cost Basis',
        'Code',
        'Adjustment',
        'Gain/(Loss)'
      ],
      rows: form.transactions.map(t => [
        t.description,
        t.dateAcquired.toLocaleDateString(),
        t.dateSold.toLocaleDateString(),
        this.formatCurrency(t.proceeds),
        this.formatCurrency(t.costBasis),
        t.adjustmentCode || '',
        t.adjustmentAmount ? this.formatCurrency(t.adjustmentAmount) : '',
        this.formatCurrency(t.gainLoss)
      ]),
      totals: {
        proceeds: this.formatCurrency(form.totals.proceeds),
        costBasis: this.formatCurrency(form.totals.costBasis),
        adjustments: this.formatCurrency(form.totals.adjustments),
        gainLoss: this.formatCurrency(form.totals.gainLoss)
      },
      footer: {
        generatedAt: form.generatedAt.toISOString(),
        disclaimer: 'This form is computer-generated. Please review for accuracy.'
      }
    };
  }

  private getCategoryDescription(category: Form8949['category']): string {
    const descriptions = {
      A: 'Short-term transactions reported on Form 1099-B with basis reported to the IRS',
      B: 'Short-term transactions reported on Form 1099-B with basis NOT reported to the IRS',
      C: 'Short-term transactions not reported on Form 1099-B',
      D: 'Long-term transactions reported on Form 1099-B with basis reported to the IRS',
      E: 'Long-term transactions reported on Form 1099-B with basis NOT reported to the IRS',
      F: 'Long-term transactions not reported on Form 1099-B'
    };
    return descriptions[category];
  }

  private formatSSN(ssn: string): string {
    return ssn.replace(/(\d{3})(\d{2})(\d{4})/, '$1-$2-$3');
  }

  private formatCurrency(amount: Decimal): string {
    return `$${amount.toFixed(2)}`;
  }

  /**
   * Split large forms into multiple pages
   */
  async splitIntoPages(form: Form8949, transactionsPerPage: number = 14): Promise<Form8949[]> {
    const pages: Form8949[] = [];

    for (let i = 0; i < form.transactions.length; i += transactionsPerPage) {
      const pageTransactions = form.transactions.slice(i, i + transactionsPerPage);

      const pageTotals = {
        proceeds: pageTransactions.reduce((sum, t) => sum.add(t.proceeds), new Decimal(0)),
        costBasis: pageTransactions.reduce((sum, t) => sum.add(t.costBasis), new Decimal(0)),
        adjustments: pageTransactions.reduce(
          (sum, t) => sum.add(t.adjustmentAmount || 0),
          new Decimal(0)
        ),
        gainLoss: pageTransactions.reduce((sum, t) => sum.add(t.gainLoss), new Decimal(0))
      };

      pages.push({
        ...form,
        transactions: pageTransactions,
        totals: pageTotals
      });
    }

    return pages;
  }
}
