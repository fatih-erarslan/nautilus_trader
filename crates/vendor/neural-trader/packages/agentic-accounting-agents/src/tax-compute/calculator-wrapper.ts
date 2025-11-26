/**
 * Calculator Wrapper
 *
 * Wraps Rust tax calculation algorithms with TypeScript-friendly interfaces
 * Handles method selection and result normalization
 */

import * as rustCore from '@neural-trader/agentic-accounting-rust-core';

export type TaxMethod = 'FIFO' | 'LIFO' | 'HIFO' | 'SPECIFIC_ID' | 'AVERAGE_COST';

export interface Transaction {
  id: string;
  transactionType: 'BUY' | 'SELL';
  asset: string;
  quantity: string;
  price: string;
  timestamp: string;
  source: string;
  fees: string;
}

export interface TaxLot {
  id: string;
  transactionId: string;
  asset: string;
  quantity: string;
  remainingQuantity: string;
  costBasis: string;
  acquisitionDate: string;
}

export interface Disposal {
  id: string;
  saleTransactionId: string;
  lotId: string;
  asset: string;
  quantity: string;
  proceeds: string;
  costBasis: string;
  gainLoss: string;
  acquisitionDate: string;
  disposalDate: string;
  isLongTerm: boolean;
}

export interface TaxCalculationResult {
  method: TaxMethod;
  disposals: Disposal[];
  totalGain: string;
  totalLoss: string;
  netGainLoss: string;
  shortTermGain: string;
  longTermGain: string;
  unusedLots: TaxLot[];
  calculationTime: number;
}

export class CalculatorWrapper {
  /**
   * Calculate using FIFO method (First In, First Out)
   */
  async calculateFifo(
    sale: Transaction,
    lots: TaxLot[]
  ): Promise<TaxCalculationResult> {
    const startTime = Date.now();

    // Sort lots by acquisition date (oldest first)
    const sortedLots = [...lots].sort((a, b) =>
      new Date(a.acquisitionDate).getTime() - new Date(b.acquisitionDate).getTime()
    );

    const result = await this.processDisposals(sale, sortedLots, 'FIFO');
    result.calculationTime = Date.now() - startTime;

    return result;
  }

  /**
   * Calculate using LIFO method (Last In, First Out)
   */
  async calculateLifo(
    sale: Transaction,
    lots: TaxLot[]
  ): Promise<TaxCalculationResult> {
    const startTime = Date.now();

    // Sort lots by acquisition date (newest first)
    const sortedLots = [...lots].sort((a, b) =>
      new Date(b.acquisitionDate).getTime() - new Date(a.acquisitionDate).getTime()
    );

    const result = await this.processDisposals(sale, sortedLots, 'LIFO');
    result.calculationTime = Date.now() - startTime;

    return result;
  }

  /**
   * Calculate using HIFO method (Highest In, First Out)
   */
  async calculateHifo(
    sale: Transaction,
    lots: TaxLot[]
  ): Promise<TaxCalculationResult> {
    const startTime = Date.now();

    // Sort lots by cost basis (highest first)
    const sortedLots = [...lots].sort((a, b) => {
      const aPrice = parseFloat(rustCore.divideDecimals(a.costBasis, a.quantity));
      const bPrice = parseFloat(rustCore.divideDecimals(b.costBasis, b.quantity));
      return bPrice - aPrice;
    });

    const result = await this.processDisposals(sale, sortedLots, 'HIFO');
    result.calculationTime = Date.now() - startTime;

    return result;
  }

  /**
   * Calculate using Specific ID method
   */
  async calculateSpecificId(
    sale: Transaction,
    lots: TaxLot[],
    selectedLotIds: string[]
  ): Promise<TaxCalculationResult> {
    const startTime = Date.now();

    // Filter and order lots by selection
    const sortedLots = selectedLotIds
      .map(id => lots.find(lot => lot.id === id))
      .filter((lot): lot is TaxLot => lot !== undefined);

    if (sortedLots.length === 0) {
      throw new Error('No valid lots selected for Specific ID method');
    }

    const result = await this.processDisposals(sale, sortedLots, 'SPECIFIC_ID');
    result.calculationTime = Date.now() - startTime;

    return result;
  }

  /**
   * Calculate using Average Cost method
   */
  async calculateAverageCost(
    sale: Transaction,
    lots: TaxLot[]
  ): Promise<TaxCalculationResult> {
    const startTime = Date.now();

    // Calculate weighted average cost basis
    let totalQuantity = '0';
    let totalCost = '0';

    for (const lot of lots) {
      totalQuantity = rustCore.addDecimals(totalQuantity, lot.remainingQuantity);
      totalCost = rustCore.addDecimals(totalCost, lot.costBasis);
    }

    const avgCostPerUnit = rustCore.divideDecimals(totalCost, totalQuantity);

    // Create virtual lot with average cost
    const avgLot: TaxLot = {
      id: 'avg-cost-lot',
      transactionId: 'average',
      asset: sale.asset,
      quantity: totalQuantity,
      remainingQuantity: totalQuantity,
      costBasis: totalCost,
      acquisitionDate: lots[0]?.acquisitionDate || sale.timestamp,
    };

    const result = await this.processDisposals(sale, [avgLot], 'AVERAGE_COST');
    result.calculationTime = Date.now() - startTime;

    return result;
  }

  /**
   * Process disposals from sale transaction and lots
   */
  private async processDisposals(
    sale: Transaction,
    lots: TaxLot[],
    method: TaxMethod
  ): Promise<TaxCalculationResult> {
    const disposals: Disposal[] = [];
    let remainingSaleQty = sale.quantity;
    const unusedLots: TaxLot[] = [];

    for (const lot of lots) {
      if (parseFloat(remainingSaleQty) <= 0) {
        unusedLots.push(lot);
        continue;
      }

      // Determine quantity to dispose
      const disposeQty = parseFloat(lot.remainingQuantity) <= parseFloat(remainingSaleQty)
        ? lot.remainingQuantity
        : remainingSaleQty;

      // Calculate proceeds and cost basis for this disposal
      const proceeds = rustCore.multiplyDecimals(disposeQty, sale.price);
      const costBasisPerUnit = rustCore.divideDecimals(lot.costBasis, lot.quantity);
      const costBasis = rustCore.multiplyDecimals(disposeQty, costBasisPerUnit);

      // Calculate gain/loss
      const gainLoss = rustCore.calculateGainLoss(
        sale.price,
        disposeQty,
        costBasisPerUnit,
        disposeQty
      );

      // Determine if long-term (> 365 days)
      const daysBetween = rustCore.daysBetween(lot.acquisitionDate, sale.timestamp);
      const isLongTerm = daysBetween > 365;

      disposals.push({
        id: `disposal-${sale.id}-${lot.id}`,
        saleTransactionId: sale.id,
        lotId: lot.id,
        asset: sale.asset,
        quantity: disposeQty,
        proceeds,
        costBasis,
        gainLoss,
        acquisitionDate: lot.acquisitionDate,
        disposalDate: sale.timestamp,
        isLongTerm,
      });

      // Update remaining quantities
      remainingSaleQty = rustCore.subtractDecimals(remainingSaleQty, disposeQty);

      // Update lot
      const newRemaining = rustCore.subtractDecimals(lot.remainingQuantity, disposeQty);
      if (parseFloat(newRemaining) > 0) {
        unusedLots.push({
          ...lot,
          remainingQuantity: newRemaining,
        });
      }
    }

    // Calculate totals
    let totalGain = '0';
    let totalLoss = '0';
    let shortTermGain = '0';
    let longTermGain = '0';

    for (const disposal of disposals) {
      const gl = parseFloat(disposal.gainLoss);

      if (gl > 0) {
        totalGain = rustCore.addDecimals(totalGain, disposal.gainLoss);
        if (disposal.isLongTerm) {
          longTermGain = rustCore.addDecimals(longTermGain, disposal.gainLoss);
        } else {
          shortTermGain = rustCore.addDecimals(shortTermGain, disposal.gainLoss);
        }
      } else {
        totalLoss = rustCore.addDecimals(totalLoss, disposal.gainLoss);
      }
    }

    const netGainLoss = rustCore.addDecimals(totalGain, totalLoss);

    return {
      method,
      disposals,
      totalGain,
      totalLoss,
      netGainLoss,
      shortTermGain,
      longTermGain,
      unusedLots,
      calculationTime: 0, // Set by caller
    };
  }
}
