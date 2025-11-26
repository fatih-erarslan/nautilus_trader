/**
 * Lot Manager
 * Manages individual lots for cost basis tracking
 */

import { Lot } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';
import Decimal from 'decimal.js';

export type AccountingMethod = 'FIFO' | 'LIFO' | 'HIFO' | 'SPECIFIC_ID' | 'AVERAGE_COST';

export class LotManager {
  private lots: Map<string, Lot[]> = new Map();

  /**
   * Add new lot to inventory
   */
  async addLot(lot: Lot): Promise<void> {
    const lots = this.lots.get(lot.asset) || [];
    lots.push(lot);
    this.lots.set(lot.asset, lots);
    logger.debug(`Added lot for ${lot.asset}`, { lotId: lot.id, quantity: lot.quantity.toString() });
  }

  /**
   * Select lots for disposal based on accounting method
   * Performance target: <1ms for 1M+ lots
   */
  async selectLotsForDisposal(
    asset: string,
    quantity: Decimal,
    method: AccountingMethod
  ): Promise<Lot[]> {
    const allLots = this.lots.get(asset) || [];
    const openLots = allLots.filter(lot => lot.isOpen && lot.remainingQuantity.gt(0));

    if (openLots.length === 0) {
      throw new Error(`No open lots available for ${asset}`);
    }

    let selectedLots: Lot[] = [];
    let remainingQuantity = quantity;

    // Sort lots based on accounting method
    const sortedLots = this.sortLotsByMethod(openLots, method);

    // Select lots until quantity is satisfied
    for (const lot of sortedLots) {
      if (remainingQuantity.lte(0)) break;

      selectedLots.push(lot);
      remainingQuantity = remainingQuantity.sub(lot.remainingQuantity);
    }

    if (remainingQuantity.gt(0)) {
      throw new Error(`Insufficient quantity in lots for ${asset}`);
    }

    return selectedLots;
  }

  private sortLotsByMethod(lots: Lot[], method: AccountingMethod): Lot[] {
    switch (method) {
      case 'FIFO':
        // First In, First Out - oldest first
        return lots.sort((a, b) => a.acquisitionDate.getTime() - b.acquisitionDate.getTime());

      case 'LIFO':
        // Last In, First Out - newest first
        return lots.sort((a, b) => b.acquisitionDate.getTime() - a.acquisitionDate.getTime());

      case 'HIFO':
        // Highest In, First Out - highest cost basis first
        return lots.sort((a, b) => {
          const costA = a.costBasis.div(a.quantity);
          const costB = b.costBasis.div(b.quantity);
          return costB.cmp(costA);
        });

      case 'AVERAGE_COST':
        // For average cost, use FIFO as lots are averaged
        return lots.sort((a, b) => a.acquisitionDate.getTime() - b.acquisitionDate.getTime());

      case 'SPECIFIC_ID':
        // Specific identification - would need additional logic
        // For now, default to FIFO
        return lots.sort((a, b) => a.acquisitionDate.getTime() - b.acquisitionDate.getTime());

      default:
        return lots;
    }
  }

  /**
   * Get all open lots for an asset
   */
  getOpenLots(asset: string): Lot[] {
    const allLots = this.lots.get(asset) || [];
    return allLots.filter(lot => lot.isOpen && lot.remainingQuantity.gt(0));
  }

  /**
   * Get total quantity across all open lots
   */
  getTotalQuantity(asset: string): Decimal {
    const openLots = this.getOpenLots(asset);
    return openLots.reduce(
      (total, lot) => total.add(lot.remainingQuantity),
      new Decimal(0)
    );
  }

  /**
   * Get average cost basis across all open lots
   */
  getAverageCostBasis(asset: string): Decimal {
    const openLots = this.getOpenLots(asset);
    if (openLots.length === 0) {
      return new Decimal(0);
    }

    const totalQuantity = this.getTotalQuantity(asset);
    if (totalQuantity.isZero()) {
      return new Decimal(0);
    }

    const totalCost = openLots.reduce(
      (total, lot) => {
        const proportionalCost = lot.costBasis
          .mul(lot.remainingQuantity)
          .div(lot.quantity);
        return total.add(proportionalCost);
      },
      new Decimal(0)
    );

    return totalCost.div(totalQuantity);
  }

  /**
   * Close a specific lot
   */
  async closeLot(lotId: string): Promise<void> {
    for (const [asset, lots] of this.lots.entries()) {
      const lot = lots.find(l => l.id === lotId);
      if (lot) {
        lot.isOpen = false;
        lot.remainingQuantity = new Decimal(0);
        logger.debug(`Closed lot ${lotId} for ${asset}`);
        return;
      }
    }
    throw new Error(`Lot ${lotId} not found`);
  }

  /**
   * Get lot by ID
   */
  getLot(lotId: string): Lot | undefined {
    for (const lots of this.lots.values()) {
      const lot = lots.find(l => l.id === lotId);
      if (lot) return lot;
    }
    return undefined;
  }
}
