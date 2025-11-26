/**
 * Position Manager
 * Tracks open positions and manages lot inventory
 */

import { Transaction, Position, Lot } from '@neural-trader/agentic-accounting-types';
import { logger } from '../utils/logger';
import { LotManager } from './lots';
import Decimal from 'decimal.js';

export class PositionManager {
  private positions: Map<string, Position> = new Map();
  private lotManager: LotManager;

  constructor() {
    this.lotManager = new LotManager();
  }

  /**
   * Update positions based on new transaction
   */
  async updatePosition(transaction: Transaction): Promise<Position> {
    const key = this.getPositionKey(transaction.asset, transaction.metadata?.wallet);

    let position = this.positions.get(key);
    if (!position) {
      position = this.createPosition(transaction.asset, transaction.metadata?.wallet);
      this.positions.set(key, position);
    }

    // Update position based on transaction type
    switch (transaction.type) {
      case 'BUY':
      case 'INCOME':
      case 'DIVIDEND':
        await this.handleAcquisition(position, transaction);
        break;
      case 'SELL':
      case 'CONVERT':
        await this.handleDisposal(position, transaction);
        break;
      case 'TRANSFER':
        await this.handleTransfer(position, transaction);
        break;
    }

    // Recalculate position metrics
    this.recalculatePosition(position);

    return position;
  }

  private async handleAcquisition(position: Position, transaction: Transaction): Promise<void> {
    // Create new lot for acquisition
    const lot: Lot = {
      id: `${transaction.id}-lot`,
      transactionId: transaction.id,
      asset: transaction.asset,
      quantity: new Decimal(transaction.quantity),
      costBasis: new Decimal(transaction.price).mul(transaction.quantity),
      acquisitionDate: transaction.timestamp,
      remainingQuantity: new Decimal(transaction.quantity),
      isOpen: true
    };

    await this.lotManager.addLot(lot);
    position.lots.push(lot);
    position.quantity = position.quantity.add(transaction.quantity);
    position.totalCost = position.totalCost.add(lot.costBasis);
  }

  private async handleDisposal(position: Position, transaction: Transaction): Promise<void> {
    // Remove quantity from position using accounting method
    // This would integrate with the tax calculation engine
    const lotsToClose = await this.lotManager.selectLotsForDisposal(
      position.asset,
      new Decimal(transaction.quantity),
      'FIFO' // Would be configurable
    );

    for (const lot of lotsToClose) {
      const quantityToClose = Decimal.min(lot.remainingQuantity, transaction.quantity);
      lot.remainingQuantity = lot.remainingQuantity.sub(quantityToClose);

      if (lot.remainingQuantity.isZero()) {
        lot.isOpen = false;
      }
    }

    position.quantity = position.quantity.sub(transaction.quantity);
  }

  private async handleTransfer(position: Position, transaction: Transaction): Promise<void> {
    // Transfers don't change cost basis, just move assets
    logger.debug(`Processing transfer for ${transaction.asset}`, { transaction });
  }

  private recalculatePosition(position: Position): void {
    // Recalculate total quantity and average cost basis
    let totalQuantity = new Decimal(0);
    let totalCost = new Decimal(0);

    for (const lot of position.lots) {
      if (lot.isOpen) {
        totalQuantity = totalQuantity.add(lot.remainingQuantity);
        const proportionalCost = lot.costBasis
          .mul(lot.remainingQuantity)
          .div(lot.quantity);
        totalCost = totalCost.add(proportionalCost);
      }
    }

    position.quantity = totalQuantity;
    position.totalCost = totalCost;
    position.averageCostBasis = totalQuantity.isZero()
      ? new Decimal(0)
      : totalCost.div(totalQuantity);
    position.lastUpdated = new Date();
  }

  private createPosition(asset: string, wallet?: string): Position {
    return {
      asset,
      wallet,
      quantity: new Decimal(0),
      totalCost: new Decimal(0),
      averageCostBasis: new Decimal(0),
      lots: [],
      lastUpdated: new Date()
    };
  }

  private getPositionKey(asset: string, wallet?: string): string {
    return wallet ? `${asset}:${wallet}` : asset;
  }

  /**
   * Get all open positions
   */
  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get position for specific asset
   */
  getPosition(asset: string, wallet?: string): Position | undefined {
    return this.positions.get(this.getPositionKey(asset, wallet));
  }

  /**
   * Calculate unrealized gains/losses
   */
  async calculateUnrealizedPnL(asset: string, currentPrice: number, wallet?: string): Promise<Decimal> {
    const position = this.getPosition(asset, wallet);
    if (!position) {
      return new Decimal(0);
    }

    const currentValue = position.quantity.mul(currentPrice);
    return currentValue.sub(position.totalCost);
  }
}
