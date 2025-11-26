/**
 * Tax-Loss Harvesting Agent
 * Autonomous agent for tax optimization
 * Target: Identify 95%+ harvestable losses with <1% wash-sale violations
 */

import { BaseAgent } from '../base/agent';
import { Position, Transaction } from '@neural-trader/agentic-accounting-types';
import { TaxLossHarvestingService, HarvestOpportunity } from '@neural-trader/agentic-accounting-core';

export interface HarvestAgentConfig {
  taxRate?: number;
  minSavingsThreshold?: number;
  autoExecute?: boolean;
  washSalePeriod?: number;
}

export class HarvestAgent extends BaseAgent {
  private harvestingService: TaxLossHarvestingService;
  private config: HarvestAgentConfig;

  constructor(config: HarvestAgentConfig = {}) {
    super('harvest-agent', 'HarvestAgent');
    this.harvestingService = new TaxLossHarvestingService();
    this.config = {
      taxRate: config.taxRate || 0.35,
      minSavingsThreshold: config.minSavingsThreshold || 100,
      autoExecute: config.autoExecute ?? false,
      washSalePeriod: config.washSalePeriod || 30
    };
  }

  /**
   * Execute harvesting task
   */
  async execute(task: {
    action: 'scan' | 'check_wash_sale' | 'find_replacements' | 'generate_plan';
    positions?: Position[];
    currentPrices?: Map<string, number>;
    recentTransactions?: Transaction[];
    asset?: string;
    opportunities?: HarvestOpportunity[];
  }): Promise<any> {
    this.logger.info(`Executing harvesting task: ${task.action}`);

    try {
      switch (task.action) {
        case 'scan':
          return await this.scanForOpportunities(
            task.positions!,
            task.currentPrices!,
            task.recentTransactions!
          );
        case 'check_wash_sale':
          return await this.checkWashSale(task.asset!, task.recentTransactions!);
        case 'find_replacements':
          return await this.findReplacements(task.asset!);
        case 'generate_plan':
          return await this.generatePlan(task.opportunities!);
        default:
          throw new Error(`Unknown action: ${task.action}`);
      }
    } catch (error) {
      this.logger.error('Harvesting task failed', { error, task });
      throw error;
    }
  }

  /**
   * Scan portfolio for harvesting opportunities
   */
  private async scanForOpportunities(
    positions: Position[],
    currentPrices: Map<string, number>,
    recentTransactions: Transaction[]
  ): Promise<any> {
    this.logger.info('Scanning for tax-loss harvesting opportunities');

    const opportunities = await this.harvestingService.scanOpportunities(
      positions,
      currentPrices,
      recentTransactions,
      this.config.taxRate
    );

    // Filter by minimum savings threshold
    const filteredOpportunities = opportunities.filter(
      o => o.potentialTaxSavings.toNumber() >= this.config.minSavingsThreshold!
    );

    // Rank opportunities
    const rankedOpportunities = this.harvestingService.rankOpportunities(filteredOpportunities);

    // Calculate summary metrics
    const summary = {
      totalOpportunities: opportunities.length,
      harvestableOpportunities: rankedOpportunities.filter(o => o.recommendation === 'HARVEST').length,
      totalPotentialSavings: rankedOpportunities
        .reduce((sum, o) => sum + o.potentialTaxSavings.toNumber(), 0)
        .toFixed(2),
      washSaleRisks: rankedOpportunities.filter(o => o.washSaleRisk).length
    };

    // Auto-execute if configured
    if (this.config.autoExecute && rankedOpportunities.length > 0) {
      await this.autoExecuteHarvests(rankedOpportunities);
    }

    // Log learning data
    await this.learn({
      action: 'scan_opportunities',
      positions: positions.length,
      opportunities: opportunities.length,
      harvestable: summary.harvestableOpportunities,
      potentialSavings: summary.totalPotentialSavings,
      washSaleRisks: summary.washSaleRisks
    });

    return {
      opportunities: rankedOpportunities,
      summary
    };
  }

  /**
   * Check for wash sale violations
   */
  private async checkWashSale(asset: string, recentTransactions: Transaction[]): Promise<any> {
    this.logger.info(`Checking wash sale for ${asset}`);

    const washSaleCheck = await this.harvestingService.checkWashSale(asset, recentTransactions);

    // Log learning data
    await this.learn({
      action: 'check_wash_sale',
      asset,
      hasViolation: washSaleCheck.hasViolation,
      daysUntilSafe: washSaleCheck.daysUntilSafe
    });

    return washSaleCheck;
  }

  /**
   * Find replacement assets
   */
  private async findReplacements(asset: string): Promise<any> {
    this.logger.info(`Finding replacement assets for ${asset}`);

    const replacements = await this.harvestingService.findReplacementAssets(asset, 0.7);

    return {
      asset,
      replacements,
      count: replacements.length
    };
  }

  /**
   * Generate execution plan
   */
  private async generatePlan(opportunities: HarvestOpportunity[]): Promise<any> {
    this.logger.info('Generating harvest execution plan');

    const plan = await this.harvestingService.generateExecutionPlan(opportunities);

    // Add agent metadata
    return {
      ...plan,
      agent: {
        name: this.name,
        config: this.config
      },
      createdAt: new Date()
    };
  }

  /**
   * Auto-execute harvest recommendations
   */
  private async autoExecuteHarvests(opportunities: HarvestOpportunity[]): Promise<void> {
    const harvestable = opportunities.filter(o => o.recommendation === 'HARVEST');

    this.logger.info(`Auto-executing ${harvestable.length} harvests`);

    // In production, this would execute actual trades
    for (const opportunity of harvestable) {
      this.logger.info(`Would harvest ${opportunity.asset}`, {
        savings: opportunity.potentialTaxSavings.toString()
      });
    }
  }

  /**
   * Monitor daily for new opportunities
   */
  async monitorDaily(
    positions: Position[],
    currentPrices: Map<string, number>,
    recentTransactions: Transaction[]
  ): Promise<any> {
    this.logger.info('Running daily harvest monitoring');

    const result = await this.scanForOpportunities(positions, currentPrices, recentTransactions);

    // Alert if significant opportunities found
    if (result.summary.harvestableOpportunities > 0) {
      this.logger.warn(`Found ${result.summary.harvestableOpportunities} harvest opportunities`, {
        totalSavings: result.summary.totalPotentialSavings
      });
    }

    return result;
  }

  /**
   * Generate annual harvest report
   */
  async generateAnnualReport(opportunities: HarvestOpportunity[], year: number): Promise<any> {
    this.logger.info(`Generating annual harvest report for ${year}`);

    const harvested = opportunities.filter(o => o.recommendation === 'HARVEST');
    const totalSavings = harvested.reduce(
      (sum, o) => sum + o.potentialTaxSavings.toNumber(),
      0
    );

    return {
      year,
      summary: {
        totalOpportunities: opportunities.length,
        harvested: harvested.length,
        totalSavings: totalSavings.toFixed(2),
        washSaleViolations: 0 // Would track actual violations
      },
      opportunities: harvested,
      generatedAt: new Date()
    };
  }
}
