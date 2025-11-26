/**
 * Tax Compute Agent
 *
 * Orchestrates all tax calculation methods with:
 * - Intelligent method selection
 * - Multi-method comparison
 * - Wash sale detection integration
 * - Result caching
 * - ReasoningBank learning
 * - Performance monitoring
 */

import { BaseAgent, AgentTask, AgentResult } from '../base/agent';
import { CalculatorWrapper, TaxMethod, Transaction, TaxLot } from './calculator-wrapper';
import { StrategySelector, TaxProfile, MethodRecommendation } from './strategy-selector';
import { TaxInputValidator, ValidationError } from './validation';
import { TaxCalculationCache } from './cache';
import * as rustCore from '@neural-trader/agentic-accounting-rust-core';

export interface TaxCalculationTask extends AgentTask {
  data: {
    sale: Transaction;
    lots: TaxLot[];
    profile?: TaxProfile;
    method?: TaxMethod;
    compareAll?: boolean;
    enableCache?: boolean;
    detectWashSales?: boolean;
  };
}

export interface TaxComputeResult {
  calculation: any;
  recommendation?: MethodRecommendation;
  comparison?: any;
  washSales?: any[];
  cacheHit?: boolean;
  performance: {
    validationTime: number;
    calculationTime: number;
    totalTime: number;
  };
}

export class TaxComputeAgent extends BaseAgent {
  private calculator: CalculatorWrapper;
  private selector: StrategySelector;
  private validator: TaxInputValidator;
  private cache: TaxCalculationCache;

  constructor(agentId: string = 'tax-compute-001') {
    super({
      agentId,
      agentType: 'TAX_COMPUTE',
      enableLearning: true,
      enableMetrics: true,
      logLevel: 'info',
    });

    this.calculator = new CalculatorWrapper();
    this.selector = new StrategySelector();
    this.validator = new TaxInputValidator();
    this.cache = new TaxCalculationCache();
  }

  /**
   * Execute tax calculation task
   */
  async execute(task: TaxCalculationTask): Promise<AgentResult<TaxComputeResult>> {
    const startTime = Date.now();
    const { sale, lots, profile, method, compareAll, enableCache, detectWashSales } = task.data;

    try {
      // 1. Validate inputs
      const validationStart = Date.now();
      this.validator.validateTransaction(sale);
      lots.forEach(lot => this.validator.validateTaxLot(lot));
      this.validator.validateSaleAndLots(sale, lots);
      const validationTime = Date.now() - validationStart;

      // 2. Check cache if enabled
      let cacheHit = false;
      if (enableCache && method) {
        const cacheKey = this.cache.generateKey(
          method,
          sale.id,
          lots.map(l => l.id)
        );
        const cached = this.cache.get(cacheKey);
        if (cached) {
          cacheHit = true;
          return {
            success: true,
            data: {
              calculation: cached,
              cacheHit: true,
              performance: {
                validationTime,
                calculationTime: 0,
                totalTime: Date.now() - startTime,
              },
            },
          };
        }
      }

      // 3. Select optimal method or use specified
      const calculationStart = Date.now();
      let selectedMethod: TaxMethod;
      let recommendation: MethodRecommendation | undefined;

      if (method) {
        selectedMethod = method;
      } else {
        const defaultProfile: TaxProfile = profile || {
          jurisdiction: 'US',
          taxBracket: 'medium',
          optimizationGoal: 'minimize_current_tax',
        };
        recommendation = await this.selector.selectOptimalMethod(
          sale,
          lots,
          defaultProfile
        );
        selectedMethod = recommendation.method;

        // Log decision
        await this.logDecision(
          'method_selection',
          selectedMethod,
          recommendation.rationale,
          'SUCCESS',
          {
            score: recommendation.score,
            saleId: sale.id,
            asset: sale.asset,
          }
        );
      }

      // 4. Calculate using selected method
      const calculation = await this.calculateWithMethod(selectedMethod, sale, lots);

      // 5. Compare all methods if requested
      let comparison;
      if (compareAll) {
        comparison = await this.compareAllMethods(sale, lots);
      }

      // 6. Detect wash sales if enabled
      let washSales;
      if (detectWashSales) {
        washSales = await this.detectWashSales(sale, calculation.disposals);

        if (washSales.length > 0) {
          await this.logDecision(
            'wash_sale_detected',
            `Found ${washSales.length} potential wash sales`,
            'Transactions within 30-day window detected',
            'SUCCESS',
            {
              count: washSales.length,
              saleId: sale.id,
            }
          );
        }
      }

      const calculationTime = Date.now() - calculationStart;

      // 7. Store in cache
      if (enableCache) {
        const cacheKey = this.cache.generateKey(
          selectedMethod,
          sale.id,
          lots.map(l => l.id)
        );
        this.cache.set(cacheKey, calculation, undefined, {
          saleId: sale.id,
          asset: sale.asset,
        });
      }

      // 8. Log successful calculation
      await this.logDecision(
        'tax_calculation',
        selectedMethod,
        `Calculated ${calculation.disposals.length} disposals with net ${calculation.netGainLoss}`,
        'SUCCESS',
        {
          method: selectedMethod,
          disposalCount: calculation.disposals.length,
          netGainLoss: calculation.netGainLoss,
          calculationTime,
        }
      );

      return {
        success: true,
        data: {
          calculation,
          recommendation,
          comparison,
          washSales,
          cacheHit,
          performance: {
            validationTime,
            calculationTime,
            totalTime: Date.now() - startTime,
          },
        },
        metrics: {
          startTime,
          endTime: Date.now(),
          duration: Date.now() - startTime,
        },
      };
    } catch (error) {
      await this.logDecision(
        'tax_calculation_error',
        'FAILED',
        error instanceof Error ? error.message : String(error),
        'FAILURE',
        {
          saleId: sale.id,
          error: error instanceof Error ? error.message : String(error),
        }
      );

      return {
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        metrics: {
          startTime,
          endTime: Date.now(),
          duration: Date.now() - startTime,
        },
      };
    }
  }

  /**
   * Calculate using specific method
   */
  private async calculateWithMethod(
    method: TaxMethod,
    sale: Transaction,
    lots: TaxLot[]
  ): Promise<any> {
    switch (method) {
      case 'FIFO':
        return await this.calculator.calculateFifo(sale, lots);
      case 'LIFO':
        return await this.calculator.calculateLifo(sale, lots);
      case 'HIFO':
        return await this.calculator.calculateHifo(sale, lots);
      case 'AVERAGE_COST':
        return await this.calculator.calculateAverageCost(sale, lots);
      case 'SPECIFIC_ID':
        // For Specific ID, use first N lots (in production, would be user-selected)
        const selectedLotIds = lots.slice(0, Math.min(5, lots.length)).map(l => l.id);
        return await this.calculator.calculateSpecificId(sale, lots, selectedLotIds);
      default:
        throw new Error(`Unsupported method: ${method}`);
    }
  }

  /**
   * Compare all calculation methods
   */
  async compareAllMethods(sale: Transaction, lots: TaxLot[]): Promise<any> {
    const results = new Map();

    // Calculate with each method
    const methods: TaxMethod[] = ['FIFO', 'LIFO', 'HIFO', 'AVERAGE_COST'];

    for (const method of methods) {
      try {
        const result = await this.calculateWithMethod(method, sale, lots);
        results.set(method, result);
      } catch (error) {
        console.warn(`Failed to calculate with ${method}:`, error);
      }
    }

    // Compare results
    const comparison = await this.selector.compareResults(results);

    await this.logDecision(
      'method_comparison',
      comparison.best,
      `Best method saves $${comparison.savings} compared to worst`,
      'SUCCESS',
      {
        savings: comparison.savings,
        methodCount: results.size,
      }
    );

    return comparison;
  }

  /**
   * Detect potential wash sales
   */
  private async detectWashSales(sale: Transaction, disposals: any[]): Promise<any[]> {
    const washSales: any[] = [];

    for (const disposal of disposals) {
      // Only check losses
      if (parseFloat(disposal.gainLoss) >= 0) {
        continue;
      }

      // Check if within wash sale window
      // In production, would check actual purchase transactions
      const isWithinWindow = rustCore.isWithinWashSalePeriod(
        disposal.disposalDate,
        disposal.acquisitionDate
      );

      if (isWithinWindow) {
        washSales.push({
          disposalId: disposal.id,
          asset: disposal.asset,
          loss: disposal.gainLoss,
          disposalDate: disposal.disposalDate,
          acquisitionDate: disposal.acquisitionDate,
          warning: 'Potential wash sale - verify no repurchases within 30 days',
        });
      }
    }

    return washSales;
  }

  /**
   * Invalidate cache for asset
   */
  invalidateCache(asset?: string): number {
    if (asset) {
      return this.cache.invalidateAsset(asset);
    }
    this.cache.clear();
    return 0;
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return this.cache.getStats();
  }

  /**
   * Get agent status with extended info
   */
  getExtendedStatus() {
    const baseStatus = this.getStatus();
    const cacheStats = this.getCacheStats();

    return {
      ...baseStatus,
      cache: cacheStats,
      calculator: 'Rust NAPI',
      methods: ['FIFO', 'LIFO', 'HIFO', 'SPECIFIC_ID', 'AVERAGE_COST'],
    };
  }
}
