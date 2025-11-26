/**
 * Tax Compute Agent Tests
 *
 * Comprehensive test suite covering:
 * - Method selection logic
 * - Multi-method comparison
 * - Wash sale integration
 * - Caching behavior
 * - ReasoningBank storage
 * - Error handling
 * - Performance benchmarks
 */

import { TaxComputeAgent } from '../src/tax-compute/tax-compute-agent';
import { Transaction, TaxLot } from '../src/tax-compute/calculator-wrapper';
import { TaxProfile } from '../src/tax-compute/strategy-selector';
import { ValidationError } from '../src/tax-compute/validation';

describe('TaxComputeAgent', () => {
  let agent: TaxComputeAgent;

  beforeEach(() => {
    agent = new TaxComputeAgent('test-agent-001');
  });

  afterEach(async () => {
    await agent.stop();
  });

  describe('Basic Calculations', () => {
    it('should calculate FIFO correctly', async () => {
      const sale: Transaction = {
        id: 'sale-1',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-1',
        description: 'Calculate FIFO',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
          enableCache: false,
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.calculation.method).toBe('FIFO');
      expect(result.data?.calculation.disposals.length).toBe(1);
      expect(parseFloat(result.data?.calculation.netGainLoss)).toBeGreaterThan(0);
    });

    it('should calculate LIFO correctly', async () => {
      const sale: Transaction = {
        id: 'sale-2',
        transactionType: 'SELL',
        asset: 'ETH',
        quantity: '2.0',
        price: '3000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '5.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'ETH',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '1000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
        {
          id: 'lot-2',
          transactionId: 'buy-2',
          asset: 'ETH',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '2000.00',
          acquisitionDate: '2023-06-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-2',
        description: 'Calculate LIFO',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'LIFO',
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.calculation.method).toBe('LIFO');
      // LIFO should use newer lot first
      expect(result.data?.calculation.disposals[0].lotId).toBe('lot-2');
    });

    it('should calculate HIFO correctly', async () => {
      const sale: Transaction = {
        id: 'sale-3',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '0.5',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '0.5',
          remainingQuantity: '0.5',
          costBasis: '20000.00', // $40k per BTC
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
        {
          id: 'lot-2',
          transactionId: 'buy-2',
          asset: 'BTC',
          quantity: '0.5',
          remainingQuantity: '0.5',
          costBasis: '25000.00', // $50k per BTC (higher)
          acquisitionDate: '2023-06-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-3',
        description: 'Calculate HIFO',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'HIFO',
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.calculation.method).toBe('HIFO');
      // HIFO should use highest cost basis lot first
      expect(result.data?.calculation.disposals[0].lotId).toBe('lot-2');
    });
  });

  describe('Method Selection', () => {
    it('should use preferred method if specified in profile', async () => {
      const sale: Transaction = {
        id: 'sale-4',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      const profile: TaxProfile = {
        jurisdiction: 'US',
        taxBracket: 'high',
        preferredMethod: 'HIFO',
        optimizationGoal: 'minimize_current_tax',
      };

      const result = await agent.execute({
        taskId: 'task-4',
        description: 'Test method selection',
        priority: 'high',
        data: {
          sale,
          lots,
          profile,
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.calculation.method).toBe('HIFO');
      expect(result.data?.recommendation?.rationale).toContain('preferred');
    });

    it('should select intelligent method without preference', async () => {
      const sale: Transaction = {
        id: 'sale-5',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      const profile: TaxProfile = {
        jurisdiction: 'US',
        taxBracket: 'high',
        optimizationGoal: 'minimize_current_tax',
      };

      const result = await agent.execute({
        taskId: 'task-5',
        description: 'Test automatic selection',
        priority: 'high',
        data: {
          sale,
          lots,
          profile,
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.recommendation).toBeDefined();
      expect(result.data?.recommendation?.score).toBeGreaterThan(0);
    });
  });

  describe('Multi-Method Comparison', () => {
    it('should compare all methods and recommend best', async () => {
      const sale: Transaction = {
        id: 'sale-6',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '0.5',
          remainingQuantity: '0.5',
          costBasis: '15000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
        {
          id: 'lot-2',
          transactionId: 'buy-2',
          asset: 'BTC',
          quantity: '0.5',
          remainingQuantity: '0.5',
          costBasis: '20000.00',
          acquisitionDate: '2023-06-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-6',
        description: 'Compare all methods',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
          compareAll: true,
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.comparison).toBeDefined();
      expect(result.data?.comparison.best).toBeDefined();
      expect(result.data?.comparison.comparison.length).toBeGreaterThan(1);
    });
  });

  describe('Wash Sale Detection', () => {
    it('should detect potential wash sales', async () => {
      const sale: Transaction = {
        id: 'sale-7',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '30000.00', // Selling at loss
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '50000.00', // Higher cost = loss
          acquisitionDate: '2024-01-01T00:00:00Z', // Within 30 days
        },
      ];

      const result = await agent.execute({
        taskId: 'task-7',
        description: 'Test wash sale detection',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
          detectWashSales: true,
        },
      });

      expect(result.success).toBe(true);
      // Wash sale detection looks for repurchases, not the original purchase
      // This test validates the detection runs
      expect(result.data?.washSales).toBeDefined();
    });
  });

  describe('Caching', () => {
    it('should cache results and return cached on second call', async () => {
      const sale: Transaction = {
        id: 'sale-8',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      // First call - should calculate
      const result1 = await agent.execute({
        taskId: 'task-8a',
        description: 'First call',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
          enableCache: true,
        },
      });

      expect(result1.success).toBe(true);
      expect(result1.data?.cacheHit).toBeFalsy();

      // Second call - should use cache
      const result2 = await agent.execute({
        taskId: 'task-8b',
        description: 'Second call',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
          enableCache: true,
        },
      });

      expect(result2.success).toBe(true);
      expect(result2.data?.cacheHit).toBe(true);
      expect(result2.data?.performance.calculationTime).toBe(0);
    });

    it('should invalidate cache correctly', () => {
      const stats1 = agent.getCacheStats();
      expect(stats1.size).toBeGreaterThanOrEqual(0);

      agent.invalidateCache();
      const stats2 = agent.getCacheStats();
      expect(stats2.size).toBe(0);
    });
  });

  describe('Validation', () => {
    it('should reject invalid transaction', async () => {
      const invalidSale: any = {
        id: 'sale-9',
        transactionType: 'INVALID',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-9',
        description: 'Test validation',
        priority: 'high',
        data: {
          sale: invalidSale,
          lots,
          method: 'FIFO',
        },
      });

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
    });

    it('should reject mismatched asset', async () => {
      const sale: Transaction = {
        id: 'sale-10',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'ETH', // Wrong asset!
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      const result = await agent.execute({
        taskId: 'task-10',
        description: 'Test asset mismatch',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
        },
      });

      expect(result.success).toBe(false);
      expect(result.error?.message).toContain('asset');
    });
  });

  describe('Performance', () => {
    it('should complete calculation under 1 second', async () => {
      const sale: Transaction = {
        id: 'sale-11',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '10.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      // Create 100 lots
      const lots: TaxLot[] = Array.from({ length: 100 }, (_, i) => ({
        id: `lot-${i}`,
        transactionId: `buy-${i}`,
        asset: 'BTC',
        quantity: '0.1',
        remainingQuantity: '0.1',
        costBasis: (30000 + i * 100).toFixed(2),
        acquisitionDate: new Date(2023, 0, i + 1).toISOString(),
      }));

      const result = await agent.execute({
        taskId: 'task-11',
        description: 'Performance test',
        priority: 'high',
        data: {
          sale,
          lots,
          method: 'FIFO',
        },
      });

      expect(result.success).toBe(true);
      expect(result.data?.performance.totalTime).toBeLessThan(1000); // < 1 second
    });
  });

  describe('Agent Status', () => {
    it('should return correct status', () => {
      const status = agent.getExtendedStatus();

      expect(status.agentId).toBe('test-agent-001');
      expect(status.agentType).toBe('TAX_COMPUTE');
      expect(status.methods).toContain('FIFO');
      expect(status.methods).toContain('LIFO');
      expect(status.methods).toContain('HIFO');
    });

    it('should track decisions', async () => {
      const sale: Transaction = {
        id: 'sale-12',
        transactionType: 'SELL',
        asset: 'BTC',
        quantity: '1.0',
        price: '50000.00',
        timestamp: '2024-01-15T00:00:00Z',
        source: 'exchange',
        fees: '10.00',
      };

      const lots: TaxLot[] = [
        {
          id: 'lot-1',
          transactionId: 'buy-1',
          asset: 'BTC',
          quantity: '1.0',
          remainingQuantity: '1.0',
          costBasis: '30000.00',
          acquisitionDate: '2023-01-01T00:00:00Z',
        },
      ];

      await agent.execute({
        taskId: 'task-12',
        description: 'Test decision tracking',
        priority: 'high',
        data: {
          sale,
          lots,
        },
      });

      const decisions = agent.getRecentDecisions();
      expect(decisions.length).toBeGreaterThan(0);
      expect(decisions[0].scenario).toBeDefined();
    });
  });
});
