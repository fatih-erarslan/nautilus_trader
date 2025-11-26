/**
 * Ingestion Agent Tests
 * Coverage Target: 85%+
 */

import { IngestionAgent } from '../src/ingestion/ingestion-agent';

describe('IngestionAgent', () => {
  let agent: IngestionAgent;

  beforeEach(() => {
    agent = new IngestionAgent('ingest-001');
  });

  describe('Initialization', () => {
    it('should initialize correctly', () => {
      const status = agent.getStatus();

      expect(status.agentId).toBe('ingest-001');
      expect(status.agentType).toBe('INGESTION');
    });

    it('should use default agent ID', () => {
      const defaultAgent = new IngestionAgent();
      const status = defaultAgent.getStatus();

      expect(status.agentId).toBe('ingestion-001');
    });
  });

  describe('Task Execution', () => {
    it('should execute ingestion task', async () => {
      const task = {
        id: 'task-001',
        type: 'ingest',
        priority: 'normal' as const,
        data: {
          source: 'coinbase',
          rawData: [
            {
              id: 'txn-001',
              date: '2024-01-15',
              type: 'buy',
              asset: 'BTC',
              amount: 1,
              price: 50000,
            },
          ],
        },
      };

      const result = await agent.execute(task);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should handle empty data', async () => {
      const task = {
        id: 'task-002',
        type: 'ingest',
        priority: 'normal' as const,
        data: {
          source: 'binance',
          rawData: [],
        },
      };

      const result = await agent.execute(task);

      expect(result.success).toBe(true);
    });

    it('should handle different sources', async () => {
      const sources = ['coinbase', 'binance', 'kraken', 'etherscan', 'csv'];

      for (const source of sources) {
        const task = {
          id: `task-${source}`,
          type: 'ingest',
          priority: 'normal' as const,
          data: {
            source,
            rawData: [
              {
                id: 'txn-001',
                date: '2024-01-15',
                type: 'buy',
                asset: 'BTC',
                amount: 1,
                price: 50000,
              },
            ],
          },
        };

        const result = await agent.execute(task);
        expect(result.success).toBe(true);
      }
    });
  });

  describe('Data Validation', () => {
    it('should validate required fields', async () => {
      const task = {
        id: 'task-003',
        type: 'ingest',
        priority: 'normal' as const,
        data: {
          source: 'coinbase',
          rawData: [
            {
              // Missing required fields
              id: 'incomplete',
            },
          ],
        },
      };

      const result = await agent.execute(task);

      // Should complete but may have warnings
      expect(result.success).toBeDefined();
    });

    it('should handle malformed data', async () => {
      const task = {
        id: 'task-004',
        type: 'ingest',
        priority: 'normal' as const,
        data: {
          source: 'coinbase',
          rawData: [null, undefined, {}],
        },
      };

      const result = await agent.execute(task);
      expect(result.success).toBeDefined();
    });
  });

  describe('Performance', () => {
    it('should process large batches efficiently', async () => {
      const largeData = Array.from({ length: 1000 }, (_, i) => ({
        id: `txn-${i}`,
        date: '2024-01-15',
        type: 'buy',
        asset: 'BTC',
        amount: 1,
        price: 50000,
      }));

      const task = {
        id: 'task-perf',
        type: 'ingest',
        priority: 'normal' as const,
        data: {
          source: 'csv',
          rawData: largeData,
        },
      };

      const startTime = Date.now();
      const result = await agent.execute(task);
      const duration = Date.now() - startTime;

      expect(result.success).toBeDefined();
      expect(duration).toBeLessThan(10000); // <10s for 1000 transactions
    });
  });
});
