/**
 * Integration tests for swarm coordination
 */

import {
  setupTestEnvironment,
  cleanupTestEnvironment,
  createMockSwarm,
  generateMarketData,
  waitForCondition
} from '@neural-trader/test-framework';

describe('Market Microstructure Swarm Integration', () => {
  let swarm: any;

  beforeAll(() => {
    setupTestEnvironment({ timeout: 60000 });
  });

  beforeEach(() => {
    swarm = createMockSwarm({ delay: 10 });
  });

  afterEach(() => {
    swarm.clear();
  });

  afterAll(() => {
    cleanupTestEnvironment();
  });

  describe('Multi-Agent Coordination', () => {
    test('should coordinate multiple analyzer agents', async () => {
      // Add multiple agents
      await swarm.addAgent({
        id: 'analyzer-1',
        role: 'order-book-analyzer',
        state: { symbol: 'AAPL' }
      });

      await swarm.addAgent({
        id: 'analyzer-2',
        role: 'pattern-learner',
        state: { symbol: 'AAPL' }
      });

      await swarm.addAgent({
        id: 'analyzer-3',
        role: 'liquidity-monitor',
        state: { symbol: 'AAPL' }
      });

      expect(swarm.getAllAgents()).toHaveLength(3);
    });

    test('should broadcast market data to all agents', async () => {
      await swarm.addAgent({
        id: 'analyzer-1',
        role: 'order-book-analyzer',
        state: {}
      });

      await swarm.addAgent({
        id: 'analyzer-2',
        role: 'pattern-learner',
        state: {}
      });

      const marketData = generateMarketData('AAPL', 10);

      await swarm.broadcast('coordinator', 'market-update', {
        data: marketData
      });

      const messages = swarm.getMessages();
      expect(messages).toHaveLength(1);
      expect(messages[0].type).toBe('market-update');
    });

    test('should aggregate analysis from multiple agents', async () => {
      // Setup agents
      const agents = ['analyzer-1', 'analyzer-2', 'analyzer-3'];

      for (const agentId of agents) {
        await swarm.addAgent({
          id: agentId,
          role: 'analyzer',
          state: {}
        });
      }

      // Simulate analysis results
      for (const agentId of agents) {
        await swarm.sendMessage({
          from: agentId,
          to: 'coordinator',
          type: 'analysis-result',
          payload: {
            confidence: 0.7 + Math.random() * 0.2,
            prediction: Math.random() > 0.5 ? 'bullish' : 'bearish'
          }
        });
      }

      const messages = swarm.getMessages();
      expect(messages).toHaveLength(3);

      // All messages should be analysis results
      expect(messages.every(m => m.type === 'analysis-result')).toBe(true);
    });
  });

  describe('Consensus Building', () => {
    test('should reach consensus on market prediction', async () => {
      // Setup 5 agents
      for (let i = 0; i < 5; i++) {
        await swarm.addAgent({
          id: `agent-${i}`,
          role: 'predictor',
          state: {}
        });
      }

      const result = await swarm.coordinate(10);

      expect(result.consensusReached).toBe(true);
      expect(result.averageQuality).toBeGreaterThan(0.5);
    });

    test('should handle disagreement among agents', async () => {
      // Setup agents with conflicting views
      await swarm.addAgent({
        id: 'bull-agent',
        role: 'predictor',
        state: { bias: 'bullish' }
      });

      await swarm.addAgent({
        id: 'bear-agent',
        role: 'predictor',
        state: { bias: 'bearish' }
      });

      const result = await swarm.coordinate(20);

      expect(result.agents).toBe(2);
      expect(result.convergenceTime).toBeGreaterThan(0);
    });
  });

  describe('Self-Learning', () => {
    test('should improve coordination over time', async () => {
      // Setup agents
      for (let i = 0; i < 3; i++) {
        await swarm.addAgent({
          id: `agent-${i}`,
          role: 'learner',
          state: { iteration: 0 }
        });
      }

      // First coordination round
      const result1 = await swarm.coordinate(5);

      // Second coordination round (should be faster)
      const result2 = await swarm.coordinate(5);

      // Coordination should become more efficient
      expect(result2.convergenceTime).toBeLessThanOrEqual(
        result1.convergenceTime * 1.2
      );
    });

    test('should share learned patterns between agents', async () => {
      await swarm.addAgent({
        id: 'teacher',
        role: 'pattern-learner',
        state: { patterns: ['pattern-1', 'pattern-2'] }
      });

      await swarm.addAgent({
        id: 'student',
        role: 'pattern-learner',
        state: { patterns: [] }
      });

      // Teacher shares patterns
      await swarm.sendMessage({
        from: 'teacher',
        to: 'student',
        type: 'share-patterns',
        payload: {
          patterns: ['pattern-1', 'pattern-2']
        }
      });

      const messages = swarm.getMessages();
      const shareMessage = messages.find(m => m.type === 'share-patterns');

      expect(shareMessage).toBeDefined();
      expect(shareMessage!.payload.patterns).toHaveLength(2);
    });
  });

  describe('Fault Tolerance', () => {
    test('should continue functioning when agent fails', async () => {
      // Setup 5 agents
      for (let i = 0; i < 5; i++) {
        await swarm.addAgent({
          id: `agent-${i}`,
          role: 'analyzer',
          state: {}
        });
      }

      // Remove one agent
      await swarm.removeAgent('agent-2');

      const result = await swarm.coordinate(5);

      expect(result.agents).toBe(4);
      expect(result.consensusReached).toBe(true);
    });

    test('should recover from communication failures', async () => {
      await swarm.addAgent({
        id: 'sender',
        role: 'analyzer',
        state: {}
      });

      await swarm.addAgent({
        id: 'receiver',
        role: 'analyzer',
        state: {}
      });

      // Simulate failed message
      try {
        await swarm.sendMessage({
          from: 'sender',
          to: 'non-existent',
          type: 'test',
          payload: {}
        });
      } catch (error) {
        // Error expected
      }

      // Swarm should still be functional
      const result = await swarm.coordinate(5);
      expect(result.agents).toBe(2);
    });
  });

  describe('Performance', () => {
    test('should scale to 10 agents efficiently', async () => {
      const agentCount = 10;

      const startTime = performance.now();

      for (let i = 0; i < agentCount; i++) {
        await swarm.addAgent({
          id: `agent-${i}`,
          role: 'analyzer',
          state: {}
        });
      }

      const setupTime = performance.now() - startTime;

      expect(setupTime).toBeLessThan(1000);
      expect(swarm.getAllAgents()).toHaveLength(agentCount);
    });

    test('should handle high message throughput', async () => {
      await swarm.addAgent({
        id: 'sender',
        role: 'broadcaster',
        state: {}
      });

      for (let i = 0; i < 5; i++) {
        await swarm.addAgent({
          id: `receiver-${i}`,
          role: 'listener',
          state: {}
        });
      }

      const startTime = performance.now();

      // Send 100 messages
      for (let i = 0; i < 100; i++) {
        await swarm.broadcast('sender', 'data', { index: i });
      }

      const duration = performance.now() - startTime;

      expect(duration).toBeLessThan(2000); // 100 messages in less than 2s
      expect(swarm.getMessages()).toHaveLength(100);
    });
  });
});
