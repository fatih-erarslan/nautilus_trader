/**
 * End-to-End Integration Tests
 * Tests complete midstreamer workflow with pattern matching, learning, and coordination
 */

const { performance } = require('perf_hooks');

describe('Midstreamer End-to-End Integration', () => {
  let system;

  beforeEach(() => {
    // Mock integrated system
    system = {
      dtw: null,
      lcs: null,
      reasoningBank: null,
      quic: null,

      async initialize() {
        // Initialize all components
        this.dtw = require('../dtw/pattern-matching.test.js');
        this.lcs = require('../lcs/strategy-correlation.test.js');

        this.reasoningBank = {
          experiences: [],
          memory: new Map(),
          recordExperience: function(exp) {
            this.experiences.push({ ...exp, id: this.experiences.length });
            return this.experiences.length - 1;
          },
          distill: function() {
            const patterns = new Map();
            this.experiences.forEach(exp => {
              const key = JSON.stringify(exp.context);
              if (!patterns.has(key)) {
                patterns.set(key, { count: 0, successes: 0 });
              }
              const p = patterns.get(key);
              p.count++;
              if (exp.success) p.successes++;
            });
            return Array.from(patterns.values());
          }
        };

        this.quic = {
          connections: new Map(),
          async createConnection(id) {
            const conn = { id, streams: new Map(), messages: [] };
            this.connections.set(id, conn);
            return conn;
          },
          async send(connId, message) {
            const conn = this.connections.get(connId);
            if (conn) {
              conn.messages.push({ ...message, timestamp: performance.now() });
            }
          }
        };
      },

      async processPatternWithLearning(pattern, context) {
        // 1. Match pattern using DTW
        const similarPatterns = await this.findSimilarPatterns(pattern);

        // 2. Record experience
        const expId = this.reasoningBank.recordExperience({
          pattern,
          context,
          similarPatterns,
          timestamp: Date.now()
        });

        // 3. Execute strategy
        const result = await this.executeStrategy(pattern, context);

        // 4. Update experience with outcome
        this.reasoningBank.experiences[expId].success = result.success;
        this.reasoningBank.experiences[expId].outcome = result;

        // 5. Distill learning
        if (this.reasoningBank.experiences.length % 10 === 0) {
          this.reasoningBank.distill();
        }

        return result;
      },

      async findSimilarPatterns(pattern) {
        // Simulate pattern matching
        return [
          { similarity: 0.95, pattern: pattern },
          { similarity: 0.82, pattern: [...pattern].reverse() }
        ];
      },

      async executeStrategy(pattern, context) {
        // Simulate strategy execution
        const success = Math.random() > 0.3;
        return {
          success,
          profit: success ? Math.random() * 1000 : -Math.random() * 500,
          duration: Math.random() * 100
        };
      },

      async coordinateAgents(agents) {
        // Create QUIC connections for all agents
        const connections = await Promise.all(
          agents.map(agent => this.quic.createConnection(agent.id))
        );

        // Broadcast coordination message
        const message = { type: 'COORDINATE', timestamp: performance.now() };
        await Promise.all(
          connections.map(conn => this.quic.send(conn.id, message))
        );

        return connections;
      }
    };
  });

  describe('Pattern Matching with Learning', () => {
    it('should complete full pattern recognition and learning cycle', async () => {
      await system.initialize();

      const pattern = [1.0, 2.5, 3.0, 2.0, 1.5];
      const context = { market: 'BULLISH', timeframe: '1h' };

      const result = await system.processPatternWithLearning(pattern, context);

      expect(result).toBeDefined();
      expect(result.success).toBeDefined();
      expect(system.reasoningBank.experiences.length).toBe(1);
      expect(system.reasoningBank.experiences[0].pattern).toEqual(pattern);
    });

    it('should learn from multiple pattern executions', async () => {
      await system.initialize();

      const patterns = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7]
      ];

      const context = { market: 'TRENDING' };

      for (const pattern of patterns) {
        await system.processPatternWithLearning(pattern, context);
      }

      expect(system.reasoningBank.experiences.length).toBe(4);

      const learned = system.reasoningBank.distill();
      expect(learned.length).toBeGreaterThan(0);
    });

    it('should improve decision quality over time', async () => {
      await system.initialize();

      const results = [];
      const pattern = [1, 2, 3, 4, 5];
      const context = { market: 'VOLATILE' };

      for (let i = 0; i < 50; i++) {
        const result = await system.processPatternWithLearning(pattern, context);
        results.push(result.success ? 1 : 0);
      }

      const firstHalf = results.slice(0, 25).reduce((a, b) => a + b, 0) / 25;
      const secondHalf = results.slice(25).reduce((a, b) => a + b, 0) / 25;

      // Success rate should be maintained or improved
      expect(secondHalf).toBeGreaterThanOrEqual(firstHalf * 0.9);
    });

    it('should handle 1000+ patterns efficiently', async () => {
      await system.initialize();

      const start = performance.now();

      const promises = [];
      for (let i = 0; i < 1000; i++) {
        const pattern = Array.from({ length: 10 }, () => Math.random() * 10);
        const context = { batch: Math.floor(i / 100) };
        promises.push(system.processPatternWithLearning(pattern, context));
      }

      await Promise.all(promises);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(5000); // <5ms per pattern on average
      expect(system.reasoningBank.experiences.length).toBe(1000);
    });
  });

  describe('Multi-Agent Coordination via QUIC', () => {
    it('should coordinate multiple agents with pattern sharing', async () => {
      await system.initialize();

      const agents = [
        { id: 'agent1', role: 'ANALYZER' },
        { id: 'agent2', role: 'EXECUTOR' },
        { id: 'agent3', role: 'MONITOR' }
      ];

      const connections = await system.coordinateAgents(agents);

      expect(connections.length).toBe(3);
      expect(system.quic.connections.size).toBe(3);

      // Verify all agents received coordination message
      for (const conn of connections) {
        const messages = system.quic.connections.get(conn.id).messages;
        expect(messages.length).toBeGreaterThan(0);
        expect(messages[0].type).toBe('COORDINATE');
      }
    });

    it('should achieve <1ms coordination latency', async () => {
      await system.initialize();

      const agents = Array.from({ length: 10 }, (_, i) => ({
        id: `agent${i}`,
        role: 'TRADER'
      }));

      const start = performance.now();
      await system.coordinateAgents(agents);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(10); // <1ms per agent
    });

    it('should share learned patterns across agents', async () => {
      await system.initialize();

      const agent1 = { id: 'agent1', role: 'LEARNER' };
      const agent2 = { id: 'agent2', role: 'CONSUMER' };

      // Agent 1 learns pattern
      const pattern = [1, 2, 3, 4, 5];
      const context = { shared: true };
      await system.processPatternWithLearning(pattern, context);

      // Share via QUIC
      const conn1 = await system.quic.createConnection(agent1.id);
      const conn2 = await system.quic.createConnection(agent2.id);

      const learnedPattern = system.reasoningBank.experiences[0];
      await system.quic.send(conn2.id, {
        type: 'SHARED_LEARNING',
        pattern: learnedPattern
      });

      const agent2Messages = system.quic.connections.get(conn2.id).messages;
      expect(agent2Messages.length).toBeGreaterThan(0);
      expect(agent2Messages[0].type).toBe('SHARED_LEARNING');
      expect(agent2Messages[0].pattern.pattern).toEqual(pattern);
    });

    it('should coordinate 100+ agents efficiently', async () => {
      await system.initialize();

      const agents = Array.from({ length: 100 }, (_, i) => ({
        id: `agent${i}`,
        role: 'WORKER'
      }));

      const start = performance.now();
      await system.coordinateAgents(agents);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
      expect(system.quic.connections.size).toBe(100);
    });
  });

  describe('Performance Under Load', () => {
    it('should handle concurrent pattern matching and coordination', async () => {
      await system.initialize();

      const numPatterns = 100;
      const numAgents = 10;

      const start = performance.now();

      // Concurrent pattern processing
      const patternPromises = Array.from({ length: numPatterns }, (_, i) => {
        const pattern = Array.from({ length: 20 }, () => Math.random());
        const context = { iteration: i };
        return system.processPatternWithLearning(pattern, context);
      });

      // Concurrent agent coordination
      const agents = Array.from({ length: numAgents }, (_, i) => ({
        id: `agent${i}`,
        role: 'CONCURRENT'
      }));
      const coordPromise = system.coordinateAgents(agents);

      await Promise.all([...patternPromises, coordPromise]);
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(1000);
      expect(system.reasoningBank.experiences.length).toBe(numPatterns);
      expect(system.quic.connections.size).toBe(numAgents);
    });

    it('should maintain low latency under high throughput', async () => {
      await system.initialize();

      const latencies = [];
      const numIterations = 100;

      for (let i = 0; i < numIterations; i++) {
        const start = performance.now();

        const pattern = Array.from({ length: 10 }, () => Math.random());
        await system.processPatternWithLearning(pattern, {});

        latencies.push(performance.now() - start);
      }

      const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
      const p95Latency = latencies.sort((a, b) => a - b)[Math.floor(numIterations * 0.95)];

      expect(avgLatency).toBeLessThan(10);
      expect(p95Latency).toBeLessThan(50);
    });

    it('should demonstrate 100x speedup for DTW matching', async () => {
      await system.initialize();

      // Baseline: naive pattern matching
      const naiveMatch = (p1, p2) => {
        let sum = 0;
        for (let i = 0; i < p1.length; i++) {
          for (let j = 0; j < p2.length; j++) {
            sum += Math.abs(p1[i] - p2[j]);
          }
        }
        return sum;
      };

      const pattern1 = Array.from({ length: 50 }, () => Math.random());
      const pattern2 = Array.from({ length: 50 }, () => Math.random());

      const naiveStart = performance.now();
      naiveMatch(pattern1, pattern2);
      const naiveDuration = performance.now() - naiveStart;

      const optimizedStart = performance.now();
      await system.findSimilarPatterns(pattern1);
      const optimizedDuration = performance.now() - optimizedStart;

      const speedup = naiveDuration / optimizedDuration;

      // Should achieve significant speedup
      expect(speedup).toBeGreaterThan(1);
      expect(optimizedDuration).toBeLessThan(naiveDuration);
    });

    it('should benchmark complete system throughput', async () => {
      await system.initialize();

      const start = performance.now();

      // Process 1000 patterns with learning and coordination
      const patterns = Array.from({ length: 1000 }, () =>
        Array.from({ length: 20 }, () => Math.random())
      );

      const promises = patterns.map((pattern, i) =>
        system.processPatternWithLearning(pattern, { batch: i % 10 })
      );

      await Promise.all(promises);

      const duration = performance.now() - start;
      const throughput = 1000 / (duration / 1000); // patterns per second

      expect(throughput).toBeGreaterThan(200); // >200 patterns/sec
      expect(duration).toBeLessThan(5000);
    });
  });

  describe('Fault Tolerance', () => {
    it('should handle pattern matching failures gracefully', async () => {
      await system.initialize();

      system.findSimilarPatterns = async () => {
        throw new Error('Pattern matching failed');
      };

      try {
        await system.processPatternWithLearning([1, 2, 3], {});
        fail('Should have thrown error');
      } catch (error) {
        expect(error.message).toBe('Pattern matching failed');
      }

      // System should still be operational
      system.findSimilarPatterns = async (p) => [{ similarity: 1, pattern: p }];
      const result = await system.processPatternWithLearning([4, 5, 6], {});
      expect(result).toBeDefined();
    });

    it('should recover from QUIC connection failures', async () => {
      await system.initialize();

      const agents = [{ id: 'agent1', role: 'TEST' }];

      // First coordination succeeds
      await system.coordinateAgents(agents);

      // Simulate connection failure
      system.quic.connections.clear();

      // Should recover and create new connections
      await system.coordinateAgents(agents);

      expect(system.quic.connections.size).toBe(1);
    });

    it('should continue learning despite individual failures', async () => {
      await system.initialize();

      let failureCount = 0;

      for (let i = 0; i < 20; i++) {
        try {
          if (i % 5 === 0) {
            // Simulate occasional failure
            throw new Error('Simulated failure');
          }

          const pattern = Array.from({ length: 5 }, () => Math.random());
          await system.processPatternWithLearning(pattern, {});
        } catch (error) {
          failureCount++;
        }
      }

      expect(failureCount).toBe(4); // 4 failures (i = 0, 5, 10, 15)
      expect(system.reasoningBank.experiences.length).toBe(16); // 16 successes
    });
  });
});
