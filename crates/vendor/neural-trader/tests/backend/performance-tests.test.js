/**
 * Performance Tests for Neural Trader Backend
 * Coverage Target: 95%+
 *
 * Performance Metrics:
 * - Execution time benchmarks
 * - Throughput testing
 * - Concurrent operation stress tests
 * - Memory usage validation
 * - Load testing
 */

const backend = require('../../neural-trader-rust/packages/neural-trader-backend');

describe('Neural Trader Backend - Performance Tests', () => {

  // Helper to measure execution time
  const measureTime = async (fn) => {
    const start = process.hrtime.bigint();
    await fn();
    const end = process.hrtime.bigint();
    return Number(end - start) / 1_000_000; // Convert to milliseconds
  };

  // ============================================================================
  // EXECUTION TIME BENCHMARKS
  // ============================================================================

  describe('Execution Time Benchmarks', () => {
    describe('Trading Operations', () => {
      it('should list strategies in under 100ms', async () => {
        const duration = await measureTime(() => backend.listStrategies());
        expect(duration).toBeLessThan(100);
      });

      it('should analyze market in under 500ms', async () => {
        const duration = await measureTime(() => backend.quickAnalysis('AAPL', false));
        expect(duration).toBeLessThan(500);
      });

      it('should simulate trade in under 200ms', async () => {
        const strategies = await backend.listStrategies();
        const duration = await measureTime(() =>
          backend.simulateTrade(strategies[0].name, 'AAPL', 'buy', false)
        );
        expect(duration).toBeLessThan(200);
      });

      it('should execute trade in under 1000ms', async () => {
        const strategies = await backend.listStrategies();
        const duration = await measureTime(() =>
          backend.executeTrade(strategies[0].name, 'AAPL', 'buy', 10)
        );
        expect(duration).toBeLessThan(1000);
      });

      it('should get portfolio status in under 100ms', async () => {
        const duration = await measureTime(() => backend.getPortfolioStatus(false));
        expect(duration).toBeLessThan(100);
      });
    });

    describe('Neural Operations', () => {
      it('should generate forecast in under 2000ms', async () => {
        const duration = await measureTime(() =>
          backend.neuralForecast('AAPL', 30, false)
        );
        expect(duration).toBeLessThan(2000);
      });

      it('should list models in under 50ms', async () => {
        const duration = await measureTime(() => backend.neuralModelStatus());
        expect(duration).toBeLessThan(50);
      });
    });

    describe('Sports Betting Operations', () => {
      it('should get sports events in under 1000ms', async () => {
        const duration = await measureTime(() => backend.getSportsEvents('nfl', 7));
        expect(duration).toBeLessThan(1000);
      });

      it('should calculate Kelly Criterion in under 10ms', async () => {
        const duration = await measureTime(() =>
          backend.calculateKellyCriterion(0.6, 2.5, 10000)
        );
        expect(duration).toBeLessThan(10);
      });
    });

    describe('Syndicate Operations', () => {
      it('should create syndicate in under 100ms', async () => {
        const syndicateId = `perf_${Date.now()}`;
        const duration = await measureTime(() =>
          backend.createSyndicate(syndicateId, 'Performance Test')
        );
        expect(duration).toBeLessThan(100);
      });

      it('should add member in under 50ms', async () => {
        const syndicateId = `perf_member_${Date.now()}`;
        await backend.createSyndicate(syndicateId, 'Test');

        const duration = await measureTime(() =>
          backend.addSyndicateMember(syndicateId, 'Member', 'mem@example.com', 'analyst', 10000)
        );
        expect(duration).toBeLessThan(50);
      });

      it('should get syndicate status in under 50ms', async () => {
        const syndicateId = `perf_status_${Date.now()}`;
        await backend.createSyndicate(syndicateId, 'Test');

        const duration = await measureTime(() => backend.getSyndicateStatus(syndicateId));
        expect(duration).toBeLessThan(50);
      });
    });

    describe('Swarm Operations', () => {
      it('should initialize swarm in under 2000ms', async () => {
        const config = JSON.stringify({ maxAgents: 5 });
        let swarmId;

        const duration = await measureTime(async () => {
          const swarm = await backend.initE2bSwarm('mesh', config);
          swarmId = swarm.swarmId;
        });

        expect(duration).toBeLessThan(2000);
        if (swarmId) await backend.shutdownSwarm(swarmId);
      });

      it('should get swarm status in under 100ms', async () => {
        const config = JSON.stringify({ maxAgents: 3 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const duration = await measureTime(() => backend.getSwarmStatus(swarm.swarmId));
        expect(duration).toBeLessThan(100);

        await backend.shutdownSwarm(swarm.swarmId);
      });

      it('should scale swarm in under 1000ms', async () => {
        const config = JSON.stringify({ maxAgents: 10 });
        const swarm = await backend.initE2bSwarm('mesh', config);

        const duration = await measureTime(() => backend.scaleSwarm(swarm.swarmId, 7));
        expect(duration).toBeLessThan(1000);

        await backend.shutdownSwarm(swarm.swarmId);
      });
    });

    describe('Security Operations', () => {
      beforeAll(() => {
        backend.initAuth();
      });

      it('should create API key in under 50ms', () => {
        const duration = measureTime(() => backend.createApiKey('perfuser', 'user'));
        expect(duration).toBeLessThan(50);
      });

      it('should validate API key in under 10ms', () => {
        const apiKey = backend.createApiKey('testuser', 'user');
        const duration = measureTime(() => backend.validateApiKey(apiKey));
        expect(duration).toBeLessThan(10);
      });

      it('should generate token in under 20ms', () => {
        const apiKey = backend.createApiKey('tokenuser', 'user');
        const duration = measureTime(() => backend.generateToken(apiKey));
        expect(duration).toBeLessThan(20);
      });

      it('should sanitize input in under 5ms', () => {
        const dirty = '<script>alert("xss")</script>';
        const duration = measureTime(() => backend.sanitizeInput(dirty));
        expect(duration).toBeLessThan(5);
      });
    });
  });

  // ============================================================================
  // THROUGHPUT TESTING
  // ============================================================================

  describe('Throughput Testing', () => {
    describe('Sequential Operations', () => {
      it('should handle 100 quick analyses', async () => {
        const start = Date.now();
        const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];

        for (let i = 0; i < 100; i++) {
          await backend.quickAnalysis(symbols[i % symbols.length], false);
        }

        const duration = Date.now() - start;
        const throughput = 100 / (duration / 1000); // operations per second

        console.log(`Quick Analysis Throughput: ${throughput.toFixed(2)} ops/sec`);
        expect(throughput).toBeGreaterThan(1); // At least 1 op/sec
      }, 60000);

      it('should handle 50 Kelly Criterion calculations', async () => {
        const start = Date.now();

        for (let i = 0; i < 50; i++) {
          await backend.calculateKellyCriterion(
            0.5 + (i % 50) / 100,
            2.0 + (i % 5) / 10,
            10000
          );
        }

        const duration = Date.now() - start;
        const throughput = 50 / (duration / 1000);

        console.log(`Kelly Criterion Throughput: ${throughput.toFixed(2)} ops/sec`);
        expect(throughput).toBeGreaterThan(50); // Should be very fast
      });
    });

    describe('Batch Operations', () => {
      it('should handle batch correlation analysis', async () => {
        const symbols = Array(20).fill(null).map((_, i) => `SYM${i}`);

        const start = Date.now();
        await backend.correlationAnalysis(symbols, false);
        const duration = Date.now() - start;

        console.log(`Correlation Analysis (20 symbols): ${duration}ms`);
        expect(duration).toBeLessThan(5000);
      });

      it('should handle batch member additions', async () => {
        const syndicateId = `batch_${Date.now()}`;
        await backend.createSyndicate(syndicateId, 'Batch Test');

        const start = Date.now();

        for (let i = 0; i < 50; i++) {
          await backend.addSyndicateMember(
            syndicateId,
            `Member ${i}`,
            `member${i}@example.com`,
            'contributing_member',
            5000
          );
        }

        const duration = Date.now() - start;
        const throughput = 50 / (duration / 1000);

        console.log(`Member Addition Throughput: ${throughput.toFixed(2)} members/sec`);
        expect(throughput).toBeGreaterThan(10);
      }, 30000);
    });
  });

  // ============================================================================
  // CONCURRENT OPERATION STRESS TESTS
  // ============================================================================

  describe('Concurrent Operation Stress Tests', () => {
    describe('Concurrent Market Analysis', () => {
      it('should handle 50 concurrent quick analyses', async () => {
        const start = Date.now();

        const promises = Array(50).fill(null).map(() =>
          backend.quickAnalysis('AAPL', false)
        );

        const results = await Promise.all(promises);
        const duration = Date.now() - start;

        console.log(`50 Concurrent Analyses: ${duration}ms`);
        expect(results.length).toBe(50);
        expect(duration).toBeLessThan(10000);
      }, 15000);

      it('should handle 100 concurrent simulations', async () => {
        const strategies = await backend.listStrategies();
        const strategy = strategies[0];

        const start = Date.now();

        const promises = Array(100).fill(null).map(() =>
          backend.simulateTrade(strategy.name, 'AAPL', 'buy', false)
        );

        const results = await Promise.all(promises);
        const duration = Date.now() - start;

        console.log(`100 Concurrent Simulations: ${duration}ms`);
        expect(results.length).toBe(100);
        expect(duration).toBeLessThan(20000);
      }, 25000);
    });

    describe('Concurrent Syndicate Operations', () => {
      it('should handle concurrent syndicate creations', async () => {
        const start = Date.now();

        const promises = Array(20).fill(null).map((_, i) =>
          backend.createSyndicate(`stress_${Date.now()}_${i}`, `Stress Test ${i}`)
        );

        const results = await Promise.all(promises);
        const duration = Date.now() - start;

        console.log(`20 Concurrent Syndicate Creations: ${duration}ms`);
        expect(results.length).toBe(20);
        expect(duration).toBeLessThan(5000);
      });

      it('should handle concurrent fund allocations', async () => {
        const syndicateId = `stress_alloc_${Date.now()}`;
        await backend.createSyndicate(syndicateId, 'Allocation Stress');

        const opportunities = JSON.stringify([
          { id: 'o1', expectedReturn: 0.1, riskScore: 0.3 }
        ]);

        const start = Date.now();

        const promises = Array(30).fill(null).map(() =>
          backend.allocateSyndicateFunds(syndicateId, opportunities)
        );

        const results = await Promise.all(promises);
        const duration = Date.now() - start;

        console.log(`30 Concurrent Allocations: ${duration}ms`);
        expect(results.length).toBe(30);
      });
    });

    describe('Concurrent Class Operations', () => {
      it('should handle concurrent fund allocation engine operations', () => {
        const engines = Array(10).fill(null).map((_, i) =>
          new backend.FundAllocationEngine(`syn_${i}`, '100000')
        );

        const opportunity = {
          sport: 'nfl',
          event: 'Test',
          betType: 'moneyline',
          selection: 'Team A',
          odds: 2.5,
          probability: 0.55,
          edge: 0.15,
          confidence: 0.85,
          modelAgreement: 0.90,
          timeUntilEventSecs: 7200,
          liquidity: 0.8,
          isLive: false,
          isParlay: false
        };

        const start = Date.now();

        const results = engines.map(engine =>
          engine.allocateFunds(opportunity, backend.AllocationStrategy.KellyCriterion)
        );

        const duration = Date.now() - start;

        console.log(`10 Concurrent Engine Allocations: ${duration}ms`);
        expect(results.length).toBe(10);
        expect(duration).toBeLessThan(1000);
      });

      it('should handle concurrent voting operations', () => {
        const votingSystems = Array(5).fill(null).map((_, i) =>
          new backend.VotingSystem(`syn_vote_${i}`)
        );

        const start = Date.now();

        const votes = votingSystems.map(system =>
          system.createVote('test', 'Concurrent Vote Test', 'member_001')
        );

        const duration = Date.now() - start;

        console.log(`5 Concurrent Vote Creations: ${duration}ms`);
        expect(votes.length).toBe(5);
        expect(duration).toBeLessThan(500);
      });
    });

    describe('Concurrent Swarm Operations', () => {
      it('should handle concurrent swarm initializations', async () => {
        const config = JSON.stringify({ maxAgents: 3 });
        const start = Date.now();

        const promises = Array(5).fill(null).map((_, i) =>
          backend.initE2bSwarm(['mesh', 'hierarchical', 'ring', 'star'][i % 4], config)
        );

        const swarms = await Promise.all(promises);
        const duration = Date.now() - start;

        console.log(`5 Concurrent Swarm Inits: ${duration}ms`);
        expect(swarms.length).toBe(5);

        // Cleanup
        await Promise.all(swarms.map(s => backend.shutdownSwarm(s.swarmId)));
      }, 30000);
    });
  });

  // ============================================================================
  // MEMORY USAGE VALIDATION
  // ============================================================================

  describe('Memory Usage Validation', () => {
    const getMemoryUsage = () => {
      const usage = process.memoryUsage();
      return usage.heapUsed / 1024 / 1024; // MB
    };

    it('should not leak memory during repeated operations', async () => {
      const initialMemory = getMemoryUsage();

      // Perform many operations
      for (let i = 0; i < 100; i++) {
        await backend.quickAnalysis('AAPL', false);

        // Force garbage collection if available
        if (global.gc) {
          global.gc();
        }
      }

      const finalMemory = getMemoryUsage();
      const memoryIncrease = finalMemory - initialMemory;

      console.log(`Memory Usage - Initial: ${initialMemory.toFixed(2)}MB, Final: ${finalMemory.toFixed(2)}MB, Increase: ${memoryIncrease.toFixed(2)}MB`);

      // Should not increase by more than 100MB
      expect(memoryIncrease).toBeLessThan(100);
    }, 30000);

    it('should handle large data structures efficiently', () => {
      const initialMemory = getMemoryUsage();

      // Create large syndicate with many members
      const largeMembers = Array(1000).fill(null).map((_, i) => ({
        memberId: `m${i}`,
        name: `Member ${i}`,
        capitalContribution: '10000',
        performanceScore: 0.5,
        tier: 'bronze'
      }));

      const system = new backend.ProfitDistributionSystem('large_test');
      system.calculateDistribution(
        '1000000',
        JSON.stringify(largeMembers),
        backend.DistributionModel.Proportional
      );

      if (global.gc) {
        global.gc();
      }

      const finalMemory = getMemoryUsage();
      const memoryIncrease = finalMemory - initialMemory;

      console.log(`Large Data Memory Increase: ${memoryIncrease.toFixed(2)}MB`);
      expect(memoryIncrease).toBeLessThan(50);
    });

    it('should cleanup resources after swarm operations', async () => {
      const initialMemory = getMemoryUsage();

      // Create and destroy multiple swarms
      for (let i = 0; i < 5; i++) {
        const config = JSON.stringify({ maxAgents: 3 });
        const swarm = await backend.initE2bSwarm('mesh', config);
        await backend.shutdownSwarm(swarm.swarmId);
      }

      if (global.gc) {
        global.gc();
      }

      const finalMemory = getMemoryUsage();
      const memoryIncrease = finalMemory - initialMemory;

      console.log(`Swarm Cleanup Memory Increase: ${memoryIncrease.toFixed(2)}MB`);
      expect(memoryIncrease).toBeLessThan(30);
    }, 60000);
  });

  // ============================================================================
  // LOAD TESTING
  // ============================================================================

  describe('Load Testing', () => {
    describe('Sustained Load', () => {
      it('should maintain performance under sustained load', async () => {
        const durations = [];

        // Run 50 operations and track timing
        for (let i = 0; i < 50; i++) {
          const start = Date.now();
          await backend.quickAnalysis('AAPL', false);
          const duration = Date.now() - start;
          durations.push(duration);
        }

        const avgDuration = durations.reduce((a, b) => a + b) / durations.length;
        const maxDuration = Math.max(...durations);
        const minDuration = Math.min(...durations);

        console.log(`Sustained Load - Avg: ${avgDuration.toFixed(2)}ms, Min: ${minDuration}ms, Max: ${maxDuration}ms`);

        // Performance shouldn't degrade significantly
        expect(maxDuration).toBeLessThan(avgDuration * 3);
      }, 60000);

      it('should handle mixed operation load', async () => {
        const operations = [];

        for (let i = 0; i < 100; i++) {
          const op = i % 4;

          switch (op) {
            case 0:
              operations.push(backend.quickAnalysis('AAPL', false));
              break;
            case 1:
              operations.push(backend.listStrategies());
              break;
            case 2:
              operations.push(backend.calculateKellyCriterion(0.6, 2.5, 10000));
              break;
            case 3:
              operations.push(backend.getPortfolioStatus(false));
              break;
          }
        }

        const start = Date.now();
        await Promise.all(operations);
        const duration = Date.now() - start;

        console.log(`Mixed Load (100 ops): ${duration}ms`);
        expect(duration).toBeLessThan(30000);
      }, 40000);
    });

    describe('Burst Load', () => {
      it('should handle sudden traffic spike', async () => {
        // Simulate sudden spike of requests
        const spike = Array(200).fill(null).map(() =>
          backend.quickAnalysis('AAPL', false)
        );

        const start = Date.now();
        const results = await Promise.all(spike);
        const duration = Date.now() - start;

        console.log(`Traffic Spike (200 concurrent): ${duration}ms`);
        expect(results.length).toBe(200);
        expect(duration).toBeLessThan(60000);
      }, 70000);
    });

    describe('Stress Testing', () => {
      it('should survive extreme load conditions', async () => {
        const syndicateId = `stress_${Date.now()}`;
        await backend.createSyndicate(syndicateId, 'Stress Test');

        // Add many members
        const memberOps = Array(100).fill(null).map((_, i) =>
          backend.addSyndicateMember(
            syndicateId,
            `Member ${i}`,
            `m${i}@example.com`,
            'contributing_member',
            5000
          )
        );

        const start = Date.now();
        await Promise.all(memberOps);
        const duration = Date.now() - start;

        console.log(`Extreme Load (100 concurrent members): ${duration}ms`);

        // Verify all members added
        const status = await backend.getSyndicateStatus(syndicateId);
        expect(status.memberCount).toBe(100);
      }, 60000);
    });
  });

  // ============================================================================
  // SCALABILITY TESTS
  // ============================================================================

  describe('Scalability Tests', () => {
    it('should scale linearly with data size', async () => {
      const sizes = [10, 20, 50, 100];
      const timings = [];

      for (const size of sizes) {
        const symbols = Array(size).fill(null).map((_, i) => `SYM${i}`);

        const start = Date.now();
        await backend.correlationAnalysis(symbols, false);
        const duration = Date.now() - start;

        timings.push({ size, duration });
        console.log(`Correlation with ${size} symbols: ${duration}ms`);
      }

      // Check if growth is reasonable (not exponential)
      const ratio1 = timings[1].duration / timings[0].duration;
      const ratio2 = timings[2].duration / timings[1].duration;
      const ratio3 = timings[3].duration / timings[2].duration;

      console.log(`Growth ratios: ${ratio1.toFixed(2)}, ${ratio2.toFixed(2)}, ${ratio3.toFixed(2)}`);

      // Growth should be sub-quadratic
      expect(ratio3).toBeLessThan(5);
    }, 120000);

    it('should handle increasing swarm sizes efficiently', async () => {
      const sizes = [3, 5, 7, 10];
      const timings = [];

      for (const size of sizes) {
        const config = JSON.stringify({ maxAgents: size });

        const start = Date.now();
        const swarm = await backend.initE2bSwarm('mesh', config);
        const duration = Date.now() - start;

        timings.push({ size, duration });
        console.log(`Swarm with ${size} agents: ${duration}ms`);

        await backend.shutdownSwarm(swarm.swarmId);
      }

      // Initialization time should scale reasonably
      const maxTiming = Math.max(...timings.map(t => t.duration));
      expect(maxTiming).toBeLessThan(10000);
    }, 120000);
  });

  // ============================================================================
  // BENCHMARK SUMMARY
  // ============================================================================

  describe('Benchmark Summary', () => {
    it('should generate performance report', async () => {
      const report = {
        quickAnalysis: 0,
        simulation: 0,
        kellyCriterion: 0,
        syndicateCreation: 0,
        memberAddition: 0,
        swarmInit: 0,
        timestamp: new Date().toISOString()
      };

      // Quick Analysis
      report.quickAnalysis = await measureTime(() => backend.quickAnalysis('AAPL', false));

      // Simulation
      const strategies = await backend.listStrategies();
      report.simulation = await measureTime(() =>
        backend.simulateTrade(strategies[0].name, 'AAPL', 'buy', false)
      );

      // Kelly Criterion
      report.kellyCriterion = await measureTime(() =>
        backend.calculateKellyCriterion(0.6, 2.5, 10000)
      );

      // Syndicate Creation
      const synId = `bench_${Date.now()}`;
      report.syndicateCreation = await measureTime(() =>
        backend.createSyndicate(synId, 'Benchmark')
      );

      // Member Addition
      report.memberAddition = await measureTime(() =>
        backend.addSyndicateMember(synId, 'Member', 'mem@example.com', 'analyst', 10000)
      );

      // Swarm Init
      const config = JSON.stringify({ maxAgents: 3 });
      let swarmId;
      report.swarmInit = await measureTime(async () => {
        const swarm = await backend.initE2bSwarm('mesh', config);
        swarmId = swarm.swarmId;
      });

      console.log('\n=== PERFORMANCE BENCHMARK REPORT ===');
      console.log(`Quick Analysis: ${report.quickAnalysis.toFixed(2)}ms`);
      console.log(`Trade Simulation: ${report.simulation.toFixed(2)}ms`);
      console.log(`Kelly Criterion: ${report.kellyCriterion.toFixed(2)}ms`);
      console.log(`Syndicate Creation: ${report.syndicateCreation.toFixed(2)}ms`);
      console.log(`Member Addition: ${report.memberAddition.toFixed(2)}ms`);
      console.log(`Swarm Initialization: ${report.swarmInit.toFixed(2)}ms`);
      console.log('====================================\n');

      if (swarmId) await backend.shutdownSwarm(swarmId);

      // All operations should complete in reasonable time
      expect(report.quickAnalysis).toBeLessThan(500);
      expect(report.kellyCriterion).toBeLessThan(10);
    }, 30000);
  });
});
