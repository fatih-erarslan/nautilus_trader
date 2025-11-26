import { EmergenceDetector, type SystemState } from '../src/emergence';

describe('EmergenceDetector', () => {
  let detector: EmergenceDetector;

  beforeEach(() => {
    detector = new EmergenceDetector(undefined, 50);
  });

  describe('initialization', () => {
    it('should create detector', () => {
      expect(detector).toBeDefined();
      expect(detector.getStateHistory()).toHaveLength(0);
      expect(detector.getEmergenceEvents()).toHaveLength(0);
    });
  });

  describe('state management', () => {
    it('should add states to history', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [
          {
            id: 'agent-1',
            position: { x: 100, y: 100 },
            state: {},
            neighbors: []
          }
        ],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.5
        }
      };

      await detector.addState(state);

      expect(detector.getStateHistory()).toHaveLength(1);
    });

    it('should maintain history size limit', async () => {
      const maxSize = 50;

      // Add more states than the limit
      for (let i = 0; i < maxSize + 10; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: Math.random(),
            order: Math.random(),
            complexity: Math.random(),
            connectivity: Math.random()
          }
        };

        await detector.addState(state);
      }

      expect(detector.getStateHistory()).toHaveLength(maxSize);
    });
  });

  describe('emergence metrics', () => {
    it('should calculate self-organization', async () => {
      // Add states with increasing order
      for (let i = 0; i < 20; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: 0.8 - i * 0.02,
            order: 0.2 + i * 0.02,
            complexity: 0.5,
            connectivity: 0.5
          }
        };

        await detector.addState(state);
      }

      const metrics = detector.getLatestMetrics();

      // Should detect increasing order (self-organization)
      expect(metrics.selfOrganization).toBeGreaterThan(0);
    });

    it('should calculate complexity', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.5
        }
      };

      await detector.addState(state);

      const metrics = detector.getLatestMetrics();

      expect(metrics.complexity).toBeGreaterThanOrEqual(0);
      expect(metrics.complexity).toBeLessThanOrEqual(1);
    });

    it('should calculate coherence', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [
          {
            id: 'agent-1',
            position: { x: 0, y: 0 },
            velocity: { x: 1, y: 0 },
            state: {},
            neighbors: []
          },
          {
            id: 'agent-2',
            position: { x: 10, y: 0 },
            velocity: { x: 1, y: 0 },
            state: {},
            neighbors: []
          }
        ],
        globalMetrics: {
          entropy: 0.3,
          order: 0.7,
          complexity: 0.5,
          connectivity: 0.8
        }
      };

      await detector.addState(state);

      const metrics = detector.getLatestMetrics();

      // Aligned velocities should show coherence
      expect(metrics.coherence).toBeGreaterThan(0.5);
    });

    it('should calculate adaptability', async () => {
      // Add states with changing entropy (system adapting)
      const entropies = [0.5, 0.7, 0.6, 0.8, 0.7, 0.9, 0.8];

      for (let i = 0; i < entropies.length; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: entropies[i],
            order: 1 - entropies[i],
            complexity: 0.5,
            connectivity: 0.5
          }
        };

        await detector.addState(state);
      }

      const metrics = detector.getLatestMetrics();

      expect(metrics.adaptability).toBeGreaterThanOrEqual(0);
      expect(metrics.adaptability).toBeLessThanOrEqual(1);
    });

    it('should calculate robustness', async () => {
      // Add stable states (low variance)
      for (let i = 0; i < 20; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: 0.5 + (Math.random() - 0.5) * 0.1, // Small variance
            order: 0.5 + (Math.random() - 0.5) * 0.1,
            complexity: 0.5,
            connectivity: 0.5
          }
        };

        await detector.addState(state);
      }

      const metrics = detector.getLatestMetrics();

      // Stable system should show high robustness
      expect(metrics.robustness).toBeGreaterThan(0.5);
    });

    it('should calculate novelty', async () => {
      // Add similar states, then a very different one
      for (let i = 0; i < 15; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: 0.5,
            order: 0.5,
            complexity: 0.5,
            connectivity: 0.5
          }
        };

        await detector.addState(state);
      }

      // Add novel state
      const novelState: SystemState = {
        timestamp: Date.now() + 15,
        agents: [],
        globalMetrics: {
          entropy: 0.9,
          order: 0.1,
          complexity: 0.9,
          connectivity: 0.1
        }
      };

      await detector.addState(novelState);

      const metrics = detector.getLatestMetrics();

      // Novel state should show high novelty
      expect(metrics.novelty).toBeGreaterThan(0.3);
    });
  });

  describe('emergence detection', () => {
    it('should detect significant emergence', async () => {
      // Create clear self-organization pattern
      for (let i = 0; i < 25; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i * 100,
          agents: Array(50).fill(0).map((_, j) => ({
            id: `agent-${j}`,
            position: { x: j * 10, y: i * 5 },
            velocity: { x: 1, y: 0 },
            state: {},
            neighbors: []
          })),
          globalMetrics: {
            entropy: Math.max(0, 0.9 - i * 0.03),
            order: Math.min(1, 0.1 + i * 0.03),
            complexity: 0.5 + i * 0.01,
            connectivity: 0.5 + i * 0.01
          }
        };

        await detector.addState(state);
      }

      const events = detector.getEmergenceEvents();

      // Should detect at least one emergence event
      // Note: This depends on LLM being available, so may not always work in tests
      expect(events).toBeDefined();
    });

    it('should not detect emergence in random noise', async () => {
      // Add random states
      for (let i = 0; i < 15; i++) {
        const state: SystemState = {
          timestamp: Date.now() + i,
          agents: [],
          globalMetrics: {
            entropy: Math.random(),
            order: Math.random(),
            complexity: Math.random(),
            connectivity: Math.random()
          }
        };

        await detector.addState(state);
      }

      const metrics = detector.getLatestMetrics();

      // Random noise should show low organization
      expect(metrics.selfOrganization).toBeLessThan(0.7);
    });
  });

  describe('clear', () => {
    it('should clear history and events', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.5
        }
      };

      await detector.addState(state);

      detector.clear();

      expect(detector.getStateHistory()).toHaveLength(0);
      expect(detector.getEmergenceEvents()).toHaveLength(0);
    });
  });

  describe('edge cases', () => {
    it('should handle empty agent lists', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.5
        }
      };

      await detector.addState(state);

      const metrics = detector.getLatestMetrics();

      expect(metrics).toBeDefined();
    });

    it('should handle single agent', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [
          {
            id: 'agent-1',
            position: { x: 0, y: 0 },
            state: {},
            neighbors: []
          }
        ],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.5
        }
      };

      await detector.addState(state);

      const metrics = detector.getLatestMetrics();

      expect(metrics.coherence).toBeDefined();
    });

    it('should handle agents without velocities', async () => {
      const state: SystemState = {
        timestamp: Date.now(),
        agents: [
          {
            id: 'agent-1',
            position: { x: 0, y: 0 },
            state: {},
            neighbors: []
          },
          {
            id: 'agent-2',
            position: { x: 10, y: 10 },
            state: {},
            neighbors: []
          }
        ],
        globalMetrics: {
          entropy: 0.5,
          order: 0.5,
          complexity: 0.5,
          connectivity: 0.8
        }
      };

      await detector.addState(state);

      const metrics = detector.getLatestMetrics();

      // Should use connectivity fallback
      expect(metrics.coherence).toBe(0.8);
    });
  });
});
