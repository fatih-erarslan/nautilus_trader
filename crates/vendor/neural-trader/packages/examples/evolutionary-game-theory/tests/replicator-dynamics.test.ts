/**
 * Tests for replicator dynamics
 */

import {
  ReplicatorDynamics,
  MultiPopulationDynamics,
} from '../src/replicator-dynamics.js';
import {
  PRISONERS_DILEMMA,
  HAWK_DOVE,
  ROCK_PAPER_SCISSORS,
} from '../src/games.js';

describe('ReplicatorDynamics', () => {
  describe('Initialization', () => {
    it('should initialize with uniform distribution', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      const state = dynamics.getState();

      expect(state.frequencies).toHaveLength(2);
      expect(state.frequencies[0]).toBeCloseTo(0.5);
      expect(state.frequencies[1]).toBeCloseTo(0.5);
      expect(state.generation).toBe(0);
    });

    it('should normalize initial population', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [3, 7]);
      const state = dynamics.getState();

      expect(state.frequencies[0]).toBeCloseTo(0.3);
      expect(state.frequencies[1]).toBeCloseTo(0.7);
    });

    it('should ensure frequencies sum to 1', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [1, 4]);
      const state = dynamics.getState();
      const sum = state.frequencies.reduce((a, b) => a + b, 0);

      expect(sum).toBeCloseTo(1.0);
    });
  });

  describe('Single Step Evolution', () => {
    it('should increase frequency of fitter strategy', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const initialState = dynamics.getState();

      dynamics.step(0.1);
      const newState = dynamics.getState();

      // Defect (strategy 1) is fitter, should increase
      expect(newState.frequencies[1]).toBeGreaterThan(initialState.frequencies[1]);
    });

    it('should preserve total frequency', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.3, 0.7]);
      dynamics.step(0.1);
      const state = dynamics.getState();
      const sum = state.frequencies.reduce((a, b) => a + b, 0);

      expect(sum).toBeCloseTo(1.0);
    });

    it('should record fitness values', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      dynamics.step(0.1);
      const state = dynamics.getState();

      expect(state.fitnessValues).toBeDefined();
      expect(state.fitnessValues).toHaveLength(2);
      expect(state.averageFitness).toBeDefined();
    });

    it('should increment generation counter', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      dynamics.step();
      dynamics.step();

      expect(dynamics.getState().generation).toBe(2);
    });
  });

  describe('Multi-step Simulation', () => {
    it('should simulate multiple steps', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const results = dynamics.simulate(10);

      expect(results).toHaveLength(10);
      expect(results[9].generation).toBe(10);
    });

    it('should converge to dominant strategy', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      dynamics.simulate(100, 0.1);
      const finalState = dynamics.getState();

      // Should converge to all-defect
      expect(finalState.frequencies[1]).toBeGreaterThan(0.9);
    });

    it('should maintain cyclic dynamics in RPS', () => {
      const dynamics = new ReplicatorDynamics(ROCK_PAPER_SCISSORS, [0.4, 0.3, 0.3]);
      dynamics.simulate(100, 0.01);
      const finalState = dynamics.getState();

      // Should not converge to a single strategy
      expect(finalState.frequencies.every((f) => f > 0.1)).toBe(true);
    });
  });

  describe('Convergence Detection', () => {
    it('should detect convergence', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const finalState = dynamics.simulateUntilConvergence(1e-4, 1000, 0.1);

      expect(finalState.frequencies[1]).toBeGreaterThan(0.95);
    });

    it('should identify fixed points', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0, 1]);
      const isFixed = dynamics.isFixedPoint(1e-6);

      expect(isFixed).toBe(true);
    });

    it('should detect non-fixed points', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const isFixed = dynamics.isFixedPoint(1e-6);

      expect(isFixed).toBe(false);
    });
  });

  describe('Diversity Calculation', () => {
    it('should calculate diversity', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const diversity = dynamics.calculateDiversity();

      expect(diversity).toBeGreaterThan(0);
      expect(diversity).toBeCloseTo(Math.log(2)); // Maximum for 2 strategies
    });

    it('should return 0 for homogeneous population', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [1, 0]);
      const diversity = dynamics.calculateDiversity();

      expect(diversity).toBeCloseTo(0);
    });

    it('should decrease as population homogenizes', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA, [0.5, 0.5]);
      const initialDiversity = dynamics.calculateDiversity();

      dynamics.simulate(50, 0.1);
      const finalDiversity = dynamics.calculateDiversity();

      expect(finalDiversity).toBeLessThan(initialDiversity);
    });
  });

  describe('History Tracking', () => {
    it('should track simulation history', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      dynamics.simulate(10);
      const history = dynamics.getHistory();

      expect(history).toHaveLength(11); // Initial + 10 steps
    });

    it('should reset history on reset', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      dynamics.simulate(10);
      dynamics.reset();
      const history = dynamics.getHistory();

      expect(history).toHaveLength(1); // Only initial state
    });
  });

  describe('Phase Portrait', () => {
    it('should calculate velocity vectors', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      const velocity = dynamics.getVelocity([0.5, 0.5]);

      expect(velocity).toHaveLength(2);
      expect(velocity[0]).toBeLessThan(0); // Cooperate frequency decreasing
      expect(velocity[1]).toBeGreaterThan(0); // Defect frequency increasing
    });

    it('should have zero velocity at fixed point', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      const velocity = dynamics.getVelocity([0, 1]);

      expect(Math.abs(velocity[0])).toBeCloseTo(0);
      expect(Math.abs(velocity[1])).toBeCloseTo(0);
    });
  });

  describe('Export Functionality', () => {
    it('should export visualization data', () => {
      const dynamics = new ReplicatorDynamics(PRISONERS_DILEMMA);
      dynamics.simulate(5);
      const exported = dynamics.exportForVisualization();

      expect(exported.game).toBe(PRISONERS_DILEMMA);
      expect(exported.states.length).toBeGreaterThan(0);
    });
  });
});

describe('MultiPopulationDynamics', () => {
  describe('Multiple Populations', () => {
    it('should initialize multiple populations', () => {
      const multi = new MultiPopulationDynamics([
        PRISONERS_DILEMMA,
        HAWK_DOVE,
      ]);

      const states = multi.getStates();
      expect(states).toHaveLength(2);
    });

    it('should evolve all populations', () => {
      const multi = new MultiPopulationDynamics([
        PRISONERS_DILEMMA,
        PRISONERS_DILEMMA,
      ]);

      const initialStates = multi.getStates();
      multi.step(0.1);
      const newStates = multi.getStates();

      expect(newStates[0].generation).toBeGreaterThan(initialStates[0].generation);
      expect(newStates[1].generation).toBeGreaterThan(initialStates[1].generation);
    });

    it('should calculate cross-population diversity', () => {
      const multi = new MultiPopulationDynamics([
        PRISONERS_DILEMMA,
        HAWK_DOVE,
      ]);

      const diversity = multi.calculateCrossDiversity();
      expect(diversity).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Coevolution', () => {
    it('should allow populations to coevolve', () => {
      const multi = new MultiPopulationDynamics([
        PRISONERS_DILEMMA,
        PRISONERS_DILEMMA,
      ]);

      const results = multi.simulate(10, 0.1);

      expect(results).toHaveLength(2);
      expect(results[0]).toHaveLength(10);
      expect(results[1]).toHaveLength(10);
    });
  });
});
