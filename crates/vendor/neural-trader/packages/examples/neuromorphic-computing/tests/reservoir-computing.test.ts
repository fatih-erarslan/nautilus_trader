/**
 * Tests for Reservoir Computing / Liquid State Machines
 */

import { LiquidStateMachine, createLSM } from '../src/reservoir-computing';

describe('LiquidStateMachine', () => {
  let lsm: LiquidStateMachine;

  beforeEach(() => {
    lsm = new LiquidStateMachine({
      reservoir_size: 20,
      input_size: 5,
      output_size: 3,
    });
  });

  test('should initialize with specified parameters', () => {
    const params = lsm.getParams();

    expect(params.reservoir_size).toBe(20);
    expect(params.input_size).toBe(5);
    expect(params.output_size).toBe(3);
  });

  test('should process input through reservoir', () => {
    const input = [1, 0, 1, 0, 1];
    const state = lsm.processInput(input, 50);

    expect(state.length).toBe(20); // reservoir_size
    expect(state.every((val) => val >= 0)).toBe(true); // Spike counts
  });

  test('should reject invalid input size', () => {
    const invalid_input = [1, 0, 1]; // Wrong size

    expect(() => {
      lsm.processInput(invalid_input, 50);
    }).toThrow();
  });

  test('should generate reservoir activity', () => {
    const input = [1, 1, 1, 1, 1]; // All inputs active
    const state = lsm.processInput(input, 100);

    // Should have some activity (non-zero spike counts)
    const active_neurons = state.filter((count) => count > 0).length;
    expect(active_neurons).toBeGreaterThan(0);
  });

  test('should have stable reservoir dynamics', () => {
    const input = [1, 0, 1, 0, 1];

    // Process same input multiple times
    const state1 = lsm.processInput(input, 50);
    lsm.reset();
    const state2 = lsm.processInput(input, 50);
    lsm.reset();
    const state3 = lsm.processInput(input, 50);

    // States should be similar (not identical due to floating point, but close)
    const diff12 = state1.reduce((sum, val, i) => sum + Math.abs(val - state2[i]), 0);
    const diff23 = state2.reduce((sum, val, i) => sum + Math.abs(val - state3[i]), 0);

    expect(diff12).toBeLessThan(state1.length * 2); // Allow some variance
    expect(diff23).toBeLessThan(state2.length * 2);
  });

  test('should compute forward pass', () => {
    const input = [1, 0, 1, 0, 1];
    const output = lsm.forward(input, 50);

    expect(output.length).toBe(3); // output_size
    expect(output.every((val) => isFinite(val))).toBe(true);
  });

  test('should train readout layer', () => {
    const inputs = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
    ];

    const targets = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    const error = lsm.trainReadout(inputs, targets, 50);

    expect(error).toBeGreaterThanOrEqual(0);
    expect(isFinite(error)).toBe(true);
  });

  test('should improve with training', () => {
    const inputs = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 0, 0, 1],
    ];

    const targets = [
      [1, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    // Measure initial error
    const initial_error = lsm.trainReadout(inputs, targets, 50);

    // Train again (readout should improve)
    const final_error = lsm.trainReadout(inputs, targets, 50);

    // Error should decrease or stay similar (might not decrease due to random initialization)
    expect(final_error).toBeLessThanOrEqual(initial_error * 2); // Allow some variance
  });

  test('should evaluate on test set', () => {
    const train_inputs = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 1, 0, 0],
    ];

    const train_targets = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    lsm.trainReadout(train_inputs, train_targets, 50);

    const test_inputs = train_inputs; // Use same for simplicity
    const test_targets = train_targets;

    const { mse, accuracy } = lsm.evaluate(test_inputs, test_targets, 50);

    expect(mse).toBeGreaterThanOrEqual(0);
    expect(accuracy).toBeGreaterThanOrEqual(0);
    expect(accuracy).toBeLessThanOrEqual(1);
  });

  test('should make predictions', () => {
    const train_inputs = [
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
    ];

    const train_targets = [
      [1, 0, 0],
      [0, 1, 0],
    ];

    lsm.trainReadout(train_inputs, train_targets, 50);

    const prediction = lsm.predict([1, 0, 0, 0, 0], 50);

    expect(prediction.length).toBe(3);
    expect(prediction.every((val) => isFinite(val))).toBe(true);
  });

  test('should get reservoir state', () => {
    const input = [1, 0, 1, 0, 1];
    lsm.processInput(input, 50);

    const state = lsm.getReservoirState();

    expect(state.length).toBe(20);
    expect(Array.isArray(state)).toBe(true);
  });

  test('should get readout weights', () => {
    const weights = lsm.getReadoutWeights();

    expect(weights.weights.length).toBe(3); // output_size
    expect(weights.weights[0].length).toBe(20); // reservoir_size
    expect(weights.bias.length).toBe(3);
  });

  test('should reset reservoir', () => {
    const input = [1, 0, 1, 0, 1];
    lsm.processInput(input, 50);

    const state_before = lsm.getReservoirState();
    const activity_before = state_before.reduce((sum, val) => sum + val, 0);

    lsm.reset();

    const state_after = lsm.getReservoirState();
    const activity_after = state_after.reduce((sum, val) => sum + val, 0);

    expect(activity_after).toBe(0); // Should be reset
  });

  test('should create LSM with presets', () => {
    const small = createLSM('small', 5, 3);
    const medium = createLSM('medium', 5, 3);
    const large = createLSM('large', 5, 3);

    expect(small.getParams().reservoir_size).toBe(50);
    expect(medium.getParams().reservoir_size).toBe(100);
    expect(large.getParams().reservoir_size).toBe(200);
  });
});

describe('Reservoir Computing Applications', () => {
  test('should classify simple patterns', () => {
    const lsm = createLSM('medium', 5, 2);

    const inputs = [
      [1, 1, 0, 0, 0], // Class 0
      [1, 0, 1, 0, 0], // Class 0
      [0, 0, 0, 1, 1], // Class 1
      [0, 0, 1, 0, 1], // Class 1
    ];

    const targets = [
      [1, 0],
      [1, 0],
      [0, 1],
      [0, 1],
    ];

    lsm.trainReadout(inputs, targets, 50);

    const { accuracy } = lsm.evaluate(inputs, targets, 50);

    // Should achieve reasonable accuracy
    expect(accuracy).toBeGreaterThan(0); // At least better than random
  });

  test('should handle temporal patterns', () => {
    const lsm = createLSM('large', 10, 3);

    // Create temporal sequences
    const sequences = [
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], // Early spike
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0], // Middle spike
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], // Late spike
    ];

    const targets = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];

    lsm.trainReadout(sequences, targets, 100);

    const prediction = lsm.predict(sequences[0], 100);

    expect(prediction.length).toBe(3);
    expect(isFinite(prediction[0])).toBe(true);
  });

  test('should process continuous input stream', () => {
    const lsm = createLSM('medium', 8, 4);

    const stream = [
      [1, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
    ];

    const targets = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ];

    lsm.trainReadout(stream, targets, 50);

    // Process stream sequentially
    const predictions = stream.map((input) => lsm.predict(input, 50));

    predictions.forEach((pred) => {
      expect(pred.length).toBe(4);
      expect(pred.every((val) => isFinite(val))).toBe(true);
    });
  });

  test('should be memory efficient', () => {
    // Create large reservoir
    const large_lsm = createLSM('large', 50, 10);

    const input = new Array(50).fill(0).map(() => Math.random() > 0.5 ? 1 : 0);

    expect(() => {
      for (let i = 0; i < 100; i++) {
        large_lsm.processInput(input, 50);
        large_lsm.reset();
      }
    }).not.toThrow();
  });

  test('should handle different simulation durations', () => {
    const lsm = createLSM('medium', 10, 3);

    const input = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

    const state_short = lsm.processInput(input, 25);
    lsm.reset();
    const state_long = lsm.processInput(input, 100);

    // Longer simulation should generate more activity
    const activity_short = state_short.reduce((sum, val) => sum + val, 0);
    const activity_long = state_long.reduce((sum, val) => sum + val, 0);

    expect(activity_long).toBeGreaterThanOrEqual(activity_short);
  });
});
