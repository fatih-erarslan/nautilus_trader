/**
 * Decision Layer Handlers - Active Inference
 *
 * Implements Layer 3 of the cognitive architecture:
 * - Active inference (Friston)
 * - Policy selection
 * - Expected free energy minimization
 * - Action generation
 */

import { QKSBridge } from './mod.js';

export interface Policy {
  id: string;
  actions: number[][];
  expected_free_energy: number;
  epistemic_value: number;
  pragmatic_value: number;
}

export class DecisionHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Update beliefs using precision-weighted prediction errors
   * Implements hierarchical Bayesian inference
   */
  async updateBeliefs(params: {
    observation: number[];
    beliefs: number[];
    precision: number[];
    learning_rate?: number;
  }): Promise<{
    updated_beliefs: number[];
    updated_precision: number[];
    prediction_errors: number[];
    mean_prediction_error: number;
    converged: boolean;
  }> {
    const { observation, beliefs, precision, learning_rate = 0.01 } = params;

    if (observation.length !== beliefs.length || observation.length !== precision.length) {
      throw new Error('Observation, beliefs, and precision must have same length');
    }

    try {
      return await this.bridge.callRust('decision.update_beliefs', {
        observation,
        beliefs,
        precision,
        learning_rate,
      });
    } catch (e) {
      // Fallback: Precision-weighted update
      const updatedBeliefs: number[] = [];
      const predictionErrors: number[] = [];
      const updatedPrecision: number[] = [];

      for (let i = 0; i < beliefs.length; i++) {
        const error = observation[i] - beliefs[i];
        predictionErrors.push(error);

        const precisionWeighted = precision[i] * error;
        const newBelief = beliefs[i] + learning_rate * precisionWeighted;
        updatedBeliefs.push(newBelief);

        const precisionUpdate = precision[i] * (1 + 0.01 * (1 - Math.abs(error)));
        updatedPrecision.push(Math.min(precisionUpdate, 100));
      }

      const meanError = predictionErrors.reduce((a, b) => Math.abs(a) + Math.abs(b), 0) / predictionErrors.length;

      return {
        updated_beliefs: updatedBeliefs,
        updated_precision: updatedPrecision,
        prediction_errors: predictionErrors,
        mean_prediction_error: meanError,
        converged: meanError < 0.01,
      };
    }
  }

  /**
   * Compute expected free energy (EFE) for policy selection
   * EFE = Epistemic Value + Pragmatic Value
   */
  async computeExpectedFreeEnergy(params: {
    policy: number[];
    beliefs: number[];
    goal: number[];
    exploration_weight?: number;
  }): Promise<{
    expected_free_energy: number;
    epistemic_value: number;
    pragmatic_value: number;
    exploration_weight: number;
  }> {
    const { policy, beliefs, goal, exploration_weight = 0.5 } = params;

    try {
      return await this.bridge.callRust('decision.expected_free_energy', {
        policy,
        beliefs,
        goal,
        exploration_weight,
      });
    } catch (e) {
      // Fallback computation
      let entropy = 0;
      for (const b of beliefs) {
        if (b > 1e-10) {
          entropy -= b * Math.log(b);
        }
      }

      let goalDistance = 0;
      for (let i = 0; i < policy.length && i < goal.length; i++) {
        const diff = policy[i] - goal[i];
        goalDistance += diff * diff;
      }

      const epistemicValue = exploration_weight * entropy;
      const pragmaticValue = -(1 - exploration_weight) * Math.sqrt(goalDistance);
      const efe = -(epistemicValue + pragmaticValue);

      return {
        expected_free_energy: efe,
        epistemic_value: epistemicValue,
        pragmatic_value: pragmaticValue,
        exploration_weight,
      };
    }
  }

  /**
   * Select optimal policy from candidates
   * Minimizes expected free energy
   */
  async selectPolicy(params: {
    policies: Policy[];
    exploration_weight?: number;
    temperature?: number;
  }): Promise<{
    selected_policy: Policy;
    selection_probabilities: number[];
    confidence: number;
  }> {
    const { policies, exploration_weight = 0.5, temperature = 1.0 } = params;

    if (policies.length === 0) {
      throw new Error('No policies provided');
    }

    try {
      return await this.bridge.callRust('decision.select_policy', {
        policies,
        exploration_weight,
        temperature,
      });
    } catch (e) {
      // Fallback: Softmax over EFE
      const efes = policies.map(p => p.expected_free_energy);
      const probs = this.softmax(efes.map(e => -e), temperature);

      const maxIdx = probs.indexOf(Math.max(...probs));
      const confidence = probs[maxIdx];

      return {
        selected_policy: policies[maxIdx],
        selection_probabilities: probs,
        confidence,
      };
    }
  }

  /**
   * Generate action from policy
   * Action minimizes expected free energy while satisfying precision constraints
   */
  async generateAction(params: {
    policy: number[];
    beliefs: number[];
    action_precision?: number;
  }): Promise<{
    action: number[];
    predicted_observation: number[];
    expected_free_energy: number;
  }> {
    const { policy, beliefs, action_precision = 1.0 } = params;

    try {
      return await this.bridge.callRust('decision.generate_action', {
        policy,
        beliefs,
        action_precision,
      });
    } catch (e) {
      // Fallback: Add precision-controlled noise
      const action: number[] = [];
      const predictedObservation: number[] = [];

      for (let i = 0; i < policy.length; i++) {
        const noise = (Math.random() - 0.5) * (1 / action_precision);
        action.push(policy[i] + noise);

        if (i < beliefs.length) {
          predictedObservation.push(beliefs[i] + 0.1 * policy[i]);
        }
      }

      let expectedFreeEnergy = 0;
      for (let i = 0; i < predictedObservation.length; i++) {
        const diff = predictedObservation[i] - beliefs[i];
        expectedFreeEnergy += diff * diff;
      }

      return {
        action,
        predicted_observation: predictedObservation,
        expected_free_energy: expectedFreeEnergy,
      };
    }
  }

  /**
   * Plan multi-step action sequence
   * Uses tree search with EFE pruning
   */
  async planActionSequence(params: {
    initial_beliefs: number[];
    goal: number[];
    horizon?: number;
    branching_factor?: number;
  }): Promise<{
    action_sequence: number[][];
    predicted_trajectory: number[][];
    total_expected_free_energy: number;
  }> {
    const { initial_beliefs, goal, horizon = 5, branching_factor = 3 } = params;

    try {
      return await this.bridge.callRust('decision.plan_sequence', {
        initial_beliefs,
        goal,
        horizon,
        branching_factor,
      });
    } catch (e) {
      // Fallback: Greedy planning
      const actionSequence: number[][] = [];
      const predictedTrajectory: number[][] = [initial_beliefs];
      let currentBeliefs = [...initial_beliefs];
      let totalEFE = 0;

      for (let t = 0; t < horizon; t++) {
        const direction = goal.map((g, i) => g - currentBeliefs[i]);
        const action = direction.map(d => Math.tanh(d));
        actionSequence.push(action);

        currentBeliefs = currentBeliefs.map((b, i) => b + 0.2 * action[i]);
        predictedTrajectory.push([...currentBeliefs]);

        const distance = Math.sqrt(
          currentBeliefs.reduce((s, b, i) => s + (b - goal[i]) ** 2, 0)
        );
        totalEFE += distance;
      }

      return {
        action_sequence: actionSequence,
        predicted_trajectory: predictedTrajectory,
        total_expected_free_energy: totalEFE,
      };
    }
  }

  /**
   * Evaluate policy performance
   * Tracks actual vs predicted outcomes
   */
  async evaluatePolicy(params: {
    policy_id: string;
    predicted_outcomes: number[][];
    actual_outcomes: number[][];
  }): Promise<{
    accuracy: number;
    precision_error: number;
    policy_fitness: number;
    update_recommendation: string;
  }> {
    const { policy_id, predicted_outcomes, actual_outcomes } = params;

    if (predicted_outcomes.length !== actual_outcomes.length) {
      throw new Error('Predicted and actual outcomes must have same length');
    }

    try {
      return await this.bridge.callRust('decision.evaluate_policy', {
        policy_id,
        predicted_outcomes,
        actual_outcomes,
      });
    } catch (e) {
      // Fallback: MSE-based evaluation
      let totalError = 0;
      let count = 0;

      for (let i = 0; i < predicted_outcomes.length; i++) {
        for (let j = 0; j < predicted_outcomes[i].length && j < actual_outcomes[i].length; j++) {
          const error = predicted_outcomes[i][j] - actual_outcomes[i][j];
          totalError += error * error;
          count++;
        }
      }

      const mse = count > 0 ? totalError / count : 0;
      const accuracy = 1 / (1 + mse);
      const policyFitness = Math.exp(-mse);

      return {
        accuracy,
        precision_error: Math.sqrt(mse),
        policy_fitness: policyFitness,
        update_recommendation: policyFitness < 0.5 ? 'update_required' : 'continue',
      };
    }
  }

  // ===== Private Helper Methods =====

  private softmax(values: number[], temperature: number): number[] {
    const scaled = values.map(v => v / temperature);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }
}
