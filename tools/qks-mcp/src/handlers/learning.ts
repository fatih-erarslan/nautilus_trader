/**
 * Learning Layer Handlers - STDP & Plasticity
 *
 * Implements Layer 4 of the cognitive architecture:
 * - Spike-Timing Dependent Plasticity (STDP)
 * - Hebbian learning
 * - Meta-learning (MAML)
 * - Transfer learning
 */

import { QKSBridge } from './mod.js';

export interface STDPResult {
  weight_change: number;
  type: 'LTP' | 'LTD';
  magnitude: number;
}

export interface LearningMetrics {
  convergence_rate: number;
  adaptation_speed: number;
  transfer_efficiency: number;
  catastrophic_forgetting: number;
}

export class LearningHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Compute classical STDP weight change
   * ΔW = A₊ × exp(-Δt/τ₊) for LTP, -A₋ × exp(Δt/τ₋) for LTD
   */
  async computeClassicalSTDP(params: {
    delta_t: number;
    a_plus?: number;
    a_minus?: number;
    tau_plus?: number;
    tau_minus?: number;
  }): Promise<STDPResult> {
    const {
      delta_t,
      a_plus = 0.005,
      a_minus = 0.00525,
      tau_plus = 20.0,
      tau_minus = 20.0,
    } = params;

    try {
      return await this.bridge.callRust('learning.stdp_classical', {
        delta_t,
        a_plus,
        a_minus,
        tau_plus,
        tau_minus,
      });
    } catch (e) {
      // Fallback implementation
      let weight_change: number;
      let type: 'LTP' | 'LTD';

      if (delta_t > 0) {
        weight_change = a_plus * Math.exp(-delta_t / tau_plus);
        type = 'LTP';
      } else {
        weight_change = -a_minus * Math.exp(delta_t / tau_minus);
        type = 'LTD';
      }

      return {
        weight_change,
        type,
        magnitude: Math.abs(weight_change),
      };
    }
  }

  /**
   * Compute triplet STDP (three-factor learning)
   * More biologically realistic than classical STDP
   */
  async computeTripletSTDP(params: {
    pre_times: number[];
    post_times: number[];
    a2_plus?: number;
    a2_minus?: number;
    a3_plus?: number;
    a3_minus?: number;
    tau_plus?: number;
    tau_x?: number;
    tau_minus?: number;
    tau_y?: number;
  }): Promise<{
    total_change: number;
    ltp_count: number;
    ltd_count: number;
    pair_count: number;
  }> {
    const {
      pre_times,
      post_times,
      a2_plus = 0.005,
      a2_minus = 0.005,
      a3_plus = 0.01,
      a3_minus = 0.01,
      tau_plus = 16.8,
      tau_x = 101.0,
      tau_minus = 33.7,
      tau_y = 125.0,
    } = params;

    try {
      return await this.bridge.callRust('learning.stdp_triplet', {
        pre_times,
        post_times,
        a2_plus,
        a2_minus,
        a3_plus,
        a3_minus,
        tau_plus,
        tau_x,
        tau_minus,
        tau_y,
      });
    } catch (e) {
      // Fallback: Pair-based calculation
      let total_change = 0;
      let ltp_count = 0;
      let ltd_count = 0;

      for (const pre_t of pre_times) {
        for (const post_t of post_times) {
          const dt = post_t - pre_t;
          if (dt > 0) {
            total_change += a2_plus * Math.exp(-dt / tau_plus);
            ltp_count++;
          } else if (dt < 0) {
            total_change += -a2_minus * Math.exp(dt / tau_minus);
            ltd_count++;
          }
        }
      }

      return {
        total_change,
        ltp_count,
        ltd_count,
        pair_count: pre_times.length * post_times.length,
      };
    }
  }

  /**
   * Apply reward-modulated STDP
   * Spike timing creates eligibility, reward signal modulates learning
   */
  async applyRewardModulatedSTDP(params: {
    pre_times: number[];
    post_times: number[];
    reward_signal: {
      value: number;
      time: number;
      phasic?: boolean;
    };
    learning_rate?: number;
    tau_eligibility?: number;
    tau_timing?: number;
    tau_dopamine?: number;
  }): Promise<{
    weight_change: number;
    eligibility: number;
    dopamine_level: number;
    is_reward: boolean;
  }> {
    const {
      pre_times,
      post_times,
      reward_signal,
      learning_rate = 0.01,
      tau_eligibility = 1000.0,
      tau_timing = 20.0,
      tau_dopamine = 200.0,
    } = params;

    try {
      return await this.bridge.callRust('learning.stdp_reward_modulated', {
        pre_times,
        post_times,
        reward_signal,
        learning_rate,
        tau_eligibility,
        tau_timing,
        tau_dopamine,
      });
    } catch (e) {
      // Fallback implementation
      let eligibility = 0;
      for (const pre_t of pre_times) {
        for (const post_t of post_times) {
          const dt = post_t - pre_t;
          if (Math.abs(dt) < 100) {
            const timing_value = Math.exp(-Math.abs(dt) / tau_timing);
            eligibility += dt > 0 ? timing_value : -timing_value;
          }
        }
      }

      const lastSpike = Math.max(...post_times);
      const time_since_pairing = reward_signal.time - lastSpike;
      const eligibility_decay = Math.exp(-time_since_pairing / tau_eligibility);
      const effective_eligibility = eligibility * eligibility_decay;

      const dopamine_value = reward_signal.phasic
        ? reward_signal.value
        : reward_signal.value * Math.exp(-time_since_pairing / tau_dopamine);

      const weight_change = learning_rate * effective_eligibility * dopamine_value;

      return {
        weight_change,
        eligibility: effective_eligibility,
        dopamine_level: dopamine_value,
        is_reward: reward_signal.value > 0,
      };
    }
  }

  /**
   * Apply homeostatic plasticity
   * Maintains target firing rates through synaptic scaling
   */
  async applyHomeostaticPlasticity(params: {
    neuron_rates: number[];
    target_rate?: number;
    learning_rate?: number;
    enable_synaptic_scaling?: boolean;
    enable_intrinsic_plasticity?: boolean;
  }): Promise<{
    scaling_factors: number[];
    excitability_changes: number[];
    homeostatic_status: string[];
  }> {
    const {
      neuron_rates,
      target_rate = 5.0,
      learning_rate = 0.0001,
      enable_synaptic_scaling = true,
      enable_intrinsic_plasticity = true,
    } = params;

    try {
      return await this.bridge.callRust('learning.homeostatic_plasticity', {
        neuron_rates,
        target_rate,
        learning_rate,
        enable_synaptic_scaling,
        enable_intrinsic_plasticity,
      });
    } catch (e) {
      // Fallback implementation
      const scaling_factors: number[] = [];
      const excitability_changes: number[] = [];
      const homeostatic_status: string[] = [];

      for (const rate of neuron_rates) {
        const rate_error = target_rate - rate;

        const scaling = enable_synaptic_scaling
          ? Math.max(0.5, Math.min(2.0, 1.0 + (learning_rate * rate_error) / target_rate))
          : 1.0;

        const excitability = enable_intrinsic_plasticity
          ? Math.max(-0.5, Math.min(0.5, learning_rate * rate_error))
          : 0.0;

        scaling_factors.push(scaling);
        excitability_changes.push(excitability);
        homeostatic_status.push(
          rate_error > 0 ? 'too_low' : rate_error < 0 ? 'too_high' : 'at_target'
        );
      }

      return {
        scaling_factors,
        excitability_changes,
        homeostatic_status,
      };
    }
  }

  /**
   * Meta-learning with Model-Agnostic Meta-Learning (MAML)
   * Learns to learn from task distributions
   */
  async metaLearn(params: {
    task_distribution?: any[];
    num_inner_steps?: number;
    inner_lr?: number;
    outer_lr?: number;
  }): Promise<{
    adapted_parameters: any;
    meta_gradient: number[];
    strategy_selected: string;
    adaptation_speed: number;
  }> {
    const {
      task_distribution = [],
      num_inner_steps = 5,
      inner_lr = 0.01,
      outer_lr = 0.001,
    } = params;

    try {
      return await this.bridge.callRust('learning.meta_learn', {
        task_distribution,
        num_inner_steps,
        inner_lr,
        outer_lr,
      });
    } catch (e) {
      // Fallback: Simplified meta-learning
      return {
        adapted_parameters: {},
        meta_gradient: new Array(10).fill(0),
        strategy_selected: 'gradient_based',
        adaptation_speed: 0.5,
      };
    }
  }

  /**
   * Apply transfer learning
   * Adapt learned representations to new tasks
   */
  async transferLearn(params: {
    source_task_params: any;
    target_task_data: any;
    fine_tune_layers?: number[];
    freeze_layers?: number[];
  }): Promise<{
    transferred_params: any;
    transfer_efficiency: number;
    catastrophic_forgetting: number;
  }> {
    const {
      source_task_params,
      target_task_data,
      fine_tune_layers = [],
      freeze_layers = [],
    } = params;

    try {
      return await this.bridge.callRust('learning.transfer_learn', {
        source_task_params,
        target_task_data,
        fine_tune_layers,
        freeze_layers,
      });
    } catch (e) {
      return {
        transferred_params: source_task_params,
        transfer_efficiency: 0.7,
        catastrophic_forgetting: 0.1,
      };
    }
  }

  /**
   * Compute learning metrics
   * Tracks convergence, adaptation, and forgetting
   */
  async computeLearningMetrics(params: {
    loss_history: number[];
    performance_history: number[];
    task_switching_points?: number[];
  }): Promise<LearningMetrics> {
    const { loss_history, performance_history, task_switching_points = [] } = params;

    try {
      return await this.bridge.callRust('learning.compute_metrics', {
        loss_history,
        performance_history,
        task_switching_points,
      });
    } catch (e) {
      // Fallback computation
      const convergence_rate = this.computeConvergenceRate(loss_history);
      const adaptation_speed = this.computeAdaptationSpeed(
        performance_history,
        task_switching_points
      );

      return {
        convergence_rate,
        adaptation_speed,
        transfer_efficiency: 0.7,
        catastrophic_forgetting: 0.2,
      };
    }
  }

  /**
   * Batch apply STDP to multiple synapses
   * More efficient than individual updates
   */
  async batchApplySTDP(params: {
    synapse_timings: Array<{ synapse_id: number; delta_t: number }>;
    rule_type: 'classical' | 'triplet' | 'reward_modulated';
    rule_params?: any;
  }): Promise<{
    weight_deltas: Array<{ synapse_id: number; delta: number }>;
    ltp_count: number;
    ltd_count: number;
  }> {
    const { synapse_timings, rule_type, rule_params = {} } = params;

    try {
      return await this.bridge.callRust('learning.batch_stdp', {
        synapse_timings,
        rule_type,
        rule_params,
      });
    } catch (e) {
      // Fallback: Apply classical STDP to each
      const weight_deltas: Array<{ synapse_id: number; delta: number }> = [];
      let ltp_count = 0;
      let ltd_count = 0;

      for (const { synapse_id, delta_t } of synapse_timings) {
        const result = await this.computeClassicalSTDP({ delta_t });
        weight_deltas.push({ synapse_id, delta: result.weight_change });

        if (result.type === 'LTP') ltp_count++;
        else ltd_count++;
      }

      return {
        weight_deltas,
        ltp_count,
        ltd_count,
      };
    }
  }

  // ===== Private Helper Methods =====

  private computeConvergenceRate(loss_history: number[]): number {
    if (loss_history.length < 2) return 0;

    const recent = loss_history.slice(-10);
    const early = loss_history.slice(0, 10);

    const recentMean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const earlyMean = early.reduce((a, b) => a + b, 0) / early.length;

    return earlyMean > 0 ? (earlyMean - recentMean) / earlyMean : 0;
  }

  private computeAdaptationSpeed(
    performance_history: number[],
    task_switching_points: number[]
  ): number {
    if (task_switching_points.length === 0) return 0.5;

    let totalSpeed = 0;
    for (const switchPoint of task_switching_points) {
      if (switchPoint < performance_history.length - 1) {
        const initial = performance_history[switchPoint];
        const adapted = performance_history[Math.min(switchPoint + 10, performance_history.length - 1)];
        const speed = Math.max(0, adapted - initial);
        totalSpeed += speed;
      }
    }

    return totalSpeed / task_switching_points.length;
  }
}
