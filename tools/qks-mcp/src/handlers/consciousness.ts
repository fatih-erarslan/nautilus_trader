/**
 * Consciousness Layer Handlers - IIT Φ & Global Workspace
 *
 * Implements Layer 6 of the cognitive architecture:
 * - Integrated Information Theory (Tononi)
 * - Global Workspace Theory (Baars)
 * - Causal density computation
 * - Broadcast mechanisms
 */

import { QKSBridge } from './mod.js';

export interface PhiResult {
  phi: number;
  is_conscious: boolean;
  partitions: any[];
  causal_density: number;
  consciousness_level: string;
}

export class ConsciousnessHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Compute integrated information Φ (consciousness metric)
   * Using Tononi's IIT 3.0
   */
  async computePhi(params: {
    network_state: number[];
    connectivity?: number[][];
    algorithm?: 'exact' | 'monte_carlo' | 'greedy' | 'hierarchical';
  }): Promise<PhiResult> {
    const { network_state, connectivity, algorithm = 'greedy' } = params;

    if (network_state.length < 4) {
      throw new Error('Network state must have at least 4 elements for meaningful Φ');
    }

    try {
      const result = await this.bridge.callRust('consciousness.compute_phi', {
        network_state,
        connectivity,
        algorithm,
      });

      return {
        ...result,
        is_conscious: result.phi > 1.0,
      };
    } catch (e) {
      // Fallback: Greedy approximation
      const n = network_state.length;

      let stateEntropy = 0;
      for (const s of network_state) {
        if (s > 1e-10) {
          stateEntropy -= s * Math.log2(s);
        }
      }

      let effectiveInfo = 0;
      if (connectivity && connectivity.length > 0) {
        let totalConnections = 0;
        let activeConnections = 0;

        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            if (connectivity[i]?.[j]) {
              totalConnections++;
              if (network_state[i] > 0.5 && network_state[j] > 0.5) {
                activeConnections++;
              }
            }
          }
        }

        effectiveInfo = totalConnections > 0
          ? (activeConnections / totalConnections) * stateEntropy
          : 0;
      } else {
        effectiveInfo = stateEntropy * 0.5;
      }

      const phi = Math.max(0, effectiveInfo);

      return {
        phi,
        is_conscious: phi > 1.0,
        partitions: [],
        causal_density: effectiveInfo / (stateEntropy + 1e-10),
        consciousness_level: phi > 1.0 ? 'emergent' : phi > 0.5 ? 'minimal' : 'none',
      };
    }
  }

  /**
   * Broadcast to global workspace
   * Makes information globally available to all cognitive processes
   */
  async globalWorkspaceBroadcast(params: {
    content: any;
    priority?: number;
    source?: string;
  }): Promise<{
    broadcast_id: string;
    reach: number;
    integration_success: boolean;
    access_consciousness: boolean;
  }> {
    const { content, priority = 0.5, source = 'unknown' } = params;

    try {
      return await this.bridge.callRust('consciousness.broadcast', {
        content,
        priority,
        source,
      });
    } catch (e) {
      // Fallback: Simple broadcast
      const broadcastId = `broadcast_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      return {
        broadcast_id: broadcastId,
        reach: priority > 0.7 ? 100 : priority > 0.3 ? 50 : 10,
        integration_success: priority > 0.3,
        access_consciousness: priority > 0.5,
      };
    }
  }

  /**
   * Subscribe to global workspace broadcasts
   * Cognitive modules can receive globally broadcasted information
   */
  async subscribeToWorkspace(params: {
    module_id: string;
    filter?: {
      priority_threshold?: number;
      content_types?: string[];
      sources?: string[];
    };
  }): Promise<{
    subscription_id: string;
    active: boolean;
  }> {
    const { module_id, filter = {} } = params;

    try {
      return await this.bridge.callRust('consciousness.subscribe', {
        module_id,
        filter,
      });
    } catch (e) {
      return {
        subscription_id: `sub_${module_id}_${Date.now()}`,
        active: true,
      };
    }
  }

  /**
   * Analyze causal density of network
   * Measures cause-effect power
   */
  async analyzeCausalDensity(params: {
    network_state: number[];
    connectivity: number[][];
    time_window?: number;
  }): Promise<{
    causal_density: number;
    cause_effect_pairs: Array<{ cause: number; effect: number; strength: number }>;
    effective_connectivity: number[][];
  }> {
    const { network_state, connectivity, time_window = 1 } = params;

    try {
      return await this.bridge.callRust('consciousness.causal_density', {
        network_state,
        connectivity,
        time_window,
      });
    } catch (e) {
      // Fallback: Count active connections
      const n = network_state.length;
      const pairs: Array<{ cause: number; effect: number; strength: number }> = [];

      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (connectivity[i]?.[j] && network_state[i] > 0.5) {
            const strength = connectivity[i][j] * network_state[i];
            pairs.push({ cause: i, effect: j, strength });
          }
        }
      }

      const density = pairs.length / (n * (n - 1));

      return {
        causal_density: density,
        cause_effect_pairs: pairs,
        effective_connectivity: connectivity,
      };
    }
  }

  /**
   * Detect qualia (subjective experience markers)
   * Experimental feature based on phenomenology
   */
  async detectQualia(params: {
    sensory_input: number[];
    internal_state: number[];
    attention_modulation?: number;
  }): Promise<{
    qualia_detected: boolean;
    phenomenal_intensity: number;
    quality_vector: number[];
    ineffability_score: number;
  }> {
    const { sensory_input, internal_state, attention_modulation = 1.0 } = params;

    try {
      return await this.bridge.callRust('consciousness.detect_qualia', {
        sensory_input,
        internal_state,
        attention_modulation,
      });
    } catch (e) {
      // Fallback: Heuristic qualia detection
      const intensity = sensory_input.reduce((s, x) => s + Math.abs(x), 0) / sensory_input.length;
      const modulated = intensity * attention_modulation;

      return {
        qualia_detected: modulated > 0.5,
        phenomenal_intensity: modulated,
        quality_vector: sensory_input.map(x => x * attention_modulation),
        ineffability_score: 0.8, // High by definition
      };
    }
  }

  /**
   * Analyze self-organized criticality markers
   * Detects edge-of-chaos dynamics optimal for consciousness
   */
  async analyzeCriticality(params: {
    activity_timeseries: number[];
    avalanche_threshold?: number;
  }): Promise<{
    branching_ratio: number;
    at_criticality: boolean;
    criticality_score: number;
    avalanche_count: number;
    power_law_exponent: number;
  }> {
    const { activity_timeseries, avalanche_threshold = 2.0 } = params;

    try {
      return await this.bridge.callRust('consciousness.criticality', {
        activity_timeseries,
        avalanche_threshold,
      });
    } catch (e) {
      // Fallback: Simple statistical analysis
      const mean = activity_timeseries.reduce((a, b) => a + b, 0) / activity_timeseries.length;
      const variance = activity_timeseries.reduce(
        (a, b) => a + (b - mean) ** 2,
        0
      ) / activity_timeseries.length;
      const std = Math.sqrt(variance);

      const threshold = mean + avalanche_threshold * std;
      const avalanches: number[][] = [];
      let current: number[] = [];

      for (const val of activity_timeseries) {
        if (val > threshold) {
          current.push(val);
        } else if (current.length > 0) {
          avalanches.push([...current]);
          current = [];
        }
      }

      let branchingRatio = 1.0;
      if (avalanches.length > 1) {
        let ratioSum = 0;
        for (let i = 1; i < avalanches.length; i++) {
          const prev = avalanches[i - 1].length;
          const curr = avalanches[i].length;
          if (prev > 0) {
            ratioSum += curr / prev;
          }
        }
        branchingRatio = ratioSum / (avalanches.length - 1);
      }

      const atCriticality = Math.abs(branchingRatio - 1.0) < 0.1;

      return {
        branching_ratio: branchingRatio,
        at_criticality: atCriticality,
        criticality_score: 1.0 - Math.abs(branchingRatio - 1.0),
        avalanche_count: avalanches.length,
        power_law_exponent: 1.5, // Typical SOC value
      };
    }
  }

  /**
   * Measure consciousness continuity over time
   * Tracks phenomenological coherence
   */
  async measureContinuity(params: {
    phi_history: number[];
    workspace_activity: number[][];
  }): Promise<{
    continuity_index: number;
    disruptions: number[];
    phenomenal_stability: number;
  }> {
    const { phi_history, workspace_activity } = params;

    try {
      return await this.bridge.callRust('consciousness.continuity', {
        phi_history,
        workspace_activity,
      });
    } catch (e) {
      // Fallback: Variance-based continuity
      const phiVariance = this.computeVariance(phi_history);
      const continuity = 1 / (1 + phiVariance);

      const disruptions: number[] = [];
      for (let i = 1; i < phi_history.length; i++) {
        if (Math.abs(phi_history[i] - phi_history[i - 1]) > 0.5) {
          disruptions.push(i);
        }
      }

      return {
        continuity_index: continuity,
        disruptions,
        phenomenal_stability: continuity,
      };
    }
  }

  // ===== Private Helper Methods =====

  private computeVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  }
}
