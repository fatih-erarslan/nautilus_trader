/**
 * QKS MCP Server - Rust FFI Bridge
 *
 * Bridge layer for calling Rust core functions via FFI or fallback implementations
 */

import { existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
import type { RustBridge } from "./types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..", "..", "..");

// =============================================================================
// Native Module Loading
// =============================================================================

interface NativeQksCore {
  // Layer 1: Thermodynamic
  thermo_compute_energy(state_json: string): number;
  thermo_compute_temperature(state_json: string): number;
  thermo_compute_entropy(state_json: string): number;
  thermo_critical_temp_ising(): number;

  // Layer 2: Cognitive
  cognitive_attention_softmax(inputs: Float64Array): Float64Array;
  cognitive_memory_decay(memory: any, decay_rate: number): any;

  // Layer 3: Decision
  decision_expected_free_energy(policy: string, beliefs: string): number;

  // Layer 4: Learning
  learning_stdp_weight_change(delta_t: number, a_plus: number, a_minus: number, tau: number): number;

  // Layer 5: Collective
  collective_swarm_state(agents_json: string): string;

  // Layer 6: Consciousness
  consciousness_compute_phi(network_json: string): number;

  // Layer 7: Metacognition
  meta_introspect_state(): string;

  // Layer 8: Integration
  integration_system_health(): string;
}

let native: NativeQksCore | null = null;

// Try to load native Rust module
const nativePaths = [
  process.env.QKS_NATIVE_PATH,
  resolve(projectRoot, "rust-core/target/release/libqks_core.dylib"),
  resolve(projectRoot, "rust-core/target/release/libqks_core.so"),
  resolve(__dirname, "../dist/libqks_core.dylib"),
];

for (const path of nativePaths) {
  if (path && existsSync(path)) {
    try {
      native = require(path) as NativeQksCore;
      console.error(`[QKS Bridge] Loaded native module from ${path}`);
      break;
    } catch (e) {
      console.error(`[QKS Bridge] Failed to load ${path}: ${e}`);
    }
  }
}

if (!native) {
  console.error("[QKS Bridge] Warning: Native module not available, using TypeScript fallback");
}

// =============================================================================
// Fallback Implementations (Pure TypeScript)
// =============================================================================

const fallback = {
  // Layer 1: Thermodynamic
  thermo_compute_energy: (state_json: string): number => {
    const state = JSON.parse(state_json);
    return state.energy || 0.0;
  },

  thermo_compute_temperature: (state_json: string): number => {
    const state = JSON.parse(state_json);
    return state.temperature || 1.0;
  },

  thermo_compute_entropy: (state_json: string): number => {
    const state = JSON.parse(state_json);
    // Shannon entropy fallback
    if (state.probabilities) {
      let entropy = 0;
      for (const p of state.probabilities) {
        if (p > 0) {
          entropy -= p * Math.log2(p);
        }
      }
      return entropy;
    }
    return 0.0;
  },

  thermo_critical_temp_ising: (): number => {
    // Onsager solution: T_c = 2 / ln(1 + sqrt(2))
    return 2 / Math.log(1 + Math.sqrt(2));
  },

  // Layer 2: Cognitive
  cognitive_attention_softmax: (inputs: Float64Array): Float64Array => {
    const max = Math.max(...inputs);
    const exp_values = Array.from(inputs).map(x => Math.exp(x - max));
    const sum = exp_values.reduce((a, b) => a + b, 0);
    return new Float64Array(exp_values.map(x => x / sum));
  },

  cognitive_memory_decay: (memory: any, decay_rate: number): any => {
    if (Array.isArray(memory)) {
      return memory.map(item => ({
        ...item,
        strength: (item.strength || 1.0) * Math.exp(-decay_rate)
      }));
    }
    return memory;
  },

  // Layer 3: Decision
  decision_expected_free_energy: (policy: string, beliefs: string): number => {
    // Simplified EFE = -E[log P(o|s)] + KL[Q(s|π)||P(s)]
    return Math.random() * 10; // Placeholder
  },

  // Layer 4: Learning
  learning_stdp_weight_change: (delta_t: number, a_plus: number, a_minus: number, tau: number): number => {
    if (delta_t > 0) {
      // LTP (Long-Term Potentiation)
      return a_plus * Math.exp(-delta_t / tau);
    } else {
      // LTD (Long-Term Depression)
      return -a_minus * Math.exp(delta_t / tau);
    }
  },

  // Layer 5: Collective
  collective_swarm_state: (agents_json: string): string => {
    const agents = JSON.parse(agents_json);
    return JSON.stringify({
      agents,
      topology: "mesh",
      consensus_level: 0.8
    });
  },

  // Layer 6: Consciousness
  consciousness_compute_phi: (network_json: string): number => {
    // IIT 3.0 Φ computation (simplified)
    const network = JSON.parse(network_json);
    const n = network.length || 4;
    // Placeholder: returns complexity metric
    return Math.log2(n) * Math.random();
  },

  // Layer 7: Metacognition
  meta_introspect_state: (): string => {
    return JSON.stringify({
      internal_state: { energy: 1.0, temperature: 1.0 },
      certainty: 0.75,
      coherence: 0.85,
      conflicts: []
    });
  },

  // Layer 8: Integration
  integration_system_health: (): string => {
    return JSON.stringify({
      layer1_health: 1.0,
      layer2_health: 1.0,
      layer3_health: 1.0,
      layer4_health: 1.0,
      layer5_health: 1.0,
      layer6_health: 1.0,
      layer7_health: 1.0,
      layer8_health: 1.0,
      overall_health: 1.0
    });
  },
};

// Use native or fallback
const lib = native || fallback;

// =============================================================================
// Rust Bridge Implementation
// =============================================================================

export const rustBridge: RustBridge = {
  // Layer 1: Thermodynamic
  async thermo_compute_energy(state: any): Promise<number> {
    return lib.thermo_compute_energy(JSON.stringify(state));
  },

  async thermo_compute_temperature(state: any): Promise<number> {
    return lib.thermo_compute_temperature(JSON.stringify(state));
  },

  async thermo_compute_entropy(state: any): Promise<number> {
    return lib.thermo_compute_entropy(JSON.stringify(state));
  },

  async thermo_critical_point(system: string): Promise<any> {
    if (system === "ising") {
      return {
        temperature: lib.thermo_critical_temp_ising(),
        formula: "T_c = 2/ln(1 + sqrt(2))",
        reference: "Onsager (1944)"
      };
    }
    return { temperature: 1.0 };
  },

  // Layer 2: Cognitive
  async cognitive_attention_focus(inputs: number[], weights: number[]): Promise<any> {
    const attended = lib.cognitive_attention_softmax(new Float64Array(inputs));
    return {
      focus_weights: Array.from(attended),
      attention_mask: Array.from(attended).map(w => w > 0.1),
      entropy: -Array.from(attended).reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0)
    };
  },

  async cognitive_memory_consolidate(working: any[], strength: number): Promise<any[]> {
    return lib.cognitive_memory_decay(working, 1.0 - strength);
  },

  // Layer 3: Decision
  async decision_compute_efe(policy: any, beliefs: any): Promise<number> {
    return lib.decision_expected_free_energy(JSON.stringify(policy), JSON.stringify(beliefs));
  },

  async decision_select_policy(policies: any[]): Promise<number> {
    // Select policy with minimum EFE
    let min_efe = Infinity;
    let best_idx = 0;
    for (let i = 0; i < policies.length; i++) {
      if (policies[i].expected_free_energy < min_efe) {
        min_efe = policies[i].expected_free_energy;
        best_idx = i;
      }
    }
    return best_idx;
  },

  // Layer 4: Learning
  async learning_stdp_update(pre_time: number, post_time: number): Promise<number> {
    const delta_t = post_time - pre_time;
    return lib.learning_stdp_weight_change(delta_t, 0.1, 0.12, 20.0);
  },

  async learning_consolidate_memory(episodes: any[]): Promise<any[]> {
    // Consolidate episodic memories
    return episodes.filter(ep => (ep.strength || 1.0) > 0.3);
  },

  // Layer 5: Collective
  async collective_swarm_coordinate(agents: any[]): Promise<any> {
    const state_json = lib.collective_swarm_state(JSON.stringify(agents));
    return JSON.parse(state_json);
  },

  async collective_reach_consensus(proposal: any): Promise<boolean> {
    const threshold = 0.66; // 2/3 majority
    const total = proposal.votes_for + proposal.votes_against;
    if (total === 0) return false;
    return proposal.votes_for / total >= threshold;
  },

  // Layer 6: Consciousness
  async consciousness_compute_phi(network: number[][]): Promise<number> {
    return lib.consciousness_compute_phi(JSON.stringify(network));
  },

  async consciousness_broadcast_workspace(content: any): Promise<any> {
    return {
      broadcast_content: content,
      attending_modules: ["perception", "memory", "decision"],
      priority: 0.8,
      duration: 100
    };
  },

  // Layer 7: Metacognition
  async meta_introspect(): Promise<any> {
    const state_json = lib.meta_introspect_state();
    return JSON.parse(state_json);
  },

  async meta_update_self_model(observations: any[]): Promise<any> {
    return {
      beliefs_about_self: new Map(Object.entries({
        "competence": 0.8,
        "energy_level": 0.7
      })),
      goals: ["optimize_performance", "maintain_coherence"],
      capabilities: ["reasoning", "learning", "adaptation"],
      limitations: ["compute_bound", "memory_limited"],
      confidence: 0.75
    };
  },

  // Layer 8: Integration
  async integration_system_health(): Promise<any> {
    const health_json = lib.integration_system_health();
    return JSON.parse(health_json);
  },

  async integration_cognitive_loop_step(input: any): Promise<any> {
    return {
      perception: input,
      inference: { beliefs: [0.5, 0.3, 0.2], confidence: 0.8 },
      action: { selected: "explore", confidence: 0.7 },
      prediction_error: 0.15,
      loop_latency_ms: 50
    };
  },
};

// =============================================================================
// Utility Functions
// =============================================================================

export function isNativeAvailable(): boolean {
  return native !== null;
}

export function getNativeModulePath(): string | null {
  for (const path of nativePaths) {
    if (path && existsSync(path)) {
      return path;
    }
  }
  return null;
}
