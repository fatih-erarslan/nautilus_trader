#!/usr/bin/env bun
/**
 * Comprehensive Wolfram Handler Validation Suite
 */

import { handleEnhancedTool } from "../src/tools/index.ts";

console.log("╔═══════════════════════════════════════════════════════════════════╗");
console.log("║     COMPREHENSIVE WOLFRAM HANDLER VALIDATION SUITE               ║");
console.log("╚═══════════════════════════════════════════════════════════════════╝");
console.log("");

const tests = [
  // Systems Dynamics
  { name: "systems_model_simulate", args: { equations: ["dx/dt = -0.1*x"], initialConditions: {x:1}, timeSpan: [0,10] } },
  { name: "systems_equilibrium_find", args: { equations: ["x^2 - 4"], variables: ["x"] } },
  { name: "systems_feedback_causal_loop", args: { variables: ["A","B"], connections: [{from:"A",to:"B",polarity:"+"},{from:"B",to:"A",polarity:"-"}] } },

  // Design Thinking
  { name: "design_empathize_analyze", args: { userResearch: "Users need fast secure AI agents" } },
  { name: "design_define_problem", args: { insights: ["Performance critical", "Security required"] } },
  { name: "design_ideate_brainstorm", args: { problemStatement: "Build secure AI", ideaCount: 3 } },

  // LLM Tools
  { name: "wolfram_llm_analyze", args: { topic: "AI Safety", analysisType: "swot" } },
  { name: "wolfram_llm_code_generate", args: { specification: "Hash function", language: "rust" } },
  { name: "wolfram_llm_reason", args: { question: "Why is post-quantum crypto important?", method: "chain_of_thought" } },

  // Agency - Core
  { name: "agency_compute_phi", args: { network_state: [0.5, 0.7, 0.3] } },
  { name: "agency_compute_free_energy", args: { observation: [0.5,0.5], beliefs: [0.6,0.4], precision: [1,1] } },
  { name: "agency_analyze_criticality", args: { activity_timeseries: [0.1,0.3,0.8,1.2,0.9,0.4] } },

  // Agency - Negentropy (Pedagogic Awareness)
  { name: "agency_compute_negentropy", args: { agent_id: "test-agent-1", beliefs: [0.3, 0.5, 0.2], precision: [1.0, 0.8, 1.2], prediction_error: 0.15, free_energy: 1.2 } },
  { name: "agency_get_bateson_level", args: { agent_id: "test-agent-1" } },
  { name: "agency_get_scaffold_mode", args: { agent_id: "test-agent-1" } },
  { name: "agency_get_intrinsic_motivation", args: { agent_id: "test-agent-1" } },
  { name: "agency_get_cognitive_state", args: { agent_id: "test-agent-1" } },
  { name: "agency_pedagogic_intervention", args: { agent_id: "test-agent-1", intervention_type: "curiosity_boost" } },

  // Agency - L4 Evolution (Holland, 1975)
  { name: "agency_set_population_context", args: { agent_id: "test-agent-1", population_size: 5 } },
  { name: "agency_update_fitness", args: { agent_id: "test-agent-1", fitness: 0.7 } },
  { name: "agency_get_l4_readiness", args: { agent_id: "test-agent-1" } },

  // Vector
  { name: "vector_db_create", args: { name: "test", dimensions: 128 } },
  { name: "vector_gnn_forward", args: { nodeFeatures: [[0.1,0.2],[0.3,0.4]], edgeIndex: [[0,1],[1,0]] } },

  // Cortex
  { name: "cortex_pbit_sample", args: { field: 0.5, temperature: 2.269 } },
  { name: "cortex_lorentz_distance", args: { point1: [1.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002], point2: [1.8, 0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005] } },

  // STDP
  { name: "stdp_classical_compute", args: { delta_t: 10, a_plus: 0.1, a_minus: 0.12, tau_plus: 20 } },
  { name: "stdp_triplet_compute", args: { pre_times: [0,50,100], post_times: [10,60,110] } },

  // SGNN (stateless tests only - stateful require network session)
  { name: "sgnn_network_create", args: { num_neurons: 100, connectivity: 0.15 } },

  // Swarm Intelligence (14+ biomimetic algorithms)
  { name: "swarm_pso", args: { objective: "sphere", bounds: [{min: -5, max: 5}, {min: -5, max: 5}, {min: -5, max: 5}], population_size: 20, max_iterations: 100 } },
  { name: "swarm_topology_create", args: { topology_type: "ring", agent_count: 10 } },
];

let passed = 0;
let failed = 0;
const results: { name: string; status: string; type: string }[] = [];

for (const test of tests) {
  try {
    const result = await handleEnhancedTool(test.name, test.args, null);
    const parsed = JSON.parse(result);
    const hasError = parsed.error !== undefined;
    const isStub = result.includes("STUB") || result.includes("pending");

    if (!hasError) {
      console.log("✓ " + test.name.padEnd(30) + (isStub ? " [STUB]" : " [REAL]"));
      passed++;
      results.push({ name: test.name, status: "PASS", type: isStub ? "STUB" : "REAL" });
    } else {
      console.log("✗ " + test.name.padEnd(30) + " [FAIL]");
      failed++;
      results.push({ name: test.name, status: "FAIL", type: "ERROR" });
    }
  } catch (e: any) {
    console.log("✗ " + test.name.padEnd(30) + " [ERROR: " + e.message?.substring(0, 20) + "]");
    failed++;
    results.push({ name: test.name, status: "ERROR", type: e.message });
  }
}

console.log("");
console.log("═══════════════════════════════════════════════════════════════════");
console.log(`Results: ${passed}/${passed + failed} passed (${Math.round(passed / (passed + failed) * 100)}%)`);
console.log("═══════════════════════════════════════════════════════════════════");

// Summary by category
const categories = {
  systems: results.filter(r => r.name.startsWith("systems_")),
  design: results.filter(r => r.name.startsWith("design_")),
  wolfram: results.filter(r => r.name.startsWith("wolfram_")),
  agency: results.filter(r => r.name.startsWith("agency_") && !r.name.includes("negentropy") && !r.name.includes("bateson") && !r.name.includes("scaffold") && !r.name.includes("motivation") && !r.name.includes("cognitive") && !r.name.includes("pedagogic") && !r.name.includes("population") && !r.name.includes("fitness") && !r.name.includes("l4_readiness") && !r.name.includes("memetic")),
  negentropy: results.filter(r => r.name.includes("negentropy") || r.name.includes("bateson") || r.name.includes("scaffold") || r.name.includes("motivation") || r.name.includes("cognitive") || r.name.includes("pedagogic")),
  evolution: results.filter(r => r.name.includes("population") || r.name.includes("fitness") || r.name.includes("l4_readiness") || r.name.includes("memetic")),
  vector: results.filter(r => r.name.startsWith("vector_")),
  cortex: results.filter(r => r.name.startsWith("cortex_")),
  stdp: results.filter(r => r.name.startsWith("stdp_")),
  sgnn: results.filter(r => r.name.startsWith("sgnn_")),
  swarm: results.filter(r => r.name.startsWith("swarm_")),
};

console.log("");
console.log("Category Summary:");
for (const [cat, items] of Object.entries(categories)) {
  const passCount = items.filter(i => i.status === "PASS").length;
  const realCount = items.filter(i => i.type === "REAL").length;
  console.log(`  ${cat.padEnd(10)}: ${passCount}/${items.length} passed, ${realCount} real implementations`);
}
