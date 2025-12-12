#!/usr/bin/env bun
/**
 * Test Agency Tools Implementation
 *
 * Validates that all agency tool handlers work with both native and fallback implementations.
 */

import { handleAgencyTool } from "./src/tools/agency-tools.js";

// Mock native module (for testing without native bindings)
const mockNative = {
  hyperbolic_distance: (p1: number[], p2: number[]) => {
    const inner = -p1[0] * p2[0] + p1.slice(1).reduce((sum, x, i) => sum + x * p2[i + 1], 0);
    return Math.acosh(Math.max(-inner, 1.0));
  }
};

async function testAgencyTools() {
  console.log("=".repeat(80));
  console.log("Testing Agency Tools - TypeScript Fallback Implementation");
  console.log("=".repeat(80));

  // Test 1: Compute Free Energy
  console.log("\n[Test 1] agency_compute_free_energy");
  try {
    const result = await handleAgencyTool("agency_compute_free_energy", {
      observation: [0.3, 0.5, 0.2],
      beliefs: [0.4, 0.4, 0.2],
      precision: [1.0, 1.0, 1.0]
    }, mockNative);
    console.log("✓ Free Energy:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 2: Compute Survival Drive
  console.log("\n[Test 2] agency_compute_survival_drive");
  try {
    const position = [1.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // H^11 point
    const result = await handleAgencyTool("agency_compute_survival_drive", {
      free_energy: 1.5,
      position,
      strength: 1.0
    }, mockNative);
    console.log("✓ Survival Drive:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 3: Compute Phi (Consciousness)
  console.log("\n[Test 3] agency_compute_phi");
  try {
    const result = await handleAgencyTool("agency_compute_phi", {
      network_state: [0.8, 0.6, 0.3, 0.9],
      algorithm: "greedy"
    }, mockNative);
    console.log("✓ Phi (Consciousness):", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 4: Analyze Criticality
  console.log("\n[Test 4] agency_analyze_criticality");
  try {
    const activity = Array(100).fill(0).map((_, i) => Math.sin(i / 10) + Math.random() * 2);
    const result = await handleAgencyTool("agency_analyze_criticality", {
      activity_timeseries: activity,
      avalanche_threshold: 2.0
    }, mockNative);
    console.log("✓ Criticality:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 5: Regulate Homeostasis
  console.log("\n[Test 5] agency_regulate_homeostasis");
  try {
    const result = await handleAgencyTool("agency_regulate_homeostasis", {
      current_state: {
        phi: 1.2,
        free_energy: 0.8,
        survival: 0.6
      },
      setpoints: {
        phi_optimal: 1.0,
        free_energy_optimal: 1.0,
        survival_optimal: 0.5
      },
      sensors: [0.5, 0.6, 0.4, 0.5]
    }, mockNative);
    console.log("✓ Homeostasis:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 6: Update Beliefs
  console.log("\n[Test 6] agency_update_beliefs");
  try {
    const result = await handleAgencyTool("agency_update_beliefs", {
      observation: [0.7, 0.3, 0.5],
      beliefs: [0.5, 0.4, 0.6],
      precision: [1.0, 1.0, 1.0],
      learning_rate: 0.1
    }, mockNative);
    console.log("✓ Beliefs Update:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 7: Generate Action
  console.log("\n[Test 7] agency_generate_action");
  try {
    const result = await handleAgencyTool("agency_generate_action", {
      policy: [0.5, 0.3, 0.8],
      beliefs: [0.4, 0.5, 0.6],
      action_precision: 2.0
    }, mockNative);
    console.log("✓ Action Generation:", JSON.stringify(result, null, 2));
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  // Test 8: Create Agent
  console.log("\n[Test 8] agency_create_agent");
  try {
    const result = await handleAgencyTool("agency_create_agent", {
      config: {
        observation_dim: 10,
        action_dim: 5,
        hidden_dim: 8,
        learning_rate: 0.01
      },
      phi_calculator_type: "greedy"
    }, mockNative);
    console.log("✓ Agent Created:", JSON.stringify(result, null, 2));

    // Test 9: Agent Step
    if (result.agent_id) {
      console.log("\n[Test 9] agency_agent_step");
      const stepResult = await handleAgencyTool("agency_agent_step", {
        agent_id: result.agent_id,
        observation: Array(10).fill(0.5)
      }, mockNative);
      console.log("✓ Agent Step:", JSON.stringify(stepResult, null, 2));

      // Test 10: Get Agent Metrics
      console.log("\n[Test 10] agency_get_agent_metrics");
      const metricsResult = await handleAgencyTool("agency_get_agent_metrics", {
        agent_id: result.agent_id
      }, mockNative);
      console.log("✓ Agent Metrics:", JSON.stringify(metricsResult, null, 2));
    }
  } catch (e) {
    console.error("✗ Failed:", e);
  }

  console.log("\n" + "=".repeat(80));
  console.log("All Agency Tools Tests Completed!");
  console.log("=".repeat(80));
}

// Run tests
testAgencyTools().catch(console.error);
