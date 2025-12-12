#!/usr/bin/env bun
/**
 * Wolfram LLM Tools Integration Test
 *
 * Tests the complete pipeline:
 * 1. Tool handler receives request
 * 2. Wolfram code is generated
 * 3. Wolfram execution (local or cloud)
 * 4. Result is structured and returned
 */

import { handleLlmTool } from "./src/tools/llm-tools.js";

console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║         WOLFRAM LLM TOOLS INTEGRATION TEST                   ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

// Test 1: wolfram_llm_synthesize
console.log("Test 1: wolfram_llm_synthesize");
console.log("─────────────────────────────────────────────────────────────");
const synthesizeArgs = {
  prompt: "Explain the Lorentz transformation in special relativity",
  format: "text",
  maxTokens: 500,
};

try {
  const result1 = await handleLlmTool("wolfram_llm_synthesize", synthesizeArgs);
  console.log("✓ Result:", JSON.stringify(result1, null, 2));
} catch (error) {
  console.error("✗ Error:", error);
}

console.log("\n");

// Test 2: wolfram_llm_code_generate
console.log("Test 2: wolfram_llm_code_generate");
console.log("─────────────────────────────────────────────────────────────");
const codeGenArgs = {
  specification: "Write a function to compute hyperbolic distance in the Poincaré ball model",
  language: "rust",
  includeTests: true,
  verify: true,
};

try {
  const result2 = await handleLlmTool("wolfram_llm_code_generate", codeGenArgs);
  console.log("✓ Result:", JSON.stringify(result2, null, 2));
} catch (error) {
  console.error("✗ Error:", error);
}

console.log("\n");

// Test 3: wolfram_llm_function
console.log("Test 3: wolfram_llm_function");
console.log("─────────────────────────────────────────────────────────────");
const functionArgs = {
  template: "Given a matrix `M`, compute its eigenvalues",
  interpreter: "JSON",
};

try {
  const result3 = await handleLlmTool("wolfram_llm_function", functionArgs);
  console.log("✓ Result:", JSON.stringify(result3, null, 2));
} catch (error) {
  console.error("✗ Error:", error);
}

console.log("\n");

// Test 4: wolfram_llm_graph
console.log("Test 4: wolfram_llm_graph");
console.log("─────────────────────────────────────────────────────────────");
const graphArgs = {
  text: "The Free Energy Principle states that biological systems minimize variational free energy. This involves active inference and precision-weighted prediction errors.",
  entityTypes: ["Concept", "Process"],
  relationTypes: ["implements", "involves"],
};

try {
  const result4 = await handleLlmTool("wolfram_llm_graph", graphArgs);
  console.log("✓ Result:", JSON.stringify(result4, null, 2));
} catch (error) {
  console.error("✗ Error:", error);
}

console.log("\n╔══════════════════════════════════════════════════════════════╗");
console.log("║                     TEST COMPLETE                            ║");
console.log("╚══════════════════════════════════════════════════════════════╝");
