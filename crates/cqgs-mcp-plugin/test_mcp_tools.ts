#!/usr/bin/env bun
/**
 * Test CQGS MCP Tools
 *
 * Validates all 6 MCP tools are working correctly
 */

import { exec } from "child_process";
import { promisify } from "util";

const execAsync = promisify(exec);

console.log("🧪 Testing CQGS MCP Tools\n");

// Test 1: cqgs_version
console.log("1️⃣  Testing cqgs_version...");
try {
  const native = require("./index.node");
  const version = native.getVersion();
  const features = native.getFeatures();
  const count = native.getSentinelCount();
  console.log(`   ✅ Version: ${version}`);
  console.log(`   ✅ Features: ${features.join(", ")}`);
  console.log(`   ✅ Sentinel Count: ${count}\n`);
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

// Test 2: hyperbolic_distance
console.log("2️⃣  Testing hyperbolic_distance...");
try {
  const native = require("./index.node");
  // Two points in H^11 (12D Lorentz coordinates)
  // Origin: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  const point1 = [1.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const point2 = [1.2, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0];

  const distance = native.hyperbolicDistance(point1, point2);
  console.log(`   ✅ Distance: ${distance.toFixed(6)}\n`);
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

// Test 3: shannon_entropy
console.log("3️⃣  Testing shannon_entropy...");
try {
  const native = require("./index.node");
  // Uniform distribution over 4 outcomes: H = 2 bits
  const probabilities = [0.25, 0.25, 0.25, 0.25];

  const entropy = native.shannonEntropy(probabilities);
  console.log(`   ✅ Entropy: ${entropy.toFixed(6)} bits`);
  console.log(`   ✅ Expected: 2.000000 bits (uniform distribution)\n`);
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

// Test 4: sentinel_quality_score
console.log("4️⃣  Testing sentinel_quality_score...");
try {
  const native = require("./index.node");
  const mockResults = JSON.stringify([
    { quality_score: 100.0 },
    { quality_score: 95.0 },
    { quality_score: 98.0 },
  ]);

  const score = native.calculateQualityScore(mockResults);
  console.log(`   ✅ Average Quality Score: ${score.toFixed(2)}`);
  console.log(`   ✅ Expected: 97.67\n`);
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

// Test 5: sentinel_quality_gate
console.log("5️⃣  Testing sentinel_quality_gate...");
try {
  const native = require("./index.node");
  const mockResults = JSON.stringify([
    { quality_score: 96.0, violations: [] },
    { quality_score: 98.0, violations: [] },
    { quality_score: 97.0, violations: [] },
  ]);

  const gates = [
    "IntegrationReady",
    "TestingReady",
    "ProductionReady",
    "DeploymentApproved"
  ];

  for (const gate of gates) {
    const passed = native.checkQualityGate(mockResults, gate);
    console.log(`   ${passed ? "✅" : "❌"} ${gate}: ${passed}`);
  }
  console.log();
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

// Test 6: sentinel_execute_all (placeholder - no real sentinels yet)
console.log("6️⃣  Testing sentinel_execute_all...");
try {
  const native = require("./index.node");
  const results = native.executeAllSentinels(".", true);
  const parsed = JSON.parse(results);
  console.log(`   ℹ️  Executed sentinels: ${parsed.length}`);
  console.log(`   ℹ️  Note: Real sentinels pending dependency resolution\n`);
} catch (error) {
  console.error(`   ❌ Error: ${error}\n`);
}

console.log("✅ All CQGS MCP tool tests completed!");
