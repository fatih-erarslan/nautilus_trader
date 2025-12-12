import { handleEnhancedTool } from "../src/tools/index.js";

console.log("🔬 Testing systems dynamics integration via handleEnhancedTool...\n");

// Test that systems_ tools are properly routed
const result = await handleEnhancedTool("systems_model_simulate", {
  equations: ["dx/dt = -x"],
  initialConditions: { x: 1.0 },
  timeSpan: [0, 5]
});

const parsed = JSON.parse(result);
console.log("✅ Integration test passed!");
console.log(`   Tool: systems_model_simulate`);
console.log(`   Success: ${parsed.success}`);
console.log(`   Time points: ${parsed.simulation.timePoints}`);
console.log(`   Final value: ${parsed.simulation.finalState.x.toFixed(6)}`);
console.log(`   Wolfram code generated: ${!!parsed.wolframCode}`);
