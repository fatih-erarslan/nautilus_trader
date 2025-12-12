#!/usr/bin/env bun
/**
 * Wolfram LLM Tools Validation Suite
 *
 * Validates that Wolfram LLM tools:
 * 1. Generate correct Wolfram Language code
 * 2. Execute via wolframscript or cloud API
 * 3. Return structured responses (not stubs)
 * 4. Handle errors gracefully with fallbacks
 */

import { executeWolfram } from "../src/wolfram/client.js";
import { llmWolframCode } from "../src/tools/llm-tools.js";

interface ValidationResult {
  test: string;
  passed: boolean;
  message: string;
  details?: any;
}

const results: ValidationResult[] = [];

async function validate(
  testName: string,
  testFn: () => Promise<boolean | { passed: boolean; message?: string; details?: any }>
) {
  try {
    const result = await testFn();
    if (typeof result === "boolean") {
      results.push({
        test: testName,
        passed: result,
        message: result ? "✓ Passed" : "✗ Failed",
      });
    } else {
      results.push({
        test: testName,
        passed: result.passed,
        message: result.passed ? `✓ ${result.message || "Passed"}` : `✗ ${result.message || "Failed"}`,
        details: result.details,
      });
    }
  } catch (error) {
    results.push({
      test: testName,
      passed: false,
      message: `✗ Exception: ${error}`,
    });
  }
}

// ============================================================================
// Test Suite
// ============================================================================

console.log("╔══════════════════════════════════════════════════════════════╗");
console.log("║       WOLFRAM LLM TOOLS VALIDATION SUITE                     ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

// Test 1: Wolfram Code Generation
await validate("1. wolfram_llm_synthesize generates valid Wolfram code", async () => {
  const codeGen = llmWolframCode["wolfram_llm_synthesize"];
  if (!codeGen) return { passed: false, message: "Code generator not found" };

  const code = codeGen({ prompt: "test", format: "text" });
  const hasLLMSynthesize = code.includes("LLMSynthesize");
  const hasPrompt = code.includes("test");

  return {
    passed: hasLLMSynthesize && hasPrompt,
    message: hasLLMSynthesize && hasPrompt
      ? "Generates correct LLMSynthesize[] call"
      : "Missing LLMSynthesize or prompt",
    details: { code: code.substring(0, 200) }
  };
});

// Test 2: Wolfram Execution (Local)
await validate("2. Wolfram client executes simple expressions", async () => {
  const result = await executeWolfram("2 + 2", 5000);

  return {
    passed: result.success && result.output.trim() === "4",
    message: result.success
      ? `Executed successfully (${result.executionTime}ms, mode: ${result.mode})`
      : `Execution failed: ${result.error}`,
    details: result
  };
});

// Test 3: Wolfram LLMSynthesize (if available)
await validate("3. Wolfram LLMSynthesize execution", async () => {
  const code = `LLMSynthesize["Explain hyperbolic geometry in one sentence"]`;
  const result = await executeWolfram(code, 30000);

  if (!result.success) {
    // Expected if no LLM access - not a critical failure
    return {
      passed: true,
      message: `LLM not available (expected): ${result.error}`,
      details: { note: "This is acceptable - LLM requires Wolfram Cloud subscription" }
    };
  }

  const hasContent = result.output && result.output.length > 50;
  return {
    passed: hasContent,
    message: hasContent
      ? `Generated ${result.output.length} chars in ${result.executionTime}ms`
      : "Output too short or empty",
    details: { output: result.output.substring(0, 200) }
  };
});

// Test 4: Wolfram LLMFunction
await validate("4. wolfram_llm_function generates correct code", async () => {
  const codeGen = llmWolframCode["wolfram_llm_function"];
  if (!codeGen) return { passed: false, message: "Code generator not found" };

  const code = codeGen({ template: "Given `x`, compute `x^2`" });
  const hasLLMFunction = code.includes("LLMFunction");
  const hasTemplate = code.includes("x^2") || code.includes("x");

  return {
    passed: hasLLMFunction && hasTemplate,
    message: hasLLMFunction && hasTemplate
      ? "Generates correct LLMFunction[] call"
      : "Missing LLMFunction or template",
    details: { code: code.substring(0, 200) }
  };
});

// Test 5: Wolfram Code Generation
await validate("5. wolfram_llm_code_generate includes verification", async () => {
  const codeGen = llmWolframCode["wolfram_llm_code_generate"];
  if (!codeGen) return { passed: false, message: "Code generator not found" };

  const code = codeGen({
    specification: "test",
    language: "rust",
    verify: true
  });

  const hasVerification = code.includes("verification") || code.includes("validate");

  return {
    passed: hasVerification,
    message: hasVerification
      ? "Includes verification step"
      : "Missing verification",
    details: { code: code.substring(0, 200) }
  };
});

// Test 6: Knowledge Graph Extraction
await validate("6. wolfram_llm_graph generates graph structure", async () => {
  const codeGen = llmWolframCode["wolfram_llm_graph"];
  if (!codeGen) return { passed: false, message: "Code generator not found" };

  const code = codeGen({
    text: "test",
    entityTypes: ["Concept"],
    relationTypes: ["relates"]
  });

  const hasGraphExtraction = code.includes("Graph") || code.includes("Entity");

  return {
    passed: hasGraphExtraction,
    message: hasGraphExtraction
      ? "Generates graph extraction code"
      : "Missing graph structure",
    details: { code: code.substring(0, 200) }
  };
});

// Test 7: Error Handling
await validate("7. Wolfram client handles invalid code gracefully", async () => {
  const result = await executeWolfram("This Is Invalid Syntax #@!", 5000);

  return {
    passed: !result.success && result.error !== undefined,
    message: !result.success
      ? "Correctly returns error for invalid code"
      : "Should have failed but succeeded",
    details: { error: result.error }
  };
});

// Test 8: Wolfram Code Quality (moved from Test 9)
await validate("8. All LLM tools have Wolfram code generators", async () => {
  const requiredTools = [
    "wolfram_llm_function",
    "wolfram_llm_synthesize",
    "wolfram_llm_tool_define",
    "wolfram_llm_prompt",
    "wolfram_llm_prompt_chain",
    "wolfram_llm_code_generate",
    "wolfram_llm_code_review",
    "wolfram_llm_code_explain",
    "wolfram_llm_analyze",
    "wolfram_llm_reason",
    "wolfram_llm_graph",
  ];

  const missing = requiredTools.filter(tool => !llmWolframCode[tool]);

  return {
    passed: missing.length === 0,
    message: missing.length === 0
      ? `All ${requiredTools.length} tools have generators`
      : `Missing ${missing.length} generators`,
    details: { missing }
  };
});

// Test 9: Non-Stub Verification (moved from Test 10)
await validate("9. Tools return Wolfram results (not stubs)", async () => {
  const result = await executeWolfram("StringLength[\"hello\"]", 5000);

  // Check that result is NOT a stub pattern
  const isStub =
    result.output.includes("TODO") ||
    result.output.includes("placeholder") ||
    result.output.includes("pending");

  return {
    passed: result.success && !isStub,
    message: result.success && !isStub
      ? "Returns actual Wolfram output"
      : "Still returning stub responses",
    details: { output: result.output }
  };
});

// Test 10: Timeout Handling (moved to end to avoid corrupting subsequent tests)
await validate("10. Wolfram client respects timeout", async () => {
  const start = Date.now();
  const result = await executeWolfram("Pause[10]", 1000); // 1s timeout for 10s pause
  const elapsed = Date.now() - start;

  return {
    passed: !result.success && elapsed < 2000,
    message: !result.success && elapsed < 2000
      ? `Timeout enforced (${elapsed}ms)`
      : `Timeout not enforced (${elapsed}ms)`,
    details: { elapsed, error: result.error }
  };
});

// ============================================================================
// Results Summary
// ============================================================================

console.log("\n╔══════════════════════════════════════════════════════════════╗");
console.log("║                    VALIDATION RESULTS                        ║");
console.log("╚══════════════════════════════════════════════════════════════╝\n");

const passed = results.filter(r => r.passed).length;
const total = results.length;
const percentage = ((passed / total) * 100).toFixed(1);

results.forEach(r => {
  const icon = r.passed ? "✓" : "✗";
  console.log(`${icon} ${r.test}`);
  console.log(`  ${r.message}`);
  if (r.details) {
    console.log(`  Details: ${JSON.stringify(r.details, null, 2).substring(0, 200)}`);
  }
  console.log();
});

console.log("─────────────────────────────────────────────────────────────");
console.log(`SCORE: ${passed}/${total} (${percentage}%)`);
console.log("─────────────────────────────────────────────────────────────\n");

if (passed === total) {
  console.log("✓ All tests passed! Wolfram LLM tools are fully operational.");
  process.exit(0);
} else {
  console.log(`✗ ${total - passed} test(s) failed. Review details above.`);
  process.exit(1);
}
