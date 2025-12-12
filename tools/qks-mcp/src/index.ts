#!/usr/bin/env bun
/**
 * QKS MCP Server v2.0
 *
 * Quantum Knowledge System - 8-Layer Cognitive Architecture for Agentic AI
 *
 * Architecture: Rust-Bun.js Bridge with 64 MCP Tools
 *
 * Layer 1: Thermodynamic Foundation (6 tools)
 *   - Energy, temperature, entropy, critical points, Landauer cost, free energy
 *
 * Layer 2: Cognitive Architecture (8 tools)
 *   - Attention, memory (working/episodic/semantic), pattern matching, perception
 *
 * Layer 3: Decision Making (8 tools)
 *   - Active inference, EFE, policy selection, belief updates, precision weighting
 *
 * Layer 4: Learning & Reasoning (8 tools)
 *   - STDP, consolidation, transfer learning, curriculum, meta-learning, reasoning
 *
 * Layer 5: Collective Intelligence (8 tools)
 *   - Swarm coordination, consensus, stigmergy, distributed memory, emergence
 *
 * Layer 6: Consciousness (8 tools)
 *   - IIT Φ, global workspace, phase coherence, integration, complexity, qualia
 *
 * Layer 7: Metacognition (10 tools)
 *   - Introspection, self-model, confidence, meta-learning, strategy selection
 *
 * Layer 8: Full Agency (8 tools)
 *   - System health, cognitive loop, homeostasis, orchestration, autopoiesis
 *
 * Environment Variables:
 *   - QKS_NATIVE_PATH: Path to Rust native module
 *   - QKS_MCP_PORT: Server port (default: 3002)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { rustBridge, isNativeAvailable, getNativeModulePath } from "./bridge.js";
import { allTools, handleToolCall, getToolStats } from "./tools/index.js";

// =============================================================================
// MCP Server Setup
// =============================================================================

const server = new Server(
  {
    name: "qks-mcp",
    version: "2.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// =============================================================================
// Request Handlers
// =============================================================================

/**
 * List all available tools
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: allTools };
});

/**
 * Handle tool execution
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    const result = await handleToolCall(name, args as Record<string, unknown>, {
      rustBridge,
      config: {}
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error),
          }),
        },
      ],
      isError: true,
    };
  }
});

// =============================================================================
// Server Lifecycle
// =============================================================================

async function main() {
  const stats = getToolStats();
  const nativePath = getNativeModulePath();

  console.error("╔══════════════════════════════════════════════════════════════╗");
  console.error("║       QKS MCP SERVER v2.0 - 8-Layer Cognitive Architecture   ║");
  console.error("║            Quantum Knowledge System for Agentic AI           ║");
  console.error("╚══════════════════════════════════════════════════════════════╝");
  console.error("");
  console.error("  Architecture: 8-Layer Cognitive System");
  console.error(`  Native Module: ${isNativeAvailable() ? "✓ Loaded" : "✗ Using TypeScript fallback"}`);
  if (nativePath) {
    console.error(`  Native Path: ${nativePath}`);
  }
  console.error(`  Total Tools: ${stats.total_tools}`);
  console.error("");

  console.error("  Tool Distribution by Layer:");
  console.error(`    L1 Thermodynamic:     ${stats.tools_by_layer.L1_thermodynamic} tools`);
  console.error(`    L2 Cognitive:         ${stats.tools_by_layer.L2_cognitive} tools`);
  console.error(`    L3 Decision:          ${stats.tools_by_layer.L3_decision} tools`);
  console.error(`    L4 Learning:          ${stats.tools_by_layer.L4_learning} tools`);
  console.error(`    L5 Collective:        ${stats.tools_by_layer.L5_collective} tools`);
  console.error(`    L6 Consciousness:     ${stats.tools_by_layer.L6_consciousness} tools`);
  console.error(`    L7 Metacognition:     ${stats.tools_by_layer.L7_metacognition} tools`);
  console.error(`    L8 Integration:       ${stats.tools_by_layer.L8_integration} tools`);
  console.error("");

  if (!isNativeAvailable()) {
    console.error("  ⚠️  WARNING: Running without native Rust module");
    console.error("  ⚠️  Using TypeScript fallback implementations");
    console.error("  ⚠️  Build native module for production performance");
    console.error("  ⚠️  Run: cd ../../rust-core && cargo build --release");
    console.error("");
  }

  console.error("  Capabilities:");
  console.error("    • Thermodynamic optimization & energy management");
  console.error("    • Attention, memory, and pattern recognition");
  console.error("    • Active inference & decision making");
  console.error("    • STDP learning & memory consolidation");
  console.error("    • Swarm intelligence & consensus protocols");
  console.error("    • IIT Φ consciousness metrics");
  console.error("    • Meta-learning & introspection");
  console.error("    • Full cybernetic agency & homeostasis");
  console.error("");

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("  [Ready] QKS MCP Server listening on stdio transport");
  console.error("");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
