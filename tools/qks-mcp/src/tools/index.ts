/**
 * QKS MCP Server - Tool Registry
 *
 * Exports all 64 tools across 8 cognitive layers
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ToolContext } from "../types.js";

// Layer 1: Thermodynamic (6 tools)
import { thermodynamicTools, handleThermodynamicTool } from "./thermodynamic.js";

// Layer 2: Cognitive (8 tools)
import { cognitiveTools, handleCognitiveTool } from "./cognitive.js";

// Layer 3: Decision (8 tools)
import { decisionTools, handleDecisionTool } from "./decision.js";

// Layer 4: Learning (8 tools)
import { learningTools, handleLearningTool } from "./learning.js";

// Layer 5: Collective (8 tools)
import { collectiveTools, handleCollectiveTool } from "./collective.js";

// Layer 6: Consciousness (8 tools)
import { consciousnessTools, handleConsciousnessTool } from "./consciousness.js";

// Layer 7: Metacognition (10 tools)
import { metacognitionTools, handleMetacognitionTool } from "./metacognition.js";

// Layer 8: Integration (8 tools)
import { integrationTools, handleIntegrationTool } from "./integration.js";

// Layer 9: Quantum Innovations (24 tools)
import { quantumTools, handleQuantumTool } from "./quantum.js";

// =============================================================================
// Tool Registry
// =============================================================================

export const allTools: Tool[] = [
  ...thermodynamicTools,      // 6 tools
  ...cognitiveTools,           // 8 tools
  ...decisionTools,            // 8 tools
  ...learningTools,            // 8 tools
  ...collectiveTools,          // 8 tools
  ...consciousnessTools,       // 8 tools
  ...metacognitionTools,       // 10 tools
  ...integrationTools,         // 8 tools
  ...quantumTools,             // 24 tools (NEW)
];

export const toolCategories = {
  thermodynamic: thermodynamicTools.map(t => t.name),
  cognitive: cognitiveTools.map(t => t.name),
  decision: decisionTools.map(t => t.name),
  learning: learningTools.map(t => t.name),
  collective: collectiveTools.map(t => t.name),
  consciousness: consciousnessTools.map(t => t.name),
  metacognition: metacognitionTools.map(t => t.name),
  integration: integrationTools.map(t => t.name),
  quantum: quantumTools.map(t => t.name),
};

export const totalToolCount = allTools.length;

// =============================================================================
// Tool Handler Router
// =============================================================================

export async function handleToolCall(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext
): Promise<any> {
  // Route to appropriate layer handler
  if (name.startsWith("qks_thermo_")) {
    return handleThermodynamicTool(name, args, context);
  }

  if (name.startsWith("qks_cognitive_")) {
    return handleCognitiveTool(name, args, context);
  }

  if (name.startsWith("qks_decision_")) {
    return handleDecisionTool(name, args, context);
  }

  if (name.startsWith("qks_learning_")) {
    return handleLearningTool(name, args, context);
  }

  if (name.startsWith("qks_collective_")) {
    return handleCollectiveTool(name, args, context);
  }

  if (name.startsWith("qks_consciousness_")) {
    return handleConsciousnessTool(name, args, context);
  }

  if (name.startsWith("qks_meta_")) {
    return handleMetacognitionTool(name, args, context);
  }

  if (name.startsWith("qks_system_") || name.startsWith("qks_cognitive_loop") ||
      name.startsWith("qks_homeostasis") || name.startsWith("qks_emergent_") ||
      name.startsWith("qks_orchestrate") || name.startsWith("qks_autopoiesis") ||
      name.startsWith("qks_criticality") || name.startsWith("qks_full_cycle")) {
    return handleIntegrationTool(name, args, context);
  }

  if (name.startsWith("qks_tensor_network_") || name.startsWith("qks_temporal_reservoir_") ||
      name.startsWith("qks_compressed_state_") || name.startsWith("qks_circuit_knitter_")) {
    return handleQuantumTool(name, args, context);
  }

  throw new Error(`Unknown tool: ${name}`);
}

// =============================================================================
// Tool Discovery
// =============================================================================

export function getToolsByLayer(layer: number): Tool[] {
  switch (layer) {
    case 1: return thermodynamicTools;
    case 2: return cognitiveTools;
    case 3: return decisionTools;
    case 4: return learningTools;
    case 5: return collectiveTools;
    case 6: return consciousnessTools;
    case 7: return metacognitionTools;
    case 8: return integrationTools;
    case 9: return quantumTools;
    default: throw new Error(`Invalid layer: ${layer}`);
  }
}

export function getToolsByPrefix(prefix: string): Tool[] {
  return allTools.filter(tool => tool.name.startsWith(prefix));
}

export function searchTools(query: string): Tool[] {
  const lowerQuery = query.toLowerCase();
  return allTools.filter(tool =>
    tool.name.toLowerCase().includes(lowerQuery) ||
    tool.description.toLowerCase().includes(lowerQuery)
  );
}

// =============================================================================
// Tool Statistics
// =============================================================================

export interface ToolStats {
  total_tools: number;
  tools_by_layer: Record<string, number>;
  categories: string[];
  layer_names: string[];
}

export function getToolStats(): ToolStats {
  return {
    total_tools: totalToolCount,
    tools_by_layer: {
      "L1_thermodynamic": thermodynamicTools.length,
      "L2_cognitive": cognitiveTools.length,
      "L3_decision": decisionTools.length,
      "L4_learning": learningTools.length,
      "L5_collective": collectiveTools.length,
      "L6_consciousness": consciousnessTools.length,
      "L7_metacognition": metacognitionTools.length,
      "L8_integration": integrationTools.length,
      "L9_quantum": quantumTools.length,
    },
    categories: Object.keys(toolCategories),
    layer_names: [
      "Thermodynamic Foundation",
      "Cognitive Architecture",
      "Decision Making",
      "Learning & Reasoning",
      "Collective Intelligence",
      "Consciousness",
      "Metacognition",
      "Full Agency Integration",
      "Quantum Innovations"
    ]
  };
}
