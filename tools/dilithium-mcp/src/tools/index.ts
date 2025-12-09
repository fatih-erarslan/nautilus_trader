/**
 * Dilithium MCP Enhanced Tools Index
 * 
 * Complete Enterprise Development Pipeline:
 * - Design Thinking (empathize, define, ideate, prototype, test)
 * - Systems Dynamics (modeling, equilibrium, control, feedback)
 * - LLM Tools (synthesize, function, code generation)
 * - Dilithium Auth (client registration, authorization)
 * - DevOps Pipeline (CI/CD, deployment, observability)
 * - Project Management (sprint, estimation, backlog)
 * - Documentation (API docs, ADRs, runbooks)
 * - Code Quality (analysis, refactoring, tech debt)
 */

export { designThinkingTools, designThinkingWolframCode } from "./design-thinking.js";
export { systemsDynamicsTools, systemsDynamicsWolframCode } from "./systems-dynamics.js";
export { llmTools, llmWolframCode } from "./llm-tools.js";
export { dilithiumAuthTools, handleDilithiumAuth } from "../auth/dilithium-sentry.js";
export { devopsPipelineTools, devopsPipelineWolframCode } from "./devops-pipeline.js";
export { projectManagementTools, projectManagementWolframCode } from "./project-management.js";
export { documentationTools, documentationWolframCode } from "./documentation.js";
export { codeQualityTools, codeQualityWolframCode } from "./code-quality.js";

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { designThinkingTools } from "./design-thinking.js";
import { systemsDynamicsTools } from "./systems-dynamics.js";
import { llmTools } from "./llm-tools.js";
import { dilithiumAuthTools } from "../auth/dilithium-sentry.js";
import { devopsPipelineTools } from "./devops-pipeline.js";
import { projectManagementTools } from "./project-management.js";
import { documentationTools } from "./documentation.js";
import { codeQualityTools } from "./code-quality.js";

/**
 * All enhanced tools combined - Complete Enterprise Pipeline
 */
export const enhancedTools: Tool[] = [
  ...designThinkingTools,      // 12 tools
  ...systemsDynamicsTools,     // 13 tools
  ...llmTools,                 // 11 tools
  ...dilithiumAuthTools,       // 7 tools
  ...devopsPipelineTools,      // 19 tools
  ...projectManagementTools,   // 13 tools
  ...documentationTools,       // 14 tools
  ...codeQualityTools,         // 16 tools
];

/**
 * Tool categories for discovery
 */
export const toolCategories = {
  designThinking: {
    name: "Design Thinking",
    description: "Cyclical development methodology: Empathize → Define → Ideate → Prototype → Test",
    tools: designThinkingTools.map(t => t.name),
    count: designThinkingTools.length,
  },
  systemsDynamics: {
    name: "Systems Dynamics",
    description: "System modeling, equilibrium analysis, control theory, feedback loops",
    tools: systemsDynamicsTools.map(t => t.name),
    count: systemsDynamicsTools.length,
  },
  llm: {
    name: "LLM Tools",
    description: "LLM capabilities: synthesize, function creation, code generation",
    tools: llmTools.map(t => t.name),
    count: llmTools.length,
  },
  auth: {
    name: "Dilithium Authorization",
    description: "Post-quantum secure client authorization for API access",
    tools: dilithiumAuthTools.map(t => t.name),
    count: dilithiumAuthTools.length,
  },
  devops: {
    name: "DevOps Pipeline",
    description: "CI/CD, deployment strategies, observability, infrastructure as code",
    tools: devopsPipelineTools.map(t => t.name),
    count: devopsPipelineTools.length,
  },
  projectManagement: {
    name: "Project Management",
    description: "Sprint planning, estimation, backlog management, DORA metrics",
    tools: projectManagementTools.map(t => t.name),
    count: projectManagementTools.length,
  },
  documentation: {
    name: "Documentation",
    description: "API docs, architecture diagrams, ADRs, runbooks, knowledge base",
    tools: documentationTools.map(t => t.name),
    count: documentationTools.length,
  },
  codeQuality: {
    name: "Code Quality",
    description: "Static analysis, refactoring, technical debt, code health metrics",
    tools: codeQualityTools.map(t => t.name),
    count: codeQualityTools.length,
  },
};

/**
 * Total tool count
 */
export const totalToolCount = enhancedTools.length;

/**
 * Handle enhanced tool calls
 */
export function handleEnhancedTool(name: string, args: Record<string, unknown>): string {
  // Design thinking, systems dynamics, llm tools handlers would go here
  // For now, return a placeholder
  return JSON.stringify({ 
    tool: name, 
    args, 
    status: "processed",
    message: "Tool handled by enhanced tools module"
  });
}
