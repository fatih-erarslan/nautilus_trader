/**
 * Wolfram LLM Tools
 * 
 * Access Wolfram's LLM capabilities:
 * - LLMFunction: Create reusable LLM-powered functions
 * - LLMSynthesize: Generate content
 * - LLMTool: Define tools for LLM agents
 * - LLMPrompt: Structured prompting
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const llmTools: Tool[] = [
  // ============================================================================
  // LLM Function Creation
  // ============================================================================
  {
    name: "wolfram_llm_function",
    description: "Create a reusable LLM-powered function that can be called multiple times with different inputs.",
    inputSchema: {
      type: "object",
      properties: {
        template: { type: "string", description: "Prompt template with `` placeholders for arguments" },
        interpreter: { type: "string", description: "Output interpreter: String, Number, Boolean, Code, JSON, etc." },
        model: { type: "string", description: "LLM model to use (default: gpt-4)" },
      },
      required: ["template"],
    },
  },
  {
    name: "wolfram_llm_synthesize",
    description: "Generate content using Wolfram's LLMSynthesize - text, code, analysis, etc.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "What to synthesize" },
        context: { type: "string", description: "Additional context" },
        format: { type: "string", enum: ["text", "code", "json", "markdown"], description: "Output format" },
        model: { type: "string", description: "LLM model" },
        maxTokens: { type: "number", description: "Maximum output tokens" },
      },
      required: ["prompt"],
    },
  },
  {
    name: "wolfram_llm_tool_define",
    description: "Define a tool that can be used by LLM agents for function calling.",
    inputSchema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Tool name" },
        description: { type: "string", description: "Tool description for the LLM" },
        parameters: { 
          type: "array", 
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              type: { type: "string" },
              description: { type: "string" }
            }
          }
        },
        implementation: { type: "string", description: "Wolfram Language implementation" },
      },
      required: ["name", "description", "implementation"],
    },
  },

  // ============================================================================
  // Prompt Engineering
  // ============================================================================
  {
    name: "wolfram_llm_prompt",
    description: "Create structured prompts using Wolfram's LLMPrompt system.",
    inputSchema: {
      type: "object",
      properties: {
        role: { type: "string", description: "System role/persona" },
        task: { type: "string", description: "Task description" },
        examples: { type: "array", items: { type: "object" }, description: "Few-shot examples" },
        constraints: { type: "array", items: { type: "string" }, description: "Output constraints" },
        format: { type: "string", description: "Expected output format" },
      },
      required: ["task"],
    },
  },
  {
    name: "wolfram_llm_prompt_chain",
    description: "Create a chain of prompts for complex multi-step reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        steps: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              prompt: { type: "string" },
              dependsOn: { type: "array", items: { type: "string" } }
            }
          }
        },
        input: { type: "object", description: "Initial input data" },
      },
      required: ["steps"],
    },
  },

  // ============================================================================
  // Code Generation
  // ============================================================================
  {
    name: "wolfram_llm_code_generate",
    description: "Generate code in any language using LLM with Wolfram verification.",
    inputSchema: {
      type: "object",
      properties: {
        specification: { type: "string", description: "What the code should do" },
        language: { type: "string", description: "Target language: rust, python, swift, typescript, wolfram" },
        style: { type: "string", description: "Code style guidelines" },
        includeTests: { type: "boolean", description: "Generate tests alongside code" },
        verify: { type: "boolean", description: "Verify with Wolfram symbolic computation" },
      },
      required: ["specification", "language"],
    },
  },
  {
    name: "wolfram_llm_code_review",
    description: "Review code using LLM with Wolfram static analysis.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string", description: "Code to review" },
        language: { type: "string" },
        reviewCriteria: { type: "array", items: { type: "string" }, description: "What to check for" },
      },
      required: ["code"],
    },
  },
  {
    name: "wolfram_llm_code_explain",
    description: "Explain code in natural language.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string" },
        language: { type: "string" },
        detailLevel: { type: "string", enum: ["brief", "detailed", "tutorial"] },
      },
      required: ["code"],
    },
  },

  // ============================================================================
  // Analysis & Reasoning
  // ============================================================================
  {
    name: "wolfram_llm_analyze",
    description: "Perform deep analysis using LLM + Wolfram knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        topic: { type: "string", description: "Topic to analyze" },
        analysisType: { 
          type: "string", 
          enum: ["swot", "root_cause", "comparative", "trend", "risk", "opportunity"],
          description: "Type of analysis"
        },
        context: { type: "string" },
        depth: { type: "string", enum: ["shallow", "medium", "deep"] },
      },
      required: ["topic", "analysisType"],
    },
  },
  {
    name: "wolfram_llm_reason",
    description: "Multi-step reasoning with chain-of-thought and verification.",
    inputSchema: {
      type: "object",
      properties: {
        question: { type: "string", description: "Question to reason about" },
        method: { type: "string", enum: ["chain_of_thought", "tree_of_thought", "self_consistency"] },
        verifySteps: { type: "boolean", description: "Verify each step with Wolfram" },
      },
      required: ["question"],
    },
  },

  // ============================================================================
  // Graph-based LLM
  // ============================================================================
  {
    name: "wolfram_llm_graph",
    description: "Create knowledge graphs from text using LLM extraction.",
    inputSchema: {
      type: "object",
      properties: {
        text: { type: "string", description: "Text to extract knowledge from" },
        entityTypes: { type: "array", items: { type: "string" }, description: "Types of entities to extract" },
        relationTypes: { type: "array", items: { type: "string" }, description: "Types of relations to extract" },
      },
      required: ["text"],
    },
  },
];

export const llmWolframCode: Record<string, (args: any) => string> = {
  "wolfram_llm_synthesize": (args) => {
    const prompt = args.prompt?.replace(/"/g, '\\"') || '';
    const model = args.model || "gpt-4";
    return `LLMSynthesize["${prompt}", LLMEvaluator -> <|"Model" -> "${model}"|>]`;
  },
  
  "wolfram_llm_function": (args) => {
    const template = args.template?.replace(/"/g, '\\"') || '';
    const interpreter = args.interpreter || "String";
    return `LLMFunction["${template}", ${interpreter}]`;
  },
  
  "wolfram_llm_code_generate": (args) => {
    const spec = args.specification?.replace(/"/g, '\\"') || '';
    const lang = args.language || 'python';
    return `LLMSynthesize["Generate ${lang} code for: ${spec}. Include comments and type hints."]`;
  },
  
  "wolfram_llm_code_review": (args) => {
    const code = args.code?.replace(/"/g, '\\"').replace(/\n/g, '\\n') || '';
    return `LLMSynthesize["Review this code for bugs, security issues, and improvements:\\n${code}"]`;
  },
  
  "wolfram_llm_graph": (args) => {
    const text = args.text?.replace(/"/g, '\\"') || '';
    return `Module[{entities, relations},
      entities = TextCases["${text}", "Entity"];
      relations = LLMSynthesize["Extract relationships between entities in: ${text}. Format as JSON array."];
      <|"entities" -> entities, "relations" -> relations|>
    ] // ToString`;
  },
  
  "wolfram_llm_analyze": (args) => {
    const topic = args.topic?.replace(/"/g, '\\"') || '';
    const type = args.analysisType || 'swot';
    return `LLMSynthesize["Perform ${type} analysis on: ${topic}. Be thorough and use data when available."]`;
  },
  
  "wolfram_llm_reason": (args) => {
    const question = args.question?.replace(/"/g, '\\"') || '';
    const method = args.method || 'chain_of_thought';
    return `LLMSynthesize["Using ${method} reasoning, answer: ${question}. Show your step-by-step reasoning."]`;
  },
};
