/**
 * Design Thinking Tools
 * 
 * Embed the complete Design Thinking cyclical methodology:
 * Empathize → Define → Ideate → Prototype → Test → (iterate)
 * 
 * Each phase has Wolfram-powered analysis capabilities.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";

export const designThinkingTools: Tool[] = [
  // ============================================================================
  // EMPATHIZE Phase
  // ============================================================================
  {
    name: "design_empathize_analyze",
    description: "Analyze user needs, pain points, and context using Wolfram NLP and data analysis. Input user research data, interviews, or observations.",
    inputSchema: {
      type: "object",
      properties: {
        userResearch: { type: "string", description: "User research notes, interview transcripts, or observations" },
        stakeholders: { type: "array", items: { type: "string" }, description: "List of stakeholder groups" },
        context: { type: "string", description: "Problem context and domain" },
      },
      required: ["userResearch"],
    },
  },
  {
    name: "design_empathize_persona",
    description: "Generate user personas from research data using clustering and pattern analysis.",
    inputSchema: {
      type: "object",
      properties: {
        userData: { type: "array", items: { type: "object" }, description: "User data points" },
        clusterCount: { type: "number", description: "Number of persona clusters (default: 3)" },
      },
      required: ["userData"],
    },
  },

  // ============================================================================
  // DEFINE Phase
  // ============================================================================
  {
    name: "design_define_problem",
    description: "Define the problem statement using structured analysis. Generates 'How Might We' statements.",
    inputSchema: {
      type: "object",
      properties: {
        insights: { type: "array", items: { type: "string" }, description: "Key insights from empathize phase" },
        constraints: { type: "array", items: { type: "string" }, description: "Known constraints" },
        goals: { type: "array", items: { type: "string" }, description: "Desired outcomes" },
      },
      required: ["insights"],
    },
  },
  {
    name: "design_define_requirements",
    description: "Extract and prioritize requirements using graph-based dependency analysis.",
    inputSchema: {
      type: "object",
      properties: {
        problemStatement: { type: "string" },
        features: { type: "array", items: { type: "string" } },
        priorities: { type: "array", items: { type: "number" }, description: "Priority weights" },
      },
      required: ["problemStatement", "features"],
    },
  },

  // ============================================================================
  // IDEATE Phase
  // ============================================================================
  {
    name: "design_ideate_brainstorm",
    description: "Generate solution ideas using LLM-powered divergent thinking and analogical reasoning.",
    inputSchema: {
      type: "object",
      properties: {
        problemStatement: { type: "string" },
        constraints: { type: "array", items: { type: "string" } },
        inspirationDomains: { type: "array", items: { type: "string" }, description: "Domains to draw analogies from" },
        ideaCount: { type: "number", description: "Number of ideas to generate (default: 10)" },
      },
      required: ["problemStatement"],
    },
  },
  {
    name: "design_ideate_evaluate",
    description: "Evaluate and rank ideas using multi-criteria decision analysis.",
    inputSchema: {
      type: "object",
      properties: {
        ideas: { type: "array", items: { type: "string" } },
        criteria: { type: "array", items: { type: "string" }, description: "Evaluation criteria" },
        weights: { type: "array", items: { type: "number" }, description: "Criteria weights" },
      },
      required: ["ideas", "criteria"],
    },
  },

  // ============================================================================
  // PROTOTYPE Phase
  // ============================================================================
  {
    name: "design_prototype_architecture",
    description: "Generate system architecture from requirements using graph modeling.",
    inputSchema: {
      type: "object",
      properties: {
        requirements: { type: "array", items: { type: "string" } },
        components: { type: "array", items: { type: "string" } },
        style: { type: "string", enum: ["microservices", "monolith", "serverless", "hybrid"] },
      },
      required: ["requirements"],
    },
  },
  {
    name: "design_prototype_code",
    description: "Generate prototype code scaffolding using LLM code synthesis.",
    inputSchema: {
      type: "object",
      properties: {
        architecture: { type: "object", description: "Architecture specification" },
        language: { type: "string", description: "Target language (rust, swift, typescript, python)" },
        framework: { type: "string", description: "Target framework" },
      },
      required: ["architecture", "language"],
    },
  },

  // ============================================================================
  // TEST Phase
  // ============================================================================
  {
    name: "design_test_generate",
    description: "Generate test cases using property-based testing and boundary analysis.",
    inputSchema: {
      type: "object",
      properties: {
        specification: { type: "string", description: "Functional specification" },
        testTypes: { type: "array", items: { type: "string" }, description: "Test types: unit, integration, e2e, property" },
        coverageTarget: { type: "number", description: "Target coverage percentage" },
      },
      required: ["specification"],
    },
  },
  {
    name: "design_test_analyze",
    description: "Analyze test results and identify failure patterns.",
    inputSchema: {
      type: "object",
      properties: {
        testResults: { type: "array", items: { type: "object" }, description: "Test result data" },
        threshold: { type: "number", description: "Failure threshold percentage" },
      },
      required: ["testResults"],
    },
  },

  // ============================================================================
  // ITERATE Phase (Cross-cutting)
  // ============================================================================
  {
    name: "design_iterate_feedback",
    description: "Analyze feedback to guide next iteration using sentiment and theme analysis.",
    inputSchema: {
      type: "object",
      properties: {
        feedback: { type: "array", items: { type: "string" } },
        currentPhase: { type: "string", enum: ["empathize", "define", "ideate", "prototype", "test"] },
      },
      required: ["feedback"],
    },
  },
  {
    name: "design_iterate_metrics",
    description: "Track design thinking metrics across iterations.",
    inputSchema: {
      type: "object",
      properties: {
        iteration: { type: "number" },
        metrics: { type: "object", description: "Key metrics for this iteration" },
      },
      required: ["iteration", "metrics"],
    },
  },
];

export const designThinkingWolframCode: Record<string, (args: any) => string> = {
  "design_empathize_analyze": (args) => `
    Module[{text, themes, sentiment},
      text = "${args.userResearch?.replace(/"/g, '\\"') || ''}";
      themes = TextCases[text, "Concept"];
      sentiment = Classify["Sentiment", text];
      <|
        "keyThemes" -> Take[Tally[themes] // SortBy[#, -Last[#]&], UpTo[10]],
        "sentiment" -> sentiment,
        "wordCloud" -> ToString[WordCloud[text]],
        "entities" -> TextCases[text, "Entity"]
      |>
    ] // ToString
  `,
  
  "design_ideate_brainstorm": (args) => `
    Module[{problem, ideas},
      problem = "${args.problemStatement?.replace(/"/g, '\\"') || ''}";
      ideas = Table[
        StringJoin["Idea ", ToString[i], ": ", 
          LLMSynthesize["Generate a creative solution for: " <> problem, 
            LLMEvaluator -> <|"Model" -> "gpt-4"|>]
        ],
        {i, ${args.ideaCount || 5}}
      ];
      ideas
    ] // ToString
  `,
  
  "design_test_generate": (args) => `
    Module[{spec, tests},
      spec = "${args.specification?.replace(/"/g, '\\"') || ''}";
      tests = {
        "unitTests" -> LLMSynthesize["Generate unit tests for: " <> spec],
        "edgeCases" -> LLMSynthesize["Identify edge cases for: " <> spec],
        "propertyTests" -> LLMSynthesize["Generate property-based tests for: " <> spec]
      };
      tests
    ] // ToString
  `,
};
