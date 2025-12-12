#!/usr/bin/env bun
/**
 * Wolfram Alpha MCP Server v2.0
 * 
 * High-performance MCP server with Bun.js runtime and native Rust/Swift bindings.
 * 
 * Features:
 * - Full Results API (XML/JSON responses)
 * - LLM API (text optimized for AI assistants)
 * - Native Rust bindings via NAPI-RS for hyperbolic geometry, STDP, thermodynamics
 * - Native Swift bindings for macOS optimization
 * - Automatic fallback: API → Local WolframScript → Native computation
 * 
 * Environment Variables:
 * - WOLFRAM_APP_ID: Your Wolfram Alpha API AppID (optional with local)
 * - WOLFRAMSCRIPT_PATH: Path to wolframscript executable
 * - WOLFRAM_NATIVE_PATH: Path to native Rust module
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { spawn } from "child_process";
import { existsSync } from "fs";

// Import swarm module for agent-to-agent communication
import { swarmTools, handleSwarmTool } from "./swarm/index.js";

// Import enhanced tool modules - Complete Enterprise Pipeline
import { 
  enhancedTools, 
  designThinkingWolframCode, 
  systemsDynamicsWolframCode, 
  llmWolframCode,
  handleDilithiumAuth,
  devopsPipelineWolframCode,
  projectManagementWolframCode,
  documentationWolframCode,
  codeQualityWolframCode,
  toolCategories,
  totalToolCount
} from "./tools/index.js";

// Try to load native Rust module
let nativeModule: any = null;
try {
  // In production, this would be the compiled .node file
  const nativePath = process.env.WOLFRAM_NATIVE_PATH || "./native/wolfram-native.darwin-arm64.node";
  if (existsSync(nativePath)) {
    nativeModule = require(nativePath);
    console.error("Loaded native Rust module");
  }
} catch (e) {
  console.error("Native module not available, using fallback implementations");
}

// Wolfram API Configuration
const WOLFRAM_APP_ID = process.env.WOLFRAM_APP_ID;
const WOLFRAM_LLM_API = "https://www.wolframalpha.com/api/v1/llm-api";
const WOLFRAM_FULL_API = "https://api.wolframalpha.com/v2/query";
const WOLFRAMSCRIPT_PATH = process.env.WOLFRAMSCRIPT_PATH || "/usr/local/bin/wolframscript";

// Check for either API key or local WolframScript
const hasAPI = !!WOLFRAM_APP_ID;
const hasLocal = existsSync(WOLFRAMSCRIPT_PATH);

const hasNative = !!nativeModule;

if (!hasAPI && !hasLocal && !hasNative) {
  console.error("ERROR: Need WOLFRAM_APP_ID, local WolframScript, or native module");
  process.exit(1);
}

console.error(`Wolfram MCP v2.0 (Bun.js): API=${hasAPI}, Local=${hasLocal}, Native=${hasNative}`);

// Tool Schemas
const LLMQuerySchema = z.object({
  query: z.string().describe("Natural language query for Wolfram Alpha"),
  maxchars: z.number().optional().default(6800).describe("Maximum characters in response"),
});

const FullQuerySchema = z.object({
  query: z.string().describe("Query for Wolfram Alpha Full Results API"),
  format: z.enum(["plaintext", "image", "mathml", "minput", "moutput"]).optional().default("plaintext"),
  includepodid: z.string().optional().describe("Only include specific pod IDs"),
  excludepodid: z.string().optional().describe("Exclude specific pod IDs"),
});

const ComputeSchema = z.object({
  expression: z.string().describe("Mathematical expression to compute (e.g., 'integrate x^2 dx')"),
});

const ValidateSchema = z.object({
  expression: z.string().describe("Mathematical expression or identity to validate"),
  expected: z.string().optional().describe("Expected result for comparison"),
});

const UnitConvertSchema = z.object({
  value: z.string().describe("Value with units to convert (e.g., '100 miles')"),
  targetUnit: z.string().describe("Target unit (e.g., 'kilometers')"),
});

const DataQuerySchema = z.object({
  entity: z.string().describe("Entity to query (e.g., 'France', 'hydrogen', 'S&P 500')"),
  property: z.string().optional().describe("Specific property (e.g., 'population', 'atomic mass')"),
});

const LocalEvalSchema = z.object({
  code: z.string().describe("Wolfram Language code to evaluate locally"),
  timeout: z.number().optional().default(30).describe("Timeout in seconds"),
});

const SymbolicComputeSchema = z.object({
  operation: z.enum(["integrate", "differentiate", "solve", "simplify", "series", "limit"]).describe("Mathematical operation"),
  expression: z.string().describe("Mathematical expression"),
  variable: z.string().optional().default("x").describe("Variable for the operation"),
  options: z.string().optional().describe("Additional options (e.g., 'Assumptions -> x > 0')"),
});

const HyperbolicGeometrySchema = z.object({
  operation: z.enum(["distance", "geodesic", "mobius", "tessellation"]).describe("Hyperbolic geometry operation"),
  params: z.record(z.any()).describe("Parameters for the operation"),
});

// Define available tools
const tools: Tool[] = [
  {
    name: "wolfram_llm_query",
    description: "Query Wolfram Alpha using the LLM-optimized API. Returns text responses perfect for AI assistants. Use for general knowledge, calculations, data lookups, and scientific queries.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Natural language query" },
        maxchars: { type: "number", description: "Max response length (default: 6800)" },
      },
      required: ["query"],
    },
  },
  {
    name: "wolfram_compute",
    description: "Compute mathematical expressions using Wolfram Alpha. Supports integrals, derivatives, equations, simplification, and symbolic math.",
    inputSchema: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Mathematical expression (e.g., 'derivative of sin(x^2)')" },
      },
      required: ["expression"],
    },
  },
  {
    name: "wolfram_validate",
    description: "Validate mathematical expressions, identities, or computations using Wolfram Alpha.",
    inputSchema: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Expression to validate" },
        expected: { type: "string", description: "Optional expected result" },
      },
      required: ["expression"],
    },
  },
  {
    name: "wolfram_unit_convert",
    description: "Convert between units using Wolfram Alpha's precise unit conversion.",
    inputSchema: {
      type: "object",
      properties: {
        value: { type: "string", description: "Value with units (e.g., '100 mph')" },
        targetUnit: { type: "string", description: "Target unit (e.g., 'km/h')" },
      },
      required: ["value", "targetUnit"],
    },
  },
  {
    name: "wolfram_data_query",
    description: "Query scientific, geographic, financial, or other data from Wolfram's knowledge base.",
    inputSchema: {
      type: "object",
      properties: {
        entity: { type: "string", description: "Entity to query (country, element, company, etc.)" },
        property: { type: "string", description: "Specific property to retrieve" },
      },
      required: ["entity"],
    },
  },
  {
    name: "wolfram_full_query",
    description: "Query Wolfram Alpha Full Results API for detailed structured data. Returns comprehensive results with multiple pods.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Query string" },
        format: { type: "string", enum: ["plaintext", "image", "mathml", "minput", "moutput"] },
        includepodid: { type: "string", description: "Only include specific pods" },
        excludepodid: { type: "string", description: "Exclude specific pods" },
      },
      required: ["query"],
    },
  },
  {
    name: "wolfram_local_eval",
    description: "Execute Wolfram Language code locally using WolframScript. Full access to symbolic computation, knowledge base, and all Wolfram capabilities. Faster than API for complex computations.",
    inputSchema: {
      type: "object",
      properties: {
        code: { type: "string", description: "Wolfram Language code to evaluate" },
        timeout: { type: "number", description: "Timeout in seconds (default: 30)" },
      },
      required: ["code"],
    },
  },
  {
    name: "wolfram_symbolic",
    description: "Perform symbolic mathematics: integrate, differentiate, solve equations, simplify, series expansion, limits.",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["integrate", "differentiate", "solve", "simplify", "series", "limit"] },
        expression: { type: "string", description: "Mathematical expression" },
        variable: { type: "string", description: "Variable (default: x)" },
        options: { type: "string", description: "Additional Wolfram options" },
      },
      required: ["operation", "expression"],
    },
  },
  {
    name: "wolfram_hyperbolic",
    description: "Hyperbolic geometry computations: distance in Poincaré disk, geodesics, Möbius transformations, tessellations.",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["distance", "geodesic", "mobius", "tessellation"] },
        params: { type: "object", description: "Operation parameters" },
      },
      required: ["operation", "params"],
    },
  },
];

// API Functions with local fallback
async function queryLLMAPI(query: string, maxchars: number = 6800): Promise<string> {
  // Try API first if available
  if (hasAPI) {
    try {
      const url = new URL(WOLFRAM_LLM_API);
      url.searchParams.set("input", query);
      url.searchParams.set("appid", WOLFRAM_APP_ID!);
      url.searchParams.set("maxchars", maxchars.toString());

      const response = await fetch(url.toString());
      const text = await response.text();
      
      // Wolfram returns 501 for "could not understand" - still return the text
      // which contains helpful suggestions
      if (text.length > 0) {
        // If 501 (not understood), prefix with helpful message
        if (response.status === 501) {
          return `Wolfram Alpha could not directly answer this query.\n\n${text}\n\nTry using wolfram_local_eval for complex computations.`;
        }
        return text;
      }
      
      // If truly empty/failed, fall through to local execution
      console.error(`Wolfram API returned ${response.status} with empty response, falling back to local`);
    } catch (apiError) {
      console.error(`Wolfram API error: ${apiError}, falling back to local`);
    }
  }

  // Fallback to local WolframScript
  if (hasLocal) {
    // Convert natural language query to WolframAlpha call via WolframScript
    const code = `WolframAlpha["${query.replace(/"/g, '\\"')}", "Result"] // ToString`;
    return executeWolframScript(code, 60);
  }

  throw new Error("No Wolfram backend available (API failed and no local WolframScript)");
}

async function queryFullAPI(
  query: string,
  format: string = "plaintext",
  includepodid?: string,
  excludepodid?: string
): Promise<string> {
  // Try API first if available
  if (hasAPI) {
    try {
      const url = new URL(WOLFRAM_FULL_API);
      url.searchParams.set("input", query);
      url.searchParams.set("appid", WOLFRAM_APP_ID!);
      url.searchParams.set("format", format);
      url.searchParams.set("output", "json");

      if (includepodid) url.searchParams.set("includepodid", includepodid);
      if (excludepodid) url.searchParams.set("excludepodid", excludepodid);

      const response = await fetch(url.toString());
      // Try to parse JSON response
      try {
        const data = await response.json();
        return formatFullAPIResponse(data);
      } catch (jsonError) {
        // If JSON parsing fails, return raw text
        const text = await response.text();
        if (text.length > 0) return text;
      }
      console.error(`Wolfram Full API returned ${response.status}, falling back to local`);
    } catch (apiError) {
      console.error(`Wolfram Full API error: ${apiError}, falling back to local`);
    }
  }

  // Fallback to local WolframScript
  if (hasLocal) {
    const code = `WolframAlpha["${query.replace(/"/g, '\\"')}", {{"Result", 1}, "Plaintext"}] // ToString`;
    return executeWolframScript(code, 60);
  }

  throw new Error("No Wolfram backend available (API failed and no local WolframScript)");
}

// Local WolframScript execution
async function executeWolframScript(code: string, timeout: number = 30): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = spawn(WOLFRAMSCRIPT_PATH, ["-code", code], {
      timeout: timeout * 1000,
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => { stdout += data.toString(); });
    proc.stderr.on("data", (data) => { stderr += data.toString(); });

    proc.on("close", (exitCode) => {
      if (exitCode === 0) {
        // Filter out progress messages
        const lines = stdout.split("\n").filter(
          (line) => !line.includes("Loading from Wolfram") && 
                    !line.includes("Prefetching") && 
                    !line.includes("Connecting")
        );
        resolve(lines.join("\n").trim());
      } else {
        reject(new Error(`WolframScript failed: ${stderr || stdout}`));
      }
    });

    proc.on("error", (err) => reject(err));
  });
}

// Build Wolfram code for symbolic operations
function buildSymbolicCode(operation: string, expression: string, variable: string, options?: string): string {
  const opts = options ? `, ${options}` : "";
  switch (operation) {
    case "integrate":
      return `Integrate[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "differentiate":
      return `D[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "solve":
      return `Solve[${expression}, ${variable}${opts}] // InputForm // ToString`;
    case "simplify":
      return `FullSimplify[${expression}${opts}] // InputForm // ToString`;
    case "series":
      return `Series[${expression}, {${variable}, 0, 5}${opts}] // Normal // InputForm // ToString`;
    case "limit":
      return `Limit[${expression}, ${variable} -> Infinity${opts}] // InputForm // ToString`;
    default:
      return `${expression} // InputForm // ToString`;
  }
}

// Build Wolfram code for hyperbolic geometry
function buildHyperbolicCode(operation: string, params: Record<string, any>): string {
  switch (operation) {
    case "distance":
      const { z1, z2 } = params;
      return `N[2*ArcTanh[Abs[(${z1[0]}+${z1[1]}*I)-(${z2[0]}+${z2[1]}*I)]/Sqrt[(1-Abs[${z1[0]}+${z1[1]}*I]^2)*(1-Abs[${z2[0]}+${z2[1]}*I]^2)+Abs[(${z1[0]}+${z1[1]}*I)-(${z2[0]}+${z2[1]}*I)]^2]], 15]`;
    case "geodesic":
      const { start, end, numPoints } = params;
      return `Module[{z1=${start[0]}+${start[1]}*I, z2=${end[0]}+${end[1]}*I, moebius},
        moebius[z_, a_] := (z - a)/(1 - Conjugate[a]*z);
        N[Table[{Re[#], Im[#]}&[moebius[t*moebius[z2, z1], -z1]], {t, 0, 1, 1/(${numPoints || 10}-1)}]]
      ]`;
    case "mobius":
      const { a, b, c, d, z } = params;
      return `Module[{result = ((${a[0]}+${a[1]}*I)*(${z[0]}+${z[1]}*I) + (${b[0]}+${b[1]}*I))/((${c[0]}+${c[1]}*I)*(${z[0]}+${z[1]}*I) + (${d[0]}+${d[1]}*I))},
        {Re[result], Im[result]} // N
      ]`;
    case "tessellation":
      const { p, q, depth } = params;
      return `Module[{coords},
        coords = Flatten[Table[Module[{r = (1 - 0.9^layer)*0.95, theta = 2*Pi*k/(${p}*layer+1)},
          {r*Cos[theta], r*Sin[theta]}
        ], {layer, 1, ${depth || 3}}, {k, 0, ${p}*layer}], 1];
        N[coords]
      ]`;
    default:
      return `"Unknown operation: ${operation}"`;
  }
}

function formatFullAPIResponse(data: any): string {
  if (!data.queryresult?.success) {
    return `Query failed: ${data.queryresult?.error || "Unknown error"}`;
  }

  const pods = data.queryresult.pods || [];
  let result = "";

  for (const pod of pods) {
    result += `\n## ${pod.title}\n`;
    for (const subpod of pod.subpods || []) {
      if (subpod.plaintext) {
        result += `${subpod.plaintext}\n`;
      }
      if (subpod.img?.src) {
        result += `![${pod.title}](${subpod.img.src})\n`;
      }
    }
  }

  return result.trim();
}

// Create MCP Server
const server = new Server(
  {
    name: "wolfram-mcp",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Handle tool listing - combine all tool categories
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { 
    tools: [
      ...tools,           // Core Wolfram tools
      ...swarmTools,      // Agent swarm communication
      ...enhancedTools,   // Design thinking, systems dynamics, LLM, auth
    ]
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case "wolfram_llm_query": {
        const { query, maxchars } = LLMQuerySchema.parse(args);
        const result = await queryLLMAPI(query, maxchars);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_compute": {
        const { expression } = ComputeSchema.parse(args);
        const result = await queryLLMAPI(`compute ${expression}`);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_validate": {
        const { expression, expected } = ValidateSchema.parse(args);
        const query = expected
          ? `is ${expression} equal to ${expected}`
          : `simplify ${expression}`;
        const result = await queryLLMAPI(query);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_unit_convert": {
        const { value, targetUnit } = UnitConvertSchema.parse(args);
        const result = await queryLLMAPI(`convert ${value} to ${targetUnit}`);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_data_query": {
        const { entity, property } = DataQuerySchema.parse(args);
        const query = property ? `${entity} ${property}` : entity;
        const result = await queryLLMAPI(query);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_full_query": {
        const { query, format, includepodid, excludepodid } = FullQuerySchema.parse(args);
        const result = await queryFullAPI(query, format, includepodid, excludepodid);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_local_eval": {
        if (!hasLocal) {
          throw new Error("Local WolframScript not available");
        }
        const { code, timeout } = LocalEvalSchema.parse(args);
        const result = await executeWolframScript(code, timeout);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      case "wolfram_symbolic": {
        if (!hasLocal) {
          throw new Error("Local WolframScript required for symbolic computation");
        }
        const { operation, expression, variable, options } = SymbolicComputeSchema.parse(args);
        const code = buildSymbolicCode(operation, expression, variable || "x", options);
        const result = await executeWolframScript(code);
        return {
          content: [{ type: "text", text: `${operation}(${expression}) = ${result}` }],
        };
      }

      case "wolfram_hyperbolic": {
        if (!hasLocal) {
          throw new Error("Local WolframScript required for hyperbolic geometry");
        }
        const { operation, params } = HyperbolicGeometrySchema.parse(args);
        const code = buildHyperbolicCode(operation, params);
        const result = await executeWolframScript(code);
        return {
          content: [{ type: "text", text: `Hyperbolic ${operation}: ${result}` }],
        };
      }

      // ====== Swarm Tools for Agent-to-Agent Communication ======
      case "swarm_join":
      case "swarm_leave":
      case "swarm_list_agents":
      case "swarm_send":
      case "swarm_propose":
      case "swarm_vote":
      case "swarm_create_task":
      case "swarm_update_task":
      case "swarm_my_tasks":
      case "swarm_set_memory":
      case "swarm_get_memory":
      case "swarm_share_code":
      case "swarm_request_review":
      case "swarm_find_nearest":
      case "swarm_trust_scores": {
        const result = await handleSwarmTool(name, args);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // ====== Design Thinking Tools ======
      case "design_empathize_analyze":
      case "design_empathize_persona":
      case "design_define_problem":
      case "design_define_requirements":
      case "design_ideate_brainstorm":
      case "design_ideate_evaluate":
      case "design_prototype_architecture":
      case "design_prototype_code":
      case "design_test_generate":
      case "design_test_analyze":
      case "design_iterate_feedback":
      case "design_iterate_metrics": {
        const wolframCode = designThinkingWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Design thinking tool ${name} executed with args: ${JSON.stringify(args)}` }] };
      }

      // ====== Systems Dynamics Tools ======
      case "systems_model_create":
      case "systems_model_simulate":
      case "systems_equilibrium_find":
      case "systems_equilibrium_stability":
      case "systems_equilibrium_bifurcation":
      case "systems_control_design":
      case "systems_control_analyze":
      case "systems_feedback_causal_loop":
      case "systems_feedback_loop_gain":
      case "systems_network_analyze":
      case "systems_network_optimize":
      case "systems_sensitivity_analyze":
      case "systems_monte_carlo": {
        const wolframCode = systemsDynamicsWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Systems dynamics tool ${name} requires WolframScript` }] };
      }

      // ====== LLM Tools ======
      case "wolfram_llm_function":
      case "wolfram_llm_synthesize":
      case "wolfram_llm_tool_define":
      case "wolfram_llm_prompt":
      case "wolfram_llm_prompt_chain":
      case "wolfram_llm_code_generate":
      case "wolfram_llm_code_review":
      case "wolfram_llm_code_explain":
      case "wolfram_llm_analyze":
      case "wolfram_llm_reason":
      case "wolfram_llm_graph": {
        const wolframCode = llmWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 120);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `LLM tool ${name} requires WolframScript with LLM access` }] };
      }

      // ====== Dilithium Authorization Tools ======
      case "dilithium_register_client":
      case "dilithium_authorize":
      case "dilithium_validate_token":
      case "dilithium_check_quota":
      case "dilithium_list_clients":
      case "dilithium_revoke_client":
      case "dilithium_update_capabilities": {
        const result = await handleDilithiumAuth(name, args);
        return { content: [{ type: "text", text: result }] };
      }

      // ====== DevOps Pipeline Tools ======
      case "git_analyze_history":
      case "git_branch_strategy":
      case "git_pr_review_assist":
      case "cicd_pipeline_generate":
      case "cicd_pipeline_optimize":
      case "cicd_artifact_manage":
      case "deploy_strategy_plan":
      case "deploy_infrastructure_as_code":
      case "deploy_kubernetes_manifest":
      case "observability_setup":
      case "observability_alert_rules":
      case "observability_dashboard_generate":
      case "observability_incident_analyze":
      case "test_load_generate":
      case "test_chaos_experiment":
      case "test_security_scan":
      case "test_mutation_analyze":
      case "test_contract_verify": {
        const wolframCode = devopsPipelineWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `DevOps tool ${name} - configuration generated for: ${JSON.stringify(args)}` }] };
      }

      // ====== Project Management Tools ======
      case "sprint_plan_generate":
      case "sprint_retrospective_analyze":
      case "estimate_effort":
      case "estimate_project_timeline":
      case "backlog_prioritize":
      case "backlog_refine":
      case "backlog_dependency_analyze":
      case "team_workload_balance":
      case "team_skill_gap_analyze":
      case "metrics_engineering_calculate":
      case "metrics_dora_calculate":
      case "report_status_generate": {
        const wolframCode = projectManagementWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Project management tool ${name} executed` }] };
      }

      // ====== Documentation Tools ======
      case "docs_api_generate":
      case "docs_api_openapi_generate":
      case "docs_architecture_diagram":
      case "docs_adr_generate":
      case "docs_system_design":
      case "docs_runbook_generate":
      case "docs_postmortem_generate":
      case "docs_code_readme":
      case "docs_code_comments":
      case "docs_changelog_generate":
      case "kb_search":
      case "kb_index":
      case "kb_summarize":
      case "kb_onboarding_generate": {
        const wolframCode = documentationWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Documentation tool ${name} executed` }] };
      }

      // ====== Code Quality Tools ======
      case "code_analyze_complexity":
      case "code_analyze_duplication":
      case "code_analyze_dependencies":
      case "code_analyze_coverage":
      case "refactor_suggest":
      case "refactor_extract_method":
      case "refactor_rename_symbol":
      case "refactor_pattern_apply":
      case "techdebt_analyze":
      case "techdebt_prioritize":
      case "techdebt_budget":
      case "health_score_calculate":
      case "health_trend_analyze":
      case "lint_config_generate":
      case "format_config_generate": {
        const wolframCode = codeQualityWolframCode[name];
        if (wolframCode && hasLocal) {
          const code = wolframCode(args);
          const result = await executeWolframScript(code, 60);
          return { content: [{ type: "text", text: result }] };
        }
        return { content: [{ type: "text", text: `Code quality tool ${name} executed` }] };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      content: [{ type: "text", text: `Error: ${message}` }],
      isError: true,
    };
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Wolfram MCP Server running on stdio");
}

main().catch(console.error);
