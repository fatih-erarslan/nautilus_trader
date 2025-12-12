#!/usr/bin/env bun
/**
 * CQGS MCP Server - Bun.JS Implementation
 *
 * Exposes 49 Code Quality Governance Sentinels via Model Context Protocol
 * with Dilithium ML-DSA-65 post-quantum authentication.
 *
 * Architecture: Bun.JS → NAPI → Rust core
 * Security: Dilithium ML-DSA-65 (NIST FIPS 204)
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";

// Load native module
let native: any;
try {
  native = require("../index.node");
  console.error("✅ Native CQGS module loaded successfully");
} catch (error) {
  console.error("❌ Failed to load native module:", error);
  console.error("💡 Run: cargo build --release --features napi");
  process.exit(1);
}

// Server configuration
const SERVER_NAME = "cqgs-mcp-server";
const SERVER_VERSION = "2.0.0";

// MCP Server
const server = new Server(
  {
    name: SERVER_NAME,
    version: SERVER_VERSION,
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// ============================================================================
// Tool Definitions
// ============================================================================

const TOOLS: Tool[] = [
  // Sentinel Execution
  {
    name: "sentinel_execute_all",
    description: "Execute all 49 CQGS sentinels on a codebase",
    inputSchema: {
      type: "object",
      properties: {
        codebase_path: {
          type: "string",
          description: "Path to codebase to analyze",
        },
        parallel: {
          type: "boolean",
          description: "Enable parallel execution (default: true)",
        },
      },
      required: ["codebase_path"],
    },
  },
  {
    name: "sentinel_quality_score",
    description: "Calculate overall quality score from sentinel results",
    inputSchema: {
      type: "object",
      properties: {
        results_json: {
          type: "string",
          description: "JSON string of sentinel results",
        },
      },
      required: ["results_json"],
    },
  },
  {
    name: "sentinel_quality_gate",
    description: "Check if results pass quality gate (GATE_1 through GATE_5)",
    inputSchema: {
      type: "object",
      properties: {
        results_json: {
          type: "string",
          description: "JSON string of sentinel results",
        },
        gate: {
          type: "string",
          enum: [
            "NoForbiddenPatterns",
            "IntegrationReady",
            "TestingReady",
            "ProductionReady",
            "DeploymentApproved",
          ],
          description: "Quality gate to check",
        },
      },
      required: ["results_json", "gate"],
    },
  },

  // Dilithium Authentication
  {
    name: "dilithium_keygen",
    description: "Generate Dilithium ML-DSA-65 key pair for authentication",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "dilithium_sign",
    description: "Sign message with Dilithium ML-DSA-65",
    inputSchema: {
      type: "object",
      properties: {
        secret_key: {
          type: "string",
          description: "Hex-encoded Dilithium secret key",
        },
        message: {
          type: "string",
          description: "Message to sign",
        },
      },
      required: ["secret_key", "message"],
    },
  },
  {
    name: "dilithium_verify",
    description: "Verify Dilithium signature",
    inputSchema: {
      type: "object",
      properties: {
        public_key: {
          type: "string",
          description: "Hex-encoded Dilithium public key",
        },
        signature: {
          type: "string",
          description: "Hex-encoded signature",
        },
        message: {
          type: "string",
          description: "Original message",
        },
      },
      required: ["public_key", "signature", "message"],
    },
  },

  // Hyperbolic Geometry
  {
    name: "hyperbolic_distance",
    description: "Compute hyperbolic distance in H^11 (Lorentz model)",
    inputSchema: {
      type: "object",
      properties: {
        point1: {
          type: "array",
          items: { type: "number" },
          minItems: 12,
          maxItems: 12,
          description: "First point (12D Lorentz coordinates)",
        },
        point2: {
          type: "array",
          items: { type: "number" },
          minItems: 12,
          maxItems: 12,
          description: "Second point (12D Lorentz coordinates)",
        },
      },
      required: ["point1", "point2"],
    },
  },

  // Symbolic Computation
  {
    name: "shannon_entropy",
    description: "Compute Shannon entropy H(X) = -Σ p(x) log₂ p(x)",
    inputSchema: {
      type: "object",
      properties: {
        probabilities: {
          type: "array",
          items: { type: "number" },
          description: "Probability distribution (must sum to 1.0)",
        },
      },
      required: ["probabilities"],
    },
  },

  // Version Info
  {
    name: "cqgs_version",
    description: "Get CQGS MCP plugin version and features",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },

  // ============================================================================
  // Individual Sentinel Tools (v2.0)
  // ============================================================================

  // Mock Detection Sentinel (47 tests passing)
  {
    name: "sentinel_mock_detection",
    description: "Analyze code for mock data patterns, synthetic generators, and placeholder implementations. Enforces TENGRI rules: NO MOCK DATA. Returns violations with severity and confidence scores.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze for mock patterns",
        },
        file_path: {
          type: "string",
          description: "Optional file path for context",
        },
        strict_mode: {
          type: "boolean",
          description: "Enable strict mode for zero-tolerance enforcement (default: true)",
        },
      },
      required: ["code"],
    },
  },

  // Reward Hacking Prevention Sentinel (14 tests passing)
  {
    name: "sentinel_reward_hacking",
    description: "Detect reward hacking patterns: test manipulation, metric gaming, circular validation, and objective misalignment. Critical for AI safety.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze",
        },
        context: {
          type: "string",
          description: "Execution context (test, production, ci)",
        },
      },
      required: ["code"],
    },
  },

  // Cross-Scale Analysis Sentinel (13 tests passing)
  {
    name: "sentinel_cross_scale",
    description: "Perform multi-granularity analysis: line-level, function-level, module-level, and system-level pattern detection.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze",
        },
      },
      required: ["code"],
    },
  },

  // Zero-Synthetic Enforcement Sentinel (8 tests passing)
  {
    name: "sentinel_zero_synthetic",
    description: "Enforce zero-synthetic data policy. Detects np.random, random.*, mock.*, hardcoded values. CRITICAL for TENGRI compliance.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze",
        },
      },
      required: ["code"],
    },
  },

  // Behavioral Analysis Sentinel (3 tests passing)
  {
    name: "sentinel_behavioral",
    description: "Analyze behavioral patterns and detect anomalies in code execution flow.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze",
        },
        execution_trace: {
          type: "string",
          description: "Optional execution trace for deeper analysis",
        },
      },
      required: ["code"],
    },
  },

  // Runtime Verification Sentinel (3 tests passing)
  {
    name: "sentinel_runtime_verification",
    description: "Verify runtime behavior against formal specifications.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to verify",
        },
        spec: {
          type: "string",
          description: "Optional specification to verify against",
        },
      },
      required: ["code"],
    },
  },

  // Batch Sentinel Analysis
  {
    name: "sentinel_batch_analyze",
    description: "Run multiple sentinels in batch. Returns aggregated quality score and quality gate achieved.",
    inputSchema: {
      type: "object",
      properties: {
        code: {
          type: "string",
          description: "Source code to analyze",
        },
        sentinels: {
          type: "array",
          items: { type: "string" },
          description: "List of sentinel IDs to run (mock-detection, reward-hacking, cross-scale, zero-synthetic)",
        },
        strict_mode: {
          type: "boolean",
          description: "Enable strict mode (default: true)",
        },
      },
      required: ["code", "sentinels"],
    },
  },

  // List Enabled Sentinels
  {
    name: "sentinel_list_enabled",
    description: "List all enabled sentinels with test counts and categories.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },

  // Fibonacci Thresholds
  {
    name: "fibonacci_thresholds",
    description: "Get Fibonacci-scaled thresholds for complexity, file size, or custom sequences.",
    inputSchema: {
      type: "object",
      properties: {
        level: {
          type: "string",
          enum: ["complexity", "file_size", "custom"],
          description: "Threshold category to retrieve",
        },
      },
      required: ["level"],
    },
  },
];

// ============================================================================
// Tool Handlers
// ============================================================================

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      // Sentinel Execution
      case "sentinel_execute_all": {
        const results = native.executeAllSentinels(
          args.codebase_path,
          args.parallel ?? true
        );
        return {
          content: [{ type: "text", text: results }],
        };
      }

      case "sentinel_quality_score": {
        const score = native.calculateQualityScore(args.results_json);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ quality_score: score }, null, 2),
            },
          ],
        };
      }

      case "sentinel_quality_gate": {
        const passed = native.checkQualityGate(args.results_json, args.gate);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { gate: args.gate, passed },
                null,
                2
              ),
            },
          ],
        };
      }

      // Dilithium Authentication
      case "dilithium_keygen": {
        const keypair = native.dilithiumKeygen();
        return {
          content: [{ type: "text", text: keypair }],
        };
      }

      case "dilithium_sign": {
        const signature = native.dilithiumSign(args.secret_key, args.message);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ signature }, null, 2),
            },
          ],
        };
      }

      case "dilithium_verify": {
        const valid = native.dilithiumVerify(
          args.public_key,
          args.signature,
          args.message
        );
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ valid }, null, 2),
            },
          ],
        };
      }

      // Hyperbolic Geometry
      case "hyperbolic_distance": {
        const distance = native.hyperbolicDistance(args.point1, args.point2);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ distance }, null, 2),
            },
          ],
        };
      }

      // Symbolic Computation
      case "shannon_entropy": {
        const entropy = native.shannonEntropy(args.probabilities);
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify({ entropy }, null, 2),
            },
          ],
        };
      }

      // Version Info
      case "cqgs_version": {
        const version = native.getVersion();
        const features = native.getFeatures();
        const sentinelCount = native.getSentinelCount();
        return {
          content: [
            {
              type: "text",
              text: JSON.stringify(
                { version, features, sentinel_count: sentinelCount },
                null,
                2
              ),
            },
          ],
        };
      }

      // ============================================================================
      // Individual Sentinel Handlers (v2.0)
      // ============================================================================

      // Mock Detection Sentinel
      case "sentinel_mock_detection": {
        const result = native.analyzeMockDetection(
          args.code,
          args.file_path ?? null
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Reward Hacking Prevention Sentinel
      case "sentinel_reward_hacking": {
        const result = native.analyzeRewardHacking(
          args.code,
          args.context ?? null
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Cross-Scale Analysis Sentinel
      case "sentinel_cross_scale": {
        const result = native.analyzeCrossScale(args.code);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Zero-Synthetic Enforcement Sentinel
      case "sentinel_zero_synthetic": {
        const result = native.analyzeZeroSynthetic(args.code);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Behavioral Analysis Sentinel
      case "sentinel_behavioral": {
        const result = native.analyzeBehavioral(
          args.code,
          args.execution_trace ?? null
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Runtime Verification Sentinel
      case "sentinel_runtime_verification": {
        const result = native.verifyRuntime(
          args.code,
          args.spec ?? null
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Batch Sentinel Analysis
      case "sentinel_batch_analyze": {
        const result = native.batchSentinelAnalysis(
          args.code,
          args.sentinels,
          args.strict_mode ?? true
        );
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // List Enabled Sentinels
      case "sentinel_list_enabled": {
        const result = native.listEnabledSentinels();
        return {
          content: [{ type: "text", text: result }],
        };
      }

      // Fibonacci Thresholds
      case "fibonacci_thresholds": {
        const result = native.getFibonacciThreshold(args.level);
        return {
          content: [{ type: "text", text: result }],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : String(error);
    return {
      content: [{ type: "text", text: `Error: ${errorMessage}` }],
      isError: true,
    };
  }
});

// ============================================================================
// Server Startup
// ============================================================================

async function main() {
  console.error("🚀 Starting CQGS MCP Server v2.0");
  console.error(`📦 Version: ${SERVER_VERSION}`);
  console.error(`🔐 Security: Dilithium ML-DSA-65 (Post-Quantum)`);
  console.error(`🛡️  Sentinels: ${native.getSentinelCount()}`);
  console.error(`⚙️  Features: ${native.getFeatures().join(", ")}`);
  console.error(`🔧 MCP Tools: ${TOOLS.length} total`);
  console.error(`   - Core: sentinel_execute_all, sentinel_quality_score, sentinel_quality_gate`);
  console.error(`   - Individual: mock_detection, reward_hacking, cross_scale, zero_synthetic`);
  console.error(`   - Batch: sentinel_batch_analyze, sentinel_list_enabled`);
  console.error(`   - Math: hyperbolic_distance, shannon_entropy, fibonacci_thresholds`);
  console.error(`   - Auth: dilithium_keygen, dilithium_sign, dilithium_verify`);

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("✅ CQGS MCP Server running on stdio");
}

main().catch((error) => {
  console.error("❌ Fatal error:", error);
  process.exit(1);
});
