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
const SERVER_VERSION = "1.0.0";

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
  console.error("🚀 Starting CQGS MCP Server");
  console.error(`📦 Version: ${SERVER_VERSION}`);
  console.error(`🔐 Security: Dilithium ML-DSA-65 (Post-Quantum)`);
  console.error(`🛡️  Sentinels: ${native.getSentinelCount()}`);
  console.error(`⚙️  Features: ${native.getFeatures().join(", ")}`);

  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("✅ CQGS MCP Server running on stdio");
}

main().catch((error) => {
  console.error("❌ Fatal error:", error);
  process.exit(1);
});
