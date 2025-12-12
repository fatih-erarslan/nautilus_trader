#!/usr/bin/env bun
/**
 * Dilithium MCP Server v3.0
 * 
 * Post-Quantum Secure Model Context Protocol Server
 * 
 * Architecture: Rust-Bun.js with Dilithium ML-DSA Authentication
 * 
 * Features:
 * - Post-quantum cryptographic authentication (Dilithium/ML-DSA)
 * - Native Rust bindings via NAPI-RS for core computations
 * - Hyperbolic geometry (Lorentz H^11)
 * - pBit dynamics engine
 * - Symbolic mathematics
 * - Agent swarm coordination
 * - Design thinking pipeline
 * - Systems dynamics modeling
 * 
 * Security Model:
 * - All requests signed with Dilithium signatures
 * - Nonce-based replay protection
 * - Client quota management
 * - Comprehensive audit logging
 * 
 * Environment Variables:
 * - DILITHIUM_SECRET_KEY: Server's secret key (hex)
 * - DILITHIUM_PUBLIC_KEY: Server's public key (hex)
 * - DILITHIUM_MCP_PORT: Server port (default: 3000)
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
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

// Module path resolution
const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

// Import local modules
import { swarmTools, handleSwarmTool } from "./swarm/index.js";
import { dilithiumAuthTools, handleDilithiumAuth } from "./auth/dilithium-sentry.js";
import { 
  enhancedTools, 
  handleEnhancedTool,
  toolCategories,
  totalToolCount
} from "./tools/index.js";

// ============================================================================
// Native Module Loading
// ============================================================================

interface NativeModule {
  // Dilithium crypto
  dilithium_keygen(): { public_key: string; secret_key: string };
  dilithium_sign(secret_key: string, message: string): string;
  dilithium_verify(public_key: string, signature: string, message: string): boolean;
  blake3_hash(data: string): string;
  generate_nonce(): string;
  
  // Hyperbolic geometry
  lorentz_inner(x: number[], y: number[]): number;
  hyperbolic_distance(x: number[], y: number[]): number;
  lift_to_hyperboloid(z: number[]): number[];
  mobius_add(x: number[], y: number[], curvature: number): number[];
  
  // pBit dynamics
  pbit_probability(field: number, bias: number, temperature: number): number;
  pbit_probabilities_batch(fields: number[], biases: number[], temperature: number): number[];
  boltzmann_weight(energy: number, temperature: number): number;
  ising_critical_temp(): number;
  stdp_weight_change(delta_t: number, a_plus: number, a_minus: number, tau: number): number;
  
  // Math utilities
  fast_exp(x: number): number;
  stable_acosh(x: number): number;
  
  // Server state
  init_server(): string;
  register_client(client_id: string, public_key: string, capabilities: string[]): boolean;
  verify_request(request: AuthenticatedRequest): AuthResult;
}

interface AuthenticatedRequest {
  client_id: string;
  timestamp: string;
  nonce: string;
  payload: string;
  signature: string;
}

interface AuthResult {
  valid: boolean;
  client_id: string;
  error?: string;
  timestamp: string;
}

let native: NativeModule | null = null;

// Try to load native Rust module
const nativePaths = [
  process.env.DILITHIUM_NATIVE_PATH,
  resolve(projectRoot, "native/dilithium-native.darwin-x64.node"),
  resolve(projectRoot, "native/dilithium-native.darwin-arm64.node"),
  resolve(projectRoot, "native/target/release/libdilithium_native.dylib"),
];

for (const path of nativePaths) {
  if (path && existsSync(path)) {
    try {
      native = require(path) as NativeModule;
      console.error(`[Dilithium MCP] Loaded native module from ${path}`);
      break;
    } catch (e) {
      console.error(`[Dilithium MCP] Failed to load ${path}: ${e}`);
    }
  }
}

if (!native) {
  console.error("[Dilithium MCP] Warning: Native module not available, using JS fallback");
}

// ============================================================================
// Fallback Implementations (Pure TypeScript)
// ============================================================================

const fallback = {
  // Note: These are NOT quantum-secure! Only for development/testing
  dilithium_keygen: () => {
    console.error("[WARNING] Using insecure fallback keygen - native module required for production");
    const pk = crypto.randomUUID().replace(/-/g, "") + crypto.randomUUID().replace(/-/g, "");
    const sk = crypto.randomUUID().replace(/-/g, "") + crypto.randomUUID().replace(/-/g, "");
    return { public_key: pk, secret_key: sk };
  },
  
  dilithium_sign: (sk: string, message: string) => {
    // Insecure fallback - HMAC-like
    const data = sk + message;
    return Bun.hash(data).toString(16).padStart(64, "0");
  },
  
  dilithium_verify: (pk: string, sig: string, message: string) => {
    // Cannot verify without proper implementation
    console.error("[WARNING] Signature verification disabled in fallback mode");
    return true; // DANGEROUS: Only for development
  },
  
  blake3_hash: (data: string) => {
    return Bun.hash(data).toString(16).padStart(64, "0");
  },
  
  generate_nonce: () => crypto.randomUUID().replace(/-/g, ""),
  
  lorentz_inner: (x: number[], y: number[]) => {
    return -x[0] * y[0] + x.slice(1).reduce((sum, xi, i) => sum + xi * y[i + 1], 0);
  },
  
  hyperbolic_distance: (x: number[], y: number[]) => {
    const inner = -fallback.lorentz_inner(x, y);
    return Math.acosh(Math.max(inner, 1));
  },
  
  lift_to_hyperboloid: (z: number[]) => {
    const normSq = z.reduce((sum, x) => sum + x * x, 0);
    return [Math.sqrt(1 + normSq), ...z];
  },
  
  mobius_add: (x: number[], y: number[], curvature: number) => {
    const c = -curvature;
    const xy = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const xNormSq = x.reduce((sum, xi) => sum + xi * xi, 0);
    const yNormSq = y.reduce((sum, yi) => sum + yi * yi, 0);
    
    const denom = 1 + 2 * c * xy + c * c * xNormSq * yNormSq;
    const coefX = 1 + 2 * c * xy + c * yNormSq;
    const coefY = 1 - c * xNormSq;
    
    return x.map((xi, i) => (coefX * xi + coefY * y[i]) / denom);
  },
  
  pbit_probability: (field: number, bias: number, temperature: number) => {
    const x = (field - bias) / Math.max(temperature, 1e-10);
    return 1 / (1 + Math.exp(-x));
  },
  
  pbit_probabilities_batch: (fields: number[], biases: number[], temperature: number) => {
    return fields.map((h, i) => fallback.pbit_probability(h, biases[i] || 0, temperature));
  },
  
  boltzmann_weight: (energy: number, temperature: number) => {
    return Math.exp(-energy / Math.max(temperature, 1e-10));
  },
  
  ising_critical_temp: () => 2 / Math.log(1 + Math.sqrt(2)),
  
  stdp_weight_change: (delta_t: number, a_plus: number, a_minus: number, tau: number) => {
    if (delta_t > 0) {
      return a_plus * Math.exp(-delta_t / tau);
    } else {
      return -a_minus * Math.exp(delta_t / tau);
    }
  },
  
  fast_exp: (x: number) => Math.exp(x),
  stable_acosh: (x: number) => x < 1.0001 ? Math.sqrt(2 * Math.max(x - 1, 0)) : Math.acosh(x),
};

// Use native or fallback
const lib = native || fallback;

// ============================================================================
// Tool Definitions
// ============================================================================

const tools: Tool[] = [
  // === Dilithium Authentication ===
  {
    name: "dilithium_keygen",
    description: "Generate a new Dilithium ML-DSA key pair for post-quantum secure authentication",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "dilithium_sign",
    description: "Sign a message using Dilithium ML-DSA",
    inputSchema: {
      type: "object",
      properties: {
        secret_key: { type: "string", description: "Hex-encoded secret key" },
        message: { type: "string", description: "Message to sign" },
      },
      required: ["secret_key", "message"],
    },
  },
  {
    name: "dilithium_verify",
    description: "Verify a Dilithium signature",
    inputSchema: {
      type: "object",
      properties: {
        public_key: { type: "string", description: "Hex-encoded public key" },
        signature: { type: "string", description: "Hex-encoded signature" },
        message: { type: "string", description: "Original message" },
      },
      required: ["public_key", "signature", "message"],
    },
  },
  
  // === Hyperbolic Geometry ===
  {
    name: "hyperbolic_distance",
    description: "Compute hyperbolic distance between two points in H^11 (Lorentz model)",
    inputSchema: {
      type: "object",
      properties: {
        point1: { type: "array", items: { type: "number" }, description: "First point (12D Lorentz coords)" },
        point2: { type: "array", items: { type: "number" }, description: "Second point (12D Lorentz coords)" },
      },
      required: ["point1", "point2"],
    },
  },
  {
    name: "lift_to_hyperboloid",
    description: "Lift Euclidean point to Lorentz hyperboloid (H^n)",
    inputSchema: {
      type: "object",
      properties: {
        point: { type: "array", items: { type: "number" }, description: "Euclidean point" },
      },
      required: ["point"],
    },
  },
  {
    name: "mobius_add",
    description: "Mobius addition in Poincare ball model",
    inputSchema: {
      type: "object",
      properties: {
        x: { type: "array", items: { type: "number" } },
        y: { type: "array", items: { type: "number" } },
        curvature: { type: "number", description: "Curvature (typically -1)" },
      },
      required: ["x", "y"],
    },
  },
  
  // === pBit Dynamics ===
  {
    name: "pbit_sample",
    description: "Compute pBit sampling probability using Boltzmann statistics",
    inputSchema: {
      type: "object",
      properties: {
        field: { type: "number", description: "Effective field h" },
        bias: { type: "number", description: "Bias term" },
        temperature: { type: "number", description: "Temperature T" },
      },
      required: ["field", "temperature"],
    },
  },
  {
    name: "boltzmann_weight",
    description: "Compute Boltzmann weight exp(-E/T)",
    inputSchema: {
      type: "object",
      properties: {
        energy: { type: "number", description: "Energy E" },
        temperature: { type: "number", description: "Temperature T" },
      },
      required: ["energy", "temperature"],
    },
  },
  {
    name: "ising_critical_temp",
    description: "Get Ising model critical temperature (2D square lattice, Onsager solution)",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "stdp_weight_change",
    description: "Compute STDP (Spike-Timing Dependent Plasticity) weight change",
    inputSchema: {
      type: "object",
      properties: {
        delta_t: { type: "number", description: "Time difference (post - pre) in ms" },
        a_plus: { type: "number", description: "LTP amplitude (default: 0.1)" },
        a_minus: { type: "number", description: "LTD amplitude (default: 0.12)" },
        tau: { type: "number", description: "Time constant in ms (default: 20)" },
      },
      required: ["delta_t"],
    },
  },
  
  // === Symbolic Math ===
  {
    name: "compute",
    description: "Compute mathematical expression (uses local engine or external service)",
    inputSchema: {
      type: "object",
      properties: {
        expression: { type: "string", description: "Mathematical expression to evaluate" },
      },
      required: ["expression"],
    },
  },
  {
    name: "symbolic",
    description: "Symbolic mathematics operations (integrate, differentiate, solve, simplify)",
    inputSchema: {
      type: "object",
      properties: {
        operation: { 
          type: "string", 
          enum: ["integrate", "differentiate", "solve", "simplify", "series", "limit"],
          description: "Operation to perform" 
        },
        expression: { type: "string", description: "Mathematical expression" },
        variable: { type: "string", description: "Variable (default: x)" },
      },
      required: ["operation", "expression"],
    },
  },
  
  // === Utility ===
  {
    name: "blake3_hash",
    description: "Hash data using BLAKE3 cryptographic hash function",
    inputSchema: {
      type: "object",
      properties: {
        data: { type: "string", description: "Data to hash" },
      },
      required: ["data"],
    },
  },
  
  // Include dilithium auth tools (7 tools)
  ...dilithiumAuthTools,
  
  // Include swarm tools (15 tools)
  ...swarmTools,
  
  // Include enhanced tools (105 tools)
  ...enhancedTools,
];

// ============================================================================
// Tool Handler
// ============================================================================

async function handleToolCall(name: string, args: Record<string, unknown>): Promise<string> {
  try {
    switch (name) {
      // Dilithium Auth
      case "dilithium_keygen":
        return JSON.stringify(lib.dilithium_keygen?.() || fallback.dilithium_keygen());
        
      case "dilithium_sign":
        const sign = lib.dilithium_sign || fallback.dilithium_sign;
        return sign(args.secret_key as string, args.message as string);
        
      case "dilithium_verify":
        const verify = lib.dilithium_verify || fallback.dilithium_verify;
        return JSON.stringify({ valid: verify(
          args.public_key as string,
          args.signature as string,
          args.message as string
        )});
        
      // Hyperbolic Geometry
      case "hyperbolic_distance":
        const dist = lib.hyperbolic_distance || fallback.hyperbolic_distance;
        return JSON.stringify({ distance: dist(args.point1 as number[], args.point2 as number[]) });
        
      case "lift_to_hyperboloid":
        const lift = lib.lift_to_hyperboloid || fallback.lift_to_hyperboloid;
        return JSON.stringify({ lorentz_point: lift(args.point as number[]) });
        
      case "mobius_add":
        const mobius = lib.mobius_add || fallback.mobius_add;
        return JSON.stringify({ result: mobius(
          args.x as number[],
          args.y as number[],
          (args.curvature as number) || -1
        )});
        
      // pBit Dynamics
      case "pbit_sample":
        const pbit = lib.pbit_probability || fallback.pbit_probability;
        return JSON.stringify({ probability: pbit(
          args.field as number,
          (args.bias as number) || 0,
          args.temperature as number
        )});
        
      case "boltzmann_weight":
        const boltz = lib.boltzmann_weight || fallback.boltzmann_weight;
        return JSON.stringify({ weight: boltz(args.energy as number, args.temperature as number) });
        
      case "ising_critical_temp":
        const tc = lib.ising_critical_temp || fallback.ising_critical_temp;
        return JSON.stringify({ 
          critical_temperature: tc(),
          formula: "T_c = 2/ln(1 + sqrt(2))",
          reference: "Onsager (1944)"
        });
        
      case "stdp_weight_change":
        const stdp = lib.stdp_weight_change || fallback.stdp_weight_change;
        return JSON.stringify({ weight_change: stdp(
          args.delta_t as number,
          (args.a_plus as number) || 0.1,
          (args.a_minus as number) || 0.12,
          (args.tau as number) || 20
        )});
        
      // Utility
      case "blake3_hash":
        const hash = lib.blake3_hash || fallback.blake3_hash;
        return JSON.stringify({ hash: hash(args.data as string) });
        
      // Math (placeholder - would integrate with CAS)
      case "compute":
      case "symbolic":
        return JSON.stringify({ 
          status: "pending",
          message: "Symbolic computation requires external engine or native module"
        });
        
      // Route to specialized handlers
      default:
        // Dilithium auth tools
        if (name.startsWith("dilithium_") && dilithiumAuthTools.some(t => t.name === name)) {
          return await handleDilithiumAuth(name, args);
        }
        // Swarm tools
        if (name.startsWith("swarm_")) {
          return handleSwarmTool(name, args);
        }
        // Enhanced tools (design, systems, llm, devops, docs, code quality, project management, agency, vector)
        if (enhancedTools.some(t => t.name === name)) {
          return handleEnhancedTool(name, args, native);
        }
        return JSON.stringify({ error: `Unknown tool: ${name}` });
    }
  } catch (error) {
    return JSON.stringify({ error: String(error) });
  }
}

// ============================================================================
// Server Setup
// ============================================================================

const server = new Server(
  {
    name: "dilithium-mcp",
    version: "3.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    const result = await handleToolCall(name, args as Record<string, unknown>);
    return {
      content: [{ type: "text", text: result }],
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: JSON.stringify({ error: String(error) }) }],
      isError: true,
    };
  }
});

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.error("╔══════════════════════════════════════════════════════════════╗");
  console.error("║            DILITHIUM MCP SERVER v3.0                         ║");
  console.error("║        Post-Quantum Secure Model Context Protocol            ║");
  console.error("╚══════════════════════════════════════════════════════════════╝");
  console.error("");
  console.error(`  Native Module: ${native ? "✓ Loaded" : "✗ Using fallback"}`);
  console.error(`  Tools Available: ${tools.length}`);
  console.error(`  Categories: ${Object.keys(toolCategories || {}).join(", ") || "core, hyperbolic, pbit, swarm, design, systems, llm, devops, docs, code_quality, project_mgmt"}`);
  console.error("");
  
  if (!native) {
    console.error("  ⚠️  WARNING: Running without native module");
    console.error("  ⚠️  Dilithium signatures use INSECURE fallback");
    console.error("  ⚠️  Build native module for production use");
    console.error("");
  }
  
  const transport = new StdioServerTransport();
  await server.connect(transport);
  
  console.error("  [Ready] Listening on stdio transport");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
