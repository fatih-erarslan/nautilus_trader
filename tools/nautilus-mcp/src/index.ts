#!/usr/bin/env bun
/**
 * Nautilus MCP Server v1.0
 *
 * High-Performance Trading Analytics MCP Server
 *
 * Architecture: Rust-Bun.js with NAPI-RS Native Bindings
 *
 * Features:
 * - Technical indicators (38 tools): MA, RSI, MACD, Bollinger, etc.
 * - Risk analytics (20 tools): VaR, CVaR, Kelly, position sizing
 * - Portfolio metrics (21 tools): Sharpe, Sortino, Calmar
 * - Execution analysis (12 tools): VWAP slippage, order flow
 * - Regime detection (12 tools): pBit dynamics, Ising model
 * - Conformal prediction (10 tools): uncertainty quantification
 * - Greeks & Options (20 tools): Delta, Gamma, Theta, Vega
 *
 * HyperPhysics Integration:
 * - pBit-based market regime detection
 * - Hyperbolic embeddings for market states
 * - Ising model for market coherence
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
    CallToolRequestSchema,
    ListToolsRequestSchema,
    Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { existsSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

// Import all 133 tools
import { allTools, toolCategories, totalToolCount } from "./tools/index.js";

// Module path resolution
const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

// ============================================================================
// Native Module Loading
// ============================================================================

interface NativeModule {
    [key: string]: (...args: unknown[]) => AnalyticsResult;
}

interface AnalyticsResult {
    success: boolean;
    value?: number;
    values?: number[];
    data?: string;
    error?: string;
}

let native: NativeModule | null = null;

// Try to load native Rust module
const nativePaths = [
    process.env.NAUTILUS_NATIVE_PATH,
    resolve(projectRoot, "native/nautilus-native.darwin-x64.node"),
    resolve(projectRoot, "native/nautilus-native.darwin-arm64.node"),
    resolve(projectRoot, "native/target/release/libnautilus_native.dylib"),
    resolve(projectRoot, "dist/libnautilus_native.dylib"),
];

for (const path of nativePaths) {
    if (path && existsSync(path)) {
        try {
            native = require(path) as NativeModule;
            console.error(`[Nautilus MCP] Loaded native module from ${path}`);
            break;
        } catch (e) {
            console.error(`[Nautilus MCP] Failed to load ${path}: ${e}`);
        }
    }
}

if (!native) {
    console.error("[Nautilus MCP] Warning: Native module not available, using JS fallback");
}

// ============================================================================
// Fallback Implementations (Pure TypeScript)
// ============================================================================

const fallback: Record<string, (...args: unknown[]) => AnalyticsResult> = {
    // Simple Moving Average
    indicator_sma: (prices: unknown, period: unknown): AnalyticsResult => {
        const p = prices as number[];
        const n = period as number;
        if (p.length < n || n === 0) return { success: false, error: "Insufficient data for SMA" };
        const sum = p.slice(-n).reduce((a, b) => a + b, 0);
        return { success: true, value: sum / n };
    },

    // Exponential Moving Average
    indicator_ema: (prices: unknown, period: unknown): AnalyticsResult => {
        const p = prices as number[];
        const n = period as number;
        if (p.length === 0 || n === 0) return { success: false, error: "Insufficient data for EMA" };
        const multiplier = 2 / (n + 1);
        let ema = p[0];
        for (let i = 1; i < p.length; i++) {
            ema = (p[i] - ema) * multiplier + ema;
        }
        return { success: true, value: ema };
    },

    // RSI
    indicator_rsi: (prices: unknown, period: unknown): AnalyticsResult => {
        const p = prices as number[];
        const n = period as number;
        if (p.length < n + 1) return { success: false, error: "Insufficient data for RSI" };
        let gains = 0, losses = 0;
        for (let i = p.length - n; i < p.length; i++) {
            const change = p[i] - p[i - 1];
            if (change > 0) gains += change;
            else losses -= change;
        }
        const avgGain = gains / n;
        const avgLoss = losses / n;
        if (avgLoss === 0) return { success: true, value: 100 };
        const rs = avgGain / avgLoss;
        return { success: true, value: 100 - 100 / (1 + rs) };
    },

    // VaR Parametric
    risk_var_parametric: (returns: unknown, confidence: unknown): AnalyticsResult => {
        const r = returns as number[];
        const c = confidence as number;
        if (r.length === 0) return { success: false, error: "No returns" };
        const mean = r.reduce((a, b) => a + b, 0) / r.length;
        const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
        const std = Math.sqrt(variance);
        const z = c === 0.99 ? 2.326 : c === 0.95 ? 1.645 : 1.282;
        return { success: true, value: -(mean - z * std) };
    },

    // Sharpe Ratio
    portfolio_sharpe: (returns: unknown, rf: unknown): AnalyticsResult => {
        const r = returns as number[];
        const riskFree = (rf as number) || 0;
        if (r.length === 0) return { success: false, error: "No returns" };
        const mean = r.reduce((a, b) => a + b, 0) / r.length;
        const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
        const std = Math.sqrt(variance);
        if (std === 0) return { success: true, value: 0 };
        return { success: true, value: (mean - riskFree) / std };
    },

    // Kelly Criterion
    risk_kelly_criterion: (win_rate: unknown, win_loss_ratio: unknown): AnalyticsResult => {
        const w = win_rate as number;
        const wl = win_loss_ratio as number;
        if (w <= 0 || w >= 1 || wl <= 0) return { success: false, error: "Invalid parameters" };
        const kelly = (w * wl - (1 - w)) / wl;
        return {
            success: true,
            data: JSON.stringify({
                kelly_fraction: kelly,
                half_kelly: kelly / 2,
                quarter_kelly: kelly / 4,
            }),
        };
    },

    // pBit State
    regime_pbit_state: (signal: unknown, volatility: unknown, temperature: unknown): AnalyticsResult => {
        const s = signal as number;
        const v = volatility as number;
        const t = temperature as number;
        const effectiveTemp = t * Math.max(v, 0.1);
        const probBullish = 1 / (1 + Math.exp(-s / effectiveTemp));
        const state = probBullish > 0.6 ? "bullish" : probBullish < 0.4 ? "bearish" : "neutral";
        return {
            success: true,
            data: JSON.stringify({
                prob_bullish: probBullish,
                prob_bearish: 1 - probBullish,
                state,
                temperature: effectiveTemp,
            }),
        };
    },
};

// ============================================================================
// Tool Handler - Dynamic dispatch to native or fallback
// ============================================================================

async function handleToolCall(name: string, args: Record<string, unknown>): Promise<string> {
    // Try native first
    if (native && typeof native[name] === "function") {
        try {
            const result = native[name](...Object.values(args));
            return JSON.stringify(result);
        } catch (e) {
            return JSON.stringify({ success: false, error: `Native error: ${e}` });
        }
    }

    // Try fallback
    if (fallback[name]) {
        try {
            const result = fallback[name](...Object.values(args));
            return JSON.stringify(result);
        } catch (e) {
            return JSON.stringify({ success: false, error: `Fallback error: ${e}` });
        }
    }

    return JSON.stringify({ success: false, error: `Unknown tool: ${name}. Native module may be required.` });
}

// ============================================================================
// Server Setup
// ============================================================================

const server = new Server(
    {
        name: "nautilus-mcp",
        version: "1.0.0",
    },
    {
        capabilities: {
            tools: {},
        },
    }
);

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: allTools };
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
            content: [{ type: "text", text: JSON.stringify({ success: false, error: String(error) }) }],
            isError: true,
        };
    }
});

// ============================================================================
// Main
// ============================================================================

async function main() {
    console.error("╔══════════════════════════════════════════════════════════════╗");
    console.error("║              NAUTILUS MCP SERVER v1.0                        ║");
    console.error("║       High-Performance Trading Analytics Platform            ║");
    console.error("╚══════════════════════════════════════════════════════════════╝");
    console.error("");
    console.error(`  Native Module: ${native ? "✓ Loaded" : "✗ Using fallback"}`);
    console.error(`  Tools Available: ${totalToolCount}`);
    console.error(`  Categories:`);
    for (const [category, count] of Object.entries(toolCategories)) {
        console.error(`    - ${category}: ${count}`);
    }
    console.error("");

    if (!native) {
        console.error("  ⚠️  WARNING: Running without native module");
        console.error("  ⚠️  Some tools require native module for full functionality");
        console.error("  ⚠️  Build with: bun run build:native");
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
