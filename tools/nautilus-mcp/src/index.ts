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
    // Dilithium Post-Quantum Crypto
    dilithium_keygen(): { public_key: string; secret_key: string };
    dilithium_sign(secret_key: string, message: string): string;
    dilithium_verify(public_key: string, signature: string, message: string): boolean;
    blake3_hash(data: string): string;
    generate_nonce(): string;

    // Trading Analytics - accepts any function name
    [key: string]: ((...args: unknown[]) => AnalyticsResult) | unknown;
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
// Fallback Implementations (Pure TypeScript) - Wolfram-verified formulas
// ============================================================================

const fallback: Record<string, (args: Record<string, unknown>) => AnalyticsResult> = {
    // Simple Moving Average
    indicator_sma: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const n = args.period as number;
        if (!p || p.length < n || n === 0) return { success: false, error: "Insufficient data for SMA" };
        const sum = p.slice(-n).reduce((a, b) => a + b, 0);
        return { success: true, value: sum / n };
    },

    // Exponential Moving Average
    indicator_ema: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const n = args.period as number;
        if (!p || p.length === 0 || n === 0) return { success: false, error: "Insufficient data for EMA" };
        const multiplier = 2 / (n + 1);
        let ema = p[0];
        for (let i = 1; i < p.length; i++) {
            ema = (p[i] - ema) * multiplier + ema;
        }
        return { success: true, value: ema };
    },

    // Triple EMA - Wolfram: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    indicator_tema: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const n = args.period as number;
        if (!p || p.length === 0 || n === 0) return { success: false, error: "Insufficient data for TEMA" };
        const multiplier = 2 / (n + 1);
        let ema1 = p[0], ema2 = p[0], ema3 = p[0];
        for (let i = 1; i < p.length; i++) {
            ema1 = (p[i] - ema1) * multiplier + ema1;
            ema2 = (ema1 - ema2) * multiplier + ema2;
            ema3 = (ema2 - ema3) * multiplier + ema3;
        }
        return { success: true, value: 3 * ema1 - 3 * ema2 + ema3 };
    },

    // RSI
    indicator_rsi: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const n = args.period as number;
        if (!p || p.length < n + 1) return { success: false, error: "Insufficient data for RSI" };
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

    // MACD
    indicator_macd: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const fast = (args.fast as number) || 12;
        const slow = (args.slow as number) || 26;
        const signal = (args.signal as number) || 9;
        if (!p || p.length < slow) return { success: false, error: "Insufficient data for MACD" };

        const fastMult = 2 / (fast + 1);
        const slowMult = 2 / (slow + 1);
        const signalMult = 2 / (signal + 1);

        let fastEma = p[0], slowEma = p[0], signalLine = 0, macdLine = 0;
        for (let i = 0; i < p.length; i++) {
            fastEma = (p[i] - fastEma) * fastMult + fastEma;
            slowEma = (p[i] - slowEma) * slowMult + slowEma;
            macdLine = fastEma - slowEma;
            if (i >= slow) signalLine = (macdLine - signalLine) * signalMult + signalLine;
        }
        return { success: true, data: JSON.stringify({ macd: macdLine, signal: signalLine, histogram: macdLine - signalLine }) };
    },

    // Bollinger Bands
    indicator_bollinger: (args): AnalyticsResult => {
        const p = args.prices as number[];
        const n = args.period as number;
        const stdDev = (args.std_dev as number) || 2;
        if (!p || p.length < n) return { success: false, error: "Insufficient data for Bollinger" };

        const recent = p.slice(-n);
        const sma = recent.reduce((a, b) => a + b, 0) / n;
        const variance = recent.reduce((sum, x) => sum + (x - sma) ** 2, 0) / n;
        const std = Math.sqrt(variance);
        return { success: true, data: JSON.stringify({ upper: sma + stdDev * std, middle: sma, lower: sma - stdDev * std, std }) };
    },

    // VaR Parametric
    risk_var_parametric: (args): AnalyticsResult => {
        const r = args.returns as number[];
        const c = args.confidence as number;
        if (!r || r.length === 0) return { success: false, error: "No returns" };
        const mean = r.reduce((a, b) => a + b, 0) / r.length;
        const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
        const std = Math.sqrt(variance);
        const z = c === 0.99 ? 2.326 : c === 0.95 ? 1.645 : 1.282;
        return { success: true, value: -(mean - z * std) };
    },

    // Omega Ratio - Wolfram: Sum(max(R-t,0)) / Sum(max(t-R,0))
    risk_omega_ratio: (args): AnalyticsResult => {
        const r = args.returns as number[];
        const threshold = (args.threshold as number) || 0;
        if (!r || r.length === 0) return { success: false, error: "No returns" };
        const gains = r.reduce((sum, x) => sum + Math.max(x - threshold, 0), 0);
        const losses = r.reduce((sum, x) => sum + Math.max(threshold - x, 0), 0);
        if (losses === 0) return { success: true, value: Infinity };
        return { success: true, value: gains / losses };
    },

    // Sharpe Ratio
    portfolio_sharpe: (args): AnalyticsResult => {
        const r = args.returns as number[];
        const riskFree = (args.risk_free_rate as number) || 0;
        if (!r || r.length === 0) return { success: false, error: "No returns" };
        const mean = r.reduce((a, b) => a + b, 0) / r.length;
        const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
        const std = Math.sqrt(variance);
        if (std === 0) return { success: true, value: 0 };
        return { success: true, value: (mean - riskFree) / std };
    },

    // Kelly Criterion
    risk_kelly_criterion: (args): AnalyticsResult => {
        const w = args.win_rate as number;
        const wl = args.win_loss_ratio as number;
        if (w <= 0 || w >= 1 || wl <= 0) return { success: false, error: "Invalid parameters" };
        const kelly = (w * wl - (1 - w)) / wl;
        return { success: true, data: JSON.stringify({ kelly_fraction: kelly, half_kelly: kelly / 2, quarter_kelly: kelly / 4 }) };
    },

    // pBit State - HyperPhysics integration
    regime_pbit_state: (args): AnalyticsResult => {
        const s = args.market_signal as number;
        const v = args.volatility as number;
        const t = args.temperature as number;
        const effectiveTemp = t * Math.max(v, 0.01);
        const probBullish = 1 / (1 + Math.exp(-s / effectiveTemp));
        const state = probBullish > 0.65 ? "bullish" : probBullish < 0.35 ? "bearish" : "neutral";
        const entropy = probBullish > 0 && probBullish < 1 ? -probBullish * Math.log(probBullish) - (1 - probBullish) * Math.log(1 - probBullish) : 0;
        return { success: true, data: JSON.stringify({ prob_bullish: probBullish, prob_bearish: 1 - probBullish, state, entropy, temperature: effectiveTemp }) };
    },

    // Max Drawdown
    risk_max_drawdown: (args): AnalyticsResult => {
        const equity = args.equity_curve as number[];
        if (!equity || equity.length === 0) return { success: false, error: "No equity data" };
        let maxDd = 0, peak = equity[0];
        for (const e of equity) {
            if (e > peak) peak = e;
            const dd = (peak - e) / peak;
            if (dd > maxDd) maxDd = dd;
        }
        return { success: true, data: JSON.stringify({ max_drawdown: maxDd, max_drawdown_pct: maxDd * 100 }) };
    },
};

// ============================================================================
// Tool Handler - Dynamic dispatch to native or fallback with NAMED args
// ============================================================================

async function handleToolCall(name: string, args: Record<string, unknown>): Promise<string> {
    // Try native first (pass as array for NAPI compatibility)
    if (native && typeof native[name] === "function") {
        try {
            const result = native[name](...Object.values(args));
            return JSON.stringify(result);
        } catch (e) {
            // Fall through to try fallback
            console.error(`[Nautilus] Native call failed for ${name}: ${e}`);
        }
    }

    // Try fallback with named args
    if (fallback[name]) {
        try {
            const result = fallback[name](args);
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
