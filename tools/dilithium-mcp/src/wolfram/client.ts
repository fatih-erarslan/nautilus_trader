/**
 * Wolfram API Client
 *
 * Hybrid execution strategy:
 * 1. Try local wolframscript first (fastest, <100ms)
 * 2. Fallback to Wolfram Cloud API if credentials available
 * 3. Return structured error if both fail
 *
 * Environment Variables:
 * - WOLFRAM_APP_ID: Wolfram Cloud App ID
 * - WOLFRAM_APP_KEY: Wolfram Cloud App Key
 * - WOLFRAM_SCRIPT_PATH: Path to wolframscript (default: /usr/local/bin/wolframscript)
 */

import { spawn } from "child_process";

/**
 * Wolfram execution mode
 */
export type WolframMode = "local" | "cloud" | "hybrid";

/**
 * Wolfram execution result
 */
export interface WolframResult {
  success: boolean;
  output: string;
  error?: string;
  executionTime: number;
  mode: "local" | "cloud";
  wolframCode: string;
}

/**
 * Wolfram Cloud API configuration
 */
interface WolframCloudConfig {
  appId?: string;
  appKey?: string;
  apiUrl: string;
}

/**
 * Wolfram API Client
 */
export class WolframClient {
  private scriptPath: string;
  private cloudConfig: WolframCloudConfig;
  private mode: WolframMode;

  constructor(mode: WolframMode = "hybrid") {
    this.scriptPath = process.env.WOLFRAM_SCRIPT_PATH || "/usr/local/bin/wolframscript";
    this.cloudConfig = {
      appId: process.env.WOLFRAM_APP_ID,
      appKey: process.env.WOLFRAM_APP_KEY,
      apiUrl: process.env.WOLFRAM_API_URL || "https://api.wolframcloud.com/v1",
    };
    this.mode = mode;
  }

  /**
   * Execute Wolfram Language code
   */
  async execute(code: string, timeoutMs: number = 30000): Promise<WolframResult> {
    const startTime = Date.now();

    // Validate code
    if (!code || code.trim().length === 0) {
      return {
        success: false,
        output: "",
        error: "Empty Wolfram Language code",
        executionTime: Date.now() - startTime,
        mode: "local",
        wolframCode: code,
      };
    }

    // Try execution based on mode
    switch (this.mode) {
      case "local":
        return await this.executeLocal(code, timeoutMs, startTime);

      case "cloud":
        return await this.executeCloud(code, timeoutMs, startTime);

      case "hybrid":
      default:
        // Try local first, fallback to cloud
        const localResult = await this.executeLocal(code, timeoutMs, startTime);
        if (localResult.success) {
          return localResult;
        }

        // If local failed and cloud credentials available, try cloud
        if (this.cloudConfig.appId && this.cloudConfig.appKey) {
          return await this.executeCloud(code, timeoutMs, startTime);
        }

        // Return local error if no cloud credentials
        return localResult;
    }
  }

  /**
   * Execute using local wolframscript
   */
  private async executeLocal(
    code: string,
    timeoutMs: number,
    startTime: number
  ): Promise<WolframResult> {
    return new Promise((resolve) => {
      let stdout = "";
      let stderr = "";
      let timedOut = false;

      // Spawn wolframscript process
      const process = spawn(this.scriptPath, ["-code", code], {
        timeout: timeoutMs,
      });

      // Set timeout
      const timer = setTimeout(() => {
        timedOut = true;
        process.kill("SIGTERM");
        resolve({
          success: false,
          output: stdout,
          error: `Execution timeout after ${timeoutMs}ms`,
          executionTime: Date.now() - startTime,
          mode: "local",
          wolframCode: code,
        });
      }, timeoutMs);

      // Capture stdout
      process.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      // Capture stderr
      process.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      // Handle process completion
      process.on("close", (exitCode) => {
        if (timedOut) return;
        clearTimeout(timer);

        const executionTime = Date.now() - startTime;

        // Check for Wolfram error messages in output
        const hasWolframError = stdout.includes("Syntax::") ||
                               stdout.includes("::sntx") ||
                               stdout.includes("::error") ||
                               stdout.includes("General::") ||
                               stdout.startsWith("$Failed");

        if (exitCode === 0 && !stderr && !hasWolframError) {
          resolve({
            success: true,
            output: stdout.trim(),
            executionTime,
            mode: "local",
            wolframCode: code,
          });
        } else {
          resolve({
            success: false,
            output: stdout.trim(),
            error: stderr || hasWolframError ? "Wolfram syntax error" : `Process exited with code ${exitCode}`,
            executionTime,
            mode: "local",
            wolframCode: code,
          });
        }
      });

      // Handle process errors
      process.on("error", (error) => {
        if (timedOut) return;
        clearTimeout(timer);

        resolve({
          success: false,
          output: stdout,
          error: `Wolframscript error: ${error.message}. Is wolframscript installed?`,
          executionTime: Date.now() - startTime,
          mode: "local",
          wolframCode: code,
        });
      });
    });
  }

  /**
   * Execute using Wolfram Cloud API
   */
  private async executeCloud(
    code: string,
    timeoutMs: number,
    startTime: number
  ): Promise<WolframResult> {
    if (!this.cloudConfig.appId || !this.cloudConfig.appKey) {
      return {
        success: false,
        output: "",
        error: "Wolfram Cloud credentials not configured. Set WOLFRAM_APP_ID and WOLFRAM_APP_KEY environment variables.",
        executionTime: Date.now() - startTime,
        mode: "cloud",
        wolframCode: code,
      };
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

      const response = await fetch(`${this.cloudConfig.apiUrl}/evaluate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Wolfram-AppId": this.cloudConfig.appId,
          "X-Wolfram-AppKey": this.cloudConfig.appKey,
        },
        body: JSON.stringify({
          input: code,
          format: "text",
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const executionTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          output: data.result || data.output || String(data),
          executionTime,
          mode: "cloud",
          wolframCode: code,
        };
      } else {
        const errorText = await response.text();
        return {
          success: false,
          output: "",
          error: `Wolfram Cloud API error (${response.status}): ${errorText}`,
          executionTime,
          mode: "cloud",
          wolframCode: code,
        };
      }
    } catch (error: any) {
      return {
        success: false,
        output: "",
        error: error.name === "AbortError"
          ? `Wolfram Cloud API timeout after ${timeoutMs}ms`
          : `Wolfram Cloud API error: ${error.message}`,
        executionTime: Date.now() - startTime,
        mode: "cloud",
        wolframCode: code,
      };
    }
  }

  /**
   * Check if local wolframscript is available
   */
  async isLocalAvailable(): Promise<boolean> {
    try {
      const result = await this.executeLocal("Print[1+1]", 5000, Date.now());
      return result.success && result.output === "2";
    } catch {
      return false;
    }
  }

  /**
   * Check if Wolfram Cloud API is configured
   */
  isCloudConfigured(): boolean {
    return Boolean(this.cloudConfig.appId && this.cloudConfig.appKey);
  }

  /**
   * Get client status
   */
  async getStatus(): Promise<{
    localAvailable: boolean;
    cloudConfigured: boolean;
    recommendedMode: WolframMode;
  }> {
    const localAvailable = await this.isLocalAvailable();
    const cloudConfigured = this.isCloudConfigured();

    let recommendedMode: WolframMode = "hybrid";
    if (localAvailable && !cloudConfigured) {
      recommendedMode = "local";
    } else if (!localAvailable && cloudConfigured) {
      recommendedMode = "cloud";
    }

    return {
      localAvailable,
      cloudConfigured,
      recommendedMode,
    };
  }
}

/**
 * Global Wolfram client instance (singleton)
 */
let globalClient: WolframClient | null = null;

/**
 * Get or create global Wolfram client
 */
export function getWolframClient(mode?: WolframMode): WolframClient {
  if (!globalClient) {
    globalClient = new WolframClient(mode || "hybrid");
  }
  return globalClient;
}

/**
 * Execute Wolfram Language code (convenience function)
 */
export async function executeWolfram(
  code: string,
  timeoutMs?: number
): Promise<WolframResult> {
  const client = getWolframClient();
  return await client.execute(code, timeoutMs);
}
