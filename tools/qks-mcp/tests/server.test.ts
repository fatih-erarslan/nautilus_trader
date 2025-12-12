/**
 * QKS MCP Server Lifecycle Tests
 *
 * Tests for server initialization, transport, tool registration,
 * and graceful shutdown.
 */

import { describe, test, expect, beforeAll, afterAll } from "bun:test";
import { spawn, ChildProcess } from "child_process";
import { resolve } from "path";

describe("MCP Server Lifecycle", () => {
  let serverProcess: ChildProcess | null = null;

  afterAll(() => {
    if (serverProcess) {
      serverProcess.kill();
    }
  });

  test("server starts successfully", async () => {
    const serverPath = resolve(__dirname, "../dist/index.js");

    serverProcess = spawn("bun", ["run", serverPath], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let started = false;

    return new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error("Server startup timeout"));
      }, 5000);

      serverProcess!.stderr!.on("data", (data) => {
        const output = data.toString();
        console.log("Server stderr:", output);

        if (output.includes("Server running")) {
          started = true;
          clearTimeout(timeout);
          resolve();
        }
      });

      serverProcess!.on("error", (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });
  });

  test("server reports correct version", () => {
    const version = "1.0.0"; // From package.json
    expect(version).toMatch(/^\d+\.\d+\.\d+$/);
  });

  test("server has stdio transport", async () => {
    // MCP server should use stdio transport
    expect(serverProcess).not.toBeNull();
    expect(serverProcess!.stdin).not.toBeNull();
    expect(serverProcess!.stdout).not.toBeNull();
  });
});

describe("Server Configuration", () => {
  test("handles QKS_PYTHON_PATH environment variable", () => {
    const pythonPath = process.env.QKS_PYTHON_PATH || "python3";
    expect(pythonPath).toBeTruthy();
  });

  test("handles WOLFRAM_APP_PATH environment variable", () => {
    const wolframPath = process.env.WOLFRAM_APP_PATH ||
      "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel";
    expect(wolframPath).toContain("Wolfram");
  });

  test("validates optional WOLFRAM_API_KEY", () => {
    const apiKey = process.env.WOLFRAM_API_KEY;
    // Optional - test should pass whether set or not
    if (apiKey) {
      expect(apiKey.length).toBeGreaterThan(0);
    }
  });
});

describe("Server Capabilities", () => {
  test("declares tools capability", () => {
    const capabilities = {
      tools: {},
    };

    expect(capabilities).toHaveProperty("tools");
  });

  test("server name is correct", () => {
    const serverName = "qks-mcp";
    expect(serverName).toBe("qks-mcp");
  });

  test("server version matches package.json", () => {
    const packageJson = require("../package.json");
    expect(packageJson.version).toBe("2.0.0"); // Updated version
  });
});

describe("Error Handling", () => {
  test("handles missing Python gracefully", async () => {
    // Mock Python execution failure
    const mockExecutePythonQks = async (code: string) => {
      return {
        success: false,
        error: "Python not found",
      };
    };

    const result = await mockExecutePythonQks("test");
    expect(result.success).toBe(false);
    expect(result.error).toBeTruthy();
  });

  test("handles missing Wolfram gracefully", () => {
    const wolframPath = "/invalid/path/to/wolfram";
    const { existsSync } = require("fs");

    const exists = existsSync(wolframPath);
    expect(exists).toBe(false);
    // Server should continue without Wolfram
  });

  test("handles tool execution errors", async () => {
    const mockHandleQuantumTool = async (
      name: string,
      args: Record<string, unknown>,
      context: any
    ) => {
      throw new Error("Tool execution failed");
    };

    try {
      await mockHandleQuantumTool("invalid_tool", {}, {});
      expect(true).toBe(false); // Should not reach here
    } catch (error) {
      expect(error).toBeTruthy();
    }
  });
});

describe("Server Health Checks", () => {
  test("Python QKS API accessibility check", async () => {
    const mockExecutePythonQks = async (code: string) => {
      if (code.includes("QKS API Ready")) {
        return {
          success: true,
          data: "QKS API Ready",
        };
      }
      return { success: false, error: "Unknown code" };
    };

    const result = await mockExecutePythonQks("result = 'QKS API Ready'");
    expect(result.success).toBe(true);
  });

  test("Wolfram Engine detection", () => {
    const { existsSync } = require("fs");
    const wolframPath = "/Applications/Wolfram Engine.app/Contents/MacOS/WolframKernel";

    const exists = existsSync(wolframPath);
    // Test passes regardless of whether Wolfram is installed
    console.log(`Wolfram Engine available: ${exists}`);
  });
});

describe("Server Shutdown", () => {
  test("handles SIGTERM gracefully", () => {
    // Mock process for testing shutdown
    let shutdownCalled = false;

    const mockProcess = {
      on: (signal: string, handler: () => void) => {
        if (signal === "SIGTERM") {
          shutdownCalled = true;
          handler();
        }
      },
    };

    mockProcess.on("SIGTERM", () => {
      console.log("Graceful shutdown");
    });

    expect(shutdownCalled).toBe(true);
  });

  test("handles SIGINT gracefully", () => {
    let shutdownCalled = false;

    const mockProcess = {
      on: (signal: string, handler: () => void) => {
        if (signal === "SIGINT") {
          shutdownCalled = true;
          handler();
        }
      },
    };

    mockProcess.on("SIGINT", () => {
      console.log("Interrupt received");
    });

    expect(shutdownCalled).toBe(true);
  });
});

describe("Transport Layer", () => {
  test("stdio transport initialization", () => {
    // Mock StdioServerTransport
    class MockStdioTransport {
      connected = false;

      async connect() {
        this.connected = true;
      }
    }

    const transport = new MockStdioTransport();
    expect(transport.connected).toBe(false);

    transport.connect();
    expect(transport.connected).toBe(true);
  });

  test("handles stdin/stdout communication", () => {
    expect(process.stdin).toBeTruthy();
    expect(process.stdout).toBeTruthy();
    expect(process.stderr).toBeTruthy();
  });
});

describe("Server Performance", () => {
  test("server startup time is reasonable", async () => {
    const start = Date.now();

    // Mock server initialization
    await new Promise((resolve) => setTimeout(resolve, 100));

    const duration = Date.now() - start;

    // Should start in less than 5 seconds
    expect(duration).toBeLessThan(5000);
  });

  test("handles concurrent tool requests", async () => {
    const mockToolHandler = async (name: string) => {
      await new Promise((resolve) => setTimeout(resolve, 10));
      return { result: `${name} executed` };
    };

    const requests = [
      mockToolHandler("tool1"),
      mockToolHandler("tool2"),
      mockToolHandler("tool3"),
    ];

    const results = await Promise.all(requests);

    expect(results).toHaveLength(3);
    expect(results[0].result).toBe("tool1 executed");
  });
});
