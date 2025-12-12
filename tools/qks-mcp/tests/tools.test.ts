/**
 * QKS MCP Tools Registration and Schema Tests
 *
 * Tests for tool definitions, input schemas, and registration.
 */

import { describe, test, expect } from "bun:test";
import { quantumTools, handleQuantumTool } from "../src/tools/index.js";

describe("Tool Registration", () => {
  test("all required tools are registered", () => {
    const expectedTools = [
      "qks_execute_circuit",
      "qks_vqe_optimize",
      "qks_state_analysis",
      "qks_hyperbolic_embedding",
      "qks_pbit_detector",
      "qks_stdp_learning",
      "qks_swarm_tune",
      "qks_wolfram_code",
      "qks_wolfram_verify",
      "qks_device_info",
    ];

    const registeredNames = quantumTools.map((tool) => tool.name);

    for (const expectedTool of expectedTools) {
      expect(registeredNames).toContain(expectedTool);
    }
  });

  test("tool count matches specification", () => {
    // Should have exactly 10 tools as defined in specification
    expect(quantumTools.length).toBe(10);
  });

  test("all tools have valid names", () => {
    for (const tool of quantumTools) {
      expect(tool.name).toBeTruthy();
      expect(tool.name.startsWith("qks_")).toBe(true);
    }
  });
});

describe("Tool Schemas", () => {
  test("qks_execute_circuit schema is valid", () => {
    const tool = quantumTools.find((t) => t.name === "qks_execute_circuit");
    expect(tool).toBeTruthy();

    const schema = tool!.inputSchema;
    expect(schema.type).toBe("object");
    expect(schema.properties).toHaveProperty("num_qubits");
    expect(schema.properties).toHaveProperty("gates");
    expect(schema.required).toContain("num_qubits");
    expect(schema.required).toContain("gates");
  });

  test("qks_vqe_optimize schema is valid", () => {
    const tool = quantumTools.find((t) => t.name === "qks_vqe_optimize");
    expect(tool).toBeTruthy();

    const schema = tool!.inputSchema;
    expect(schema.properties).toHaveProperty("hamiltonian");
    expect(schema.properties).toHaveProperty("ansatz");
    expect(schema.properties).toHaveProperty("strategy");
    expect(schema.required).toContain("hamiltonian");
  });

  test("qks_state_analysis schema is valid", () => {
    const tool = quantumTools.find((t) => t.name === "qks_state_analysis");
    expect(tool).toBeTruthy();

    const schema = tool!.inputSchema;
    expect(schema.properties).toHaveProperty("state");
    expect(schema.properties).toHaveProperty("metrics");
    expect(schema.required).toContain("state");
    expect(schema.required).toContain("metrics");
  });

  test("qks_wolfram_code schema has enum validation", () => {
    const tool = quantumTools.find((t) => t.name === "qks_wolfram_code");
    expect(tool).toBeTruthy();

    const schema = tool!.inputSchema;
    const languageEnum = schema.properties.language.enum;

    expect(languageEnum).toContain("python");
    expect(languageEnum).toContain("rust");
    expect(languageEnum).toContain("typescript");
    expect(languageEnum).toContain("wolfram");
  });
});

describe("Tool Descriptions", () => {
  test("all tools have descriptions", () => {
    for (const tool of quantumTools) {
      expect(tool.description).toBeTruthy();
      expect(tool.description.length).toBeGreaterThan(10);
    }
  });

  test("descriptions are informative", () => {
    const tool = quantumTools.find((t) => t.name === "qks_execute_circuit");
    expect(tool!.description).toContain("quantum circuit");
    expect(tool!.description).toContain("Metal GPU");
  });
});

describe("Tool Input Validation", () => {
  test("validates required fields", () => {
    const tool = quantumTools.find((t) => t.name === "qks_execute_circuit");
    const required = tool!.inputSchema.required;

    expect(required).toContain("num_qubits");
    expect(required).toContain("gates");
    expect(required).not.toContain("shots"); // Optional
  });

  test("validates array types", () => {
    const tool = quantumTools.find((t) => t.name === "qks_execute_circuit");
    const gatesProperty = tool!.inputSchema.properties.gates;

    expect(gatesProperty.type).toBe("array");
    expect(gatesProperty.items).toBeTruthy();
  });

  test("validates enum constraints", () => {
    const tool = quantumTools.find((t) => t.name === "qks_vqe_optimize");
    const strategyProperty = tool!.inputSchema.properties.strategy;

    expect(strategyProperty.enum).toContain("grey_wolf");
    expect(strategyProperty.enum).toContain("particle_swarm");
    expect(strategyProperty.enum).toContain("whale");
  });
});

describe("Tool Handler Dispatch", () => {
  test("dispatches qks_device_info correctly", async () => {
    const mockContext = {
      executePythonQks: async (code: string) => ({
        success: true,
        data: { devices: [{ type: "cpu", info: { max_qubits: 30 } }] },
      }),
      executeWolframLocal: async () => ({ success: true }),
      queryWolframLLM: async () => ({ success: true }),
    };

    const result = await handleQuantumTool("qks_device_info", {}, mockContext);

    expect(result.success).toBe(true);
    expect(result.data.devices).toBeTruthy();
  });

  test("handles unknown tool names", async () => {
    const mockContext = {
      executePythonQks: async () => ({ success: true }),
      executeWolframLocal: async () => ({ success: true }),
      queryWolframLLM: async () => ({ success: true }),
    };

    try {
      await handleQuantumTool("invalid_tool_name", {}, mockContext);
      expect(true).toBe(false); // Should not reach here
    } catch (error: any) {
      expect(error.message).toContain("Unknown tool");
    }
  });
});

describe("Tool Categories", () => {
  test("quantum circuit tools are present", () => {
    const circuitTools = quantumTools.filter((t) =>
      t.name.includes("circuit") || t.name.includes("vqe")
    );

    expect(circuitTools.length).toBeGreaterThan(0);
  });

  test("HyperPhysics tools are present", () => {
    const hyperphysicsTools = quantumTools.filter(
      (t) =>
        t.name.includes("hyperbolic") ||
        t.name.includes("pbit") ||
        t.name.includes("stdp") ||
        t.name.includes("swarm")
    );

    expect(hyperphysicsTools.length).toBeGreaterThan(0);
  });

  test("Wolfram integration tools are present", () => {
    const wolframTools = quantumTools.filter((t) =>
      t.name.includes("wolfram")
    );

    expect(wolframTools.length).toBe(2); // code and verify
  });
});

describe("Tool Documentation", () => {
  test("tools have property descriptions", () => {
    const tool = quantumTools.find((t) => t.name === "qks_execute_circuit");
    const properties = tool!.inputSchema.properties;

    expect(properties.num_qubits.description).toBeTruthy();
    expect(properties.gates.description).toBeTruthy();
  });

  test("complex types have detailed descriptions", () => {
    const tool = quantumTools.find((t) => t.name === "qks_state_analysis");
    const stateProperty = tool!.inputSchema.properties.state;

    expect(stateProperty.description).toContain("State vector");
    expect(stateProperty.description).toContain("complex");
  });
});

describe("Tool Completeness", () => {
  test("Layer 1 thermodynamic tools", () => {
    // Covered by qks_pbit_detector, qks_state_analysis
    const layer1Tools = quantumTools.filter(
      (t) => t.name.includes("pbit") || t.name.includes("state")
    );
    expect(layer1Tools.length).toBeGreaterThan(0);
  });

  test("Layer 3 decision tools", () => {
    // Covered by qks_swarm_tune, qks_vqe_optimize
    const layer3Tools = quantumTools.filter(
      (t) => t.name.includes("swarm") || t.name.includes("vqe")
    );
    expect(layer3Tools.length).toBeGreaterThan(0);
  });

  test("Layer 4 learning tools", () => {
    // Covered by qks_stdp_learning
    const layer4Tools = quantumTools.filter((t) => t.name.includes("stdp"));
    expect(layer4Tools.length).toBeGreaterThan(0);
  });
});

describe("Tool Versioning", () => {
  test("tool definitions are stable", () => {
    // Verify that tool names haven't changed
    const expectedNames = [
      "qks_execute_circuit",
      "qks_vqe_optimize",
      "qks_state_analysis",
    ];

    const actualNames = quantumTools.map((t) => t.name);

    for (const name of expectedNames) {
      expect(actualNames).toContain(name);
    }
  });
});
