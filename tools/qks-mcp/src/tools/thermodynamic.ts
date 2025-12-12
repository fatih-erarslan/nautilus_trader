/**
 * Layer 1: Thermodynamic Substrate Tools
 *
 * Scientific foundation for energy, entropy, and temperature dynamics
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ToolContext } from "../types.js";

export const thermodynamicTools: Tool[] = [
  {
    name: "qks_thermo_energy",
    description: "Compute system energy from thermodynamic state. Returns Helmholtz free energy F = E - TS.",
    inputSchema: {
      type: "object",
      properties: {
        state: {
          type: "object",
          description: "Thermodynamic state with energy, temperature, entropy",
          properties: {
            energy: { type: "number" },
            temperature: { type: "number" },
            entropy: { type: "number" }
          }
        }
      },
      required: ["state"]
    }
  },

  {
    name: "qks_thermo_temperature",
    description: "Compute effective temperature from system state. Uses Boltzmann statistics.",
    inputSchema: {
      type: "object",
      properties: {
        state: {
          type: "object",
          description: "System state"
        }
      },
      required: ["state"]
    }
  },

  {
    name: "qks_thermo_entropy",
    description: "Compute Shannon entropy S = -Σ p_i log₂(p_i) from probability distribution.",
    inputSchema: {
      type: "object",
      properties: {
        probabilities: {
          type: "array",
          items: { type: "number" },
          description: "Probability distribution (must sum to 1)"
        }
      },
      required: ["probabilities"]
    }
  },

  {
    name: "qks_thermo_critical_point",
    description: "Get critical point for phase transitions (Ising model, etc.). Returns temperature and reference.",
    inputSchema: {
      type: "object",
      properties: {
        system: {
          type: "string",
          enum: ["ising", "liquid_gas", "ferromagnet"],
          description: "Physical system type"
        }
      },
      required: ["system"]
    }
  },

  {
    name: "qks_thermo_landauer_cost",
    description: "Compute Landauer's minimum energy cost for bit erasure: E = kT ln(2). Fundamental thermodynamic limit.",
    inputSchema: {
      type: "object",
      properties: {
        temperature: {
          type: "number",
          description: "Temperature in Kelvin"
        },
        num_bits: {
          type: "number",
          description: "Number of bits to erase"
        },
        operation: {
          type: "string",
          enum: ["erase", "reversible", "irreversible", "measurement"],
          description: "Operation type"
        }
      },
      required: ["temperature"]
    }
  },

  {
    name: "qks_thermo_free_energy",
    description: "Compute Helmholtz free energy F = E - TS. Minimizing F drives system evolution.",
    inputSchema: {
      type: "object",
      properties: {
        energy: { type: "number" },
        temperature: { type: "number" },
        entropy: { type: "number" }
      },
      required: ["energy", "temperature", "entropy"]
    }
  }
];

export async function handleThermodynamicTool(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext
): Promise<any> {
  const { rustBridge } = context;

  switch (name) {
    case "qks_thermo_energy": {
      const { state } = args;
      const energy = await rustBridge!.thermo_compute_energy(state);
      return { energy, unit: "joules" };
    }

    case "qks_thermo_temperature": {
      const { state } = args;
      const temperature = await rustBridge!.thermo_compute_temperature(state);
      return { temperature, unit: "kelvin" };
    }

    case "qks_thermo_entropy": {
      const { probabilities } = args as { probabilities: number[] };
      // Validate probability distribution
      const sum = probabilities.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1.0) > 1e-6) {
        throw new Error(`Probabilities must sum to 1.0, got ${sum}`);
      }

      const entropy = await rustBridge!.thermo_compute_entropy({ probabilities });
      return {
        entropy,
        unit: "bits",
        formula: "S = -Σ p_i log₂(p_i)",
        max_entropy: Math.log2(probabilities.length)
      };
    }

    case "qks_thermo_critical_point": {
      const { system } = args as { system: string };
      const result = await rustBridge!.thermo_critical_point(system);
      return result;
    }

    case "qks_thermo_landauer_cost": {
      const { temperature, num_bits, operation } = args as any;
      const k_B = 1.380649e-23; // Boltzmann constant (J/K)
      const T = temperature || 300; // Default room temperature
      const n = num_bits || 1;

      let efficiency_factor = 1.0;
      if (operation === "reversible") {
        efficiency_factor = 1.1; // Near-ideal
      } else if (operation === "irreversible") {
        efficiency_factor = 100; // Realistic overhead
      } else if (operation === "measurement") {
        efficiency_factor = 10;
      }

      const landauer_cost = k_B * T * Math.log(2) * n;
      const total_cost = landauer_cost * efficiency_factor;

      return {
        landauer_cost,
        efficiency_factor,
        total_cost,
        unit: "joules",
        formula: "E_min = kT ln(2) per bit",
        reference: "Landauer (1961), Bérut et al. (2012)"
      };
    }

    case "qks_thermo_free_energy": {
      const { energy, temperature, entropy } = args as any;
      const free_energy = energy - temperature * entropy;

      return {
        free_energy,
        energy,
        temperature,
        entropy,
        formula: "F = E - TS",
        interpretation: free_energy < 0
          ? "Spontaneous process (thermodynamically favorable)"
          : "Non-spontaneous (requires energy input)"
      };
    }

    default:
      throw new Error(`Unknown thermodynamic tool: ${name}`);
  }
}
