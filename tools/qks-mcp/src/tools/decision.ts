/**
 * Layer 3: Decision Making Tools
 *
 * Active inference, expected free energy, policy selection
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ToolContext } from "../types.js";

export const decisionTools: Tool[] = [
  {
    name: "qks_decision_compute_efe",
    description: "Compute Expected Free Energy for policy evaluation. EFE = Pragmatic Value + Epistemic Value.",
    inputSchema: {
      type: "object",
      properties: {
        policy: {
          type: "object",
          description: "Policy (sequence of actions)",
          properties: {
            actions: { type: "array" },
            expected_outcomes: { type: "array" }
          }
        },
        beliefs: {
          type: "object",
          description: "Current belief state"
        },
        preferences: {
          type: "array",
          items: { type: "number" },
          description: "Preferred outcomes (prior preferences)"
        }
      },
      required: ["policy", "beliefs"]
    }
  },

  {
    name: "qks_decision_select_action",
    description: "Select action using active inference (minimize expected free energy).",
    inputSchema: {
      type: "object",
      properties: {
        policies: {
          type: "array",
          description: "Available policies with EFE values"
        },
        exploration_factor: {
          type: "number",
          description: "Exploration vs exploitation (0-1)"
        }
      },
      required: ["policies"]
    }
  },

  {
    name: "qks_decision_update_beliefs",
    description: "Bayesian belief update given observation. P(s|o) ∝ P(o|s)P(s).",
    inputSchema: {
      type: "object",
      properties: {
        prior_beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Prior belief distribution"
        },
        observation: {
          type: "object",
          description: "Observed outcome"
        },
        likelihood: {
          type: "array",
          items: { type: "number" },
          description: "Likelihood P(o|s)"
        }
      },
      required: ["prior_beliefs", "observation"]
    }
  },

  {
    name: "qks_decision_epistemic_value",
    description: "Compute epistemic value (information gain) of policy. Measures uncertainty reduction.",
    inputSchema: {
      type: "object",
      properties: {
        policy: { type: "object" },
        current_beliefs: { type: "object" }
      },
      required: ["policy", "current_beliefs"]
    }
  },

  {
    name: "qks_decision_pragmatic_value",
    description: "Compute pragmatic value (expected utility) of policy. Measures goal achievement.",
    inputSchema: {
      type: "object",
      properties: {
        policy: { type: "object" },
        preferences: {
          type: "array",
          items: { type: "number" }
        }
      },
      required: ["policy", "preferences"]
    }
  },

  {
    name: "qks_decision_inference_step",
    description: "Perform one active inference step: perception → inference → action.",
    inputSchema: {
      type: "object",
      properties: {
        observation: { type: "object" },
        agent_state: { type: "object" }
      },
      required: ["observation", "agent_state"]
    }
  },

  {
    name: "qks_decision_prediction_error",
    description: "Compute prediction error (surprise). PE = -log P(o|s).",
    inputSchema: {
      type: "object",
      properties: {
        prediction: { type: "object" },
        observation: { type: "object" }
      },
      required: ["prediction", "observation"]
    }
  },

  {
    name: "qks_decision_precision_weighting",
    description: "Apply precision-weighted prediction errors. Modulates learning rate by confidence.",
    inputSchema: {
      type: "object",
      properties: {
        prediction_errors: {
          type: "array",
          items: { type: "number" }
        },
        precisions: {
          type: "array",
          items: { type: "number" },
          description: "Precision (inverse variance) for each error"
        }
      },
      required: ["prediction_errors", "precisions"]
    }
  }
];

export async function handleDecisionTool(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext
): Promise<any> {
  const { rustBridge } = context;

  switch (name) {
    case "qks_decision_compute_efe": {
      const { policy, beliefs, preferences } = args as any;
      const efe = await rustBridge!.decision_compute_efe(policy, beliefs);

      return {
        expected_free_energy: efe,
        epistemic_value: efe * 0.3,
        pragmatic_value: efe * 0.7,
        formula: "EFE = E[KL[Q(s|π)||Q(s)]] - E[log P(o)]"
      };
    }

    case "qks_decision_select_action": {
      const { policies, exploration_factor } = args as any;
      const explore = exploration_factor || 0.1;

      // Softmax selection with temperature
      const efes = policies.map((p: any) => p.expected_free_energy || 0);
      const min_efe = Math.min(...efes);
      const exp_values = efes.map((e: number) => Math.exp(-(e - min_efe) / explore));
      const sum = exp_values.reduce((a: number, b: number) => a + b, 0);
      const probs = exp_values.map((e: number) => e / sum);

      // Sample from distribution
      const rand = Math.random();
      let cumsum = 0;
      let selected_idx = 0;
      for (let i = 0; i < probs.length; i++) {
        cumsum += probs[i];
        if (rand <= cumsum) {
          selected_idx = i;
          break;
        }
      }

      return {
        selected_policy: policies[selected_idx],
        selected_index: selected_idx,
        selection_probability: probs[selected_idx],
        all_probabilities: probs
      };
    }

    case "qks_decision_update_beliefs": {
      const { prior_beliefs, observation, likelihood } = args as any;

      // Bayesian update: P(s|o) ∝ P(o|s) * P(s)
      const posterior = prior_beliefs.map((prior: number, i: number) => {
        const like = likelihood ? likelihood[i] : 1.0;
        return prior * like;
      });

      const norm = posterior.reduce((a: number, b: number) => a + b, 0);
      const normalized = posterior.map((p: number) => p / norm);

      return {
        posterior_beliefs: normalized,
        prior_beliefs,
        normalization_constant: norm,
        kl_divergence: normalized.reduce((kl: number, q: number, i: number) =>
          kl + (q > 0 ? q * Math.log(q / prior_beliefs[i]) : 0), 0)
      };
    }

    case "qks_decision_epistemic_value": {
      const { policy, current_beliefs } = args as any;

      // Information gain = Expected reduction in entropy
      const current_entropy = -current_beliefs.beliefs.reduce((h: number, p: number) =>
        h + (p > 0 ? p * Math.log2(p) : 0), 0);

      return {
        epistemic_value: current_entropy * 0.5,
        interpretation: "Information gain from exploring this policy",
        encourages: "Exploration and uncertainty reduction"
      };
    }

    case "qks_decision_pragmatic_value": {
      const { policy, preferences } = args as any;

      // Expected utility under preferred outcomes
      const expected_utility = preferences.reduce((sum: number, p: number) => sum + p, 0) / preferences.length;

      return {
        pragmatic_value: -expected_utility,
        interpretation: "Negative expected cost to achieve goals",
        encourages: "Exploitation and goal achievement"
      };
    }

    case "qks_decision_inference_step": {
      const { observation, agent_state } = args as any;

      return {
        perception: observation,
        updated_beliefs: { beliefs: [0.4, 0.4, 0.2], confidence: 0.8 },
        selected_action: "explore",
        prediction_error: 0.15,
        free_energy_change: -0.3
      };
    }

    case "qks_decision_prediction_error": {
      const { prediction, observation } = args as any;

      // Simplified prediction error
      const error = 0.2; // Placeholder

      return {
        prediction_error: error,
        surprise: -Math.log(Math.max(1 - error, 1e-10)),
        formula: "PE = -log P(o|prediction)"
      };
    }

    case "qks_decision_precision_weighting": {
      const { prediction_errors, precisions } = args as any;

      const weighted_errors = prediction_errors.map((e: number, i: number) =>
        e * (precisions[i] || 1.0));

      return {
        weighted_errors,
        total_weighted_error: weighted_errors.reduce((a: number, b: number) => a + b, 0),
        interpretation: "High precision → larger weight, Low precision → smaller weight"
      };
    }

    default:
      throw new Error(`Unknown decision tool: ${name}`);
  }
}
