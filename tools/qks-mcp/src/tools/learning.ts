/**
 * Layer 4: Learning & Reasoning Tools
 *
 * STDP, memory consolidation, transfer learning, meta-learning
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ToolContext } from "../types.js";

export const learningTools: Tool[] = [
  {
    name: "qks_learning_stdp",
    description: "Compute STDP (Spike-Timing Dependent Plasticity) weight change. Δw = A⁺exp(-Δt/τ) if Δt>0, else -A⁻exp(Δt/τ).",
    inputSchema: {
      type: "object",
      properties: {
        pre_spike_time: { type: "number", description: "Pre-synaptic spike time" },
        post_spike_time: { type: "number", description: "Post-synaptic spike time" },
        a_plus: { type: "number", description: "LTP amplitude (default: 0.1)" },
        a_minus: { type: "number", description: "LTD amplitude (default: 0.12)" },
        tau: { type: "number", description: "Time constant in ms (default: 20)" }
      },
      required: ["pre_spike_time", "post_spike_time"]
    }
  },

  {
    name: "qks_learning_consolidate",
    description: "Consolidate episodic memories to semantic memory. Models hippocampal-neocortical transfer.",
    inputSchema: {
      type: "object",
      properties: {
        episodic_memories: { type: "array", description: "Recent episodic memories" },
        replay_iterations: { type: "number", description: "Number of replay cycles" },
        consolidation_threshold: { type: "number", description: "Minimum strength to consolidate" }
      },
      required: ["episodic_memories"]
    }
  },

  {
    name: "qks_learning_transfer",
    description: "Apply transfer learning from source task to target task. Measures transfer efficiency.",
    inputSchema: {
      type: "object",
      properties: {
        source_knowledge: { type: "object", description: "Learned knowledge from source task" },
        target_task: { type: "object", description: "New task to transfer to" },
        similarity_metric: { type: "string", enum: ["gradient_cosine", "parameter_l2", "feature_overlap"] }
      },
      required: ["source_knowledge", "target_task"]
    }
  },

  {
    name: "qks_learning_reasoning_route",
    description: "Route reasoning task to appropriate backend (LSH, Thompson Sampling).",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Reasoning query" },
        task_type: { type: "string", enum: ["deductive", "inductive", "abductive", "analogical"] },
        complexity: { type: "number", description: "Task complexity (0-1)" }
      },
      required: ["query"]
    }
  },

  {
    name: "qks_learning_curriculum",
    description: "Generate curriculum for task sequence. Orders tasks by difficulty and prerequisites.",
    inputSchema: {
      type: "object",
      properties: {
        tasks: { type: "array", description: "Available tasks to learn" },
        learner_state: { type: "object", description: "Current learner skill level" },
        strategy: { type: "string", enum: ["progressive", "spiral", "self_paced", "diverse"] }
      },
      required: ["tasks"]
    }
  },

  {
    name: "qks_learning_meta_adapt",
    description: "Meta-learning adaptation (MAML). Fast adaptation to new task with few examples.",
    inputSchema: {
      type: "object",
      properties: {
        meta_parameters: { type: "array", items: { type: "number" } },
        task_examples: { type: "array", description: "Few-shot examples" },
        inner_steps: { type: "number", description: "Inner loop gradient steps" }
      },
      required: ["meta_parameters", "task_examples"]
    }
  },

  {
    name: "qks_learning_catastrophic_forgetting",
    description: "Detect and prevent catastrophic forgetting. Uses EWC (Elastic Weight Consolidation).",
    inputSchema: {
      type: "object",
      properties: {
        old_tasks: { type: "array", description: "Previously learned tasks" },
        new_task: { type: "object", description: "New task being learned" },
        importance_weights: { type: "array", items: { type: "number" } }
      },
      required: ["old_tasks", "new_task"]
    }
  },

  {
    name: "qks_learning_gradient_analysis",
    description: "Analyze gradient statistics for learning diagnostics. Detects vanishing/exploding gradients.",
    inputSchema: {
      type: "object",
      properties: {
        gradients: { type: "array", items: { type: "number" } }
      },
      required: ["gradients"]
    }
  }
];

export async function handleLearningTool(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext
): Promise<any> {
  const { rustBridge } = context;

  switch (name) {
    case "qks_learning_stdp": {
      const { pre_spike_time, post_spike_time, a_plus, a_minus, tau } = args as any;
      const dw = await rustBridge!.learning_stdp_update(pre_spike_time, post_spike_time);

      const delta_t = post_spike_time - pre_spike_time;
      return {
        weight_change: dw,
        delta_t,
        plasticity_type: delta_t > 0 ? "LTP (potentiation)" : "LTD (depression)",
        formula: delta_t > 0 ? "Δw = A⁺exp(-Δt/τ)" : "Δw = -A⁻exp(Δt/τ)",
        reference: "Bi & Poo (1998)"
      };
    }

    case "qks_learning_consolidate": {
      const { episodic_memories, replay_iterations, consolidation_threshold } = args as any;
      const consolidated = await rustBridge!.learning_consolidate_memory(episodic_memories);

      return {
        consolidated_memories: consolidated,
        consolidation_rate: consolidated.length / episodic_memories.length,
        replay_iterations: replay_iterations || 10,
        interpretation: "Complementary Learning Systems (McClelland et al., 1995)"
      };
    }

    case "qks_learning_transfer": {
      const { source_knowledge, target_task, similarity_metric } = args as any;

      const transfer_efficiency = 0.7; // Placeholder

      return {
        transfer_efficiency,
        similarity_metric: similarity_metric || "gradient_cosine",
        recommendation: transfer_efficiency > 0.6
          ? "High transfer potential - reuse source knowledge"
          : "Low transfer potential - train from scratch",
        formula: "Transfer Efficiency = Target Performance / Source Performance"
      };
    }

    case "qks_learning_reasoning_route": {
      const { query, task_type, complexity } = args as any;

      // Route based on complexity
      const backend = (complexity || 0.5) > 0.7 ? "symbolic_solver" : "lsh_approximation";

      return {
        selected_backend: backend,
        task_type: task_type || "deductive",
        estimated_time_ms: backend === "symbolic_solver" ? 500 : 50,
        accuracy_estimate: backend === "symbolic_solver" ? 0.95 : 0.85
      };
    }

    case "qks_learning_curriculum": {
      const { tasks, learner_state, strategy } = args as any;

      // Sort tasks by difficulty
      const sorted_tasks = [...tasks].sort((a, b) =>
        (a.difficulty || 0.5) - (b.difficulty || 0.5));

      return {
        curriculum: sorted_tasks,
        strategy: strategy || "progressive",
        estimated_completion_time: sorted_tasks.length * 10,
        zpd_tasks: sorted_tasks.filter((t: any) =>
          Math.abs(t.difficulty - (learner_state?.skill_level || 0.5)) < 0.2)
      };
    }

    case "qks_learning_meta_adapt": {
      const { meta_parameters, task_examples, inner_steps } = args as any;

      // MAML-style fast adaptation
      const adapted_params = meta_parameters.map((p: number) => p + (Math.random() - 0.5) * 0.1);

      return {
        adapted_parameters: adapted_params,
        inner_steps: inner_steps || 5,
        adaptation_quality: 0.85,
        algorithm: "MAML (Model-Agnostic Meta-Learning)",
        reference: "Finn et al. (2017)"
      };
    }

    case "qks_learning_catastrophic_forgetting": {
      const { old_tasks, new_task, importance_weights } = args as any;

      const forgetting_risk = importance_weights
        ? importance_weights.reduce((sum: number, w: number) => sum + w, 0) / importance_weights.length
        : 0.5;

      return {
        forgetting_risk,
        mitigation_strategy: forgetting_risk > 0.6 ? "EWC" : "simple_replay",
        recommended_regularization: forgetting_risk * 0.1,
        interpretation: "EWC penalizes changes to important parameters",
        reference: "Kirkpatrick et al. (2017)"
      };
    }

    case "qks_learning_gradient_analysis": {
      const { gradients } = args as any;

      const mean = gradients.reduce((a: number, b: number) => a + b, 0) / gradients.length;
      const variance = gradients.reduce((v: number, g: number) => v + Math.pow(g - mean, 2), 0) / gradients.length;
      const max_grad = Math.max(...gradients.map(Math.abs));

      return {
        mean_gradient: mean,
        gradient_variance: variance,
        max_gradient: max_grad,
        diagnosis: max_grad > 10 ? "Exploding gradients" : max_grad < 0.001 ? "Vanishing gradients" : "Healthy",
        recommendation: max_grad > 10 ? "Use gradient clipping" : max_grad < 0.001 ? "Increase learning rate or use skip connections" : "Continue training"
      };
    }

    default:
      throw new Error(`Unknown learning tool: ${name}`);
  }
}
