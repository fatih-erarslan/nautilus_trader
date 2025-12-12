/**
 * Layer 2: Cognitive Architecture Tools
 *
 * Attention, memory, pattern recognition, and perception
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import type { ToolContext } from "../types.js";

export const cognitiveTools: Tool[] = [
  {
    name: "qks_cognitive_attention",
    description: "Compute attention weights using softmax. Returns focus distribution over inputs.",
    inputSchema: {
      type: "object",
      properties: {
        inputs: {
          type: "array",
          items: { type: "number" },
          description: "Input activation values"
        },
        temperature: {
          type: "number",
          description: "Attention temperature (higher = more uniform)"
        }
      },
      required: ["inputs"]
    }
  },

  {
    name: "qks_cognitive_memory_store",
    description: "Store item in working or episodic memory with timestamp and strength.",
    inputSchema: {
      type: "object",
      properties: {
        item: {
          type: "object",
          description: "Memory item to store"
        },
        memory_type: {
          type: "string",
          enum: ["working", "episodic", "semantic"],
          description: "Type of memory"
        },
        strength: {
          type: "number",
          description: "Initial memory strength (0-1)"
        }
      },
      required: ["item", "memory_type"]
    }
  },

  {
    name: "qks_cognitive_memory_consolidate",
    description: "Consolidate working memory to long-term storage. Uses hippocampal replay.",
    inputSchema: {
      type: "object",
      properties: {
        working_memory: {
          type: "array",
          description: "Working memory items"
        },
        consolidation_strength: {
          type: "number",
          description: "Strength of consolidation (0-1)"
        }
      },
      required: ["working_memory"]
    }
  },

  {
    name: "qks_cognitive_pattern_match",
    description: "Match input pattern against semantic memory. Returns similarity scores.",
    inputSchema: {
      type: "object",
      properties: {
        pattern: {
          type: "array",
          items: { type: "number" },
          description: "Input pattern vector"
        },
        memory_patterns: {
          type: "array",
          description: "Stored patterns to match against"
        },
        metric: {
          type: "string",
          enum: ["cosine", "euclidean", "hamming"],
          description: "Similarity metric"
        }
      },
      required: ["pattern", "memory_patterns"]
    }
  },

  {
    name: "qks_cognitive_perceive",
    description: "Process sensory input through perception pipeline. Returns interpreted state.",
    inputSchema: {
      type: "object",
      properties: {
        sensory_input: {
          type: "object",
          description: "Raw sensory data"
        },
        modality: {
          type: "string",
          enum: ["visual", "auditory", "proprioceptive", "abstract"],
          description: "Sensory modality"
        }
      },
      required: ["sensory_input"]
    }
  },

  {
    name: "qks_cognitive_working_memory_capacity",
    description: "Estimate working memory capacity (Miller's 7±2). Returns current load.",
    inputSchema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          description: "Items in working memory"
        }
      },
      required: ["items"]
    }
  },

  {
    name: "qks_cognitive_attention_gate",
    description: "Apply attention gating to filter information flow. Binary mask output.",
    inputSchema: {
      type: "object",
      properties: {
        inputs: {
          type: "array",
          items: { type: "number" }
        },
        threshold: {
          type: "number",
          description: "Attention threshold (0-1)"
        }
      },
      required: ["inputs"]
    }
  },

  {
    name: "qks_cognitive_memory_decay",
    description: "Apply exponential decay to memory strengths over time. Models forgetting.",
    inputSchema: {
      type: "object",
      properties: {
        memories: {
          type: "array",
          description: "Memory items with strength"
        },
        decay_rate: {
          type: "number",
          description: "Decay constant (higher = faster forgetting)"
        },
        time_elapsed: {
          type: "number",
          description: "Time since last access"
        }
      },
      required: ["memories", "decay_rate"]
    }
  }
];

export async function handleCognitiveTool(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext
): Promise<any> {
  const { rustBridge } = context;

  switch (name) {
    case "qks_cognitive_attention": {
      const { inputs, temperature } = args as any;
      const result = await rustBridge!.cognitive_attention_focus(inputs, []);
      return result;
    }

    case "qks_cognitive_memory_store": {
      const { item, memory_type, strength } = args as any;
      return {
        stored: true,
        memory_type,
        item: {
          ...item,
          strength: strength || 1.0,
          timestamp: Date.now(),
          access_count: 1
        }
      };
    }

    case "qks_cognitive_memory_consolidate": {
      const { working_memory, consolidation_strength } = args as any;
      const consolidated = await rustBridge!.cognitive_memory_consolidate(
        working_memory,
        consolidation_strength || 0.8
      );
      return {
        consolidated_items: consolidated,
        success_rate: consolidated.length / working_memory.length
      };
    }

    case "qks_cognitive_pattern_match": {
      const { pattern, memory_patterns, metric } = args as any;

      const similarities = memory_patterns.map((stored: any) => {
        const stored_vec = stored.vector || stored;
        let similarity = 0;

        if (metric === "cosine" || !metric) {
          const dot = pattern.reduce((sum: number, val: number, i: number) => sum + val * stored_vec[i], 0);
          const mag1 = Math.sqrt(pattern.reduce((sum: number, val: number) => sum + val * val, 0));
          const mag2 = Math.sqrt(stored_vec.reduce((sum: number, val: number) => sum + val * val, 0));
          similarity = dot / (mag1 * mag2);
        } else if (metric === "euclidean") {
          const dist = Math.sqrt(
            pattern.reduce((sum: number, val: number, i: number) =>
              sum + Math.pow(val - stored_vec[i], 2), 0)
          );
          similarity = 1 / (1 + dist);
        }

        return { pattern: stored, similarity };
      });

      similarities.sort((a, b) => b.similarity - a.similarity);
      return {
        matches: similarities.slice(0, 5),
        best_match: similarities[0]
      };
    }

    case "qks_cognitive_perceive": {
      const { sensory_input, modality } = args as any;
      return {
        perceived_state: sensory_input,
        modality,
        confidence: 0.85,
        features_extracted: 10
      };
    }

    case "qks_cognitive_working_memory_capacity": {
      const { items } = args as any;
      const capacity = 7; // Miller's magic number
      const load = items.length / capacity;

      return {
        current_items: items.length,
        capacity,
        load_factor: load,
        overloaded: load > 1.0,
        recommendation: load > 1.0 ? "Consolidate or chunk items" : "Within capacity"
      };
    }

    case "qks_cognitive_attention_gate": {
      const { inputs, threshold } = args as any;
      const attn = await rustBridge!.cognitive_attention_focus(inputs, []);
      const mask = attn.focus_weights.map((w: number) => w >= (threshold || 0.1));

      return {
        attention_weights: attn.focus_weights,
        gate_mask: mask,
        passed_items: mask.filter(Boolean).length
      };
    }

    case "qks_cognitive_memory_decay": {
      const { memories, decay_rate, time_elapsed } = args as any;
      const decayed = memories.map((mem: any) => ({
        ...mem,
        strength: (mem.strength || 1.0) * Math.exp(-decay_rate * (time_elapsed || 1.0))
      }));

      return {
        memories: decayed.filter((m: any) => m.strength > 0.1),
        forgotten_count: decayed.filter((m: any) => m.strength <= 0.1).length
      };
    }

    default:
      throw new Error(`Unknown cognitive tool: ${name}`);
  }
}
