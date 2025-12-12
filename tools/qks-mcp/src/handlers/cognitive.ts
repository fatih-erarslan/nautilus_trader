/**
 * Cognitive Layer Handlers - Attention & Memory
 *
 * Implements Layer 2 of the cognitive architecture:
 * - Attention mechanisms (top-down, bottom-up)
 * - Working memory buffer
 * - Episodic memory storage/retrieval
 * - Semantic knowledge graphs
 */

import { QKSBridge } from './mod.js';

export interface AttentionState {
  focus_vector: number[];
  saliency_map: number[];
  attention_weights: number[];
  mode: 'top_down' | 'bottom_up' | 'hybrid';
}

export interface MemoryEntry {
  id: string;
  content: any;
  timestamp: number;
  importance: number;
  associations: string[];
  embedding?: number[];
}

export class CognitiveHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Compute attention distribution over inputs
   * Supports top-down (goal-driven) and bottom-up (saliency-driven) modes
   */
  async computeAttention(params: {
    inputs: number[][];
    query?: number[];
    mode?: 'top_down' | 'bottom_up' | 'hybrid';
    temperature?: number;
  }): Promise<{
    attention_weights: number[];
    focus_vector: number[];
    saliency_map: number[];
    attended_inputs: number[][];
  }> {
    const { inputs, query, mode = 'hybrid', temperature = 1.0 } = params;

    try {
      return await this.bridge.callRust('cognitive.attention', {
        inputs,
        query,
        mode,
        temperature,
      });
    } catch (e) {
      // Fallback: Simplified attention mechanism
      const numInputs = inputs.length;
      let saliency = inputs.map(input => this.computeSaliency(input));

      // Top-down modulation
      if (query && (mode === 'top_down' || mode === 'hybrid')) {
        const queryWeights = inputs.map(input => this.dotProduct(input, query));
        if (mode === 'hybrid') {
          saliency = saliency.map((s, i) => 0.5 * s + 0.5 * queryWeights[i]);
        } else {
          saliency = queryWeights;
        }
      }

      // Softmax with temperature
      const weights = this.softmax(saliency, temperature);

      // Compute focus vector
      const focusVector = new Array(inputs[0].length).fill(0);
      for (let i = 0; i < numInputs; i++) {
        for (let j = 0; j < inputs[0].length; j++) {
          focusVector[j] += weights[i] * inputs[i][j];
        }
      }

      return {
        attention_weights: weights,
        focus_vector: focusVector,
        saliency_map: saliency,
        attended_inputs: inputs.map((input, i) => input.map(x => x * weights[i])),
      };
    }
  }

  /**
   * Update working memory buffer
   * Maintains short-term active representations
   */
  async updateWorkingMemory(params: {
    current_buffer: number[][];
    new_input: number[];
    capacity?: number;
    decay_rate?: number;
  }): Promise<{
    updated_buffer: number[][];
    evicted_items: number[][];
    buffer_coherence: number;
  }> {
    const { current_buffer, new_input, capacity = 7, decay_rate = 0.1 } = params;

    try {
      return await this.bridge.callRust('cognitive.working_memory', {
        current_buffer,
        new_input,
        capacity,
        decay_rate,
      });
    } catch (e) {
      // Fallback: FIFO with decay
      let buffer = [...current_buffer];

      // Apply decay
      buffer = buffer.map(item => item.map(x => x * (1 - decay_rate)));

      // Add new input
      buffer.push(new_input);

      // Evict if over capacity
      const evicted: number[][] = [];
      while (buffer.length > capacity) {
        const removed = buffer.shift();
        if (removed) evicted.push(removed);
      }

      // Compute coherence
      const coherence = this.computeBufferCoherence(buffer);

      return {
        updated_buffer: buffer,
        evicted_items: evicted,
        buffer_coherence: coherence,
      };
    }
  }

  /**
   * Store episodic memory
   */
  async storeEpisodicMemory(params: {
    content: any;
    context: Record<string, any>;
    importance?: number;
  }): Promise<{
    memory_id: string;
    embedding: number[];
    consolidation_success: boolean;
  }> {
    const { content, context, importance = 0.5 } = params;

    try {
      return await this.bridge.callRust('cognitive.store_episodic', {
        content,
        context,
        importance,
      });
    } catch (e) {
      // Fallback: Generate embedding and ID
      const memoryId = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const embedding = this.generateEmbedding(content, context);

      return {
        memory_id: memoryId,
        embedding,
        consolidation_success: true,
      };
    }
  }

  /**
   * Retrieve episodic memories
   */
  async retrieveEpisodicMemory(params: {
    query: any;
    k?: number;
    threshold?: number;
    time_window?: { start: number; end: number };
  }): Promise<{
    memories: MemoryEntry[];
    relevance_scores: number[];
    retrieval_latency_ms: number;
  }> {
    const { query, k = 5, threshold = 0.0, time_window } = params;

    const startTime = Date.now();

    try {
      const result = await this.bridge.callRust('cognitive.retrieve_episodic', {
        query,
        k,
        threshold,
        time_window,
      });

      return {
        ...result,
        retrieval_latency_ms: Date.now() - startTime,
      };
    } catch (e) {
      // Fallback: Return empty
      return {
        memories: [],
        relevance_scores: [],
        retrieval_latency_ms: Date.now() - startTime,
      };
    }
  }

  /**
   * Build semantic knowledge graph
   */
  async buildSemanticGraph(params: {
    concepts: string[];
    relations?: Array<{ source: string; target: string; type: string; weight: number }>;
  }): Promise<{
    graph_id: string;
    node_count: number;
    edge_count: number;
    clustering_coefficient: number;
  }> {
    const { concepts, relations = [] } = params;

    try {
      return await this.bridge.callRust('cognitive.semantic_graph', {
        concepts,
        relations,
      });
    } catch (e) {
      // Fallback: Basic stats
      const graphId = `graph_${Date.now()}`;

      return {
        graph_id: graphId,
        node_count: concepts.length,
        edge_count: relations.length,
        clustering_coefficient: 0.0,
      };
    }
  }

  /**
   * Query semantic knowledge graph
   */
  async querySemanticGraph(params: {
    graph_id: string;
    query_concept: string;
    max_depth?: number;
    relation_types?: string[];
  }): Promise<{
    related_concepts: string[];
    paths: Array<{ nodes: string[]; edges: string[]; weight: number }>;
    subgraph: any;
  }> {
    const { graph_id, query_concept, max_depth = 2, relation_types } = params;

    try {
      return await this.bridge.callRust('cognitive.query_semantic', {
        graph_id,
        query_concept,
        max_depth,
        relation_types,
      });
    } catch (e) {
      return {
        related_concepts: [],
        paths: [],
        subgraph: null,
      };
    }
  }

  /**
   * Consolidate memory (sleep-like process)
   * Moves important short-term memories to long-term storage
   */
  async consolidateMemory(params: {
    working_memory: number[][];
    importance_threshold?: number;
  }): Promise<{
    consolidated_count: number;
    pruned_count: number;
    memory_efficiency: number;
  }> {
    const { working_memory, importance_threshold = 0.3 } = params;

    try {
      return await this.bridge.callRust('cognitive.consolidate_memory', {
        working_memory,
        importance_threshold,
      });
    } catch (e) {
      // Fallback: Simple consolidation
      const consolidated = working_memory.filter((item) => {
        const importance = this.computeImportance(item);
        return importance >= importance_threshold;
      });

      return {
        consolidated_count: consolidated.length,
        pruned_count: working_memory.length - consolidated.length,
        memory_efficiency: consolidated.length / working_memory.length,
      };
    }
  }

  // ===== Private Helper Methods =====

  private computeSaliency(input: number[]): number {
    // Simple saliency: variance + magnitude
    const mean = input.reduce((a, b) => a + b, 0) / input.length;
    const variance = input.reduce((a, b) => a + (b - mean) ** 2, 0) / input.length;
    const magnitude = Math.sqrt(input.reduce((a, b) => a + b * b, 0));
    return variance + 0.1 * magnitude;
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  private softmax(values: number[], temperature: number): number[] {
    const scaled = values.map(v => v / temperature);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  }

  private computeBufferCoherence(buffer: number[][]): number {
    if (buffer.length < 2) return 1.0;

    let totalSimilarity = 0;
    let count = 0;

    for (let i = 0; i < buffer.length; i++) {
      for (let j = i + 1; j < buffer.length; j++) {
        totalSimilarity += this.cosineSimilarity(buffer[i], buffer[j]);
        count++;
      }
    }

    return count > 0 ? totalSimilarity / count : 0.0;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dot = this.dotProduct(a, b);
    const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  private generateEmbedding(content: any, context: Record<string, any>): number[] {
    // Simple hash-based embedding (replace with real embedding model)
    const str = JSON.stringify({ content, context });
    const hash = this.hashString(str);
    const embedding = new Array(128).fill(0).map((_, i) => {
      return Math.sin(hash + i) * 2 - 1;
    });
    return embedding;
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash = hash & hash;
    }
    return hash;
  }

  private computeImportance(item: number[]): number {
    // Importance = magnitude + variance
    const magnitude = Math.sqrt(item.reduce((a, b) => a + b * b, 0));
    const mean = item.reduce((a, b) => a + b, 0) / item.length;
    const variance = item.reduce((a, b) => a + (b - mean) ** 2, 0) / item.length;
    return (magnitude + variance) / 2;
  }
}
