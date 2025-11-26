/**
 * Emergence Detection and Quantification
 *
 * Analyzes multi-agent systems to detect and measure emergent behavior:
 * - Self-organization metrics
 * - Complexity measures
 * - Pattern formation detection
 * - Phase transitions
 * - Collective intelligence indicators
 *
 * Uses OpenRouter for pattern analysis and AgentDB for memory
 */

import { AgentDB } from 'agentdb';
import OpenAI from 'openai';
import { z } from 'zod';

export interface SystemState {
  timestamp: number;
  agents: AgentState[];
  globalMetrics: {
    entropy: number;
    order: number;
    complexity: number;
    connectivity: number;
  };
}

export interface AgentState {
  id: string;
  position: { x: number; y: number };
  velocity?: { x: number; y: number };
  state: any;
  neighbors: string[];
}

export interface EmergenceMetrics {
  selfOrganization: number; // 0-1, measures degree of self-organization
  complexity: number; // 0-1, system complexity
  coherence: number; // 0-1, collective behavior coherence
  adaptability: number; // 0-1, system's ability to adapt
  robustness: number; // 0-1, resistance to perturbations
  novelty: number; // 0-1, novelty of patterns
}

export interface EmergenceEvent {
  timestamp: number;
  type: 'phase-transition' | 'pattern-formation' | 'synchronization' | 'bifurcation';
  description: string;
  metrics: EmergenceMetrics;
  confidence: number;
}

const EmergenceSchema = z.object({
  timestamp: z.number(),
  type: z.enum(['phase-transition', 'pattern-formation', 'synchronization', 'bifurcation']),
  description: z.string(),
  metrics: z.object({
    selfOrganization: z.number(),
    complexity: z.number(),
    coherence: z.number(),
    adaptability: z.number(),
    robustness: z.number(),
    novelty: z.number()
  }),
  confidence: z.number()
});

export class EmergenceDetector {
  private agentDB: AgentDB;
  private openai: OpenAI;
  private stateHistory: SystemState[] = [];
  private maxHistorySize: number;
  private emergenceEvents: EmergenceEvent[] = [];

  constructor(
    openaiApiKey?: string,
    maxHistorySize: number = 100
  ) {
    this.agentDB = new AgentDB({
      enableCache: true,
      enableMemory: true,
      memorySize: 10000
    });

    this.openai = new OpenAI({
      apiKey: openaiApiKey || process.env.OPENAI_API_KEY,
      baseURL: 'https://openrouter.ai/api/v1',
      defaultHeaders: {
        'HTTP-Referer': 'https://neural-trader.dev',
        'X-Title': 'Neural Trader Adaptive Systems'
      }
    });

    this.maxHistorySize = maxHistorySize;
  }

  /**
   * Add system state snapshot
   */
  async addState(state: SystemState): Promise<void> {
    this.stateHistory.push(state);

    // Maintain history size limit
    if (this.stateHistory.length > this.maxHistorySize) {
      this.stateHistory.shift();
    }

    // Store in AgentDB
    await this.storeState(state);

    // Analyze for emergence
    if (this.stateHistory.length >= 10) {
      await this.detectEmergence();
    }
  }

  /**
   * Detect emergent behavior from recent states
   */
  private async detectEmergence(): Promise<void> {
    const metrics = this.calculateEmergenceMetrics();

    // Check for significant changes indicating emergence
    const isSignificant = this.isSignificantEmergence(metrics);

    if (isSignificant) {
      // Use LLM to analyze and describe the emergence
      const event = await this.analyzeEmergenceWithLLM(metrics);

      if (event) {
        this.emergenceEvents.push(event);
        await this.storeEmergenceEvent(event);
      }
    }
  }

  /**
   * Calculate emergence metrics from state history
   */
  private calculateEmergenceMetrics(): EmergenceMetrics {
    if (this.stateHistory.length < 2) {
      return {
        selfOrganization: 0,
        complexity: 0,
        coherence: 0,
        adaptability: 0,
        robustness: 0,
        novelty: 0
      };
    }

    const recentStates = this.stateHistory.slice(-20);

    // Self-organization: measure of order increase over time
    const selfOrganization = this.measureSelfOrganization(recentStates);

    // Complexity: balance between order and randomness
    const complexity = this.measureComplexity(recentStates);

    // Coherence: collective behavior alignment
    const coherence = this.measureCoherence(recentStates);

    // Adaptability: response to changes
    const adaptability = this.measureAdaptability(recentStates);

    // Robustness: stability under perturbations
    const robustness = this.measureRobustness(recentStates);

    // Novelty: uniqueness of current patterns
    const novelty = this.measureNovelty(recentStates);

    return {
      selfOrganization,
      complexity,
      coherence,
      adaptability,
      robustness,
      novelty
    };
  }

  /**
   * Measure self-organization (order parameter increase)
   */
  private measureSelfOrganization(states: SystemState[]): number {
    if (states.length < 2) return 0;

    const firstOrder = states[0].globalMetrics.order;
    const lastOrder = states[states.length - 1].globalMetrics.order;

    // Positive change in order indicates self-organization
    const orderChange = (lastOrder - firstOrder) / Math.max(firstOrder, 0.01);

    // Also consider entropy decrease
    const firstEntropy = states[0].globalMetrics.entropy;
    const lastEntropy = states[states.length - 1].globalMetrics.entropy;
    const entropyChange = (firstEntropy - lastEntropy) / Math.max(firstEntropy, 0.01);

    // Combine both signals
    const score = (orderChange + entropyChange) / 2;

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Measure complexity (edge of chaos)
   */
  private measureComplexity(states: SystemState[]): number {
    const avgEntropy = states.reduce((sum, s) => sum + s.globalMetrics.entropy, 0) / states.length;
    const avgOrder = states.reduce((sum, s) => sum + s.globalMetrics.order, 0) / states.length;

    // Complexity is highest when there's balance between order and chaos
    // Use a measure that peaks around 0.5 for both entropy and order
    const entropyComplexity = 1 - Math.abs(avgEntropy - 0.5) * 2;
    const orderComplexity = 1 - Math.abs(avgOrder - 0.5) * 2;

    return (entropyComplexity + orderComplexity) / 2;
  }

  /**
   * Measure coherence (collective behavior)
   */
  private measureCoherence(states: SystemState[]): number {
    if (states.length === 0) return 0;

    const recentState = states[states.length - 1];
    const agents = recentState.agents;

    if (agents.length < 2) return 0;

    // Calculate velocity alignment (for boids-like systems)
    if (agents[0].velocity) {
      let alignmentSum = 0;
      let count = 0;

      for (let i = 0; i < agents.length; i++) {
        for (let j = i + 1; j < agents.length; j++) {
          const a = agents[i];
          const b = agents[j];

          if (a.velocity && b.velocity) {
            // Cosine similarity of velocities
            const dot = a.velocity.x * b.velocity.x + a.velocity.y * b.velocity.y;
            const magA = Math.sqrt(a.velocity.x ** 2 + a.velocity.y ** 2);
            const magB = Math.sqrt(b.velocity.x ** 2 + b.velocity.y ** 2);

            if (magA > 0 && magB > 0) {
              alignmentSum += dot / (magA * magB);
              count++;
            }
          }
        }
      }

      if (count > 0) {
        // Normalize from [-1, 1] to [0, 1]
        return (alignmentSum / count + 1) / 2;
      }
    }

    // Fallback: measure spatial clustering
    const avgConnectivity = states.reduce((sum, s) => sum + s.globalMetrics.connectivity, 0) / states.length;
    return avgConnectivity;
  }

  /**
   * Measure adaptability (response to changes)
   */
  private measureAdaptability(states: SystemState[]): number {
    if (states.length < 3) return 0;

    // Measure how quickly system responds to changes
    const metrics = states.map(s => s.globalMetrics);

    let responseSum = 0;
    let count = 0;

    for (let i = 1; i < metrics.length - 1; i++) {
      const prevChange = Math.abs(metrics[i].entropy - metrics[i - 1].entropy);
      const nextChange = Math.abs(metrics[i + 1].entropy - metrics[i].entropy);

      if (prevChange > 0.1) {
        // Significant change detected, measure response
        const response = nextChange / prevChange;
        responseSum += Math.min(response, 2); // Cap at 2
        count++;
      }
    }

    if (count === 0) return 0.5; // Neutral if no changes detected

    const avgResponse = responseSum / count;
    return Math.max(0, Math.min(1, avgResponse / 2));
  }

  /**
   * Measure robustness (stability)
   */
  private measureRobustness(states: SystemState[]): number {
    if (states.length < 2) return 1;

    // Measure variance in key metrics
    const entropies = states.map(s => s.globalMetrics.entropy);
    const orders = states.map(s => s.globalMetrics.order);

    const entropyVariance = this.variance(entropies);
    const orderVariance = this.variance(orders);

    // Low variance indicates high robustness
    const entropyRobustness = 1 - Math.min(entropyVariance, 1);
    const orderRobustness = 1 - Math.min(orderVariance, 1);

    return (entropyRobustness + orderRobustness) / 2;
  }

  /**
   * Measure novelty (uniqueness of patterns)
   */
  private measureNovelty(states: SystemState[]): number {
    if (states.length < 2) return 1;

    const recentState = states[states.length - 1];

    // Compare recent state to historical states
    let minSimilarity = 1;

    for (let i = 0; i < states.length - 1; i++) {
      const similarity = this.calculateStateSimilarity(recentState, states[i]);
      minSimilarity = Math.min(minSimilarity, similarity);
    }

    // High novelty = low similarity to past states
    return 1 - minSimilarity;
  }

  /**
   * Calculate similarity between two states
   */
  private calculateStateSimilarity(a: SystemState, b: SystemState): number {
    const metricSimilarity =
      1 - Math.abs(a.globalMetrics.entropy - b.globalMetrics.entropy) / 2 +
      1 - Math.abs(a.globalMetrics.order - b.globalMetrics.order) / 2 +
      1 - Math.abs(a.globalMetrics.complexity - b.globalMetrics.complexity) / 2;

    return metricSimilarity / 3;
  }

  /**
   * Calculate variance of an array
   */
  private variance(values: number[]): number {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const squaredDiffs = values.map(v => (v - mean) ** 2);
    return squaredDiffs.reduce((sum, v) => sum + v, 0) / values.length;
  }

  /**
   * Check if metrics indicate significant emergence
   */
  private isSignificantEmergence(metrics: EmergenceMetrics): boolean {
    // Significant if any metric is high or there's a strong overall signal
    const threshold = 0.7;

    return (
      metrics.selfOrganization > threshold ||
      metrics.novelty > threshold ||
      metrics.complexity > threshold ||
      (metrics.selfOrganization + metrics.coherence + metrics.complexity) / 3 > 0.6
    );
  }

  /**
   * Use LLM to analyze and describe emergence
   */
  private async analyzeEmergenceWithLLM(metrics: EmergenceMetrics): Promise<EmergenceEvent | null> {
    try {
      const prompt = this.buildAnalysisPrompt(metrics);

      const completion = await this.openai.chat.completions.create({
        model: 'anthropic/claude-3.5-sonnet',
        messages: [
          {
            role: 'system',
            content: 'You are an expert in complex systems, emergence theory, and multi-agent systems. Analyze emergence metrics and provide structured insights.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 500
      });

      const response = completion.choices[0]?.message?.content;
      if (!response) return null;

      return this.parseEmergenceResponse(response, metrics);
    } catch (error) {
      console.error('LLM analysis failed:', error);
      return null;
    }
  }

  /**
   * Build prompt for LLM analysis
   */
  private buildAnalysisPrompt(metrics: EmergenceMetrics): string {
    const recentStates = this.stateHistory.slice(-10);

    return `Analyze the following emergence metrics from a multi-agent system:

Self-Organization: ${metrics.selfOrganization.toFixed(3)} (0=none, 1=high)
Complexity: ${metrics.complexity.toFixed(3)} (0=simple, 1=complex)
Coherence: ${metrics.coherence.toFixed(3)} (0=chaotic, 1=aligned)
Adaptability: ${metrics.adaptability.toFixed(3)} (0=rigid, 1=flexible)
Robustness: ${metrics.robustness.toFixed(3)} (0=fragile, 1=stable)
Novelty: ${metrics.novelty.toFixed(3)} (0=repetitive, 1=novel)

Recent system states:
${recentStates.map((s, i) => `
State ${i}: entropy=${s.globalMetrics.entropy.toFixed(3)}, order=${s.globalMetrics.order.toFixed(3)}, agents=${s.agents.length}
`).join('\n')}

Classify this as one of: phase-transition, pattern-formation, synchronization, bifurcation

Respond in JSON format:
{
  "type": "phase-transition|pattern-formation|synchronization|bifurcation",
  "description": "Brief description of the emergent behavior",
  "confidence": 0.0-1.0
}`;
  }

  /**
   * Parse LLM response into emergence event
   */
  private parseEmergenceResponse(response: string, metrics: EmergenceMetrics): EmergenceEvent {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error('No JSON found in response');

      const parsed = JSON.parse(jsonMatch[0]);

      return {
        timestamp: Date.now(),
        type: parsed.type || 'pattern-formation',
        description: parsed.description || 'Emergent behavior detected',
        metrics,
        confidence: parsed.confidence || 0.5
      };
    } catch (error) {
      // Fallback if parsing fails
      return {
        timestamp: Date.now(),
        type: 'pattern-formation',
        description: response.slice(0, 200),
        metrics,
        confidence: 0.5
      };
    }
  }

  /**
   * Store state in AgentDB
   */
  private async storeState(state: SystemState): Promise<void> {
    const embedding = this.createStateEmbedding(state);

    await this.agentDB.store(
      `state:${state.timestamp}`,
      JSON.stringify(state),
      embedding
    );
  }

  /**
   * Store emergence event in AgentDB
   */
  private async storeEmergenceEvent(event: EmergenceEvent): Promise<void> {
    const embedding = [
      event.metrics.selfOrganization,
      event.metrics.complexity,
      event.metrics.coherence,
      event.metrics.adaptability,
      event.metrics.robustness,
      event.metrics.novelty
    ];

    await this.agentDB.store(
      `emergence:${event.timestamp}`,
      JSON.stringify(event),
      embedding
    );
  }

  /**
   * Create embedding from system state
   */
  private createStateEmbedding(state: SystemState): number[] {
    return [
      state.globalMetrics.entropy,
      state.globalMetrics.order,
      state.globalMetrics.complexity,
      state.globalMetrics.connectivity,
      state.agents.length / 100 // Normalize agent count
    ];
  }

  /**
   * Get emergence events
   */
  getEmergenceEvents(): EmergenceEvent[] {
    return [...this.emergenceEvents];
  }

  /**
   * Get state history
   */
  getStateHistory(): SystemState[] {
    return [...this.stateHistory];
  }

  /**
   * Get latest metrics
   */
  getLatestMetrics(): EmergenceMetrics {
    return this.calculateEmergenceMetrics();
  }

  /**
   * Clear history
   */
  clear(): void {
    this.stateHistory = [];
    this.emergenceEvents = [];
  }
}
