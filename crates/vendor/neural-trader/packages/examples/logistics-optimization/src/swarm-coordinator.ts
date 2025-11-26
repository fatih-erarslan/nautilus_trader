/**
 * Multi-agent swarm coordinator for parallel route optimization
 * Uses agentic-flow for swarm orchestration
 */

import { SwarmConfig, Solution, OptimizationConfig, Customer, Vehicle } from './types';
import { VRPRouter } from './router';
import OpenAI from 'openai';

export interface SwarmAgent {
  id: string;
  algorithm: 'genetic' | 'simulated-annealing' | 'ant-colony';
  bestSolution: Solution | null;
  iterations: number;
  status: 'idle' | 'working' | 'completed';
}

export interface SwarmMessage {
  from: string;
  to: string | 'broadcast';
  type: 'solution-share' | 'diversity-request' | 'convergence-signal';
  payload: any;
  timestamp: number;
}

export class SwarmCoordinator {
  private config: SwarmConfig;
  private agents: Map<string, SwarmAgent>;
  private router: VRPRouter;
  private messageQueue: SwarmMessage[];
  private globalBestSolution: Solution | null;
  private openai: OpenAI;
  private iterationCount: number;

  constructor(
    config: SwarmConfig,
    customers: Customer[],
    vehicles: Vehicle[],
    openRouterApiKey?: string
  ) {
    this.config = config;
    this.agents = new Map();
    this.router = new VRPRouter(customers, vehicles);
    this.messageQueue = [];
    this.globalBestSolution = null;
    this.iterationCount = 0;

    // Initialize OpenRouter client for constraint reasoning
    this.openai = new OpenAI({
      apiKey: openRouterApiKey || process.env.OPENROUTER_API_KEY,
      baseURL: 'https://openrouter.ai/api/v1'
    });

    this.initializeAgents();
  }

  /**
   * Initialize swarm agents with different algorithms
   */
  private initializeAgents(): void {
    const algorithms: Array<'genetic' | 'simulated-annealing' | 'ant-colony'> = [
      'genetic',
      'simulated-annealing',
      'ant-colony'
    ];

    for (let i = 0; i < this.config.numAgents; i++) {
      const algorithm = algorithms[i % algorithms.length];
      const agent: SwarmAgent = {
        id: `agent-${i}`,
        algorithm,
        bestSolution: null,
        iterations: 0,
        status: 'idle'
      };
      this.agents.set(agent.id, agent);
    }
  }

  /**
   * Run swarm optimization in parallel
   */
  async optimize(): Promise<Solution> {
    console.log(`Starting swarm optimization with ${this.agents.size} agents...`);

    const optimizationPromises: Promise<void>[] = [];

    // Launch all agents in parallel
    for (const [agentId, agent] of this.agents.entries()) {
      optimizationPromises.push(this.runAgent(agentId, agent));
    }

    // Wait for all agents to complete or convergence
    await Promise.race([
      Promise.all(optimizationPromises),
      this.waitForConvergence()
    ]);

    console.log(`Swarm optimization completed after ${this.iterationCount} iterations`);

    return this.globalBestSolution || this.createEmptySolution();
  }

  /**
   * Run individual agent optimization
   */
  private async runAgent(agentId: string, agent: SwarmAgent): Promise<void> {
    agent.status = 'working';

    const config: OptimizationConfig = {
      algorithm: agent.algorithm,
      maxIterations: Math.floor(
        this.config.convergenceCriteria.maxIterations / this.config.numAgents
      ),
      populationSize: 30,
      mutationRate: 0.1,
      crossoverRate: 0.8,
      temperature: 1000,
      coolingRate: 0.995,
      pheromoneEvaporation: 0.1
    };

    while (agent.iterations < config.maxIterations && !this.hasConverged()) {
      let solution: Solution;

      switch (agent.algorithm) {
        case 'genetic':
          solution = await this.router.solveGenetic(config);
          break;
        case 'simulated-annealing':
          solution = await this.router.solveSimulatedAnnealing(config);
          break;
        case 'ant-colony':
          solution = await this.router.solveAntColony(config);
          break;
      }

      solution.metadata.agentId = agentId;
      agent.bestSolution = solution;
      agent.iterations++;
      this.iterationCount++;

      // Update global best
      if (!this.globalBestSolution || solution.fitness < this.globalBestSolution.fitness) {
        this.globalBestSolution = solution;
        console.log(
          `Agent ${agentId} found new best solution: fitness=${solution.fitness.toFixed(2)}`
        );

        // Broadcast to other agents
        if (this.config.communicationStrategy === 'broadcast') {
          this.broadcastSolution(agentId, solution);
        }
      }

      // Handle communication
      await this.processCommunication();

      // Small delay to prevent CPU thrashing
      await this.sleep(10);
    }

    agent.status = 'completed';
  }

  /**
   * Process inter-agent communication
   */
  private async processCommunication(): Promise<void> {
    if (this.config.communicationStrategy === 'best-solution') {
      // Share global best with all agents periodically
      if (this.iterationCount % 10 === 0 && this.globalBestSolution) {
        for (const [agentId, agent] of this.agents.entries()) {
          if (
            agent.bestSolution &&
            this.globalBestSolution.fitness < agent.bestSolution.fitness
          ) {
            // Agent learns from global best
            agent.bestSolution = { ...this.globalBestSolution };
          }
        }
      }
    } else if (this.config.communicationStrategy === 'diversity') {
      // Maintain diversity by sharing different solutions
      if (this.iterationCount % 20 === 0) {
        await this.maintainDiversity();
      }
    }
  }

  /**
   * Broadcast solution to all agents
   */
  private broadcastSolution(fromAgentId: string, solution: Solution): void {
    const message: SwarmMessage = {
      from: fromAgentId,
      to: 'broadcast',
      type: 'solution-share',
      payload: solution,
      timestamp: Date.now()
    };
    this.messageQueue.push(message);
  }

  /**
   * Maintain solution diversity across agents
   */
  private async maintainDiversity(): Promise<void> {
    const solutions = Array.from(this.agents.values())
      .filter(a => a.bestSolution)
      .map(a => a.bestSolution!);

    if (solutions.length < 2) return;

    // Calculate diversity metric (simplified)
    const diversityScore = this.calculateDiversityScore(solutions);

    if (diversityScore < 0.3) {
      // Low diversity - inject randomness into some agents
      let resetCount = 0;
      for (const [agentId, agent] of this.agents.entries()) {
        if (resetCount >= this.config.numAgents / 3) break;
        if (
          agent.bestSolution &&
          agent.bestSolution.fitness > this.globalBestSolution!.fitness * 1.2
        ) {
          console.log(`Resetting agent ${agentId} to maintain diversity`);
          agent.bestSolution = null;
          resetCount++;
        }
      }
    }
  }

  /**
   * Calculate diversity score for solutions
   */
  private calculateDiversityScore(solutions: Solution[]): number {
    if (solutions.length < 2) return 1.0;

    let totalDifference = 0;
    let comparisons = 0;

    for (let i = 0; i < solutions.length; i++) {
      for (let j = i + 1; j < solutions.length; j++) {
        const diff = Math.abs(solutions[i].fitness - solutions[j].fitness);
        totalDifference += diff;
        comparisons++;
      }
    }

    const avgDifference = totalDifference / comparisons;
    const avgFitness = solutions.reduce((sum, s) => sum + s.fitness, 0) / solutions.length;

    return avgDifference / (avgFitness + 1);
  }

  /**
   * Check if swarm has converged
   */
  private hasConverged(): boolean {
    if (this.iterationCount >= this.config.convergenceCriteria.maxIterations) {
      return true;
    }

    if (
      this.config.convergenceCriteria.fitnessThreshold &&
      this.globalBestSolution &&
      this.globalBestSolution.fitness <= this.config.convergenceCriteria.fitnessThreshold
    ) {
      return true;
    }

    // Check for no improvement
    if (this.config.convergenceCriteria.noImprovementSteps) {
      // Implementation would track improvement history
      // Simplified for now
    }

    return false;
  }

  /**
   * Wait for convergence criteria
   */
  private async waitForConvergence(): Promise<void> {
    while (!this.hasConverged()) {
      await this.sleep(100);
    }
  }

  /**
   * Use OpenRouter LLM for constraint reasoning
   */
  async reasonAboutConstraints(solution: Solution): Promise<string> {
    try {
      const prompt = `Analyze this vehicle routing solution and identify constraint violations or optimization opportunities:

Total Routes: ${solution.routes.length}
Total Cost: $${solution.totalCost.toFixed(2)}
Total Distance: ${solution.totalDistance.toFixed(2)} km
Unassigned Customers: ${solution.unassignedCustomers.length}

Routes:
${solution.routes.map((r, i) => `
Route ${i + 1} (Vehicle ${r.vehicleId}):
- Customers: ${r.customers.length}
- Distance: ${r.totalDistance.toFixed(2)} km
- Time Window Violations: ${r.timeWindowViolations}
- Capacity Violations: ${r.capacityViolations}
`).join('\n')}

Provide specific recommendations for improvement.`;

      const response = await this.openai.chat.completions.create({
        model: 'openai/gpt-4-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are a logistics optimization expert analyzing vehicle routing solutions.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 500
      });

      return response.choices[0]?.message?.content || 'No recommendations available.';
    } catch (error) {
      console.error('Error calling OpenRouter:', error);
      return 'Unable to generate recommendations at this time.';
    }
  }

  /**
   * Get swarm status and metrics
   */
  getStatus(): {
    iteration: number;
    agentsWorking: number;
    agentsCompleted: number;
    globalBestFitness: number | null;
    convergence: number;
  } {
    const agentsWorking = Array.from(this.agents.values()).filter(
      a => a.status === 'working'
    ).length;
    const agentsCompleted = Array.from(this.agents.values()).filter(
      a => a.status === 'completed'
    ).length;

    return {
      iteration: this.iterationCount,
      agentsWorking,
      agentsCompleted,
      globalBestFitness: this.globalBestSolution?.fitness || null,
      convergence: this.iterationCount / this.config.convergenceCriteria.maxIterations
    };
  }

  /**
   * Get agent details
   */
  getAgents(): SwarmAgent[] {
    return Array.from(this.agents.values());
  }

  private createEmptySolution(): Solution {
    return {
      routes: [],
      totalCost: 0,
      totalDistance: 0,
      unassignedCustomers: [],
      fitness: Infinity,
      metadata: { algorithm: 'none', iterations: 0, computeTime: 0 }
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
