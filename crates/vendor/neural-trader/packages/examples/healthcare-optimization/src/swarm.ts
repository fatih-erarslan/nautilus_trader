/**
 * Swarm Coordinator
 *
 * Swarm intelligence for exploring scheduling heuristics and optimization strategies.
 * Uses agentic-flow for multi-agent coordination.
 */

import { AgenticFlow } from 'agentic-flow';
import { AgentDB } from 'agentdb';
import type {
  SwarmAgent,
  SwarmConfig,
  SwarmResult,
  ScheduleSolution,
  ScheduleConstraints,
  ForecastResult,
  OptimizationObjective
} from './types.js';
import { Scheduler, type SchedulerConfig } from './scheduler.js';

export class SwarmCoordinator {
  private flow: AgenticFlow;
  private memory: AgentDB;
  private config: SwarmConfig;
  private agents: SwarmAgent[];
  private bestSolution: ScheduleSolution | null;

  constructor(
    config: SwarmConfig,
    agentdbPath: string
  ) {
    this.config = config;
    this.agents = [];
    this.bestSolution = null;

    // Initialize agentic-flow
    this.flow = new AgenticFlow({
      maxConcurrency: config.populationSize,
      coordinationStrategy: 'swarm',
      enableLearning: true
    });

    // Initialize AgentDB for swarm memory
    this.memory = new AgentDB({
      dbPath: agentdbPath,
      enableHNSW: true,
      hnswM: 16,
      hnswEfConstruction: 200
    });
  }

  /**
   * Optimize schedule using swarm intelligence
   */
  async optimize(
    forecasts: ForecastResult[],
    constraints: ScheduleConstraints,
    objective: OptimizationObjective,
    schedulerConfig: SchedulerConfig,
    startDate: Date
  ): Promise<SwarmResult> {
    console.log('🐝 Initializing swarm optimization...');

    // Initialize swarm population
    await this.initializePopulation(schedulerConfig);

    const convergenceHistory: number[] = [];
    let iteration = 0;
    let bestFitness = -Infinity;
    let stableIterations = 0;

    while (iteration < this.config.maxIterations && stableIterations < 10) {
      console.log(`\n📊 Iteration ${iteration + 1}/${this.config.maxIterations}`);

      // Evaluate all agents in parallel
      await this.evaluatePopulation(
        forecasts,
        constraints,
        objective,
        schedulerConfig,
        startDate
      );

      // Update best solution
      const currentBest = this.agents.reduce((best, agent) =>
        agent.fitness > best.fitness ? agent : best
      );

      if (currentBest.fitness > bestFitness) {
        bestFitness = currentBest.fitness;
        this.bestSolution = currentBest.currentSolution;
        stableIterations = 0;
        console.log(`✨ New best fitness: ${bestFitness.toFixed(4)}`);
      } else {
        stableIterations++;
      }

      convergenceHistory.push(bestFitness);

      // Check convergence
      if (this.hasConverged(convergenceHistory)) {
        console.log('✅ Swarm converged');
        break;
      }

      // Evolve population
      await this.evolvePopulation(objective);

      iteration++;
    }

    // Store best solution in memory
    await this.storeBestSolution();

    return {
      bestSolution: this.bestSolution!,
      convergenceHistory,
      iterations: iteration,
      exploredSolutions: this.agents.length * iteration
    };
  }

  /**
   * Initialize swarm population with diverse strategies
   */
  private async initializePopulation(schedulerConfig: SchedulerConfig): Promise<void> {
    this.agents = [];

    const strategies = [
      'greedy_cost',
      'greedy_coverage',
      'balanced',
      'fair_distribution',
      'preference_optimized'
    ];

    for (let i = 0; i < this.config.populationSize; i++) {
      const role = i < this.config.populationSize * this.config.explorationRate
        ? 'explorer'
        : (i < this.config.populationSize * 0.7 ? 'exploiter' : 'evaluator');

      const strategy = strategies[i % strategies.length];

      const agent: SwarmAgent = {
        id: `agent-${i}`,
        role,
        currentSolution: {
          shifts: [],
          totalCost: Infinity,
          coverageScore: 0,
          fairnessScore: 0,
          constraintViolations: []
        },
        fitness: 0,
        explorationRadius: role === 'explorer' ? 0.5 : 0.1
      };

      this.agents.push(agent);

      // Store agent strategy in memory
      await this.memory.store(`agent:${agent.id}`, {
        strategy,
        role,
        initialized: new Date().toISOString()
      });
    }

    console.log(`🐝 Initialized ${this.agents.length} agents`);
  }

  /**
   * Evaluate all agents in parallel
   */
  private async evaluatePopulation(
    forecasts: ForecastResult[],
    constraints: ScheduleConstraints,
    objective: OptimizationObjective,
    schedulerConfig: SchedulerConfig,
    startDate: Date
  ): Promise<void> {
    // Create evaluation tasks for each agent
    const evaluationTasks = this.agents.map(agent => async () => {
      // Get agent's strategy
      const agentData = await this.memory.retrieve(`agent:${agent.id}`);
      const strategy = agentData?.strategy || 'balanced';

      // Create modified scheduler config based on strategy
      const modifiedScheduler = this.applyStrategy(schedulerConfig, strategy, agent.explorationRadius);
      const scheduler = new Scheduler(modifiedScheduler);

      // Add staff (in production, this would come from database)
      const staff = this.generateSyntheticStaff();
      staff.forEach(s => scheduler.addStaff(s));

      // Generate schedule
      const solution = await scheduler.generateSchedule(forecasts, constraints, startDate);

      // Calculate fitness
      const fitness = this.calculateFitness(solution, objective);

      // Update agent
      agent.currentSolution = solution;
      agent.fitness = fitness;

      return { agentId: agent.id, fitness };
    });

    // Execute in parallel using agentic-flow
    await this.flow.parallel(evaluationTasks);

    console.log(`✅ Evaluated ${this.agents.length} solutions`);
  }

  /**
   * Apply strategy to scheduler config
   */
  private applyStrategy(
    baseConfig: SchedulerConfig,
    strategy: string,
    explorationRadius: number
  ): SchedulerConfig {
    const config = { ...baseConfig };

    // Add random perturbation for exploration
    const perturbation = (Math.random() - 0.5) * 2 * explorationRadius;

    switch (strategy) {
      case 'greedy_cost':
        config.costPerConstraintViolation = config.costPerConstraintViolation * (1 + perturbation * 0.5);
        break;
      case 'greedy_coverage':
        config.costPerConstraintViolation = config.costPerConstraintViolation * (1 - perturbation * 0.5);
        break;
      case 'fair_distribution':
        config.planningHorizonDays = Math.max(7, config.planningHorizonDays);
        break;
      case 'preference_optimized':
        // Strategies handled in fitness calculation
        break;
      default: // balanced
        break;
    }

    return config;
  }

  /**
   * Calculate fitness of solution based on objectives
   */
  private calculateFitness(
    solution: ScheduleSolution,
    objective: OptimizationObjective
  ): number {
    // Normalize cost (inverse, lower is better)
    const maxCost = 100000; // assumed max
    const costScore = 1 - Math.min(solution.totalCost / maxCost, 1);

    // Coverage score (higher is better)
    const coverageScore = solution.coverageScore;

    // Fairness score (higher is better)
    const fairnessScore = solution.fairnessScore;

    // Constraint penalty
    const constraintPenalty = solution.constraintViolations.length * 0.1;

    // Weighted combination
    const fitness =
      objective.minimizeCost * costScore +
      objective.maximizeUtilization * coverageScore +
      objective.maximizePatientOutcomes * fairnessScore -
      constraintPenalty;

    return Math.max(0, fitness);
  }

  /**
   * Evolve population using swarm intelligence
   */
  private async evolvePopulation(objective: OptimizationObjective): Promise<void> {
    // Sort agents by fitness
    this.agents.sort((a, b) => b.fitness - a.fitness);

    // Keep elite agents
    const eliteCount = Math.floor(this.config.populationSize * this.config.elitismRate);
    const eliteAgents = this.agents.slice(0, eliteCount);

    // Replace weak agents with variations of elite
    for (let i = eliteCount; i < this.agents.length; i++) {
      const agent = this.agents[i];

      // Select random elite to learn from
      const elite = eliteAgents[Math.floor(Math.random() * eliteAgents.length)];

      // Update exploration radius based on fitness
      if (agent.fitness < elite.fitness * 0.5) {
        agent.explorationRadius = Math.min(0.8, agent.explorationRadius * 1.2);
      } else {
        agent.explorationRadius = Math.max(0.05, agent.explorationRadius * 0.9);
      }

      // Copy elite's strategy with perturbation
      const eliteData = await this.memory.retrieve(`agent:${elite.id}`);
      await this.memory.store(`agent:${agent.id}`, {
        ...eliteData,
        explorationRadius: agent.explorationRadius,
        parentId: elite.id
      });
    }
  }

  /**
   * Check if swarm has converged
   */
  private hasConverged(history: number[]): boolean {
    if (history.length < 5) {
      return false;
    }

    const recent = history.slice(-5);
    const improvement = recent[recent.length - 1] - recent[0];

    return improvement < this.config.convergenceThreshold;
  }

  /**
   * Store best solution in memory
   */
  private async storeBestSolution(): Promise<void> {
    if (!this.bestSolution) {
      return;
    }

    await this.memory.store('best_solution', {
      solution: this.bestSolution,
      timestamp: new Date().toISOString(),
      fitness: this.agents[0].fitness
    });

    console.log('💾 Stored best solution in memory');
  }

  /**
   * Generate synthetic staff for testing
   */
  private generateSyntheticStaff() {
    const staff = [];
    const roles = ['physician', 'nurse', 'technician', 'specialist'] as const;

    for (let i = 0; i < 50; i++) {
      const role = roles[i % roles.length];
      staff.push({
        id: `staff-${i}`,
        name: `Staff ${i}`,
        role,
        skills: this.getSkillsForRole(role),
        shiftPreference: ['day', 'evening', 'night', 'any'][Math.floor(Math.random() * 4)] as any,
        maxHoursPerWeek: 40 + Math.floor(Math.random() * 20),
        costPerHour: this.getCostForRole(role)
      });
    }

    return staff;
  }

  /**
   * Get skills for role
   */
  private getSkillsForRole(role: string): string[] {
    const skillMap: Record<string, string[]> = {
      physician: ['diagnosis', 'emergency_care', 'procedures'],
      nurse: ['patient_care', 'medication', 'monitoring'],
      technician: ['lab', 'imaging', 'equipment'],
      specialist: ['surgery', 'cardiology', 'neurology']
    };

    return skillMap[role] || [];
  }

  /**
   * Get cost per hour for role
   */
  private getCostForRole(role: string): number {
    const costMap: Record<string, number> = {
      physician: 150,
      nurse: 50,
      technician: 35,
      specialist: 200
    };

    return costMap[role] || 40;
  }

  /**
   * Get best solution
   */
  getBestSolution(): ScheduleSolution | null {
    return this.bestSolution;
  }

  /**
   * Get swarm statistics
   */
  getStatistics() {
    const fitnesses = this.agents.map(a => a.fitness);

    return {
      population: this.agents.length,
      bestFitness: Math.max(...fitnesses),
      avgFitness: fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length,
      worstFitness: Math.min(...fitnesses),
      diversity: this.calculateDiversity()
    };
  }

  /**
   * Calculate population diversity
   */
  private calculateDiversity(): number {
    const fitnesses = this.agents.map(a => a.fitness);
    const mean = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    const variance = fitnesses.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / fitnesses.length;

    return Math.sqrt(variance);
  }
}
