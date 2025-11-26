# Swarm Coordination Patterns

Comprehensive guide to multi-agent swarm coordination patterns.

## Table of Contents

- [Introduction](#introduction)
- [Core Concepts](#core-concepts)
- [Swarm Topologies](#swarm-topologies)
- [Agent Types](#agent-types)
- [Evolution Strategies](#evolution-strategies)
- [Consensus Mechanisms](#consensus-mechanisms)
- [Communication Patterns](#communication-patterns)
- [Advanced Patterns](#advanced-patterns)

---

## Introduction

Swarm intelligence leverages collective behavior of decentralized, self-organized agents to solve complex problems. Neural Trader examples use swarm coordination for:

- **Feature engineering**: Evolve optimal feature combinations
- **Strategy exploration**: Test multiple strategies in parallel
- **Anomaly detection**: Consensus-based outlier identification
- **Parameter optimization**: Navigate complex solution spaces
- **Ensemble learning**: Combine predictions from multiple agents

**Benefits**:
- **2.8-4.4x faster** than sequential approaches
- **32.3% token reduction** through intelligent coordination
- **84.8% SWE-Bench solve rate** for complex tasks
- Robust to local optima through diversity

---

## Core Concepts

### Swarm Architecture

```
┌─────────────────────────────────────┐
│      Swarm Coordinator              │
│  - Initialize population            │
│  - Orchestrate evolution            │
│  - Collect results                  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴──────┬──────────────┐
    │             │              │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐
│Explorer│   │Optimizer│   │Validator│
│ Agents │   │ Agents  │   │ Agents  │
└────────┘   └─────────┘   └─────────┘
    │             │              │
    └──────┬──────┴──────┬───────┘
           │             │
    ┌──────▼─────────────▼──────┐
    │    Shared Memory Pool      │
    │  (AgentDB or in-memory)    │
    └────────────────────────────┘
```

### Agent Lifecycle

```typescript
// 1. Initialization
const agents = Array.from({ length: populationSize }, () =>
  createRandomAgent()
);

// 2. Evaluation
await Promise.all(agents.map(agent => evaluateFitness(agent)));

// 3. Selection
const parents = tournamentSelection(agents, selectionSize);

// 4. Crossover
const offspring = parents.map((p1, i) =>
  crossover(p1, parents[(i + 1) % parents.length])
);

// 5. Mutation
offspring.forEach(agent => {
  if (Math.random() < mutationRate) {
    mutate(agent);
  }
});

// 6. Replacement
replaceWorstAgents(agents, offspring);

// 7. Repeat steps 2-6 for N generations
```

---

## Swarm Topologies

### 1. Hierarchical Swarm

Best for complex tasks with subtask delegation.

```typescript
export class HierarchicalSwarm {
  private queen: Agent;
  private workers: Agent[] = [];

  async initialize(populationSize: number): Promise<void> {
    // Queen agent coordinates
    this.queen = this.createQueenAgent();

    // Worker agents execute
    this.workers = Array.from({ length: populationSize }, () =>
      this.createWorkerAgent()
    );
  }

  async solve(problem: Problem): Promise<Solution> {
    // Queen decomposes problem
    const subtasks = await this.queen.decompose(problem);

    // Workers solve subtasks in parallel
    const solutions = await Promise.all(
      subtasks.map((task, i) =>
        this.workers[i % this.workers.length].solve(task)
      )
    );

    // Queen aggregates solutions
    return await this.queen.aggregate(solutions);
  }
}
```

### 2. Mesh Swarm

Fully connected agents with peer-to-peer communication.

```typescript
export class MeshSwarm {
  private agents: Agent[] = [];

  async evolve(generations: number): Promise<void> {
    for (let gen = 0; gen < generations; gen++) {
      // Each agent communicates with all others
      for (let i = 0; i < this.agents.length; i++) {
        const agent = this.agents[i];
        const peers = this.agents.filter((_, j) => i !== j);

        // Share knowledge
        const sharedKnowledge = peers.map(p => p.getKnowledge());
        await agent.learn(sharedKnowledge);

        // Update based on peers
        await agent.update();
      }

      await this.evaluatePopulation();
    }
  }
}
```

### 3. Ring Swarm

Agents communicate with neighbors only.

```typescript
export class RingSwarm {
  private agents: Agent[] = [];

  async evolve(generations: number): Promise<void> {
    for (let gen = 0; gen < generations; gen++) {
      for (let i = 0; i < this.agents.length; i++) {
        const agent = this.agents[i];

        // Get neighbors (left and right in ring)
        const leftNeighbor = this.agents[(i - 1 + this.agents.length) % this.agents.length];
        const rightNeighbor = this.agents[(i + 1) % this.agents.length];

        // Learn from neighbors
        await agent.learn([
          leftNeighbor.getKnowledge(),
          rightNeighbor.getKnowledge()
        ]);

        await agent.update();
      }

      await this.evaluatePopulation();
    }
  }
}
```

### 4. Adaptive Swarm

Dynamically adjusts topology based on performance.

```typescript
export class AdaptiveSwarm {
  private topology: 'hierarchical' | 'mesh' | 'ring' = 'mesh';

  async evolve(generations: number): Promise<void> {
    for (let gen = 0; gen < generations; gen++) {
      // Evolve using current topology
      await this.evolveWithTopology(this.topology);

      // Evaluate performance
      const performance = this.evaluatePerformance();

      // Adjust topology if stagnating
      if (performance.stagnating) {
        this.topology = this.selectNewTopology(performance);
        console.log(`Switching to ${this.topology} topology at generation ${gen}`);
      }
    }
  }

  private selectNewTopology(performance: Performance): Topology {
    if (performance.needsExploration) return 'mesh';
    if (performance.needsExploitation) return 'ring';
    if (performance.needsDecomposition) return 'hierarchical';
    return 'mesh';
  }
}
```

---

## Agent Types

### 1. Explorer Agents

High mutation rate for discovering new solutions.

```typescript
export class ExplorerAgent {
  private mutationRate = 0.5;  // High mutation
  private explorationBonus = 0.2;

  async explore(searchSpace: SearchSpace): Promise<Solution> {
    // Random initialization
    let solution = this.randomSolution(searchSpace);

    // Aggressive mutation
    for (let i = 0; i < 10; i++) {
      const mutated = this.mutate(solution, this.mutationRate);

      if (this.fitness(mutated) > this.fitness(solution)) {
        solution = mutated;
      }

      // Bonus for novelty
      const noveltyScore = this.measureNovelty(mutated);
      if (noveltyScore > 0.8) {
        solution = mutated;
      }
    }

    return solution;
  }

  private mutate(solution: Solution, rate: number): Solution {
    return solution.map(gene =>
      Math.random() < rate
        ? gene + (Math.random() - 0.5) * 2  // Large perturbation
        : gene
    );
  }
}
```

### 2. Optimizer Agents

Gradient-based refinement of existing solutions.

```typescript
export class OptimizerAgent {
  private learningRate = 0.01;
  private mutationRate = 0.1;  // Low mutation

  async optimize(solution: Solution): Promise<Solution> {
    let current = solution;
    let bestFitness = this.fitness(current);

    for (let iter = 0; iter < 100; iter++) {
      // Gradient approximation
      const gradient = this.estimateGradient(current);

      // Gradient ascent
      const updated = current.map((gene, i) =>
        gene + this.learningRate * gradient[i]
      );

      const newFitness = this.fitness(updated);

      if (newFitness > bestFitness) {
        current = updated;
        bestFitness = newFitness;
      } else {
        // Reduce learning rate
        this.learningRate *= 0.9;
      }

      // Small random perturbation
      if (Math.random() < this.mutationRate) {
        current = this.mutate(current, 0.01);
      }
    }

    return current;
  }

  private estimateGradient(solution: Solution): number[] {
    const epsilon = 0.001;
    const baseline = this.fitness(solution);

    return solution.map((gene, i) => {
      const perturbed = [...solution];
      perturbed[i] += epsilon;
      return (this.fitness(perturbed) - baseline) / epsilon;
    });
  }
}
```

### 3. Validator Agents

Cross-validation and robustness testing.

```typescript
export class ValidatorAgent {
  async validate(
    solution: Solution,
    testCases: TestCase[]
  ): Promise<ValidationResult> {
    let passedTests = 0;
    let totalTests = testCases.length;
    const failures: TestFailure[] = [];

    for (const testCase of testCases) {
      try {
        const result = await this.applySolution(solution, testCase);

        if (this.isValid(result, testCase.expected)) {
          passedTests++;
        } else {
          failures.push({
            testCase,
            result,
            expected: testCase.expected
          });
        }
      } catch (error) {
        failures.push({
          testCase,
          error: error.message
        });
      }
    }

    return {
      passRate: passedTests / totalTests,
      passedTests,
      totalTests,
      failures,
      isValid: passedTests === totalTests
    };
  }
}
```

### 4. Anomaly Detector Agents

Specialized for identifying outliers.

```typescript
export class AnomalyDetectorAgent {
  private threshold = 2.5;  // Standard deviations

  async detect(data: DataPoint[]): Promise<Anomaly[]> {
    // Calculate statistics
    const mean = this.calculateMean(data);
    const std = this.calculateStd(data, mean);

    // Find outliers
    const anomalies: Anomaly[] = [];

    for (let i = 0; i < data.length; i++) {
      const zScore = Math.abs((data[i].value - mean) / std);

      if (zScore > this.threshold) {
        anomalies.push({
          index: i,
          value: data[i].value,
          zScore,
          confidence: Math.min(zScore / 5, 1),
          type: data[i].value > mean ? 'high' : 'low'
        });
      }
    }

    return anomalies;
  }

  // Adaptive threshold based on data distribution
  adaptThreshold(data: DataPoint[]): void {
    const skewness = this.calculateSkewness(data);

    if (Math.abs(skewness) > 1) {
      // Heavy-tailed distribution - increase threshold
      this.threshold = 3.0;
    } else {
      // Normal distribution - standard threshold
      this.threshold = 2.5;
    }
  }
}
```

---

## Evolution Strategies

### 1. Genetic Algorithm

```typescript
export class GeneticSwarm {
  private populationSize = 50;
  private eliteSize = 5;
  private mutationRate = 0.2;
  private crossoverRate = 0.7;

  async evolve(generations: number): Promise<Agent> {
    let population = this.initializePopulation();

    for (let gen = 0; gen < generations; gen++) {
      // Evaluate fitness
      await Promise.all(
        population.map(agent => this.evaluateFitness(agent))
      );

      // Sort by fitness
      population.sort((a, b) => b.fitness - a.fitness);

      // Keep elite
      const elite = population.slice(0, this.eliteSize);

      // Select parents
      const parents = this.tournamentSelection(population);

      // Generate offspring
      const offspring: Agent[] = [];

      for (let i = 0; i < parents.length - 1; i += 2) {
        if (Math.random() < this.crossoverRate) {
          const [child1, child2] = this.crossover(parents[i], parents[i + 1]);
          offspring.push(child1, child2);
        } else {
          offspring.push(parents[i], parents[i + 1]);
        }
      }

      // Mutate offspring
      offspring.forEach(agent => {
        if (Math.random() < this.mutationRate) {
          this.mutate(agent);
        }
      });

      // New population = elite + offspring
      population = [...elite, ...offspring].slice(0, this.populationSize);

      // Report progress
      if (gen % 10 === 0) {
        console.log(`Generation ${gen}: Best fitness = ${elite[0].fitness}`);
      }
    }

    return population[0];  // Best agent
  }

  private crossover(parent1: Agent, parent2: Agent): [Agent, Agent] {
    const point = Math.floor(Math.random() * parent1.genome.length);

    const child1 = {
      genome: [
        ...parent1.genome.slice(0, point),
        ...parent2.genome.slice(point)
      ]
    };

    const child2 = {
      genome: [
        ...parent2.genome.slice(0, point),
        ...parent1.genome.slice(point)
      ]
    };

    return [child1, child2];
  }
}
```

### 2. Particle Swarm Optimization

```typescript
export class ParticleSwarm {
  private particles: Particle[] = [];
  private globalBest: Solution;
  private inertiaWeight = 0.7;
  private cognitiveWeight = 1.5;
  private socialWeight = 1.5;

  async optimize(dimensions: number, iterations: number): Promise<Solution> {
    // Initialize particles
    this.particles = Array.from({ length: 30 }, () => ({
      position: this.randomPosition(dimensions),
      velocity: this.randomVelocity(dimensions),
      bestPosition: null,
      bestFitness: -Infinity
    }));

    for (let iter = 0; iter < iterations; iter++) {
      for (const particle of this.particles) {
        // Evaluate fitness
        const fitness = this.evaluateFitness(particle.position);

        // Update personal best
        if (fitness > particle.bestFitness) {
          particle.bestPosition = [...particle.position];
          particle.bestFitness = fitness;
        }

        // Update global best
        if (fitness > this.globalBestFitness) {
          this.globalBest = [...particle.position];
          this.globalBestFitness = fitness;
        }

        // Update velocity and position
        for (let d = 0; d < dimensions; d++) {
          const r1 = Math.random();
          const r2 = Math.random();

          particle.velocity[d] =
            this.inertiaWeight * particle.velocity[d] +
            this.cognitiveWeight * r1 * (particle.bestPosition[d] - particle.position[d]) +
            this.socialWeight * r2 * (this.globalBest[d] - particle.position[d]);

          particle.position[d] += particle.velocity[d];

          // Clamp to bounds
          particle.position[d] = Math.max(-10, Math.min(10, particle.position[d]));
        }
      }

      // Decay inertia weight
      this.inertiaWeight *= 0.99;
    }

    return this.globalBest;
  }
}
```

### 3. Differential Evolution

```typescript
export class DifferentialEvolutionSwarm {
  private populationSize = 50;
  private F = 0.8;  // Mutation factor
  private CR = 0.9;  // Crossover rate

  async evolve(generations: number): Promise<Agent> {
    let population = this.initializePopulation();

    for (let gen = 0; gen < generations; gen++) {
      const newPopulation: Agent[] = [];

      for (let i = 0; i < population.length; i++) {
        // Select three random agents (different from i)
        const [a, b, c] = this.selectRandomAgents(population, i, 3);

        // Mutation: v = a + F * (b - c)
        const mutant = a.genome.map((gene, j) =>
          gene + this.F * (b.genome[j] - c.genome[j])
        );

        // Crossover
        const trial = population[i].genome.map((gene, j) =>
          Math.random() < this.CR ? mutant[j] : gene
        );

        // Selection
        const trialFitness = await this.evaluateFitness({ genome: trial });
        const targetFitness = await this.evaluateFitness(population[i]);

        if (trialFitness > targetFitness) {
          newPopulation.push({ genome: trial, fitness: trialFitness });
        } else {
          newPopulation.push(population[i]);
        }
      }

      population = newPopulation;
    }

    return population.reduce((best, agent) =>
      agent.fitness > best.fitness ? agent : best
    );
  }
}
```

---

## Consensus Mechanisms

### 1. Voting Consensus

```typescript
export class VotingConsensus {
  async decide(agents: Agent[], data: any): Promise<Decision> {
    // Each agent votes
    const votes = await Promise.all(
      agents.map(agent => agent.makeDecision(data))
    );

    // Count votes
    const voteCounts = new Map<string, number>();

    votes.forEach(vote => {
      voteCounts.set(vote, (voteCounts.get(vote) || 0) + 1);
    });

    // Majority wins
    const sortedVotes = Array.from(voteCounts.entries())
      .sort((a, b) => b[1] - a[1]);

    const [decision, count] = sortedVotes[0];

    return {
      decision,
      confidence: count / agents.length,
      votesFor: count,
      votesAgainst: agents.length - count,
      totalAgents: agents.length
    };
  }
}
```

### 2. Weighted Consensus

```typescript
export class WeightedConsensus {
  async decide(agents: Agent[], data: any): Promise<Decision> {
    // Each agent votes with weight
    const weightedVotes = await Promise.all(
      agents.map(async agent => ({
        decision: await agent.makeDecision(data),
        weight: agent.fitness  // Weight by fitness
      }))
    );

    // Sum weights by decision
    const weightSums = new Map<string, number>();

    weightedVotes.forEach(({ decision, weight }) => {
      weightSums.set(decision, (weightSums.get(decision) || 0) + weight);
    });

    // Highest weighted decision wins
    const sortedDecisions = Array.from(weightSums.entries())
      .sort((a, b) => b[1] - a[1]);

    const [decision, totalWeight] = sortedDecisions[0];
    const totalPossibleWeight = agents.reduce((sum, a) => sum + a.fitness, 0);

    return {
      decision,
      confidence: totalWeight / totalPossibleWeight,
      weightedScore: totalWeight
    };
  }
}
```

### 3. Ensemble Consensus

```typescript
export class EnsembleConsensus {
  async decide(agents: Agent[], data: any): Promise<Decision> {
    // Get predictions from all agents
    const predictions = await Promise.all(
      agents.map(agent => agent.predict(data))
    );

    // Average predictions
    const avgPrediction = predictions.reduce((sum, p) => sum + p, 0) / predictions.length;

    // Calculate confidence based on agreement
    const variance = predictions.reduce((sum, p) =>
      sum + Math.pow(p - avgPrediction, 2), 0
    ) / predictions.length;

    const agreement = 1 / (1 + variance);

    return {
      prediction: avgPrediction,
      confidence: agreement,
      predictions,
      variance
    };
  }
}
```

---

## Communication Patterns

### 1. Broadcast

```typescript
export class BroadcastCommunication {
  async broadcast(sender: Agent, message: Message, recipients: Agent[]): Promise<void> {
    await Promise.all(
      recipients.map(agent => agent.receive(message))
    );
  }
}
```

### 2. Selective Sharing

```typescript
export class SelectiveSharingCommunication {
  async shareWithSimilar(
    sender: Agent,
    message: Message,
    allAgents: Agent[],
    similarityThreshold: number = 0.7
  ): Promise<void> {
    const recipients = allAgents.filter(agent => {
      if (agent === sender) return false;

      const similarity = this.calculateSimilarity(sender, agent);
      return similarity >= similarityThreshold;
    });

    await Promise.all(
      recipients.map(agent => agent.receive(message))
    );
  }

  private calculateSimilarity(agent1: Agent, agent2: Agent): number {
    // Cosine similarity of agent genomes
    const dot = agent1.genome.reduce((sum, val, i) =>
      sum + val * agent2.genome[i], 0
    );

    const mag1 = Math.sqrt(agent1.genome.reduce((sum, val) => sum + val * val, 0));
    const mag2 = Math.sqrt(agent2.genome.reduce((sum, val) => sum + val * val, 0));

    return dot / (mag1 * mag2);
  }
}
```

### 3. Stigmergy

Indirect communication through environment.

```typescript
export class StigmergyCommunication {
  private pheromones = new Map<string, number>();
  private evaporationRate = 0.1;

  depositPheromone(location: string, intensity: number): void {
    const current = this.pheromones.get(location) || 0;
    this.pheromones.set(location, current + intensity);
  }

  readPheromone(location: string): number {
    return this.pheromones.get(location) || 0;
  }

  evaporate(): void {
    for (const [location, intensity] of this.pheromones.entries()) {
      const newIntensity = intensity * (1 - this.evaporationRate);

      if (newIntensity < 0.01) {
        this.pheromones.delete(location);
      } else {
        this.pheromones.set(location, newIntensity);
      }
    }
  }
}
```

---

## Advanced Patterns

### 1. Coevolution

```typescript
export class CoevolutionarySwarm {
  private predators: Agent[] = [];
  private prey: Agent[] = [];

  async evolve(generations: number): Promise<void> {
    for (let gen = 0; gen < generations; gen++) {
      // Predators evolve to catch prey
      await this.evolvePredators();

      // Prey evolve to evade predators
      await this.evolvePrey();

      // Evaluate both populations
      await this.evaluateFitness();
    }
  }

  private async evolvePredators(): Promise<void> {
    // Fitness = success rate at catching prey
    for (const predator of this.predators) {
      let catches = 0;

      for (const prey of this.prey) {
        if (predator.canCatch(prey)) {
          catches++;
        }
      }

      predator.fitness = catches / this.prey.length;
    }

    // Standard genetic algorithm
    this.predators = this.geneticStep(this.predators);
  }

  private async evolvePrey(): Promise<void> {
    // Fitness = success rate at evading predators
    for (const prey of this.prey) {
      let escapes = 0;

      for (const predator of this.predators) {
        if (!predator.canCatch(prey)) {
          escapes++;
        }
      }

      prey.fitness = escapes / this.predators.length;
    }

    this.prey = this.geneticStep(this.prey);
  }
}
```

### 2. Island Model

```typescript
export class IslandModelSwarm {
  private islands: Agent[][] = [];
  private migrationInterval = 10;
  private migrationSize = 2;

  async evolve(generations: number): Promise<void> {
    for (let gen = 0; gen < generations; gen++) {
      // Evolve each island independently
      await Promise.all(
        this.islands.map(island => this.evolveIsland(island))
      );

      // Periodic migration
      if (gen % this.migrationInterval === 0) {
        this.migrate();
      }
    }
  }

  private migrate(): void {
    for (let i = 0; i < this.islands.length; i++) {
      const sourceIsland = this.islands[i];
      const targetIsland = this.islands[(i + 1) % this.islands.length];

      // Sort by fitness
      sourceIsland.sort((a, b) => b.fitness - a.fitness);
      targetIsland.sort((a, b) => b.fitness - a.fitness);

      // Move best agents to next island
      const migrants = sourceIsland.slice(0, this.migrationSize);
      const replaced = targetIsland.slice(-this.migrationSize);

      targetIsland.splice(-this.migrationSize, this.migrationSize, ...migrants);
      sourceIsland.splice(0, this.migrationSize, ...replaced);
    }
  }
}
```

---

## Complete Example

```typescript
export class AdaptiveFeatureEngineeringSwarm {
  private explorers: ExplorerAgent[] = [];
  private optimizers: OptimizerAgent[] = [];
  private validators: ValidatorAgent[] = [];

  async discoverFeatures(
    data: DataPoint[],
    targetMetric: string
  ): Promise<FeatureSet[]> {
    // Phase 1: Exploration (generations 1-20)
    console.log('Phase 1: Exploration');
    const candidateFeatures = await this.exploreFeatureSpace(data);

    // Phase 2: Optimization (generations 21-40)
    console.log('Phase 2: Optimization');
    const optimizedFeatures = await this.optimizeFeatures(
      candidateFeatures,
      data,
      targetMetric
    );

    // Phase 3: Validation (cross-validation)
    console.log('Phase 3: Validation');
    const validatedFeatures = await this.validateFeatures(
      optimizedFeatures,
      data
    );

    // Consensus: Select best features
    return await this.selectBestFeatures(validatedFeatures);
  }

  private async exploreFeatureSpace(data: DataPoint[]): Promise<FeatureSet[]> {
    const features = await Promise.all(
      this.explorers.map(explorer => explorer.explore(data))
    );

    return features;
  }

  private async optimizeFeatures(
    features: FeatureSet[],
    data: DataPoint[],
    targetMetric: string
  ): Promise<FeatureSet[]> {
    return await Promise.all(
      this.optimizers.map((optimizer, i) =>
        optimizer.optimize(features[i % features.length], data, targetMetric)
      )
    );
  }

  private async validateFeatures(
    features: FeatureSet[],
    data: DataPoint[]
  ): Promise<FeatureSet[]> {
    const validationResults = await Promise.all(
      features.map(featureSet =>
        Promise.all(
          this.validators.map(validator =>
            validator.validate(featureSet, data)
          )
        )
      )
    );

    // Keep only features that pass validation
    return features.filter((featureSet, i) => {
      const passRate = validationResults[i].reduce((sum, result) =>
        sum + result.passRate, 0
      ) / this.validators.length;

      return passRate >= 0.8;  // 80% pass threshold
    });
  }

  private async selectBestFeatures(
    features: FeatureSet[]
  ): Promise<FeatureSet[]> {
    // Sort by fitness and diversity
    features.sort((a, b) => b.fitness - a.fitness);

    // Select diverse top features
    const selected: FeatureSet[] = [features[0]];

    for (const featureSet of features.slice(1)) {
      const minSimilarity = Math.min(
        ...selected.map(s => this.calculateSimilarity(s, featureSet))
      );

      // Add if sufficiently different
      if (minSimilarity < 0.7) {
        selected.push(featureSet);
      }

      if (selected.length >= 10) break;
    }

    return selected;
  }
}
```

---

## References

- [Architecture Overview](./ARCHITECTURE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [AgentDB Guide](./AGENTDB_GUIDE.md)
- [OpenRouter Config](./OPENROUTER_CONFIG.md)

---

Built with ❤️ by the Neural Trader team
