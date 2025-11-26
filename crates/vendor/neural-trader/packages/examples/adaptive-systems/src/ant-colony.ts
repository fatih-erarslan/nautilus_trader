/**
 * Ant Colony Optimization (ACO)
 *
 * Implements ant colony optimization for path finding and TSP-like problems.
 * Uses pheromone trails, heuristic information, and evaporation.
 *
 * Features:
 * - Multiple ant agents exploring in parallel
 * - Pheromone deposit and evaporation
 * - Probabilistic path selection
 * - Best path memory using AgentDB
 */

import { AgentDB } from 'agentdb';
import { z } from 'zod';

export interface Node {
  id: string;
  x: number;
  y: number;
  type?: 'start' | 'goal' | 'obstacle' | 'normal';
}

export interface Edge {
  from: string;
  to: string;
  distance: number;
  pheromone: number;
}

export interface Ant {
  id: string;
  currentNode: string;
  visitedNodes: string[];
  pathLength: number;
  isActive: boolean;
}

export interface AntColonyConfig {
  numAnts?: number;
  alpha?: number; // Pheromone importance
  beta?: number; // Distance importance
  evaporationRate?: number;
  pheromoneDeposit?: number;
  maxIterations?: number;
  convergenceThreshold?: number;
}

const PathSchema = z.object({
  nodes: z.array(z.string()),
  length: z.number(),
  pheromoneLevel: z.number(),
  iteration: z.number(),
  timestamp: z.number()
});

export class AntColonyOptimization {
  private nodes: Map<string, Node> = new Map();
  private edges: Map<string, Edge> = new Map();
  private ants: Ant[] = [];
  private agentDB: AgentDB;
  private config: Required<AntColonyConfig>;
  private bestPath: { nodes: string[]; length: number } | null = null;
  private iteration: number = 0;

  constructor(config: AntColonyConfig = {}) {
    this.config = {
      numAnts: config.numAnts ?? 20,
      alpha: config.alpha ?? 1.0, // Pheromone importance
      beta: config.beta ?? 2.0, // Distance importance
      evaporationRate: config.evaporationRate ?? 0.5,
      pheromoneDeposit: config.pheromoneDeposit ?? 100,
      maxIterations: config.maxIterations ?? 100,
      convergenceThreshold: config.convergenceThreshold ?? 0.01
    };

    this.agentDB = new AgentDB({
      enableCache: true,
      enableMemory: true,
      memorySize: 10000
    });
  }

  /**
   * Add a node to the graph
   */
  addNode(node: Node): void {
    this.nodes.set(node.id, node);
  }

  /**
   * Add an edge between two nodes
   */
  addEdge(fromId: string, toId: string, bidirectional: boolean = true): void {
    const from = this.nodes.get(fromId);
    const to = this.nodes.get(toId);

    if (!from || !to) {
      throw new Error(`Nodes ${fromId} or ${toId} not found`);
    }

    const distance = this.calculateDistance(from, to);
    const edgeKey = `${fromId}->${toId}`;

    this.edges.set(edgeKey, {
      from: fromId,
      to: toId,
      distance,
      pheromone: 1.0 // Initial pheromone level
    });

    if (bidirectional) {
      const reverseKey = `${toId}->${fromId}`;
      this.edges.set(reverseKey, {
        from: toId,
        to: fromId,
        distance,
        pheromone: 1.0
      });
    }
  }

  /**
   * Run ACO algorithm to find optimal path
   */
  async optimize(startNodeId: string, goalNodeId: string): Promise<{
    path: string[];
    length: number;
    iterations: number;
  }> {
    const startNode = this.nodes.get(startNodeId);
    const goalNode = this.nodes.get(goalNodeId);

    if (!startNode || !goalNode) {
      throw new Error('Start or goal node not found');
    }

    this.bestPath = null;
    this.iteration = 0;

    let consecutiveNoImprovement = 0;
    let previousBestLength = Infinity;

    // Main ACO loop
    while (this.iteration < this.config.maxIterations) {
      // Initialize ants
      this.initializeAnts(startNodeId);

      // Move all ants until they reach the goal
      await this.moveAllAnts(goalNodeId);

      // Update pheromones
      this.evaporatePheromones();
      this.depositPheromones(goalNodeId);

      // Store best paths in AgentDB for learning
      if (this.bestPath) {
        await this.storeBestPath();
      }

      // Check for convergence
      if (this.bestPath) {
        const improvement = previousBestLength - this.bestPath.length;
        if (improvement < this.config.convergenceThreshold) {
          consecutiveNoImprovement++;
          if (consecutiveNoImprovement >= 10) {
            break; // Converged
          }
        } else {
          consecutiveNoImprovement = 0;
        }
        previousBestLength = this.bestPath.length;
      }

      this.iteration++;
    }

    if (!this.bestPath) {
      throw new Error('No path found');
    }

    return {
      path: this.bestPath.nodes,
      length: this.bestPath.length,
      iterations: this.iteration
    };
  }

  /**
   * Initialize ants at starting position
   */
  private initializeAnts(startNodeId: string): void {
    this.ants = [];

    for (let i = 0; i < this.config.numAnts; i++) {
      this.ants.push({
        id: `ant-${i}`,
        currentNode: startNodeId,
        visitedNodes: [startNodeId],
        pathLength: 0,
        isActive: true
      });
    }
  }

  /**
   * Move all ants until they reach goal or get stuck
   */
  private async moveAllAnts(goalNodeId: string): Promise<void> {
    let activeAnts = this.ants.filter(ant => ant.isActive);

    while (activeAnts.length > 0) {
      for (const ant of activeAnts) {
        if (!ant.isActive) continue;

        // Check if ant reached goal
        if (ant.currentNode === goalNodeId) {
          ant.isActive = false;
          this.updateBestPath(ant);
          continue;
        }

        // Select next node
        const nextNode = this.selectNextNode(ant, goalNodeId);

        if (nextNode) {
          const edgeKey = `${ant.currentNode}->${nextNode}`;
          const edge = this.edges.get(edgeKey);

          if (edge) {
            ant.pathLength += edge.distance;
            ant.currentNode = nextNode;
            ant.visitedNodes.push(nextNode);
          }
        } else {
          // Ant is stuck
          ant.isActive = false;
        }
      }

      activeAnts = this.ants.filter(ant => ant.isActive);
    }
  }

  /**
   * Select next node using probabilistic decision rule
   */
  private selectNextNode(ant: Ant, goalNodeId: string): string | null {
    const currentNode = ant.currentNode;
    const neighbors = this.getUnvisitedNeighbors(ant);

    if (neighbors.length === 0) {
      return null;
    }

    // Calculate probabilities for each neighbor
    const probabilities = neighbors.map(neighborId => {
      const edgeKey = `${currentNode}->${neighborId}`;
      const edge = this.edges.get(edgeKey);

      if (!edge) return { nodeId: neighborId, probability: 0 };

      // Pheromone factor
      const pheromoneFactor = Math.pow(edge.pheromone, this.config.alpha);

      // Heuristic factor (inverse of distance)
      const heuristicFactor = Math.pow(1 / edge.distance, this.config.beta);

      // Add goal attraction heuristic
      const goalNode = this.nodes.get(goalNodeId);
      const neighborNode = this.nodes.get(neighborId);
      const goalDistance = goalNode && neighborNode
        ? this.calculateDistance(neighborNode, goalNode)
        : Infinity;
      const goalAttraction = 1 / (1 + goalDistance);

      const probability = pheromoneFactor * heuristicFactor * (1 + goalAttraction);

      return { nodeId: neighborId, probability };
    });

    // Normalize probabilities
    const totalProbability = probabilities.reduce((sum, p) => sum + p.probability, 0);

    if (totalProbability === 0) {
      return neighbors[Math.floor(Math.random() * neighbors.length)];
    }

    const normalizedProbs = probabilities.map(p => ({
      nodeId: p.nodeId,
      probability: p.probability / totalProbability
    }));

    // Roulette wheel selection
    const random = Math.random();
    let cumulativeProbability = 0;

    for (const prob of normalizedProbs) {
      cumulativeProbability += prob.probability;
      if (random <= cumulativeProbability) {
        return prob.nodeId;
      }
    }

    return neighbors[0];
  }

  /**
   * Get unvisited neighboring nodes
   */
  private getUnvisitedNeighbors(ant: Ant): string[] {
    const neighbors: string[] = [];

    for (const [key, edge] of this.edges) {
      if (edge.from === ant.currentNode && !ant.visitedNodes.includes(edge.to)) {
        neighbors.push(edge.to);
      }
    }

    return neighbors;
  }

  /**
   * Evaporate pheromones on all edges
   */
  private evaporatePheromones(): void {
    for (const [key, edge] of this.edges) {
      edge.pheromone *= (1 - this.config.evaporationRate);

      // Ensure minimum pheromone level
      edge.pheromone = Math.max(edge.pheromone, 0.01);
    }
  }

  /**
   * Deposit pheromones on paths taken by ants
   */
  private depositPheromones(goalNodeId: string): void {
    for (const ant of this.ants) {
      // Only deposit if ant reached goal
      if (ant.visitedNodes[ant.visitedNodes.length - 1] !== goalNodeId) {
        continue;
      }

      // Amount of pheromone to deposit (inversely proportional to path length)
      const deposit = this.config.pheromoneDeposit / ant.pathLength;

      // Deposit on each edge in the path
      for (let i = 0; i < ant.visitedNodes.length - 1; i++) {
        const from = ant.visitedNodes[i];
        const to = ant.visitedNodes[i + 1];
        const edgeKey = `${from}->${to}`;
        const edge = this.edges.get(edgeKey);

        if (edge) {
          edge.pheromone += deposit;
        }
      }
    }
  }

  /**
   * Update best path if ant found better solution
   */
  private updateBestPath(ant: Ant): void {
    if (!this.bestPath || ant.pathLength < this.bestPath.length) {
      this.bestPath = {
        nodes: [...ant.visitedNodes],
        length: ant.pathLength
      };
    }
  }

  /**
   * Store best path in AgentDB for learning
   */
  private async storeBestPath(): Promise<void> {
    if (!this.bestPath) return;

    const pathData = {
      nodes: this.bestPath.nodes,
      length: this.bestPath.length,
      pheromoneLevel: this.getAveragePheromone(this.bestPath.nodes),
      iteration: this.iteration,
      timestamp: Date.now()
    };

    // Create embedding from path coordinates
    const pathEmbedding = this.createPathEmbedding(this.bestPath.nodes);

    await this.agentDB.store(
      `path:${this.iteration}`,
      JSON.stringify(pathData),
      pathEmbedding
    );
  }

  /**
   * Get average pheromone level on a path
   */
  private getAveragePheromone(path: string[]): number {
    let totalPheromone = 0;
    let edgeCount = 0;

    for (let i = 0; i < path.length - 1; i++) {
      const edgeKey = `${path[i]}->${path[i + 1]}`;
      const edge = this.edges.get(edgeKey);

      if (edge) {
        totalPheromone += edge.pheromone;
        edgeCount++;
      }
    }

    return edgeCount > 0 ? totalPheromone / edgeCount : 0;
  }

  /**
   * Create embedding from path for AgentDB storage
   */
  private createPathEmbedding(path: string[]): number[] {
    const embedding: number[] = [];

    for (const nodeId of path) {
      const node = this.nodes.get(nodeId);
      if (node) {
        embedding.push(node.x, node.y);
      }
    }

    // Pad or truncate to fixed size (e.g., 128 dimensions)
    const targetSize = 128;
    while (embedding.length < targetSize) {
      embedding.push(0);
    }

    return embedding.slice(0, targetSize);
  }

  /**
   * Calculate Euclidean distance between two nodes
   */
  private calculateDistance(a: Node, b: Node): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /**
   * Get all nodes
   */
  getNodes(): Node[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Get all edges
   */
  getEdges(): Edge[] {
    return Array.from(this.edges.values());
  }

  /**
   * Get best path found so far
   */
  getBestPath(): { nodes: string[]; length: number } | null {
    return this.bestPath;
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.ants = [];
    this.bestPath = null;
    this.iteration = 0;
  }
}
