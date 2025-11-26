/**
 * @neural-trader/example-adaptive-systems
 *
 * Self-organizing multi-agent systems with swarm intelligence,
 * emergence detection, and adaptive behavior.
 *
 * @example
 * ```typescript
 * import { BoidsSimulation, AntColonyOptimization, EmergenceDetector } from '@neural-trader/example-adaptive-systems';
 *
 * // Boids flocking simulation
 * const boids = new BoidsSimulation({ width: 800, height: 600 });
 * for (let i = 0; i < 50; i++) {
 *   boids.addBoid(`boid-${i}`, { x: Math.random() * 800, y: Math.random() * 600 });
 * }
 * await boids.update();
 *
 * // Ant colony optimization
 * const aco = new AntColonyOptimization();
 * // ... add nodes and edges
 * const result = await aco.optimize('start', 'goal');
 *
 * // Emergence detection
 * const detector = new EmergenceDetector();
 * // ... add system states
 * const metrics = detector.getLatestMetrics();
 * ```
 */

// Boids
export {
  BoidsSimulation,
  type Boid,
  type BoidConfig,
  type Vector2D
} from './boids';

// Ant Colony Optimization
export {
  AntColonyOptimization,
  type Node,
  type Edge,
  type Ant,
  type AntColonyConfig
} from './ant-colony';

// Cellular Automata
export {
  CellularAutomata,
  type CellState,
  type GridConfig,
  type AutomatonRule,
  ConwaysGameOfLife,
  LangtonsAnt,
  BriansBrain,
  Seeds
} from './cellular-automata';

// Emergence Detection
export {
  EmergenceDetector,
  type SystemState,
  type AgentState,
  type EmergenceMetrics,
  type EmergenceEvent
} from './emergence';

/**
 * Utility functions for adaptive systems
 */
export const utils = {
  /**
   * Generate random position within boundaries
   */
  randomPosition(width: number, height: number): Vector2D {
    return {
      x: Math.random() * width,
      y: Math.random() * height
    };
  },

  /**
   * Generate random velocity
   */
  randomVelocity(maxSpeed: number): Vector2D {
    const angle = Math.random() * 2 * Math.PI;
    const speed = Math.random() * maxSpeed;
    return {
      x: Math.cos(angle) * speed,
      y: Math.sin(angle) * speed
    };
  },

  /**
   * Create grid graph for pathfinding
   */
  createGridGraph(
    width: number,
    height: number,
    spacing: number = 50
  ): { nodes: Node[]; edges: Array<[string, string]> } {
    const nodes: Node[] = [];
    const edges: Array<[string, string]> = [];

    // Create nodes
    for (let y = 0; y < height; y += spacing) {
      for (let x = 0; x < width; x += spacing) {
        nodes.push({
          id: `node-${x}-${y}`,
          x,
          y,
          type: 'normal'
        });
      }
    }

    // Create edges (4-connected grid)
    for (let y = 0; y < height; y += spacing) {
      for (let x = 0; x < width; x += spacing) {
        const currentId = `node-${x}-${y}`;

        // Connect to right neighbor
        if (x + spacing < width) {
          const rightId = `node-${x + spacing}-${y}`;
          edges.push([currentId, rightId]);
        }

        // Connect to bottom neighbor
        if (y + spacing < height) {
          const bottomId = `node-${x}-${y + spacing}`;
          edges.push([currentId, bottomId]);
        }
      }
    }

    return { nodes, edges };
  },

  /**
   * Calculate distance between two points
   */
  distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  },

  /**
   * Calculate average position of points
   */
  centroid(points: Array<{ x: number; y: number }>): { x: number; y: number } {
    if (points.length === 0) return { x: 0, y: 0 };

    const sum = points.reduce(
      (acc, p) => ({ x: acc.x + p.x, y: acc.y + p.y }),
      { x: 0, y: 0 }
    );

    return {
      x: sum.x / points.length,
      y: sum.y / points.length
    };
  },

  /**
   * Calculate Shannon entropy of a distribution
   */
  entropy(values: number[]): number {
    const total = values.reduce((sum, v) => sum + v, 0);
    if (total === 0) return 0;

    let entropy = 0;
    for (const value of values) {
      if (value > 0) {
        const probability = value / total;
        entropy -= probability * Math.log2(probability);
      }
    }

    return entropy;
  },

  /**
   * Calculate order parameter (opposite of entropy)
   */
  orderParameter(distribution: number[]): number {
    const maxEntropy = Math.log2(distribution.length);
    const actualEntropy = this.entropy(distribution);
    return 1 - (actualEntropy / maxEntropy);
  }
};

// Re-export Vector2D type for convenience
import type { Vector2D } from './boids';
import type { Node } from './ant-colony';
