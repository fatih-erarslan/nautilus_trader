/**
 * Boids Algorithm - Flocking Behavior Simulation
 *
 * Implements Reynolds' boids algorithm with three core rules:
 * 1. Separation: Avoid crowding neighbors
 * 2. Alignment: Steer towards average heading of neighbors
 * 3. Cohesion: Move towards average position of neighbors
 *
 * Uses AgentDB for spatial indexing and memory-based learning
 */

import { AgentDB } from 'agentdb';
import { z } from 'zod';

export interface Vector2D {
  x: number;
  y: number;
}

export interface Boid {
  id: string;
  position: Vector2D;
  velocity: Vector2D;
  acceleration: Vector2D;
  maxSpeed: number;
  maxForce: number;
  perceptionRadius: number;
}

export interface BoidConfig {
  separationWeight?: number;
  alignmentWeight?: number;
  cohesionWeight?: number;
  separationRadius?: number;
  alignmentRadius?: number;
  cohesionRadius?: number;
  maxSpeed?: number;
  maxForce?: number;
  boundaryBehavior?: 'wrap' | 'bounce' | 'attract';
}

const BoidStateSchema = z.object({
  id: z.string(),
  position: z.object({ x: z.number(), y: z.number() }),
  velocity: z.object({ x: z.number(), y: z.number() }),
  separationScore: z.number(),
  alignmentScore: z.number(),
  cohesionScore: z.number(),
  timestamp: z.number()
});

export class BoidsSimulation {
  private boids: Map<string, Boid> = new Map();
  private agentDB: AgentDB;
  private config: Required<BoidConfig>;
  private boundaries: { width: number; height: number };

  constructor(
    boundaries: { width: number; height: number },
    config: BoidConfig = {}
  ) {
    this.boundaries = boundaries;
    this.config = {
      separationWeight: config.separationWeight ?? 1.5,
      alignmentWeight: config.alignmentWeight ?? 1.0,
      cohesionWeight: config.cohesionWeight ?? 1.0,
      separationRadius: config.separationRadius ?? 25,
      alignmentRadius: config.alignmentRadius ?? 50,
      cohesionRadius: config.cohesionRadius ?? 50,
      maxSpeed: config.maxSpeed ?? 4,
      maxForce: config.maxForce ?? 0.1,
      boundaryBehavior: config.boundaryBehavior ?? 'wrap'
    };

    // Initialize AgentDB for spatial queries
    this.agentDB = new AgentDB({
      enableCache: true,
      enableMemory: true,
      memorySize: 10000
    });
  }

  /**
   * Add a boid to the simulation
   */
  addBoid(
    id: string,
    position: Vector2D,
    velocity: Vector2D = { x: 0, y: 0 }
  ): void {
    const boid: Boid = {
      id,
      position,
      velocity,
      acceleration: { x: 0, y: 0 },
      maxSpeed: this.config.maxSpeed,
      maxForce: this.config.maxForce,
      perceptionRadius: Math.max(
        this.config.separationRadius,
        this.config.alignmentRadius,
        this.config.cohesionRadius
      )
    };

    this.boids.set(id, boid);
  }

  /**
   * Update simulation for one time step
   */
  async update(deltaTime: number = 1): Promise<void> {
    // Store current states in AgentDB for spatial queries
    await this.storeBoidStates();

    // Calculate forces for each boid
    const forces = new Map<string, Vector2D>();

    for (const [id, boid] of this.boids) {
      const neighbors = await this.findNeighbors(boid);

      const separation = this.calculateSeparation(boid, neighbors);
      const alignment = this.calculateAlignment(boid, neighbors);
      const cohesion = this.calculateCohesion(boid, neighbors);

      // Weighted sum of forces
      const totalForce = this.addVectors([
        this.scaleVector(separation, this.config.separationWeight),
        this.scaleVector(alignment, this.config.alignmentWeight),
        this.scaleVector(cohesion, this.config.cohesionWeight)
      ]);

      forces.set(id, totalForce);
    }

    // Apply forces and update positions
    for (const [id, force] of forces) {
      const boid = this.boids.get(id)!;
      this.applyForce(boid, force);
      this.updateBoid(boid, deltaTime);
    }
  }

  /**
   * Separation: Steer to avoid crowding local flockmates
   */
  private calculateSeparation(boid: Boid, neighbors: Boid[]): Vector2D {
    const steer: Vector2D = { x: 0, y: 0 };
    let count = 0;

    for (const neighbor of neighbors) {
      const distance = this.distance(boid.position, neighbor.position);

      if (distance > 0 && distance < this.config.separationRadius) {
        // Calculate vector pointing away from neighbor
        const diff = this.subtractVectors(boid.position, neighbor.position);
        const normalized = this.normalize(diff);
        // Weight by distance (closer = stronger)
        const weighted = this.scaleVector(normalized, 1 / distance);

        steer.x += weighted.x;
        steer.y += weighted.y;
        count++;
      }
    }

    if (count > 0) {
      steer.x /= count;
      steer.y /= count;
    }

    return this.limitVector(steer, boid.maxForce);
  }

  /**
   * Alignment: Steer towards the average heading of local flockmates
   */
  private calculateAlignment(boid: Boid, neighbors: Boid[]): Vector2D {
    const avgVelocity: Vector2D = { x: 0, y: 0 };
    let count = 0;

    for (const neighbor of neighbors) {
      const distance = this.distance(boid.position, neighbor.position);

      if (distance > 0 && distance < this.config.alignmentRadius) {
        avgVelocity.x += neighbor.velocity.x;
        avgVelocity.y += neighbor.velocity.y;
        count++;
      }
    }

    if (count > 0) {
      avgVelocity.x /= count;
      avgVelocity.y /= count;

      // Steer towards average velocity
      const desired = this.normalize(avgVelocity);
      const scaled = this.scaleVector(desired, boid.maxSpeed);
      const steer = this.subtractVectors(scaled, boid.velocity);

      return this.limitVector(steer, boid.maxForce);
    }

    return { x: 0, y: 0 };
  }

  /**
   * Cohesion: Steer to move towards the average position of local flockmates
   */
  private calculateCohesion(boid: Boid, neighbors: Boid[]): Vector2D {
    const avgPosition: Vector2D = { x: 0, y: 0 };
    let count = 0;

    for (const neighbor of neighbors) {
      const distance = this.distance(boid.position, neighbor.position);

      if (distance > 0 && distance < this.config.cohesionRadius) {
        avgPosition.x += neighbor.position.x;
        avgPosition.y += neighbor.position.y;
        count++;
      }
    }

    if (count > 0) {
      avgPosition.x /= count;
      avgPosition.y /= count;

      // Steer towards average position
      return this.seek(boid, avgPosition);
    }

    return { x: 0, y: 0 };
  }

  /**
   * Seek behavior: steer towards a target
   */
  private seek(boid: Boid, target: Vector2D): Vector2D {
    const desired = this.subtractVectors(target, boid.position);
    const normalized = this.normalize(desired);
    const scaled = this.scaleVector(normalized, boid.maxSpeed);
    const steer = this.subtractVectors(scaled, boid.velocity);

    return this.limitVector(steer, boid.maxForce);
  }

  /**
   * Apply force to boid's acceleration
   */
  private applyForce(boid: Boid, force: Vector2D): void {
    boid.acceleration.x += force.x;
    boid.acceleration.y += force.y;
  }

  /**
   * Update boid position and velocity
   */
  private updateBoid(boid: Boid, deltaTime: number): void {
    // Update velocity
    boid.velocity.x += boid.acceleration.x * deltaTime;
    boid.velocity.y += boid.acceleration.y * deltaTime;

    // Limit velocity
    boid.velocity = this.limitVector(boid.velocity, boid.maxSpeed);

    // Update position
    boid.position.x += boid.velocity.x * deltaTime;
    boid.position.y += boid.velocity.y * deltaTime;

    // Handle boundaries
    this.handleBoundaries(boid);

    // Reset acceleration
    boid.acceleration.x = 0;
    boid.acceleration.y = 0;
  }

  /**
   * Handle boundary conditions
   */
  private handleBoundaries(boid: Boid): void {
    switch (this.config.boundaryBehavior) {
      case 'wrap':
        if (boid.position.x < 0) boid.position.x = this.boundaries.width;
        if (boid.position.x > this.boundaries.width) boid.position.x = 0;
        if (boid.position.y < 0) boid.position.y = this.boundaries.height;
        if (boid.position.y > this.boundaries.height) boid.position.y = 0;
        break;

      case 'bounce':
        if (boid.position.x < 0 || boid.position.x > this.boundaries.width) {
          boid.velocity.x *= -1;
          boid.position.x = Math.max(0, Math.min(this.boundaries.width, boid.position.x));
        }
        if (boid.position.y < 0 || boid.position.y > this.boundaries.height) {
          boid.velocity.y *= -1;
          boid.position.y = Math.max(0, Math.min(this.boundaries.height, boid.position.y));
        }
        break;

      case 'attract':
        const margin = 50;
        const turnForce = 0.5;

        if (boid.position.x < margin) {
          this.applyForce(boid, { x: turnForce, y: 0 });
        }
        if (boid.position.x > this.boundaries.width - margin) {
          this.applyForce(boid, { x: -turnForce, y: 0 });
        }
        if (boid.position.y < margin) {
          this.applyForce(boid, { x: 0, y: turnForce });
        }
        if (boid.position.y > this.boundaries.height - margin) {
          this.applyForce(boid, { x: 0, y: -turnForce });
        }
        break;
    }
  }

  /**
   * Store boid states in AgentDB for spatial queries
   */
  private async storeBoidStates(): Promise<void> {
    for (const [id, boid] of this.boids) {
      const state = {
        id,
        position: boid.position,
        velocity: boid.velocity,
        separationScore: 0,
        alignmentScore: 0,
        cohesionScore: 0,
        timestamp: Date.now()
      };

      await this.agentDB.store(
        `boid:${id}`,
        JSON.stringify(state),
        [boid.position.x, boid.position.y]
      );
    }
  }

  /**
   * Find neighbors within perception radius using AgentDB
   */
  private async findNeighbors(boid: Boid): Promise<Boid[]> {
    const results = await this.agentDB.query(
      [boid.position.x, boid.position.y],
      Math.floor(boid.perceptionRadius / 10) // k neighbors
    );

    const neighbors: Boid[] = [];

    for (const result of results) {
      if (result.id === `boid:${boid.id}`) continue;

      const state = JSON.parse(result.text);
      const neighborBoid = this.boids.get(state.id);

      if (neighborBoid) {
        const distance = this.distance(boid.position, neighborBoid.position);
        if (distance <= boid.perceptionRadius) {
          neighbors.push(neighborBoid);
        }
      }
    }

    return neighbors;
  }

  /**
   * Get all boids
   */
  getBoids(): Boid[] {
    return Array.from(this.boids.values());
  }

  /**
   * Get boid by id
   */
  getBoid(id: string): Boid | undefined {
    return this.boids.get(id);
  }

  /**
   * Clear all boids
   */
  clear(): void {
    this.boids.clear();
  }

  // Vector math utilities

  private distance(a: Vector2D, b: Vector2D): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private magnitude(v: Vector2D): number {
    return Math.sqrt(v.x * v.x + v.y * v.y);
  }

  private normalize(v: Vector2D): Vector2D {
    const mag = this.magnitude(v);
    if (mag === 0) return { x: 0, y: 0 };
    return { x: v.x / mag, y: v.y / mag };
  }

  private limitVector(v: Vector2D, max: number): Vector2D {
    const mag = this.magnitude(v);
    if (mag > max) {
      return this.scaleVector(this.normalize(v), max);
    }
    return v;
  }

  private scaleVector(v: Vector2D, scalar: number): Vector2D {
    return { x: v.x * scalar, y: v.y * scalar };
  }

  private addVectors(vectors: Vector2D[]): Vector2D {
    return vectors.reduce(
      (acc, v) => ({ x: acc.x + v.x, y: acc.y + v.y }),
      { x: 0, y: 0 }
    );
  }

  private subtractVectors(a: Vector2D, b: Vector2D): Vector2D {
    return { x: a.x - b.x, y: a.y - b.y };
  }
}
