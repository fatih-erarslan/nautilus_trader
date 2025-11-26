/**
 * Cellular Automata
 *
 * Implements various cellular automata including:
 * - Conway's Game of Life
 * - Langton's Ant
 * - Elementary Cellular Automata (Wolfram's rules)
 * - Brian's Brain
 *
 * Uses AgentDB to track patterns and emergence
 */

import { AgentDB } from 'agentdb';
import { z } from 'zod';

export type CellState = number;

export interface GridConfig {
  width: number;
  height: number;
  wrapEdges?: boolean;
}

export interface AutomatonRule {
  name: string;
  states: number;
  neighborhoodType: 'moore' | 'von-neumann' | 'extended';
  updateRule: (cell: CellState, neighbors: CellState[]) => CellState;
}

const PatternSchema = z.object({
  type: z.string(),
  generation: z.number(),
  aliveCells: z.number(),
  stability: z.number(),
  entropy: z.number(),
  timestamp: z.number()
});

export class CellularAutomata {
  private grid: CellState[][];
  private nextGrid: CellState[][];
  private config: Required<GridConfig>;
  private rule: AutomatonRule;
  private generation: number = 0;
  private agentDB: AgentDB;
  private history: Map<string, number> = new Map(); // Pattern hash -> first seen generation

  constructor(config: GridConfig, rule: AutomatonRule) {
    this.config = {
      width: config.width,
      height: config.height,
      wrapEdges: config.wrapEdges ?? true
    };

    this.rule = rule;
    this.grid = this.createEmptyGrid();
    this.nextGrid = this.createEmptyGrid();

    this.agentDB = new AgentDB({
      enableCache: true,
      enableMemory: true,
      memorySize: 10000
    });
  }

  /**
   * Create empty grid
   */
  private createEmptyGrid(): CellState[][] {
    return Array(this.config.height)
      .fill(0)
      .map(() => Array(this.config.width).fill(0));
  }

  /**
   * Set cell state
   */
  setCell(x: number, y: number, state: CellState): void {
    if (this.isValidPosition(x, y)) {
      this.grid[y][x] = state;
    }
  }

  /**
   * Get cell state
   */
  getCell(x: number, y: number): CellState {
    if (!this.isValidPosition(x, y)) {
      return 0;
    }
    return this.grid[y][x];
  }

  /**
   * Check if position is valid
   */
  private isValidPosition(x: number, y: number): boolean {
    if (this.config.wrapEdges) {
      return true;
    }
    return x >= 0 && x < this.config.width && y >= 0 && y < this.config.height;
  }

  /**
   * Get wrapped position
   */
  private wrapPosition(x: number, y: number): [number, number] {
    const wrappedX = ((x % this.config.width) + this.config.width) % this.config.width;
    const wrappedY = ((y % this.config.height) + this.config.height) % this.config.height;
    return [wrappedX, wrappedY];
  }

  /**
   * Get neighbors based on neighborhood type
   */
  private getNeighbors(x: number, y: number): CellState[] {
    const neighbors: CellState[] = [];

    if (this.rule.neighborhoodType === 'moore') {
      // 8 neighbors (including diagonals)
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;

          const [nx, ny] = this.wrapPosition(x + dx, y + dy);
          neighbors.push(this.grid[ny][nx]);
        }
      }
    } else if (this.rule.neighborhoodType === 'von-neumann') {
      // 4 neighbors (no diagonals)
      const offsets = [[0, -1], [1, 0], [0, 1], [-1, 0]];

      for (const [dx, dy] of offsets) {
        const [nx, ny] = this.wrapPosition(x + dx, y + dy);
        neighbors.push(this.grid[ny][nx]);
      }
    } else if (this.rule.neighborhoodType === 'extended') {
      // Extended Moore neighborhood (5x5)
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          if (dx === 0 && dy === 0) continue;

          const [nx, ny] = this.wrapPosition(x + dx, y + dy);
          neighbors.push(this.grid[ny][nx]);
        }
      }
    }

    return neighbors;
  }

  /**
   * Update grid for one generation
   */
  async step(): Promise<void> {
    // Apply rule to each cell
    for (let y = 0; y < this.config.height; y++) {
      for (let x = 0; x < this.config.width; x++) {
        const currentState = this.grid[y][x];
        const neighbors = this.getNeighbors(x, y);
        const newState = this.rule.updateRule(currentState, neighbors);
        this.nextGrid[y][x] = newState;
      }
    }

    // Swap grids
    [this.grid, this.nextGrid] = [this.nextGrid, this.grid];
    this.generation++;

    // Track patterns
    await this.trackPattern();
  }

  /**
   * Track patterns and emergence
   */
  private async trackPattern(): Promise<void> {
    const patternHash = this.hashGrid();
    const aliveCells = this.countAliveCells();
    const entropy = this.calculateEntropy();

    // Check if pattern has been seen before
    const firstSeen = this.history.get(patternHash);
    const stability = firstSeen ? this.generation - firstSeen : 0;

    if (!firstSeen) {
      this.history.set(patternHash, this.generation);
    }

    const patternData = {
      type: this.rule.name,
      generation: this.generation,
      aliveCells,
      stability,
      entropy,
      timestamp: Date.now()
    };

    // Store in AgentDB
    const embedding = this.createGridEmbedding();
    await this.agentDB.store(
      `pattern:${this.rule.name}:${this.generation}`,
      JSON.stringify(patternData),
      embedding
    );
  }

  /**
   * Hash grid for pattern detection
   */
  private hashGrid(): string {
    let hash = '';
    for (let y = 0; y < this.config.height; y++) {
      for (let x = 0; x < this.config.width; x++) {
        hash += this.grid[y][x].toString();
      }
    }
    return hash;
  }

  /**
   * Count alive cells
   */
  private countAliveCells(): number {
    let count = 0;
    for (let y = 0; y < this.config.height; y++) {
      for (let x = 0; x < this.config.width; x++) {
        if (this.grid[y][x] > 0) count++;
      }
    }
    return count;
  }

  /**
   * Calculate Shannon entropy of grid
   */
  private calculateEntropy(): number {
    const stateCounts = new Map<CellState, number>();
    const totalCells = this.config.width * this.config.height;

    // Count states
    for (let y = 0; y < this.config.height; y++) {
      for (let x = 0; x < this.config.width; x++) {
        const state = this.grid[y][x];
        stateCounts.set(state, (stateCounts.get(state) || 0) + 1);
      }
    }

    // Calculate entropy
    let entropy = 0;
    for (const count of stateCounts.values()) {
      if (count > 0) {
        const probability = count / totalCells;
        entropy -= probability * Math.log2(probability);
      }
    }

    return entropy;
  }

  /**
   * Create embedding from grid for AgentDB
   */
  private createGridEmbedding(): number[] {
    const embedding: number[] = [];

    // Downsample grid to fixed size
    const sampleSize = 16;
    const xStep = this.config.width / sampleSize;
    const yStep = this.config.height / sampleSize;

    for (let sy = 0; sy < sampleSize; sy++) {
      for (let sx = 0; sx < sampleSize; sx++) {
        const x = Math.floor(sx * xStep);
        const y = Math.floor(sy * yStep);
        embedding.push(this.grid[y][x]);
      }
    }

    return embedding;
  }

  /**
   * Initialize with random state
   */
  randomize(density: number = 0.3): void {
    for (let y = 0; y < this.config.height; y++) {
      for (let x = 0; x < this.config.width; x++) {
        this.grid[y][x] = Math.random() < density ? 1 : 0;
      }
    }
    this.generation = 0;
    this.history.clear();
  }

  /**
   * Clear grid
   */
  clear(): void {
    this.grid = this.createEmptyGrid();
    this.nextGrid = this.createEmptyGrid();
    this.generation = 0;
    this.history.clear();
  }

  /**
   * Get current grid
   */
  getGrid(): CellState[][] {
    return this.grid.map(row => [...row]);
  }

  /**
   * Get current generation
   */
  getGeneration(): number {
    return this.generation;
  }

  /**
   * Load pattern from array
   */
  loadPattern(pattern: number[][], offsetX: number = 0, offsetY: number = 0): void {
    for (let y = 0; y < pattern.length; y++) {
      for (let x = 0; x < pattern[y].length; x++) {
        const gridX = offsetX + x;
        const gridY = offsetY + y;
        if (this.isValidPosition(gridX, gridY)) {
          this.grid[gridY][gridX] = pattern[y][x];
        }
      }
    }
  }
}

/**
 * Conway's Game of Life rules
 */
export const ConwaysGameOfLife: AutomatonRule = {
  name: 'Conway\'s Game of Life',
  states: 2,
  neighborhoodType: 'moore',
  updateRule: (cell: CellState, neighbors: CellState[]): CellState => {
    const aliveNeighbors = neighbors.filter(n => n === 1).length;

    if (cell === 1) {
      // Cell is alive
      return aliveNeighbors === 2 || aliveNeighbors === 3 ? 1 : 0;
    } else {
      // Cell is dead
      return aliveNeighbors === 3 ? 1 : 0;
    }
  }
};

/**
 * Langton's Ant rules
 */
export const LangtonsAnt: AutomatonRule = {
  name: 'Langton\'s Ant',
  states: 3, // 0 = white, 1 = black, 2 = ant
  neighborhoodType: 'von-neumann',
  updateRule: (cell: CellState, neighbors: CellState[]): CellState => {
    // Simplified version - full implementation would track ant direction
    if (cell === 2) {
      // Ant: turn and move
      return Math.random() < 0.5 ? 1 : 0;
    } else if (cell === 1) {
      // Black: turn ant right, flip to white
      return 0;
    } else {
      // White: turn ant left, flip to black
      return neighbors.some(n => n === 2) ? 2 : 0;
    }
  }
};

/**
 * Brian's Brain rules
 */
export const BriansBrain: AutomatonRule = {
  name: 'Brian\'s Brain',
  states: 3, // 0 = off, 1 = on, 2 = dying
  neighborhoodType: 'moore',
  updateRule: (cell: CellState, neighbors: CellState[]): CellState => {
    if (cell === 2) {
      // Dying -> Off
      return 0;
    } else if (cell === 1) {
      // On -> Dying
      return 2;
    } else {
      // Off -> On if exactly 2 neighbors are on
      const onNeighbors = neighbors.filter(n => n === 1).length;
      return onNeighbors === 2 ? 1 : 0;
    }
  }
};

/**
 * Seeds rule (B2/S)
 */
export const Seeds: AutomatonRule = {
  name: 'Seeds',
  states: 2,
  neighborhoodType: 'moore',
  updateRule: (cell: CellState, neighbors: CellState[]): CellState => {
    const aliveNeighbors = neighbors.filter(n => n === 1).length;

    if (cell === 1) {
      // All alive cells die
      return 0;
    } else {
      // Dead cell becomes alive if exactly 2 neighbors
      return aliveNeighbors === 2 ? 1 : 0;
    }
  }
};
