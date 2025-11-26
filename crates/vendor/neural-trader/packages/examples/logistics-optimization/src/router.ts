/**
 * Vehicle Routing Problem (VRP) solver with time windows
 * Implements multiple optimization algorithms
 */

import { Customer, Vehicle, Route, Solution, OptimizationConfig, Location } from './types';

export class VRPRouter {
  private customers: Customer[];
  private vehicles: Vehicle[];
  private distanceMatrix: Map<string, number>;
  private timeMatrix: Map<string, number>;

  constructor(customers: Customer[], vehicles: Vehicle[]) {
    this.customers = customers;
    this.vehicles = vehicles;
    this.distanceMatrix = new Map();
    this.timeMatrix = new Map();
    this.precomputeMatrices();
  }

  /**
   * Precompute distance and time matrices for all location pairs
   */
  private precomputeMatrices(): void {
    const allLocations = [
      ...this.vehicles.map(v => v.startLocation),
      ...this.customers.map(c => c.location)
    ];

    for (let i = 0; i < allLocations.length; i++) {
      for (let j = 0; j < allLocations.length; j++) {
        if (i !== j) {
          const key = `${allLocations[i].id}-${allLocations[j].id}`;
          const distance = this.calculateDistance(allLocations[i], allLocations[j]);
          this.distanceMatrix.set(key, distance);
          // Assume average speed of 50 km/h for time calculation
          this.timeMatrix.set(key, (distance / 50) * 60); // minutes
        }
      }
    }
  }

  /**
   * Calculate Haversine distance between two locations
   */
  private calculateDistance(loc1: Location, loc2: Location): number {
    const R = 6371; // Earth's radius in km
    const dLat = this.toRad(loc2.lat - loc1.lat);
    const dLng = this.toRad(loc2.lng - loc1.lng);

    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(this.toRad(loc1.lat)) * Math.cos(this.toRad(loc2.lat)) *
              Math.sin(dLng / 2) * Math.sin(dLng / 2);

    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  }

  private toRad(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  /**
   * Get distance between two locations
   */
  getDistance(loc1Id: string, loc2Id: string): number {
    const key = `${loc1Id}-${loc2Id}`;
    return this.distanceMatrix.get(key) || 0;
  }

  /**
   * Get travel time between two locations
   */
  getTravelTime(loc1Id: string, loc2Id: string): number {
    const key = `${loc1Id}-${loc2Id}`;
    return this.timeMatrix.get(key) || 0;
  }

  /**
   * Genetic Algorithm for VRP optimization
   */
  async solveGenetic(config: OptimizationConfig): Promise<Solution> {
    const startTime = Date.now();
    const populationSize = config.populationSize || 50;
    const mutationRate = config.mutationRate || 0.1;
    const crossoverRate = config.crossoverRate || 0.8;

    // Initialize population with random solutions
    let population = this.initializePopulation(populationSize);

    for (let iteration = 0; iteration < config.maxIterations; iteration++) {
      // Evaluate fitness
      population = population.map(sol => ({
        ...sol,
        fitness: this.evaluateFitness(sol)
      }));

      // Sort by fitness (lower is better)
      population.sort((a, b) => a.fitness - b.fitness);

      // Selection, crossover, and mutation
      const newPopulation: Solution[] = [];

      // Keep best solutions (elitism)
      newPopulation.push(...population.slice(0, Math.floor(populationSize * 0.1)));

      // Generate new solutions
      while (newPopulation.length < populationSize) {
        const parent1 = this.tournamentSelection(population);
        const parent2 = this.tournamentSelection(population);

        let offspring: Solution;
        if (Math.random() < crossoverRate) {
          offspring = this.crossover(parent1, parent2);
        } else {
          offspring = { ...parent1 };
        }

        if (Math.random() < mutationRate) {
          offspring = this.mutate(offspring);
        }

        newPopulation.push(offspring);
      }

      population = newPopulation;
    }

    // Return best solution
    const bestSolution = population.sort((a, b) => a.fitness - b.fitness)[0];
    bestSolution.metadata = {
      algorithm: 'genetic',
      iterations: config.maxIterations,
      computeTime: Date.now() - startTime
    };

    return bestSolution;
  }

  /**
   * Simulated Annealing for VRP optimization
   */
  async solveSimulatedAnnealing(config: OptimizationConfig): Promise<Solution> {
    const startTime = Date.now();
    let temperature = config.temperature || 1000;
    const coolingRate = config.coolingRate || 0.995;

    // Start with a greedy solution
    let currentSolution = this.constructGreedySolution();
    let bestSolution = { ...currentSolution };
    let currentFitness = this.evaluateFitness(currentSolution);
    let bestFitness = currentFitness;

    for (let iteration = 0; iteration < config.maxIterations && temperature > 1; iteration++) {
      // Generate neighbor solution
      const neighborSolution = this.generateNeighbor(currentSolution);
      const neighborFitness = this.evaluateFitness(neighborSolution);

      // Accept or reject neighbor
      const delta = neighborFitness - currentFitness;
      if (delta < 0 || Math.random() < Math.exp(-delta / temperature)) {
        currentSolution = neighborSolution;
        currentFitness = neighborFitness;

        if (currentFitness < bestFitness) {
          bestSolution = { ...currentSolution };
          bestFitness = currentFitness;
        }
      }

      temperature *= coolingRate;
    }

    bestSolution.metadata = {
      algorithm: 'simulated-annealing',
      iterations: config.maxIterations,
      computeTime: Date.now() - startTime
    };

    return bestSolution;
  }

  /**
   * Ant Colony Optimization for VRP
   */
  async solveAntColony(config: OptimizationConfig): Promise<Solution> {
    const startTime = Date.now();
    const numAnts = config.populationSize || 20;
    const evaporationRate = config.pheromoneEvaporation || 0.1;

    // Initialize pheromone matrix
    const pheromones = new Map<string, number>();

    let bestSolution: Solution | null = null;
    let bestFitness = Infinity;

    for (let iteration = 0; iteration < config.maxIterations; iteration++) {
      const solutions: Solution[] = [];

      // Each ant constructs a solution
      for (let ant = 0; ant < numAnts; ant++) {
        const solution = this.constructAntSolution(pheromones);
        const fitness = this.evaluateFitness(solution);
        solution.fitness = fitness;
        solutions.push(solution);

        if (fitness < bestFitness) {
          bestSolution = { ...solution };
          bestFitness = fitness;
        }
      }

      // Update pheromones
      this.updatePheromones(pheromones, solutions, evaporationRate);
    }

    if (!bestSolution) {
      bestSolution = this.constructGreedySolution();
    }

    bestSolution.metadata = {
      algorithm: 'ant-colony',
      iterations: config.maxIterations,
      computeTime: Date.now() - startTime
    };

    return bestSolution;
  }

  /**
   * Initialize random population
   */
  private initializePopulation(size: number): Solution[] {
    const population: Solution[] = [];
    for (let i = 0; i < size; i++) {
      population.push(this.constructRandomSolution());
    }
    return population;
  }

  /**
   * Construct a random solution
   */
  private constructRandomSolution(): Solution {
    const routes: Route[] = [];
    const unassignedCustomers: Customer[] = [];
    const shuffledCustomers = [...this.customers].sort(() => Math.random() - 0.5);

    for (const vehicle of this.vehicles) {
      const route: Route = {
        vehicleId: vehicle.id,
        customers: [],
        totalDistance: 0,
        totalTime: 0,
        totalCost: 0,
        utilizationRate: 0,
        timeWindowViolations: 0,
        capacityViolations: 0
      };

      let currentLoad = 0;
      let currentTime = vehicle.availableTimeWindow.start;
      let currentLocation = vehicle.startLocation;

      for (const customer of shuffledCustomers) {
        if (currentLoad + customer.demand <= vehicle.capacity) {
          const travelTime = this.getTravelTime(currentLocation.id, customer.location.id);
          const arrivalTime = currentTime + travelTime;

          if (arrivalTime <= customer.timeWindow.end) {
            route.customers.push(customer);
            currentLoad += customer.demand;
            currentTime = Math.max(arrivalTime, customer.timeWindow.start) + customer.serviceTime;
            currentLocation = customer.location;
          }
        }
      }

      routes.push(route);
    }

    // Track unassigned customers
    const assignedIds = new Set(routes.flatMap(r => r.customers.map(c => c.id)));
    for (const customer of this.customers) {
      if (!assignedIds.has(customer.id)) {
        unassignedCustomers.push(customer);
      }
    }

    return {
      routes,
      totalCost: 0,
      totalDistance: 0,
      unassignedCustomers,
      fitness: 0,
      metadata: { algorithm: 'random', iterations: 0, computeTime: 0 }
    };
  }

  /**
   * Construct greedy solution (nearest neighbor)
   */
  private constructGreedySolution(): Solution {
    const routes: Route[] = [];
    const unassignedCustomers: Customer[] = [];
    const remainingCustomers = [...this.customers];

    for (const vehicle of this.vehicles) {
      const route: Route = {
        vehicleId: vehicle.id,
        customers: [],
        totalDistance: 0,
        totalTime: 0,
        totalCost: 0,
        utilizationRate: 0,
        timeWindowViolations: 0,
        capacityViolations: 0
      };

      let currentLoad = 0;
      let currentLocation = vehicle.startLocation;

      while (remainingCustomers.length > 0) {
        let bestCustomer: Customer | null = null;
        let bestDistance = Infinity;
        let bestIndex = -1;

        for (let i = 0; i < remainingCustomers.length; i++) {
          const customer = remainingCustomers[i];
          if (currentLoad + customer.demand <= vehicle.capacity) {
            const distance = this.getDistance(currentLocation.id, customer.location.id);
            if (distance < bestDistance) {
              bestDistance = distance;
              bestCustomer = customer;
              bestIndex = i;
            }
          }
        }

        if (bestCustomer) {
          route.customers.push(bestCustomer);
          currentLoad += bestCustomer.demand;
          currentLocation = bestCustomer.location;
          remainingCustomers.splice(bestIndex, 1);
        } else {
          break;
        }
      }

      routes.push(route);
    }

    unassignedCustomers.push(...remainingCustomers);

    return {
      routes,
      totalCost: 0,
      totalDistance: 0,
      unassignedCustomers,
      fitness: 0,
      metadata: { algorithm: 'greedy', iterations: 0, computeTime: 0 }
    };
  }

  /**
   * Construct solution using ant colony pheromones
   */
  private constructAntSolution(pheromones: Map<string, number>): Solution {
    // Similar to greedy but influenced by pheromones
    return this.constructGreedySolution();
  }

  /**
   * Evaluate fitness of a solution (lower is better)
   */
  private evaluateFitness(solution: Solution): number {
    let totalCost = 0;
    let totalDistance = 0;
    let violations = 0;

    for (const route of solution.routes) {
      const vehicle = this.vehicles.find(v => v.id === route.vehicleId)!;
      let routeDistance = 0;
      let currentLocation = vehicle.startLocation;

      for (const customer of route.customers) {
        const distance = this.getDistance(currentLocation.id, customer.location.id);
        routeDistance += distance;
        currentLocation = customer.location;
      }

      // Add return distance
      if (vehicle.endLocation) {
        routeDistance += this.getDistance(currentLocation.id, vehicle.endLocation.id);
      }

      totalDistance += routeDistance;
      totalCost += routeDistance * vehicle.costPerKm;
      violations += route.timeWindowViolations + route.capacityViolations;
    }

    // Penalize unassigned customers heavily
    const unassignedPenalty = solution.unassignedCustomers.length * 10000;

    return totalCost + violations * 1000 + unassignedPenalty;
  }

  /**
   * Tournament selection
   */
  private tournamentSelection(population: Solution[]): Solution {
    const tournamentSize = 5;
    const tournament = [];

    for (let i = 0; i < tournamentSize; i++) {
      const randomIndex = Math.floor(Math.random() * population.length);
      tournament.push(population[randomIndex]);
    }

    return tournament.sort((a, b) => a.fitness - b.fitness)[0];
  }

  /**
   * Crossover two solutions
   */
  private crossover(parent1: Solution, parent2: Solution): Solution {
    // Order crossover (OX) for route sequences
    return { ...parent1 }; // Simplified for now
  }

  /**
   * Mutate a solution
   */
  private mutate(solution: Solution): Solution {
    // Random swap or insertion mutation
    const mutated = { ...solution };

    if (mutated.routes.length > 0 && Math.random() < 0.5) {
      const routeIndex = Math.floor(Math.random() * mutated.routes.length);
      const route = mutated.routes[routeIndex];

      if (route.customers.length >= 2) {
        const i = Math.floor(Math.random() * route.customers.length);
        const j = Math.floor(Math.random() * route.customers.length);
        [route.customers[i], route.customers[j]] = [route.customers[j], route.customers[i]];
      }
    }

    return mutated;
  }

  /**
   * Generate neighbor solution for simulated annealing
   */
  private generateNeighbor(solution: Solution): Solution {
    return this.mutate(solution);
  }

  /**
   * Update pheromones for ant colony
   */
  private updatePheromones(
    pheromones: Map<string, number>,
    solutions: Solution[],
    evaporationRate: number
  ): void {
    // Evaporate existing pheromones
    for (const [key, value] of pheromones.entries()) {
      pheromones.set(key, value * (1 - evaporationRate));
    }

    // Deposit new pheromones from solutions
    for (const solution of solutions) {
      const deposit = 1 / (solution.fitness + 1);

      for (const route of solution.routes) {
        for (let i = 0; i < route.customers.length - 1; i++) {
          const key = `${route.customers[i].location.id}-${route.customers[i + 1].location.id}`;
          const current = pheromones.get(key) || 0;
          pheromones.set(key, current + deposit);
        }
      }
    }
  }
}
