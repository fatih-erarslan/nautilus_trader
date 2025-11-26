/**
 * Core types for logistics optimization
 */

export interface Location {
  id: string;
  lat: number;
  lng: number;
  name?: string;
}

export interface TimeWindow {
  start: number; // Unix timestamp
  end: number;   // Unix timestamp
}

export interface Customer {
  id: string;
  location: Location;
  demand: number;
  timeWindow: TimeWindow;
  serviceTime: number; // Minutes
  priority: number;    // 1-10
}

export interface Vehicle {
  id: string;
  capacity: number;
  startLocation: Location;
  endLocation?: Location;
  availableTimeWindow: TimeWindow;
  costPerKm: number;
  costPerHour: number;
  maxWorkingHours: number;
}

export interface Route {
  vehicleId: string;
  customers: Customer[];
  totalDistance: number;
  totalTime: number;
  totalCost: number;
  utilizationRate: number;
  timeWindowViolations: number;
  capacityViolations: number;
}

export interface Solution {
  routes: Route[];
  totalCost: number;
  totalDistance: number;
  unassignedCustomers: Customer[];
  fitness: number;
  metadata: {
    algorithm: string;
    iterations: number;
    computeTime: number;
    agentId?: string;
  };
}

export interface TrafficPattern {
  fromLocationId: string;
  toLocationId: string;
  timeOfDay: number; // Hour of day (0-23)
  dayOfWeek: number; // 0-6
  avgSpeed: number;  // km/h
  reliability: number; // 0-1
}

export interface OptimizationConfig {
  algorithm: 'genetic' | 'simulated-annealing' | 'ant-colony' | 'hybrid';
  maxIterations: number;
  populationSize?: number;
  mutationRate?: number;
  crossoverRate?: number;
  temperature?: number;
  coolingRate?: number;
  pheromoneEvaporation?: number;
  parallelAgents?: number;
}

export interface SwarmConfig {
  numAgents: number;
  topology: 'mesh' | 'hierarchical' | 'ring';
  communicationStrategy: 'broadcast' | 'best-solution' | 'diversity';
  convergenceCriteria: {
    maxIterations: number;
    fitnessThreshold?: number;
    noImprovementSteps?: number;
  };
}

export interface LearningMetrics {
  episodeId: string;
  timestamp: number;
  solutionQuality: number;
  computeTime: number;
  customersServed: number;
  trafficPredictionAccuracy?: number;
  routeAdherence?: number;
}
