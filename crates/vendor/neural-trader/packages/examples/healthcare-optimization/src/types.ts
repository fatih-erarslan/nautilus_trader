/**
 * Healthcare Optimization Types
 *
 * Type definitions for patient flow, scheduling, and optimization
 */

// Patient and arrival types
export interface Patient {
  id: string;
  arrivalTime: Date;
  acuity: 1 | 2 | 3 | 4 | 5; // 1=critical, 5=non-urgent
  chiefComplaint: string;
  estimatedServiceTime: number; // minutes
  requiredSkills: string[];
}

export interface ArrivalPattern {
  hourOfDay: number;
  dayOfWeek: number;
  month: number;
  expectedArrivals: number;
  variance: number;
  seasonalFactor: number;
}

export interface ForecastResult {
  timestamp: Date;
  predictedArrivals: number;
  lowerBound: number;
  upperBound: number;
  confidence: number;
  seasonalComponent: number;
  trendComponent: number;
}

// Staff and scheduling types
export interface StaffMember {
  id: string;
  name: string;
  role: 'physician' | 'nurse' | 'technician' | 'specialist';
  skills: string[];
  shiftPreference: 'day' | 'evening' | 'night' | 'any';
  maxHoursPerWeek: number;
  costPerHour: number;
}

export interface Shift {
  id: string;
  staffId: string;
  start: Date;
  end: Date;
  role: string;
  assignedArea: 'ed' | 'or' | 'icu' | 'ward';
}

export interface ScheduleConstraints {
  minStaffPerShift: Record<string, number>; // role -> count
  maxConsecutiveHours: number;
  minRestBetweenShifts: number;
  requiredSkillCoverage: string[];
}

export interface ScheduleSolution {
  shifts: Shift[];
  totalCost: number;
  coverageScore: number; // 0-1
  fairnessScore: number; // 0-1
  constraintViolations: string[];
}

// Queue and flow types
export interface QueueMetrics {
  timestamp: Date;
  queueLength: number;
  averageWaitTime: number;
  maxWaitTime: number;
  throughput: number; // patients per hour
  utilization: number; // 0-1
  abandonmentRate: number;
}

export interface ResourcePool {
  id: string;
  type: 'exam_room' | 'or' | 'icu_bed' | 'imaging';
  capacity: number;
  available: number;
  utilizationTarget: number; // optimal 0-1
}

export interface QueueState {
  waitingPatients: Patient[];
  inServicePatients: Patient[];
  resources: ResourcePool[];
  metrics: QueueMetrics;
}

// Optimization types
export interface OptimizationObjective {
  minimizeWaitTime: number; // weight
  maximizeUtilization: number; // weight
  minimizeCost: number; // weight
  maximizePatientOutcomes: number; // weight
}

export interface OptimizationResult {
  schedule: ScheduleSolution;
  expectedWaitTime: number;
  expectedUtilization: number;
  totalCost: number;
  qualityScore: number;
  simulationRuns: number;
}

// Swarm types
export interface SwarmAgent {
  id: string;
  role: 'explorer' | 'exploiter' | 'evaluator';
  currentSolution: ScheduleSolution;
  fitness: number;
  explorationRadius: number;
}

export interface SwarmConfig {
  populationSize: number;
  maxIterations: number;
  explorationRate: number;
  convergenceThreshold: number;
  elitismRate: number;
}

export interface SwarmResult {
  bestSolution: ScheduleSolution;
  convergenceHistory: number[];
  iterations: number;
  exploredSolutions: number;
}

// Memory and learning types
export interface HospitalMemory {
  historicalPatterns: ArrivalPattern[];
  successfulSchedules: ScheduleSolution[];
  performanceMetrics: QueueMetrics[];
  learnedHeuristics: Record<string, number>;
  seasonalModels: Record<string, any>;
}

export interface LearningMetrics {
  forecastAccuracy: number;
  scheduleQuality: number;
  adaptationRate: number;
  patternsLearned: number;
}

// Triage and emergency types
export interface TriageDecision {
  patientId: string;
  assignedAcuity: 1 | 2 | 3 | 4 | 5;
  recommendedPath: 'immediate' | 'urgent' | 'standard' | 'fast_track';
  reasoning: string;
  confidence: number;
}

export interface EmergencyFlowState {
  triageQueue: Patient[];
  treatmentAreas: Record<string, Patient[]>;
  availableStaff: StaffMember[];
  criticalAlerts: string[];
}

// Privacy and compliance
export interface PrivacyConfig {
  useSyntheticData: boolean;
  anonymizationLevel: 'none' | 'basic' | 'full';
  dataRetentionDays: number;
  complianceMode: 'hipaa' | 'gdpr' | 'both';
}

// Configuration
export interface HealthcareOptimizationConfig {
  openRouterApiKey?: string;
  openRouterModel?: string;
  agentdbPath: string;
  enableNapiRS: boolean;
  privacy: PrivacyConfig;
  optimization: OptimizationObjective;
  swarm: SwarmConfig;
  constraints: ScheduleConstraints;
}
