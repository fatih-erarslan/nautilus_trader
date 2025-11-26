/**
 * Energy Grid Optimization Types
 * Comprehensive type definitions for grid operations, forecasting, and scheduling
 */

/**
 * Time horizon for load forecasting
 */
export enum ForecastHorizon {
  /** 5-minute intervals for real-time dispatch */
  MINUTES_5 = '5min',
  /** 15-minute intervals for intra-hour balancing */
  MINUTES_15 = '15min',
  /** 1-hour intervals for hourly commitment */
  HOUR_1 = '1h',
  /** 4-hour intervals for short-term planning */
  HOUR_4 = '4h',
  /** 24-hour intervals for day-ahead scheduling */
  HOUR_24 = '24h',
  /** 168-hour (1 week) for weekly planning */
  HOUR_168 = '168h',
}

/**
 * Load forecast with prediction intervals
 */
export interface LoadForecast {
  /** Forecast timestamp */
  timestamp: Date;
  /** Predicted load in MW */
  loadMW: number;
  /** Lower bound of prediction interval (MW) */
  lowerBound: number;
  /** Upper bound of prediction interval (MW) */
  upperBound: number;
  /** Confidence level (e.g., 0.95 for 95%) */
  confidence: number;
  /** Forecast horizon */
  horizon: ForecastHorizon;
  /** Forecast error metrics */
  metrics?: {
    mae?: number;
    rmse?: number;
    mape?: number;
  };
}

/**
 * Generator unit types
 */
export enum GeneratorType {
  COAL = 'coal',
  NATURAL_GAS = 'natural_gas',
  NUCLEAR = 'nuclear',
  HYDRO = 'hydro',
  WIND = 'wind',
  SOLAR = 'solar',
  BATTERY = 'battery',
  DIESEL = 'diesel',
}

/**
 * Generator unit operational constraints
 */
export interface GeneratorUnit {
  /** Unique generator ID */
  id: string;
  /** Generator type */
  type: GeneratorType;
  /** Minimum output capacity (MW) */
  minCapacityMW: number;
  /** Maximum output capacity (MW) */
  maxCapacityMW: number;
  /** Ramp-up rate (MW/hour) */
  rampUpRate: number;
  /** Ramp-down rate (MW/hour) */
  rampDownRate: number;
  /** Minimum up time (hours) */
  minUpTime: number;
  /** Minimum down time (hours) */
  minDownTime: number;
  /** Startup cost ($) */
  startupCost: number;
  /** Shutdown cost ($) */
  shutdownCost: number;
  /** Variable cost ($/MWh) */
  variableCost: number;
  /** Fixed cost ($/hour when online) */
  fixedCost: number;
  /** Startup time (hours) */
  startupTime: number;
  /** Current operational status */
  status: {
    isOnline: boolean;
    currentOutputMW: number;
    hoursOnline: number;
    hoursOffline: number;
  };
}

/**
 * Renewable energy forecast
 */
export interface RenewableForecast {
  /** Forecast timestamp */
  timestamp: Date;
  /** Generator ID */
  generatorId: string;
  /** Expected output (MW) */
  expectedOutputMW: number;
  /** Forecast uncertainty (standard deviation) */
  uncertaintyMW: number;
  /** Weather-dependent factors */
  weatherFactors?: {
    windSpeed?: number;
    solarIrradiance?: number;
    cloudCover?: number;
    temperature?: number;
  };
}

/**
 * Battery storage system
 */
export interface BatteryStorage {
  /** Unique battery ID */
  id: string;
  /** Maximum charge/discharge rate (MW) */
  maxPowerMW: number;
  /** Total energy capacity (MWh) */
  capacityMWh: number;
  /** Current state of charge (MWh) */
  currentChargeMWh: number;
  /** Charging efficiency (0-1) */
  chargeEfficiency: number;
  /** Discharging efficiency (0-1) */
  dischargeEfficiency: number;
  /** Minimum state of charge (MWh) */
  minChargeMWh: number;
  /** Maximum state of charge (MWh) */
  maxChargeMWh: number;
  /** Degradation rate per cycle */
  degradationRate: number;
}

/**
 * Unit commitment solution for a single time period
 */
export interface UnitCommitment {
  /** Time period timestamp */
  timestamp: Date;
  /** Generator commitments */
  commitments: Array<{
    generatorId: string;
    isCommitted: boolean;
    outputMW: number;
    startingUp: boolean;
    shuttingDown: boolean;
  }>;
  /** Battery operations */
  batteryOperations: Array<{
    batteryId: string;
    chargeMW: number;
    dischargeMW: number;
    stateOfChargeMWh: number;
  }>;
  /** Total generation (MW) */
  totalGenerationMW: number;
  /** Total load (MW) */
  totalLoadMW: number;
  /** Spinning reserve (MW) */
  spinningReserveMW: number;
  /** Total cost ($) */
  totalCost: number;
  /** Feasibility status */
  isFeasible: boolean;
  /** Constraint violations */
  violations?: string[];
}

/**
 * Demand response program
 */
export interface DemandResponseProgram {
  /** Program ID */
  id: string;
  /** Available load reduction (MW) */
  availableReductionMW: number;
  /** Cost per MWh of reduction */
  costPerMWh: number;
  /** Maximum duration (hours) */
  maxDurationHours: number;
  /** Minimum notice time (hours) */
  minNoticeHours: number;
  /** Customer segment */
  segment: 'residential' | 'commercial' | 'industrial';
}

/**
 * Grid state for self-learning pattern recognition
 */
export interface GridState {
  /** Timestamp */
  timestamp: Date;
  /** Current load (MW) */
  loadMW: number;
  /** Total generation (MW) */
  generationMW: number;
  /** Renewable penetration (%) */
  renewablePenetration: number;
  /** Grid frequency (Hz) */
  frequency: number;
  /** Voltage stability indicator */
  voltageStability: number;
  /** Active generators */
  activeGenerators: string[];
  /** Weather conditions */
  weather?: {
    temperature: number;
    windSpeed: number;
    solarIrradiance: number;
    precipitation: number;
  };
  /** Day of week (0-6) */
  dayOfWeek: number;
  /** Hour of day (0-23) */
  hourOfDay: number;
  /** Is holiday */
  isHoliday: boolean;
}

/**
 * Swarm scheduling strategy
 */
export interface SchedulingStrategy {
  /** Strategy ID */
  id: string;
  /** Strategy name */
  name: string;
  /** Objective weights */
  objectives: {
    /** Minimize cost */
    cost: number;
    /** Maximize renewable usage */
    renewable: number;
    /** Minimize emissions */
    emissions: number;
    /** Maximize reliability */
    reliability: number;
  };
  /** Constraint penalties */
  penalties: {
    /** Load balance violation */
    loadBalance: number;
    /** Reserve shortage */
    reserve: number;
    /** Ramp rate violation */
    ramp: number;
  };
  /** Performance metrics */
  performance?: {
    averageCost: number;
    renewableUtilization: number;
    feasibilityRate: number;
    computeTimeMs: number;
  };
}

/**
 * Optimization result from swarm scheduler
 */
export interface OptimizationResult {
  /** Schedule ID */
  scheduleId: string;
  /** Strategy used */
  strategy: SchedulingStrategy;
  /** Time horizon commitments */
  commitments: UnitCommitment[];
  /** Total cost over horizon */
  totalCost: number;
  /** Average renewable utilization */
  renewableUtilization: number;
  /** Total emissions (tons CO2) */
  totalEmissions: number;
  /** Feasibility status */
  isFeasible: boolean;
  /** Computation time (ms) */
  computeTimeMs: number;
  /** Solution quality score */
  qualityScore: number;
}

/**
 * Self-learning forecast error correction
 */
export interface ForecastErrorCorrection {
  /** Hour of day patterns */
  hourlyBias: number[];
  /** Day of week patterns */
  dailyBias: number[];
  /** Weather-based corrections */
  weatherCorrections: Map<string, number>;
  /** Recent error statistics */
  recentErrors: {
    mean: number;
    stdDev: number;
    mae: number;
    mape: number;
  };
  /** Last update timestamp */
  lastUpdate: Date;
}

/**
 * Configuration for load forecaster
 */
export interface LoadForecasterConfig {
  /** Horizons to forecast */
  horizons: ForecastHorizon[];
  /** Historical data window (days) */
  historyWindowDays: number;
  /** Confidence level for prediction intervals */
  confidenceLevel: number;
  /** Enable self-learning error correction */
  enableErrorCorrection: boolean;
  /** Update frequency for corrections (hours) */
  correctionUpdateFrequency: number;
  /** AgentDB namespace for memory */
  memoryNamespace: string;
}

/**
 * Configuration for unit commitment optimizer
 */
export interface UnitCommitmentConfig {
  /** Planning horizon (hours) */
  planningHorizonHours: number;
  /** Time step resolution (minutes) */
  timeStepMinutes: number;
  /** Required spinning reserve (% of load) */
  reserveMarginPercent: number;
  /** Maximum computation time (ms) */
  maxComputeTimeMs: number;
  /** Sublinear solver tolerance */
  solverTolerance: number;
  /** Enable battery optimization */
  enableBatteryOptimization: boolean;
}

/**
 * Configuration for swarm scheduler
 */
export interface SwarmSchedulerConfig {
  /** Number of parallel strategies */
  swarmSize: number;
  /** Strategy exploration vs exploitation (0-1) */
  explorationRate: number;
  /** Number of iterations */
  maxIterations: number;
  /** Convergence threshold */
  convergenceThreshold: number;
  /** Enable OpenRouter for strategy generation */
  enableOpenRouter: boolean;
  /** OpenRouter API key */
  openRouterApiKey?: string;
  /** Memory namespace for learned strategies */
  memoryNamespace: string;
}
