/**
 * Thermodynamic Layer Handlers - Energy Management
 *
 * Implements Layer 1 of the cognitive architecture:
 * - Free Energy Principle (Friston)
 * - Survival drive computation
 * - Homeostatic regulation
 * - Metabolic cost tracking
 */

import { QKSBridge } from './mod.js';

export interface HomeostasisState {
  energy: { current: number; setpoint: number; error: number };
  temperature: { current: number; setpoint: number; error: number };
  activity: { current: number; setpoint: number; error: number };
  coherence: { current: number; setpoint: number; error: number };
  criticality: { current: number; setpoint: number; error: number };
  integration: { current: number; setpoint: number; error: number };
}

export class ThermodynamicHandlers {
  private bridge: QKSBridge;

  constructor(bridge: QKSBridge) {
    this.bridge = bridge;
  }

  /**
   * Compute variational free energy F = Complexity - Accuracy
   * Based on Karl Friston's Free Energy Principle
   */
  async computeFreeEnergy(params: {
    observation: number[];
    beliefs: number[];
    precision: number[];
  }): Promise<{
    free_energy: number;
    complexity: number;
    accuracy: number;
    valid: boolean;
  }> {
    const { observation, beliefs, precision } = params;

    if (observation.length !== beliefs.length || observation.length !== precision.length) {
      throw new Error('Observation, beliefs, and precision must have same length');
    }

    try {
      // Try Rust core first for performance
      return await this.bridge.callRust('thermodynamic.free_energy', {
        observation,
        beliefs,
        precision,
      });
    } catch (e) {
      // Fallback to TypeScript implementation
      return this.computeFreeEnergyFallback(observation, beliefs, precision);
    }
  }

  /**
   * Compute survival drive from free energy and hyperbolic position
   * Drive increases with high free energy (danger) and distance from safe region
   */
  async computeSurvivalDrive(params: {
    free_energy: number;
    position: number[];
    strength?: number;
  }): Promise<{
    survival_drive: number;
    threat_level: string;
    homeostatic_status: string;
    hyperbolic_distance: number;
    crisis: boolean;
  }> {
    const { free_energy, position, strength = 1.0 } = params;

    if (position.length !== 12) {
      throw new Error('Position must be 12D Lorentz coordinates for H^11');
    }

    try {
      return await this.bridge.callRust('thermodynamic.survival_drive', {
        free_energy,
        position,
        strength,
      });
    } catch (e) {
      return this.computeSurvivalDriveFallback(free_energy, position, strength);
    }
  }

  /**
   * Assess threat across multiple dimensions
   */
  async assessThreat(params: {
    free_energy: number;
    free_energy_history?: number[];
    position: number[];
    prediction_errors?: number[];
  }): Promise<{
    overall_threat: number;
    components: {
      free_energy_gradient: number;
      hyperbolic_distance: number;
      prediction_volatility: number;
      environmental_volatility: number;
    };
    threat_level: string;
  }> {
    const {
      free_energy,
      free_energy_history = [],
      position,
      prediction_errors = [],
    } = params;

    try {
      return await this.bridge.callRust('thermodynamic.assess_threat', {
        free_energy,
        free_energy_history,
        position,
        prediction_errors,
      });
    } catch (e) {
      // Fallback computation
      const feGradient = free_energy_history.length > 1
        ? free_energy_history[free_energy_history.length - 1] - free_energy_history[0]
        : 0;

      const hyperbolicDistance = this.computeHyperbolicDistance(position);

      const predictionVolatility = prediction_errors.length > 0
        ? this.computeStdDev(prediction_errors)
        : 0;

      const environmentalVolatility = free_energy_history.length > 0
        ? this.computeStdDev(free_energy_history)
        : 0;

      const overallThreat = 0.3 * Math.tanh(feGradient) +
                            0.25 * Math.tanh(hyperbolicDistance) +
                            0.25 * Math.tanh(predictionVolatility) +
                            0.2 * Math.tanh(environmentalVolatility);

      return {
        overall_threat: Math.max(0, Math.min(1, overallThreat)),
        components: {
          free_energy_gradient: feGradient,
          hyperbolic_distance: hyperbolicDistance,
          prediction_volatility: predictionVolatility,
          environmental_volatility: environmentalVolatility,
        },
        threat_level: overallThreat > 0.7 ? 'critical' : overallThreat > 0.4 ? 'elevated' : 'nominal',
      };
    }
  }

  /**
   * Regulate homeostasis using PID control + allostatic prediction
   */
  async regulateHomeostasis(params: {
    current_state: {
      phi: number;
      free_energy: number;
      survival: number;
    };
    setpoints?: {
      phi_optimal?: number;
      free_energy_optimal?: number;
      survival_optimal?: number;
    };
    sensors?: number[];
  }): Promise<{
    control_signals: {
      phi_adjustment: number;
      free_energy_adjustment: number;
      survival_adjustment: number;
    };
    errors: {
      phi_error: number;
      free_energy_error: number;
      survival_error: number;
    };
    homeostatic_status: string;
  }> {
    const { current_state, setpoints = {}, sensors = [] } = params;

    const phiOptimal = setpoints.phi_optimal ?? 1.0;
    const feOptimal = setpoints.free_energy_optimal ?? 1.0;
    const survivalOptimal = setpoints.survival_optimal ?? 0.5;

    try {
      return await this.bridge.callRust('thermodynamic.regulate_homeostasis', {
        current_state,
        setpoints: {
          phi_optimal: phiOptimal,
          free_energy_optimal: feOptimal,
          survival_optimal: survivalOptimal,
        },
        sensors,
      });
    } catch (e) {
      // PID control fallback
      const Kp = 0.5;
      const phiError = phiOptimal - current_state.phi;
      const feError = feOptimal - current_state.free_energy;
      const survivalError = survivalOptimal - current_state.survival;

      const phiAdjustment = Kp * phiError;
      const feAdjustment = Kp * feError;
      const survivalAdjustment = Kp * survivalError;

      // Allostatic bias from sensors
      let allostaticBias = 0;
      if (sensors.length > 0) {
        const sensorMean = sensors.reduce((a, b) => a + b, 0) / sensors.length;
        allostaticBias = (sensorMean - 0.5) * 0.1;
      }

      return {
        control_signals: {
          phi_adjustment: Math.max(-1, Math.min(1, phiAdjustment + allostaticBias)),
          free_energy_adjustment: Math.max(-1, Math.min(1, feAdjustment)),
          survival_adjustment: Math.max(-1, Math.min(1, survivalAdjustment)),
        },
        errors: {
          phi_error: phiError,
          free_energy_error: feError,
          survival_error: survivalError,
        },
        homeostatic_status: Math.abs(phiError) < 0.1 && Math.abs(feError) < 0.2 ? 'stable' : 'regulating',
      };
    }
  }

  /**
   * Track metabolic cost of cognitive operations
   */
  async trackMetabolicCost(params: {
    operation: string;
    duration_ms: number;
    layer_activations: number[];
  }): Promise<{
    energy_consumed: number;
    efficiency: number;
    cost_breakdown: Record<string, number>;
  }> {
    try {
      return await this.bridge.callRust('thermodynamic.metabolic_cost', params);
    } catch (e) {
      // Simplified cost model
      const baseCost = 0.1;
      const durationCost = params.duration_ms * 0.001;
      const activationCost = params.layer_activations.reduce((a, b) => a + Math.abs(b), 0) * 0.01;

      const totalCost = baseCost + durationCost + activationCost;
      const efficiency = 1.0 / (1.0 + totalCost);

      return {
        energy_consumed: totalCost,
        efficiency,
        cost_breakdown: {
          base: baseCost,
          duration: durationCost,
          activation: activationCost,
        },
      };
    }
  }

  // ===== Private Fallback Methods =====

  private computeFreeEnergyFallback(
    observation: number[],
    beliefs: number[],
    precision: number[]
  ): {
    free_energy: number;
    complexity: number;
    accuracy: number;
    valid: boolean;
  } {
    const n = observation.length;
    const epsilon = 1e-10;

    // Normalize to probabilities
    const beliefsSum = beliefs.reduce((a, b) => a + Math.abs(b), 0);
    const obsSum = observation.reduce((a, b) => a + Math.abs(b), 0);

    const normalizedBeliefs = beliefs.map(b => Math.abs(b) / (beliefsSum + epsilon));
    const normalizedObs = observation.map(o => Math.abs(o) / (obsSum + epsilon));

    // Complexity: KL divergence
    let complexity = 0;
    for (let i = 0; i < n; i++) {
      if (normalizedBeliefs[i] > epsilon && normalizedObs[i] > epsilon) {
        complexity += normalizedBeliefs[i] * Math.log(normalizedBeliefs[i] / normalizedObs[i]);
      }
    }

    // Accuracy: Expected log likelihood
    let accuracy = 0;
    for (let i = 0; i < n; i++) {
      const error = observation[i] - beliefs[i];
      accuracy -= 0.5 * error * error * precision[i];
    }

    const freeEnergy = complexity - accuracy;

    return {
      free_energy: isFinite(freeEnergy) ? freeEnergy : 1.0,
      complexity: isFinite(complexity) ? complexity : 0.0,
      accuracy: isFinite(accuracy) ? accuracy : 0.0,
      valid: isFinite(freeEnergy),
    };
  }

  private computeSurvivalDriveFallback(
    free_energy: number,
    position: number[],
    strength: number
  ): {
    survival_drive: number;
    threat_level: string;
    homeostatic_status: string;
    hyperbolic_distance: number;
    crisis: boolean;
  } {
    const hyperbolicDist = this.computeHyperbolicDistance(position);

    const optimalFE = 1.0;
    const feComponent = 1 / (1 + Math.exp(-(free_energy - optimalFE)));
    const distComponent = Math.tanh(1.5 * hyperbolicDist);

    const drive = strength * (0.7 * feComponent + 0.3 * distComponent);
    const clampedDrive = Math.max(0, Math.min(1, drive));

    return {
      survival_drive: clampedDrive,
      threat_level: clampedDrive > 0.7 ? 'danger' : clampedDrive > 0.3 ? 'caution' : 'safe',
      homeostatic_status: clampedDrive < 0.8 ? 'stable' : 'critical',
      hyperbolic_distance: hyperbolicDist,
      crisis: clampedDrive > 0.8,
    };
  }

  private computeHyperbolicDistance(position: number[]): number {
    // Lorentz distance: d = acosh(-⟨p,q⟩_L) where ⟨p,q⟩_L = -t₁t₂ + x₁·x₂
    const origin = [1.0, ...Array(11).fill(0)];
    const lorentzInner = -position[0] * origin[0] +
                         position.slice(1).reduce((s, x, i) => s + x * origin[i + 1], 0);
    return Math.acosh(Math.max(-lorentzInner, 1.0));
  }

  private computeStdDev(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }
}
