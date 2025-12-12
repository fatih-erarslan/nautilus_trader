/**
 * Dilithium MCP Integration Bridge for Quantum Knowledge System
 *
 * This module provides the integration layer between QKS and dilithium-mcp,
 * bridging quantum knowledge representations with post-quantum cryptography,
 * hyperbolic geometry, thermodynamic state management, and neural plasticity.
 *
 * @module dilithium-bridge
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

/**
 * Post-quantum cryptographic key pair (Dilithium ML-DSA)
 */
export interface DilithiumKeyPair {
  publicKey: string;   // Hex-encoded public key
  secretKey: string;   // Hex-encoded secret key
}

/**
 * Signed message with post-quantum signature
 */
export interface SignedMessage {
  message: string;
  signature: string;   // Hex-encoded Dilithium signature
  publicKey: string;
}

/**
 * Point in H^11 hyperbolic space (Lorentz model)
 */
export interface HyperbolicPoint {
  coordinates: number[];  // 12D Lorentz coordinates [x0, x1, ..., x11]
  curvature: number;      // Hyperbolic curvature (typically -1)
}

/**
 * Hyperbolic distance result
 */
export interface HyperbolicDistance {
  distance: number;       // Hyperbolic distance in H^11
  point1: number[];
  point2: number[];
}

/**
 * Möbius addition result in Poincaré ball
 */
export interface MobiusResult {
  result: number[];       // Resulting vector after Möbius addition
  curvature: number;
}

/**
 * Probabilistic bit sampling result
 */
export interface PBitSample {
  probability: number;    // P(s=+1) = σ((h-bias)/T)
  state: number;          // Sampled state: +1 or -1
  field: number;
  temperature: number;
  bias: number;
}

/**
 * Boltzmann weight calculation
 */
export interface BoltzmannWeight {
  weight: number;         // exp(-E/T)
  energy: number;
  temperature: number;
}

/**
 * STDP weight change calculation
 */
export interface STDPWeightChange {
  deltaWeight: number;    // Weight change: Δw
  deltaT: number;         // Time difference (post - pre) in ms
  type: 'LTP' | 'LTD';   // Long-term potentiation or depression
}

/**
 * Ising model critical temperature
 */
export interface IsingCriticalTemp {
  criticalTemp: number;   // Tc ≈ 2.269 for 2D square lattice (Onsager)
  model: '2D_square_lattice';
}

/**
 * QKS embedding mapped to hyperbolic space
 */
export interface HyperbolicEmbedding {
  euclideanEmbedding: number[];   // Original R^n embedding
  hyperbolicPoint: HyperbolicPoint; // Lifted to H^11
  norm: number;
}

/**
 * Thermodynamic state with dilithium integration
 */
export interface ThermodynamicState {
  temperature: number;
  energy: number;
  entropy: number;
  boltzmannWeight: number;
  decayRate: number;
  timestamp: number;
}

/**
 * Neural plasticity state
 */
export interface PlasticityState {
  synapticWeight: number;
  learningRate: number;
  stdpWindow: number;      // Time window for STDP (ms)
  lastUpdate: number;
  totalChanges: number;
}

/**
 * Quantum consensus state
 */
export interface QuantumConsensus {
  ghzState: number[];      // GHZ state coefficients
  criticalTemp: number;
  magnetization: number;
  isingCoupling: number;
  consensusReached: boolean;
}

/**
 * Specialist communication with security
 */
export interface SecureSpecialistMessage {
  from: string;            // Specialist ID
  to: string[];            // Target specialist IDs
  payload: unknown;        // Message payload
  signature: string;       // Dilithium signature
  publicKey: string;
  timestamp: number;
  verified: boolean;
}

/**
 * Bridge configuration
 */
export interface BridgeConfig {
  mcpServerPath: string;
  enableSecurity: boolean;
  defaultTemperature: number;
  stdpTimeConstant: number;
  hyperbolicCurvature: number;
}

// ============================================================================
// DILITHIUM BRIDGE CLASS
// ============================================================================

/**
 * Main integration bridge between QKS and dilithium-mcp
 */
export class DilithiumBridge {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private keyPair: DilithiumKeyPair | null = null;
  private config: BridgeConfig;
  private connected: boolean = false;

  constructor(config: Partial<BridgeConfig> = {}) {
    this.config = {
      mcpServerPath: '/Volumes/Tengritek/Ashina/HyperPhysics/tools/dilithium-mcp/dist/index.js',
      enableSecurity: true,
      defaultTemperature: 1.0,
      stdpTimeConstant: 20.0,
      hyperbolicCurvature: -1.0,
      ...config,
    };
  }

  // ============================================================================
  // CONNECTION MANAGEMENT
  // ============================================================================

  /**
   * Connect to dilithium-mcp server
   */
  async connect(): Promise<void> {
    if (this.connected) {
      return;
    }

    this.transport = new StdioClientTransport({
      command: 'bun',
      args: ['run', this.config.mcpServerPath],
    });

    this.client = new Client({
      name: 'qks-dilithium-bridge',
      version: '1.0.0',
    }, {
      capabilities: {},
    });

    await this.client.connect(this.transport);
    this.connected = true;

    // Generate key pair for security if enabled
    if (this.config.enableSecurity) {
      await this.generateKeyPair();
    }
  }

  /**
   * Disconnect from dilithium-mcp server
   */
  async disconnect(): Promise<void> {
    if (!this.connected) {
      return;
    }

    await this.client?.close();
    await this.transport?.close();
    this.connected = false;
    this.client = null;
    this.transport = null;
  }

  /**
   * Ensure connection is active
   */
  private ensureConnected(): void {
    if (!this.connected || !this.client) {
      throw new Error('Bridge not connected. Call connect() first.');
    }
  }

  // ============================================================================
  // 1. HYPERBOLIC OPERATIONS
  // ============================================================================

  /**
   * Lift Euclidean embedding to H^11 hyperboloid
   *
   * @param embedding - QKS knowledge embedding (R^n)
   * @returns Hyperbolic point on H^11
   */
  async liftToHyperboloid(embedding: number[]): Promise<HyperbolicEmbedding> {
    this.ensureConnected();

    // Ensure 11-dimensional input (pad or truncate)
    const normalized = this.normalizeToR11(embedding);
    const norm = Math.sqrt(normalized.reduce((sum, x) => sum + x * x, 0));

    const result = await this.client!.callTool('lift_to_hyperboloid', {
      point: normalized,
    });

    const hyperbolicCoords = this.extractToolResult<number[]>(result);

    return {
      euclideanEmbedding: embedding,
      hyperbolicPoint: {
        coordinates: hyperbolicCoords,
        curvature: this.config.hyperbolicCurvature,
      },
      norm,
    };
  }

  /**
   * Compute hyperbolic distance between two QKS embeddings
   *
   * @param embedding1 - First embedding
   * @param embedding2 - Second embedding
   * @returns Hyperbolic distance in H^11
   */
  async computeHyperbolicDistance(
    embedding1: number[],
    embedding2: number[]
  ): Promise<HyperbolicDistance> {
    this.ensureConnected();

    // Lift both embeddings to H^11
    const point1 = await this.liftToHyperboloid(embedding1);
    const point2 = await this.liftToHyperboloid(embedding2);

    const result = await this.client!.callTool('hyperbolic_distance', {
      point1: point1.hyperbolicPoint.coordinates,
      point2: point2.hyperbolicPoint.coordinates,
    });

    const distance = this.extractToolResult<number>(result);

    return {
      distance,
      point1: point1.hyperbolicPoint.coordinates,
      point2: point2.hyperbolicPoint.coordinates,
    };
  }

  /**
   * Perform Möbius addition in Poincaré ball model
   *
   * @param x - First vector in Poincaré ball
   * @param y - Second vector in Poincaré ball
   * @returns Result of Möbius addition
   */
  async mobiusAdd(x: number[], y: number[]): Promise<MobiusResult> {
    this.ensureConnected();

    const result = await this.client!.callTool('mobius_add', {
      x,
      y,
      curvature: this.config.hyperbolicCurvature,
    });

    const resultVector = this.extractToolResult<number[]>(result);

    return {
      result: resultVector,
      curvature: this.config.hyperbolicCurvature,
    };
  }

  /**
   * Compute hyperbolic centroid of multiple embeddings
   *
   * @param embeddings - Array of QKS embeddings
   * @returns Centroid in hyperbolic space
   */
  async computeHyperbolicCentroid(embeddings: number[][]): Promise<HyperbolicPoint> {
    this.ensureConnected();

    if (embeddings.length === 0) {
      throw new Error('Cannot compute centroid of empty embeddings');
    }

    if (embeddings.length === 1) {
      const lifted = await this.liftToHyperboloid(embeddings[0]);
      return lifted.hyperbolicPoint;
    }

    // Use Möbius addition to compute centroid in Poincaré ball
    // Project to Poincaré ball, average, then lift
    const normalized = embeddings.map(e => this.normalizeToR11(e));
    let centroid = normalized[0];

    for (let i = 1; i < normalized.length; i++) {
      const mobiusResult = await this.mobiusAdd(centroid, normalized[i]);
      centroid = mobiusResult.result;
    }

    // Scale by 1/n
    const scale = 1.0 / embeddings.length;
    centroid = centroid.map(x => x * scale);

    const lifted = await this.liftToHyperboloid(centroid);
    return lifted.hyperbolicPoint;
  }

  // ============================================================================
  // 2. THERMODYNAMIC OPERATIONS
  // ============================================================================

  /**
   * Sample probabilistic bit using Boltzmann statistics
   *
   * @param field - Effective field h
   * @param temperature - Temperature T
   * @param bias - Bias term (default: 0)
   * @returns pBit sample result
   */
  async samplePBit(
    field: number,
    temperature: number = this.config.defaultTemperature,
    bias: number = 0
  ): Promise<PBitSample> {
    this.ensureConnected();

    const result = await this.client!.callTool('pbit_sample', {
      field,
      temperature,
      bias,
    });

    const sample = this.extractToolResult<{ probability: number; state: number }>(result);

    return {
      probability: sample.probability,
      state: sample.state,
      field,
      temperature,
      bias,
    };
  }

  /**
   * Compute Boltzmann weight for thermodynamic state
   *
   * @param energy - Energy E
   * @param temperature - Temperature T
   * @returns Boltzmann weight exp(-E/T)
   */
  async computeBoltzmannWeight(
    energy: number,
    temperature: number = this.config.defaultTemperature
  ): Promise<BoltzmannWeight> {
    this.ensureConnected();

    const result = await this.client!.callTool('boltzmann_weight', {
      energy,
      temperature,
    });

    const weight = this.extractToolResult<number>(result);

    return {
      weight,
      energy,
      temperature,
    };
  }

  /**
   * Compute decay rate for QKS knowledge using thermodynamics
   *
   * @param currentEnergy - Current knowledge energy
   * @param temperature - System temperature
   * @returns Thermodynamic state with decay rate
   */
  async computeThermodynamicDecay(
    currentEnergy: number,
    temperature: number = this.config.defaultTemperature
  ): Promise<ThermodynamicState> {
    this.ensureConnected();

    const boltzmann = await this.computeBoltzmannWeight(currentEnergy, temperature);

    // Entropy calculation: S = -k_B * ln(W)
    // For single state: S ≈ -ln(boltzmann.weight)
    const entropy = -Math.log(boltzmann.weight);

    // Decay rate proportional to Boltzmann factor
    const decayRate = 1.0 - boltzmann.weight;

    return {
      temperature,
      energy: currentEnergy,
      entropy,
      boltzmannWeight: boltzmann.weight,
      decayRate,
      timestamp: Date.now(),
    };
  }

  /**
   * Update QKS knowledge strength using thermodynamic annealing
   *
   * @param initialStrength - Initial knowledge strength
   * @param accessCount - Number of accesses (increases energy)
   * @param timeElapsed - Time since last access (ms)
   * @returns Updated strength based on thermodynamics
   */
  async annealKnowledgeStrength(
    initialStrength: number,
    accessCount: number,
    timeElapsed: number
  ): Promise<number> {
    this.ensureConnected();

    // Energy increases with access count
    const energy = -Math.log(initialStrength + 0.01) * (1.0 + accessCount * 0.1);

    // Temperature decreases with time (cooling)
    const temperature = this.config.defaultTemperature * Math.exp(-timeElapsed / 10000);

    const state = await this.computeThermodynamicDecay(energy, temperature);

    // New strength based on Boltzmann weight
    return initialStrength * state.boltzmannWeight;
  }

  // ============================================================================
  // 3. NEURAL PLASTICITY
  // ============================================================================

  /**
   * Compute STDP weight change
   *
   * @param deltaT - Time difference (post - pre) in ms
   * @param aPlus - LTP amplitude (default: 0.1)
   * @param aMinus - LTD amplitude (default: 0.12)
   * @param tau - Time constant in ms (default: 20)
   * @returns STDP weight change
   */
  async computeSTDPWeightChange(
    deltaT: number,
    aPlus: number = 0.1,
    aMinus: number = 0.12,
    tau: number = this.config.stdpTimeConstant
  ): Promise<STDPWeightChange> {
    this.ensureConnected();

    const result = await this.client!.callTool('stdp_weight_change', {
      delta_t: deltaT,
      a_plus: aPlus,
      a_minus: aMinus,
      tau,
    });

    const deltaWeight = this.extractToolResult<number>(result);
    const type = deltaT > 0 ? 'LTP' : 'LTD';

    return {
      deltaWeight,
      deltaT,
      type,
    };
  }

  /**
   * Update specialist learning rate using STDP
   *
   * @param currentRate - Current learning rate
   * @param rewardSignal - Reward/error signal (-1 to 1)
   * @param timeWindow - Time window for plasticity (ms)
   * @returns Updated plasticity state
   */
  async updateSpecialistPlasticity(
    currentRate: number,
    rewardSignal: number,
    timeWindow: number = 100
  ): Promise<PlasticityState> {
    this.ensureConnected();

    // Map reward signal to STDP timing
    const deltaT = rewardSignal * timeWindow;

    const stdp = await this.computeSTDPWeightChange(deltaT);

    const newWeight = Math.max(0, Math.min(1, currentRate + stdp.deltaWeight));

    return {
      synapticWeight: newWeight,
      learningRate: newWeight,
      stdpWindow: timeWindow,
      lastUpdate: Date.now(),
      totalChanges: 1,
    };
  }

  /**
   * Compute synaptic scaling for homeostatic plasticity
   *
   * @param weights - Current synaptic weights
   * @param targetActivity - Target activity level
   * @param currentActivity - Current activity level
   * @returns Scaled weights
   */
  async homeostaticScaling(
    weights: number[],
    targetActivity: number,
    currentActivity: number
  ): Promise<number[]> {
    this.ensureConnected();

    // Multiplicative scaling factor
    const scaleFactor = targetActivity / (currentActivity + 1e-8);

    // Apply scaling with Boltzmann modulation
    const energy = Math.abs(Math.log(scaleFactor));
    const boltzmann = await this.computeBoltzmannWeight(energy);

    return weights.map(w => w * scaleFactor * boltzmann.weight);
  }

  // ============================================================================
  // 4. QUANTUM OPERATIONS
  // ============================================================================

  /**
   * Get Ising model critical temperature
   *
   * @returns Critical temperature (Onsager solution for 2D square lattice)
   */
  async getIsingCriticalTemp(): Promise<IsingCriticalTemp> {
    this.ensureConnected();

    const result = await this.client!.callTool('ising_critical_temp', {});

    const criticalTemp = this.extractToolResult<number>(result);

    return {
      criticalTemp,
      model: '2D_square_lattice',
    };
  }

  /**
   * Compute quantum consensus using Ising model
   *
   * @param ghzState - GHZ state coefficients
   * @param temperature - System temperature
   * @returns Quantum consensus state
   */
  async computeQuantumConsensus(
    ghzState: number[],
    temperature: number = this.config.defaultTemperature
  ): Promise<QuantumConsensus> {
    this.ensureConnected();

    const criticalTemp = await this.getIsingCriticalTemp();

    // Magnetization from GHZ state
    const magnetization = ghzState.reduce((sum, coeff) => sum + coeff, 0) / ghzState.length;

    // Ising coupling strength
    const isingCoupling = magnetization * temperature / criticalTemp.criticalTemp;

    // Consensus reached if magnetization > threshold and T < Tc
    const consensusReached =
      Math.abs(magnetization) > 0.8 &&
      temperature < criticalTemp.criticalTemp;

    return {
      ghzState,
      criticalTemp: criticalTemp.criticalTemp,
      magnetization,
      isingCoupling,
      consensusReached,
    };
  }

  /**
   * Sample specialist agreement using pBit statistics
   *
   * @param specialistVotes - Array of specialist votes (-1 or +1)
   * @param temperature - Decision temperature
   * @returns Consensus probability and state
   */
  async sampleSpecialistConsensus(
    specialistVotes: number[],
    temperature: number = this.config.defaultTemperature
  ): Promise<PBitSample> {
    this.ensureConnected();

    // Effective field from majority vote
    const field = specialistVotes.reduce((sum, vote) => sum + vote, 0);

    return await this.samplePBit(field, temperature);
  }

  // ============================================================================
  // 5. SECURITY
  // ============================================================================

  /**
   * Generate Dilithium key pair
   */
  async generateKeyPair(): Promise<DilithiumKeyPair> {
    this.ensureConnected();

    const result = await this.client!.callTool('dilithium_keygen', {});

    const keyPair = this.extractToolResult<DilithiumKeyPair>(result);
    this.keyPair = keyPair;

    return keyPair;
  }

  /**
   * Sign message with Dilithium
   *
   * @param message - Message to sign
   * @returns Signed message
   */
  async signMessage(message: string): Promise<SignedMessage> {
    this.ensureConnected();

    if (!this.keyPair) {
      throw new Error('No key pair available. Generate key pair first.');
    }

    const result = await this.client!.callTool('dilithium_sign', {
      message,
      secret_key: this.keyPair.secretKey,
    });

    const signature = this.extractToolResult<string>(result);

    return {
      message,
      signature,
      publicKey: this.keyPair.publicKey,
    };
  }

  /**
   * Verify Dilithium signature
   *
   * @param signedMessage - Signed message to verify
   * @returns True if signature is valid
   */
  async verifySignature(signedMessage: SignedMessage): Promise<boolean> {
    this.ensureConnected();

    const result = await this.client!.callTool('dilithium_verify', {
      message: signedMessage.message,
      signature: signedMessage.signature,
      public_key: signedMessage.publicKey,
    });

    return this.extractToolResult<boolean>(result);
  }

  /**
   * Send authenticated specialist broadcast
   *
   * @param from - Sender specialist ID
   * @param to - Target specialist IDs
   * @param payload - Message payload
   * @returns Secure message
   */
  async sendSecureSpecialistMessage(
    from: string,
    to: string[],
    payload: unknown
  ): Promise<SecureSpecialistMessage> {
    this.ensureConnected();

    if (!this.config.enableSecurity) {
      return {
        from,
        to,
        payload,
        signature: '',
        publicKey: '',
        timestamp: Date.now(),
        verified: false,
      };
    }

    const message = JSON.stringify({ from, to, payload, timestamp: Date.now() });
    const signed = await this.signMessage(message);

    return {
      from,
      to,
      payload,
      signature: signed.signature,
      publicKey: signed.publicKey,
      timestamp: Date.now(),
      verified: true,
    };
  }

  /**
   * Verify specialist message authenticity
   *
   * @param message - Secure specialist message
   * @returns True if message is authentic
   */
  async verifySpecialistMessage(message: SecureSpecialistMessage): Promise<boolean> {
    this.ensureConnected();

    if (!this.config.enableSecurity) {
      return true;
    }

    const messageContent = JSON.stringify({
      from: message.from,
      to: message.to,
      payload: message.payload,
      timestamp: message.timestamp,
    });

    return await this.verifySignature({
      message: messageContent,
      signature: message.signature,
      publicKey: message.publicKey,
    });
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Normalize embedding to R^11
   */
  private normalizeToR11(embedding: number[]): number[] {
    if (embedding.length === 11) {
      return embedding;
    }

    if (embedding.length > 11) {
      // Truncate and normalize
      return embedding.slice(0, 11);
    }

    // Pad with zeros
    return [...embedding, ...new Array(11 - embedding.length).fill(0)];
  }

  /**
   * Extract result from MCP tool response
   */
  private extractToolResult<T>(result: unknown): T {
    if (typeof result === 'object' && result !== null && 'content' in result) {
      const content = (result as { content: Array<{ text?: string }> }).content;
      if (content && content.length > 0 && content[0].text) {
        return JSON.parse(content[0].text) as T;
      }
    }
    return result as T;
  }

  /**
   * Get current connection status
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get current key pair
   */
  getKeyPair(): DilithiumKeyPair | null {
    return this.keyPair;
  }

  /**
   * Get bridge configuration
   */
  getConfig(): BridgeConfig {
    return { ...this.config };
  }
}

// ============================================================================
// FACTORY FUNCTION
// ============================================================================

/**
 * Create and connect a DilithiumBridge instance
 *
 * @param config - Bridge configuration
 * @returns Connected bridge instance
 */
export async function createDilithiumBridge(
  config?: Partial<BridgeConfig>
): Promise<DilithiumBridge> {
  const bridge = new DilithiumBridge(config);
  await bridge.connect();
  return bridge;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default DilithiumBridge;
