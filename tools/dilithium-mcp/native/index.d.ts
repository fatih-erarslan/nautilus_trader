/**
 * Dilithium Native - Post-Quantum Secure MCP Infrastructure
 *
 * Native Rust bindings providing:
 * - Post-quantum cryptographic authentication (Dilithium ML-DSA)
 * - Hyperbolic geometry computations
 * - pBit dynamics engine
 * - Symbolic mathematics
 * - HyperPhysics Agency functions
 */

// Dilithium Cryptography
export interface KeyPairResult {
  publicKey: string;
  secretKey: string;
}

export function dilithiumKeygen(): KeyPairResult;
export function dilithiumSign(secretKeyHex: string, message: string): string;
export function dilithiumVerify(publicKeyHex: string, signatureHex: string, message: string): boolean;
export function blake3Hash(data: string): string;
export function generateNonce(): string;

// MCP Authentication
export interface AuthenticatedRequest {
  clientId: string;
  timestamp: string;
  nonce: string;
  payload: string;
  signature: string;
}

export interface AuthResult {
  valid: boolean;
  clientId: string;
  error?: string;
  timestamp: string;
}

export function initServer(): string;
export function registerClient(clientId: string, publicKey: string, capabilities: string[]): boolean;
export function verifyRequest(request: AuthenticatedRequest): AuthResult;

// Hyperbolic Geometry
export function lorentzInner(x: number[], y: number[]): number;
export function hyperbolicDistance(x: number[], y: number[]): number;
export function liftToHyperboloid(z: number[]): number[];
export function mobiusAdd(x: number[], y: number[], curvature: number): number[];

// pBit Dynamics
export function pbitProbability(field: number, bias: number, temperature: number): number;
export function pbitProbabilitiesBatch(fields: number[], biases: number[], temperature: number): number[];
export function boltzmannWeight(energy: number, temperature: number): number;
export function isingCriticalTemp(): number;
export function stdpWeightChange(deltaT: number, aPlus: number, aMinus: number, tau: number): number;

// Mathematical Utilities
export function fastExp(x: number): number;
export function stableAcosh(x: number): number;

// HyperPhysics Agency
export interface AgencyResult {
  success: boolean;
  agentId?: string;
  error?: string;
  data?: string;
}

export function agencyCreateAgent(configJson: string): AgencyResult;
export function agencyAgentStep(agentId: string, observationJson: string): AgencyResult;
export function agencyComputeFreeEnergy(observation: number[], beliefs: number[], precision: number[]): AgencyResult;
export function agencyComputeSurvivalDrive(freeEnergy: number, position: number[]): AgencyResult;
export function agencyComputePhi(networkState: number[]): AgencyResult;
export function agencyAnalyzeCriticality(timeseries: number[]): AgencyResult;
export function agencyRegulateHomeostasis(currentStateJson: string, setpointsJson?: string): AgencyResult;
