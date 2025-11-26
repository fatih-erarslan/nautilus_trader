/**
 * Variational Quantum Eigensolver (VQE)
 *
 * Hybrid quantum-classical algorithm for finding ground state energies
 * and solving optimization problems through Hamiltonian minimization.
 *
 * Applications: Molecular simulation, Portfolio optimization, Machine learning
 */

import { create, all } from 'mathjs';
import type { MathJsStatic } from 'mathjs';

const math = create(all) as MathJsStatic;

export interface VQEConfig {
  numQubits: number;
  ansatzType: 'hardware-efficient' | 'uccsd' | 'custom';
  ansatzDepth: number;
  maxIterations: number;
  optimizer: 'gradient-descent' | 'adam' | 'cobyla';
  learningRate: number;
  tolerance: number;
}

export interface Hamiltonian {
  pauliStrings: PauliString[];  // e.g., [{pauli: 'XYZZ', coeff: 0.5}, ...]
}

export interface PauliString {
  pauli: string;      // String of I, X, Y, Z operators
  coefficient: number;
}

export interface VQEResult {
  groundStateEnergy: number;
  optimalParameters: number[];
  groundState: Complex[];
  iterations: number;
  converged: boolean;
  energyHistory: number[];
  executionTime: number;
}

export interface Complex {
  re: number;
  im: number;
}

/**
 * VQE Solver for Hamiltonian ground state problems
 */
export class VQESolver {
  private config: VQEConfig;
  private hamiltonian: Hamiltonian;
  private parameterHistory: number[][] = [];
  private energyHistory: number[] = [];

  constructor(config: VQEConfig, hamiltonian: Hamiltonian) {
    this.config = config;
    this.hamiltonian = hamiltonian;
  }

  /**
   * Run VQE optimization to find ground state
   */
  async solve(): Promise<VQEResult> {
    const startTime = Date.now();

    // Initialize variational parameters
    const numParams = this.getNumParameters();
    let params = Array(numParams).fill(0).map(() => Math.random() * 2 * Math.PI);

    let bestEnergy = Infinity;
    let bestParams: number[] = [];
    let bestState: Complex[] = [];
    let converged = false;

    // Adam optimizer state (if using Adam)
    const adamState = this.initializeAdamState(numParams);

    for (let iter = 0; iter < this.config.maxIterations; iter++) {
      // Prepare quantum state with ansatz
      const state = this.prepareAnsatz(params);

      // Compute expectation value of Hamiltonian
      const energy = this.computeHamiltonianExpectation(state);

      this.energyHistory.push(energy);
      this.parameterHistory.push([...params]);

      if (energy < bestEnergy) {
        bestEnergy = energy;
        bestParams = [...params];
        bestState = [...state];
      }

      // Check convergence
      if (iter > 0 && Math.abs(energy - this.energyHistory[iter - 1]) < this.config.tolerance) {
        converged = true;
        break;
      }

      // Update parameters using selected optimizer
      params = this.updateParameters(params, state, adamState, iter);
    }

    const executionTime = Date.now() - startTime;

    return {
      groundStateEnergy: bestEnergy,
      optimalParameters: bestParams,
      groundState: bestState,
      iterations: this.energyHistory.length,
      converged,
      energyHistory: this.energyHistory,
      executionTime
    };
  }

  /**
   * Get number of parameters for chosen ansatz
   */
  private getNumParameters(): number {
    switch (this.config.ansatzType) {
      case 'hardware-efficient':
        // Each layer has n rotations + n-1 entangling gates
        return this.config.numQubits * this.config.ansatzDepth * 2;
      case 'uccsd':
        // UCCSD has O(n^4) parameters for n qubits
        const n = this.config.numQubits;
        return n * (n - 1) / 2 + n * (n - 1) * (n - 2) * (n - 3) / 8;
      default:
        return this.config.numQubits * this.config.ansatzDepth;
    }
  }

  /**
   * Prepare quantum state using variational ansatz
   */
  private prepareAnsatz(params: number[]): Complex[] {
    const n = this.config.numQubits;
    const dim = Math.pow(2, n);

    // Initialize to |0⟩^n
    let state: Complex[] = Array(dim).fill({ re: 0, im: 0 });
    state[0] = { re: 1, im: 0 };

    switch (this.config.ansatzType) {
      case 'hardware-efficient':
        state = this.applyHardwareEfficientAnsatz(state, params);
        break;
      case 'uccsd':
        state = this.applyUCCSDAnsatz(state, params);
        break;
      default:
        state = this.applyCustomAnsatz(state, params);
    }

    return state;
  }

  /**
   * Apply hardware-efficient ansatz
   * Pattern: (Ry rotations) -> (CNOT ladder) repeated depth times
   */
  private applyHardwareEfficientAnsatz(state: Complex[], params: number[]): Complex[] {
    const n = this.config.numQubits;
    let currentState = [...state];
    let paramIdx = 0;

    for (let layer = 0; layer < this.config.ansatzDepth; layer++) {
      // Apply Ry rotations to all qubits
      for (let qubit = 0; qubit < n; qubit++) {
        currentState = this.applyRyRotation(currentState, qubit, params[paramIdx++]);
      }

      // Apply CNOT ladder for entanglement
      for (let qubit = 0; qubit < n - 1; qubit++) {
        currentState = this.applyCNOT(currentState, qubit, qubit + 1);
      }

      // Apply Rz rotations
      for (let qubit = 0; qubit < n; qubit++) {
        currentState = this.applyRzRotation(currentState, qubit, params[paramIdx++]);
      }
    }

    return currentState;
  }

  /**
   * Apply UCCSD (Unitary Coupled Cluster) ansatz
   * Used for molecular electronic structure
   */
  private applyUCCSDAnsatz(state: Complex[], params: number[]): Complex[] {
    // Simplified UCCSD for demonstration
    const n = this.config.numQubits;
    let currentState = [...state];
    let paramIdx = 0;

    // Single excitations: i -> a
    for (let i = 0; i < n / 2; i++) {
      for (let a = n / 2; a < n; a++) {
        const theta = params[paramIdx++];
        currentState = this.applySingleExcitation(currentState, i, a, theta);
      }
    }

    // Double excitations: i,j -> a,b (simplified)
    for (let i = 0; i < n / 2 - 1; i++) {
      for (let j = i + 1; j < n / 2; j++) {
        for (let a = n / 2; a < n - 1; a++) {
          for (let b = a + 1; b < n; b++) {
            if (paramIdx < params.length) {
              const theta = params[paramIdx++];
              currentState = this.applyDoubleExcitation(currentState, i, j, a, b, theta);
            }
          }
        }
      }
    }

    return currentState;
  }

  /**
   * Apply custom ansatz
   */
  private applyCustomAnsatz(state: Complex[], params: number[]): Complex[] {
    // Default to hardware-efficient
    return this.applyHardwareEfficientAnsatz(state, params);
  }

  /**
   * Compute expectation value ⟨ψ|H|ψ⟩
   */
  private computeHamiltonianExpectation(state: Complex[]): number {
    let expectation = 0;

    for (const pauliTerm of this.hamiltonian.pauliStrings) {
      const pauliExpectation = this.computePauliExpectation(state, pauliTerm.pauli);
      expectation += pauliTerm.coefficient * pauliExpectation;
    }

    return expectation;
  }

  /**
   * Compute expectation of a Pauli string
   */
  private computePauliExpectation(state: Complex[], pauliString: string): number {
    const transformedState = this.applyPauliString(state, pauliString);
    let expectation = 0;

    state.forEach((amp, idx) => {
      const transformedAmp = transformedState[idx];
      expectation += amp.re * transformedAmp.re + amp.im * transformedAmp.im;
    });

    return expectation;
  }

  /**
   * Apply Pauli string operator to state
   */
  private applyPauliString(state: Complex[], pauliString: string): Complex[] {
    let result = [...state];

    for (let qubit = 0; qubit < pauliString.length; qubit++) {
      const pauli = pauliString[qubit];

      switch (pauli) {
        case 'X':
          result = this.applyPauliX(result, qubit);
          break;
        case 'Y':
          result = this.applyPauliY(result, qubit);
          break;
        case 'Z':
          result = this.applyPauliZ(result, qubit);
          break;
        case 'I':
          // Identity - do nothing
          break;
      }
    }

    return result;
  }

  /**
   * Update parameters using selected optimizer
   */
  private updateParameters(
    params: number[],
    state: Complex[],
    adamState: AdamState,
    iteration: number
  ): number[] {
    const gradients = this.computeGradients(params);

    switch (this.config.optimizer) {
      case 'adam':
        return this.adamUpdate(params, gradients, adamState, iteration);
      case 'gradient-descent':
        return params.map((p, i) => p - this.config.learningRate * gradients[i]);
      case 'cobyla':
        return this.cobylaUpdate(params, gradients);
      default:
        return params.map((p, i) => p - this.config.learningRate * gradients[i]);
    }
  }

  /**
   * Compute gradients using parameter shift rule
   */
  private computeGradients(params: number[]): number[] {
    const shift = Math.PI / 2;
    const gradients: number[] = [];

    for (let i = 0; i < params.length; i++) {
      const paramsPlus = [...params];
      const paramsMinus = [...params];
      paramsPlus[i] += shift;
      paramsMinus[i] -= shift;

      const statePlus = this.prepareAnsatz(paramsPlus);
      const stateMinus = this.prepareAnsatz(paramsMinus);

      const energyPlus = this.computeHamiltonianExpectation(statePlus);
      const energyMinus = this.computeHamiltonianExpectation(stateMinus);

      gradients.push((energyPlus - energyMinus) / 2);
    }

    return gradients;
  }

  // Gate operations
  private applyRyRotation(state: Complex[], qubit: number, angle: number): Complex[] {
    const dim = state.length;
    const newState = Array(dim).fill({ re: 0, im: 0 });
    const cos = Math.cos(angle / 2);
    const sin = Math.sin(angle / 2);

    for (let idx = 0; idx < dim; idx++) {
      const flippedIdx = idx ^ (1 << qubit);
      const bit = (idx >> qubit) & 1;

      if (bit === 0) {
        newState[idx] = this.complexAdd(
          this.complexScale(state[idx], cos),
          this.complexScale(state[flippedIdx], sin)
        );
      } else {
        newState[idx] = this.complexAdd(
          this.complexScale(state[flippedIdx], -sin),
          this.complexScale(state[idx], cos)
        );
      }
    }

    return newState;
  }

  private applyRzRotation(state: Complex[], qubit: number, angle: number): Complex[] {
    return state.map((amp, idx) => {
      const bit = (idx >> qubit) & 1;
      const phase = bit === 0 ? -angle / 2 : angle / 2;
      return this.complexMult(amp, { re: Math.cos(phase), im: Math.sin(phase) });
    });
  }

  private applyCNOT(state: Complex[], control: number, target: number): Complex[] {
    const newState = [...state];
    const controlMask = 1 << control;
    const targetMask = 1 << target;

    for (let idx = 0; idx < state.length; idx++) {
      if ((idx & controlMask) !== 0) {
        const flippedIdx = idx ^ targetMask;
        [newState[idx], newState[flippedIdx]] = [newState[flippedIdx], newState[idx]];
      }
    }

    return newState;
  }

  private applyPauliX(state: Complex[], qubit: number): Complex[] {
    const newState = [...state];
    const mask = 1 << qubit;

    for (let idx = 0; idx < state.length; idx++) {
      const flippedIdx = idx ^ mask;
      if (idx < flippedIdx) {
        [newState[idx], newState[flippedIdx]] = [newState[flippedIdx], newState[idx]];
      }
    }

    return newState;
  }

  private applyPauliY(state: Complex[], qubit: number): Complex[] {
    const newState = [...state];
    const mask = 1 << qubit;

    for (let idx = 0; idx < state.length; idx++) {
      const bit = (idx >> qubit) & 1;
      const flippedIdx = idx ^ mask;
      const phase = { re: 0, im: bit === 0 ? -1 : 1 };

      if (idx < flippedIdx) {
        [newState[idx], newState[flippedIdx]] = [
          this.complexMult(newState[flippedIdx], phase),
          this.complexMult(newState[idx], this.complexConj(phase))
        ];
      }
    }

    return newState;
  }

  private applyPauliZ(state: Complex[], qubit: number): Complex[] {
    return state.map((amp, idx) => {
      const bit = (idx >> qubit) & 1;
      return bit === 1 ? this.complexScale(amp, -1) : amp;
    });
  }

  private applySingleExcitation(state: Complex[], i: number, a: number, theta: number): Complex[] {
    // Simplified single excitation operator
    let result = [...state];
    result = this.applyRyRotation(result, i, theta);
    result = this.applyCNOT(result, i, a);
    return result;
  }

  private applyDoubleExcitation(
    state: Complex[],
    i: number,
    j: number,
    a: number,
    b: number,
    theta: number
  ): Complex[] {
    // Simplified double excitation operator
    let result = [...state];
    result = this.applyRyRotation(result, i, theta / 2);
    result = this.applyCNOT(result, i, j);
    result = this.applyCNOT(result, j, a);
    result = this.applyCNOT(result, a, b);
    return result;
  }

  // Adam optimizer
  private initializeAdamState(numParams: number): AdamState {
    return {
      m: Array(numParams).fill(0),
      v: Array(numParams).fill(0),
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8
    };
  }

  private adamUpdate(
    params: number[],
    gradients: number[],
    state: AdamState,
    iteration: number
  ): number[] {
    const { m, v, beta1, beta2, epsilon } = state;
    const t = iteration + 1;

    return params.map((p, i) => {
      m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
      v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] ** 2;

      const mHat = m[i] / (1 - Math.pow(beta1, t));
      const vHat = v[i] / (1 - Math.pow(beta2, t));

      return p - this.config.learningRate * mHat / (Math.sqrt(vHat) + epsilon);
    });
  }

  private cobylaUpdate(params: number[], gradients: number[]): number[] {
    // Simplified COBYLA-like update
    return params.map((p, i) => p - this.config.learningRate * gradients[i]);
  }

  // Complex number helpers
  private complexAdd(a: Complex, b: Complex): Complex {
    return { re: a.re + b.re, im: a.im + b.im };
  }

  private complexMult(a: Complex, b: Complex): Complex {
    return {
      re: a.re * b.re - a.im * b.im,
      im: a.re * b.im + a.im * b.re
    };
  }

  private complexScale(a: Complex, scalar: number): Complex {
    return { re: a.re * scalar, im: a.im * scalar };
  }

  private complexConj(a: Complex): Complex {
    return { re: a.re, im: -a.im };
  }
}

interface AdamState {
  m: number[];
  v: number[];
  beta1: number;
  beta2: number;
  epsilon: number;
}

/**
 * Create Hamiltonian for common problems
 */
export function createIsingHamiltonian(couplings: number[][], fields: number[]): Hamiltonian {
  const pauliStrings: PauliString[] = [];

  // Add ZZ coupling terms
  for (let i = 0; i < couplings.length; i++) {
    for (let j = i + 1; j < couplings[i].length; j++) {
      if (couplings[i][j] !== 0) {
        const pauli = 'I'.repeat(i) + 'Z' + 'I'.repeat(j - i - 1) + 'Z' + 'I'.repeat(couplings.length - j - 1);
        pauliStrings.push({ pauli, coefficient: couplings[i][j] });
      }
    }
  }

  // Add Z field terms
  fields.forEach((field, i) => {
    if (field !== 0) {
      const pauli = 'I'.repeat(i) + 'Z' + 'I'.repeat(fields.length - i - 1);
      pauliStrings.push({ pauli, coefficient: field });
    }
  });

  return { pauliStrings };
}
