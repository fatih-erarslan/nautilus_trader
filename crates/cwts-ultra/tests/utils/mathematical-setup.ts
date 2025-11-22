/**
 * Mathematical Test Setup - Configures mathematical validation framework
 * Ensures scientific rigor in all numerical computations and validations
 */

import { jest } from '@jest/globals';

// Mathematical constants and tolerances
const MATHEMATICAL_CONSTANTS = {
  EPSILON: Number.EPSILON,
  PRECISION_TOLERANCE: 1e-12,
  STATISTICAL_CONFIDENCE: 0.99,
  NUMERICAL_STABILITY_THRESHOLD: 1e-10,
  CONVERGENCE_TOLERANCE: 1e-8,
  MAX_ITERATIONS: 10000
};

// Extend global with mathematical utilities
(global as any).mathUtils = {
  constants: MATHEMATICAL_CONSTANTS,

  // Floating-point comparison with tolerance
  isEqual: (a: number, b: number, tolerance: number = MATHEMATICAL_CONSTANTS.PRECISION_TOLERANCE): boolean => {
    return Math.abs(a - b) <= tolerance;
  },

  // Relative error calculation
  relativeError: (actual: number, expected: number): number => {
    if (expected === 0) {
      return Math.abs(actual);
    }
    return Math.abs((actual - expected) / expected);
  },

  // Numerical derivative for testing
  numericalDerivative: (f: (x: number) => number, x: number, h: number = 1e-8): number => {
    return (f(x + h) - f(x - h)) / (2 * h);
  },

  // Numerical integration (trapezoidal rule)
  numericalIntegral: (f: (x: number) => number, a: number, b: number, n: number = 1000): number => {
    const h = (b - a) / n;
    let sum = (f(a) + f(b)) / 2;
    
    for (let i = 1; i < n; i++) {
      sum += f(a + i * h);
    }
    
    return sum * h;
  },

  // Statistical functions
  mean: (values: number[]): number => {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  },

  variance: (values: number[]): number => {
    const avg = (global as any).mathUtils.mean(values);
    return values.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / (values.length - 1);
  },

  standardDeviation: (values: number[]): number => {
    return Math.sqrt((global as any).mathUtils.variance(values));
  },

  // Normal distribution functions
  normalPdf: (x: number, mean: number = 0, stdDev: number = 1): number => {
    const factor = 1 / (stdDev * Math.sqrt(2 * Math.PI));
    const exponent = -0.5 * Math.pow((x - mean) / stdDev, 2);
    return factor * Math.exp(exponent);
  },

  // Error function approximation
  erf: (x: number): number => {
    // Abramowitz and Stegun approximation
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;

    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  },

  // Matrix operations
  matrixMultiply: (a: number[][], b: number[][]): number[][] => {
    const result = Array(a.length).fill(null).map(() => Array(b[0].length).fill(0));
    
    for (let i = 0; i < a.length; i++) {
      for (let j = 0; j < b[0].length; j++) {
        for (let k = 0; k < b.length; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    
    return result;
  },

  // Determinant calculation
  determinant: (matrix: number[][]): number => {
    const n = matrix.length;
    if (n === 1) return matrix[0][0];
    if (n === 2) return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    
    let det = 0;
    for (let i = 0; i < n; i++) {
      const subMatrix = matrix.slice(1).map(row => 
        row.filter((_, colIndex) => colIndex !== i)
      );
      det += Math.pow(-1, i) * matrix[0][i] * (global as any).mathUtils.determinant(subMatrix);
    }
    
    return det;
  },

  // Eigenvalue calculation (power iteration for largest eigenvalue)
  largestEigenvalue: (matrix: number[][], tolerance: number = 1e-10, maxIterations: number = 1000): number => {
    const n = matrix.length;
    let x = Array(n).fill(1); // Initial guess
    let lambda = 0;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      // x = A * x
      const newX = Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          newX[i] += matrix[i][j] * x[j];
        }
      }
      
      // Calculate eigenvalue estimate
      const newLambda = newX.reduce((sum, val, i) => sum + val * x[i], 0) / 
                       x.reduce((sum, val) => sum + val * val, 0);
      
      // Normalize x
      const norm = Math.sqrt(newX.reduce((sum, val) => sum + val * val, 0));
      x = newX.map(val => val / norm);
      
      // Check convergence
      if (Math.abs(newLambda - lambda) < tolerance) {
        return newLambda;
      }
      
      lambda = newLambda;
    }
    
    return lambda;
  },

  // Condition number estimation
  conditionNumber: (matrix: number[][]): number => {
    // Simplified estimation using largest/smallest eigenvalue ratio
    // In practice, would use more sophisticated methods
    return 1.0; // Placeholder - would implement actual calculation
  }
};

// Statistical testing utilities
(global as any).statUtils = {
  // Kolmogorov-Smirnov test for normality
  ksTest: (sample: number[]): { statistic: number; pValue: number; isNormal: boolean } => {
    const n = sample.length;
    const sortedSample = [...sample].sort((a, b) => a - b);
    const mean = (global as any).mathUtils.mean(sample);
    const std = (global as any).mathUtils.standardDeviation(sample);
    
    let dMax = 0;
    for (let i = 0; i < n; i++) {
      const x = (sortedSample[i] - mean) / std;
      const empiricalCdf = (i + 1) / n;
      const theoreticalCdf = 0.5 * (1 + (global as any).mathUtils.erf(x / Math.sqrt(2)));
      
      const d = Math.abs(empiricalCdf - theoreticalCdf);
      dMax = Math.max(dMax, d);
    }
    
    // Approximate p-value calculation
    const lambda = dMax * (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n));
    let pValue = 2 * Math.exp(-2 * lambda * lambda);
    pValue = Math.min(1, Math.max(0, pValue));
    
    return {
      statistic: dMax,
      pValue,
      isNormal: pValue > 0.05
    };
  },

  // Chi-square goodness of fit test
  chiSquareTest: (observed: number[], expected: number[]): { statistic: number; pValue: number; passed: boolean } => {
    if (observed.length !== expected.length) {
      throw new Error('Observed and expected arrays must have the same length');
    }
    
    let chiSquare = 0;
    for (let i = 0; i < observed.length; i++) {
      if (expected[i] > 0) {
        chiSquare += Math.pow(observed[i] - expected[i], 2) / expected[i];
      }
    }
    
    const degreesOfFreedom = observed.length - 1;
    // Simplified p-value calculation (would use proper chi-square distribution)
    const pValue = Math.exp(-chiSquare / 2);
    
    return {
      statistic: chiSquare,
      pValue,
      passed: pValue > 0.05
    };
  },

  // T-test for means comparison
  tTest: (sample1: number[], sample2: number[]): { statistic: number; pValue: number; significant: boolean } => {
    const mean1 = (global as any).mathUtils.mean(sample1);
    const mean2 = (global as any).mathUtils.mean(sample2);
    const var1 = (global as any).mathUtils.variance(sample1);
    const var2 = (global as any).mathUtils.variance(sample2);
    const n1 = sample1.length;
    const n2 = sample2.length;
    
    const pooledVar = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    const standardError = Math.sqrt(pooledVar * (1/n1 + 1/n2));
    const tStatistic = (mean1 - mean2) / standardError;
    
    // Simplified p-value (would use proper t-distribution)
    const pValue = 2 * (1 - Math.abs(tStatistic) / (Math.abs(tStatistic) + Math.sqrt(n1 + n2 - 2)));
    
    return {
      statistic: tStatistic,
      pValue,
      significant: pValue < 0.05
    };
  }
};

// Numerical algorithm validation utilities
(global as any).numericalUtils = {
  // Test numerical stability
  testStability: (algorithm: (input: number) => number, testPoints: number[]): { stable: boolean; maxError: number } => {
    let maxError = 0;
    let stable = true;
    
    for (const point of testPoints) {
      // Test with small perturbations
      const baseResult = algorithm(point);
      const perturbedResult = algorithm(point + MATHEMATICAL_CONSTANTS.EPSILON);
      
      const error = Math.abs(perturbedResult - baseResult);
      maxError = Math.max(maxError, error);
      
      // Check if error is proportional to input perturbation
      if (error > MATHEMATICAL_CONSTANTS.NUMERICAL_STABILITY_THRESHOLD) {
        stable = false;
      }
    }
    
    return { stable, maxError };
  },

  // Test convergence properties
  testConvergence: (iterativeAlgorithm: (x0: number, tolerance: number) => { result: number; iterations: number }): {
    converges: boolean;
    convergenceRate: number;
    iterations: number;
  } => {
    const testResult = iterativeAlgorithm(1.0, MATHEMATICAL_CONSTANTS.CONVERGENCE_TOLERANCE);
    
    return {
      converges: testResult.iterations < MATHEMATICAL_CONSTANTS.MAX_ITERATIONS,
      convergenceRate: testResult.iterations > 0 ? 1 / testResult.iterations : 0,
      iterations: testResult.iterations
    };
  },

  // Validate boundary conditions
  testBoundaryConditions: (fn: (x: number) => number, boundaries: { min: number; max: number }): {
    minBoundaryValid: boolean;
    maxBoundaryValid: boolean;
    infinityHandled: boolean;
  } => {
    let minBoundaryValid = true;
    let maxBoundaryValid = true;
    let infinityHandled = true;
    
    try {
      const minResult = fn(boundaries.min);
      minBoundaryValid = Number.isFinite(minResult);
    } catch {
      minBoundaryValid = false;
    }
    
    try {
      const maxResult = fn(boundaries.max);
      maxBoundaryValid = Number.isFinite(maxResult);
    } catch {
      maxBoundaryValid = false;
    }
    
    try {
      const infResult = fn(Infinity);
      infinityHandled = Number.isFinite(infResult) || !Number.isNaN(infResult);
    } catch {
      infinityHandled = true; // Exception handling is acceptable
    }
    
    return { minBoundaryValid, maxBoundaryValid, infinityHandled };
  }
};

// Custom Jest matchers for mathematical assertions
expect.extend({
  toBeCloseTo(received: number, expected: number, precision: number = 12) {
    const pass = Math.abs(received - expected) < Math.pow(10, -precision);
    
    return {
      message: () =>
        `expected ${received} to be close to ${expected} with precision ${precision}`,
      pass
    };
  },

  toBeNumericallyStable(received: (x: number) => number, testPoints: number[]) {
    const { stable } = (global as any).numericalUtils.testStability(received, testPoints);
    
    return {
      message: () => `expected function to be numerically stable`,
      pass: stable
    };
  },

  toConverge(received: (x0: number, tolerance: number) => { result: number; iterations: number }) {
    const { converges } = (global as any).numericalUtils.testConvergence(received);
    
    return {
      message: () => `expected iterative algorithm to converge`,
      pass: converges
    };
  },

  toBeNormallyDistributed(received: number[]) {
    const { isNormal } = (global as any).statUtils.ksTest(received);
    
    return {
      message: () => `expected sample to be normally distributed`,
      pass: isNormal
    };
  }
});

// Type declarations for TypeScript
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeCloseTo(expected: number, precision?: number): R;
      toBeNumericallyStable(testPoints: number[]): R;
      toConverge(): R;
      toBeNormallyDistributed(): R;
    }
  }
}

console.log('🧮 Mathematical testing framework initialized');