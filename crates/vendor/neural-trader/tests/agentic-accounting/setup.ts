/**
 * Global Test Setup
 *
 * Configure test environment, custom matchers, and global utilities
 */

import { beforeAll, afterAll } from 'vitest';
import './fixtures/custom-matchers';

// Configure Decimal.js for financial precision
import Decimal from 'decimal.js';

Decimal.set({
  precision: 30, // High precision for crypto
  rounding: Decimal.ROUND_HALF_UP,
  toExpNeg: -9, // 9 decimal places before exponential notation
  toExpPos: 20,
});

// Global test configuration
beforeAll(() => {
  console.log('🧪 Starting Agentic Accounting Test Suite');
  console.log('📊 Phase 2: Tax Calculation Components');
  console.log('');
});

afterAll(() => {
  console.log('');
  console.log('✅ Test Suite Complete');
});

// Global error handlers
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Test utilities available globally
export const TEST_CONFIG = {
  COVERAGE_THRESHOLD: {
    rust: 95,
    typescript: 90,
    integration: 85,
  },
  PERFORMANCE_TARGETS: {
    fifo_1000_lots: 10, // milliseconds
    lifo_1000_lots: 10,
    hifo_1000_lots: 10,
    specific_id_1000_lots: 10,
    average_cost_1000_lots: 10,
    tax_compute_agent: 1000, // 1 second
  },
  DECIMAL_PRECISION: 8, // satoshi precision
};
