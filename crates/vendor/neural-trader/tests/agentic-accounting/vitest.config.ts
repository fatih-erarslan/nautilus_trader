import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    name: 'Agentic Accounting Tests',
    environment: 'node',

    // Test patterns
    include: [
      'unit/**/*.test.ts',
      'integration/**/*.test.ts',
      'e2e/**/*.test.ts',
    ],

    // Setup
    setupFiles: ['./utils/test-setup.ts'],

    // Coverage configuration
    coverage: {
      provider: 'v8',
      reporter: ['text', 'text-summary', 'html', 'lcov', 'json'],
      reportsDirectory: './coverage',
      include: [
        '../../packages/agentic-accounting/**/src/**/*.ts',
      ],
      exclude: [
        '**/*.d.ts',
        '**/node_modules/**',
        '**/dist/**',
        '**/coverage/**',
      ],

      // 90% minimum thresholds
      thresholds: {
        branches: 90,
        functions: 90,
        lines: 90,
        statements: 90,
      },
    },

    // Timeouts
    testTimeout: 30000, // 30 seconds
    hookTimeout: 10000, // 10 seconds

    // Isolation
    isolate: true,
    clearMocks: true,
    mockReset: true,
    restoreMocks: true,

    // Concurrency
    pool: 'threads',
    poolOptions: {
      threads: {
        singleThread: false,
      },
    },

    // Output
    reporter: ['default', 'verbose'],
  },

  resolve: {
    alias: {
      '@': path.resolve(__dirname, '../../packages/agentic-accounting'),
      '@fixtures': path.resolve(__dirname, './fixtures'),
      '@utils': path.resolve(__dirname, './utils'),
    },
  },
});
