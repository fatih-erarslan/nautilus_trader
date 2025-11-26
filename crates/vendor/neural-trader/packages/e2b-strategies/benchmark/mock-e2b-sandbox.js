/**
 * Mock E2B Sandbox for benchmarking when E2B API is not available
 * Simulates realistic sandbox behavior including:
 * - Creation latency (50-200ms)
 * - Execution time (100-500ms)
 * - Success/failure rates (95% success)
 * - Resource cleanup
 */

class MockSandbox {
    constructor(options = {}) {
        this.sandboxId = `mock-${Date.now()}-${Math.random().toString(36).substring(7)}`;
        this.timeoutMs = options.timeoutMs || 300000;
        this.createdAt = Date.now();
        this._closed = false;
    }

    static async create(options = {}) {
        // Simulate sandbox creation latency (50-200ms)
        const creationLatency = Math.random() * 150 + 50;
        await new Promise(resolve => setTimeout(resolve, creationLatency));

        // 2% chance of creation failure (simulating real-world conditions)
        if (Math.random() < 0.02) {
            throw new Error('Mock sandbox creation failed (simulated error)');
        }

        return new MockSandbox(options);
    }

    async runCode(code) {
        if (this._closed) {
            throw new Error('Cannot run code on closed sandbox');
        }

        // Simulate code execution time (100-500ms)
        const executionTime = Math.random() * 400 + 100;
        await new Promise(resolve => setTimeout(resolve, executionTime));

        // 5% chance of execution failure (simulating real-world conditions)
        if (Math.random() < 0.05) {
            return {
                success: false,
                error: 'Mock execution error (simulated failure)',
                logs: [],
                executionTime
            };
        }

        return {
            success: true,
            output: `Mock execution result for: ${code.substring(0, 50)}...`,
            logs: ['Mock log 1', 'Mock log 2'],
            executionTime
        };
    }

    async close() {
        if (this._closed) {
            return;
        }

        // Simulate cleanup latency (10-50ms)
        const cleanupLatency = Math.random() * 40 + 10;
        await new Promise(resolve => setTimeout(resolve, cleanupLatency));

        this._closed = true;
    }

    get isClosed() {
        return this._closed;
    }
}

module.exports = { MockSandbox };
