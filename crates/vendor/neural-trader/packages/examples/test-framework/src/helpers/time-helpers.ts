/**
 * Time-related testing utilities
 */

/**
 * Mock timer that can be controlled in tests
 */
export class MockTimer {
  private currentTime: number;
  private timers: Map<number, { callback: () => void; time: number }>;
  private nextId: number;

  constructor(startTime = Date.now()) {
    this.currentTime = startTime;
    this.timers = new Map();
    this.nextId = 1;
  }

  /**
   * Get current mock time
   */
  now(): number {
    return this.currentTime;
  }

  /**
   * Advance time by specified milliseconds
   */
  tick(ms: number): void {
    const targetTime = this.currentTime + ms;

    // Execute all timers that should fire
    const timersToExecute = Array.from(this.timers.entries())
      .filter(([_, timer]) => timer.time <= targetTime)
      .sort((a, b) => a[1].time - b[1].time);

    for (const [id, timer] of timersToExecute) {
      this.currentTime = timer.time;
      timer.callback();
      this.timers.delete(id);
    }

    this.currentTime = targetTime;
  }

  /**
   * Set a mock timeout
   */
  setTimeout(callback: () => void, ms: number): number {
    const id = this.nextId++;
    this.timers.set(id, {
      callback,
      time: this.currentTime + ms
    });
    return id;
  }

  /**
   * Clear a mock timeout
   */
  clearTimeout(id: number): void {
    this.timers.delete(id);
  }

  /**
   * Clear all timers
   */
  clearAll(): void {
    this.timers.clear();
  }

  /**
   * Get number of pending timers
   */
  pendingTimers(): number {
    return this.timers.size;
  }
}

/**
 * Generate timestamps for testing
 */
export function generateTimestamps(
  count: number,
  intervalMs: number,
  startTime = Date.now()
): number[] {
  return Array.from({ length: count }, (_, i) =>
    startTime + i * intervalMs
  );
}

/**
 * Create time series data for testing
 */
export function generateTimeSeries(
  count: number,
  intervalMs: number,
  options: {
    trend?: number;
    seasonality?: number;
    noise?: number;
    startValue?: number;
  } = {}
): { timestamps: number[]; values: number[] } {
  const {
    trend = 0,
    seasonality = 0,
    noise = 0.1,
    startValue = 100
  } = options;

  const timestamps = generateTimestamps(count, intervalMs);
  const values: number[] = [];

  for (let i = 0; i < count; i++) {
    const trendComponent = trend * i;
    const seasonalComponent = seasonality * Math.sin(2 * Math.PI * i / 24);
    const noiseComponent = (Math.random() - 0.5) * noise * 2;

    values.push(startValue + trendComponent + seasonalComponent + noiseComponent);
  }

  return { timestamps, values };
}

/**
 * Measure elapsed time
 */
export class StopWatch {
  private startTime: number | null = null;
  private stopTime: number | null = null;

  start(): void {
    this.startTime = performance.now();
    this.stopTime = null;
  }

  stop(): number {
    if (this.startTime === null) {
      throw new Error('StopWatch not started');
    }
    this.stopTime = performance.now();
    return this.elapsed();
  }

  elapsed(): number {
    if (this.startTime === null) {
      throw new Error('StopWatch not started');
    }
    const end = this.stopTime ?? performance.now();
    return end - this.startTime;
  }

  reset(): void {
    this.startTime = null;
    this.stopTime = null;
  }
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Format duration in human-readable format
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms.toFixed(2)}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(2)}s`;
  } else if (ms < 3600000) {
    return `${(ms / 60000).toFixed(2)}m`;
  } else {
    return `${(ms / 3600000).toFixed(2)}h`;
  }
}
