/**
 * Retry Handler with Exponential Backoff
 * 
 * Implements intelligent retry logic with exponential backoff,
 * jitter, and configurable retry policies.
 */

import { RetryConfig, AtsCoreError } from '../types';

interface RetryMetrics {
  totalRetries: number;
  successfulRetries: number;
  failedRetries: number;
  averageAttempts: number;
}

export class RetryHandler {
  private config: Required<RetryConfig>;
  private metrics: RetryMetrics = {
    totalRetries: 0,
    successfulRetries: 0,
    failedRetries: 0,
    averageAttempts: 0,
  };

  constructor(config: RetryConfig) {
    this.config = {
      enabled: true,
      maxRetries: 3,
      baseDelay: 1000,
      maxDelay: 10000,
      backoffMultiplier: 2,
      ...config,
    };
  }

  /**
   * Execute operation with retry logic
   */
  public async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (!this.config.enabled) {
      return operation();
    }

    let lastError: Error;
    let attempt = 0;

    while (attempt <= this.config.maxRetries) {
      try {
        const result = await operation();
        
        // Update metrics
        if (attempt > 0) {
          this.metrics.successfulRetries++;
          this.updateAverageAttempts(attempt + 1);
        }
        
        return result;
      } catch (error) {
        lastError = error as Error;
        attempt++;

        // Don't retry on certain error types
        if (!this.shouldRetry(error as Error) || attempt > this.config.maxRetries) {
          if (attempt > 1) {
            this.metrics.failedRetries++;
            this.updateAverageAttempts(attempt);
          }
          break;
        }

        // Wait before retry with exponential backoff
        const delay = this.calculateDelay(attempt - 1);
        await this.sleep(delay);

        this.metrics.totalRetries++;
      }
    }

    throw lastError!;
  }

  /**
   * Check if error should be retried
   */
  private shouldRetry(error: Error): boolean {
    // Don't retry validation errors or authentication errors
    if (error instanceof AtsCoreError) {
      const nonRetryableCodes = [
        'VALIDATION_ERROR',
        'AUTHENTICATION_ERROR',
        'AUTHORIZATION_ERROR',
        'NOT_FOUND_ERROR',
        'CIRCUIT_BREAKER_OPEN',
      ];
      
      if (nonRetryableCodes.includes(error.code)) {
        return false;
      }
    }

    // Don't retry 4xx errors (except 408, 429)
    if (error.message.includes('HTTP 4') && 
        !error.message.includes('HTTP 408') && 
        !error.message.includes('HTTP 429')) {
      return false;
    }

    return true;
  }

  /**
   * Calculate delay with exponential backoff and jitter
   */
  private calculateDelay(attempt: number): number {
    const exponentialDelay = this.config.baseDelay * Math.pow(this.config.backoffMultiplier, attempt);
    
    // Add jitter (±25% random variation)
    const jitter = exponentialDelay * 0.25 * (Math.random() - 0.5);
    const delayWithJitter = exponentialDelay + jitter;
    
    // Cap at max delay
    return Math.min(delayWithJitter, this.config.maxDelay);
  }

  /**
   * Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Update average attempts metric
   */
  private updateAverageAttempts(attempts: number): void {
    const totalOperations = this.metrics.successfulRetries + this.metrics.failedRetries;
    if (totalOperations === 1) {
      this.metrics.averageAttempts = attempts;
    } else {
      // Rolling average
      this.metrics.averageAttempts = 
        (this.metrics.averageAttempts * (totalOperations - 1) + attempts) / totalOperations;
    }
  }

  /**
   * Get retry metrics
   */
  public getMetrics(): RetryMetrics & { config: Required<RetryConfig> } {
    return {
      ...this.metrics,
      config: this.config,
    };
  }

  /**
   * Reset metrics
   */
  public resetMetrics(): void {
    this.metrics = {
      totalRetries: 0,
      successfulRetries: 0,
      failedRetries: 0,
      averageAttempts: 0,
    };
  }

  /**
   * Update retry configuration
   */
  public updateConfig(config: Partial<RetryConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Check if retry is enabled
   */
  public isEnabled(): boolean {
    return this.config.enabled;
  }
}