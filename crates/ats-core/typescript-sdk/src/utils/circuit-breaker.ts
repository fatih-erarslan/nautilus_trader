/**
 * Circuit Breaker Pattern Implementation
 * 
 * Prevents cascade failures by monitoring service health and temporarily
 * blocking requests to failing services.
 */

import EventEmitter from 'eventemitter3';
import { CircuitBreakerConfig } from '../types';

export enum CircuitBreakerState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN', 
  HALF_OPEN = 'HALF_OPEN'
}

interface CircuitBreakerEvents {
  'open': [];
  'closed': [];
  'halfOpen': [];
  'failure': [Error];
  'success': [];
}

export class CircuitBreaker extends EventEmitter<CircuitBreakerEvents> {
  private state: CircuitBreakerState = CircuitBreakerState.CLOSED;
  private failureCount = 0;
  private successCount = 0;
  private nextAttempt = 0;
  private config: Required<CircuitBreakerConfig>;

  constructor(config: CircuitBreakerConfig) {
    super();
    
    this.config = {
      enabled: true,
      failureThreshold: 5,
      resetTimeout: 60000,
      monitoringPeriod: 10000,
      ...config,
    };
  }

  /**
   * Check if circuit breaker is open
   */
  public isOpen(): boolean {
    if (!this.config.enabled) {
      return false;
    }

    if (this.state === CircuitBreakerState.OPEN) {
      // Check if we should transition to half-open
      if (Date.now() >= this.nextAttempt) {
        this.state = CircuitBreakerState.HALF_OPEN;
        this.emit('halfOpen');
      }
    }

    return this.state === CircuitBreakerState.OPEN;
  }

  /**
   * Record a successful operation
   */
  public recordSuccess(): void {
    if (!this.config.enabled) {
      return;
    }

    this.successCount++;
    this.failureCount = 0;
    this.emit('success');

    if (this.state === CircuitBreakerState.HALF_OPEN) {
      this.state = CircuitBreakerState.CLOSED;
      this.emit('closed');
    }
  }

  /**
   * Record a failed operation
   */
  public recordFailure(error?: Error): void {
    if (!this.config.enabled) {
      return;
    }

    this.failureCount++;
    this.emit('failure', error || new Error('Circuit breaker failure'));

    if (this.state === CircuitBreakerState.CLOSED || this.state === CircuitBreakerState.HALF_OPEN) {
      if (this.failureCount >= this.config.failureThreshold) {
        this.open();
      }
    }
  }

  /**
   * Open the circuit breaker
   */
  private open(): void {
    this.state = CircuitBreakerState.OPEN;
    this.nextAttempt = Date.now() + this.config.resetTimeout;
    this.emit('open');
  }

  /**
   * Get current state
   */
  public getState(): CircuitBreakerState {
    return this.state;
  }

  /**
   * Get failure count
   */
  public getFailureCount(): number {
    return this.failureCount;
  }

  /**
   * Get success count
   */
  public getSuccessCount(): number {
    return this.successCount;
  }

  /**
   * Reset circuit breaker to closed state
   */
  public reset(): void {
    this.state = CircuitBreakerState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.nextAttempt = 0;
    this.emit('closed');
  }

  /**
   * Get circuit breaker metrics
   */
  public getMetrics() {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      nextAttempt: this.nextAttempt,
      isOpen: this.isOpen(),
      config: this.config,
    };
  }

  /**
   * Execute operation with circuit breaker protection
   */
  public async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (this.isOpen()) {
      throw new Error('Circuit breaker is open');
    }

    try {
      const result = await operation();
      this.recordSuccess();
      return result;
    } catch (error) {
      this.recordFailure(error as Error);
      throw error;
    }
  }
}