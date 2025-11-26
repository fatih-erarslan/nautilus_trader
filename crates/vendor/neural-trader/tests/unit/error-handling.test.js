/**
 * Unit Tests for Error Handling Framework
 *
 * Tests:
 * - Error class hierarchy
 * - Retry logic with exponential backoff
 * - Circuit breaker state transitions
 * - Dead letter queue operations
 * - Error handler manager integration
 */

const {
  TradingError,
  NetworkError,
  ValidationError,
  ExternalAPIError,
  DatabaseError,
  ErrorCategory,
  ErrorSeverity,
  RetryStrategy,
  RetryExecutor,
  CircuitBreaker,
  CircuitState,
  DeadLetterQueue,
  ErrorHandlerManager,
} = require('../../src/infrastructure/error-handling');

describe('Error Handling Framework - Unit Tests', () => {
  describe('Error Classes', () => {
    test('TradingError base class', () => {
      const error = new TradingError('Test error', {
        code: 'TEST_ERROR',
        category: ErrorCategory.SYSTEM,
        severity: ErrorSeverity.HIGH,
        context: { test: 'data' },
      });

      expect(error.name).toBe('TradingError');
      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_ERROR');
      expect(error.category).toBe(ErrorCategory.SYSTEM);
      expect(error.severity).toBe(ErrorSeverity.HIGH);
      expect(error.context).toEqual({ test: 'data' });
      expect(error.retryable).toBe(false);
    });

    test('NetworkError is retryable by default', () => {
      const error = new NetworkError('Connection failed');

      expect(error.category).toBe(ErrorCategory.NETWORK);
      expect(error.retryable).toBe(true);
    });

    test('ValidationError is not retryable', () => {
      const error = new ValidationError('Invalid input');

      expect(error.category).toBe(ErrorCategory.VALIDATION);
      expect(error.severity).toBe(ErrorSeverity.LOW);
      expect(error.retryable).toBe(false);
    });

    test('ExternalAPIError includes API details', () => {
      const error = new ExternalAPIError('API request failed', {
        apiName: 'alpaca',
        statusCode: 503,
      });

      expect(error.category).toBe(ErrorCategory.EXTERNAL_API);
      expect(error.apiName).toBe('alpaca');
      expect(error.statusCode).toBe(503);
      expect(error.retryable).toBe(true);
    });

    test('DatabaseError is critical and retryable', () => {
      const error = new DatabaseError('Connection lost');

      expect(error.category).toBe(ErrorCategory.DATABASE);
      expect(error.severity).toBe(ErrorSeverity.CRITICAL);
      expect(error.retryable).toBe(true);
    });

    test('Error serialization to JSON', () => {
      const error = new TradingError('Test error', {
        code: 'TEST_CODE',
        context: { foo: 'bar' },
      });

      const json = error.toJSON();

      expect(json.name).toBe('TradingError');
      expect(json.message).toBe('Test error');
      expect(json.code).toBe('TEST_CODE');
      expect(json.context).toEqual({ foo: 'bar' });
      expect(json.timestamp).toBeDefined();
      expect(json.stack).toBeDefined();
    });
  });

  describe('Retry Strategy', () => {
    test('calculates exponential backoff', () => {
      const strategy = new RetryStrategy({
        initialDelay: 100,
        backoffMultiplier: 2,
        maxDelay: 1000,
        jitter: false,
      });

      expect(strategy.calculateDelay(0)).toBe(100);   // 100 * 2^0
      expect(strategy.calculateDelay(1)).toBe(200);   // 100 * 2^1
      expect(strategy.calculateDelay(2)).toBe(400);   // 100 * 2^2
      expect(strategy.calculateDelay(3)).toBe(800);   // 100 * 2^3
      expect(strategy.calculateDelay(4)).toBe(1000);  // Capped at maxDelay
    });

    test('adds jitter to prevent thundering herd', () => {
      const strategy = new RetryStrategy({
        initialDelay: 1000,
        jitter: true,
      });

      const delay = strategy.calculateDelay(0);

      // With jitter, delay should be between 50% and 100% of base delay
      expect(delay).toBeGreaterThanOrEqual(500);
      expect(delay).toBeLessThanOrEqual(1000);
    });

    test('determines retryability based on error type', () => {
      const strategy = new RetryStrategy({ maxRetries: 3 });

      // NetworkError is retryable
      expect(strategy.shouldRetry(new NetworkError('test'), 1)).toBe(true);

      // ValidationError is not retryable
      expect(strategy.shouldRetry(new ValidationError('test'), 1)).toBe(false);

      // ExternalAPIError with 5xx is retryable
      expect(strategy.shouldRetry(
        new ExternalAPIError('test', { statusCode: 503 }),
        1
      )).toBe(true);

      // ExternalAPIError with 4xx is not retryable
      expect(strategy.shouldRetry(
        new ExternalAPIError('test', { statusCode: 400 }),
        1
      )).toBe(false);

      // Max retries exceeded
      expect(strategy.shouldRetry(new NetworkError('test'), 3)).toBe(false);
    });
  });

  describe('Retry Executor', () => {
    test('executes successfully on first attempt', async () => {
      const executor = new RetryExecutor();
      const fn = jest.fn().mockResolvedValue('success');

      const result = await executor.execute(fn);

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(1);
    });

    test('retries on retryable error and succeeds', async () => {
      const executor = new RetryExecutor(
        new RetryStrategy({ maxRetries: 3, initialDelay: 10 })
      );

      let attemptCount = 0;
      const fn = jest.fn(async () => {
        attemptCount++;
        if (attemptCount < 3) {
          throw new NetworkError('Temporary failure');
        }
        return 'success';
      });

      const result = await executor.execute(fn);

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(3);
    });

    test('throws error after max retries', async () => {
      const executor = new RetryExecutor(
        new RetryStrategy({ maxRetries: 2, initialDelay: 10 })
      );

      const fn = jest.fn().mockRejectedValue(new NetworkError('Persistent failure'));

      await expect(executor.execute(fn)).rejects.toThrow('Persistent failure');
      expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });

    test('does not retry non-retryable errors', async () => {
      const executor = new RetryExecutor();
      const fn = jest.fn().mockRejectedValue(new ValidationError('Invalid input'));

      await expect(executor.execute(fn)).rejects.toThrow('Invalid input');
      expect(fn).toHaveBeenCalledTimes(1); // No retries
    });
  });

  describe('Circuit Breaker', () => {
    test('starts in CLOSED state', () => {
      const breaker = new CircuitBreaker();
      expect(breaker.state).toBe(CircuitState.CLOSED);
    });

    test('opens circuit after failure threshold', async () => {
      const breaker = new CircuitBreaker({
        failureThreshold: 3,
        timeout: 1000,
      });

      const failingFn = jest.fn().mockRejectedValue(new Error('Service down'));

      // Execute failing function 3 times
      for (let i = 0; i < 3; i++) {
        try {
          await breaker.execute(failingFn);
        } catch (err) {
          // Expected
        }
      }

      expect(breaker.state).toBe(CircuitState.OPEN);
      expect(breaker.failures).toBe(3);
    });

    test('rejects requests when circuit is OPEN', async () => {
      const breaker = new CircuitBreaker({
        failureThreshold: 1,
        timeout: 60000,
      });

      // Trigger circuit to open
      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch (err) {
        // Expected
      }

      expect(breaker.state).toBe(CircuitState.OPEN);

      // Next request should be rejected immediately
      await expect(
        breaker.execute(() => Promise.resolve('success'))
      ).rejects.toThrow('Circuit breaker is OPEN');
    });

    test('uses fallback when circuit is OPEN', async () => {
      const breaker = new CircuitBreaker({
        failureThreshold: 1,
        timeout: 60000,
      });

      // Trigger circuit to open
      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch (err) {
        // Expected
      }

      const fallback = jest.fn().mockReturnValue('fallback-value');
      const result = await breaker.execute(() => Promise.resolve('success'), fallback);

      expect(result).toBe('fallback-value');
      expect(fallback).toHaveBeenCalled();
    });

    test('transitions to HALF_OPEN after timeout', async () => {
      const breaker = new CircuitBreaker({
        failureThreshold: 1,
        timeout: 100, // 100ms timeout
      });

      // Trigger circuit to open
      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch (err) {
        // Expected
      }

      expect(breaker.state).toBe(CircuitState.OPEN);

      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 150));

      // Next request should transition to HALF_OPEN
      const successFn = jest.fn().mockResolvedValue('success');
      await breaker.execute(successFn);

      expect(successFn).toHaveBeenCalled();
    });

    test('closes circuit after success threshold in HALF_OPEN', async () => {
      const breaker = new CircuitBreaker({
        failureThreshold: 1,
        successThreshold: 2,
        timeout: 50,
      });

      // Open circuit
      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch (err) {
        // Expected
      }

      // Wait for timeout
      await new Promise(resolve => setTimeout(resolve, 100));

      // Execute 2 successful requests
      await breaker.execute(() => Promise.resolve('success'));
      await breaker.execute(() => Promise.resolve('success'));

      expect(breaker.state).toBe(CircuitState.CLOSED);
    });

    test('tracks statistics', async () => {
      const breaker = new CircuitBreaker();

      await breaker.execute(() => Promise.resolve('success'));

      try {
        await breaker.execute(() => Promise.reject(new Error('Failure')));
      } catch (err) {
        // Expected
      }

      const stats = breaker.getStats();

      expect(stats.totalRequests).toBe(2);
      expect(stats.totalSuccesses).toBe(1);
      expect(stats.totalFailures).toBe(1);
    });
  });

  describe('Dead Letter Queue', () => {
    test('enqueues failed operations', () => {
      const dlq = new DeadLetterQueue();

      dlq.enqueue({ operation: 'test', data: 'value' });

      expect(dlq.queue.length).toBe(1);
      expect(dlq.stats.totalEnqueued).toBe(1);
    });

    test('dequeues items in FIFO order', () => {
      const dlq = new DeadLetterQueue();

      dlq.enqueue({ id: 1 });
      dlq.enqueue({ id: 2 });
      dlq.enqueue({ id: 3 });

      expect(dlq.dequeue().operation.id).toBe(1);
      expect(dlq.dequeue().operation.id).toBe(2);
      expect(dlq.dequeue().operation.id).toBe(3);
      expect(dlq.dequeue()).toBeUndefined();
    });

    test('respects max size', () => {
      const dlq = new DeadLetterQueue({ maxSize: 3 });

      dlq.enqueue({ id: 1 });
      dlq.enqueue({ id: 2 });
      dlq.enqueue({ id: 3 });
      dlq.enqueue({ id: 4 }); // Should remove id: 1

      expect(dlq.queue.length).toBe(3);
      expect(dlq.dequeue().operation.id).toBe(2);
    });

    test('processes queue with handler', async () => {
      const dlq = new DeadLetterQueue();
      const handler = jest.fn().mockResolvedValue('processed');

      dlq.enqueue({ id: 1 });
      dlq.enqueue({ id: 2 });

      await dlq.process(handler);

      expect(handler).toHaveBeenCalledTimes(2);
      expect(dlq.queue.length).toBe(0);
      expect(dlq.stats.totalProcessed).toBe(2);
    });

    test('re-enqueues failed processing with attempt limit', async () => {
      const dlq = new DeadLetterQueue();
      const handler = jest.fn().mockRejectedValue(new Error('Processing failed'));

      dlq.enqueue({ id: 1 });

      await dlq.process(handler);

      // Item should be re-enqueued with incremented attempts
      expect(dlq.queue.length).toBe(1);
      expect(dlq.queue[0].attempts).toBe(1);
      expect(dlq.stats.totalFailed).toBe(1);
    });
  });

  describe('Error Handler Manager', () => {
    test('creates circuit breakers per service', () => {
      const manager = new ErrorHandlerManager();

      const breaker1 = manager.getCircuitBreaker('service1');
      const breaker2 = manager.getCircuitBreaker('service2');
      const breaker1Again = manager.getCircuitBreaker('service1');

      expect(breaker1).toBeInstanceOf(CircuitBreaker);
      expect(breaker2).toBeInstanceOf(CircuitBreaker);
      expect(breaker1).toBe(breaker1Again); // Same instance
      expect(breaker1).not.toBe(breaker2);  // Different instances
    });

    test('executes with full protection (retry + circuit breaker)', async () => {
      const manager = new ErrorHandlerManager();
      const fn = jest.fn().mockResolvedValue('success');

      const result = await manager.executeWithProtection(fn, {
        serviceName: 'test-service',
      });

      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(1);
    });

    test('adds critical errors to dead letter queue', async () => {
      const manager = new ErrorHandlerManager();

      const criticalError = new DatabaseError('Critical failure', {
        severity: ErrorSeverity.CRITICAL,
      });

      const fn = jest.fn().mockRejectedValue(criticalError);

      try {
        await manager.executeWithProtection(fn, {
          serviceName: 'test-service',
        });
      } catch (err) {
        // Expected
      }

      const stats = manager.getAllStats();
      expect(stats.deadLetterQueue.queueSize).toBeGreaterThan(0);
    });

    test('returns comprehensive statistics', async () => {
      const manager = new ErrorHandlerManager();

      await manager.executeWithProtection(
        () => Promise.resolve('success'),
        { serviceName: 'service1' }
      );

      const stats = manager.getAllStats();

      expect(stats.circuitBreakers).toHaveProperty('service1');
      expect(stats.deadLetterQueue).toBeDefined();
    });
  });
});
