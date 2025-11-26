/**
 * Async testing utilities
 */

/**
 * Wait for a promise with timeout
 */
export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage = 'Operation timed out'
): Promise<T> {
  let timeoutHandle: NodeJS.Timeout;

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutHandle = setTimeout(() => reject(new Error(errorMessage)), timeoutMs);
  });

  try {
    return await Promise.race([promise, timeoutPromise]);
  } finally {
    clearTimeout(timeoutHandle!);
  }
}

/**
 * Run promises in parallel with concurrency limit
 */
export async function parallelLimit<T>(
  items: T[],
  concurrency: number,
  fn: (item: T) => Promise<any>
): Promise<any[]> {
  const results: any[] = [];
  const executing: Promise<any>[] = [];

  for (const item of items) {
    const promise = fn(item).then(result => {
      executing.splice(executing.indexOf(promise), 1);
      return result;
    });

    results.push(promise);
    executing.push(promise);

    if (executing.length >= concurrency) {
      await Promise.race(executing);
    }
  }

  return Promise.all(results);
}

/**
 * Debounce async function calls
 */
export function debounceAsync<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  delayMs: number
): T {
  let timeoutHandle: NodeJS.Timeout | null = null;
  let pendingPromise: Promise<any> | null = null;

  return ((...args: any[]) => {
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
    }

    if (!pendingPromise) {
      pendingPromise = new Promise((resolve, reject) => {
        timeoutHandle = setTimeout(async () => {
          try {
            const result = await fn(...args);
            resolve(result);
          } catch (error) {
            reject(error);
          } finally {
            pendingPromise = null;
            timeoutHandle = null;
          }
        }, delayMs);
      });
    }

    return pendingPromise;
  }) as T;
}

/**
 * Throttle async function calls
 */
export function throttleAsync<T extends (...args: any[]) => Promise<any>>(
  fn: T,
  limitMs: number
): T {
  let lastCall = 0;
  let pendingPromise: Promise<any> | null = null;

  return (async (...args: any[]) => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCall;

    if (timeSinceLastCall >= limitMs) {
      lastCall = now;
      return fn(...args);
    }

    if (!pendingPromise) {
      const delay = limitMs - timeSinceLastCall;
      pendingPromise = new Promise((resolve) => {
        setTimeout(async () => {
          lastCall = Date.now();
          const result = await fn(...args);
          pendingPromise = null;
          resolve(result);
        }, delay);
      });
    }

    return pendingPromise;
  }) as T;
}

/**
 * Create a deferred promise
 */
export interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
}

export function createDeferred<T>(): Deferred<T> {
  let resolve: (value: T) => void;
  let reject: (error: Error) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve: resolve!, reject: reject! };
}

/**
 * Wait for all promises to settle (fulfilled or rejected)
 */
export async function allSettled<T>(
  promises: Promise<T>[]
): Promise<Array<{ status: 'fulfilled' | 'rejected'; value?: T; reason?: any }>> {
  return Promise.all(
    promises.map(promise =>
      promise
        .then(value => ({ status: 'fulfilled' as const, value }))
        .catch(reason => ({ status: 'rejected' as const, reason }))
    )
  );
}
