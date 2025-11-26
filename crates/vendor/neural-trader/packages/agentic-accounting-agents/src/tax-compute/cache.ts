/**
 * Result Caching System
 *
 * Caches tax calculation results with TTL
 * Invalidates cache on new transactions
 */

import crypto from 'crypto';
import { TaxCalculationResult, TaxMethod } from './calculator-wrapper';

export interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  ttl: number;
  metadata?: Record<string, unknown>;
}

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  hitRate: number;
}

export class TaxCalculationCache {
  private cache: Map<string, CacheEntry<TaxCalculationResult>> = new Map();
  private hits: number = 0;
  private misses: number = 0;
  private readonly defaultTtl: number;
  private readonly maxSize: number;

  constructor(
    defaultTtl: number = 24 * 60 * 60 * 1000, // 24 hours
    maxSize: number = 1000
  ) {
    this.defaultTtl = defaultTtl;
    this.maxSize = maxSize;
  }

  /**
   * Generate cache key from calculation parameters
   */
  generateKey(
    method: TaxMethod,
    saleId: string,
    lotIds: string[]
  ): string {
    const data = JSON.stringify({
      method,
      saleId,
      lotIds: [...lotIds].sort(), // Sort for consistency
    });

    return crypto
      .createHash('sha256')
      .update(data)
      .digest('hex')
      .substring(0, 16);
  }

  /**
   * Get cached result
   */
  get(key: string): TaxCalculationResult | null {
    const entry = this.cache.get(key);

    if (!entry) {
      this.misses++;
      return null;
    }

    // Check if expired
    const now = Date.now();
    if (now - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      this.misses++;
      return null;
    }

    this.hits++;
    return entry.value;
  }

  /**
   * Store result in cache
   */
  set(
    key: string,
    value: TaxCalculationResult,
    ttl?: number,
    metadata?: Record<string, unknown>
  ): void {
    // Enforce max size (LRU eviction)
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) {
        this.cache.delete(firstKey);
      }
    }

    const entry: CacheEntry<TaxCalculationResult> = {
      key,
      value,
      timestamp: Date.now(),
      ttl: ttl || this.defaultTtl,
      metadata,
    };

    this.cache.set(key, entry);
  }

  /**
   * Invalidate cache entries matching pattern
   */
  invalidate(pattern?: string): number {
    if (!pattern) {
      // Clear all
      const size = this.cache.size;
      this.cache.clear();
      return size;
    }

    let removed = 0;
    const keysToDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (entry.metadata?.saleId === pattern || entry.metadata?.asset === pattern) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      this.cache.delete(key);
      removed++;
    }

    return removed;
  }

  /**
   * Invalidate all entries for a specific asset
   */
  invalidateAsset(asset: string): number {
    let removed = 0;
    const keysToDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (entry.metadata?.asset === asset) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      this.cache.delete(key);
      removed++;
    }

    return removed;
  }

  /**
   * Invalidate expired entries
   */
  cleanup(): number {
    const now = Date.now();
    const keysToDelete: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        keysToDelete.push(key);
      }
    }

    for (const key of keysToDelete) {
      this.cache.delete(key);
    }

    return keysToDelete.length;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const total = this.hits + this.misses;
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.cache.size,
      hitRate: total > 0 ? this.hits / total : 0,
    };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Clear all cache
   */
  clear(): void {
    this.cache.clear();
    this.resetStats();
  }

  /**
   * Get cache size
   */
  size(): number {
    return this.cache.size;
  }
}
