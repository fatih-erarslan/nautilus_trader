/**
 * PostgreSQL Database Client
 * Connection pool and query interface with pgvector support
 */

import { Pool, PoolClient, QueryResult } from 'pg';
import type { QueryResultRow } from 'pg';
import { getDatabaseConfig } from './config';
import { getQueryCache } from './query-cache';

let pool: Pool | null = null;

/**
 * Initialize database connection pool
 */
export const initializeDatabase = async (): Promise<Pool> => {
  if (pool) {
    return pool;
  }

  const config = getDatabaseConfig();

  // Optimized connection pool configuration
  pool = new Pool({
    ...config,
    max: 20, // Maximum pool size
    min: 5, // Minimum pool size (keep connections warm)
    idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
    connectionTimeoutMillis: 2000, // Return error after 2 seconds if no connection available
    maxUses: 7500, // Close connections after 7500 uses to prevent memory leaks
  });

  // Test connection
  try {
    const client = await pool.connect();
    await client.query('SELECT NOW()');
    console.log('✅ Database connection established');
    client.release();
  } catch (error) {
    console.error('❌ Database connection failed:', error);
    throw error;
  }

  // Handle pool errors
  pool.on('error', (err) => {
    console.error('Unexpected database pool error:', err);
  });

  return pool;
};

/**
 * Get database pool instance
 */
export const getPool = (): Pool => {
  if (!pool) {
    throw new Error('Database not initialized. Call initializeDatabase() first.');
  }
  return pool;
};

/**
 * Close database connection pool
 */
export const closeDatabase = async (): Promise<void> => {
  if (pool) {
    await pool.end();
    pool = null;
    console.log('✅ Database connection closed');
  }
};

/**
 * Execute a query with parameters and optional caching
 */
export const query = async <T extends QueryResultRow = any>(
  text: string,
  params?: any[],
  options?: { cache?: boolean; cacheTtl?: number }
): Promise<QueryResult<T>> => {
  // Generate cache key for SELECT queries
  const isSelect = text.trim().toUpperCase().startsWith('SELECT');
  const useCache = options?.cache && isSelect;

  if (useCache) {
    const cache = getQueryCache({ ttl: options.cacheTtl });
    const cacheKey = `query:${text}:${JSON.stringify(params || [])}`;
    const cached = cache.get<QueryResult<T>>(cacheKey);

    if (cached) {
      return cached;
    }

    // Execute query and cache result
    const client = getPool();
    const result = await client.query<T>(text, params);
    cache.set(cacheKey, result);
    return result;
  }

  const client = getPool();
  return client.query<T>(text, params);
};

/**
 * Execute a transaction
 */
export const transaction = async <T>(
  callback: (client: PoolClient) => Promise<T>
): Promise<T> => {
  const client = await getPool().connect();

  try {
    await client.query('BEGIN');
    const result = await callback(client);
    await client.query('COMMIT');
    return result;
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
};

/**
 * Check if pgvector extension is installed
 */
export const checkPgVector = async (): Promise<boolean> => {
  try {
    const result = await query(
      "SELECT * FROM pg_extension WHERE extname = 'vector'"
    );
    return result.rows.length > 0;
  } catch (error) {
    return false;
  }
};

/**
 * Install pgvector extension
 */
export const installPgVector = async (): Promise<void> => {
  try {
    await query('CREATE EXTENSION IF NOT EXISTS vector');
    console.log('✅ pgvector extension installed');
  } catch (error) {
    console.error('❌ Failed to install pgvector:', error);
    throw error;
  }
};

/**
 * Health check
 */
export const healthCheck = async (): Promise<{
  healthy: boolean;
  latency: number;
  pgvector: boolean;
}> => {
  const start = Date.now();

  try {
    await query('SELECT 1');
    const latency = Date.now() - start;
    const pgvector = await checkPgVector();

    return {
      healthy: true,
      latency,
      pgvector,
    };
  } catch (error) {
    return {
      healthy: false,
      latency: Date.now() - start,
      pgvector: false,
    };
  }
};

/**
 * Get database statistics
 */
export const getStats = async (): Promise<{
  totalConnections: number;
  idleConnections: number;
  waitingConnections: number;
}> => {
  const pool = getPool();

  return {
    totalConnections: pool.totalCount,
    idleConnections: pool.idleCount,
    waitingConnections: pool.waitingCount,
  };
};
