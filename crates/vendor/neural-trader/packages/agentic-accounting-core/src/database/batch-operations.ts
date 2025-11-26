/**
 * Batch Database Operations
 * Optimized bulk insert and update operations
 */

import { getPool } from './postgresql';
import { PoolClient } from 'pg';

export interface BatchInsertOptions {
  batchSize?: number;
  onProgress?: (processed: number, total: number) => void;
}

/**
 * Batch insert records efficiently
 */
export async function batchInsert<T extends Record<string, any>>(
  table: string,
  columns: string[],
  records: T[],
  options: BatchInsertOptions = {}
): Promise<number> {
  const { batchSize = 500, onProgress } = options;
  const pool = getPool();
  let totalInserted = 0;

  // Process in batches
  for (let i = 0; i < records.length; i += batchSize) {
    const batch = records.slice(i, i + batchSize);

    // Build parameterized query
    const values: any[] = [];
    const valuePlaceholders: string[] = [];

    batch.forEach((record, batchIdx) => {
      const rowPlaceholders: string[] = [];

      columns.forEach((col, colIdx) => {
        const paramIdx = batchIdx * columns.length + colIdx + 1;
        rowPlaceholders.push(`$${paramIdx}`);
        values.push(record[col]);
      });

      valuePlaceholders.push(`(${rowPlaceholders.join(', ')})`);
    });

    const query = `
      INSERT INTO ${table} (${columns.join(', ')})
      VALUES ${valuePlaceholders.join(', ')}
      ON CONFLICT DO NOTHING
    `;

    await pool.query(query, values);
    totalInserted += batch.length;

    if (onProgress) {
      onProgress(totalInserted, records.length);
    }
  }

  return totalInserted;
}

/**
 * Batch update records efficiently using CASE statements
 */
export async function batchUpdate<T extends Record<string, any>>(
  table: string,
  idColumn: string,
  updates: { id: any; changes: Partial<T> }[],
  options: BatchInsertOptions = {}
): Promise<number> {
  if (updates.length === 0) return 0;

  const { batchSize = 500, onProgress } = options;
  const pool = getPool();
  let totalUpdated = 0;

  // Get all columns to update
  const updateColumns = new Set<string>();
  updates.forEach(u => Object.keys(u.changes).forEach(k => updateColumns.add(k)));
  const columns = Array.from(updateColumns);

  // Process in batches
  for (let i = 0; i < updates.length; i += batchSize) {
    const batch = updates.slice(i, i + batchSize);
    const ids = batch.map(u => u.id);

    // Build CASE statements for each column
    const caseStatements = columns.map(col => {
      const cases = batch
        .filter(u => u.changes[col] !== undefined)
        .map((u, idx) => `WHEN ${idColumn} = $${idx + 1} THEN $${ids.length + batch.indexOf(u) * columns.length + columns.indexOf(col) + 1}`)
        .join(' ');

      return `${col} = CASE ${cases} ELSE ${col} END`;
    });

    // Build values array
    const values = [
      ...ids,
      ...batch.flatMap(u => columns.map(col => u.changes[col]))
    ];

    const query = `
      UPDATE ${table}
      SET ${caseStatements.join(', ')}
      WHERE ${idColumn} IN (${ids.map((_, idx) => `$${idx + 1}`).join(', ')})
    `;

    await pool.query(query, values);
    totalUpdated += batch.length;

    if (onProgress) {
      onProgress(totalUpdated, updates.length);
    }
  }

  return totalUpdated;
}

/**
 * Execute operations in a transaction with retries
 */
export async function withTransaction<T>(
  operations: (client: PoolClient) => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  const pool = getPool();
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const client = await pool.connect();

    try {
      await client.query('BEGIN');
      const result = await operations(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on certain errors
      if (error instanceof Error && error.message.includes('unique constraint')) {
        throw error;
      }

      // Exponential backoff
      if (attempt < maxRetries - 1) {
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 100));
      }
    } finally {
      client.release();
    }
  }

  throw lastError || new Error('Transaction failed after retries');
}
