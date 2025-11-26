/**
 * Database Module Entry Point
 * Exports all database clients and utilities
 */

// Import for internal use
import * as postgresql from './postgresql';
import * as agentdb from './agentdb';

// PostgreSQL client
export {
  initializeDatabase,
  closeDatabase,
  getPool,
  query,
  transaction,
  checkPgVector,
  installPgVector,
  healthCheck,
  getStats,
} from './postgresql';

// AgentDB client
export {
  AgentDBClient,
  getAgentDB,
  closeAgentDB,
  type VectorRecord,
  type SearchResult,
  type AgentDBOptions,
} from './agentdb';

// Configuration
export {
  getDatabaseConfig,
  getMigrationConfig,
  getAgentDBConfig,
  type DatabaseConfig,
  type AgentDBConfig,
} from './config';

/**
 * Initialize all database connections
 */
export const initializeAllDatabases = async (): Promise<void> => {
  console.log('🔄 Initializing databases...');

  try {
    // Initialize PostgreSQL
    await postgresql.initializeDatabase();

    // Install pgvector if needed
    const hasPgVector = await postgresql.checkPgVector();
    if (!hasPgVector) {
      await postgresql.installPgVector();
    }

    // Initialize AgentDB
    const agentDB = agentdb.getAgentDB();
    await agentDB.initialize();

    console.log('✅ All databases initialized successfully');
  } catch (error) {
    console.error('❌ Database initialization failed:', error);
    throw error;
  }
};

/**
 * Close all database connections
 */
export const closeAllDatabases = async (): Promise<void> => {
  console.log('🔄 Closing databases...');

  try {
    await postgresql.closeDatabase();
    await agentdb.closeAgentDB();

    console.log('✅ All databases closed successfully');
  } catch (error) {
    console.error('❌ Error closing databases:', error);
    throw error;
  }
};

/**
 * Health check for all databases
 */
export const healthCheckAll = async (): Promise<{
  postgresql: Awaited<ReturnType<typeof postgresql.healthCheck>>;
  agentdb: Awaited<ReturnType<agentdb.AgentDBClient['healthCheck']>>;
}> => {
  const [postgresqlHealth, agentdbHealth] = await Promise.all([
    postgresql.healthCheck(),
    agentdb.getAgentDB().healthCheck(),
  ]);

  return { postgresql: postgresqlHealth, agentdb: agentdbHealth };
};

// Re-export types
import type { Pool, PoolClient, QueryResult } from 'pg';
export type { Pool, PoolClient, QueryResult };
