/**
 * Database Test Helpers
 *
 * Utilities for database setup, seeding, and cleanup in tests
 */

import { Client, Pool, PoolConfig } from 'pg';
import { Redis } from 'ioredis';

// ============================================================================
// PostgreSQL Test Database
// ============================================================================

export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
}

export const TEST_DB_CONFIG: DatabaseConfig = {
  host: process.env.TEST_DB_HOST || 'localhost',
  port: parseInt(process.env.TEST_DB_PORT || '5433'),
  database: process.env.TEST_DB_NAME || 'agentic_accounting_test',
  user: process.env.TEST_DB_USER || 'test_user',
  password: process.env.TEST_DB_PASSWORD || 'test_password',
};

/**
 * Creates a test database connection pool
 */
export function createTestDatabasePool(config: Partial<PoolConfig> = {}): Pool {
  return new Pool({
    ...TEST_DB_CONFIG,
    max: 10,
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 2000,
    ...config,
  });
}

/**
 * Creates a test database client
 */
export async function createTestDatabase(): Promise<Client> {
  const client = new Client(TEST_DB_CONFIG);
  await client.connect();
  return client;
}

/**
 * Runs database migrations
 */
export async function runMigrations(client: Client): Promise<void> {
  // Enable pgvector extension
  await client.query('CREATE EXTENSION IF NOT EXISTS vector');

  // Create tables
  await client.query(`
    CREATE TABLE IF NOT EXISTS transactions (
      id UUID PRIMARY KEY,
      type VARCHAR(20) NOT NULL,
      asset VARCHAR(20) NOT NULL,
      quantity DECIMAL(30, 18) NOT NULL,
      price DECIMAL(30, 18) NOT NULL,
      timestamp TIMESTAMP NOT NULL,
      source VARCHAR(100),
      fees DECIMAL(30, 18) DEFAULT 0,
      notes TEXT,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);

  await client.query(`
    CREATE TABLE IF NOT EXISTS tax_lots (
      id UUID PRIMARY KEY,
      asset VARCHAR(20) NOT NULL,
      quantity DECIMAL(30, 18) NOT NULL,
      cost_basis DECIMAL(30, 18) NOT NULL,
      acquired_date TIMESTAMP NOT NULL,
      transaction_id UUID REFERENCES transactions(id),
      disposed BOOLEAN DEFAULT FALSE,
      disposed_date TIMESTAMP,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);

  await client.query(`
    CREATE TABLE IF NOT EXISTS disposals (
      id UUID PRIMARY KEY,
      asset VARCHAR(20) NOT NULL,
      quantity DECIMAL(30, 18) NOT NULL,
      proceeds DECIMAL(30, 18) NOT NULL,
      cost_basis DECIMAL(30, 18) NOT NULL,
      gain DECIMAL(30, 18) NOT NULL,
      disposal_date TIMESTAMP NOT NULL,
      acquired_date TIMESTAMP NOT NULL,
      term VARCHAR(10) NOT NULL,
      wash_sale BOOLEAN DEFAULT FALSE,
      disallowed_loss DECIMAL(30, 18),
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);

  await client.query(`
    CREATE TABLE IF NOT EXISTS audit_trail (
      id UUID PRIMARY KEY,
      timestamp TIMESTAMP NOT NULL,
      action VARCHAR(50) NOT NULL,
      entity VARCHAR(50) NOT NULL,
      entity_id UUID NOT NULL,
      user_id VARCHAR(100),
      changes JSONB,
      hash VARCHAR(64) NOT NULL,
      previous_hash VARCHAR(64) NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);

  await client.query(`
    CREATE TABLE IF NOT EXISTS embeddings (
      id UUID PRIMARY KEY,
      entity_type VARCHAR(50) NOT NULL,
      entity_id UUID NOT NULL,
      embedding vector(768),
      created_at TIMESTAMP DEFAULT NOW()
    )
  `);

  // Create indexes
  await client.query('CREATE INDEX IF NOT EXISTS idx_transactions_asset ON transactions(asset)');
  await client.query('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)');
  await client.query('CREATE INDEX IF NOT EXISTS idx_tax_lots_asset ON tax_lots(asset)');
  await client.query('CREATE INDEX IF NOT EXISTS idx_tax_lots_acquired_date ON tax_lots(acquired_date)');
  await client.query('CREATE INDEX IF NOT EXISTS idx_audit_trail_entity ON audit_trail(entity, entity_id)');
  await client.query('CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_type, entity_id)');
}

/**
 * Cleans all test data from database
 */
export async function cleanDatabase(client: Client): Promise<void> {
  await client.query('TRUNCATE TABLE embeddings CASCADE');
  await client.query('TRUNCATE TABLE audit_trail CASCADE');
  await client.query('TRUNCATE TABLE disposals CASCADE');
  await client.query('TRUNCATE TABLE tax_lots CASCADE');
  await client.query('TRUNCATE TABLE transactions CASCADE');
}

/**
 * Drops all test tables
 */
export async function dropTables(client: Client): Promise<void> {
  await client.query('DROP TABLE IF EXISTS embeddings CASCADE');
  await client.query('DROP TABLE IF EXISTS audit_trail CASCADE');
  await client.query('DROP TABLE IF EXISTS disposals CASCADE');
  await client.query('DROP TABLE IF EXISTS tax_lots CASCADE');
  await client.query('DROP TABLE IF EXISTS transactions CASCADE');
}

/**
 * Seeds database with test data
 */
export async function seedDatabase(client: Client, data: any): Promise<void> {
  // Insert transactions
  if (data.transactions) {
    for (const tx of data.transactions) {
      await client.query(
        `INSERT INTO transactions (id, type, asset, quantity, price, timestamp, source, fees, notes)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
        [tx.id, tx.type, tx.asset, tx.quantity, tx.price, tx.timestamp, tx.source, tx.fees, tx.notes]
      );
    }
  }

  // Insert tax lots
  if (data.lots) {
    for (const lot of data.lots) {
      await client.query(
        `INSERT INTO tax_lots (id, asset, quantity, cost_basis, acquired_date, transaction_id, disposed)
         VALUES ($1, $2, $3, $4, $5, $6, $7)`,
        [lot.id, lot.asset, lot.quantity, lot.costBasis, lot.acquiredDate, lot.transactionId, lot.disposed]
      );
    }
  }
}

// ============================================================================
// Redis Test Database
// ============================================================================

export const TEST_REDIS_CONFIG = {
  host: process.env.TEST_REDIS_HOST || 'localhost',
  port: parseInt(process.env.TEST_REDIS_PORT || '6380'),
  password: process.env.TEST_REDIS_PASSWORD || 'test_redis_password',
  db: parseInt(process.env.TEST_REDIS_DB || '0'),
};

/**
 * Creates a test Redis client
 */
export function createTestRedis(): Redis {
  return new Redis(TEST_REDIS_CONFIG);
}

/**
 * Flushes all Redis test data
 */
export async function flushRedis(redis: Redis): Promise<void> {
  await redis.flushdb();
}

// ============================================================================
// AgentDB Test Helpers
// ============================================================================

export interface AgentDBTestConfig {
  dbPath?: string;
  dimensions?: number;
}

/**
 * Creates a test AgentDB instance
 */
export async function createTestAgentDB(config: AgentDBTestConfig = {}) {
  const AgentDB = require('agentdb');

  const db = new AgentDB({
    dbPath: config.dbPath || ':memory:',
    dimensions: config.dimensions || 768,
    quantization: 'none', // Disable for tests
  });

  await db.init();
  return db;
}

/**
 * Seeds AgentDB with test vectors
 */
export async function seedAgentDB(db: any, count: number): Promise<void> {
  const vectors = Array.from({ length: count }, (_, i) => ({
    id: `test-vector-${i}`,
    vector: Array.from({ length: 768 }, () => Math.random()),
    metadata: { index: i, type: 'test' },
  }));

  await db.addVectors(vectors);
}

// ============================================================================
// Test Database Lifecycle
// ============================================================================

export class TestDatabaseLifecycle {
  private pool: Pool | null = null;
  private redis: Redis | null = null;
  private agentdb: any = null;

  async setup(): Promise<void> {
    // Setup PostgreSQL
    this.pool = createTestDatabasePool();
    const client = await this.pool.connect();
    try {
      await runMigrations(client);
    } finally {
      client.release();
    }

    // Setup Redis
    this.redis = createTestRedis();
    await this.redis.ping();

    // Setup AgentDB
    this.agentdb = await createTestAgentDB();
  }

  async cleanup(): Promise<void> {
    // Cleanup PostgreSQL
    if (this.pool) {
      const client = await this.pool.connect();
      try {
        await cleanDatabase(client);
      } finally {
        client.release();
      }
    }

    // Cleanup Redis
    if (this.redis) {
      await flushRedis(this.redis);
    }

    // AgentDB cleanup (in-memory, nothing needed)
  }

  async teardown(): Promise<void> {
    if (this.pool) {
      await this.pool.end();
    }

    if (this.redis) {
      await this.redis.quit();
    }

    if (this.agentdb) {
      await this.agentdb.close();
    }
  }

  getPool(): Pool {
    if (!this.pool) throw new Error('Database pool not initialized');
    return this.pool;
  }

  getRedis(): Redis {
    if (!this.redis) throw new Error('Redis not initialized');
    return this.redis;
  }

  getAgentDB(): any {
    if (!this.agentdb) throw new Error('AgentDB not initialized');
    return this.agentdb;
  }
}

// ============================================================================
// Export all helpers
// ============================================================================

export const dbHelpers = {
  createTestDatabasePool,
  createTestDatabase,
  runMigrations,
  cleanDatabase,
  dropTables,
  seedDatabase,
  createTestRedis,
  flushRedis,
  createTestAgentDB,
  seedAgentDB,
  TestDatabaseLifecycle,
};
