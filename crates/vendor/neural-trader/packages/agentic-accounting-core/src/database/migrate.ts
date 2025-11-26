/**
 * Database Migration Runner
 * Runs SQL migration files in order
 */

import { readdir, readFile } from 'fs/promises';
import { join } from 'path';
import { Pool } from 'pg';
import { getDatabaseConfig } from './config';

interface Migration {
  id: number;
  name: string;
  path: string;
  sql: string;
}

/**
 * Get all migration files
 */
async function getMigrations(dir: string): Promise<Migration[]> {
  const files = await readdir(dir);
  const sqlFiles = files.filter((f) => f.endsWith('.sql')).sort();

  const migrations: Migration[] = [];

  for (const file of sqlFiles) {
    const match = file.match(/^(\d+)_(.+)\.sql$/);
    if (!match) continue;

    const [, idStr, name] = match;
    const id = parseInt(idStr, 10);
    const path = join(dir, file);
    const sql = await readFile(path, 'utf-8');

    migrations.push({ id, name, path, sql });
  }

  return migrations;
}

/**
 * Create migrations table if it doesn't exist
 */
async function createMigrationsTable(pool: Pool): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id INTEGER PRIMARY KEY,
      name VARCHAR(255) NOT NULL,
      executed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
  `);
}

/**
 * Get executed migrations
 */
async function getExecutedMigrations(pool: Pool): Promise<Set<number>> {
  const result = await pool.query(
    'SELECT id FROM schema_migrations ORDER BY id'
  );
  return new Set(result.rows.map((r) => r.id));
}

/**
 * Execute a migration
 */
async function executeMigration(
  pool: Pool,
  migration: Migration
): Promise<void> {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    console.log(`🔄 Running migration ${migration.id}: ${migration.name}`);

    // Execute migration SQL
    await client.query(migration.sql);

    // Record migration
    await client.query(
      'INSERT INTO schema_migrations (id, name) VALUES ($1, $2)',
      [migration.id, migration.name]
    );

    await client.query('COMMIT');
    console.log(`✅ Migration ${migration.id} completed`);
  } catch (error) {
    await client.query('ROLLBACK');
    console.error(`❌ Migration ${migration.id} failed:`, error);
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Run all pending migrations
 */
export async function runMigrations(
  migrationsDir?: string
): Promise<void> {
  const dir = migrationsDir || join(__dirname, 'migrations');
  const config = getDatabaseConfig();
  const pool = new Pool(config);

  try {
    console.log('🔄 Starting database migrations...');

    // Create migrations table
    await createMigrationsTable(pool);

    // Get all migrations
    const migrations = await getMigrations(dir);
    console.log(`📋 Found ${migrations.length} migration(s)`);

    // Get executed migrations
    const executed = await getExecutedMigrations(pool);
    console.log(`✅ ${executed.size} migration(s) already executed`);

    // Run pending migrations
    const pending = migrations.filter((m) => !executed.has(m.id));

    if (pending.length === 0) {
      console.log('✅ No pending migrations');
      return;
    }

    console.log(`🔄 Running ${pending.length} pending migration(s)...`);

    for (const migration of pending) {
      await executeMigration(pool, migration);
    }

    console.log('✅ All migrations completed successfully');
  } catch (error) {
    console.error('❌ Migration failed:', error);
    throw error;
  } finally {
    await pool.end();
  }
}

/**
 * Run seed files
 */
export async function runSeeds(seedsDir?: string): Promise<void> {
  const dir = seedsDir || join(__dirname, 'seeds');
  const config = getDatabaseConfig();
  const pool = new Pool(config);

  try {
    console.log('🔄 Starting database seeds...');

    const seeds = await getMigrations(dir);
    console.log(`📋 Found ${seeds.length} seed file(s)`);

    for (const seed of seeds) {
      console.log(`🔄 Running seed: ${seed.name}`);
      await pool.query(seed.sql);
      console.log(`✅ Seed ${seed.name} completed`);
    }

    console.log('✅ All seeds completed successfully');
  } catch (error) {
    console.error('❌ Seeding failed:', error);
    throw error;
  } finally {
    await pool.end();
  }
}

/**
 * Rollback last migration
 */
export async function rollbackMigration(): Promise<void> {
  const config = getDatabaseConfig();
  const pool = new Pool(config);

  try {
    // Get last migration
    const result = await pool.query(
      'SELECT id, name FROM schema_migrations ORDER BY id DESC LIMIT 1'
    );

    if (result.rows.length === 0) {
      console.log('⚠️  No migrations to rollback');
      return;
    }

    const { id, name } = result.rows[0];
    console.log(`🔄 Rolling back migration ${id}: ${name}`);

    // Note: SQL migrations don't have down migrations by default
    // This would require separate down migration files
    console.warn('⚠️  Manual rollback required. SQL migrations do not have automatic down migrations.');
    console.warn('   Please create and run a down migration manually.');

    // Remove from migrations table
    await pool.query('DELETE FROM schema_migrations WHERE id = $1', [id]);
    console.log(`✅ Migration ${id} removed from tracking`);
  } catch (error) {
    console.error('❌ Rollback failed:', error);
    throw error;
  } finally {
    await pool.end();
  }
}

// CLI interface
if (require.main === module) {
  const command = process.argv[2];

  (async () => {
    try {
      switch (command) {
        case 'up':
          await runMigrations();
          break;
        case 'seed':
          await runSeeds();
          break;
        case 'rollback':
          await rollbackMigration();
          break;
        case 'reset':
          await rollbackMigration();
          await runMigrations();
          break;
        default:
          console.log('Usage: ts-node migrate.ts <up|seed|rollback|reset>');
          process.exit(1);
      }
    } catch (error) {
      console.error(error);
      process.exit(1);
    }
  })();
}
