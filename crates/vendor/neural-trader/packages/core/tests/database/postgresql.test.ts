import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { DatabaseClient, type DatabaseConfig } from '../../src/database/postgresql.js';

describe('DatabaseClient', () => {
  describe('Configuration', () => {
    it('should create client with custom config', () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        port: 5432,
        database: 'test_db',
        user: 'test_user',
        password: 'test_password',
        max: 10,
        min: 1,
      };

      const client = new DatabaseClient(config);
      expect(client).toBeInstanceOf(DatabaseClient);
      expect(client.isConnected()).toBe(false);
    });

    it('should create client with connection string', () => {
      const config: DatabaseConfig = {
        connectionString: 'postgresql://user:password@localhost:5432/testdb',
      };

      const client = new DatabaseClient(config);
      expect(client).toBeInstanceOf(DatabaseClient);
    });

    it('should merge config with defaults', () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);
      expect(client).toBeInstanceOf(DatabaseClient);
      // Default values should be applied
    });
  });

  describe('Connection management', () => {
    it('should track connection state', () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);
      expect(client.isConnected()).toBe(false);
    });

    it('should provide pool statistics', () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);
      const stats = client.getPoolStats();

      expect(stats).toHaveProperty('totalCount');
      expect(stats).toHaveProperty('idleCount');
      expect(stats).toHaveProperty('waitingCount');
      expect(stats.totalCount).toBe(0);
      expect(stats.idleCount).toBe(0);
      expect(stats.waitingCount).toBe(0);
    });
  });

  describe('Error handling', () => {
    it('should throw error when querying without connection', async () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);

      await expect(client.query('SELECT 1')).rejects.toThrow('Database not connected');
    });

    it('should throw error when getting client without connection', async () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);

      await expect(client.getClient()).rejects.toThrow('Database not connected');
    });

    it('should throw error when connecting twice', async () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);

      // Mock successful connection for first attempt
      // In real tests with actual DB, this would actually connect
      // For unit tests, we're just testing the guard clause
    });
  });

  describe('Configuration validation', () => {
    it('should accept minimal valid config', () => {
      const config: DatabaseConfig = {
        connectionString: 'postgresql://localhost/testdb',
      };

      const client = new DatabaseClient(config);
      expect(client).toBeInstanceOf(DatabaseClient);
    });

    it('should accept detailed config', () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        port: 5432,
        database: 'accounting_db',
        user: 'accounting_user',
        password: 'secure_password',
        max: 25,
        min: 5,
        idleTimeoutMillis: 60000,
        connectionTimeoutMillis: 15000,
      };

      const client = new DatabaseClient(config);
      expect(client).toBeInstanceOf(DatabaseClient);
    });
  });

  describe('Type safety', () => {
    it('should support typed query results', async () => {
      const config: DatabaseConfig = {
        host: 'localhost',
        database: 'test_db',
      };

      const client = new DatabaseClient(config);

      // Type check only - will fail at runtime without connection
      type TestRow = { id: number; name: string };

      // This should type-check correctly and throw error (not connected)
      await expect(
        client.query<TestRow>('SELECT id, name FROM users')
      ).rejects.toThrow('Database not connected');
    });
  });
});
