/**
 * Database Configuration
 * PostgreSQL connection settings with pgvector support
 */

export interface DatabaseConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  password: string;
  max: number;
  idleTimeoutMillis: number;
  connectionTimeoutMillis: number;
  ssl?: boolean | { rejectUnauthorized: boolean };
}

export const getDatabaseConfig = (): DatabaseConfig => {
  return {
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '5432', 10),
    database: process.env.DB_NAME || 'agentic_accounting',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASSWORD || 'postgres',
    max: parseInt(process.env.DB_POOL_SIZE || '20', 10),
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 10000,
    ssl: process.env.DB_SSL === 'true' ? { rejectUnauthorized: false } : false,
  };
};

export const getMigrationConfig = () => {
  const config = getDatabaseConfig();
  return {
    databaseUrl: `postgresql://${config.user}:${config.password}@${config.host}:${config.port}/${config.database}`,
    migrationsTable: 'pgmigrations',
    dir: 'src/database/migrations',
    direction: 'up',
    checkOrder: true,
    verbose: true,
  };
};

// AgentDB Configuration
export interface AgentDBConfig {
  dimensions: number;
  distanceMetric: 'cosine' | 'euclidean' | 'dot';
  indexType: 'hnsw';
  hnswParams: {
    m: number;
    efConstruction: number;
    efSearch: number;
  };
  quantization: 'none' | 'int8' | 'binary';
  persistence: {
    enabled: boolean;
    path: string;
    syncInterval: number;
  };
}

export const getAgentDBConfig = (): AgentDBConfig => {
  return {
    dimensions: parseInt(process.env.AGENTDB_DIMENSIONS || '768', 10),
    distanceMetric: (process.env.AGENTDB_METRIC as any) || 'cosine',
    indexType: 'hnsw',
    hnswParams: {
      m: 16,
      efConstruction: 200,
      efSearch: 100,
    },
    quantization: (process.env.AGENTDB_QUANTIZATION as any) || 'int8',
    persistence: {
      enabled: process.env.AGENTDB_PERSISTENCE !== 'false',
      path: process.env.AGENTDB_PATH || './data/agentdb',
      syncInterval: 60000,
    },
  };
};
