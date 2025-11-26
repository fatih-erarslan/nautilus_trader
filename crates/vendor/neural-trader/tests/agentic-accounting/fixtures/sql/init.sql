-- Initialization script for test database
-- This runs automatically when the PostgreSQL container starts

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE transaction_type AS ENUM ('BUY', 'SELL', 'TRADE', 'INCOME', 'TRANSFER');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE term_type AS ENUM ('SHORT', 'LONG');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE agentic_accounting_test TO test_user;

-- Create schema
CREATE SCHEMA IF NOT EXISTS accounting;

-- Set default schema
ALTER DATABASE agentic_accounting_test SET search_path TO public, accounting;

COMMENT ON DATABASE agentic_accounting_test IS 'Test database for agentic accounting system';
