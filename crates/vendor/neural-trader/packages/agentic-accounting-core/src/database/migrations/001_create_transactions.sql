-- Migration: Create transactions table
-- Description: Core transaction records with all buy/sell/trade events

CREATE TYPE transaction_type AS ENUM (
  'BUY',
  'SELL',
  'TRADE',
  'INCOME',
  'EXPENSE',
  'TRANSFER'
);

CREATE TABLE transactions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp TIMESTAMPTZ NOT NULL,
  type transaction_type NOT NULL,
  asset VARCHAR(50) NOT NULL,
  quantity DECIMAL(30, 18) NOT NULL CHECK (quantity > 0),
  price DECIMAL(30, 18) NOT NULL CHECK (price >= 0),
  fees DECIMAL(30, 18) NOT NULL DEFAULT 0 CHECK (fees >= 0),
  currency VARCHAR(10) NOT NULL DEFAULT 'USD',
  source VARCHAR(100) NOT NULL,
  source_id VARCHAR(255),
  taxable BOOLEAN NOT NULL DEFAULT true,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Ensure unique external transaction IDs per source
  CONSTRAINT unique_source_transaction UNIQUE (source, source_id)
);

-- Indexes for common queries
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_asset ON transactions(asset);
CREATE INDEX idx_transactions_type ON transactions(type);
CREATE INDEX idx_transactions_source ON transactions(source);
CREATE INDEX idx_transactions_taxable ON transactions(taxable) WHERE taxable = true;

-- Composite index for asset + timestamp queries
CREATE INDEX idx_transactions_asset_timestamp ON transactions(asset, timestamp DESC);

-- Index for metadata JSONB queries
CREATE INDEX idx_transactions_metadata ON transactions USING gin(metadata);

-- Add updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_transactions_updated_at
BEFORE UPDATE ON transactions
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Add comment
COMMENT ON TABLE transactions IS 'Core transaction records for all buy/sell/trade/income events';
