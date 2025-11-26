-- Migration: Create tax_lots table
-- Description: Individual tax lot tracking for cost basis calculations

CREATE TYPE accounting_method AS ENUM (
  'FIFO',
  'LIFO',
  'HIFO',
  'SPECIFIC_ID',
  'AVERAGE_COST'
);

CREATE TYPE lot_status AS ENUM (
  'OPEN',
  'PARTIAL',
  'CLOSED'
);

CREATE TABLE tax_lots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  transaction_id UUID NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
  asset VARCHAR(50) NOT NULL,
  acquired_date TIMESTAMPTZ NOT NULL,
  quantity DECIMAL(30, 18) NOT NULL CHECK (quantity >= 0),
  original_quantity DECIMAL(30, 18) NOT NULL CHECK (original_quantity > 0),
  cost_basis DECIMAL(30, 18) NOT NULL CHECK (cost_basis >= 0),
  unit_cost_basis DECIMAL(30, 18) NOT NULL CHECK (unit_cost_basis >= 0),
  currency VARCHAR(10) NOT NULL DEFAULT 'USD',
  source VARCHAR(100) NOT NULL,
  method accounting_method NOT NULL DEFAULT 'FIFO',
  status lot_status NOT NULL DEFAULT 'OPEN',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Ensure quantity doesn't exceed original
  CONSTRAINT check_lot_quantity CHECK (quantity <= original_quantity)
);

-- Indexes for lot queries
CREATE INDEX idx_tax_lots_transaction_id ON tax_lots(transaction_id);
CREATE INDEX idx_tax_lots_asset ON tax_lots(asset);
CREATE INDEX idx_tax_lots_status ON tax_lots(status) WHERE status != 'CLOSED';
CREATE INDEX idx_tax_lots_acquired_date ON tax_lots(acquired_date);
CREATE INDEX idx_tax_lots_method ON tax_lots(method);

-- Composite index for FIFO/LIFO queries (asset + acquired_date)
CREATE INDEX idx_tax_lots_asset_acquired ON tax_lots(asset, acquired_date ASC) WHERE status != 'CLOSED';

-- Composite index for HIFO queries (asset + unit_cost_basis)
CREATE INDEX idx_tax_lots_asset_cost ON tax_lots(asset, unit_cost_basis DESC) WHERE status != 'CLOSED';

-- Add updated_at trigger
CREATE TRIGGER update_tax_lots_updated_at
BEFORE UPDATE ON tax_lots
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Add comment
COMMENT ON TABLE tax_lots IS 'Individual tax lot tracking for cost basis and holding period calculations';
