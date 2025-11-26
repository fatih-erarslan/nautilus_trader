-- Migration: Create disposals table
-- Description: Records of asset sales/trades with realized gains/losses

CREATE TYPE capital_gain_term AS ENUM (
  'SHORT',
  'LONG'
);

CREATE TABLE disposals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lot_id UUID NOT NULL REFERENCES tax_lots(id) ON DELETE CASCADE,
  transaction_id UUID NOT NULL REFERENCES transactions(id) ON DELETE CASCADE,
  disposal_date TIMESTAMPTZ NOT NULL,
  quantity DECIMAL(30, 18) NOT NULL CHECK (quantity > 0),
  proceeds DECIMAL(30, 18) NOT NULL CHECK (proceeds >= 0),
  cost_basis DECIMAL(30, 18) NOT NULL CHECK (cost_basis >= 0),
  gain DECIMAL(30, 18) NOT NULL,
  term capital_gain_term NOT NULL,
  tax_year INTEGER NOT NULL CHECK (tax_year >= 2000 AND tax_year <= 2100),
  method accounting_method NOT NULL,
  wash_sale_disallowed DECIMAL(30, 18) DEFAULT 0 CHECK (wash_sale_disallowed >= 0),
  wash_sale_deferred_to UUID REFERENCES tax_lots(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for disposal queries
CREATE INDEX idx_disposals_lot_id ON disposals(lot_id);
CREATE INDEX idx_disposals_transaction_id ON disposals(transaction_id);
CREATE INDEX idx_disposals_disposal_date ON disposals(disposal_date DESC);
CREATE INDEX idx_disposals_tax_year ON disposals(tax_year);
CREATE INDEX idx_disposals_term ON disposals(term);

-- Composite index for tax reporting (tax_year + term)
CREATE INDEX idx_disposals_year_term ON disposals(tax_year, term);

-- Index for wash sale tracking
CREATE INDEX idx_disposals_wash_sale ON disposals(wash_sale_deferred_to) WHERE wash_sale_deferred_to IS NOT NULL;

-- Add updated_at trigger
CREATE TRIGGER update_disposals_updated_at
BEFORE UPDATE ON disposals
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate holding period
CREATE OR REPLACE FUNCTION calculate_holding_period(
  acquired_date TIMESTAMPTZ,
  disposal_date TIMESTAMPTZ
) RETURNS capital_gain_term AS $$
BEGIN
  IF disposal_date - acquired_date >= INTERVAL '1 year' THEN
    RETURN 'LONG'::capital_gain_term;
  ELSE
    RETURN 'SHORT'::capital_gain_term;
  END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Add comment
COMMENT ON TABLE disposals IS 'Records of asset disposals with realized gains/losses and wash sale tracking';
