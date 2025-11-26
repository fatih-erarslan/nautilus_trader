-- Migration: Create tax_summaries table
-- Description: Annual tax summaries with aggregated gains/losses

CREATE TYPE income_type AS ENUM (
  'INTEREST',
  'DIVIDEND',
  'STAKING',
  'MINING',
  'AIRDROP'
);

CREATE TABLE tax_summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tax_year INTEGER NOT NULL CHECK (tax_year >= 2000 AND tax_year <= 2100),
  short_term_gains DECIMAL(30, 18) NOT NULL DEFAULT 0,
  long_term_gains DECIMAL(30, 18) NOT NULL DEFAULT 0,
  total_gains DECIMAL(30, 18) NOT NULL DEFAULT 0,
  total_losses DECIMAL(30, 18) NOT NULL DEFAULT 0,
  net_gains DECIMAL(30, 18) NOT NULL DEFAULT 0,
  harvested_losses DECIMAL(30, 18) NOT NULL DEFAULT 0,
  wash_sale_adjustments DECIMAL(30, 18) NOT NULL DEFAULT 0,
  total_income DECIMAL(30, 18) NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- One summary per tax year
  CONSTRAINT unique_tax_year UNIQUE (tax_year)
);

-- Income tracking table
CREATE TABLE income_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tax_summary_id UUID NOT NULL REFERENCES tax_summaries(id) ON DELETE CASCADE,
  type income_type NOT NULL,
  amount DECIMAL(30, 18) NOT NULL CHECK (amount > 0),
  asset VARCHAR(50) NOT NULL,
  date TIMESTAMPTZ NOT NULL,
  source VARCHAR(100) NOT NULL,
  transaction_id UUID REFERENCES transactions(id),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_tax_summaries_year ON tax_summaries(tax_year DESC);
CREATE INDEX idx_income_records_summary_id ON income_records(tax_summary_id);
CREATE INDEX idx_income_records_type ON income_records(type);
CREATE INDEX idx_income_records_date ON income_records(date);
CREATE INDEX idx_income_records_asset ON income_records(asset);

-- Add updated_at trigger
CREATE TRIGGER update_tax_summaries_updated_at
BEFORE UPDATE ON tax_summaries
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Function to compute tax summary for a year
CREATE OR REPLACE FUNCTION compute_tax_summary(
  year INTEGER
) RETURNS UUID AS $$
DECLARE
  summary_id UUID;
  short_gains DECIMAL(30, 18);
  long_gains DECIMAL(30, 18);
  total_losses DECIMAL(30, 18);
  wash_adjustments DECIMAL(30, 18);
  total_income DECIMAL(30, 18);
BEGIN
  -- Calculate short-term gains
  SELECT COALESCE(SUM(gain), 0) INTO short_gains
  FROM disposals
  WHERE tax_year = year AND term = 'SHORT' AND gain > 0;

  -- Calculate long-term gains
  SELECT COALESCE(SUM(gain), 0) INTO long_gains
  FROM disposals
  WHERE tax_year = year AND term = 'LONG' AND gain > 0;

  -- Calculate total losses
  SELECT COALESCE(SUM(ABS(gain)), 0) INTO total_losses
  FROM disposals
  WHERE tax_year = year AND gain < 0;

  -- Calculate wash sale adjustments
  SELECT COALESCE(SUM(wash_sale_disallowed), 0) INTO wash_adjustments
  FROM disposals
  WHERE tax_year = year;

  -- Calculate total income
  SELECT COALESCE(SUM(amount), 0) INTO total_income
  FROM income_records ir
  JOIN tax_summaries ts ON ir.tax_summary_id = ts.id
  WHERE ts.tax_year = year;

  -- Insert or update summary
  INSERT INTO tax_summaries (
    tax_year,
    short_term_gains,
    long_term_gains,
    total_gains,
    total_losses,
    net_gains,
    wash_sale_adjustments,
    total_income
  ) VALUES (
    year,
    short_gains,
    long_gains,
    short_gains + long_gains,
    total_losses,
    (short_gains + long_gains) - total_losses,
    wash_adjustments,
    total_income
  )
  ON CONFLICT (tax_year) DO UPDATE SET
    short_term_gains = EXCLUDED.short_term_gains,
    long_term_gains = EXCLUDED.long_term_gains,
    total_gains = EXCLUDED.total_gains,
    total_losses = EXCLUDED.total_losses,
    net_gains = EXCLUDED.net_gains,
    wash_sale_adjustments = EXCLUDED.wash_sale_adjustments,
    total_income = EXCLUDED.total_income,
    updated_at = NOW()
  RETURNING id INTO summary_id;

  RETURN summary_id;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON TABLE tax_summaries IS 'Annual tax summaries with aggregated capital gains and income';
COMMENT ON TABLE income_records IS 'Individual income records (interest, dividends, staking, etc.)';
