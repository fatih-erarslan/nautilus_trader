-- Agentic Accounting Initial Schema Migration
-- Version: 0.1.0
-- Date: 2025-11-16

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- =====================================================
-- 1. TRANSACTIONS TABLE
-- =====================================================
CREATE TABLE transactions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  timestamp TIMESTAMPTZ NOT NULL,
  type VARCHAR(20) NOT NULL CHECK (type IN ('BUY', 'SELL', 'TRADE', 'INCOME', 'EXPENSE', 'TRANSFER')),
  asset VARCHAR(50) NOT NULL,
  quantity DECIMAL(30, 10) NOT NULL CHECK (quantity >= 0),
  price DECIMAL(30, 10) NOT NULL CHECK (price >= 0),
  fees DECIMAL(30, 10) NOT NULL DEFAULT 0 CHECK (fees >= 0),
  currency VARCHAR(10) NOT NULL DEFAULT 'USD',
  source VARCHAR(100) NOT NULL,
  source_id VARCHAR(255),
  taxable BOOLEAN NOT NULL DEFAULT true,
  metadata JSONB DEFAULT '{}',
  embedding vector(768),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE(source, source_id)
);

CREATE INDEX idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX idx_transactions_asset ON transactions(asset);
CREATE INDEX idx_transactions_type ON transactions(type);
CREATE INDEX idx_transactions_source ON transactions(source);
CREATE INDEX idx_transactions_embedding ON transactions USING hnsw (embedding vector_cosine_ops);

-- =====================================================
-- 2. TAX LOTS TABLE
-- =====================================================
CREATE TABLE tax_lots (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  transaction_id UUID NOT NULL REFERENCES transactions(id),
  asset VARCHAR(50) NOT NULL,
  acquired_date TIMESTAMPTZ NOT NULL,
  quantity DECIMAL(30, 10) NOT NULL CHECK (quantity >= 0),
  original_quantity DECIMAL(30, 10) NOT NULL CHECK (original_quantity >= 0),
  cost_basis DECIMAL(30, 10) NOT NULL,
  unit_cost_basis DECIMAL(30, 10) NOT NULL,
  currency VARCHAR(10) NOT NULL DEFAULT 'USD',
  source VARCHAR(100) NOT NULL,
  method VARCHAR(20) NOT NULL CHECK (method IN ('FIFO', 'LIFO', 'HIFO', 'SPECIFIC_ID', 'AVERAGE_COST')),
  status VARCHAR(20) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'PARTIAL', 'CLOSED')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tax_lots_asset ON tax_lots(asset);
CREATE INDEX idx_tax_lots_acquired_date ON tax_lots(acquired_date);
CREATE INDEX idx_tax_lots_status ON tax_lots(status) WHERE status != 'CLOSED';
CREATE INDEX idx_tax_lots_transaction ON tax_lots(transaction_id);

-- =====================================================
-- 3. DISPOSALS TABLE
-- =====================================================
CREATE TABLE disposals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  lot_id UUID NOT NULL REFERENCES tax_lots(id),
  transaction_id UUID NOT NULL REFERENCES transactions(id),
  disposal_date TIMESTAMPTZ NOT NULL,
  quantity DECIMAL(30, 10) NOT NULL CHECK (quantity > 0),
  proceeds DECIMAL(30, 10) NOT NULL,
  cost_basis DECIMAL(30, 10) NOT NULL,
  gain DECIMAL(30, 10) NOT NULL,
  term VARCHAR(10) NOT NULL CHECK (term IN ('SHORT', 'LONG')),
  tax_year INTEGER NOT NULL,
  method VARCHAR(20) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_disposals_lot ON disposals(lot_id);
CREATE INDEX idx_disposals_transaction ON disposals(transaction_id);
CREATE INDEX idx_disposals_tax_year ON disposals(tax_year);
CREATE INDEX idx_disposals_term ON disposals(term);
CREATE INDEX idx_disposals_date ON disposals(disposal_date);

-- =====================================================
-- 4. POSITIONS (MATERIALIZED VIEW)
-- =====================================================
CREATE MATERIALIZED VIEW positions AS
SELECT
  asset,
  SUM(quantity) as total_quantity,
  SUM(cost_basis) as total_cost_basis,
  SUM(cost_basis) / NULLIF(SUM(quantity), 0) as average_cost_basis,
  0::DECIMAL(30, 10) as current_price,
  0::DECIMAL(30, 10) as market_value,
  0::DECIMAL(30, 10) as unrealized_gain,
  0::NUMERIC as unrealized_gain_percent,
  NOW() as last_updated
FROM tax_lots
WHERE status IN ('OPEN', 'PARTIAL')
GROUP BY asset
HAVING SUM(quantity) > 0;

CREATE UNIQUE INDEX idx_positions_asset ON positions(asset);

-- =====================================================
-- 5. TAX SUMMARIES TABLE
-- =====================================================
CREATE TABLE tax_summaries (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tax_year INTEGER NOT NULL,
  user_id UUID,
  short_term_gains DECIMAL(30, 10) NOT NULL DEFAULT 0,
  long_term_gains DECIMAL(30, 10) NOT NULL DEFAULT 0,
  total_gains DECIMAL(30, 10) NOT NULL DEFAULT 0,
  total_losses DECIMAL(30, 10) NOT NULL DEFAULT 0,
  net_gains DECIMAL(30, 10) NOT NULL DEFAULT 0,
  harvested_losses DECIMAL(30, 10) NOT NULL DEFAULT 0,
  wash_sale_adjustments DECIMAL(30, 10) NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE(tax_year, user_id)
);

CREATE INDEX idx_tax_summaries_year ON tax_summaries(tax_year);
CREATE INDEX idx_tax_summaries_user ON tax_summaries(user_id);

-- =====================================================
-- 6. COMPLIANCE RULES TABLE
-- =====================================================
CREATE TABLE compliance_rules (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name VARCHAR(255) NOT NULL,
  type VARCHAR(50) NOT NULL CHECK (type IN ('WASH_SALE', 'TRADING_LIMIT', 'SEGREGATION_DUTY', 'SUSPICIOUS_ACTIVITY', 'POLICY_VIOLATION')),
  condition JSONB NOT NULL,
  action VARCHAR(30) NOT NULL CHECK (action IN ('ALERT', 'BLOCK', 'FLAG', 'REQUIRE_APPROVAL')),
  severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
  enabled BOOLEAN NOT NULL DEFAULT true,
  jurisdictions TEXT[] DEFAULT ARRAY['US'],
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_compliance_rules_type ON compliance_rules(type);
CREATE INDEX idx_compliance_rules_enabled ON compliance_rules(enabled) WHERE enabled = true;

-- =====================================================
-- 7. AUDIT TRAIL TABLE
-- =====================================================
CREATE TABLE audit_trail (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  agent_id VARCHAR(100) NOT NULL,
  agent_type VARCHAR(50) NOT NULL,
  action VARCHAR(100) NOT NULL,
  entity_type VARCHAR(50) NOT NULL,
  entity_id UUID NOT NULL,
  changes JSONB NOT NULL DEFAULT '[]',
  reason TEXT,
  hash VARCHAR(64) NOT NULL,
  signature VARCHAR(128) NOT NULL,
  previous_hash VARCHAR(64),
  merkle_root VARCHAR(64),
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX idx_audit_trail_agent ON audit_trail(agent_id);
CREATE INDEX idx_audit_trail_entity ON audit_trail(entity_type, entity_id);
CREATE INDEX idx_audit_trail_hash ON audit_trail(hash);

-- =====================================================
-- 8. EMBEDDINGS TABLE
-- =====================================================
CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  entity_type VARCHAR(50) NOT NULL,
  entity_id UUID NOT NULL,
  vector vector(768) NOT NULL,
  model VARCHAR(100) NOT NULL,
  metadata JSONB DEFAULT '{}',
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE(entity_type, entity_id)
);

CREATE INDEX idx_embeddings_entity ON embeddings(entity_type, entity_id);
CREATE INDEX idx_embeddings_vector ON embeddings USING hnsw (vector vector_cosine_ops);

-- =====================================================
-- 9. REASONING BANK TABLE
-- =====================================================
CREATE TABLE reasoning_bank (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  agent_type VARCHAR(50) NOT NULL,
  scenario TEXT NOT NULL,
  decision TEXT NOT NULL,
  rationale TEXT NOT NULL,
  outcome VARCHAR(20) NOT NULL CHECK (outcome IN ('SUCCESS', 'FAILURE', 'PARTIAL', 'PENDING')),
  metrics JSONB DEFAULT '{}',
  embedding vector(768),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  feedback_score DECIMAL(3, 2),
  references UUID[]
);

CREATE INDEX idx_reasoning_bank_agent ON reasoning_bank(agent_type);
CREATE INDEX idx_reasoning_bank_outcome ON reasoning_bank(outcome);
CREATE INDEX idx_reasoning_bank_timestamp ON reasoning_bank(timestamp DESC);
CREATE INDEX idx_reasoning_bank_embedding ON reasoning_bank USING hnsw (embedding vector_cosine_ops);

-- =====================================================
-- 10. VERIFICATION PROOFS TABLE
-- =====================================================
CREATE TABLE verification_proofs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  theorem TEXT NOT NULL,
  proof TEXT NOT NULL,
  invariant VARCHAR(50) NOT NULL CHECK (invariant IN ('BALANCE_CONSISTENCY', 'NON_NEGATIVE_HOLDINGS', 'SEGREGATION_DUTIES', 'COST_BASIS_ACCURACY', 'WASH_SALE_COMPLIANCE')),
  verified BOOLEAN NOT NULL DEFAULT false,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  context JSONB DEFAULT '{}',
  error TEXT
);

CREATE INDEX idx_verification_proofs_invariant ON verification_proofs(invariant);
CREATE INDEX idx_verification_proofs_verified ON verification_proofs(verified);
CREATE INDEX idx_verification_proofs_timestamp ON verification_proofs(timestamp DESC);

-- =====================================================
-- TRIGGERS FOR UPDATED_AT
-- =====================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_transactions_updated_at BEFORE UPDATE ON transactions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tax_lots_updated_at BEFORE UPDATE ON tax_lots
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tax_summaries_updated_at BEFORE UPDATE ON tax_summaries
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_compliance_rules_updated_at BEFORE UPDATE ON compliance_rules
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- REFRESH POSITIONS MATERIALIZED VIEW FUNCTION
-- =====================================================
CREATE OR REPLACE FUNCTION refresh_positions()
RETURNS VOID AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY positions;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- COMMENTS
-- =====================================================
COMMENT ON TABLE transactions IS 'All financial transactions from various sources';
COMMENT ON TABLE tax_lots IS 'Individual tax lot tracking with FIFO/LIFO/HIFO support';
COMMENT ON TABLE disposals IS 'Sale records with realized gains/losses';
COMMENT ON TABLE positions IS 'Current holdings aggregated from tax lots';
COMMENT ON TABLE tax_summaries IS 'Annual tax summary data';
COMMENT ON TABLE compliance_rules IS 'Configurable compliance rule definitions';
COMMENT ON TABLE audit_trail IS 'Immutable audit log with cryptographic verification';
COMMENT ON TABLE embeddings IS 'Vector embeddings for semantic search';
COMMENT ON TABLE reasoning_bank IS 'Agent decision history for learning';
COMMENT ON TABLE verification_proofs IS 'Formal verification proofs from Lean4';
