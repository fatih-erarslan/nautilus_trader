-- Migration: Create verification_proofs table
-- Description: Formal verification proofs using Lean4

CREATE TYPE invariant AS ENUM (
  'BALANCE_CONSISTENCY',
  'NON_NEGATIVE_HOLDINGS',
  'SEGREGATION_DUTIES',
  'COST_BASIS_ACCURACY',
  'WASH_SALE_COMPLIANCE'
);

CREATE TABLE verification_proofs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  theorem TEXT NOT NULL,
  proof TEXT NOT NULL,
  invariant invariant NOT NULL,
  verified BOOLEAN NOT NULL DEFAULT false,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  context JSONB DEFAULT '{}',
  error TEXT,
  verification_time_ms INTEGER,
  lean_version VARCHAR(50) DEFAULT '4.0.0',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_verification_proofs_invariant ON verification_proofs(invariant);
CREATE INDEX idx_verification_proofs_verified ON verification_proofs(verified);
CREATE INDEX idx_verification_proofs_timestamp ON verification_proofs(timestamp DESC);

-- Index for failed proofs
CREATE INDEX idx_verification_proofs_failed ON verification_proofs(verified) WHERE verified = false;

-- Function to record verification attempt
CREATE OR REPLACE FUNCTION record_verification(
  p_theorem TEXT,
  p_proof TEXT,
  p_invariant invariant,
  p_verified BOOLEAN,
  p_context JSONB DEFAULT '{}',
  p_error TEXT DEFAULT NULL,
  p_verification_time_ms INTEGER DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
  proof_id UUID;
BEGIN
  INSERT INTO verification_proofs (
    theorem,
    proof,
    invariant,
    verified,
    context,
    error,
    verification_time_ms
  ) VALUES (
    p_theorem,
    p_proof,
    p_invariant,
    p_verified,
    p_context,
    p_error,
    p_verification_time_ms
  ) RETURNING id INTO proof_id;

  RETURN proof_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get verification statistics
CREATE OR REPLACE FUNCTION get_verification_stats() RETURNS TABLE(
  invariant_type invariant,
  total_attempts INTEGER,
  successful_verifications INTEGER,
  failed_verifications INTEGER,
  success_rate DECIMAL,
  avg_verification_time_ms DECIMAL
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    vp.invariant,
    COUNT(*)::INTEGER AS total_attempts,
    COUNT(*) FILTER (WHERE vp.verified = true)::INTEGER AS successful_verifications,
    COUNT(*) FILTER (WHERE vp.verified = false)::INTEGER AS failed_verifications,
    CASE
      WHEN COUNT(*) > 0
      THEN ROUND(COUNT(*) FILTER (WHERE vp.verified = true)::DECIMAL / COUNT(*) * 100, 2)
      ELSE 0
    END AS success_rate,
    ROUND(AVG(vp.verification_time_ms)::DECIMAL, 2) AS avg_verification_time_ms
  FROM verification_proofs vp
  GROUP BY vp.invariant
  ORDER BY vp.invariant;
END;
$$ LANGUAGE plpgsql;

-- View for recent verification failures
CREATE VIEW recent_verification_failures AS
SELECT
  id,
  invariant,
  theorem,
  error,
  timestamp,
  context
FROM verification_proofs
WHERE verified = false
ORDER BY timestamp DESC
LIMIT 100;

-- Add comment
COMMENT ON TABLE verification_proofs IS 'Formal verification proofs for accounting invariants using Lean4';
COMMENT ON VIEW recent_verification_failures IS 'Recent verification failures for debugging';
