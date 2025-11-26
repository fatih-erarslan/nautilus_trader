-- Migration: Create audit_trail table
-- Description: Immutable audit log with cryptographic verification

CREATE TABLE audit_trail (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  agent_id VARCHAR(255) NOT NULL,
  agent_type VARCHAR(100) NOT NULL,
  action VARCHAR(255) NOT NULL,
  entity_type VARCHAR(100) NOT NULL,
  entity_id UUID NOT NULL,
  changes JSONB DEFAULT '[]',
  reason TEXT,
  hash VARCHAR(64) NOT NULL,
  signature VARCHAR(128) NOT NULL,
  previous_hash VARCHAR(64),
  merkle_root VARCHAR(64),
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Immutable audit trail (prevent updates and deletes)
CREATE RULE audit_trail_no_update AS ON UPDATE TO audit_trail DO INSTEAD NOTHING;
CREATE RULE audit_trail_no_delete AS ON DELETE TO audit_trail DO INSTEAD NOTHING;

-- Indexes
CREATE INDEX idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
CREATE INDEX idx_audit_trail_agent_id ON audit_trail(agent_id);
CREATE INDEX idx_audit_trail_agent_type ON audit_trail(agent_type);
CREATE INDEX idx_audit_trail_entity_type ON audit_trail(entity_type);
CREATE INDEX idx_audit_trail_entity_id ON audit_trail(entity_id);
CREATE INDEX idx_audit_trail_action ON audit_trail(action);

-- Composite index for entity queries
CREATE INDEX idx_audit_trail_entity ON audit_trail(entity_type, entity_id, timestamp DESC);

-- Index for chain verification
CREATE INDEX idx_audit_trail_hash ON audit_trail(hash);
CREATE INDEX idx_audit_trail_previous_hash ON audit_trail(previous_hash);

-- Function to compute SHA-256 hash
CREATE OR REPLACE FUNCTION compute_audit_hash(
  agent_id VARCHAR,
  action VARCHAR,
  entity_type VARCHAR,
  entity_id UUID,
  changes JSONB,
  previous_hash VARCHAR
) RETURNS VARCHAR AS $$
DECLARE
  data TEXT;
BEGIN
  data := agent_id || '|' || action || '|' || entity_type || '|' ||
          entity_id::TEXT || '|' || changes::TEXT || '|' ||
          COALESCE(previous_hash, '');

  -- PostgreSQL's built-in SHA-256 (requires pgcrypto extension)
  RETURN encode(digest(data, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get last audit entry hash
CREATE OR REPLACE FUNCTION get_last_audit_hash() RETURNS VARCHAR AS $$
DECLARE
  last_hash VARCHAR;
BEGIN
  SELECT hash INTO last_hash
  FROM audit_trail
  ORDER BY timestamp DESC
  LIMIT 1;

  RETURN COALESCE(last_hash, '');
END;
$$ LANGUAGE plpgsql;

-- Function to insert audit entry with automatic hashing
CREATE OR REPLACE FUNCTION insert_audit_entry(
  p_agent_id VARCHAR,
  p_agent_type VARCHAR,
  p_action VARCHAR,
  p_entity_type VARCHAR,
  p_entity_id UUID,
  p_changes JSONB,
  p_reason TEXT,
  p_signature VARCHAR,
  p_metadata JSONB DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
  entry_id UUID;
  prev_hash VARCHAR;
  entry_hash VARCHAR;
BEGIN
  -- Get previous hash
  prev_hash := get_last_audit_hash();

  -- Compute hash for this entry
  entry_hash := compute_audit_hash(
    p_agent_id,
    p_action,
    p_entity_type,
    p_entity_id,
    p_changes,
    prev_hash
  );

  -- Insert entry
  INSERT INTO audit_trail (
    agent_id,
    agent_type,
    action,
    entity_type,
    entity_id,
    changes,
    reason,
    hash,
    signature,
    previous_hash,
    metadata
  ) VALUES (
    p_agent_id,
    p_agent_type,
    p_action,
    p_entity_type,
    p_entity_id,
    p_changes,
    p_reason,
    entry_hash,
    p_signature,
    prev_hash,
    p_metadata
  ) RETURNING id INTO entry_id;

  RETURN entry_id;
END;
$$ LANGUAGE plpgsql;

-- Function to verify audit chain integrity
CREATE OR REPLACE FUNCTION verify_audit_chain() RETURNS TABLE(
  entry_id UUID,
  valid BOOLEAN,
  error TEXT
) AS $$
BEGIN
  RETURN QUERY
  WITH chain AS (
    SELECT
      id,
      agent_id,
      action,
      entity_type,
      entity_id,
      changes,
      hash,
      previous_hash,
      LAG(hash) OVER (ORDER BY timestamp) AS expected_previous_hash
    FROM audit_trail
    ORDER BY timestamp
  )
  SELECT
    id,
    CASE
      WHEN previous_hash IS NULL THEN true  -- First entry
      WHEN previous_hash = expected_previous_hash THEN true
      ELSE false
    END AS valid,
    CASE
      WHEN previous_hash IS NULL THEN NULL
      WHEN previous_hash = expected_previous_hash THEN NULL
      ELSE 'Hash chain broken'
    END AS error
  FROM chain;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON TABLE audit_trail IS 'Immutable audit log with cryptographic chain verification';
