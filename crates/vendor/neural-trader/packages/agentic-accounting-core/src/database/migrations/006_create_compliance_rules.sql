-- Migration: Create compliance_rules table
-- Description: Configurable compliance rules for transaction validation

CREATE TYPE rule_type AS ENUM (
  'WASH_SALE',
  'TRADING_LIMIT',
  'SEGREGATION_DUTY',
  'SUSPICIOUS_ACTIVITY',
  'POLICY_VIOLATION'
);

CREATE TYPE rule_action AS ENUM (
  'ALERT',
  'BLOCK',
  'FLAG',
  'REQUIRE_APPROVAL'
);

CREATE TYPE severity AS ENUM (
  'INFO',
  'WARNING',
  'ERROR',
  'CRITICAL'
);

CREATE TABLE compliance_rules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  type rule_type NOT NULL,
  condition JSONB NOT NULL,
  action rule_action NOT NULL DEFAULT 'ALERT',
  severity severity NOT NULL DEFAULT 'WARNING',
  enabled BOOLEAN NOT NULL DEFAULT true,
  jurisdictions TEXT[] DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- Unique rule names
  CONSTRAINT unique_rule_name UNIQUE (name)
);

-- Compliance violations table
CREATE TABLE compliance_violations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  rule_id UUID NOT NULL REFERENCES compliance_rules(id) ON DELETE CASCADE,
  transaction_id UUID REFERENCES transactions(id),
  detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  severity severity NOT NULL,
  message TEXT NOT NULL,
  details JSONB DEFAULT '{}',
  resolved BOOLEAN NOT NULL DEFAULT false,
  resolved_at TIMESTAMPTZ,
  resolved_by VARCHAR(255),
  resolution_notes TEXT
);

-- Indexes
CREATE INDEX idx_compliance_rules_type ON compliance_rules(type);
CREATE INDEX idx_compliance_rules_enabled ON compliance_rules(enabled) WHERE enabled = true;
CREATE INDEX idx_compliance_violations_rule_id ON compliance_violations(rule_id);
CREATE INDEX idx_compliance_violations_transaction_id ON compliance_violations(transaction_id);
CREATE INDEX idx_compliance_violations_detected_at ON compliance_violations(detected_at DESC);
CREATE INDEX idx_compliance_violations_resolved ON compliance_violations(resolved) WHERE resolved = false;
CREATE INDEX idx_compliance_violations_severity ON compliance_violations(severity);

-- Add updated_at trigger
CREATE TRIGGER update_compliance_rules_updated_at
BEFORE UPDATE ON compliance_rules
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Function to evaluate compliance rules
CREATE OR REPLACE FUNCTION evaluate_compliance_rules(
  transaction_data JSONB
) RETURNS TABLE(
  rule_id UUID,
  rule_name VARCHAR,
  violated BOOLEAN,
  action rule_action,
  severity severity,
  message TEXT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    cr.id,
    cr.name,
    -- Simple JSON path evaluation (extend with custom logic)
    jsonb_path_exists(transaction_data, cr.condition::jsonpath) AS violated,
    cr.action,
    cr.severity,
    CONCAT('Rule "', cr.name, '" triggered') AS message
  FROM compliance_rules cr
  WHERE cr.enabled = true;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON TABLE compliance_rules IS 'Configurable compliance rules for transaction validation and monitoring';
COMMENT ON TABLE compliance_violations IS 'Records of compliance rule violations with resolution tracking';
