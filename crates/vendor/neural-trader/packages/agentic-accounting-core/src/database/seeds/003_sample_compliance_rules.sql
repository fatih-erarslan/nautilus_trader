-- Seed: Sample compliance rules
-- Description: Pre-configured compliance rules for common scenarios

-- Wash sale rule (30-day rule)
INSERT INTO compliance_rules (id, name, description, type, condition, action, severity, enabled, jurisdictions)
VALUES
  ('c1111111-1111-1111-1111-111111111111',
   'Wash Sale Detection',
   'Detects potential wash sales within 30-day window',
   'WASH_SALE',
   '{"operator": "within_days", "field": "timestamp", "value": 30}'::jsonb,
   'FLAG',
   'WARNING',
   true,
   ARRAY['US']);

-- Trading volume limit
INSERT INTO compliance_rules (id, name, description, type, condition, action, severity, enabled, jurisdictions)
VALUES
  ('c2222222-2222-2222-2222-222222222222',
   'High Volume Trading Alert',
   'Alert when single transaction exceeds $100,000',
   'TRADING_LIMIT',
   '{"operator": "greater_than", "field": "amount", "value": 100000}'::jsonb,
   'ALERT',
   'INFO',
   true,
   ARRAY['US', 'EU']);

-- Suspicious activity
INSERT INTO compliance_rules (id, name, description, type, condition, action, severity, enabled, jurisdictions)
VALUES
  ('c3333333-3333-3333-3333-333333333333',
   'Round Amount Transactions',
   'Flag suspiciously round transaction amounts',
   'SUSPICIOUS_ACTIVITY',
   '{"operator": "modulo_zero", "field": "quantity", "value": 1}'::jsonb,
   'FLAG',
   'WARNING',
   true,
   ARRAY['US', 'EU', 'UK']);

-- Segregation of duties
INSERT INTO compliance_rules (id, name, description, type, condition, action, severity, enabled, jurisdictions)
VALUES
  ('c4444444-4444-4444-4444-444444444444',
   'Same-Day Buy-Sell',
   'Require approval for same-day buy and sell',
   'SEGREGATION_DUTY',
   '{"operator": "same_day", "field": "timestamp", "value": true}'::jsonb,
   'REQUIRE_APPROVAL',
   'INFO',
   true,
   ARRAY['US']);

-- Policy violation
INSERT INTO compliance_rules (id, name, description, type, condition, action, severity, enabled, jurisdictions)
VALUES
  ('c5555555-5555-5555-5555-555555555555',
   'Minimum Transaction Value',
   'Block transactions below $10',
   'POLICY_VIOLATION',
   '{"operator": "less_than", "field": "amount", "value": 10}'::jsonb,
   'BLOCK',
   'ERROR',
   true,
   ARRAY['US', 'EU', 'UK']);

-- Add comment
COMMENT ON TABLE compliance_rules IS 'Seeded with common compliance rules for transaction monitoring';
