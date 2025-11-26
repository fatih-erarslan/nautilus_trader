-- Migration: Create reasoning_bank table
-- Description: Agent learning memory with decision tracking

CREATE TYPE outcome AS ENUM (
  'SUCCESS',
  'FAILURE',
  'PARTIAL',
  'PENDING'
);

CREATE TABLE reasoning_bank (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_type VARCHAR(100) NOT NULL,
  scenario TEXT NOT NULL,
  decision TEXT NOT NULL,
  rationale TEXT NOT NULL,
  outcome outcome NOT NULL,
  metrics JSONB DEFAULT '{}',
  embedding vector(768),  -- For similarity search
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  feedback_score DECIMAL(3, 2) CHECK (feedback_score >= 0 AND feedback_score <= 1),
  references UUID[] DEFAULT '{}',
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_reasoning_bank_agent_type ON reasoning_bank(agent_type);
CREATE INDEX idx_reasoning_bank_outcome ON reasoning_bank(outcome);
CREATE INDEX idx_reasoning_bank_timestamp ON reasoning_bank(timestamp DESC);
CREATE INDEX idx_reasoning_bank_feedback_score ON reasoning_bank(feedback_score DESC NULLS LAST);

-- HNSW index for similar decision search
CREATE INDEX idx_reasoning_bank_embedding ON reasoning_bank
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE embedding IS NOT NULL;

-- Function to find similar decisions
CREATE OR REPLACE FUNCTION find_similar_decisions(
  query_scenario TEXT,
  query_embedding vector(768),
  agent_filter VARCHAR DEFAULT NULL,
  outcome_filter outcome DEFAULT NULL,
  min_feedback_score DECIMAL DEFAULT 0.7,
  result_limit INTEGER DEFAULT 5
) RETURNS TABLE(
  reasoning_id UUID,
  agent_type VARCHAR,
  scenario TEXT,
  decision TEXT,
  rationale TEXT,
  outcome outcome,
  similarity FLOAT,
  feedback_score DECIMAL
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    rb.id,
    rb.agent_type,
    rb.scenario,
    rb.decision,
    rb.rationale,
    rb.outcome,
    CASE
      WHEN rb.embedding IS NOT NULL AND query_embedding IS NOT NULL
      THEN 1 - (rb.embedding <=> query_embedding)
      ELSE NULL
    END AS similarity,
    rb.feedback_score
  FROM reasoning_bank rb
  WHERE
    (agent_filter IS NULL OR rb.agent_type = agent_filter)
    AND (outcome_filter IS NULL OR rb.outcome = outcome_filter)
    AND (rb.feedback_score IS NULL OR rb.feedback_score >= min_feedback_score)
    AND (
      rb.embedding IS NULL OR query_embedding IS NULL
      OR (1 - (rb.embedding <=> query_embedding)) >= 0.7
    )
  ORDER BY
    CASE
      WHEN rb.embedding IS NOT NULL AND query_embedding IS NOT NULL
      THEN rb.embedding <=> query_embedding
      ELSE 1
    END,
    rb.feedback_score DESC NULLS LAST,
    rb.timestamp DESC
  LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to record agent decision
CREATE OR REPLACE FUNCTION record_agent_decision(
  p_agent_type VARCHAR,
  p_scenario TEXT,
  p_decision TEXT,
  p_rationale TEXT,
  p_outcome outcome,
  p_metrics JSONB DEFAULT '{}',
  p_embedding vector(768) DEFAULT NULL,
  p_references UUID[] DEFAULT '{}'
) RETURNS UUID AS $$
DECLARE
  entry_id UUID;
BEGIN
  INSERT INTO reasoning_bank (
    agent_type,
    scenario,
    decision,
    rationale,
    outcome,
    metrics,
    embedding,
    references
  ) VALUES (
    p_agent_type,
    p_scenario,
    p_decision,
    p_rationale,
    p_outcome,
    p_metrics,
    p_embedding,
    p_references
  ) RETURNING id INTO entry_id;

  RETURN entry_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update feedback score
CREATE OR REPLACE FUNCTION update_reasoning_feedback(
  reasoning_id UUID,
  score DECIMAL
) RETURNS BOOLEAN AS $$
BEGIN
  UPDATE reasoning_bank
  SET feedback_score = score
  WHERE id = reasoning_id;

  RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- View for successful patterns
CREATE VIEW successful_reasoning_patterns AS
SELECT
  agent_type,
  scenario,
  decision,
  rationale,
  outcome,
  feedback_score,
  COUNT(*) OVER (PARTITION BY agent_type, decision) AS usage_count
FROM reasoning_bank
WHERE outcome = 'SUCCESS' AND (feedback_score IS NULL OR feedback_score >= 0.8)
ORDER BY feedback_score DESC NULLS LAST, timestamp DESC;

-- Add comment
COMMENT ON TABLE reasoning_bank IS 'Agent learning memory with decision tracking and similarity search';
COMMENT ON VIEW successful_reasoning_patterns IS 'High-performing decision patterns for agent learning';
