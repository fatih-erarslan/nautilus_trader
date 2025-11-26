-- Migration: Create embeddings table with pgvector
-- Description: Vector embeddings for semantic search (requires pgvector extension)

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type VARCHAR(100) NOT NULL,
  entity_id UUID NOT NULL,
  vector vector(768) NOT NULL,  -- 768 dimensions (BERT-base, all-MiniLM-L6-v2)
  model VARCHAR(100) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
  metadata JSONB DEFAULT '{}',
  timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  -- One embedding per entity
  CONSTRAINT unique_entity_embedding UNIQUE (entity_type, entity_id)
);

-- Indexes for entity lookups
CREATE INDEX idx_embeddings_entity_type ON embeddings(entity_type);
CREATE INDEX idx_embeddings_entity_id ON embeddings(entity_id);
CREATE INDEX idx_embeddings_timestamp ON embeddings(timestamp DESC);

-- HNSW index for fast vector similarity search (cosine distance)
CREATE INDEX idx_embeddings_vector_cosine ON embeddings
USING hnsw (vector vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat index (faster build, slower query)
-- CREATE INDEX idx_embeddings_vector_ivfflat ON embeddings
-- USING ivfflat (vector vector_cosine_ops)
-- WITH (lists = 100);

-- Function for similarity search
CREATE OR REPLACE FUNCTION find_similar_entities(
  query_vector vector(768),
  entity_filter VARCHAR DEFAULT NULL,
  similarity_threshold FLOAT DEFAULT 0.8,
  result_limit INTEGER DEFAULT 10
) RETURNS TABLE(
  entity_type VARCHAR,
  entity_id UUID,
  similarity FLOAT,
  metadata JSONB
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.entity_type,
    e.entity_id,
    1 - (e.vector <=> query_vector) AS similarity,
    e.metadata
  FROM embeddings e
  WHERE
    (entity_filter IS NULL OR e.entity_type = entity_filter)
    AND (1 - (e.vector <=> query_vector)) >= similarity_threshold
  ORDER BY e.vector <=> query_vector
  LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get embedding for transaction
CREATE OR REPLACE FUNCTION get_transaction_embedding(
  transaction_id UUID
) RETURNS vector(768) AS $$
DECLARE
  embedding_vec vector(768);
BEGIN
  SELECT vector INTO embedding_vec
  FROM embeddings
  WHERE entity_type = 'transaction' AND entity_id = transaction_id;

  RETURN embedding_vec;
END;
$$ LANGUAGE plpgsql;

-- Function to find similar transactions
CREATE OR REPLACE FUNCTION find_similar_transactions(
  transaction_id UUID,
  similarity_threshold FLOAT DEFAULT 0.8,
  result_limit INTEGER DEFAULT 10
) RETURNS TABLE(
  similar_transaction_id UUID,
  similarity FLOAT,
  asset VARCHAR,
  type transaction_type,
  amount DECIMAL,
  timestamp TIMESTAMPTZ
) AS $$
DECLARE
  query_vec vector(768);
BEGIN
  -- Get embedding for the query transaction
  query_vec := get_transaction_embedding(transaction_id);

  IF query_vec IS NULL THEN
    RAISE EXCEPTION 'No embedding found for transaction %', transaction_id;
  END IF;

  RETURN QUERY
  SELECT
    t.id,
    1 - (e.vector <=> query_vec) AS similarity,
    t.asset,
    t.type,
    t.quantity * t.price AS amount,
    t.timestamp
  FROM embeddings e
  JOIN transactions t ON e.entity_id = t.id
  WHERE
    e.entity_type = 'transaction'
    AND e.entity_id != transaction_id
    AND (1 - (e.vector <=> query_vec)) >= similarity_threshold
  ORDER BY e.vector <=> query_vec
  LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON TABLE embeddings IS 'Vector embeddings for semantic similarity search using pgvector';
COMMENT ON INDEX idx_embeddings_vector_cosine IS 'HNSW index for fast cosine similarity search (768-dim vectors)';
