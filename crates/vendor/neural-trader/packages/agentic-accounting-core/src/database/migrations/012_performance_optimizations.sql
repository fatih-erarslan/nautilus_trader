-- Migration: Advanced Performance Optimizations
-- Description: Additional indexes, query optimization, and performance tuning

-- ========================================
-- COVERING INDEXES (Include frequently accessed columns)
-- ========================================

-- Tax lot selection queries (FIFO/LIFO/HIFO with all needed data)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tax_lots_fifo_covering
ON tax_lots(asset, acquired_date ASC, status)
INCLUDE (id, quantity, original_quantity, cost_basis, unit_cost_basis)
WHERE status IN ('OPEN', 'PARTIAL');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tax_lots_lifo_covering
ON tax_lots(asset, acquired_date DESC, status)
INCLUDE (id, quantity, original_quantity, cost_basis, unit_cost_basis)
WHERE status IN ('OPEN', 'PARTIAL');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tax_lots_hifo_covering
ON tax_lots(asset, unit_cost_basis DESC, status)
INCLUDE (id, acquired_date, quantity, original_quantity, cost_basis)
WHERE status IN ('OPEN', 'PARTIAL');

-- Transaction lookup queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_asset_time_covering
ON transactions(asset, timestamp DESC)
INCLUDE (id, type, quantity, price, fees, source)
WHERE taxable = true;

-- Disposal reporting queries (avoid table scans)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_disposals_reporting_covering
ON disposals(tax_year, term, disposal_date DESC)
INCLUDE (asset, quantity, proceeds, cost_basis, gain);

-- ========================================
-- HASH INDEXES (for exact match queries)
-- ========================================

-- Fast transaction ID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_id_hash
ON transactions USING hash(id);

-- Fast tax lot ID lookups
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tax_lots_id_hash
ON tax_lots USING hash(id);

-- ========================================
-- EXPRESSION INDEXES (for computed values)
-- ========================================

-- Index on year extracted from timestamp for fast year filtering
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_year
ON transactions(EXTRACT(YEAR FROM timestamp));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_disposals_year
ON disposals(EXTRACT(YEAR FROM disposal_date));

-- Index on asset + wallet combination for position queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tax_lots_asset_wallet
ON tax_lots((metadata->>'wallet'), asset)
WHERE status IN ('OPEN', 'PARTIAL');

-- ========================================
-- BLOOM FILTERS (for multi-column OR queries)
-- ========================================

-- Install bloom extension if available
CREATE EXTENSION IF NOT EXISTS bloom;

-- Bloom index for transaction filtering
CREATE INDEX IF NOT EXISTS idx_transactions_bloom
ON transactions USING bloom(asset, type, source, taxable)
WITH (length=80, col1=2, col2=2, col3=2, col4=2);

-- ========================================
-- PARTIAL INDEXES (for frequently filtered queries)
-- ========================================

-- Only index recent transactions (last 2 years)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_recent
ON transactions(timestamp DESC, asset, type)
WHERE timestamp > NOW() - INTERVAL '2 years';

-- Only index transactions with fees
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_transactions_with_fees
ON transactions(asset, timestamp DESC, fees)
WHERE fees > 0;

-- Only index short-term disposals (for tax planning)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_disposals_short_term
ON disposals(asset, disposal_date DESC, gain)
WHERE term = 'SHORT';

-- ========================================
-- STATISTICS TARGETS (improve query planning)
-- ========================================

-- Increase statistics targets for high-cardinality columns
ALTER TABLE transactions ALTER COLUMN asset SET STATISTICS 1000;
ALTER TABLE transactions ALTER COLUMN timestamp SET STATISTICS 1000;
ALTER TABLE tax_lots ALTER COLUMN asset SET STATISTICS 1000;
ALTER TABLE tax_lots ALTER COLUMN acquired_date SET STATISTICS 1000;
ALTER TABLE disposals ALTER COLUMN asset SET STATISTICS 500;

-- ========================================
-- QUERY OPTIMIZATION FUNCTIONS
-- ========================================

-- Function to efficiently get available lots for disposal
CREATE OR REPLACE FUNCTION get_available_lots_for_disposal(
  p_asset VARCHAR(50),
  p_method VARCHAR(20),
  p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
  id UUID,
  quantity DECIMAL(30, 10),
  cost_basis DECIMAL(30, 10),
  unit_cost_basis DECIMAL(30, 10),
  acquired_date TIMESTAMPTZ
) AS $$
BEGIN
  CASE p_method
    WHEN 'FIFO' THEN
      RETURN QUERY
      SELECT tl.id, tl.quantity, tl.cost_basis, tl.unit_cost_basis, tl.acquired_date
      FROM tax_lots tl
      WHERE tl.asset = p_asset AND tl.status IN ('OPEN', 'PARTIAL')
      ORDER BY tl.acquired_date ASC
      LIMIT p_limit;

    WHEN 'LIFO' THEN
      RETURN QUERY
      SELECT tl.id, tl.quantity, tl.cost_basis, tl.unit_cost_basis, tl.acquired_date
      FROM tax_lots tl
      WHERE tl.asset = p_asset AND tl.status IN ('OPEN', 'PARTIAL')
      ORDER BY tl.acquired_date DESC
      LIMIT p_limit;

    WHEN 'HIFO' THEN
      RETURN QUERY
      SELECT tl.id, tl.quantity, tl.cost_basis, tl.unit_cost_basis, tl.acquired_date
      FROM tax_lots tl
      WHERE tl.asset = p_asset AND tl.status IN ('OPEN', 'PARTIAL')
      ORDER BY tl.unit_cost_basis DESC
      LIMIT p_limit;

    ELSE
      RAISE EXCEPTION 'Unsupported method: %', p_method;
  END CASE;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get tax summary for year (cached)
CREATE OR REPLACE FUNCTION get_tax_summary_for_year(p_tax_year INTEGER)
RETURNS TABLE (
  short_term_gains DECIMAL(30, 10),
  long_term_gains DECIMAL(30, 10),
  total_gains DECIMAL(30, 10),
  total_losses DECIMAL(30, 10),
  net_gains DECIMAL(30, 10)
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COALESCE(SUM(CASE WHEN term = 'SHORT' AND gain > 0 THEN gain ELSE 0 END), 0),
    COALESCE(SUM(CASE WHEN term = 'LONG' AND gain > 0 THEN gain ELSE 0 END), 0),
    COALESCE(SUM(CASE WHEN gain > 0 THEN gain ELSE 0 END), 0),
    COALESCE(SUM(CASE WHEN gain < 0 THEN ABS(gain) ELSE 0 END), 0),
    COALESCE(SUM(gain), 0)
  FROM disposals
  WHERE tax_year = p_tax_year;
END;
$$ LANGUAGE plpgsql STABLE;

-- ========================================
-- MATERIALIZED VIEW OPTIMIZATIONS
-- ========================================

-- Drop old positions view
DROP MATERIALIZED VIEW IF EXISTS positions CASCADE;

-- Create optimized positions view with better aggregation
CREATE MATERIALIZED VIEW positions AS
SELECT
  tl.asset,
  SUM(tl.quantity) as total_quantity,
  SUM(tl.cost_basis) as total_cost_basis,
  SUM(tl.cost_basis) / NULLIF(SUM(tl.quantity), 0) as average_cost_basis,
  MIN(tl.acquired_date) as first_acquired,
  MAX(tl.acquired_date) as last_acquired,
  COUNT(*) as lot_count,
  NOW() as last_updated
FROM tax_lots tl
WHERE tl.status IN ('OPEN', 'PARTIAL') AND tl.quantity > 0
GROUP BY tl.asset;

CREATE UNIQUE INDEX idx_positions_asset ON positions(asset);
CREATE INDEX idx_positions_quantity ON positions(total_quantity DESC);

-- Materialized view for hot assets (frequently traded)
CREATE MATERIALIZED VIEW IF NOT EXISTS hot_assets_30d AS
SELECT
  t.asset,
  COUNT(*) AS transaction_count,
  SUM(t.quantity * t.price) AS total_volume,
  AVG(t.price) AS avg_price,
  MAX(t.timestamp) AS last_transaction
FROM transactions t
WHERE t.timestamp > NOW() - INTERVAL '30 days'
GROUP BY t.asset
HAVING COUNT(*) >= 5
ORDER BY total_volume DESC
LIMIT 100;

CREATE UNIQUE INDEX idx_hot_assets_30d_asset ON hot_assets_30d(asset);

-- ========================================
-- VACUUM AND ANALYZE CONFIGURATION
-- ========================================

-- Tune autovacuum for high-volume tables
ALTER TABLE transactions SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE tax_lots SET (
  autovacuum_vacuum_scale_factor = 0.05,
  autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE disposals SET (
  autovacuum_vacuum_scale_factor = 0.1,
  autovacuum_analyze_scale_factor = 0.05
);

-- ========================================
-- QUERY PLAN HINTS (for common queries)
-- ========================================

-- Analyze all tables to update statistics
ANALYZE transactions;
ANALYZE tax_lots;
ANALYZE disposals;
ANALYZE audit_trail;
ANALYZE embeddings;
ANALYZE reasoning_bank;

-- ========================================
-- COMMENTS
-- ========================================

COMMENT ON INDEX idx_tax_lots_fifo_covering IS 'Covering index for FIFO lot selection - includes all needed columns';
COMMENT ON INDEX idx_tax_lots_lifo_covering IS 'Covering index for LIFO lot selection - includes all needed columns';
COMMENT ON INDEX idx_tax_lots_hifo_covering IS 'Covering index for HIFO lot selection - includes all needed columns';
COMMENT ON FUNCTION get_available_lots_for_disposal IS 'Optimized function for fetching available lots using proper indexes';
COMMENT ON FUNCTION get_tax_summary_for_year IS 'Cached tax summary calculation for a given year';
