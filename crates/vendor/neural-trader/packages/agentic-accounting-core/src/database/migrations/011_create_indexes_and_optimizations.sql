-- Migration: Additional indexes and performance optimizations
-- Description: Composite indexes, partial indexes, and performance tuning

-- ======================
-- PERFORMANCE INDEXES
-- ======================

-- Transaction analysis queries
CREATE INDEX idx_transactions_asset_type_timestamp
ON transactions(asset, type, timestamp DESC)
WHERE taxable = true;

-- Tax lot matching queries (for disposal calculations)
CREATE INDEX idx_tax_lots_fifo ON tax_lots(asset, acquired_date ASC, status)
WHERE status IN ('OPEN', 'PARTIAL');

CREATE INDEX idx_tax_lots_lifo ON tax_lots(asset, acquired_date DESC, status)
WHERE status IN ('OPEN', 'PARTIAL');

-- Wash sale detection queries (30-day window)
CREATE INDEX idx_disposals_wash_sale_detection
ON disposals(asset, disposal_date DESC)
INCLUDE (quantity, cost_basis, transaction_id);

-- Tax reporting queries
CREATE INDEX idx_disposals_tax_report
ON disposals(tax_year, term, gain)
INCLUDE (quantity, proceeds, cost_basis);

-- Compliance monitoring queries
CREATE INDEX idx_compliance_violations_active
ON compliance_violations(severity, detected_at DESC)
WHERE resolved = false;

-- Audit trail queries by time range
CREATE INDEX idx_audit_trail_time_range
ON audit_trail(timestamp DESC, entity_type)
INCLUDE (agent_id, action, entity_id);

-- ======================
-- MATERIALIZED VIEWS
-- ======================

-- Hot assets (high volume trading)
CREATE MATERIALIZED VIEW hot_assets AS
SELECT
  asset,
  COUNT(*) AS transaction_count,
  SUM(quantity * price) AS total_volume,
  MAX(timestamp) AS last_transaction,
  COUNT(DISTINCT source) AS exchange_count
FROM transactions
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY asset
HAVING COUNT(*) > 10
ORDER BY total_volume DESC;

CREATE UNIQUE INDEX idx_hot_assets_asset ON hot_assets(asset);

-- Tax year summary
CREATE MATERIALIZED VIEW tax_year_summary AS
SELECT
  EXTRACT(YEAR FROM d.disposal_date)::INTEGER AS tax_year,
  COUNT(DISTINCT d.asset) AS assets_disposed,
  COUNT(*) AS total_disposals,
  SUM(CASE WHEN d.term = 'SHORT' THEN d.gain ELSE 0 END) AS short_term_total,
  SUM(CASE WHEN d.term = 'LONG' THEN d.gain ELSE 0 END) AS long_term_total,
  SUM(d.gain) AS total_gain,
  SUM(d.wash_sale_disallowed) AS total_wash_sales
FROM disposals d
GROUP BY EXTRACT(YEAR FROM d.disposal_date)::INTEGER
ORDER BY tax_year DESC;

CREATE UNIQUE INDEX idx_tax_year_summary_year ON tax_year_summary(tax_year);

-- ======================
-- PARTITIONING
-- ======================

-- Partition audit_trail by year for better performance
-- (Note: This requires recreating the table, left as a future optimization)
-- ALTER TABLE audit_trail PARTITION BY RANGE (timestamp);

-- ======================
-- STATISTICS
-- ======================

-- Update statistics for query planner
ANALYZE transactions;
ANALYZE tax_lots;
ANALYZE disposals;
ANALYZE embeddings;
ANALYZE reasoning_bank;
ANALYZE audit_trail;

-- ======================
-- CONSTRAINTS
-- ======================

-- Add check constraint to ensure disposal date is after acquisition
ALTER TABLE disposals
ADD CONSTRAINT check_disposal_after_acquisition
CHECK (
  disposal_date >= (
    SELECT acquired_date
    FROM tax_lots
    WHERE id = lot_id
  )
);

-- ======================
-- FUNCTIONS FOR PERFORMANCE
-- ======================

-- Batch refresh for all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views() RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY positions;
  REFRESH MATERIALIZED VIEW CONCURRENTLY hot_assets;
  REFRESH MATERIALIZED VIEW CONCURRENTLY tax_year_summary;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh (requires pg_cron extension)
-- SELECT cron.schedule('refresh_views', '0 * * * *', 'SELECT refresh_all_materialized_views()');

-- Add comment
COMMENT ON INDEX idx_transactions_asset_type_timestamp IS 'Optimized index for transaction analysis queries';
COMMENT ON INDEX idx_tax_lots_fifo IS 'FIFO lot matching optimization';
COMMENT ON INDEX idx_tax_lots_lifo IS 'LIFO lot matching optimization';
COMMENT ON MATERIALIZED VIEW hot_assets IS 'High-volume trading assets (last 30 days)';
COMMENT ON MATERIALIZED VIEW tax_year_summary IS 'Annual tax summary aggregated from disposals';
