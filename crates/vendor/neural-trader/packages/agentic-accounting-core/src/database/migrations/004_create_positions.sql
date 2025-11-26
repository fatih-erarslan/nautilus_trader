-- Migration: Create positions materialized view
-- Description: Current holdings aggregated from tax lots

CREATE MATERIALIZED VIEW positions AS
SELECT
  asset,
  SUM(quantity) AS total_quantity,
  SUM(cost_basis) AS total_cost_basis,
  CASE
    WHEN SUM(quantity) > 0 THEN SUM(cost_basis) / SUM(quantity)
    ELSE 0
  END AS average_cost_basis,
  0::DECIMAL(30, 18) AS current_price,  -- Updated via external price feed
  0::DECIMAL(30, 18) AS market_value,   -- Updated via refresh
  0::DECIMAL(30, 18) AS unrealized_gain,
  0::DECIMAL(30, 2) AS unrealized_gain_percent,
  NOW() AS last_updated
FROM tax_lots
WHERE status != 'CLOSED'
GROUP BY asset
HAVING SUM(quantity) > 0;

-- Unique index on asset (also enables REFRESH CONCURRENTLY)
CREATE UNIQUE INDEX idx_positions_asset ON positions(asset);

-- Index for sorting by value
CREATE INDEX idx_positions_market_value ON positions(market_value DESC);

-- Index for filtering by gain
CREATE INDEX idx_positions_unrealized_gain ON positions(unrealized_gain);

-- Function to refresh positions with current prices
CREATE OR REPLACE FUNCTION refresh_positions_with_prices(
  price_updates JSONB
) RETURNS void AS $$
BEGIN
  -- Refresh the materialized view
  REFRESH MATERIALIZED VIEW CONCURRENTLY positions;

  -- Update prices from JSONB input
  -- Expected format: {"BTC": 50000, "ETH": 3000, ...}
  UPDATE positions p
  SET
    current_price = (price_updates->>p.asset)::DECIMAL,
    market_value = total_quantity * (price_updates->>p.asset)::DECIMAL,
    unrealized_gain = (total_quantity * (price_updates->>p.asset)::DECIMAL) - total_cost_basis,
    unrealized_gain_percent =
      CASE
        WHEN total_cost_basis > 0 THEN
          ((total_quantity * (price_updates->>p.asset)::DECIMAL) - total_cost_basis) / total_cost_basis * 100
        ELSE 0
      END,
    last_updated = NOW()
  WHERE price_updates ? p.asset;
END;
$$ LANGUAGE plpgsql;

-- Add comment
COMMENT ON MATERIALIZED VIEW positions IS 'Current holdings with unrealized gains/losses (refreshed from tax_lots)';
