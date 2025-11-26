-- Seed: Sample transactions for testing
-- Description: Realistic transaction data for development and testing

-- Bitcoin transactions
INSERT INTO transactions (id, timestamp, type, asset, quantity, price, fees, source, source_id, taxable)
VALUES
  ('11111111-1111-1111-1111-111111111111', '2024-01-15 10:30:00+00', 'BUY', 'BTC', 0.5, 45000, 50, 'coinbase', 'CB-001', true),
  ('11111111-1111-1111-1111-111111111112', '2024-03-20 14:15:00+00', 'BUY', 'BTC', 0.3, 52000, 30, 'coinbase', 'CB-002', true),
  ('11111111-1111-1111-1111-111111111113', '2024-06-10 09:45:00+00', 'SELL', 'BTC', 0.2, 65000, 40, 'coinbase', 'CB-003', true);

-- Ethereum transactions
INSERT INTO transactions (id, timestamp, type, asset, quantity, price, fees, source, source_id, taxable)
VALUES
  ('22222222-2222-2222-2222-222222222221', '2024-02-01 11:00:00+00', 'BUY', 'ETH', 5, 3000, 25, 'binance', 'BN-001', true),
  ('22222222-2222-2222-2222-222222222222', '2024-04-15 16:30:00+00', 'BUY', 'ETH', 3, 3500, 20, 'binance', 'BN-002', true),
  ('22222222-2222-2222-2222-222222222223', '2024-07-20 13:20:00+00', 'SELL', 'ETH', 2, 4200, 15, 'binance', 'BN-003', true);

-- Staking income
INSERT INTO transactions (id, timestamp, type, asset, quantity, price, fees, source, source_id, taxable)
VALUES
  ('33333333-3333-3333-3333-333333333331', '2024-03-01 00:00:00+00', 'INCOME', 'ETH', 0.05, 3200, 0, 'staking', 'STAKE-001', true),
  ('33333333-3333-3333-3333-333333333332', '2024-06-01 00:00:00+00', 'INCOME', 'ETH', 0.06, 3800, 0, 'staking', 'STAKE-002', true);

-- Trading activity
INSERT INTO transactions (id, timestamp, type, asset, quantity, price, fees, source, source_id, taxable)
VALUES
  ('44444444-4444-4444-4444-444444444441', '2024-05-10 10:00:00+00', 'BUY', 'SOL', 100, 150, 15, 'kraken', 'KR-001', true),
  ('44444444-4444-4444-4444-444444444442', '2024-08-15 14:30:00+00', 'SELL', 'SOL', 50, 180, 10, 'kraken', 'KR-002', true);

-- Add comment
COMMENT ON TABLE transactions IS 'Seeded with sample cryptocurrency transactions for testing';
