-- Seed: Sample tax lots derived from transactions
-- Description: Tax lots created from buy transactions

-- Bitcoin tax lots
INSERT INTO tax_lots (id, transaction_id, asset, acquired_date, quantity, original_quantity, cost_basis, unit_cost_basis, source, method, status)
VALUES
  ('a1111111-1111-1111-1111-111111111111', '11111111-1111-1111-1111-111111111111', 'BTC',
   '2024-01-15 10:30:00+00', 0.5, 0.5, 22550, 45100, 'coinbase', 'FIFO', 'OPEN'),
  ('a1111111-1111-1111-1111-111111111112', '11111111-1111-1111-1111-111111111112', 'BTC',
   '2024-03-20 14:15:00+00', 0.3, 0.3, 15630, 52100, 'coinbase', 'FIFO', 'OPEN');

-- Ethereum tax lots
INSERT INTO tax_lots (id, transaction_id, asset, acquired_date, quantity, original_quantity, cost_basis, unit_cost_basis, source, method, status)
VALUES
  ('a2222222-2222-2222-2222-222222222221', '22222222-2222-2222-2222-222222222221', 'ETH',
   '2024-02-01 11:00:00+00', 5, 5, 15025, 3005, 'binance', 'FIFO', 'OPEN'),
  ('a2222222-2222-2222-2222-222222222222', '22222222-2222-2222-2222-222222222222', 'ETH',
   '2024-04-15 16:30:00+00', 3, 3, 10520, 3506.67, 'binance', 'FIFO', 'OPEN');

-- Staking income lots
INSERT INTO tax_lots (id, transaction_id, asset, acquired_date, quantity, original_quantity, cost_basis, unit_cost_basis, source, method, status)
VALUES
  ('a3333333-3333-3333-3333-333333333331', '33333333-3333-3333-3333-333333333331', 'ETH',
   '2024-03-01 00:00:00+00', 0.05, 0.05, 160, 3200, 'staking', 'FIFO', 'OPEN'),
  ('a3333333-3333-3333-3333-333333333332', '33333333-3333-3333-3333-333333333332', 'ETH',
   '2024-06-01 00:00:00+00', 0.06, 0.06, 228, 3800, 'staking', 'FIFO', 'OPEN');

-- Solana tax lot
INSERT INTO tax_lots (id, transaction_id, asset, acquired_date, quantity, original_quantity, cost_basis, unit_cost_basis, source, method, status)
VALUES
  ('a4444444-4444-4444-4444-444444444441', '44444444-4444-4444-4444-444444444441', 'SOL',
   '2024-05-10 10:00:00+00', 100, 100, 15015, 150.15, 'kraken', 'FIFO', 'OPEN');

-- Add comment
COMMENT ON TABLE tax_lots IS 'Seeded with sample tax lots for testing cost basis calculations';
