-- Beefy Finance Crypto Trading Database Schema
-- SQLite implementation with comprehensive tables for vault positions, yield tracking, and portfolio management

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Vault positions table
-- Tracks individual positions in Beefy Finance vaults
CREATE TABLE IF NOT EXISTS vault_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id TEXT NOT NULL,
    vault_name TEXT NOT NULL,
    chain TEXT NOT NULL,
    amount_deposited REAL NOT NULL,
    shares_owned REAL NOT NULL,
    current_value REAL DEFAULT 0,
    entry_price REAL NOT NULL,
    entry_apy REAL NOT NULL,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'closed', 'pending')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Yield history table
-- Records historical yield data for positions
CREATE TABLE IF NOT EXISTS yield_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vault_id TEXT NOT NULL,
    position_id INTEGER NOT NULL,
    earned_amount REAL NOT NULL,
    apy_snapshot REAL NOT NULL,
    tvl_snapshot REAL,
    price_per_share REAL NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (position_id) REFERENCES vault_positions(id) ON DELETE CASCADE
);

-- Crypto transactions table
-- Logs all blockchain transactions
CREATE TABLE IF NOT EXISTS crypto_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('deposit', 'withdraw', 'claim', 'compound')),
    vault_id TEXT NOT NULL,
    chain TEXT NOT NULL,
    amount REAL NOT NULL,
    gas_used REAL,
    tx_hash TEXT UNIQUE,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'confirmed', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio summary table
-- Aggregated portfolio metrics over time
CREATE TABLE IF NOT EXISTS portfolio_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    total_value_usd REAL NOT NULL,
    total_yield_earned REAL NOT NULL,
    average_apy REAL NOT NULL,
    chains_active TEXT NOT NULL, -- JSON array of active chains
    vaults_count INTEGER NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_vault_positions_vault_id ON vault_positions(vault_id);
CREATE INDEX idx_vault_positions_chain ON vault_positions(chain);
CREATE INDEX idx_vault_positions_status ON vault_positions(status);
CREATE INDEX idx_vault_positions_created ON vault_positions(created_at);

CREATE INDEX idx_yield_history_vault_id ON yield_history(vault_id);
CREATE INDEX idx_yield_history_position_id ON yield_history(position_id);
CREATE INDEX idx_yield_history_recorded ON yield_history(recorded_at);

CREATE INDEX idx_crypto_transactions_vault_id ON crypto_transactions(vault_id);
CREATE INDEX idx_crypto_transactions_chain ON crypto_transactions(chain);
CREATE INDEX idx_crypto_transactions_status ON crypto_transactions(status);
CREATE INDEX idx_crypto_transactions_tx_hash ON crypto_transactions(tx_hash);

CREATE INDEX idx_portfolio_summary_recorded ON portfolio_summary(recorded_at);

-- Create triggers for updated_at
CREATE TRIGGER update_vault_positions_timestamp 
AFTER UPDATE ON vault_positions
BEGIN
    UPDATE vault_positions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- View for active positions with latest yields
CREATE VIEW IF NOT EXISTS active_positions_summary AS
SELECT 
    vp.id,
    vp.vault_id,
    vp.vault_name,
    vp.chain,
    vp.amount_deposited,
    vp.shares_owned,
    vp.current_value,
    vp.entry_price,
    vp.entry_apy,
    vp.created_at,
    COALESCE(yh.total_earned, 0) as total_earned,
    COALESCE(yh.latest_apy, vp.entry_apy) as current_apy
FROM vault_positions vp
LEFT JOIN (
    SELECT 
        position_id,
        SUM(earned_amount) as total_earned,
        MAX(apy_snapshot) as latest_apy
    FROM yield_history
    GROUP BY position_id
) yh ON vp.id = yh.position_id
WHERE vp.status = 'active';

-- View for daily portfolio performance
CREATE VIEW IF NOT EXISTS daily_portfolio_performance AS
SELECT 
    DATE(recorded_at) as date,
    AVG(total_value_usd) as avg_portfolio_value,
    AVG(average_apy) as avg_apy,
    MAX(total_yield_earned) as total_yield,
    MAX(vaults_count) as active_vaults
FROM portfolio_summary
GROUP BY DATE(recorded_at)
ORDER BY date DESC;