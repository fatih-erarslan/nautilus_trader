-- Neural Trader Database Initialization Script
-- PostgreSQL 14+

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO trading, public;

-- Orders table
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20, 8),
    order_type VARCHAR(20) NOT NULL DEFAULT 'market' CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    time_in_force VARCHAR(10) DEFAULT 'day' CHECK (time_in_force IN ('day', 'gtc', 'ioc', 'fok')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'submitted', 'filled', 'partial', 'cancelled', 'rejected')),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_price DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    submitted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_orders_user_id ON trading.orders(user_id);
CREATE INDEX idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX idx_orders_status ON trading.orders(status);
CREATE INDEX idx_orders_created_at ON trading.orders(created_at DESC);

-- Positions table
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    average_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,
    UNIQUE(user_id, symbol)
);

CREATE INDEX idx_positions_user_id ON trading.positions(user_id);
CREATE INDEX idx_positions_symbol ON trading.positions(symbol);

-- Portfolio history table
CREATE TABLE IF NOT EXISTS trading.portfolio_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    total_value DECIMAL(20, 8) NOT NULL,
    cash DECIMAL(20, 8) NOT NULL,
    positions_value DECIMAL(20, 8) NOT NULL,
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_portfolio_history_user_id ON trading.portfolio_history(user_id);
CREATE INDEX idx_portfolio_history_timestamp ON trading.portfolio_history(timestamp DESC);

-- Trades table (executed trades)
CREATE TABLE IF NOT EXISTS trading.trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES trading.orders(id),
    user_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    commission DECIMAL(20, 8) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_trades_order_id ON trading.trades(order_id);
CREATE INDEX idx_trades_user_id ON trading.trades(user_id);
CREATE INDEX idx_trades_symbol ON trading.trades(symbol);
CREATE INDEX idx_trades_executed_at ON trading.trades(executed_at DESC);

-- Market data table
CREATE TABLE IF NOT EXISTS trading.market_data (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    timeframe VARCHAR(10) NOT NULL DEFAULT '1m',
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE INDEX idx_market_data_symbol_timestamp ON trading.market_data(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timeframe ON trading.market_data(timeframe);

-- Strategy performance table
CREATE TABLE IF NOT EXISTS analytics.strategy_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_name VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20, 8) DEFAULT 0,
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),
    start_date TIMESTAMP WITH TIME ZONE,
    end_date TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_strategy_performance_user_id ON analytics.strategy_performance(user_id);
CREATE INDEX idx_strategy_performance_strategy_name ON analytics.strategy_performance(strategy_name);

-- Risk events table
CREATE TABLE IF NOT EXISTS trading.risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    description TEXT NOT NULL,
    triggered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

CREATE INDEX idx_risk_events_user_id ON trading.risk_events(user_id);
CREATE INDEX idx_risk_events_severity ON trading.risk_events(severity);
CREATE INDEX idx_risk_events_triggered_at ON trading.risk_events(triggered_at DESC);

-- Audit log table
CREATE TABLE IF NOT EXISTS trading.audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    old_value JSONB,
    new_value JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_log_user_id ON trading.audit_log(user_id);
CREATE INDEX idx_audit_log_timestamp ON trading.audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_action ON trading.audit_log(action);

-- Functions

-- Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views

-- Active positions view
CREATE OR REPLACE VIEW trading.active_positions AS
SELECT * FROM trading.positions
WHERE closed_at IS NULL;

-- Open orders view
CREATE OR REPLACE VIEW trading.open_orders AS
SELECT * FROM trading.orders
WHERE status IN ('pending', 'submitted', 'partial');

-- Daily portfolio performance
CREATE OR REPLACE VIEW analytics.daily_portfolio_performance AS
SELECT
    user_id,
    DATE(timestamp) as date,
    MAX(total_value) as end_value,
    MIN(total_value) as low_value,
    MAX(total_value) as high_value,
    MAX(unrealized_pnl) as unrealized_pnl,
    MAX(realized_pnl) as realized_pnl
FROM trading.portfolio_history
GROUP BY user_id, DATE(timestamp)
ORDER BY date DESC;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON SCHEMA trading TO neural;
-- GRANT ALL PRIVILEGES ON SCHEMA analytics TO neural;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO neural;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO neural;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO neural;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO neural;

-- Insert sample data (optional, for development)
-- Uncomment for development environments
/*
INSERT INTO trading.orders (user_id, symbol, side, quantity, order_type, status, price)
VALUES
    (uuid_generate_v4(), 'AAPL', 'buy', 10, 'limit', 'filled', 150.00),
    (uuid_generate_v4(), 'MSFT', 'buy', 5, 'market', 'filled', 300.00),
    (uuid_generate_v4(), 'GOOGL', 'sell', 2, 'limit', 'filled', 2800.00);
*/

-- Completion message
DO $$
BEGIN
    RAISE NOTICE 'Neural Trader database initialized successfully';
END $$;
