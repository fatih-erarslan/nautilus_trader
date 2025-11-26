-- Neural Trader Database Schema
-- Comprehensive schema for trading data, neural networks, and real-time capabilities

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Custom types
CREATE TYPE bot_status AS ENUM ('active', 'paused', 'stopped', 'error', 'training');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'rejected', 'partial');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop', 'stop_limit');
CREATE TYPE position_side AS ENUM ('long', 'short');
CREATE TYPE model_status AS ENUM ('training', 'trained', 'deployed', 'deprecated');
CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'error', 'critical');
CREATE TYPE deployment_status AS ENUM ('pending', 'running', 'stopped', 'failed', 'terminated');

-- Users and authentication
CREATE TABLE profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    full_name TEXT,
    avatar_url TEXT,
    tier TEXT DEFAULT 'basic' CHECK (tier IN ('basic', 'pro', 'enterprise')),
    api_quota INTEGER DEFAULT 1000,
    api_usage INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market data tables
CREATE TABLE symbols (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    exchange TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK (asset_type IN ('stock', 'crypto', 'forex', 'commodity')),
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE market_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    symbol_id UUID REFERENCES symbols(id) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    timeframe TEXT NOT NULL CHECK (timeframe IN ('1m', '5m', '15m', '30m', '1h', '4h', '1d')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol_id, timestamp, timeframe)
);

CREATE TABLE news_data (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    source TEXT NOT NULL,
    url TEXT,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols TEXT[] DEFAULT '{}',
    sentiment_score DECIMAL(3,2), -- -1.0 to 1.0
    relevance_score DECIMAL(3,2), -- 0.0 to 1.0
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading data tables
CREATE TABLE trading_accounts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    name TEXT NOT NULL,
    broker TEXT NOT NULL,
    account_type TEXT DEFAULT 'paper' CHECK (account_type IN ('paper', 'live')),
    balance DECIMAL(18,8) DEFAULT 0,
    equity DECIMAL(18,8) DEFAULT 0,
    margin_used DECIMAL(18,8) DEFAULT 0,
    margin_available DECIMAL(18,8) DEFAULT 0,
    api_credentials JSONB, -- encrypted
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE positions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    account_id UUID REFERENCES trading_accounts(id) NOT NULL,
    symbol_id UUID REFERENCES symbols(id) NOT NULL,
    side position_side NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    entry_price DECIMAL(18,8) NOT NULL,
    current_price DECIMAL(18,8),
    unrealized_pnl DECIMAL(18,8) DEFAULT 0,
    realized_pnl DECIMAL(18,8) DEFAULT 0,
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE orders (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    account_id UUID REFERENCES trading_accounts(id) NOT NULL,
    symbol_id UUID REFERENCES symbols(id) NOT NULL,
    order_type order_type NOT NULL,
    side position_side NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(18,8),
    stop_price DECIMAL(18,8),
    status order_status DEFAULT 'pending',
    filled_quantity DECIMAL(18,8) DEFAULT 0,
    average_fill_price DECIMAL(18,8),
    commission DECIMAL(18,8) DEFAULT 0,
    external_order_id TEXT,
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Neural network and AI model tables
CREATE TABLE neural_models (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('lstm', 'transformer', 'cnn', 'ensemble')),
    architecture JSONB NOT NULL,
    parameters JSONB DEFAULT '{}',
    status model_status DEFAULT 'training',
    version INTEGER DEFAULT 1,
    performance_metrics JSONB DEFAULT '{}',
    training_data_hash TEXT,
    model_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE training_runs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_id UUID REFERENCES neural_models(id) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    epoch INTEGER DEFAULT 0,
    loss DECIMAL(10,6),
    accuracy DECIMAL(5,4),
    validation_loss DECIMAL(10,6),
    validation_accuracy DECIMAL(5,4),
    hyperparameters JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    logs TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE model_predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_id UUID REFERENCES neural_models(id) NOT NULL,
    symbol_id UUID REFERENCES symbols(id) NOT NULL,
    prediction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    prediction_value DECIMAL(18,8) NOT NULL,
    confidence DECIMAL(3,2), -- 0.0 to 1.0
    actual_value DECIMAL(18,8),
    error DECIMAL(18,8),
    features JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading bot tables
CREATE TABLE trading_bots (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    account_id UUID REFERENCES trading_accounts(id) NOT NULL,
    name TEXT NOT NULL,
    strategy_type TEXT NOT NULL,
    configuration JSONB NOT NULL,
    model_ids UUID[] DEFAULT '{}',
    symbols TEXT[] NOT NULL,
    status bot_status DEFAULT 'paused',
    max_position_size DECIMAL(18,8) DEFAULT 1000,
    risk_limit DECIMAL(18,8) DEFAULT 0.05, -- 5% max risk
    is_active BOOLEAN DEFAULT true,
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE bot_executions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    bot_id UUID REFERENCES trading_bots(id) NOT NULL,
    symbol_id UUID REFERENCES symbols(id) NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('buy', 'sell', 'hold')),
    signal_strength DECIMAL(3,2), -- 0.0 to 1.0
    reasoning TEXT,
    order_id UUID REFERENCES orders(id),
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- E2B sandbox integration tables
CREATE TABLE sandbox_deployments (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    bot_id UUID REFERENCES trading_bots(id),
    sandbox_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    template TEXT DEFAULT 'base',
    configuration JSONB DEFAULT '{}',
    status deployment_status DEFAULT 'pending',
    cpu_count INTEGER DEFAULT 1,
    memory_mb INTEGER DEFAULT 512,
    timeout_seconds INTEGER DEFAULT 300,
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    resource_usage JSONB DEFAULT '{}',
    logs TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance and analytics tables
CREATE TABLE performance_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('bot', 'model', 'account', 'system')),
    entity_id UUID NOT NULL,
    metric_type TEXT NOT NULL,
    metric_value DECIMAL(18,8) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE alerts (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    severity alert_severity DEFAULT 'info',
    entity_type TEXT,
    entity_id UUID,
    is_read BOOLEAN DEFAULT false,
    expires_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit and logging
CREATE TABLE audit_logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id),
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol_id, timestamp DESC);
CREATE INDEX idx_market_data_timeframe ON market_data(timeframe, timestamp DESC);
CREATE INDEX idx_news_data_published ON news_data(published_at DESC);
CREATE INDEX idx_news_data_symbols ON news_data USING GIN(symbols);
CREATE INDEX idx_positions_account ON positions(account_id, opened_at DESC);
CREATE INDEX idx_orders_account_status ON orders(account_id, status, placed_at DESC);
CREATE INDEX idx_training_runs_model ON training_runs(model_id, started_at DESC);
CREATE INDEX idx_predictions_model_timestamp ON model_predictions(model_id, prediction_timestamp DESC);
CREATE INDEX idx_bot_executions_bot_timestamp ON bot_executions(bot_id, executed_at DESC);
CREATE INDEX idx_performance_metrics_entity ON performance_metrics(entity_type, entity_id, timestamp DESC);
CREATE INDEX idx_alerts_user_created ON alerts(user_id, created_at DESC);
CREATE INDEX idx_audit_logs_user_created ON audit_logs(user_id, created_at DESC);

-- Updated at triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_profiles_updated_at BEFORE UPDATE ON profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_neural_models_updated_at BEFORE UPDATE ON neural_models FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_training_runs_updated_at BEFORE UPDATE ON training_runs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trading_bots_updated_at BEFORE UPDATE ON trading_bots FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_sandbox_deployments_updated_at BEFORE UPDATE ON sandbox_deployments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();