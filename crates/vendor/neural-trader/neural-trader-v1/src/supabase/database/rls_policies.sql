-- Row Level Security Policies
-- Comprehensive RLS setup for multi-tenant neural trading platform

-- Enable RLS on all tables
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE symbols ENABLE ROW LEVEL SECURITY;
ALTER TABLE market_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE neural_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_bots ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE sandbox_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Helper function to check user access
CREATE OR REPLACE FUNCTION auth.user_id() RETURNS UUID AS $$
  SELECT auth.uid()
$$ LANGUAGE sql STABLE;

-- Profiles policies
CREATE POLICY "Users can view own profile" ON profiles
  FOR SELECT USING (auth.user_id() = id);

CREATE POLICY "Users can update own profile" ON profiles
  FOR UPDATE USING (auth.user_id() = id);

CREATE POLICY "Users can insert own profile" ON profiles
  FOR INSERT WITH CHECK (auth.user_id() = id);

-- Symbols policies (public read, admin write)
CREATE POLICY "Anyone can view symbols" ON symbols
  FOR SELECT USING (true);

CREATE POLICY "Admins can manage symbols" ON symbols
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles 
      WHERE id = auth.user_id() 
      AND tier = 'enterprise'
    )
  );

-- Market data policies (public read, admin write)
CREATE POLICY "Anyone can view market data" ON market_data
  FOR SELECT USING (true);

CREATE POLICY "Admins can manage market data" ON market_data
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles 
      WHERE id = auth.user_id() 
      AND tier = 'enterprise'
    )
  );

-- News data policies (public read, admin write)
CREATE POLICY "Anyone can view news data" ON news_data
  FOR SELECT USING (true);

CREATE POLICY "Admins can manage news data" ON news_data
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM profiles 
      WHERE id = auth.user_id() 
      AND tier = 'enterprise'
    )
  );

-- Trading accounts policies
CREATE POLICY "Users can view own trading accounts" ON trading_accounts
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "Users can manage own trading accounts" ON trading_accounts
  FOR ALL USING (auth.user_id() = user_id);

-- Positions policies
CREATE POLICY "Users can view own positions" ON positions
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM trading_accounts 
      WHERE id = positions.account_id 
      AND user_id = auth.user_id()
    )
  );

CREATE POLICY "Users can manage own positions" ON positions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM trading_accounts 
      WHERE id = positions.account_id 
      AND user_id = auth.user_id()
    )
  );

-- Orders policies
CREATE POLICY "Users can view own orders" ON orders
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM trading_accounts 
      WHERE id = orders.account_id 
      AND user_id = auth.user_id()
    )
  );

CREATE POLICY "Users can manage own orders" ON orders
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM trading_accounts 
      WHERE id = orders.account_id 
      AND user_id = auth.user_id()
    )
  );

-- Neural models policies
CREATE POLICY "Users can view own models" ON neural_models
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "Users can manage own models" ON neural_models
  FOR ALL USING (auth.user_id() = user_id);

-- Training runs policies
CREATE POLICY "Users can view own training runs" ON training_runs
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM neural_models 
      WHERE id = training_runs.model_id 
      AND user_id = auth.user_id()
    )
  );

CREATE POLICY "Users can manage own training runs" ON training_runs
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM neural_models 
      WHERE id = training_runs.model_id 
      AND user_id = auth.user_id()
    )
  );

-- Model predictions policies
CREATE POLICY "Users can view own predictions" ON model_predictions
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM neural_models 
      WHERE id = model_predictions.model_id 
      AND user_id = auth.user_id()
    )
  );

CREATE POLICY "Users can manage own predictions" ON model_predictions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM neural_models 
      WHERE id = model_predictions.model_id 
      AND user_id = auth.user_id()
    )
  );

-- Trading bots policies
CREATE POLICY "Users can view own bots" ON trading_bots
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "Users can manage own bots" ON trading_bots
  FOR ALL USING (auth.user_id() = user_id);

-- Bot executions policies
CREATE POLICY "Users can view own bot executions" ON bot_executions
  FOR SELECT USING (
    EXISTS (
      SELECT 1 FROM trading_bots 
      WHERE id = bot_executions.bot_id 
      AND user_id = auth.user_id()
    )
  );

CREATE POLICY "Users can manage own bot executions" ON bot_executions
  FOR ALL USING (
    EXISTS (
      SELECT 1 FROM trading_bots 
      WHERE id = bot_executions.bot_id 
      AND user_id = auth.user_id()
    )
  );

-- Sandbox deployments policies
CREATE POLICY "Users can view own deployments" ON sandbox_deployments
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "Users can manage own deployments" ON sandbox_deployments
  FOR ALL USING (auth.user_id() = user_id);

-- Performance metrics policies
CREATE POLICY "Users can view own metrics" ON performance_metrics
  FOR SELECT USING (
    CASE entity_type
      WHEN 'bot' THEN EXISTS (
        SELECT 1 FROM trading_bots 
        WHERE id = performance_metrics.entity_id::UUID 
        AND user_id = auth.user_id()
      )
      WHEN 'model' THEN EXISTS (
        SELECT 1 FROM neural_models 
        WHERE id = performance_metrics.entity_id::UUID 
        AND user_id = auth.user_id()
      )
      WHEN 'account' THEN EXISTS (
        SELECT 1 FROM trading_accounts 
        WHERE id = performance_metrics.entity_id::UUID 
        AND user_id = auth.user_id()
      )
      ELSE false
    END
  );

CREATE POLICY "System can insert metrics" ON performance_metrics
  FOR INSERT WITH CHECK (true);

-- Alerts policies
CREATE POLICY "Users can view own alerts" ON alerts
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "Users can update own alerts" ON alerts
  FOR UPDATE USING (auth.user_id() = user_id);

CREATE POLICY "System can create alerts" ON alerts
  FOR INSERT WITH CHECK (true);

-- Audit logs policies
CREATE POLICY "Users can view own audit logs" ON audit_logs
  FOR SELECT USING (auth.user_id() = user_id);

CREATE POLICY "System can insert audit logs" ON audit_logs
  FOR INSERT WITH CHECK (true);

-- Service role policies (for backend operations)
CREATE POLICY "Service role full access" ON profiles
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role market data access" ON market_data
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role news data access" ON news_data
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role performance access" ON performance_metrics
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role alerts access" ON alerts
  FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role audit access" ON audit_logs
  FOR ALL USING (auth.role() = 'service_role');