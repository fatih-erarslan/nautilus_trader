-- Database Functions
-- Custom functions for neural trading platform

-- Function to calculate portfolio performance
CREATE OR REPLACE FUNCTION calculate_portfolio_performance(
  account_id_param UUID,
  start_date TIMESTAMP WITH TIME ZONE DEFAULT NULL,
  end_date TIMESTAMP WITH TIME ZONE DEFAULT NULL
)
RETURNS TABLE (
  total_return DECIMAL(18,8),
  realized_pnl DECIMAL(18,8),
  unrealized_pnl DECIMAL(18,8),
  win_rate DECIMAL(5,4),
  sharpe_ratio DECIMAL(8,4),
  max_drawdown DECIMAL(8,4)
) AS $$
DECLARE
  start_dt TIMESTAMP WITH TIME ZONE;
  end_dt TIMESTAMP WITH TIME ZONE;
BEGIN
  start_dt := COALESCE(start_date, NOW() - INTERVAL '30 days');
  end_dt := COALESCE(end_date, NOW());
  
  RETURN QUERY
  WITH portfolio_stats AS (
    SELECT 
      SUM(p.realized_pnl) as total_realized,
      SUM(p.unrealized_pnl) as total_unrealized,
      COUNT(CASE WHEN p.realized_pnl > 0 THEN 1 END)::DECIMAL / NULLIF(COUNT(CASE WHEN p.closed_at IS NOT NULL THEN 1 END), 0) as win_percentage,
      ta.balance
    FROM positions p
    JOIN trading_accounts ta ON ta.id = p.account_id
    WHERE p.account_id = account_id_param
      AND p.opened_at >= start_dt
      AND p.opened_at <= end_dt
    GROUP BY ta.balance
  )
  SELECT 
    ps.total_realized + ps.total_unrealized as total_return,
    ps.total_realized as realized_pnl,
    ps.total_unrealized as unrealized_pnl,
    ps.win_percentage as win_rate,
    -- Simplified Sharpe ratio calculation
    CASE 
      WHEN ps.total_realized + ps.total_unrealized > 0 THEN
        (ps.total_realized + ps.total_unrealized) / NULLIF(ps.balance * 0.01, 0)
      ELSE 0
    END as sharpe_ratio,
    -- Simplified max drawdown
    CASE 
      WHEN ps.total_realized + ps.total_unrealized < 0 THEN
        ABS(ps.total_realized + ps.total_unrealized) / NULLIF(ps.balance, 0)
      ELSE 0
    END as max_drawdown
  FROM portfolio_stats ps;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to update model performance metrics
CREATE OR REPLACE FUNCTION update_model_performance(
  model_id_param UUID,
  predictions_count INTEGER DEFAULT 100
)
RETURNS JSONB AS $$
DECLARE
  accuracy DECIMAL(5,4);
  mae DECIMAL(10,6);
  rmse DECIMAL(10,6);
  result JSONB;
BEGIN
  -- Calculate model accuracy metrics
  WITH prediction_stats AS (
    SELECT 
      AVG(ABS(prediction_value - actual_value)) as mean_absolute_error,
      SQRT(AVG(POWER(prediction_value - actual_value, 2))) as root_mean_square_error,
      COUNT(*) as total_predictions,
      COUNT(CASE WHEN ABS(prediction_value - actual_value) / NULLIF(actual_value, 0) <= 0.05 THEN 1 END) as accurate_predictions
    FROM model_predictions
    WHERE model_id = model_id_param
      AND actual_value IS NOT NULL
      AND created_at >= NOW() - INTERVAL '7 days'
    ORDER BY created_at DESC
    LIMIT predictions_count
  )
  SELECT 
    ps.accurate_predictions::DECIMAL / NULLIF(ps.total_predictions, 0) as acc,
    ps.mean_absolute_error as mae_val,
    ps.root_mean_square_error as rmse_val
  INTO accuracy, mae, rmse
  FROM prediction_stats ps;
  
  -- Build result JSON
  result := jsonb_build_object(
    'accuracy', COALESCE(accuracy, 0),
    'mae', COALESCE(mae, 0),
    'rmse', COALESCE(rmse, 0),
    'predictions_analyzed', predictions_count,
    'last_updated', NOW()
  );
  
  -- Update the model's performance metrics
  UPDATE neural_models
  SET 
    performance_metrics = result,
    updated_at = NOW()
  WHERE id = model_id_param;
  
  RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to generate trading signals
CREATE OR REPLACE FUNCTION generate_trading_signal(
  symbol_param TEXT,
  model_ids UUID[],
  lookback_periods INTEGER DEFAULT 20
)
RETURNS JSONB AS $$
DECLARE
  signal_strength DECIMAL(3,2);
  signal_direction TEXT;
  confidence DECIMAL(3,2);
  reasoning TEXT[];
  result JSONB;
BEGIN
  -- Get latest predictions from specified models
  WITH model_signals AS (
    SELECT 
      mp.model_id,
      mp.prediction_value,
      mp.confidence,
      nm.name as model_name,
      ROW_NUMBER() OVER (PARTITION BY mp.model_id ORDER BY mp.prediction_timestamp DESC) as rn
    FROM model_predictions mp
    JOIN neural_models nm ON nm.id = mp.model_id
    JOIN symbols s ON s.id = mp.symbol_id
    WHERE s.symbol = symbol_param
      AND mp.model_id = ANY(model_ids)
      AND mp.prediction_timestamp >= NOW() - INTERVAL '1 hour'
  ),
  aggregated_signal AS (
    SELECT 
      AVG(ms.prediction_value) as avg_prediction,
      AVG(ms.confidence) as avg_confidence,
      COUNT(*) as model_count,
      ARRAY_AGG(ms.model_name || ': ' || ms.prediction_value::TEXT) as model_predictions
    FROM model_signals ms
    WHERE ms.rn = 1
  )
  SELECT 
    CASE 
      WHEN avg_prediction > 0.6 THEN 0.8
      WHEN avg_prediction > 0.3 THEN 0.5
      WHEN avg_prediction < -0.6 THEN -0.8
      WHEN avg_prediction < -0.3 THEN -0.5
      ELSE 0
    END,
    CASE 
      WHEN avg_prediction > 0.3 THEN 'BUY'
      WHEN avg_prediction < -0.3 THEN 'SELL'
      ELSE 'HOLD'
    END,
    avg_confidence,
    model_predictions
  INTO signal_strength, signal_direction, confidence, reasoning
  FROM aggregated_signal;
  
  -- Build result
  result := jsonb_build_object(
    'symbol', symbol_param,
    'signal', signal_direction,
    'strength', COALESCE(signal_strength, 0),
    'confidence', COALESCE(confidence, 0),
    'reasoning', COALESCE(reasoning, ARRAY[]::TEXT[]),
    'generated_at', NOW()
  );
  
  RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to calculate risk metrics
CREATE OR REPLACE FUNCTION calculate_position_risk(
  account_id_param UUID,
  symbol_param TEXT,
  position_size DECIMAL(18,8)
)
RETURNS JSONB AS $$
DECLARE
  account_balance DECIMAL(18,8);
  current_exposure DECIMAL(18,8);
  symbol_volatility DECIMAL(8,4);
  var_95 DECIMAL(18,8);
  risk_percentage DECIMAL(5,4);
  result JSONB;
BEGIN
  -- Get account balance
  SELECT balance INTO account_balance
  FROM trading_accounts
  WHERE id = account_id_param;
  
  -- Calculate current exposure for this symbol
  SELECT COALESCE(SUM(ABS(quantity * current_price)), 0)
  INTO current_exposure
  FROM positions p
  JOIN symbols s ON s.id = p.symbol_id
  WHERE p.account_id = account_id_param
    AND s.symbol = symbol_param
    AND p.closed_at IS NULL;
  
  -- Calculate symbol volatility (simplified using recent data)
  WITH price_changes AS (
    SELECT 
      (close - LAG(close) OVER (ORDER BY timestamp)) / LAG(close) OVER (ORDER BY timestamp) as return
    FROM market_data md
    JOIN symbols s ON s.id = md.symbol_id
    WHERE s.symbol = symbol_param
      AND md.timeframe = '1d'
      AND md.timestamp >= NOW() - INTERVAL '30 days'
    ORDER BY md.timestamp DESC
    LIMIT 30
  )
  SELECT STDDEV(return) INTO symbol_volatility
  FROM price_changes
  WHERE return IS NOT NULL;
  
  -- Calculate VaR (95% confidence)
  var_95 := position_size * COALESCE(symbol_volatility, 0.02) * 1.65; -- 95% confidence interval
  
  -- Calculate risk as percentage of account
  risk_percentage := var_95 / NULLIF(account_balance, 0);
  
  result := jsonb_build_object(
    'account_balance', account_balance,
    'current_exposure', current_exposure,
    'new_position_size', position_size,
    'symbol_volatility', COALESCE(symbol_volatility, 0),
    'var_95', var_95,
    'risk_percentage', COALESCE(risk_percentage, 0),
    'risk_level', CASE 
      WHEN risk_percentage > 0.1 THEN 'HIGH'
      WHEN risk_percentage > 0.05 THEN 'MEDIUM'
      ELSE 'LOW'
    END,
    'calculated_at', NOW()
  );
  
  RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to clean old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS INTEGER AS $$
DECLARE
  deleted_count INTEGER := 0;
BEGIN
  -- Clean old market data (keep 1 year)
  DELETE FROM market_data 
  WHERE timestamp < NOW() - INTERVAL '1 year';
  GET DIAGNOSTICS deleted_count = ROW_COUNT;
  
  -- Clean old predictions (keep 3 months)
  DELETE FROM model_predictions 
  WHERE created_at < NOW() - INTERVAL '3 months';
  
  -- Clean old performance metrics (keep 6 months)
  DELETE FROM performance_metrics 
  WHERE timestamp < NOW() - INTERVAL '6 months';
  
  -- Clean old audit logs (keep 1 year)
  DELETE FROM audit_logs 
  WHERE created_at < NOW() - INTERVAL '1 year';
  
  -- Clean read alerts (keep 1 month)
  DELETE FROM alerts 
  WHERE is_read = true 
    AND created_at < NOW() - INTERVAL '1 month';
  
  RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to create audit log entry
CREATE OR REPLACE FUNCTION create_audit_log()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit_logs (
    user_id,
    action,
    entity_type,
    entity_id,
    old_values,
    new_values,
    created_at
  ) VALUES (
    auth.user_id(),
    TG_OP,
    TG_TABLE_NAME,
    COALESCE(NEW.id, OLD.id),
    CASE WHEN TG_OP = 'DELETE' THEN to_jsonb(OLD) ELSE NULL END,
    CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN to_jsonb(NEW) ELSE NULL END,
    NOW()
  );
  
  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Apply audit triggers to important tables
CREATE TRIGGER audit_trading_accounts AFTER INSERT OR UPDATE OR DELETE ON trading_accounts
  FOR EACH ROW EXECUTE FUNCTION create_audit_log();

CREATE TRIGGER audit_orders AFTER INSERT OR UPDATE OR DELETE ON orders
  FOR EACH ROW EXECUTE FUNCTION create_audit_log();

CREATE TRIGGER audit_positions AFTER INSERT OR UPDATE OR DELETE ON positions
  FOR EACH ROW EXECUTE FUNCTION create_audit_log();

CREATE TRIGGER audit_trading_bots AFTER INSERT OR UPDATE OR DELETE ON trading_bots
  FOR EACH ROW EXECUTE FUNCTION create_audit_log();