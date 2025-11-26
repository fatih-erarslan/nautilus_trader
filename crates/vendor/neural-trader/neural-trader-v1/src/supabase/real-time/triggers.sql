-- Real-time Triggers for Neural Trading Platform
-- Triggers to enhance real-time capabilities and data consistency

-- Function to notify real-time clients of significant events
CREATE OR REPLACE FUNCTION notify_realtime_event()
RETURNS TRIGGER AS $$
DECLARE
  payload JSON;
  channel TEXT;
  event_type TEXT;
BEGIN
  -- Determine the channel and event type based on table and operation
  CASE TG_TABLE_NAME
    WHEN 'market_data' THEN
      channel := 'market_data_channel';
      event_type := 'market_update';
    WHEN 'bot_executions' THEN
      channel := 'trading_signals_channel';
      event_type := 'signal_generated';
    WHEN 'trading_bots' THEN
      channel := 'bot_status_channel';
      event_type := 'bot_status_change';
    WHEN 'alerts' THEN
      channel := 'alerts_channel';
      event_type := 'alert_triggered';
    WHEN 'performance_metrics' THEN
      channel := 'performance_channel';
      event_type := 'performance_update';
    WHEN 'training_runs' THEN
      channel := 'neural_training_channel';
      event_type := 'training_progress';
    WHEN 'orders' THEN
      channel := 'orders_channel';
      event_type := 'order_executed';
    WHEN 'positions' THEN
      channel := 'positions_channel';
      event_type := 'position_update';
    ELSE
      RETURN COALESCE(NEW, OLD);
  END CASE;

  -- Build the payload
  payload := json_build_object(
    'event_type', event_type,
    'table', TG_TABLE_NAME,
    'operation', TG_OP,
    'timestamp', NOW(),
    'old', CASE WHEN TG_OP = 'DELETE' THEN to_json(OLD) ELSE NULL END,
    'new', CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN to_json(NEW) ELSE NULL END
  );

  -- Send notification
  PERFORM pg_notify(channel, payload::text);

  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Function to update position current prices and PnL
CREATE OR REPLACE FUNCTION update_position_pnl()
RETURNS TRIGGER AS $$
DECLARE
  pos RECORD;
BEGIN
  -- Update positions for this symbol with new market price
  FOR pos IN 
    SELECT p.id, p.side, p.quantity, p.entry_price
    FROM positions p
    WHERE p.symbol_id = NEW.symbol_id
      AND p.closed_at IS NULL
  LOOP
    UPDATE positions
    SET 
      current_price = NEW.close,
      unrealized_pnl = CASE 
        WHEN pos.side = 'long' THEN 
          pos.quantity * (NEW.close - pos.entry_price)
        WHEN pos.side = 'short' THEN 
          pos.quantity * (pos.entry_price - NEW.close)
        ELSE 0
      END,
      updated_at = NOW()
    WHERE id = pos.id;
  END LOOP;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically create performance metrics
CREATE OR REPLACE FUNCTION create_performance_metrics()
RETURNS TRIGGER AS $$
BEGIN
  -- Create metrics for different events
  CASE TG_TABLE_NAME
    WHEN 'bot_executions' THEN
      INSERT INTO performance_metrics (
        entity_type, entity_id, metric_type, metric_value, metadata
      ) VALUES (
        'bot', NEW.bot_id, 'execution_count', 1,
        jsonb_build_object('action', NEW.action, 'signal_strength', NEW.signal_strength)
      );
      
    WHEN 'orders' THEN
      IF NEW.status = 'filled' AND (OLD.status IS NULL OR OLD.status != 'filled') THEN
        INSERT INTO performance_metrics (
          entity_type, entity_id, metric_type, metric_value, metadata
        ) VALUES (
          'account', NEW.account_id, 'trade_executed', NEW.filled_quantity,
          jsonb_build_object('order_type', NEW.order_type, 'side', NEW.side, 'symbol_id', NEW.symbol_id)
        );
      END IF;
      
    WHEN 'positions' THEN
      IF NEW.closed_at IS NOT NULL AND (OLD.closed_at IS NULL) THEN
        INSERT INTO performance_metrics (
          entity_type, entity_id, metric_type, metric_value, metadata
        ) VALUES (
          'account', NEW.account_id, 'realized_pnl', NEW.realized_pnl,
          jsonb_build_object('symbol_id', NEW.symbol_id, 'side', NEW.side, 'duration_minutes', 
            EXTRACT(EPOCH FROM (NEW.closed_at - NEW.opened_at))/60)
        );
      END IF;
      
    WHEN 'training_runs' THEN
      IF NEW.status = 'completed' AND (OLD.status IS NULL OR OLD.status != 'completed') THEN
        INSERT INTO performance_metrics (
          entity_type, entity_id, metric_type, metric_value, metadata
        ) VALUES (
          'model', NEW.model_id, 'training_accuracy', COALESCE(NEW.accuracy, 0),
          jsonb_build_object('loss', NEW.loss, 'epochs', NEW.epoch, 'validation_accuracy', NEW.validation_accuracy)
        );
      END IF;
  END CASE;

  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Function to manage trading bot risk limits
CREATE OR REPLACE FUNCTION check_trading_bot_risk()
RETURNS TRIGGER AS $$
DECLARE
  bot RECORD;
  account_balance DECIMAL(18,8);
  total_exposure DECIMAL(18,8);
  risk_ratio DECIMAL(5,4);
BEGIN
  -- Get bot information
  SELECT * INTO bot FROM trading_bots WHERE id = NEW.bot_id;
  
  IF NOT FOUND THEN
    RETURN NEW;
  END IF;

  -- Get account balance
  SELECT balance INTO account_balance 
  FROM trading_accounts 
  WHERE id = bot.account_id;

  -- Calculate total exposure for this bot
  SELECT COALESCE(SUM(ABS(quantity * current_price)), 0)
  INTO total_exposure
  FROM positions p
  WHERE p.account_id = bot.account_id
    AND p.closed_at IS NULL;

  -- Calculate risk ratio
  risk_ratio := total_exposure / NULLIF(account_balance, 0);

  -- Check if risk limit exceeded
  IF risk_ratio > bot.risk_limit THEN
    -- Create alert
    INSERT INTO alerts (
      user_id, title, message, severity, entity_type, entity_id, metadata
    ) VALUES (
      bot.user_id,
      'Risk Limit Exceeded',
      format('Bot %s has exceeded risk limit. Current exposure: %.2f%%, Limit: %.2f%%', 
        bot.name, risk_ratio * 100, bot.risk_limit * 100),
      'warning',
      'bot',
      bot.id,
      jsonb_build_object('risk_ratio', risk_ratio, 'risk_limit', bot.risk_limit, 'total_exposure', total_exposure)
    );

    -- Optionally pause the bot
    UPDATE trading_bots 
    SET status = 'paused', updated_at = NOW()
    WHERE id = bot.id;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically update account equity
CREATE OR REPLACE FUNCTION update_account_equity()
RETURNS TRIGGER AS $$
DECLARE
  account_id_val UUID;
  total_unrealized DECIMAL(18,8);
BEGIN
  -- Get account ID from the position
  account_id_val := COALESCE(NEW.account_id, OLD.account_id);
  
  -- Calculate total unrealized PnL for the account
  SELECT COALESCE(SUM(unrealized_pnl), 0)
  INTO total_unrealized
  FROM positions
  WHERE account_id = account_id_val
    AND closed_at IS NULL;

  -- Update account equity
  UPDATE trading_accounts
  SET 
    equity = balance + total_unrealized,
    updated_at = NOW()
  WHERE id = account_id_val;

  RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Function to manage model prediction accuracy
CREATE OR REPLACE FUNCTION update_prediction_accuracy()
RETURNS TRIGGER AS $$
DECLARE
  model_accuracy DECIMAL(5,4);
  recent_predictions INTEGER;
BEGIN
  -- Only process when actual_value is updated
  IF NEW.actual_value IS NOT NULL AND (OLD.actual_value IS NULL OR OLD.actual_value != NEW.actual_value) THEN
    -- Calculate prediction error
    NEW.error := ABS(NEW.prediction_value - NEW.actual_value);
    
    -- Update model performance metrics periodically
    SELECT COUNT(*) INTO recent_predictions
    FROM model_predictions
    WHERE model_id = NEW.model_id
      AND actual_value IS NOT NULL
      AND created_at >= NOW() - INTERVAL '24 hours';
    
    -- Update model performance every 10 predictions
    IF recent_predictions % 10 = 0 THEN
      PERFORM update_model_performance(NEW.model_id, 100);
    END IF;
  END IF;

  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply real-time notification triggers
CREATE TRIGGER notify_market_data_changes
  AFTER INSERT OR UPDATE ON market_data
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_bot_execution_changes
  AFTER INSERT ON bot_executions
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_bot_status_changes
  AFTER UPDATE ON trading_bots
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_alert_changes
  AFTER INSERT ON alerts
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_performance_changes
  AFTER INSERT ON performance_metrics
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_training_changes
  AFTER UPDATE ON training_runs
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_order_changes
  AFTER INSERT OR UPDATE ON orders
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

CREATE TRIGGER notify_position_changes
  AFTER INSERT OR UPDATE OR DELETE ON positions
  FOR EACH ROW EXECUTE FUNCTION notify_realtime_event();

-- Apply business logic triggers
CREATE TRIGGER update_position_pnl_trigger
  AFTER INSERT OR UPDATE ON market_data
  FOR EACH ROW EXECUTE FUNCTION update_position_pnl();

CREATE TRIGGER create_performance_metrics_trigger
  AFTER INSERT OR UPDATE ON bot_executions
  FOR EACH ROW EXECUTE FUNCTION create_performance_metrics();

CREATE TRIGGER create_order_metrics_trigger
  AFTER UPDATE ON orders
  FOR EACH ROW EXECUTE FUNCTION create_performance_metrics();

CREATE TRIGGER create_position_metrics_trigger
  AFTER UPDATE ON positions
  FOR EACH ROW EXECUTE FUNCTION create_performance_metrics();

CREATE TRIGGER create_training_metrics_trigger
  AFTER UPDATE ON training_runs
  FOR EACH ROW EXECUTE FUNCTION create_performance_metrics();

CREATE TRIGGER check_bot_risk_trigger
  AFTER INSERT ON bot_executions
  FOR EACH ROW EXECUTE FUNCTION check_trading_bot_risk();

CREATE TRIGGER update_account_equity_trigger
  AFTER INSERT OR UPDATE OR DELETE ON positions
  FOR EACH ROW EXECUTE FUNCTION update_account_equity();

CREATE TRIGGER update_prediction_accuracy_trigger
  BEFORE UPDATE ON model_predictions
  FOR EACH ROW EXECUTE FUNCTION update_prediction_accuracy();

-- Create a trigger to automatically clean up old data
CREATE OR REPLACE FUNCTION schedule_data_cleanup()
RETURNS TRIGGER AS $$
BEGIN
  -- Schedule cleanup every 1000 inserts to performance_metrics
  IF (SELECT COUNT(*) FROM performance_metrics) % 1000 = 0 THEN
    PERFORM cleanup_old_data();
  END IF;
  
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER schedule_cleanup_trigger
  AFTER INSERT ON performance_metrics
  FOR EACH ROW EXECUTE FUNCTION schedule_data_cleanup();