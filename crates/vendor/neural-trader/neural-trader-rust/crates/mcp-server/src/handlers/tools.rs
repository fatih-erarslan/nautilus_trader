//! Tool handler - routes tool calls to implementations

use neural_trader_mcp_protocol::{JsonRpcRequest, ProtocolError};
use serde_json::Value;
use crate::tools::{trading, neural, neural_extended, brokers, sports, prediction, news, system, risk, account, config};

/// Handle tool call requests
pub async fn handle_tool_call(request: &JsonRpcRequest) -> Result<Value, ProtocolError> {
    let method = &request.method;
    let params = request.params.clone().unwrap_or(Value::Null);

    match method.as_str() {
        // Core trading tools (Priority 0)
        "ping" => Ok(trading::ping().await),
        "list_strategies" => Ok(trading::list_strategies().await),
        "get_strategy_info" => Ok(trading::get_strategy_info(params).await),
        "quick_analysis" => Ok(trading::quick_analysis(params).await),
        "simulate_trade" => Ok(trading::simulate_trade(params).await),
        "get_portfolio_status" => Ok(trading::get_portfolio_status(params).await),
        "execute_trade" => Ok(trading::execute_trade(params).await),
        "run_backtest" => Ok(trading::run_backtest(params).await),
        "optimize_strategy" => Ok(trading::optimize_strategy(params).await),
        "risk_analysis" => Ok(trading::risk_analysis(params).await),

        // Neural & Advanced tools (Priority 1)
        "neural_forecast" => Ok(neural::neural_forecast(params).await),
        "neural_train" => Ok(neural::neural_train(params).await),
        "neural_evaluate" => Ok(neural::neural_evaluate(params).await),
        "neural_backtest" => Ok(neural::neural_backtest(params).await),
        "neural_model_status" => Ok(neural::neural_model_status(params).await),
        "neural_optimize" => Ok(neural::neural_optimize(params).await),
        "correlation_analysis" => Ok(neural::correlation_analysis(params).await),
        "performance_report" => Ok(neural::performance_report(params).await),

        // News analysis tools
        "analyze_news" => Ok(news::analyze_news(params).await),
        "get_news_sentiment" => Ok(news::get_news_sentiment(params).await),

        // Multi-broker tools (Priority 2)
        "list_brokers" => Ok(brokers::list_brokers().await),
        "connect_broker" => Ok(brokers::connect_broker(params).await),
        "get_broker_status" => Ok(brokers::get_broker_status(params).await),
        "execute_broker_order" => Ok(brokers::execute_broker_order(params).await),
        "get_broker_positions" => Ok(brokers::get_broker_positions(params).await),
        "get_broker_orders" => Ok(brokers::get_broker_orders(params).await),

        // Sports betting tools (Priority 3)
        "get_sports_events" => Ok(sports::get_sports_events(params).await),
        "get_sports_odds" => Ok(sports::get_sports_odds(params).await),
        "find_sports_arbitrage" => Ok(sports::find_sports_arbitrage(params).await),
        "analyze_betting_market_depth" => Ok(sports::analyze_betting_market_depth(params).await),
        "calculate_kelly_criterion" => Ok(sports::calculate_kelly_criterion(params).await),
        "simulate_betting_strategy" => Ok(sports::simulate_betting_strategy(params).await),
        "get_betting_portfolio_status" => Ok(sports::get_betting_portfolio_status(params).await),
        "execute_sports_bet" => Ok(sports::execute_sports_bet(params).await),
        "get_sports_betting_performance" => Ok(sports::get_sports_betting_performance(params).await),

        // Prediction markets tools (Priority 4)
        "get_prediction_markets" => Ok(prediction::get_prediction_markets(params).await),
        "analyze_market_sentiment" => Ok(prediction::analyze_market_sentiment(params).await),
        "get_market_orderbook" => Ok(prediction::get_market_orderbook(params).await),
        "place_prediction_order" => Ok(prediction::place_prediction_order(params).await),
        "get_prediction_positions" => Ok(prediction::get_prediction_positions().await),
        "calculate_expected_value" => Ok(prediction::calculate_expected_value(params).await),

        // System monitoring tools
        "run_benchmark" => Ok(system::run_benchmark(params).await),
        "get_system_metrics" => Ok(system::get_system_metrics(params).await),
        "monitor_strategy_health" => Ok(system::monitor_strategy_health(params).await),
        "get_execution_analytics" => Ok(system::get_execution_analytics(params).await),

        // **NEW: 20 Critical MCP Tools (Priority 0 - High Value)**

        // Account & Trading Operations (8 tools)
        "get_account_info" => Ok(account::get_account_info(params).await),
        "get_positions" => Ok(account::get_positions(params).await),
        "get_orders" => Ok(account::get_orders(params).await),
        "cancel_order" => Ok(account::cancel_order(params).await),
        "modify_order" => Ok(account::modify_order(params).await),
        "get_fills" => Ok(account::get_fills(params).await),
        "get_portfolio_value" => Ok(account::get_portfolio_value(params).await),
        "get_market_status" => Ok(account::get_market_status(params).await),

        // Neural Network Training & Management (5 tools)
        "neural_train_model" => Ok(neural_extended::neural_train_model(params).await),
        "neural_get_status" => Ok(neural_extended::neural_get_status(params).await),
        "neural_stop_training" => Ok(neural_extended::neural_stop_training(params).await),
        "neural_save_model" => Ok(neural_extended::neural_save_model(params).await),
        "neural_load_model" => Ok(neural_extended::neural_load_model(params).await),

        // Risk Management (4 tools)
        "calculate_position_size" => Ok(risk::calculate_position_size(params).await),
        "check_risk_limits" => Ok(risk::check_risk_limits(params).await),
        "get_portfolio_risk" => Ok(risk::get_portfolio_risk(params).await),
        "stress_test_portfolio" => Ok(risk::stress_test_portfolio(params).await),

        // System Configuration (3 tools)
        "get_config" => Ok(config::get_config(params).await),
        "set_config" => Ok(config::set_config(params).await),
        "health_check" => Ok(config::health_check(params).await),

        _ => Err(ProtocolError::MethodNotFound(format!("Unknown tool: {}", method))),
    }
}

/// Handle list tools request
pub async fn handle_list_tools() -> Value {
    serde_json::json!({
        "tools": [
            // Core Trading Tools (Priority 0)
            {"name": "ping", "description": "Verify server is responsive", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "list_strategies", "description": "List all available trading strategies", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "get_strategy_info", "description": "Get detailed information about a strategy", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}}, "required": ["strategy"]}},
            {"name": "quick_analysis", "description": "Quick market analysis for a symbol", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "use_gpu": {"type": "boolean"}}, "required": ["symbol"]}},
            {"name": "simulate_trade", "description": "Simulate a trade operation", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}, "symbol": {"type": "string"}, "action": {"type": "string"}}, "required": ["strategy", "symbol", "action"]}},
            {"name": "get_portfolio_status", "description": "Get current portfolio status", "inputSchema": {"type": "object", "properties": {"include_analytics": {"type": "boolean"}}}},
            {"name": "execute_trade", "description": "Execute a live trade", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}, "symbol": {"type": "string"}, "action": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["strategy", "symbol", "action", "quantity"]}},
            {"name": "run_backtest", "description": "Run historical backtest", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}, "symbol": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["strategy", "symbol", "start_date", "end_date"]}},
            {"name": "optimize_strategy", "description": "Optimize strategy parameters", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}, "symbol": {"type": "string"}, "parameter_ranges": {"type": "object"}}, "required": ["strategy", "symbol", "parameter_ranges"]}},
            {"name": "risk_analysis", "description": "Comprehensive portfolio risk analysis", "inputSchema": {"type": "object", "properties": {"portfolio": {"type": "array"}, "use_gpu": {"type": "boolean"}}, "required": ["portfolio"]}},

            // Neural & Advanced Tools (Priority 1)
            {"name": "neural_forecast", "description": "Generate neural network forecasts", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "horizon": {"type": "integer"}}, "required": ["symbol", "horizon"]}},
            {"name": "neural_train", "description": "Train a neural forecasting model", "inputSchema": {"type": "object", "properties": {"model_type": {"type": "string"}, "data_path": {"type": "string"}}, "required": ["model_type", "data_path"]}},
            {"name": "neural_evaluate", "description": "Evaluate a trained neural model", "inputSchema": {"type": "object", "properties": {"model_id": {"type": "string"}, "test_data": {"type": "string"}}, "required": ["model_id", "test_data"]}},
            {"name": "neural_backtest", "description": "Run historical backtest with neural model", "inputSchema": {"type": "object", "properties": {"model_id": {"type": "string"}, "start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["model_id", "start_date", "end_date"]}},
            {"name": "neural_model_status", "description": "Get neural model status", "inputSchema": {"type": "object", "properties": {"model_id": {"type": "string"}}}},
            {"name": "neural_optimize", "description": "Optimize neural model hyperparameters", "inputSchema": {"type": "object", "properties": {"model_id": {"type": "string"}, "parameter_ranges": {"type": "object"}}, "required": ["model_id", "parameter_ranges"]}},
            {"name": "correlation_analysis", "description": "Analyze asset correlations", "inputSchema": {"type": "object", "properties": {"symbols": {"type": "array"}}, "required": ["symbols"]}},
            {"name": "performance_report", "description": "Generate performance report", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}}, "required": ["strategy"]}},

            // News Analysis Tools
            {"name": "analyze_news", "description": "AI sentiment analysis of market news", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
            {"name": "get_news_sentiment", "description": "Get real-time news sentiment", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},

            // Multi-Broker Tools (Priority 2)
            {"name": "list_brokers", "description": "List supported brokers", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "connect_broker", "description": "Connect to a broker", "inputSchema": {"type": "object", "properties": {"broker_id": {"type": "string"}}, "required": ["broker_id"]}},
            {"name": "get_broker_status", "description": "Get broker account status", "inputSchema": {"type": "object", "properties": {"broker_id": {"type": "string"}}, "required": ["broker_id"]}},
            {"name": "execute_broker_order", "description": "Execute multi-broker order", "inputSchema": {"type": "object", "properties": {"broker_id": {"type": "string"}, "symbol": {"type": "string"}, "side": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["broker_id", "symbol", "side", "quantity"]}},
            {"name": "get_broker_positions", "description": "Get broker positions", "inputSchema": {"type": "object", "properties": {"broker_id": {"type": "string"}}, "required": ["broker_id"]}},
            {"name": "get_broker_orders", "description": "Get broker order history", "inputSchema": {"type": "object", "properties": {"broker_id": {"type": "string"}}, "required": ["broker_id"]}},

            // Sports Betting Tools (Priority 3)
            {"name": "get_sports_events", "description": "Get upcoming sports events", "inputSchema": {"type": "object", "properties": {"sport": {"type": "string"}}, "required": ["sport"]}},
            {"name": "get_sports_odds", "description": "Get sports betting odds", "inputSchema": {"type": "object", "properties": {"sport": {"type": "string"}}, "required": ["sport"]}},
            {"name": "find_sports_arbitrage", "description": "Find sports arbitrage opportunities", "inputSchema": {"type": "object", "properties": {"sport": {"type": "string"}}, "required": ["sport"]}},
            {"name": "analyze_betting_market_depth", "description": "Analyze betting market depth", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}, "sport": {"type": "string"}}, "required": ["market_id", "sport"]}},
            {"name": "calculate_kelly_criterion", "description": "Calculate Kelly Criterion bet sizing", "inputSchema": {"type": "object", "properties": {"probability": {"type": "number"}, "odds": {"type": "number"}, "bankroll": {"type": "number"}}, "required": ["probability", "odds", "bankroll"]}},
            {"name": "simulate_betting_strategy", "description": "Simulate betting strategy", "inputSchema": {"type": "object", "properties": {"strategy_config": {"type": "object"}}, "required": ["strategy_config"]}},
            {"name": "get_betting_portfolio_status", "description": "Get betting portfolio status", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "execute_sports_bet", "description": "Execute sports bet", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}, "selection": {"type": "string"}, "stake": {"type": "number"}, "odds": {"type": "number"}}, "required": ["market_id", "selection", "stake", "odds"]}},
            {"name": "get_sports_betting_performance", "description": "Get sports betting performance", "inputSchema": {"type": "object", "properties": {}}},

            // Prediction Markets Tools (Priority 4)
            {"name": "get_prediction_markets", "description": "Get available prediction markets", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "analyze_market_sentiment", "description": "Analyze prediction market sentiment", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}}, "required": ["market_id"]}},
            {"name": "get_market_orderbook", "description": "Get market orderbook", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}}, "required": ["market_id"]}},
            {"name": "place_prediction_order", "description": "Place prediction market order", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}, "outcome": {"type": "string"}, "side": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["market_id", "outcome", "side", "quantity"]}},
            {"name": "get_prediction_positions", "description": "Get prediction positions", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "calculate_expected_value", "description": "Calculate expected value", "inputSchema": {"type": "object", "properties": {"market_id": {"type": "string"}, "investment_amount": {"type": "number"}}, "required": ["market_id", "investment_amount"]}},

            // System Monitoring Tools
            {"name": "run_benchmark", "description": "Run performance benchmarks", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}}, "required": ["strategy"]}},
            {"name": "get_system_metrics", "description": "Get system metrics and health", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "monitor_strategy_health", "description": "Monitor strategy health", "inputSchema": {"type": "object", "properties": {"strategy": {"type": "string"}}, "required": ["strategy"]}},
            {"name": "get_execution_analytics", "description": "Get execution analytics", "inputSchema": {"type": "object", "properties": {}}},

            // **NEW: 20 Critical MCP Tools (Priority 0 - High Value)**

            // Account & Trading Operations (8 tools)
            {"name": "get_account_info", "description": "Get account balance, buying power, and status", "inputSchema": {"type": "object", "properties": {"broker": {"type": "string"}, "include_positions": {"type": "boolean"}}}},
            {"name": "get_positions", "description": "Get current open positions", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "include_closed": {"type": "boolean"}}}},
            {"name": "get_orders", "description": "Get order history and status", "inputSchema": {"type": "object", "properties": {"status": {"type": "string"}, "limit": {"type": "integer"}, "symbol": {"type": "string"}}}},
            {"name": "cancel_order", "description": "Cancel a pending order", "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
            {"name": "modify_order", "description": "Modify an existing order", "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}, "quantity": {"type": "number"}, "limit_price": {"type": "number"}}, "required": ["order_id"]}},
            {"name": "get_fills", "description": "Get trade execution fills", "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}, "limit": {"type": "integer"}}}},
            {"name": "get_portfolio_value", "description": "Get total portfolio value with breakdown", "inputSchema": {"type": "object", "properties": {"include_history": {"type": "boolean"}, "include_breakdown": {"type": "boolean"}}}},
            {"name": "get_market_status", "description": "Get market hours and trading status", "inputSchema": {"type": "object", "properties": {"market": {"type": "string"}}}},

            // Neural Network Training & Management (5 tools)
            {"name": "neural_train_model", "description": "Start neural network training", "inputSchema": {"type": "object", "properties": {"model_type": {"type": "string"}, "dataset": {"type": "string"}, "epochs": {"type": "integer"}, "batch_size": {"type": "integer"}, "learning_rate": {"type": "number"}}, "required": ["model_type"]}},
            {"name": "neural_get_status", "description": "Get training status and metrics", "inputSchema": {"type": "object", "properties": {"training_id": {"type": "string"}}}},
            {"name": "neural_stop_training", "description": "Stop ongoing training", "inputSchema": {"type": "object", "properties": {"training_id": {"type": "string"}, "save_checkpoint": {"type": "boolean"}}}},
            {"name": "neural_save_model", "description": "Save model checkpoint", "inputSchema": {"type": "object", "properties": {"model_id": {"type": "string"}, "save_path": {"type": "string"}, "include_optimizer": {"type": "boolean"}}, "required": ["model_id"]}},
            {"name": "neural_load_model", "description": "Load saved model checkpoint", "inputSchema": {"type": "object", "properties": {"model_path": {"type": "string"}, "load_optimizer": {"type": "boolean"}, "device": {"type": "string"}}, "required": ["model_path"]}},

            // Risk Management (4 tools)
            {"name": "calculate_position_size", "description": "Calculate optimal position size using Kelly Criterion", "inputSchema": {"type": "object", "properties": {"bankroll": {"type": "number"}, "win_probability": {"type": "number"}, "win_loss_ratio": {"type": "number"}, "risk_fraction": {"type": "number"}}, "required": ["bankroll", "win_probability", "win_loss_ratio"]}},
            {"name": "check_risk_limits", "description": "Check if trade violates risk limits", "inputSchema": {"type": "object", "properties": {"symbol": {"type": "string"}, "quantity": {"type": "number"}, "price": {"type": "number"}, "side": {"type": "string"}, "portfolio_value": {"type": "number"}}, "required": ["symbol", "quantity", "price", "side"]}},
            {"name": "get_portfolio_risk", "description": "Get comprehensive portfolio risk metrics (VaR, CVaR)", "inputSchema": {"type": "object", "properties": {"confidence_level": {"type": "number"}, "time_horizon_days": {"type": "integer"}, "use_monte_carlo": {"type": "boolean"}, "use_gpu": {"type": "boolean"}}}},
            {"name": "stress_test_portfolio", "description": "Run stress test scenarios on portfolio", "inputSchema": {"type": "object", "properties": {"scenarios": {"type": "array", "items": {"type": "string"}}, "portfolio_value": {"type": "number"}, "use_gpu": {"type": "boolean"}}}},

            // System Configuration (3 tools)
            {"name": "get_config", "description": "Get current system configuration", "inputSchema": {"type": "object", "properties": {"section": {"type": "string"}}}},
            {"name": "set_config", "description": "Update system configuration", "inputSchema": {"type": "object", "properties": {"section": {"type": "string"}, "updates": {"type": "object"}}, "required": ["section", "updates"]}},
            {"name": "health_check", "description": "Comprehensive system health check", "inputSchema": {"type": "object", "properties": {"detailed": {"type": "boolean"}}}}
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_mcp_protocol::RequestId;

    #[tokio::test]
    async fn test_handle_ping() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "ping".to_string(),
            params: None,
            id: Some(RequestId::String("test-1".to_string())),
        };

        let result = handle_tool_call(&request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_handle_unknown_tool() {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            method: "unknown_tool".to_string(),
            params: None,
            id: Some(RequestId::String("test-2".to_string())),
        };

        let result = handle_tool_call(&request).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_list_tools() {
        let result = handle_list_tools().await;
        assert!(result["tools"].is_array());
        assert!(result["tools"].as_array().unwrap().len() >= 2);
    }
}
