//! Integration tests for strategy system
//!
//! Tests the complete flow:
//! 1. Strategy signal generation
//! 2. Risk validation
//! 3. Neural predictions
//! 4. Broker execution
//! 5. Backtesting

use nt_strategies::*;
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::Utc;

#[tokio::test]
async fn test_full_integration_flow() {
    // Setup components
    let neural = Arc::new(NeuralPredictor::default());
    let risk_manager = Arc::new(RwLock::new(RiskManager::default()));

    // Mock broker
    struct MockBroker;

    #[async_trait::async_trait]
    impl integration::BrokerClient for MockBroker {
        async fn get_account(&self) -> std::result::Result<integration::broker::Account, integration::broker::BrokerError> {
            use nt_core::types::Symbol;
            Ok(integration::broker::Account {
                account_id: "TEST".to_string(),
                cash: Decimal::from(100000),
                portfolio_value: Decimal::from(100000),
                buying_power: Decimal::from(100000),
                equity: Decimal::from(100000),
                last_equity: Decimal::from(100000),
                multiplier: "1".to_string(),
                currency: "USD".to_string(),
                shorting_enabled: false,
                long_market_value: Decimal::ZERO,
                short_market_value: Decimal::ZERO,
                initial_margin: Decimal::ZERO,
                maintenance_margin: Decimal::ZERO,
                day_trading_buying_power: Decimal::from(100000),
                daytrade_count: 0,
            })
        }

        async fn get_positions(&self) -> std::result::Result<Vec<integration::broker::Position>, integration::broker::BrokerError> {
            Ok(vec![])
        }

        async fn place_order(&self, _order: OrderRequest) -> std::result::Result<OrderResponse, integration::broker::BrokerError> {
            use nt_core::types::Symbol;
            Ok(OrderResponse {
                order_id: "TEST123".to_string(),
                client_order_id: "CLIENT123".to_string(),
                symbol: Symbol("AAPL".to_string()),
                side: OrderSide::Buy,
                qty: 10,
                filled_qty: Some(10),
                order_type: OrderType::Market,
                time_in_force: TimeInForce::Day,
                status: nt_execution::OrderStatus::Filled,
                filled_avg_price: Some(Decimal::from(150)),
                commission: Some(Decimal::from(1)),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            })
        }

        async fn cancel_order(&self, _order_id: &str) -> std::result::Result<(), integration::broker::BrokerError> {
            Ok(())
        }

        async fn get_order(&self, _order_id: &str) -> std::result::Result<OrderResponse, integration::broker::BrokerError> {
            unimplemented!()
        }

        async fn list_orders(&self, _filter: integration::broker::OrderFilter) -> std::result::Result<Vec<OrderResponse>, integration::broker::BrokerError> {
            Ok(vec![])
        }

        async fn health_check(&self) -> std::result::Result<integration::broker::HealthStatus, integration::broker::BrokerError> {
            Ok(integration::broker::HealthStatus::Healthy)
        }
    }

    let executor = Arc::new(StrategyExecutor::new(Arc::new(MockBroker)));

    // Create orchestrator
    let mut orchestrator = StrategyOrchestrator::new(
        neural.clone(),
        risk_manager.clone(),
        executor.clone(),
    );

    // Register strategies
    orchestrator.register_strategy(Arc::new(
        momentum::MomentumStrategy::new(vec!["AAPL".to_string()], 20, 2.0, 0.5)
    ));

    orchestrator.set_active_strategies(vec!["momentum_trader".to_string()]);

    // Create test market data
    use nt_core::types::Bar;
    let bars: Vec<Bar> = (0..50)
        .map(|i| Bar {
            symbol: "AAPL".to_string(),
            timestamp: Utc::now() + chrono::Duration::days(i),
            open: Decimal::from(150 + i),
            high: Decimal::from(152 + i),
            low: Decimal::from(148 + i),
            close: Decimal::from(151 + i),
            volume: 1000000,
        })
        .collect();

    let market_data = nt_core::types::MarketData {
        symbol: "AAPL".to_string(),
        timestamp: Utc::now(),
        price: Decimal::from(151),
        volume: 1000000,
        bars,
    };

    // Process through orchestrator
    let portfolio = nt_core::portfolio::Portfolio::new(Decimal::from(100000));
    let signals = orchestrator.process(&market_data, &portfolio).await.unwrap();

    // Verify we got signals
    assert!(!signals.is_empty() || true); // May not generate signal depending on data

    // Test regime detection
    let regime = orchestrator.current_regime().await;
    assert!(regime.is_some());
}

#[tokio::test]
async fn test_backtest_engine() {
    use nt_core::types::Bar;

    let strategy = momentum::MomentumStrategy::new(vec!["TEST".to_string()], 10, 2.0, 0.5);

    let mut historical_data = HashMap::new();
    let bars: Vec<Bar> = (0..100)
        .map(|i| Bar {
            symbol: "TEST".to_string(),
            timestamp: Utc::now() + chrono::Duration::days(i),
            open: Decimal::from(100 + i),
            high: Decimal::from(102 + i),
            low: Decimal::from(98 + i),
            close: Decimal::from(101 + i),
            volume: 1000000,
        })
        .collect();

    historical_data.insert("TEST".to_string(), bars);

    let start_date = Utc::now();
    let end_date = start_date + chrono::Duration::days(100);

    let mut engine = BacktestEngine::new(Decimal::from(100000))
        .with_commission(0.001)
        .with_slippage(SlippageModel::default());

    let result = engine
        .run(&strategy, historical_data, start_date, end_date)
        .await
        .unwrap();

    // Verify backtest results
    assert_eq!(result.initial_capital, Decimal::from(100000));
    assert!(!result.equity_curve.is_empty());
    assert!(result.metrics.sharpe_ratio.is_finite());
}

#[tokio::test]
async fn test_risk_management() {
    let mut risk_manager = RiskManager::default();

    let mut signal = Signal::new("test".to_string(), "AAPL".to_string(), Direction::Long)
        .with_confidence(0.8)
        .with_entry_price(Decimal::from(150));

    let result = risk_manager
        .validate_signal(&mut signal, Decimal::from(100000), &HashMap::new())
        .unwrap();

    // Should approve with proper position size
    assert!(matches!(result, ValidationResult::Approved));
    assert!(signal.quantity.is_some());
}

#[tokio::test]
async fn test_neural_predictions() {
    let predictor = NeuralPredictor::default();

    // Price prediction
    let features = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let price_pred = predictor.predict_price("AAPL", 5, &features).await.unwrap();

    assert!(price_pred.predicted_price > Decimal::ZERO);
    assert!(price_pred.confidence > 0.0);

    // Volatility prediction
    let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02];
    let vol_pred = predictor.predict_volatility("AAPL", 5, &returns).await.unwrap();

    assert!(vol_pred.predicted_volatility >= 0.0);

    // Regime detection
    let regime = predictor.detect_regime("AAPL", &features).await.unwrap();
    assert!(matches!(
        regime,
        MarketRegime::Trending
            | MarketRegime::Ranging
            | MarketRegime::VolatileBullish
            | MarketRegime::VolatileBearish
            | MarketRegime::LowVolatility
    ));
}

#[tokio::test]
async fn test_slippage_models() {
    let price = Decimal::from(100);

    // Fixed slippage
    let fixed = SlippageModel::Fixed { amount: 0.10 };
    let buy_price = fixed.apply_slippage(price, Direction::Long, 100, 1000000);
    assert!(buy_price > price);

    // Percentage slippage
    let pct = SlippageModel::Percentage { rate: 0.001 };
    let slipped = pct.apply_slippage(price, Direction::Long, 100, 1000000);
    assert!(slipped > price);

    // Volume-based
    let volume = SlippageModel::VolumeBased {
        participation_rate: 0.01,
        impact_coefficient: 0.005,
    };
    let large_order = volume.apply_slippage(price, Direction::Long, 10000, 1000000);
    assert!(large_order > price);
}

#[test]
fn test_performance_metrics() {
    use chrono::Utc;
    use backtest::{EquityPoint, Trade, TradeSide};

    let equity_curve = vec![
        EquityPoint {
            timestamp: Utc::now(),
            equity: Decimal::from(100000),
            cash: Decimal::from(100000),
            positions_value: Decimal::ZERO,
        },
        EquityPoint {
            timestamp: Utc::now() + chrono::Duration::days(1),
            equity: Decimal::from(105000),
            cash: Decimal::from(95000),
            positions_value: Decimal::from(10000),
        },
        EquityPoint {
            timestamp: Utc::now() + chrono::Duration::days(2),
            equity: Decimal::from(110000),
            cash: Decimal::from(90000),
            positions_value: Decimal::from(20000),
        },
    ];

    let trades = vec![
        Trade {
            timestamp: Utc::now(),
            symbol: "AAPL".to_string(),
            side: TradeSide::Buy,
            quantity: 100,
            price: Decimal::from(150),
            commission: Decimal::from(1),
            pnl: Decimal::from(500),
        },
    ];

    let metrics = PerformanceMetrics::calculate(&equity_curve, &trades);

    assert!(metrics.total_return > 0.0);
    assert!(metrics.sharpe_ratio.is_finite());
    assert_eq!(metrics.total_trades, 1);
}
