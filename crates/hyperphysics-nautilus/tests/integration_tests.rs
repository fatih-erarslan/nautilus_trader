//! Integration tests for HyperPhysics-Nautilus bridge.
//!
//! These tests validate the complete integration between HyperPhysics
//! physics-based trading pipeline and Nautilus Trader's event-driven architecture.

use hyperphysics_nautilus::{
    adapter::{NautilusDataAdapter, NautilusExecBridge},
    backtest::{BacktestConfig, BacktestRunner, DataLoader, SlippageModel},
    config::IntegrationConfig,
    strategy::HyperPhysicsStrategy,
    types::{
        HyperPhysicsOrderCommand, NautilusBar, NautilusQuoteTick, NautilusTradeTick,
        OrderSide, OrderType, TimeInForce,
    },
};

/// Test basic type conversions from Nautilus to HyperPhysics formats.
mod type_conversion_tests {
    use super::*;

    #[test]
    fn test_quote_tick_to_market_snapshot() {
        let quote = NautilusQuoteTick {
            instrument_id: 12345,
            bid_price: 5000000, // $50,000.00 with precision 2
            ask_price: 5000100, // $50,001.00 with precision 2
            bid_size: 100,
            ask_size: 150,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let snapshot = quote.to_market_snapshot();

        assert!((snapshot.bid - 50000.0).abs() < 0.01);
        assert!((snapshot.ask - 50001.0).abs() < 0.01);
        assert!((snapshot.spread - 1.0).abs() < 0.01);
        assert!((snapshot.mid - 50000.5).abs() < 0.01);
    }

    #[test]
    fn test_quote_tick_to_market_tick() {
        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 50,
            ask_size: 50,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let tick = quote.to_market_tick().expect("Should convert to tick");

        assert!((tick.price - 100.05).abs() < 0.01); // Mid price
        assert!(tick.volume > 0.0);
        assert_eq!(tick.timestamp, 1700000000_000_000_000);
    }

    #[test]
    fn test_bar_to_market_feed() {
        let bar = NautilusBar {
            instrument_id: 1,
            open: 10000,
            high: 10200,
            low: 9900,
            close: 10100,
            volume: 1000,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let feed = bar.to_market_feed();

        assert!((feed.snapshot.bid - 101.0).abs() < 0.01); // Close price
        assert!((feed.snapshot.ask - 101.0).abs() < 0.01);
    }

    #[test]
    fn test_precision_scale_boundary() {
        // Test max precision (8 decimal places)
        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 12345678900000000,
            ask_price: 12345678900000001,
            bid_size: 1,
            ask_size: 1,
            price_precision: 8,
            size_precision: 0,
            ts_event: 0,
            ts_init: 0,
        };

        let snapshot = quote.to_market_snapshot();
        assert!(snapshot.bid > 0.0);
        assert!(snapshot.ask > snapshot.bid);
    }
}

/// Test data adapter functionality.
mod data_adapter_tests {
    use super::*;

    #[tokio::test]
    async fn test_data_adapter_creation() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        // Adapter should be created without errors
        assert!(adapter.get_snapshot(0).await.is_none());
    }

    #[tokio::test]
    async fn test_quote_processing() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let feed = adapter.on_quote(&quote).await.expect("Should process quote");

        assert!(feed.snapshot.bid > 0.0);
        assert!(feed.snapshot.ask > feed.snapshot.bid);
        assert_eq!(feed.snapshot.timestamp, 1700000000_000_000_000);
    }

    #[tokio::test]
    async fn test_trade_processing() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        let trade = NautilusTradeTick {
            instrument_id: 1,
            price: 10005,
            size: 50,
            aggressor_side: 1, // Buy
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
            trade_id: 12345,
        };

        let feed = adapter.on_trade(&trade).await.expect("Should process trade");

        assert!(feed.snapshot.last_price > 0.0);
        assert!(feed.snapshot.volume > 0.0);
    }

    #[tokio::test]
    async fn test_bar_processing() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        let bar = NautilusBar {
            instrument_id: 1,
            open: 10000,
            high: 10100,
            low: 9900,
            close: 10050,
            volume: 500,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let feed = adapter.on_bar(&bar).await.expect("Should process bar");

        assert!(!feed.bars.is_empty());
    }

    #[tokio::test]
    async fn test_snapshot_retrieval() {
        let config = IntegrationConfig::default();
        let adapter = NautilusDataAdapter::new(config);

        // Process a quote first
        let quote = NautilusQuoteTick {
            instrument_id: 42,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        adapter.on_quote(&quote).await.unwrap();

        // Should be able to retrieve snapshot
        let snapshot = adapter.get_snapshot(42).await;
        assert!(snapshot.is_some());

        let snap = snapshot.unwrap();
        assert!((snap.bid - 100.0).abs() < 0.01);
    }
}

/// Test execution bridge functionality.
mod exec_bridge_tests {
    use super::*;
    use hyperphysics_hft_ecosystem::core::unified_pipeline::PipelineResult;

    #[tokio::test]
    async fn test_exec_bridge_creation() {
        let config = IntegrationConfig::default();
        let bridge = NautilusExecBridge::new(config);

        let stats = bridge.get_stats().await;
        assert_eq!(stats.signals_processed, 0);
        assert_eq!(stats.orders_generated, 0);
    }

    #[tokio::test]
    async fn test_order_generation_sequence() {
        let config = IntegrationConfig::default();
        let bridge = NautilusExecBridge::new(config);

        // Verify order sequence is unique
        let first_id = bridge.next_order_id();
        let second_id = bridge.next_order_id();
        let third_id = bridge.next_order_id();

        assert!(second_id > first_id);
        assert!(third_id > second_id);
    }

    #[tokio::test]
    async fn test_instrument_setting() {
        let config = IntegrationConfig::default();
        let bridge = NautilusExecBridge::new(config);

        bridge.set_instrument("BTCUSDT.BINANCE").await;

        // Verify instrument is set (would need getter or check via order generation)
    }
}

/// Test strategy lifecycle and event handling.
mod strategy_tests {
    use super::*;

    #[tokio::test]
    async fn test_strategy_creation() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await;

        assert!(strategy.is_ok(), "Strategy should be created successfully");
    }

    #[tokio::test]
    async fn test_strategy_lifecycle() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        // Initial state
        assert!(!strategy.is_running().await);

        // Start
        strategy.start().await.unwrap();
        assert!(strategy.is_running().await);

        // Stop
        strategy.stop().await.unwrap();
        assert!(!strategy.is_running().await);
    }

    #[tokio::test]
    async fn test_strategy_not_running_no_processing() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        // Don't start strategy

        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let result = strategy.on_quote(&quote).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_none()); // No order when not running
    }

    #[tokio::test]
    async fn test_quote_event_processing() {
        let config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..Default::default()
        };
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.set_instrument("TEST.SIM").await;
        strategy.start().await.unwrap();

        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 1700000000_000_000_000,
            ts_init: 1700000000_000_000_000,
        };

        let result = strategy.on_quote(&quote).await;
        assert!(result.is_ok());

        // Check metrics updated
        let metrics = strategy.get_metrics().await;
        assert_eq!(metrics.quotes_processed, 1);
    }

    #[tokio::test]
    async fn test_multiple_quote_processing() {
        let config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..Default::default()
        };
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.set_instrument("TEST.SIM").await;
        strategy.start().await.unwrap();

        // Process multiple quotes
        for i in 0..10 {
            let quote = NautilusQuoteTick {
                instrument_id: 1,
                bid_price: 10000 + i * 10,
                ask_price: 10010 + i * 10,
                bid_size: 100,
                ask_size: 100,
                price_precision: 2,
                size_precision: 0,
                ts_event: 1700000000_000_000_000 + i as u64 * 1000000,
                ts_init: 1700000000_000_000_000 + i as u64 * 1000000,
            };

            strategy.on_quote(&quote).await.unwrap();
        }

        let metrics = strategy.get_metrics().await;
        assert_eq!(metrics.quotes_processed, 10);
    }

    #[tokio::test]
    async fn test_strategy_reset() {
        let config = IntegrationConfig::default();
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.start().await.unwrap();

        // Process some data
        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 0,
            ts_init: 0,
        };
        strategy.on_quote(&quote).await.unwrap();

        // Reset
        strategy.reset().await.unwrap();

        // Verify reset
        assert!(!strategy.is_running().await);
        let metrics = strategy.get_metrics().await;
        assert_eq!(metrics.quotes_processed, 0);
    }
}

/// Test backtest runner functionality.
mod backtest_tests {
    use super::*;
    use hyperphysics_nautilus::backtest::MarketDataEvent;

    #[tokio::test]
    async fn test_backtest_runner_creation() {
        let bt_config = BacktestConfig::default();
        let strategy_config = IntegrationConfig::backtest();

        let runner = BacktestRunner::new(bt_config, strategy_config).await;
        assert!(runner.is_ok());
    }

    #[tokio::test]
    async fn test_empty_backtest() {
        let bt_config = BacktestConfig::default();
        let strategy_config = IntegrationConfig::backtest();

        let mut runner = BacktestRunner::new(bt_config.clone(), strategy_config)
            .await
            .unwrap();

        let results = runner.run(vec![]).await.unwrap();

        assert_eq!(results.total_trades, 0);
        assert!((results.final_equity - bt_config.initial_capital).abs() < 0.01);
        assert!(results.backtest_runtime_secs > 0.0);
    }

    #[tokio::test]
    async fn test_backtest_with_synthetic_data() {
        let bt_config = BacktestConfig {
            initial_capital: 100_000.0,
            commission_rate: 0.001,
            slippage_model: SlippageModel::FixedBps(1.0),
            ..Default::default()
        };
        let strategy_config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..IntegrationConfig::backtest()
        };

        let mut runner = BacktestRunner::new(bt_config, strategy_config)
            .await
            .unwrap();

        // Generate synthetic data
        let events = DataLoader::generate_synthetic_quotes(
            1,
            100,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        let results = runner.run(events).await.unwrap();

        // Verify results structure
        assert!(results.backtest_runtime_secs > 0.0);
        assert!(results.sharpe_ratio.is_finite());
        assert!(results.max_drawdown >= 0.0);
    }

    #[tokio::test]
    async fn test_backtest_time_filtering() {
        let bt_config = BacktestConfig {
            start_time_ns: 1000000000 + 20 * 1000000,
            end_time_ns: 1000000000 + 80 * 1000000,
            initial_capital: 100_000.0,
            ..Default::default()
        };
        let strategy_config = IntegrationConfig::backtest();

        let mut runner = BacktestRunner::new(bt_config, strategy_config)
            .await
            .unwrap();

        // Generate events outside time range
        let events = DataLoader::generate_synthetic_quotes(
            1,
            100,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        let results = runner.run(events).await.unwrap();

        // Should process only events within time range
        assert!(results.backtest_runtime_secs > 0.0);
    }
}

/// Test data loader functionality.
mod data_loader_tests {
    use super::*;

    #[test]
    fn test_synthetic_quote_generation() {
        let events = DataLoader::generate_synthetic_quotes(
            1,
            1000,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        assert_eq!(events.len(), 1000);

        // Verify ordering
        for window in events.windows(2) {
            assert!(window[0].timestamp() < window[1].timestamp());
        }

        // Verify first event
        if let hyperphysics_nautilus::backtest::MarketDataEvent::Quote(q) = &events[0] {
            assert_eq!(q.instrument_id, 1);
            assert!(q.bid_price > 0);
            assert!(q.ask_price > q.bid_price);
        } else {
            panic!("Expected quote event");
        }
    }

    #[test]
    fn test_time_range_filter() {
        let events = DataLoader::generate_synthetic_quotes(
            1,
            100,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        let start_ns = 1000000000 + 20 * 1000000;
        let end_ns = 1000000000 + 50 * 1000000;

        let filtered = DataLoader::filter_time_range(events, start_ns, end_ns);

        assert!(filtered.len() < 100);
        for e in &filtered {
            let ts = e.timestamp();
            assert!(ts >= start_ns && ts <= end_ns);
        }
    }

    #[test]
    fn test_instrument_filter() {
        let mut events = DataLoader::generate_synthetic_quotes(
            1,
            50,
            100.0,
            0.001,
            1000000000,
            1000000,
        );

        // Add events for another instrument
        let events2 = DataLoader::generate_synthetic_quotes(
            2,
            50,
            200.0,
            0.001,
            1000000000,
            1000000,
        );

        events.extend(events2);
        assert_eq!(events.len(), 100);

        let filtered = DataLoader::filter_instrument(events, 1);

        assert_eq!(filtered.len(), 50);
        for e in &filtered {
            assert_eq!(e.instrument_id(), 1);
        }
    }

    #[test]
    fn test_sort_by_time() {
        use hyperphysics_nautilus::backtest::MarketDataEvent;

        // Create events with unsorted timestamps
        let mut events: Vec<MarketDataEvent> = (0..10)
            .rev()
            .map(|i| {
                MarketDataEvent::Quote(NautilusQuoteTick {
                    instrument_id: 1,
                    bid_price: 10000,
                    ask_price: 10010,
                    bid_size: 100,
                    ask_size: 100,
                    price_precision: 2,
                    size_precision: 0,
                    ts_event: (i as u64) * 1000000,
                    ts_init: (i as u64) * 1000000,
                })
            })
            .collect();

        DataLoader::sort_by_time(&mut events);

        // Verify sorted
        for window in events.windows(2) {
            assert!(window[0].timestamp() <= window[1].timestamp());
        }
    }
}

/// Test configuration handling.
mod config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IntegrationConfig::default();

        assert!(!config.strategy_id.is_empty());
        assert!(config.min_confidence_threshold >= 0.0);
        assert!(config.min_confidence_threshold <= 1.0);
    }

    #[test]
    fn test_backtest_config() {
        let config = IntegrationConfig::backtest();

        assert!(!config.enable_consensus);
        assert!((config.min_confidence_threshold - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_backtest_default_config() {
        let config = BacktestConfig::default();

        assert!(config.initial_capital > 0.0);
        assert!(config.commission_rate >= 0.0);
        assert!(config.commission_rate < 1.0);
    }

    #[test]
    fn test_slippage_models() {
        // Test None slippage
        let _none = SlippageModel::None;

        // Test Fixed BPS
        let _fixed = SlippageModel::FixedBps(5.0);

        // Test Volatility based
        let _vol = SlippageModel::VolatilityBased { multiplier: 2.0 };
    }
}

/// Test order command generation.
mod order_command_tests {
    use super::*;

    #[test]
    fn test_order_command_structure() {
        let cmd = HyperPhysicsOrderCommand {
            client_order_id: "TEST-001".to_string(),
            instrument_id: "BTCUSDT.BINANCE".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 1.0,
            price: None,
            time_in_force: TimeInForce::IOC,
            reduce_only: false,
            post_only: false,
            hp_confidence: 0.85,
            hp_algorithm: "PhysicsSimulation".to_string(),
            hp_latency_us: 100,
            hp_consensus_term: 42,
        };

        assert_eq!(cmd.client_order_id, "TEST-001");
        assert_eq!(cmd.instrument_id, "BTCUSDT.BINANCE");
        assert!(matches!(cmd.side, OrderSide::Buy));
        assert!(matches!(cmd.order_type, OrderType::Market));
        assert!(cmd.price.is_none());
        assert!((cmd.hp_confidence - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_order_side_variants() {
        let buy = OrderSide::Buy;
        let sell = OrderSide::Sell;
        let no_side = OrderSide::NoSide;

        assert!(matches!(buy, OrderSide::Buy));
        assert!(matches!(sell, OrderSide::Sell));
        assert!(matches!(no_side, OrderSide::NoSide));
    }

    #[test]
    fn test_order_type_variants() {
        let market = OrderType::Market;
        let limit = OrderType::Limit;
        let stop_market = OrderType::StopMarket;
        let stop_limit = OrderType::StopLimit;

        assert!(matches!(market, OrderType::Market));
        assert!(matches!(limit, OrderType::Limit));
        assert!(matches!(stop_market, OrderType::StopMarket));
        assert!(matches!(stop_limit, OrderType::StopLimit));
    }

    #[test]
    fn test_time_in_force_variants() {
        let gtc = TimeInForce::GTC;
        let ioc = TimeInForce::IOC;
        let fok = TimeInForce::FOK;
        let day = TimeInForce::Day;

        assert!(matches!(gtc, TimeInForce::GTC));
        assert!(matches!(ioc, TimeInForce::IOC));
        assert!(matches!(fok, TimeInForce::FOK));
        assert!(matches!(day, TimeInForce::Day));
    }
}

/// Performance tests.
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_quote_processing_latency() {
        let config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..Default::default()
        };
        let strategy = HyperPhysicsStrategy::new(config).await.unwrap();

        strategy.set_instrument("PERF.TEST").await;
        strategy.start().await.unwrap();

        let quote = NautilusQuoteTick {
            instrument_id: 1,
            bid_price: 10000,
            ask_price: 10010,
            bid_size: 100,
            ask_size: 100,
            price_precision: 2,
            size_precision: 0,
            ts_event: 0,
            ts_init: 0,
        };

        // Warm up
        for _ in 0..10 {
            strategy.on_quote(&quote).await.unwrap();
        }

        // Measure
        let start = Instant::now();
        let iterations = 100;
        for _ in 0..iterations {
            strategy.on_quote(&quote).await.unwrap();
        }
        let elapsed = start.elapsed();

        let avg_latency_us = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "Average quote processing latency: {:.2}μs ({} iterations)",
            avg_latency_us, iterations
        );

        // Strategy metrics should track latency
        let metrics = strategy.get_metrics().await;
        assert!(metrics.avg_signal_latency_us > 0.0);
    }

    #[test]
    fn test_type_conversion_throughput() {
        let start = Instant::now();
        let iterations = 10_000;

        for i in 0..iterations {
            let quote = NautilusQuoteTick {
                instrument_id: 1,
                bid_price: 10000 + i,
                ask_price: 10010 + i,
                bid_size: 100,
                ask_size: 100,
                price_precision: 2,
                size_precision: 0,
                ts_event: i as u64,
                ts_init: i as u64,
            };

            let _snapshot = quote.to_market_snapshot();
        }

        let elapsed = start.elapsed();
        let throughput = iterations as f64 / elapsed.as_secs_f64();

        println!(
            "Type conversion throughput: {:.0} conversions/sec",
            throughput
        );

        // Should achieve at least 100,000 conversions per second
        assert!(throughput > 100_000.0);
    }

    #[test]
    fn test_synthetic_data_generation_performance() {
        let start = Instant::now();
        let events = DataLoader::generate_synthetic_quotes(
            1,
            100_000,
            100.0,
            0.001,
            1000000000,
            1000000,
        );
        let elapsed = start.elapsed();

        let throughput = events.len() as f64 / elapsed.as_secs_f64();

        println!(
            "Synthetic data generation: {} events in {:?} ({:.0} events/sec)",
            events.len(),
            elapsed,
            throughput
        );

        assert_eq!(events.len(), 100_000);
        // Should generate at least 100,000 events per second
        assert!(throughput > 100_000.0);
    }
}

/// Integration scenario tests.
mod scenario_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_backtest_scenario() {
        // Configure backtest
        let bt_config = BacktestConfig {
            initial_capital: 100_000.0,
            commission_rate: 0.0005,
            slippage_model: SlippageModel::FixedBps(0.5),
            start_time_ns: 0,
            end_time_ns: u64::MAX,
            verbose: false,
        };

        let strategy_config = IntegrationConfig {
            strategy_id: "SCENARIO_TEST".to_string(),
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..IntegrationConfig::backtest()
        };

        // Create runner
        let mut runner = BacktestRunner::new(bt_config.clone(), strategy_config)
            .await
            .expect("Should create runner");

        // Generate market data
        let events = DataLoader::generate_synthetic_quotes(
            1,
            5000,
            100.0,
            0.002,
            1000000000,
            100000, // 100μs intervals
        );

        // Run backtest
        let results = runner.run(events).await.expect("Should complete backtest");

        // Validate results
        assert!(results.backtest_runtime_secs > 0.0);
        assert!(results.final_equity > 0.0);
        assert!(results.sharpe_ratio.is_finite());
        assert!(results.max_drawdown >= 0.0 && results.max_drawdown <= 1.0);
        assert!(results.win_rate >= 0.0 && results.win_rate <= 1.0);

        println!("Backtest Results:");
        println!("  Total Return: {:.2}%", results.total_return * 100.0);
        println!("  Sharpe Ratio: {:.2}", results.sharpe_ratio);
        println!("  Max Drawdown: {:.2}%", results.max_drawdown * 100.0);
        println!("  Total Trades: {}", results.total_trades);
        println!("  Win Rate: {:.2}%", results.win_rate * 100.0);
        println!("  Profit Factor: {:.2}", results.profit_factor);
        println!("  Final Equity: ${:.2}", results.final_equity);
        println!("  Runtime: {:.3}s", results.backtest_runtime_secs);
    }

    #[tokio::test]
    async fn test_multi_instrument_scenario() {
        let config = IntegrationConfig {
            enable_consensus: false,
            min_confidence_threshold: 0.0,
            ..Default::default()
        };
        let adapter = NautilusDataAdapter::new(config);

        // Process quotes for multiple instruments
        for instrument_id in 1..=5 {
            let quote = NautilusQuoteTick {
                instrument_id,
                bid_price: 10000 * instrument_id as i64,
                ask_price: 10010 * instrument_id as i64,
                bid_size: 100,
                ask_size: 100,
                price_precision: 2,
                size_precision: 0,
                ts_event: 1700000000_000_000_000,
                ts_init: 1700000000_000_000_000,
            };

            adapter.on_quote(&quote).await.unwrap();
        }

        // Verify each instrument has its own state
        for instrument_id in 1..=5 {
            let snapshot = adapter.get_snapshot(instrument_id).await;
            assert!(snapshot.is_some());

            let snap = snapshot.unwrap();
            let expected_bid = (10000 * instrument_id as i64) as f64 / 100.0;
            assert!((snap.bid - expected_bid).abs() < 0.01);
        }
    }
}
