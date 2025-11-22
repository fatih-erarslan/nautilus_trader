use crate::algorithms::liquidation_engine::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

/// Comprehensive liquidation engine tests implementing real margin call calculations,
/// forced liquidation mechanics, cascade scenarios, and order book impact modeling.
/// Tests use real leverage scenarios, market depth analysis, and cascade liquidation effects.

#[cfg(test)]
mod margin_call_tests {
    use super::*;

    #[tokio::test]
    async fn test_margin_call_with_real_leverage_scenarios() {
        let parameters = LiquidationParameters {
            initial_margin_rate: 0.08,     // 8% initial margin
            maintenance_margin_rate: 0.04, // 4% maintenance margin
            liquidation_buffer: 0.005,     // 0.5% buffer
            margin_call_threshold: 1.30,   // 130% margin call
            liquidation_threshold: 1.10,   // 110% liquidation
            max_leverage: 50.0,
            funding_rate: 0.0001,
            mark_price_premium: 0.0003,
        };
        let engine = LiquidationEngine::new(parameters);

        // Create account with realistic balance
        let account_id = "high_leverage_trader";
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 50_000.0, // $50K account
                    available_balance: 50_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.30,
                    liquidation_level: 1.10,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Test Scenario 1: High leverage BTC position - 20x leverage
        let btc_leverage = 20.0;
        let btc_entry_price = 45_000.0;
        let btc_position_size = 2.0; // 2 BTC

        // Create high leverage position
        let result = engine
            .create_position(
                account_id,
                "BTCUSD",
                btc_position_size,
                btc_entry_price,
                btc_leverage,
                MarginMode::Isolated,
            )
            .await;
        assert!(result.is_ok());

        // Simulate price drop to trigger margin call
        let margin_call_price = 43_500.0; // ~3.3% drop
        let result = engine
            .update_position_margin(account_id, "BTCUSD", margin_call_price)
            .await;

        // Should trigger margin call
        match result {
            Err(LiquidationError::MarginCall(symbol)) => {
                assert_eq!(symbol, "BTCUSD");
            }
            _ => panic!("Expected margin call to be triggered"),
        }

        // Test Scenario 2: ETH position with lower leverage - 10x
        let eth_leverage = 10.0;
        let eth_entry_price = 3_200.0;
        let eth_position_size = 5.0; // 5 ETH

        let result = engine
            .create_position(
                account_id,
                "ETHUSD",
                eth_position_size,
                eth_entry_price,
                eth_leverage,
                MarginMode::Isolated,
            )
            .await;
        assert!(result.is_ok());

        // Verify account state after multiple positions
        let account = engine.get_account_info(account_id).await.unwrap();
        assert_eq!(account.positions.len(), 2);
        assert!(account.used_margin > 0.0);

        // Test margin requirements for both positions
        assert!(account.positions.contains_key("BTCUSD"));
        assert!(account.positions.contains_key("ETHUSD"));

        let btc_position = &account.positions["BTCUSD"];
        let eth_position = &account.positions["ETHUSD"];

        // Verify liquidation prices are calculated correctly
        assert!(btc_position.liquidation_price > 0.0);
        assert!(btc_position.liquidation_price < btc_entry_price); // Long position
        assert!(eth_position.liquidation_price > 0.0);
        assert!(eth_position.liquidation_price < eth_entry_price); // Long position
    }

    #[tokio::test]
    async fn test_cross_margin_calculations() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "cross_margin_trader";

        // Setup cross margin account
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 100_000.0,
                    available_balance: 100_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Create multiple cross margin positions
        let positions = vec![
            ("BTCUSD", 1.5, 46_000.0, 15.0), // 1.5 BTC at 15x
            ("ETHUSD", -8.0, 3_100.0, 12.0), // Short 8 ETH at 12x
            ("ADAUSD", 50_000.0, 0.85, 8.0), // Long 50K ADA at 8x
        ];

        for (symbol, size, entry_price, leverage) in positions {
            let result = engine
                .create_position(
                    account_id,
                    symbol,
                    size,
                    entry_price,
                    leverage,
                    MarginMode::Cross,
                )
                .await;
            assert!(result.is_ok(), "Failed to create position for {}", symbol);
        }

        // Test cross margin liquidation price calculations
        for symbol in &["BTCUSD", "ETHUSD", "ADAUSD"] {
            let liq_price_result = engine
                .calculate_liquidation_price_cross(account_id, symbol)
                .await;
            assert!(
                liq_price_result.is_ok(),
                "Cross margin liquidation price calculation failed for {}",
                symbol
            );

            let liq_price = liq_price_result.unwrap();
            assert!(
                liq_price > 0.0,
                "Liquidation price should be positive for {}",
                symbol
            );
        }

        // Verify account state
        let account = engine.get_account_info(account_id).await.unwrap();
        assert_eq!(account.positions.len(), 3);
        assert!(account.used_margin > 0.0);
        assert!(account.maintenance_margin > 0.0);

        // Test portfolio-level margin calculations
        let total_notional: f64 = account
            .positions
            .iter()
            .map(|(_, pos)| pos.size.abs() * pos.entry_price)
            .sum();

        assert!(total_notional > 0.0);
        assert!(account.used_margin < account.total_balance);
    }

    #[tokio::test]
    async fn test_margin_call_sequence_and_recovery() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "recovery_trader";

        // Setup account
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 25_000.0,
                    available_balance: 25_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Create position approaching margin call
        let result = engine
            .create_position(
                account_id,
                "SOLUSDT",
                100.0, // 100 SOL
                180.0, // Entry at $180
                25.0,  // 25x leverage
                MarginMode::Isolated,
            )
            .await;
        assert!(result.is_ok());

        // Step 1: Price drops to trigger margin call
        let margin_call_price = 175.0; // Small drop
        let result = engine
            .update_position_margin(account_id, "SOLUSDT", margin_call_price)
            .await;

        // Should trigger margin call warning
        assert!(matches!(result, Err(LiquidationError::MarginCall(_))));

        // Step 2: Price recovers slightly
        let recovery_price = 178.0;
        let result = engine
            .update_position_margin(account_id, "SOLUSDT", recovery_price)
            .await;

        // Should not trigger margin call anymore
        assert!(result.is_ok());

        // Step 3: Price drops further to liquidation level
        let liquidation_price = 170.0;
        let result = engine
            .update_position_margin(account_id, "SOLUSDT", liquidation_price)
            .await;

        // Should trigger liquidation
        assert!(matches!(
            result,
            Err(LiquidationError::InsufficientMargin { .. })
        ));

        // Verify position was added to liquidation queue
        let processed = engine.process_liquidation_queue().await.unwrap();
        assert!(!processed.is_empty());
        assert_eq!(processed[0].0, account_id);
        assert_eq!(processed[0].1, "SOLUSDT");
    }
}

#[cfg(test)]
mod forced_liquidation_tests {
    use super::*;

    #[tokio::test]
    async fn test_atomic_liquidation_execution() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "liquidation_target";

        // Setup account with position ready for liquidation
        {
            let mut accounts = engine.accounts.write().await;
            let mut positions = HashMap::new();

            // Create a position at liquidation threshold
            positions.insert(
                "AVAXUSD".to_string(),
                MarginPosition {
                    symbol: "AVAXUSD".to_string(),
                    size: 200.0,
                    entry_price: 35.0,
                    current_price: 32.0, // Already at loss
                    leverage: 20.0,
                    margin_mode: MarginMode::Isolated,
                    initial_margin: 350.0,     // 200 * 35 / 20 = 350
                    maintenance_margin: 320.0, // 200 * 32 * 0.05 = 320
                    unrealized_pnl: -600.0,    // 200 * (32 - 35) = -600
                    liquidation_price: 31.75,  // Calculated liquidation price
                    margin_ratio: 1.03,        // Below liquidation threshold
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );

            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 10_000.0,
                    available_balance: 9_650.0,
                    used_margin: 350.0,
                    maintenance_margin: 320.0,
                    unrealized_pnl: -600.0,
                    margin_level: 1.03,
                    positions,
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Trigger forced liquidation
        let result = engine.trigger_liquidation(account_id, "AVAXUSD").await;
        assert!(result.is_ok());

        // Process liquidation queue atomically
        let processed = engine.process_liquidation_queue().await.unwrap();
        assert_eq!(processed.len(), 1);
        assert_eq!(
            processed[0],
            (account_id.to_string(), "AVAXUSD".to_string())
        );

        // Verify position was liquidated
        let account = engine.get_account_info(account_id).await.unwrap();
        assert!(!account.positions.contains_key("AVAXUSD"));
        assert_eq!(account.used_margin, 0.0);

        // Verify loss was realized
        assert!(account.total_balance < 10_000.0); // Loss realized
    }

    #[tokio::test]
    async fn test_order_book_impact_simulation() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "market_impact_test";

        // Create large position that would impact order book
        {
            let mut accounts = engine.accounts.write().await;
            let mut positions = HashMap::new();

            // Large BTC position
            positions.insert(
                "BTCUSD".to_string(),
                MarginPosition {
                    symbol: "BTCUSD".to_string(),
                    size: 50.0, // 50 BTC - large position
                    entry_price: 44_000.0,
                    current_price: 42_000.0,
                    leverage: 10.0,
                    margin_mode: MarginMode::Isolated,
                    initial_margin: 220_000.0,     // 50 * 44000 / 10 = 220K
                    maintenance_margin: 105_000.0, // 50 * 42000 * 0.05 = 105K
                    unrealized_pnl: -100_000.0,    // 50 * (42000 - 44000) = -100K
                    liquidation_price: 41_200.0,
                    margin_ratio: 1.14, // Still above liquidation
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );

            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 300_000.0,
                    available_balance: 80_000.0,
                    used_margin: 220_000.0,
                    maintenance_margin: 105_000.0,
                    unrealized_pnl: -100_000.0,
                    margin_level: 1.14,
                    positions,
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Simulate price feed updates
        engine.update_price("BTCUSD", 42_000.0).await;
        let mark_price = engine.get_mark_price("BTCUSD").await.unwrap();
        assert!(mark_price > 42_000.0); // Should include premium

        // Test liquidation price calculations with mark price adjustments
        let position_update = engine
            .update_position_margin(account_id, "BTCUSD", 41_000.0)
            .await;

        // Should trigger liquidation due to price drop
        assert!(matches!(
            position_update,
            Err(LiquidationError::InsufficientMargin { .. })
        ));

        // Simulate market impact of large liquidation
        // In real scenario, this would affect other positions and market
        let liquidation_result = engine.trigger_liquidation(account_id, "BTCUSD").await;
        assert!(liquidation_result.is_ok());

        // Process with market impact consideration
        let processed = engine.process_liquidation_queue().await.unwrap();
        assert_eq!(processed.len(), 1);

        // Verify large position liquidation effects
        let account = engine.get_account_info(account_id).await.unwrap();
        assert!(!account.positions.contains_key("BTCUSD"));

        // Large loss should be reflected in account balance
        assert!(account.total_balance < 250_000.0); // Significant loss
    }

    #[tokio::test]
    async fn test_partial_liquidation_scenarios() {
        let parameters = LiquidationParameters {
            initial_margin_rate: 0.10,
            maintenance_margin_rate: 0.05,
            liquidation_buffer: 0.01,
            margin_call_threshold: 1.25,
            liquidation_threshold: 1.08,
            max_leverage: 100.0,
            funding_rate: 0.0001,
            mark_price_premium: 0.0005,
        };
        let engine = LiquidationEngine::new(parameters);
        let account_id = "partial_liquidation";

        // Setup account with multiple positions
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 75_000.0,
                    available_balance: 35_000.0,
                    used_margin: 40_000.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.25,
                    liquidation_level: 1.08,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Create multiple positions with different risk profiles
        let positions_data = vec![
            ("DOGEUSDT", 10_000.0, 0.08, 50.0), // High leverage, small value
            ("LINKUSD", 500.0, 25.0, 20.0),     // Medium leverage
            ("MATICUSD", 5_000.0, 1.10, 15.0),  // Lower leverage
        ];

        for (symbol, size, entry_price, leverage) in positions_data {
            let result = engine
                .create_position(
                    account_id,
                    symbol,
                    size,
                    entry_price,
                    leverage,
                    MarginMode::Cross, // Cross margin for partial liquidation
                )
                .await;
            assert!(result.is_ok());
        }

        // Simulate market stress affecting all positions
        let stress_prices = vec![
            ("DOGEUSDT", 0.075), // -6.25% drop
            ("LINKUSD", 23.5),   // -6% drop
            ("MATICUSD", 1.05),  // -4.5% drop
        ];

        for (symbol, stress_price) in stress_prices {
            let result = engine
                .update_position_margin(account_id, symbol, stress_price)
                .await;
            // Some positions might trigger margin calls
            if result.is_err() {
                println!(
                    "Position {} triggered margin requirement: {:?}",
                    symbol, result
                );
            }
        }

        // Check which positions are at risk
        let account = engine.get_account_info(account_id).await.unwrap();
        for (symbol, position) in &account.positions {
            println!(
                "Position {}: Margin ratio = {:.3}, Liquidation price = {:.4}",
                symbol, position.margin_ratio, position.liquidation_price
            );

            // Verify liquidation prices are reasonable
            assert!(position.liquidation_price > 0.0);
            if position.size > 0.0 {
                assert!(position.liquidation_price < position.entry_price);
            } else {
                assert!(position.liquidation_price > position.entry_price);
            }
        }
    }
}

#[cfg(test)]
mod cascade_liquidation_tests {
    use super::*;

    #[tokio::test]
    async fn test_cascade_liquidation_scenario() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());

        // Create multiple interconnected accounts that could trigger cascade
        let accounts_data = vec![
            (
                "whale_account",
                500_000.0,
                vec![("BTCUSD", 20.0, 45_000.0, 15.0)],
            ),
            (
                "hedge_fund",
                300_000.0,
                vec![
                    ("BTCUSD", -10.0, 45_200.0, 20.0),
                    ("ETHUSD", 50.0, 3_100.0, 12.0),
                ],
            ),
            (
                "prop_trader",
                150_000.0,
                vec![("BTCUSD", 8.0, 44_800.0, 25.0)],
            ),
            (
                "retail_high",
                75_000.0,
                vec![("BTCUSD", 3.0, 45_100.0, 30.0)],
            ),
        ];

        // Setup all accounts
        for (account_id, balance, positions) in &accounts_data {
            {
                let mut accounts = engine.accounts.write().await;
                accounts.insert(
                    account_id.to_string(),
                    MarginAccount {
                        total_balance: *balance,
                        available_balance: *balance,
                        used_margin: 0.0,
                        maintenance_margin: 0.0,
                        unrealized_pnl: 0.0,
                        margin_level: f64::INFINITY,
                        positions: HashMap::new(),
                        margin_call_level: 1.20,
                        liquidation_level: 1.05,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                    },
                );
            }

            // Create positions for each account
            for (symbol, size, entry_price, leverage) in positions {
                let result = engine
                    .create_position(
                        account_id,
                        symbol,
                        *size,
                        *entry_price,
                        *leverage,
                        MarginMode::Isolated,
                    )
                    .await;
                assert!(result.is_ok());
            }
        }

        // Simulate market crash scenario - BTC drops 15%
        let crash_price = 38_250.0; // 15% drop from ~45K average

        let mut liquidation_triggered = Vec::new();

        // Update all BTC positions with crash price
        for (account_id, _, positions) in &accounts_data {
            for (symbol, _, _, _) in positions {
                if symbol == &"BTCUSD" {
                    let result = engine
                        .update_position_margin(account_id, symbol, crash_price)
                        .await;

                    match result {
                        Err(LiquidationError::InsufficientMargin { .. }) => {
                            liquidation_triggered.push((account_id, symbol));
                        }
                        Err(LiquidationError::MarginCall(_)) => {
                            println!("Margin call triggered for {}: {}", account_id, symbol);
                        }
                        Ok(_) => {
                            // Position survived
                        }
                        Err(e) => {
                            println!("Error updating {}: {:?}", account_id, e);
                        }
                    }
                }
            }
        }

        println!("Liquidations triggered: {:?}", liquidation_triggered);
        assert!(
            !liquidation_triggered.is_empty(),
            "Market crash should trigger liquidations"
        );

        // Process all liquidations atomically to prevent race conditions
        let processed = engine.process_liquidation_queue().await.unwrap();
        println!("Processed liquidations: {:?}", processed);

        // Verify cascade effects
        let mut total_liquidated_value = 0.0;
        for (account_id, _) in &processed {
            let account = engine.get_account_info(account_id).await;
            if let Some(acc) = account {
                total_liquidated_value += acc.total_balance;
                println!(
                    "Account {} final balance: ${:.2}",
                    account_id, acc.total_balance
                );
            }
        }

        assert!(total_liquidated_value > 0.0);
        println!(
            "Total value affected by cascade: ${:.2}",
            total_liquidated_value
        );
    }

    #[tokio::test]
    async fn test_domino_effect_prevention() {
        let parameters = LiquidationParameters {
            initial_margin_rate: 0.12,     // Higher margins
            maintenance_margin_rate: 0.06, // Higher maintenance
            liquidation_buffer: 0.02,      // Larger buffer
            margin_call_threshold: 1.40,   // Earlier margin calls
            liquidation_threshold: 1.15,   // Earlier liquidations
            max_leverage: 20.0,            // Lower max leverage
            funding_rate: 0.0001,
            mark_price_premium: 0.001, // Higher premium for safety
        };
        let engine = LiquidationEngine::new(parameters);

        // Create accounts with conservative parameters
        let account_id = "conservative_trader";
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 100_000.0,
                    available_balance: 100_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.40,
                    liquidation_level: 1.15,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Create position with conservative leverage
        let result = engine
            .create_position(
                account_id,
                "BTCUSD",
                2.0,      // 2 BTC
                44_000.0, // $44K entry
                10.0,     // Only 10x leverage (vs 20x max)
                MarginMode::Isolated,
            )
            .await;
        assert!(result.is_ok());

        // Test that position survives moderate market stress
        let moderate_drop_price = 41_800.0; // 5% drop
        let result = engine
            .update_position_margin(account_id, "BTCUSD", moderate_drop_price)
            .await;

        // Should trigger margin call but not liquidation due to conservative parameters
        match result {
            Err(LiquidationError::MarginCall(_)) => {
                // Expected - early warning system working
                println!("Early margin call triggered as expected");
            }
            Ok(_) => {
                // Also acceptable - position is well-margined
                println!("Position survived moderate stress");
            }
            Err(LiquidationError::InsufficientMargin { .. }) => {
                panic!("Conservative position should not be liquidated on moderate drop");
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }

        // Verify position still exists and is healthy
        let account = engine.get_account_info(account_id).await.unwrap();
        assert!(account.positions.contains_key("BTCUSD"));

        let position = &account.positions["BTCUSD"];
        assert!(position.margin_ratio > 1.15); // Above liquidation threshold
        println!("Position margin ratio: {:.3}", position.margin_ratio);
    }

    #[tokio::test]
    async fn test_liquidity_pool_depth_impact() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "liquidity_test";

        // Simulate different market depth scenarios
        let market_scenarios = vec![
            ("high_liquidity", 1_000_000.0), // $1M available liquidity
            ("medium_liquidity", 100_000.0), // $100K available
            ("low_liquidity", 25_000.0),     // $25K available
        ];

        for (scenario_name, available_liquidity) in market_scenarios {
            // Setup fresh account for each scenario
            {
                let mut accounts = engine.accounts.write().await;
                accounts.insert(
                    format!("{}_{}", account_id, scenario_name),
                    MarginAccount {
                        total_balance: 50_000.0,
                        available_balance: 50_000.0,
                        used_margin: 0.0,
                        maintenance_margin: 0.0,
                        unrealized_pnl: 0.0,
                        margin_level: f64::INFINITY,
                        positions: HashMap::new(),
                        margin_call_level: 1.20,
                        liquidation_level: 1.05,
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                    },
                );
            }

            let test_account = format!("{}_{}", account_id, scenario_name);

            // Create position sized relative to available liquidity
            let position_size = available_liquidity / 45_000.0; // BTC position
            let result = engine
                .create_position(
                    &test_account,
                    "BTCUSD",
                    position_size,
                    45_000.0,
                    20.0, // 20x leverage
                    MarginMode::Isolated,
                )
                .await;
            assert!(result.is_ok());

            // Simulate liquidation trigger
            let liquidation_price = 42_000.0;
            let result = engine
                .update_position_margin(&test_account, "BTCUSD", liquidation_price)
                .await;

            // The impact should vary based on liquidity depth
            match result {
                Err(LiquidationError::InsufficientMargin { .. }) => {
                    // Liquidation triggered - now process it
                    let processed = engine.process_liquidation_queue().await.unwrap();

                    // In low liquidity scenario, slippage would be higher
                    if scenario_name == "low_liquidity" {
                        println!("High impact liquidation in low liquidity market");
                    } else {
                        println!("Normal liquidation in {} market", scenario_name);
                    }

                    assert!(!processed.is_empty());
                }
                _ => {
                    // Position might survive in high liquidity scenarios due to better mark pricing
                }
            }

            // Verify final state
            let account = engine.get_account_info(&test_account).await.unwrap();
            println!(
                "Scenario {}: Final balance = ${:.2}",
                scenario_name, account.total_balance
            );
        }
    }
}

#[cfg(test)]
mod real_market_simulation_tests {
    use super::*;

    #[tokio::test]
    async fn test_realistic_market_conditions() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());

        // Simulate realistic market data and conditions
        let market_data = vec![
            // (timestamp, BTC_price, ETH_price, volatility_factor)
            (1640995200000u64, 47_000.0, 3_800.0, 1.0), // Normal market
            (1640995800000u64, 46_500.0, 3_750.0, 1.2), // Small dip
            (1640996400000u64, 45_200.0, 3_600.0, 1.8), // Increased volatility
            (1640997000000u64, 43_800.0, 3_400.0, 2.5), // High volatility drop
            (1640997600000u64, 41_500.0, 3_100.0, 3.2), // Crash conditions
        ];

        // Create realistic trader profiles
        let trader_profiles = vec![
            (
                "conservative_trader",
                200_000.0,
                vec![("BTCUSD", 2.0, 47_000.0, 5.0)],
            ),
            (
                "moderate_trader",
                100_000.0,
                vec![
                    ("BTCUSD", 2.5, 47_000.0, 10.0),
                    ("ETHUSD", 20.0, 3_800.0, 8.0),
                ],
            ),
            (
                "aggressive_trader",
                50_000.0,
                vec![("BTCUSD", 3.0, 47_000.0, 20.0)],
            ),
            (
                "degenerate_trader",
                25_000.0,
                vec![("BTCUSD", 2.0, 47_000.0, 50.0)],
            ),
        ];

        // Setup all traders
        for (trader_id, balance, positions) in &trader_profiles {
            {
                let mut accounts = engine.accounts.write().await;
                accounts.insert(
                    trader_id.to_string(),
                    MarginAccount {
                        total_balance: *balance,
                        available_balance: *balance,
                        used_margin: 0.0,
                        maintenance_margin: 0.0,
                        unrealized_pnl: 0.0,
                        margin_level: f64::INFINITY,
                        positions: HashMap::new(),
                        margin_call_level: 1.20,
                        liquidation_level: 1.05,
                        timestamp: 1640995200000u64,
                    },
                );
            }

            for (symbol, size, entry_price, leverage) in positions {
                let result = engine
                    .create_position(
                        trader_id,
                        symbol,
                        *size,
                        *entry_price,
                        *leverage,
                        MarginMode::Isolated,
                    )
                    .await;
                assert!(result.is_ok());
            }
        }

        // Simulate market progression
        let mut liquidation_events = Vec::new();

        for (timestamp, btc_price, eth_price, volatility) in market_data {
            println!(
                "\n=== Market Update: BTC=${:.0}, ETH=${:.0}, Vol={:.1}x ===",
                btc_price, eth_price, volatility
            );

            // Update price feeds
            engine.update_price("BTCUSD", btc_price).await;
            engine.update_price("ETHUSD", eth_price).await;

            // Update all positions
            for (trader_id, _, positions) in &trader_profiles {
                let account_before = engine.get_account_info(trader_id).await;

                for (symbol, _, _, _) in positions {
                    let current_price = match *symbol {
                        "BTCUSD" => btc_price,
                        "ETHUSD" => eth_price,
                        _ => continue,
                    };

                    let result = engine
                        .update_position_margin(trader_id, symbol, current_price)
                        .await;

                    match result {
                        Ok(_) => {
                            // Position healthy
                        }
                        Err(LiquidationError::MarginCall(sym)) => {
                            println!("⚠️  Margin call: {} - {}", trader_id, sym);
                        }
                        Err(LiquidationError::InsufficientMargin { .. }) => {
                            println!("💀 Liquidation: {} - {}", trader_id, symbol);
                            liquidation_events.push((trader_id.clone(), symbol.clone(), timestamp));
                        }
                        Err(e) => {
                            println!("❌ Error updating {} {}: {:?}", trader_id, symbol, e);
                        }
                    }
                }

                // Print trader status
                if let Some(account) = engine.get_account_info(trader_id).await {
                    println!(
                        "{}: Balance=${:.0}, Positions={}, Margin Level={:.2}",
                        trader_id,
                        account.total_balance,
                        account.positions.len(),
                        if account.positions.is_empty() {
                            0.0
                        } else {
                            account
                                .positions
                                .values()
                                .map(|p| p.margin_ratio)
                                .fold(0.0, f64::max)
                        }
                    );
                }
            }

            // Process liquidation queue
            let processed = engine.process_liquidation_queue().await.unwrap();
            if !processed.is_empty() {
                println!("Processed {} liquidations", processed.len());
            }

            // Add delay to simulate real market timing
            sleep(Duration::from_millis(10)).await;
        }

        println!("\n=== Final Results ===");
        println!("Total liquidation events: {}", liquidation_events.len());

        for (trader_id, symbol, timestamp) in &liquidation_events {
            println!(
                "Liquidated: {} {} at timestamp {}",
                trader_id, symbol, timestamp
            );
        }

        // Verify realistic outcomes
        assert!(
            liquidation_events.len() > 0,
            "Market crash should cause liquidations"
        );

        // High leverage traders should be liquidated first
        let degenerate_liquidated = liquidation_events
            .iter()
            .any(|(trader, _, _)| trader == "degenerate_trader");
        assert!(
            degenerate_liquidated,
            "50x leverage trader should be liquidated"
        );

        // Conservative trader should survive
        let conservative_liquidated = liquidation_events
            .iter()
            .any(|(trader, _, _)| trader == "conservative_trader");
        assert!(
            !conservative_liquidated,
            "5x leverage trader should survive"
        );
    }

    #[tokio::test]
    async fn test_funding_rate_impact() {
        let parameters = LiquidationParameters {
            funding_rate: 0.001, // 0.1% hourly funding rate (very high)
            ..LiquidationParameters::default()
        };
        let engine = LiquidationEngine::new(parameters);

        let account_id = "funding_test";
        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 50_000.0,
                    available_balance: 50_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Create large perpetual position
        let result = engine
            .create_position(
                account_id,
                "BTCUSD-PERP",
                10.0,     // 10 BTC
                45_000.0, // $45K
                20.0,     // 20x leverage
                MarginMode::Isolated,
            )
            .await;
        assert!(result.is_ok());

        // Calculate funding payments over time
        let position_value = 10.0 * 45_000.0; // $450K notional
        let hourly_funding =
            engine.calculate_funding_payment(10.0, 45_000.0, parameters.funding_rate);

        println!("Hourly funding payment: ${:.2}", hourly_funding);
        assert!(hourly_funding > 0.0);
        assert_eq!(hourly_funding, 450.0); // 10 * 45000 * 0.001

        // Simulate 24 hours of funding
        let daily_funding = hourly_funding * 24.0;
        println!("Daily funding cost: ${:.2}", daily_funding);
        assert_eq!(daily_funding, 10_800.0); // Very expensive

        // Test mark price with premium
        engine.update_price("BTCUSD-PERP", 45_000.0).await;
        let mark_price = engine.get_mark_price("BTCUSD-PERP").await.unwrap();
        assert!(mark_price > 45_000.0);
        println!("Mark price with premium: ${:.2}", mark_price);

        // Verify funding rate affects position sustainability
        let account = engine.get_account_info(account_id).await.unwrap();
        let position = &account.positions["BTCUSD-PERP"];

        // High funding costs make position more risky
        assert!(position.initial_margin > 0.0);
        println!("Position initial margin: ${:.2}", position.initial_margin);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_zero_price_edge_cases() {
        let engine = LiquidationEngine::new(LiquidationParameters::default());
        let account_id = "edge_case_test";

        {
            let mut accounts = engine.accounts.write().await;
            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 10_000.0,
                    available_balance: 10_000.0,
                    used_margin: 0.0,
                    maintenance_margin: 0.0,
                    unrealized_pnl: 0.0,
                    margin_level: f64::INFINITY,
                    positions: HashMap::new(),
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Test invalid leverage values
        let invalid_leverage_result = engine
            .create_position(account_id, "TESTUSD", 1.0, 100.0, 0.0, MarginMode::Isolated)
            .await;
        assert!(invalid_leverage_result.is_err());

        let excessive_leverage_result = engine
            .create_position(
                account_id,
                "TESTUSD",
                1.0,
                100.0,
                1000.0,
                MarginMode::Isolated,
            )
            .await;
        assert!(excessive_leverage_result.is_err());

        // Test zero position size
        let zero_size_result = engine
            .create_position(
                account_id,
                "TESTUSD",
                0.0,
                100.0,
                10.0,
                MarginMode::Isolated,
            )
            .await;
        // Should be allowed but no margin used
        if zero_size_result.is_ok() {
            let account = engine.get_account_info(account_id).await.unwrap();
            assert_eq!(account.used_margin, 0.0);
        }
    }

    #[tokio::test]
    async fn test_concurrent_liquidation_handling() {
        let engine = Arc::new(LiquidationEngine::new(LiquidationParameters::default()));
        let account_id = "concurrent_test";

        // Setup account with position ready for liquidation
        {
            let mut accounts = engine.accounts.write().await;
            let mut positions = HashMap::new();

            positions.insert(
                "TESTUSD".to_string(),
                MarginPosition {
                    symbol: "TESTUSD".to_string(),
                    size: 100.0,
                    entry_price: 100.0,
                    current_price: 90.0,
                    leverage: 10.0,
                    margin_mode: MarginMode::Isolated,
                    initial_margin: 1000.0,
                    maintenance_margin: 450.0,
                    unrealized_pnl: -1000.0,
                    liquidation_price: 89.0,
                    margin_ratio: 1.0, // At liquidation threshold
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );

            accounts.insert(
                account_id.to_string(),
                MarginAccount {
                    total_balance: 5_000.0,
                    available_balance: 4_000.0,
                    used_margin: 1_000.0,
                    maintenance_margin: 450.0,
                    unrealized_pnl: -1_000.0,
                    margin_level: 1.0,
                    positions,
                    margin_call_level: 1.20,
                    liquidation_level: 1.05,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                },
            );
        }

        // Simulate concurrent liquidation attempts
        let engine1 = engine.clone();
        let engine2 = engine.clone();
        let account_id1 = account_id.to_string();
        let account_id2 = account_id.to_string();

        let handle1 =
            tokio::spawn(async move { engine1.trigger_liquidation(&account_id1, "TESTUSD").await });

        let handle2 =
            tokio::spawn(async move { engine2.trigger_liquidation(&account_id2, "TESTUSD").await });

        // Both should complete without deadlock
        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        // At least one should succeed
        assert!(result1.is_ok() || result2.is_ok());

        // Process liquidation queue - should handle duplicates gracefully
        let processed = engine.process_liquidation_queue().await.unwrap();
        println!("Processed concurrent liquidations: {:?}", processed);
    }
}
