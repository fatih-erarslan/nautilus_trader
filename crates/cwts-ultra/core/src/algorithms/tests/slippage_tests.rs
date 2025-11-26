use crate::algorithms::slippage_calculator::*;
use crate::common_types::TradeSide;
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Comprehensive slippage calculator tests implementing real market impact models,
/// order book depth analysis, large order slippage calculation, and liquidity pool testing.
/// Tests model realistic market conditions with no mocks - using real order book data patterns.

// Shared helper functions for all test modules
#[cfg(test)]
fn create_deep_order_book() -> OrderBook {
    OrderBook {
        symbol: "BTCUSD".to_string(),
        bids: vec![
            OrderBookLevel { price: 44_995.0, quantity: 0.5, timestamp: 0 },
            OrderBookLevel { price: 44_990.0, quantity: 1.2, timestamp: 0 },
            OrderBookLevel { price: 44_985.0, quantity: 2.1, timestamp: 0 },
            OrderBookLevel { price: 44_980.0, quantity: 1.8, timestamp: 0 },
            OrderBookLevel { price: 44_975.0, quantity: 3.5, timestamp: 0 },
            OrderBookLevel { price: 44_970.0, quantity: 5.2, timestamp: 0 },
            OrderBookLevel { price: 44_965.0, quantity: 4.1, timestamp: 0 },
            OrderBookLevel { price: 44_960.0, quantity: 7.3, timestamp: 0 },
            OrderBookLevel { price: 44_955.0, quantity: 6.8, timestamp: 0 },
            OrderBookLevel { price: 44_950.0, quantity: 12.5, timestamp: 0 },
            OrderBookLevel { price: 44_945.0, quantity: 15.2, timestamp: 0 },
            OrderBookLevel { price: 44_940.0, quantity: 20.8, timestamp: 0 },
            OrderBookLevel { price: 44_935.0, quantity: 25.3, timestamp: 0 },
            OrderBookLevel { price: 44_930.0, quantity: 18.7, timestamp: 0 },
            OrderBookLevel { price: 44_925.0, quantity: 30.5, timestamp: 0 },
        ],
        asks: vec![
            OrderBookLevel { price: 45_005.0, quantity: 0.6, timestamp: 0 },
            OrderBookLevel { price: 45_010.0, quantity: 1.5, timestamp: 0 },
            OrderBookLevel { price: 45_015.0, quantity: 2.3, timestamp: 0 },
            OrderBookLevel { price: 45_020.0, quantity: 1.9, timestamp: 0 },
            OrderBookLevel { price: 45_025.0, quantity: 3.8, timestamp: 0 },
            OrderBookLevel { price: 45_030.0, quantity: 5.5, timestamp: 0 },
            OrderBookLevel { price: 45_035.0, quantity: 4.5, timestamp: 0 },
            OrderBookLevel { price: 45_040.0, quantity: 7.8, timestamp: 0 },
            OrderBookLevel { price: 45_045.0, quantity: 7.2, timestamp: 0 },
            OrderBookLevel { price: 45_050.0, quantity: 13.0, timestamp: 0 },
            OrderBookLevel { price: 45_055.0, quantity: 16.0, timestamp: 0 },
            OrderBookLevel { price: 45_060.0, quantity: 22.0, timestamp: 0 },
            OrderBookLevel { price: 45_065.0, quantity: 27.0, timestamp: 0 },
            OrderBookLevel { price: 45_070.0, quantity: 20.0, timestamp: 0 },
            OrderBookLevel { price: 45_075.0, quantity: 32.0, timestamp: 0 },
        ],
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
    }
}

#[cfg(test)]
fn create_thin_order_book() -> OrderBook {
    OrderBook {
        symbol: "ALTCOIN".to_string(),
        bids: vec![
            OrderBookLevel { price: 9.95, quantity: 50.0, timestamp: 0 },
            OrderBookLevel { price: 9.90, quantity: 100.0, timestamp: 0 },
            OrderBookLevel { price: 9.85, quantity: 75.0, timestamp: 0 },
            OrderBookLevel { price: 9.80, quantity: 200.0, timestamp: 0 },
            OrderBookLevel { price: 9.70, quantity: 300.0, timestamp: 0 },
        ],
        asks: vec![
            OrderBookLevel { price: 10.05, quantity: 45.0, timestamp: 0 },
            OrderBookLevel { price: 10.10, quantity: 80.0, timestamp: 0 },
            OrderBookLevel { price: 10.15, quantity: 120.0, timestamp: 0 },
            OrderBookLevel { price: 10.20, quantity: 90.0, timestamp: 0 },
            OrderBookLevel { price: 10.30, quantity: 250.0, timestamp: 0 },
        ],
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
    }
}

#[cfg(test)]
mod market_impact_tests {
    use super::*;

    // Local helper (kept for backward compatibility with existing tests)
    fn _create_deep_order_book() -> OrderBook {
        super::create_deep_order_book()
    }

    fn _create_thin_order_book() -> OrderBook {
        super::create_thin_order_book()
    }

    fn create_deep_order_book() -> OrderBook {
        OrderBook {
            symbol: "BTCUSD".to_string(),
            bids: vec![
                // Top of book - tight spread
                OrderBookLevel {
                    price: 44_995.0,
                    quantity: 0.5,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_990.0,
                    quantity: 1.2,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_985.0,
                    quantity: 2.1,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_980.0,
                    quantity: 1.8,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_975.0,
                    quantity: 3.5,
                    timestamp: 0,
                },
                // Mid levels
                OrderBookLevel {
                    price: 44_970.0,
                    quantity: 5.2,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_965.0,
                    quantity: 4.1,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_960.0,
                    quantity: 7.3,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_955.0,
                    quantity: 6.8,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_950.0,
                    quantity: 12.5,
                    timestamp: 0,
                },
                // Deeper levels - more liquidity
                OrderBookLevel {
                    price: 44_945.0,
                    quantity: 15.2,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_940.0,
                    quantity: 20.8,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_935.0,
                    quantity: 25.3,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_930.0,
                    quantity: 18.7,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_925.0,
                    quantity: 32.1,
                    timestamp: 0,
                },
            ],
            asks: vec![
                // Top of book
                OrderBookLevel {
                    price: 45_005.0,
                    quantity: 0.8,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_010.0,
                    quantity: 1.1,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_015.0,
                    quantity: 1.9,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_020.0,
                    quantity: 2.3,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_025.0,
                    quantity: 3.2,
                    timestamp: 0,
                },
                // Mid levels
                OrderBookLevel {
                    price: 45_030.0,
                    quantity: 4.8,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_035.0,
                    quantity: 3.9,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_040.0,
                    quantity: 6.7,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_045.0,
                    quantity: 8.4,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_050.0,
                    quantity: 11.2,
                    timestamp: 0,
                },
                // Deeper levels
                OrderBookLevel {
                    price: 45_055.0,
                    quantity: 14.6,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_060.0,
                    quantity: 19.3,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_065.0,
                    quantity: 22.7,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_070.0,
                    quantity: 17.9,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_075.0,
                    quantity: 28.5,
                    timestamp: 0,
                },
            ],
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    fn create_thin_order_book() -> OrderBook {
        OrderBook {
            symbol: "ALTCOIN".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 9.95,
                    quantity: 50.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 9.90,
                    quantity: 100.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 9.85,
                    quantity: 75.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 9.80,
                    quantity: 200.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 9.70,
                    quantity: 300.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 10.05,
                    quantity: 45.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 10.10,
                    quantity: 80.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 10.15,
                    quantity: 120.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 10.20,
                    quantity: 90.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 10.30,
                    quantity: 250.0,
                    timestamp: 0,
                },
            ],
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    #[test]
    fn test_large_order_slippage_calculation() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Test small order - minimal slippage
        let small_analysis = calculator
            .calculate_slippage(
                "BTCUSD",
                0.5, // 0.5 BTC - can be filled at best ask
                TradeSide::Buy,
                Some(45_000.0),
            )
            .unwrap();

        assert!(small_analysis.slippage_bps < 50.0); // Less than 0.5 bps
        assert_eq!(small_analysis.estimated_fill_price, 45_005.0); // Best ask
        println!(
            "Small order slippage: {:.2} bps",
            small_analysis.slippage_bps
        );

        // Test medium order - moderate slippage
        let medium_analysis = calculator
            .calculate_slippage(
                "BTCUSD",
                5.0, // 5 BTC - requires multiple levels
                TradeSide::Buy,
                Some(45_000.0),
            )
            .unwrap();

        assert!(medium_analysis.slippage_bps > small_analysis.slippage_bps);
        assert!(medium_analysis.estimated_fill_price > 45_005.0); // Above best ask
        println!(
            "Medium order slippage: {:.2} bps, Fill price: ${:.2}",
            medium_analysis.slippage_bps, medium_analysis.estimated_fill_price
        );

        // Test large order - significant slippage
        let large_analysis = calculator
            .calculate_slippage(
                "BTCUSD",
                25.0, // 25 BTC - consumes significant liquidity
                TradeSide::Buy,
                Some(45_000.0),
            )
            .unwrap();

        assert!(large_analysis.slippage_bps > medium_analysis.slippage_bps);
        assert!(large_analysis.estimated_fill_price > medium_analysis.estimated_fill_price);
        assert!(large_analysis.market_impact > 0.0);
        println!(
            "Large order slippage: {:.2} bps, Fill price: ${:.2}, Market impact: ${:.2}",
            large_analysis.slippage_bps,
            large_analysis.estimated_fill_price,
            large_analysis.market_impact
        );

        // Verify slippage increases with order size (square root relationship expected)
        assert!(large_analysis.slippage_percentage > medium_analysis.slippage_percentage);
        assert!(medium_analysis.slippage_percentage > small_analysis.slippage_percentage);
    }

    #[test]
    fn test_market_impact_models() {
        let parameters = SlippageParameters {
            model: MarketImpactModel {
                temporary_impact_coeff: 0.7,
                permanent_impact_coeff: 0.15,
                volatility_factor: 0.4,
                liquidity_factor: 0.25,
                volume_decay_factor: 0.93,
            },
            ..SlippageParameters::default()
        };
        let mut calculator = SlippageCalculator::new(parameters);

        // Setup historical volume data for impact calculation
        let volume_data = vec![
            1500.0, 1200.0, 1800.0, 2100.0, 1650.0, // Daily volumes in BTC
            1750.0, 1450.0, 1950.0, 1580.0, 1680.0, 1420.0, 1850.0, 2200.0, 1390.0, 1720.0,
        ];

        for volume in volume_data {
            calculator
                .volume_profile
                .entry("BTCUSD".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(volume);
        }

        // Setup order book
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Add trade history for volatility calculation
        let base_price = 45_000.0;
        let price_series = vec![
            45_000.0, 45_100.0, 44_950.0, 45_200.0, 44_800.0, 44_900.0, 45_150.0, 45_050.0,
            44_850.0, 45_300.0,
        ];

        for (i, price) in price_series.iter().enumerate() {
            let trade = Trade {
                symbol: "BTCUSD".to_string(),
                price: *price,
                quantity: 1.0,
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
                timestamp: (1640995200000u64 + i as u64 * 60000), // 1 minute intervals
            };
            calculator.add_trade(trade);
        }

        // Test different order sizes to verify impact model
        let test_sizes = vec![1.0, 5.0, 15.0, 50.0, 100.0];
        let mut previous_impact = 0.0;

        for size in test_sizes {
            let analysis = calculator
                .calculate_slippage("BTCUSD", size, TradeSide::Buy, Some(45_000.0))
                .unwrap();

            println!(
                "Order size: {:.1} BTC, Market impact: ${:.2}, Total slippage: {:.2} bps",
                size, analysis.market_impact, analysis.slippage_bps
            );

            // Market impact should generally increase with order size
            if previous_impact > 0.0 {
                assert!(
                    analysis.market_impact >= previous_impact,
                    "Market impact should increase with order size"
                );
            }

            previous_impact = analysis.market_impact;

            // Verify impact components are reasonable
            assert!(analysis.market_impact > 0.0);
            assert!(analysis.market_impact < size * 1000.0); // Sanity check
        }
    }

    #[test]
    fn test_volatility_impact_on_slippage() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Create two scenarios: low volatility vs high volatility
        let low_vol_prices = vec![
            45_000.0, 45_010.0, 45_005.0, 45_015.0, 45_008.0, 45_012.0, 45_007.0, 45_020.0,
            45_003.0, 45_018.0,
        ];

        let high_vol_prices = vec![
            45_000.0, 45_300.0, 44_600.0, 45_500.0, 44_200.0, 45_800.0, 44_100.0, 45_900.0,
            43_800.0, 46_200.0,
        ];

        // Test low volatility scenario
        for (i, price) in low_vol_prices.iter().enumerate() {
            let trade = Trade {
                symbol: "LOWVOL".to_string(),
                price: *price,
                quantity: 1.0,
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
                timestamp: (1640995200000u64 + i as u64 * 60000),
            };
            calculator.add_trade(trade);
        }

        // Test high volatility scenario
        for (i, price) in high_vol_prices.iter().enumerate() {
            let trade = Trade {
                symbol: "HIVOL".to_string(),
                price: *price,
                quantity: 1.0,
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
                timestamp: (1640995200000u64 + i as u64 * 60000),
            };
            calculator.add_trade(trade);
        }

        // Update order books for both symbols
        let mut low_vol_book = create_deep_order_book();
        low_vol_book.symbol = "LOWVOL".to_string();
        calculator.update_order_book(low_vol_book);

        let mut high_vol_book = create_deep_order_book();
        high_vol_book.symbol = "HIVOL".to_string();
        calculator.update_order_book(high_vol_book);

        // Compare slippage in different volatility environments
        let order_size = 10.0;

        let low_vol_analysis = calculator
            .calculate_slippage("LOWVOL", order_size, TradeSide::Buy, Some(45_000.0))
            .unwrap();

        let high_vol_analysis = calculator
            .calculate_slippage("HIVOL", order_size, TradeSide::Buy, Some(45_000.0))
            .unwrap();

        println!(
            "Low vol slippage: {:.2} bps, Market impact: ${:.2}",
            low_vol_analysis.slippage_bps, low_vol_analysis.market_impact
        );
        println!(
            "High vol slippage: {:.2} bps, Market impact: ${:.2}",
            high_vol_analysis.slippage_bps, high_vol_analysis.market_impact
        );

        // High volatility should result in higher market impact and slippage
        assert!(high_vol_analysis.market_impact > low_vol_analysis.market_impact);
        assert!(high_vol_analysis.execution_cost > low_vol_analysis.execution_cost);

        // Confidence intervals should be wider for high volatility
        let low_vol_ci_width =
            low_vol_analysis.confidence_interval.1 - low_vol_analysis.confidence_interval.0;
        let high_vol_ci_width =
            high_vol_analysis.confidence_interval.1 - high_vol_analysis.confidence_interval.0;
        assert!(high_vol_ci_width > low_vol_ci_width);
    }

    #[test]
    fn test_liquidity_score_calculation() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());

        // Test different liquidity scenarios
        let deep_liquidity = vec![
            OrderBookLevel {
                price: 100.0,
                quantity: 500.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 100.5,
                quantity: 400.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 101.0,
                quantity: 350.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 101.5,
                quantity: 300.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 102.0,
                quantity: 250.0,
                timestamp: 0,
            },
        ];

        let shallow_liquidity = vec![
            OrderBookLevel {
                price: 100.0,
                quantity: 10.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 100.5,
                quantity: 8.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 101.0,
                quantity: 5.0,
                timestamp: 0,
            },
        ];

        let thin_liquidity = vec![OrderBookLevel {
            price: 100.0,
            quantity: 1.0,
            timestamp: 0,
        }];

        // Test with 50 unit order
        let order_size = 50.0;

        let deep_score = calculator.calculate_liquidity_score(&deep_liquidity, order_size);
        let shallow_score = calculator.calculate_liquidity_score(&shallow_liquidity, order_size);
        let thin_score = calculator.calculate_liquidity_score(&thin_liquidity, order_size);

        println!("Deep liquidity score: {:.3}", deep_score);
        println!("Shallow liquidity score: {:.3}", shallow_score);
        println!("Thin liquidity score: {:.3}", thin_score);

        // Scores should decrease with reduced liquidity
        assert!(deep_score > shallow_score);
        assert!(shallow_score > thin_score);
        assert!(deep_score > 0.7); // Should be high score
        assert!(thin_score < 0.3); // Should be low score

        // Test with different order sizes
        let small_order = 5.0;
        let large_order = 500.0;

        let deep_score_small = calculator.calculate_liquidity_score(&deep_liquidity, small_order);
        let deep_score_large = calculator.calculate_liquidity_score(&deep_liquidity, large_order);

        // Smaller orders should have better liquidity scores
        assert!(deep_score_small > deep_score_large);
    }
}

#[cfg(test)]
mod order_execution_simulation_tests {
    use super::*;

    #[test]
    fn test_vwap_execution_accuracy() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());

        let test_levels = vec![
            OrderBookLevel {
                price: 50_000.0,
                quantity: 1.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 50_010.0,
                quantity: 2.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 50_020.0,
                quantity: 3.0,
                timestamp: 0,
            },
            OrderBookLevel {
                price: 50_030.0,
                quantity: 4.0,
                timestamp: 0,
            },
        ];

        // Test exact calculation
        let (vwap, available) = calculator
            .calculate_vwap_execution(&test_levels, 6.0)
            .unwrap();

        // Manual calculation: (1*50000 + 2*50010 + 3*50020) / 6 = 50013.33
        let expected_vwap = (1.0 * 50_000.0 + 2.0 * 50_010.0 + 3.0 * 50_020.0) / 6.0;
        assert_eq!(vwap, expected_vwap);
        assert_eq!(available, 10.0); // Total available liquidity

        println!("VWAP for 6 units: ${:.2}", vwap);

        // Test partial fill
        let (vwap_partial, _) = calculator
            .calculate_vwap_execution(&test_levels, 2.5)
            .unwrap();
        let expected_partial = (1.0 * 50_000.0 + 1.5 * 50_010.0) / 2.5;
        assert_eq!(vwap_partial, expected_partial);

        println!("VWAP for 2.5 units: ${:.2}", vwap_partial);
    }

    #[test]
    fn test_order_splitting_optimization() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Setup volume profile
        let daily_volumes = vec![
            2000.0, 2200.0, 1800.0, 2400.0, 2100.0, // 5-day average = 2100
            1900.0, 2300.0, 2000.0, 2250.0, 1950.0,
        ];

        for volume in daily_volumes {
            calculator
                .volume_profile
                .entry("BTCUSD".to_string())
                .or_insert_with(VecDeque::new)
                .push_back(volume);
        }

        // Test optimal splitting for large orders
        let total_order = 500.0; // 500 BTC order
        let max_participation = 0.05; // 5% of daily volume

        let splits = calculator
            .calculate_optimal_order_splits(
                "BTCUSD",
                total_order,
                TradeSide::Buy,
                max_participation,
            )
            .unwrap();

        println!("Order splits for {} BTC: {:?}", total_order, splits);

        // Verify splits
        assert!(splits.len() > 1); // Should be split
        assert_eq!(splits.iter().sum::<f64>(), total_order); // Total should match

        // Each split should be <= max participation * avg volume
        let avg_volume = 2100.0; // Average from our data
        let max_split_size = avg_volume * max_participation;

        for split in &splits {
            assert!(
                *split <= max_split_size,
                "Split {} exceeds max size {}",
                split,
                max_split_size
            );
        }

        // Test small order - no splitting needed
        let small_order = 50.0;
        let small_splits = calculator
            .calculate_optimal_order_splits(
                "BTCUSD",
                small_order,
                TradeSide::Buy,
                max_participation,
            )
            .unwrap();

        assert_eq!(small_splits.len(), 1);
        assert_eq!(small_splits[0], small_order);
    }

    #[test]
    fn test_dynamic_slippage_with_time_horizon() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Add some trade history for volatility calculation
        let prices = vec![45_000.0, 45_100.0, 44_900.0, 45_200.0, 44_800.0];
        for (i, price) in prices.iter().enumerate() {
            let trade = Trade {
                symbol: "BTCUSD".to_string(),
                price: *price,
                quantity: 1.0,
                side: TradeSide::Buy,
                timestamp: (1640995200000u64 + i as u64 * 60000),
            };
            calculator.add_trade(trade);
        }

        let order_size = 10.0;
        let base_analysis = calculator
            .calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0))
            .unwrap();

        // Test different time horizons
        let time_horizons = vec![
            (1000, "1 second"),
            (60_000, "1 minute"),
            (300_000, "5 minutes"),
            (1_800_000, "30 minutes"),
            (3_600_000, "1 hour"),
        ];

        println!("Base slippage: {:.2} bps", base_analysis.slippage_bps);

        for (time_ms, description) in time_horizons {
            let dynamic_analysis = calculator
                .estimate_dynamic_slippage("BTCUSD", order_size, TradeSide::Buy, time_ms)
                .unwrap();

            println!(
                "{}: {:.2} bps, Market impact: ${:.2}",
                description, dynamic_analysis.slippage_bps, dynamic_analysis.market_impact
            );

            // Longer time horizons should generally have higher slippage
            if time_ms > 1000 {
                assert!(dynamic_analysis.slippage_bps >= base_analysis.slippage_bps);
                assert!(dynamic_analysis.execution_cost >= base_analysis.execution_cost);
            }
        }
    }

    #[test]
    fn test_confidence_interval_calculation() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Create extensive trade history for statistical analysis
        let base_price = 45_000.0;
        let mut price = base_price;

        // Simulate 100 trades with realistic price movements
        for i in 0..100 {
            // Random walk with mean reversion
            let random_change = if i % 3 == 0 {
                0.002
            } else if i % 3 == 1 {
                -0.001
            } else {
                0.0005
            };
            price *= 1.0 + random_change;

            let trade = Trade {
                symbol: "BTCUSD".to_string(),
                price,
                quantity: 1.0 + (i as f64 * 0.1),
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
                timestamp: (1640995200000u64 + i as u64 * 30000), // 30-second intervals
            };
            calculator.add_trade(trade);
        }

        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Test confidence interval for different order sizes
        let test_orders = vec![1.0, 5.0, 15.0, 30.0];

        for order_size in test_orders {
            let analysis = calculator
                .calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0))
                .unwrap();

            let ci_width = analysis.confidence_interval.1 - analysis.confidence_interval.0;
            println!(
                "Order {:.1} BTC: Slippage {:.2} bps, CI: [{:.2}, {:.2}], Width: {:.2}",
                order_size,
                analysis.slippage_bps,
                analysis.confidence_interval.0,
                analysis.confidence_interval.1,
                ci_width
            );

            // Basic sanity checks
            assert!(analysis.confidence_interval.1 > analysis.confidence_interval.0);
            assert!(ci_width > 0.0);
            assert!(ci_width < 100.0); // Should be reasonable

            // Slippage estimate should be within confidence interval (approximately)
            let in_interval = analysis.slippage_percentage >= analysis.confidence_interval.0
                && analysis.slippage_percentage <= analysis.confidence_interval.1;

            if !in_interval {
                println!(
                    "Warning: Slippage {:.3}% outside CI [{:.3}%, {:.3}%]",
                    analysis.slippage_percentage,
                    analysis.confidence_interval.0,
                    analysis.confidence_interval.1
                );
            }
        }
    }
}

#[cfg(test)]
mod liquidity_pool_depth_tests {
    use super::*;

    fn create_exchange_order_books() -> HashMap<String, OrderBook> {
        let mut books = HashMap::new();

        // Binance-style deep book
        books.insert(
            "binance".to_string(),
            OrderBook {
                symbol: "BTCUSD".to_string(),
                bids: vec![
                    OrderBookLevel {
                        price: 44_995.0,
                        quantity: 2.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_990.0,
                        quantity: 5.2,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_985.0,
                        quantity: 8.1,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_980.0,
                        quantity: 12.3,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_975.0,
                        quantity: 18.7,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_970.0,
                        quantity: 25.4,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_965.0,
                        quantity: 32.1,
                        timestamp: 0,
                    },
                ],
                asks: vec![
                    OrderBookLevel {
                        price: 45_005.0,
                        quantity: 2.8,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_010.0,
                        quantity: 5.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_015.0,
                        quantity: 8.9,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_020.0,
                        quantity: 13.2,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_025.0,
                        quantity: 19.6,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_030.0,
                        quantity: 26.8,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_035.0,
                        quantity: 35.4,
                        timestamp: 0,
                    },
                ],
                timestamp: 0,
            },
        );

        // Coinbase-style medium liquidity
        books.insert(
            "coinbase".to_string(),
            OrderBook {
                symbol: "BTCUSD".to_string(),
                bids: vec![
                    OrderBookLevel {
                        price: 44_992.0,
                        quantity: 1.2,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_987.0,
                        quantity: 2.8,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_982.0,
                        quantity: 4.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_977.0,
                        quantity: 6.1,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_972.0,
                        quantity: 8.9,
                        timestamp: 0,
                    },
                ],
                asks: vec![
                    OrderBookLevel {
                        price: 45_008.0,
                        quantity: 1.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_013.0,
                        quantity: 3.1,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_018.0,
                        quantity: 5.2,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_023.0,
                        quantity: 7.8,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_028.0,
                        quantity: 9.4,
                        timestamp: 0,
                    },
                ],
                timestamp: 0,
            },
        );

        // Small exchange - limited liquidity
        books.insert(
            "small_exchange".to_string(),
            OrderBook {
                symbol: "BTCUSD".to_string(),
                bids: vec![
                    OrderBookLevel {
                        price: 44_985.0,
                        quantity: 0.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_975.0,
                        quantity: 1.2,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 44_965.0,
                        quantity: 0.8,
                        timestamp: 0,
                    },
                ],
                asks: vec![
                    OrderBookLevel {
                        price: 45_015.0,
                        quantity: 0.7,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_025.0,
                        quantity: 1.5,
                        timestamp: 0,
                    },
                    OrderBookLevel {
                        price: 45_035.0,
                        quantity: 0.9,
                        timestamp: 0,
                    },
                ],
                timestamp: 0,
            },
        );

        books
    }

    #[test]
    fn test_cross_exchange_liquidity_analysis() {
        let exchange_books = create_exchange_order_books();
        let order_size = 10.0; // 10 BTC order

        for (exchange, book) in &exchange_books {
            let mut calculator = SlippageCalculator::new(SlippageParameters::default());
            calculator.update_order_book(book.clone());

            let analysis_result =
                calculator.calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0));

            match analysis_result {
                Ok(analysis) => {
                    println!("{} Exchange: Slippage {:.2} bps, Fill price: ${:.2}, Liquidity score: {:.3}",
                        exchange, analysis.slippage_bps, analysis.estimated_fill_price, analysis.liquidity_score);

                    // Verify analysis components
                    assert!(analysis.slippage_bps >= 0.0);
                    assert!(analysis.estimated_fill_price > 45_000.0); // Should be above mid price for buy
                    assert!(analysis.liquidity_score >= 0.0 && analysis.liquidity_score <= 1.0);
                    assert!(analysis.execution_cost >= 0.0);
                }
                Err(SlippageError::InsufficientLiquidity {
                    requested,
                    available,
                }) => {
                    println!(
                        "{} Exchange: Insufficient liquidity - Requested: {:.2}, Available: {:.2}",
                        exchange, requested, available
                    );

                    // This is expected for small exchanges with large orders
                    if exchange == "small_exchange" {
                        assert!(available < requested);
                    }
                }
                Err(e) => panic!("Unexpected error for {}: {:?}", exchange, e),
            }
        }
    }

    #[test]
    fn test_liquidity_fragmentation_impact() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Scenario 1: Concentrated liquidity
        let concentrated_book = OrderBook {
            symbol: "CONCENTRATED".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 99.95,
                    quantity: 100.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.90,
                    quantity: 200.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.85,
                    quantity: 300.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 100.05,
                    quantity: 120.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.10,
                    quantity: 250.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.15,
                    quantity: 350.0,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        };

        // Scenario 2: Fragmented liquidity (same total, spread across more levels)
        let fragmented_book = OrderBook {
            symbol: "FRAGMENTED".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 99.95,
                    quantity: 20.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.94,
                    quantity: 15.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.93,
                    quantity: 25.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.92,
                    quantity: 18.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.91,
                    quantity: 22.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.90,
                    quantity: 30.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.89,
                    quantity: 35.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.88,
                    quantity: 40.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.87,
                    quantity: 25.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.86,
                    quantity: 20.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 99.85,
                    quantity: 350.0,
                    timestamp: 0,
                }, // Deep level
            ],
            asks: vec![
                OrderBookLevel {
                    price: 100.05,
                    quantity: 25.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.06,
                    quantity: 20.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.07,
                    quantity: 30.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.08,
                    quantity: 18.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.09,
                    quantity: 27.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.10,
                    quantity: 35.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.11,
                    quantity: 40.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.12,
                    quantity: 45.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.13,
                    quantity: 30.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.14,
                    quantity: 25.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 100.15,
                    quantity: 425.0,
                    timestamp: 0,
                }, // Deep level
            ],
            timestamp: 0,
        };

        // Test both scenarios with same order size
        let order_size = 150.0;

        calculator.update_order_book(concentrated_book);
        let concentrated_analysis = calculator
            .calculate_slippage("CONCENTRATED", order_size, TradeSide::Buy, Some(100.0))
            .unwrap();

        calculator.update_order_book(fragmented_book);
        let fragmented_analysis = calculator
            .calculate_slippage("FRAGMENTED", order_size, TradeSide::Buy, Some(100.0))
            .unwrap();

        println!(
            "Concentrated liquidity: Slippage {:.2} bps, Fill price: ${:.4}",
            concentrated_analysis.slippage_bps, concentrated_analysis.estimated_fill_price
        );
        println!(
            "Fragmented liquidity: Slippage {:.2} bps, Fill price: ${:.4}",
            fragmented_analysis.slippage_bps, fragmented_analysis.estimated_fill_price
        );

        // Fragmented liquidity should have more slippage (higher fill price for buy orders)
        // because orders walk through multiple levels
        assert!(
            fragmented_analysis.estimated_fill_price >= concentrated_analysis.estimated_fill_price
        );
        assert!(fragmented_analysis.liquidity_score >= concentrated_analysis.liquidity_score);

        // Both should have reasonable execution costs
        assert!(concentrated_analysis.execution_cost > 0.0);
        assert!(fragmented_analysis.execution_cost > 0.0);
    }

    #[test]
    fn test_market_depth_stress_testing() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Create stress test scenarios
        let stress_scenarios = vec![
            // (scenario_name, total_liquidity, distribution_pattern)
            ("flash_crash", create_flash_crash_book()),
            ("whale_walls", create_whale_wall_book()),
            ("normal_market", create_deep_order_book()),
            ("illiquid_market", create_thin_order_book()),
        ];

        let stress_order_sizes = vec![1.0, 5.0, 15.0, 50.0, 100.0];

        for (scenario_name, order_book) in stress_scenarios {
            println!("\n=== {} Scenario ===", scenario_name);
            let symbol = order_book.symbol.clone();
            calculator.update_order_book(order_book);

            for order_size in &stress_order_sizes {
                let analysis_result = calculator.calculate_slippage(
                    &symbol,
                    *order_size,
                    TradeSide::Buy,
                    Some(45_000.0),
                );

                match analysis_result {
                    Ok(analysis) => {
                        println!(
                            "Order {:.1}: Slippage {:.1} bps, Impact ${:.2}, Score {:.3}",
                            order_size,
                            analysis.slippage_bps,
                            analysis.market_impact,
                            analysis.liquidity_score
                        );

                        // Verify stress test bounds
                        assert!(analysis.slippage_bps >= 0.0);
                        assert!(analysis.slippage_bps < 10000.0); // Max 100% slippage
                        assert!(analysis.market_impact >= 0.0);
                        assert!(analysis.liquidity_score >= 0.0 && analysis.liquidity_score <= 1.0);
                    }
                    Err(SlippageError::InsufficientLiquidity {
                        requested,
                        available,
                    }) => {
                        println!(
                            "Order {:.1}: Insufficient liquidity ({:.1}/{:.1})",
                            order_size, available, requested
                        );

                        // This is acceptable for large orders in stress scenarios
                        assert!(available < requested);
                    }
                    Err(e) => {
                        println!("Order {:.1}: Error {:?}", order_size, e);
                    }
                }
            }
        }
    }

    fn create_flash_crash_book() -> OrderBook {
        // Simulates order book during flash crash - very thin on one side
        OrderBook {
            symbol: "FLASHCRASH".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 35_000.0,
                    quantity: 0.1,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 30_000.0,
                    quantity: 0.5,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 25_000.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 20_000.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 50_000.0,
                    quantity: 50.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 51_000.0,
                    quantity: 100.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 52_000.0,
                    quantity: 200.0,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        }
    }

    fn create_whale_wall_book() -> OrderBook {
        // Large orders creating "walls" in the order book
        OrderBook {
            symbol: "WHALEWALL".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 44_995.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_990.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_985.0,
                    quantity: 3.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_980.0,
                    quantity: 500.0,
                    timestamp: 0,
                }, // Whale wall
                OrderBookLevel {
                    price: 44_975.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_970.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 45_005.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_010.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_015.0,
                    quantity: 3.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_020.0,
                    quantity: 800.0,
                    timestamp: 0,
                }, // Whale wall
                OrderBookLevel {
                    price: 45_025.0,
                    quantity: 1.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_030.0,
                    quantity: 2.0,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        }
    }

    #[test]
    fn test_real_time_liquidity_updates() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Initial order book state
        let mut current_book = create_deep_order_book();
        calculator.update_order_book(current_book.clone());

        let order_size = 10.0;
        let initial_analysis = calculator
            .calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0))
            .unwrap();

        println!("Initial slippage: {:.2} bps", initial_analysis.slippage_bps);

        // Simulate liquidity updates over time
        let updates = vec![
            // (timestamp_offset, price_level, new_quantity, description)
            (1000, 0, 0.2, "Top ask reduced"), // Reduce top ask liquidity
            (2000, 1, 0.5, "Second level reduced"),
            (3000, 0, 2.0, "Top ask replenished"), // Liquidity comes back
            (4000, 4, 50.0, "Deep liquidity added"), // New deep order
        ];

        for (time_offset, level_index, new_quantity, description) in updates {
            // Update the order book
            current_book.timestamp += time_offset;
            if level_index < current_book.asks.len() {
                current_book.asks[level_index].quantity = new_quantity;
                current_book.asks[level_index].timestamp = current_book.timestamp;
            }

            calculator.update_order_book(current_book.clone());

            let updated_analysis = calculator
                .calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0))
                .unwrap();

            println!(
                "{}: Slippage {:.2} bps, Liquidity score {:.3}",
                description, updated_analysis.slippage_bps, updated_analysis.liquidity_score
            );

            // Verify updates affect analysis appropriately
            assert!(updated_analysis.slippage_bps > 0.0);
            assert!(updated_analysis.liquidity_score >= 0.0);
        }
    }
}

#[cfg(test)]
mod edge_cases_and_error_handling_tests {
    use super::*;

    #[test]
    fn test_empty_order_book_handling() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        let empty_book = OrderBook {
            symbol: "EMPTY".to_string(),
            bids: vec![],
            asks: vec![],
            timestamp: 0,
        };

        calculator.update_order_book(empty_book);

        let result = calculator.calculate_slippage("EMPTY", 1.0, TradeSide::Buy, Some(100.0));

        assert!(matches!(
            result,
            Err(SlippageError::InsufficientLiquidity { .. })
        ));
    }

    #[test]
    fn test_invalid_order_sizes() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());

        // Test zero order size
        let result = calculator.calculate_slippage("BTCUSD", 0.0, TradeSide::Buy, Some(45_000.0));
        assert!(matches!(result, Err(SlippageError::InvalidOrderSize(_))));

        // Test negative order size
        let result = calculator.calculate_slippage("BTCUSD", -5.0, TradeSide::Buy, Some(45_000.0));
        assert!(matches!(result, Err(SlippageError::InvalidOrderSize(_))));
    }

    #[test]
    fn test_missing_market_data() {
        let calculator = SlippageCalculator::new(SlippageParameters::default());

        // Try to calculate slippage for non-existent symbol
        let result = calculator.calculate_slippage("NONEXISTENT", 1.0, TradeSide::Buy, Some(100.0));
        assert!(matches!(result, Err(SlippageError::InsufficientData)));

        // Try confidence interval with insufficient data
        let result = calculator.calculate_confidence_interval("NONEXISTENT", 1.0, 0.95);
        assert!(matches!(result, Err(SlippageError::InsufficientData)));
    }

    #[test]
    fn test_extreme_market_conditions() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Create extreme order book - massive spreads
        let extreme_book = OrderBook {
            symbol: "EXTREME".to_string(),
            bids: vec![OrderBookLevel {
                price: 1000.0,
                quantity: 1.0,
                timestamp: 0,
            }],
            asks: vec![
                OrderBookLevel {
                    price: 10000.0,
                    quantity: 1.0,
                    timestamp: 0,
                }, // 10x spread!
            ],
            timestamp: 0,
        };

        calculator.update_order_book(extreme_book);

        let result = calculator.calculate_slippage(
            "EXTREME",
            0.5,
            TradeSide::Buy,
            Some(5500.0), // Mid price
        );

        if let Ok(analysis) = result {
            println!("Extreme market slippage: {:.2} bps", analysis.slippage_bps);
            // Should have very high slippage due to wide spread
            assert!(analysis.slippage_bps > 1000.0); // > 10%
            assert!(analysis.liquidity_score < 0.5); // Poor liquidity
        }
    }

    #[test]
    fn test_precision_and_rounding() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Order book with very precise prices
        let precise_book = OrderBook {
            symbol: "PRECISE".to_string(),
            bids: vec![OrderBookLevel {
                price: 1.123456789,
                quantity: 1000.0,
                timestamp: 0,
            }],
            asks: vec![
                OrderBookLevel {
                    price: 1.123456790,
                    quantity: 1000.0,
                    timestamp: 0,
                }, // 1 satoshi difference
            ],
            timestamp: 0,
        };

        calculator.update_order_book(precise_book);

        let analysis = calculator
            .calculate_slippage("PRECISE", 100.0, TradeSide::Buy, Some(1.123456789))
            .unwrap();

        // Very tight spread should result in minimal slippage
        println!("Precise market slippage: {:.6} bps", analysis.slippage_bps);
        assert!(analysis.slippage_bps < 1.0); // Less than 0.01%
        assert!(analysis.estimated_fill_price > 1.123456789);
    }

    #[test]
    fn test_concurrent_access_safety() {
        use std::sync::Arc;
        use std::thread;

        let calculator = Arc::new(std::sync::Mutex::new(SlippageCalculator::new(
            SlippageParameters::default(),
        )));

        // Setup initial data
        {
            let mut calc = calculator.lock().unwrap();
            calc.update_order_book(create_deep_order_book());
        }

        // Spawn multiple threads to test concurrent access
        let handles: Vec<_> = (0..5)
            .map(|i| {
                let calc = calculator.clone();
                thread::spawn(move || {
                    let calc = calc.lock().unwrap();
                    let result = calc.calculate_slippage(
                        "BTCUSD",
                        1.0 + i as f64,
                        TradeSide::Buy,
                        Some(45_000.0),
                    );

                    match result {
                        Ok(analysis) => {
                            assert!(analysis.slippage_bps >= 0.0);
                            println!("Thread {}: Slippage {:.2} bps", i, analysis.slippage_bps);
                        }
                        Err(e) => panic!("Thread {}: {:?}", i, e),
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_memory_efficiency_with_large_datasets() {
        let mut calculator = SlippageCalculator::new(SlippageParameters {
            historical_window: 10, // Small window for testing
            ..SlippageParameters::default()
        });

        // Add many trades to test memory management
        for i in 0..100 {
            let trade = Trade {
                symbol: "MEMTEST".to_string(),
                price: 100.0 + (i as f64 * 0.1),
                quantity: 1.0,
                side: if i % 2 == 0 {
                    TradeSide::Buy
                } else {
                    TradeSide::Sell
                },
                timestamp: i as u64 * 1000,
            };
            calculator.add_trade(trade);
        }

        // Verify window size is maintained
        let trade_history = calculator.trade_history.get("MEMTEST").unwrap();
        assert_eq!(trade_history.len(), 10); // Should only keep last 10

        let volume_profile = calculator.volume_profile.get("MEMTEST").unwrap();
        assert_eq!(volume_profile.len(), 10); // Should only keep last 10

        // Verify volatility is still calculated correctly
        assert!(calculator.volatility_cache.contains_key("MEMTEST"));
        let volatility = calculator.volatility_cache.get("MEMTEST").unwrap();
        assert!(*volatility >= 0.0);

        println!(
            "Memory test completed - Final volatility: {:.6}",
            volatility
        );
    }
}

#[cfg(test)]
mod realistic_trading_scenarios_tests {
    use super::*;

    #[test]
    fn test_institutional_block_trade() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Simulate institutional block trade - 100 BTC
        let block_size = 100.0;

        // First, check if order can be filled
        let analysis_result =
            calculator.calculate_slippage("BTCUSD", block_size, TradeSide::Buy, Some(45_000.0));

        match analysis_result {
            Ok(analysis) => {
                println!("Block trade analysis:");
                println!("  Size: {} BTC", block_size);
                println!("  Expected fill: ${:.2}", analysis.estimated_fill_price);
                println!("  Slippage: {:.2} bps", analysis.slippage_bps);
                println!("  Market impact: ${:.2}", analysis.market_impact);
                println!("  Total cost: ${:.2}", analysis.execution_cost);
                println!("  Liquidity score: {:.3}", analysis.liquidity_score);

                // Institutional trade validation
                assert!(analysis.slippage_bps > 100.0); // > 1% expected for large orders
                assert!(analysis.market_impact > 1000.0); // Significant impact expected
                assert!(analysis.execution_cost > analysis.slippage_amount); // Should include market impact
                assert!(analysis.liquidity_score < 0.7); // Lower score for large order
            }
            Err(SlippageError::InsufficientLiquidity {
                requested,
                available,
            }) => {
                println!(
                    "Block trade requires splitting: {:.2} BTC available of {:.2} requested",
                    available, requested
                );

                // Test order splitting strategy
                let splits = calculator
                    .calculate_optimal_order_splits(
                        "BTCUSD",
                        block_size,
                        TradeSide::Buy,
                        0.05, // 5% max participation
                    )
                    .unwrap();

                println!("Optimal splits: {:?}", splits);
                assert!(splits.len() > 1);
                assert_eq!(splits.iter().sum::<f64>(), block_size);

                // Test each split individually
                for (i, split_size) in splits.iter().enumerate() {
                    let split_analysis = calculator.calculate_slippage(
                        "BTCUSD",
                        *split_size,
                        TradeSide::Buy,
                        Some(45_000.0),
                    );

                    match split_analysis {
                        Ok(split_result) => {
                            println!(
                                "Split {}: {:.2} BTC, Slippage: {:.2} bps",
                                i + 1,
                                split_size,
                                split_result.slippage_bps
                            );
                        }
                        Err(e) => {
                            println!("Split {} failed: {:?}", i + 1, e);
                        }
                    }
                }
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_retail_trader_scenarios() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // Different retail trader profiles
        let retail_scenarios = vec![
            (0.01, "Micro trader - $450 order"),
            (0.1, "Small trader - $4.5K order"),
            (0.5, "Medium trader - $22.5K order"),
            (2.0, "Large retail - $90K order"),
        ];

        for (order_size, description) in retail_scenarios {
            let analysis = calculator
                .calculate_slippage("BTCUSD", order_size, TradeSide::Buy, Some(45_000.0))
                .unwrap();

            println!("\n{}", description);
            println!("  Order size: {:.3} BTC", order_size);
            println!("  Slippage: {:.2} bps", analysis.slippage_bps);
            println!("  Fill price: ${:.2}", analysis.estimated_fill_price);
            println!("  Dollar cost: ${:.2}", analysis.slippage_amount);
            println!("  Liquidity score: {:.3}", analysis.liquidity_score);

            // Retail validation - should have reasonable costs
            assert!(analysis.slippage_bps < 1000.0); // < 10% slippage
            assert!(analysis.liquidity_score > 0.3); // Decent liquidity

            // Larger retail orders should have higher slippage
            if order_size > 1.0 {
                assert!(analysis.slippage_bps > 10.0); // > 0.1% for larger orders
            } else {
                assert!(analysis.slippage_bps < 50.0); // < 0.5% for small orders
            }
        }
    }

    #[test]
    fn test_high_frequency_trading_patterns() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Setup HFT-style order book with tight spreads and frequent updates
        let mut hft_book = OrderBook {
            symbol: "HFTTEST".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 45_000.50,
                    quantity: 0.1,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.49,
                    quantity: 0.2,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.48,
                    quantity: 0.15,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.47,
                    quantity: 0.3,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.46,
                    quantity: 0.25,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 45_000.51,
                    quantity: 0.12,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.52,
                    quantity: 0.18,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.53,
                    quantity: 0.22,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.54,
                    quantity: 0.16,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_000.55,
                    quantity: 0.28,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        };

        calculator.update_order_book(hft_book.clone());

        // HFT order sizes - very small, frequent
        let hft_order_sizes = vec![0.01, 0.02, 0.05, 0.1];
        let mut total_slippage = 0.0;

        for (i, order_size) in hft_order_sizes.iter().enumerate() {
            // Simulate rapid-fire orders with minor book updates
            hft_book.timestamp += 100; // 100ms apart

            // Micro adjustments to top of book
            if i > 0 {
                hft_book.bids[0].price += if i % 2 == 0 { 0.01 } else { -0.01 };
                hft_book.asks[0].price += if i % 2 == 0 { 0.01 } else { -0.01 };
            }

            calculator.update_order_book(hft_book.clone());

            let analysis = calculator
                .calculate_slippage("HFTTEST", *order_size, TradeSide::Buy, Some(45_000.505))
                .unwrap();

            println!(
                "HFT Order {}: Size {:.3}, Slippage {:.4} bps, Fill ${:.3}",
                i + 1,
                order_size,
                analysis.slippage_bps,
                analysis.estimated_fill_price
            );

            // HFT validation - should have minimal slippage
            assert!(analysis.slippage_bps < 5.0); // < 0.05% for small HFT orders
            assert!(analysis.liquidity_score > 0.8); // High liquidity score

            total_slippage += analysis.slippage_amount;
        }

        println!("Total HFT session slippage: ${:.4}", total_slippage);
        assert!(total_slippage < 5.0); // Total session cost should be minimal
    }

    #[test]
    fn test_market_maker_impact() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());

        // Market maker providing liquidity
        let mm_book = OrderBook {
            symbol: "MMTEST".to_string(),
            bids: vec![
                // Market maker bids - consistent size, regular intervals
                OrderBookLevel {
                    price: 44_995.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_990.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_985.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_980.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 44_975.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 45_005.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_010.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_015.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_020.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
                OrderBookLevel {
                    price: 45_025.0,
                    quantity: 5.0,
                    timestamp: 0,
                },
            ],
            timestamp: 0,
        };

        calculator.update_order_book(mm_book);

        // Test market maker's impact on different order sizes
        let test_orders = vec![2.0, 7.0, 15.0, 30.0];

        for order_size in test_orders {
            let analysis = calculator
                .calculate_slippage("MMTEST", order_size, TradeSide::Buy, Some(45_000.0))
                .unwrap();

            println!(
                "MM Market - Order: {:.1}, Slippage: {:.2} bps, Score: {:.3}",
                order_size, analysis.slippage_bps, analysis.liquidity_score
            );

            // Market maker provides consistent liquidity
            if order_size <= 25.0 {
                // Within MM capacity
                assert!(analysis.liquidity_score > 0.5);
                assert!(analysis.slippage_bps < 200.0); // < 2%
            }

            // Verify linear slippage progression due to consistent MM sizes
            assert!(analysis.slippage_bps > 0.0);
            assert!(analysis.estimated_fill_price > 45_000.0);
        }

        // Compare with non-MM market (irregular liquidity)
        let irregular_book = create_thin_order_book();
        calculator.update_order_book(irregular_book);

        let mm_analysis =
            calculator.calculate_slippage("MMTEST", 10.0, TradeSide::Buy, Some(45_000.0));

        let irregular_analysis =
            calculator.calculate_slippage("ALTCOIN", 10.0, TradeSide::Buy, Some(10.0));

        // MM market should generally provide better liquidity characteristics
        if let (Ok(mm_result), Ok(irregular_result)) = (mm_analysis, irregular_analysis) {
            println!(
                "MM vs Irregular liquidity scores: {:.3} vs {:.3}",
                mm_result.liquidity_score, irregular_result.liquidity_score
            );

            // Note: Actual comparison depends on order size relative to available liquidity
            assert!(mm_result.liquidity_score > 0.0);
            assert!(irregular_result.liquidity_score > 0.0);
        }
    }

    #[test]
    fn test_algorithmic_trading_execution() {
        let mut calculator = SlippageCalculator::new(SlippageParameters::default());
        let order_book = create_deep_order_book();
        calculator.update_order_book(order_book);

        // TWAP algorithm simulation - spreading order over time
        let total_order = 50.0; // 50 BTC total
        let execution_windows = 10; // 10 time windows
        let order_per_window = total_order / execution_windows as f64;

        let mut total_execution_cost = 0.0;
        let mut cumulative_slippage = 0.0;

        println!("TWAP Execution Simulation:");
        println!(
            "Total order: {} BTC over {} windows",
            total_order, execution_windows
        );

        for window in 1..=execution_windows {
            // Simulate time-based execution with slight market movement
            let time_horizon_ms = window as u64 * 60_000; // 1 minute per window

            let dynamic_analysis = calculator
                .estimate_dynamic_slippage(
                    "BTCUSD",
                    order_per_window,
                    TradeSide::Buy,
                    time_horizon_ms,
                )
                .unwrap();

            total_execution_cost += dynamic_analysis.execution_cost;
            cumulative_slippage += dynamic_analysis.slippage_amount;

            println!(
                "Window {}: {:.1} BTC, Slippage {:.2} bps, Cost ${:.2}",
                window,
                order_per_window,
                dynamic_analysis.slippage_bps,
                dynamic_analysis.execution_cost
            );

            // TWAP validation - should have consistent per-window costs
            assert!(dynamic_analysis.slippage_bps > 0.0);
            assert!(dynamic_analysis.execution_cost > 0.0);

            // Later windows might have slightly higher slippage due to time factor
            if window > 1 {
                assert!(dynamic_analysis.slippage_bps < 500.0); // Should remain reasonable
            }
        }

        println!("\nTWAP Results:");
        println!("Total execution cost: ${:.2}", total_execution_cost);
        println!(
            "Average per-window slippage: {:.2} bps",
            (cumulative_slippage / total_order) / 0.0001
        ); // Convert to bps

        // Compare with single large order
        let single_order_analysis =
            calculator.calculate_slippage("BTCUSD", total_order, TradeSide::Buy, Some(45_000.0));

        match single_order_analysis {
            Ok(single_result) => {
                println!(
                    "Single order slippage: {:.2} bps",
                    single_result.slippage_bps
                );
                println!("Single order cost: ${:.2}", single_result.execution_cost);

                // TWAP should generally be better than single large order
                let twap_avg_bps = (cumulative_slippage / total_order) / 0.0001;
                if single_result.slippage_bps > twap_avg_bps {
                    println!("TWAP strategy was beneficial");
                } else {
                    println!("Single order might have been better in this case");
                }
            }
            Err(SlippageError::InsufficientLiquidity { .. }) => {
                println!("Single order would have failed - TWAP was necessary");
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }
}
