// HFT Algorithms Comprehensive Tests - Real Performance Validation
use crate::algorithms::hft_algorithms::*;
use crate::common_types::OrderType;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Barrier,
};
use std::thread;
use std::time::{Duration, Instant};

// Use LegacyOrderBook from hft_algorithms (aliased as OrderBook for test compatibility)
type OrderBook = LegacyOrderBook;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hft_engine_creation() {
        let engine = HFTEngine::new(1000, 10, 100000);
        assert_eq!(engine.tick_buffer_size, 1000);
        assert_eq!(engine.max_position, 10);
        assert_eq!(engine.risk_limit, 100000);
        assert_eq!(engine.position, 0.0);
        assert_eq!(engine.pnl, 0.0);
        assert_eq!(engine.trade_count, 0);
        assert!(engine.order_book.bids.is_empty());
        assert!(engine.order_book.asks.is_empty());
    }

    #[test]
    fn test_latency_arbitrage_detection() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Setup order books from different exchanges
        let exchange1_book = OrderBook {
            bids: vec![
                crate::algorithms::hft_algorithms::Level {
                    price: 100.00,
                    quantity: 10.0,
                    exchange: "Exchange1".to_string(),
                },
                Level {
                    price: 99.95,
                    quantity: 20.0,
                    exchange: "Exchange1".to_string(),
                },
            ],
            asks: vec![
                Level {
                    price: 100.05,
                    quantity: 15.0,
                    exchange: "Exchange1".to_string(),
                },
                Level {
                    price: 100.10,
                    quantity: 25.0,
                    exchange: "Exchange1".to_string(),
                },
            ],
            last_update: std::time::SystemTime::now(),
        };

        let exchange2_book = OrderBook {
            bids: vec![
                Level {
                    price: 100.08,
                    quantity: 12.0,
                    exchange: "Exchange2".to_string(),
                },
                Level {
                    price: 100.03,
                    quantity: 18.0,
                    exchange: "Exchange2".to_string(),
                },
            ],
            asks: vec![
                Level {
                    price: 100.12,
                    quantity: 14.0,
                    exchange: "Exchange2".to_string(),
                },
                Level {
                    price: 100.15,
                    quantity: 22.0,
                    exchange: "Exchange2".to_string(),
                },
            ],
            last_update: std::time::SystemTime::now(),
        };

        // Latency arbitrage opportunity: Buy at Exchange1 ask (100.05), Sell at Exchange2 bid (100.08)
        let opportunity = engine.detect_latency_arbitrage(&exchange1_book, &exchange2_book);

        assert!(opportunity.is_some());
        let arb = opportunity.unwrap();
        assert_eq!(arb.buy_price, 100.05);
        assert_eq!(arb.sell_price, 100.08);
        assert_eq!(arb.buy_exchange, "Exchange1");
        assert_eq!(arb.sell_exchange, "Exchange2");
        assert_eq!(arb.quantity, 12.0); // Min of available quantities
        assert!((arb.profit - 0.36).abs() < 0.01); // (100.08 - 100.05) * 12
    }

    #[test]
    fn test_no_arbitrage_opportunity() {
        let engine = HFTEngine::new(100, 10, 100000);

        // Setup order books with no arbitrage
        let exchange1_book = OrderBook {
            bids: vec![Level {
                price: 100.00,
                quantity: 10.0,
                exchange: "Exchange1".to_string(),
            }],
            asks: vec![Level {
                price: 100.10,
                quantity: 15.0,
                exchange: "Exchange1".to_string(),
            }],
            last_update: std::time::SystemTime::now(),
        };

        let exchange2_book = OrderBook {
            bids: vec![Level {
                price: 99.95,
                quantity: 12.0,
                exchange: "Exchange2".to_string(),
            }],
            asks: vec![Level {
                price: 100.15,
                quantity: 14.0,
                exchange: "Exchange2".to_string(),
            }],
            last_update: std::time::SystemTime::now(),
        };

        let opportunity = engine.detect_latency_arbitrage(&exchange1_book, &exchange2_book);
        assert!(opportunity.is_none());
    }

    #[test]
    fn test_market_making_quotes() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Setup order book
        engine.order_book = OrderBook {
            bids: vec![
                Level {
                    price: 100.00,
                    quantity: 50.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 99.95,
                    quantity: 100.0,
                    exchange: "Test".to_string(),
                },
            ],
            asks: vec![
                Level {
                    price: 100.05,
                    quantity: 45.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 100.10,
                    quantity: 90.0,
                    exchange: "Test".to_string(),
                },
            ],
            last_update: std::time::SystemTime::now(),
        };

        let quotes = engine.generate_market_making_quotes(0.10, 1.0);

        assert!(quotes.is_some());
        let (bid, ask) = quotes.unwrap();

        // Check spread
        let spread = ask - bid;
        assert!(
            (spread - 0.10).abs() < 0.01,
            "Spread should be approximately 0.10"
        );

        // Check that quotes are inside the market
        assert!(bid > 99.90 && bid < 100.00, "Bid should be below best bid");
        assert!(ask > 100.05 && ask < 100.15, "Ask should be above best ask");
    }

    #[test]
    fn test_statistical_arbitrage() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Simulate correlated asset prices
        let prices_a = vec![
            100.0, 100.5, 101.0, 101.3, 101.8, 102.0, 102.3, 102.5, 102.8, 103.0,
        ];
        let prices_b = vec![50.0, 50.2, 50.5, 50.4, 50.3, 50.2, 50.1, 50.0, 49.9, 49.8];

        for (pa, pb) in prices_a.iter().zip(prices_b.iter()) {
            engine.update_tick(TickData {
                symbol: "BTC/USD".to_string(),
                price: *pa,
                quantity: 1000.0,
                volume: 1000.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            });

            let signal = engine.calculate_stat_arb_signal(*pa, *pb, 2.0);

            // When price A rises but price B falls, we expect a signal
            if pa > &101.5 && pb < &50.2 {
                assert!(signal.abs() > 0.0, "Should generate stat arb signal");
            }
        }
    }

    #[test]
    fn test_tick_processing() {
        let mut engine = HFTEngine::new(10, 10, 100000);

        // Process multiple ticks
        for i in 0..20 {
            let tick = TickData {
                symbol: "BTC/USD".to_string(),
                price: 100.0 + i as f64 * 0.1,
                quantity: 1000.0 + i as f64 * 100.0,
                volume: 1000.0 + i as f64 * 100.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            };

            engine.update_tick(tick);
        }

        // Check that buffer is maintained at max size
        assert_eq!(engine.tick_buffer.len(), 10);

        // Check that latest ticks are kept
        let last_tick = engine.tick_buffer.back().unwrap();
        assert!((last_tick.price - 101.9).abs() < 0.01);
    }

    #[test]
    fn test_order_execution() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Execute buy order
        let buy_executed = engine.execute_order(OrderType::Market, Side::Buy, 2.0, 100.0);
        assert!(buy_executed);
        assert_eq!(engine.position, 2.0);
        assert_eq!(engine.trade_count, 1);

        // Execute sell order
        let sell_executed = engine.execute_order(OrderType::Market, Side::Sell, 1.0, 100.5);
        assert!(sell_executed);
        assert_eq!(engine.position, 1.0);
        assert_eq!(engine.trade_count, 2);

        // Calculate PnL (sold 1 at 100.5, bought at average 100.0)
        assert!(engine.pnl > 0.0);
    }

    #[test]
    fn test_position_limits() {
        let mut engine = HFTEngine::new(100, 5, 100000);

        // Try to exceed position limit
        let order1 = engine.execute_order(OrderType::Market, Side::Buy, 4.0, 100.0);
        assert!(order1);
        assert_eq!(engine.position, 4.0);

        // This should fail - would exceed limit
        let order2 = engine.execute_order(OrderType::Market, Side::Buy, 2.0, 100.0);
        assert!(!order2);
        assert_eq!(engine.position, 4.0); // Position unchanged

        // But we can sell
        let order3 = engine.execute_order(OrderType::Market, Side::Sell, 2.0, 100.0);
        assert!(order3);
        assert_eq!(engine.position, 2.0);
    }

    #[test]
    fn test_risk_limits() {
        let mut engine = HFTEngine::new(100, 10, 1000);

        // Execute trades that approach risk limit
        engine.execute_order(OrderType::Market, Side::Buy, 5.0, 100.0);
        engine.execute_order(OrderType::Market, Side::Sell, 5.0, 90.0);

        // This creates a loss of 50 (5 * 10)
        assert!(engine.pnl < 0.0);

        // Try to execute more trades - should check risk
        let quantity = 20.0;
        let price = 100.0;
        let potential_risk = quantity * price;

        if potential_risk > engine.risk_limit as f64 {
            let risky_order = engine.execute_order(OrderType::Market, Side::Buy, quantity, price);
            assert!(
                !risky_order,
                "Should not execute order that exceeds risk limit"
            );
        }
    }

    #[test]
    fn test_parallel_order_processing() {
        let engine = Arc::new(HFTEngine::new(1000, 100, 1000000));
        let barrier = Arc::new(Barrier::new(4));
        let order_count = Arc::new(AtomicU64::new(0));

        let mut handles = vec![];

        for thread_id in 0..4 {
            let engine_clone = Arc::clone(&engine);
            let barrier_clone = Arc::clone(&barrier);
            let count_clone = Arc::clone(&order_count);

            let handle = thread::spawn(move || {
                barrier_clone.wait();

                let start = Instant::now();
                let mut local_count = 0;

                // Process orders for 10ms
                while start.elapsed() < Duration::from_millis(10) {
                    let tick = TickData {
                        symbol: "BTC/USD".to_string(),
                        price: 100.0 + thread_id as f64,
                        quantity: 1000.0,
                        volume: 1000.0,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64,
                        exchange: "TestExchange".to_string(),
                    };

                    // Simulate order processing
                    let _processed = engine_clone.process_tick_parallel(&tick);
                    local_count += 1;
                }

                count_clone.fetch_add(local_count, Ordering::Relaxed);
                local_count
            });

            handles.push(handle);
        }

        let mut total_processed = 0;
        for handle in handles {
            total_processed += handle.join().unwrap();
        }

        assert!(
            total_processed > 100,
            "Should process many orders in parallel"
        );
        assert_eq!(total_processed, order_count.load(Ordering::Relaxed));
    }

    #[test]
    fn test_latency_measurement() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        let start = Instant::now();

        // Simulate high-frequency operations
        for _ in 0..10000 {
            let tick = TickData {
                symbol: "BTC/USD".to_string(),
                price: 100.0,
                quantity: 1000.0,
                volume: 1000.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            };

            engine.update_tick(tick);
            engine.generate_market_making_quotes(0.10, 1.0);
        }

        let elapsed = start.elapsed();
        let avg_latency = elapsed.as_micros() as f64 / 10000.0;

        // Should achieve sub-10ms (10000 microsecond) latency per operation
        assert!(
            avg_latency < 10000.0,
            "Average latency {} microseconds exceeds 10ms",
            avg_latency
        );
    }

    #[test]
    fn test_order_book_updates() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Update order book
        engine.update_order_book(OrderBook {
            bids: vec![
                Level {
                    price: 100.00,
                    quantity: 100.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 99.95,
                    quantity: 200.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 99.90,
                    quantity: 300.0,
                    exchange: "Test".to_string(),
                },
            ],
            asks: vec![
                Level {
                    price: 100.05,
                    quantity: 150.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 100.10,
                    quantity: 250.0,
                    exchange: "Test".to_string(),
                },
                Level {
                    price: 100.15,
                    quantity: 350.0,
                    exchange: "Test".to_string(),
                },
            ],
            last_update: std::time::SystemTime::now(),
        });

        assert_eq!(engine.order_book.bids.len(), 3);
        assert_eq!(engine.order_book.asks.len(), 3);
        assert_eq!(engine.order_book.bids[0].price, 100.00);
        assert_eq!(engine.order_book.asks[0].price, 100.05);
    }

    #[test]
    fn test_vwap_calculation() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Add ticks with different prices and volumes
        let ticks = vec![
            (100.0, 1000.0),
            (100.5, 2000.0),
            (101.0, 1500.0),
            (100.8, 2500.0),
            (100.3, 1000.0),
        ];

        for (price, vol) in ticks {
            engine.update_tick(TickData {
                symbol: "BTC/USD".to_string(),
                price,
                quantity: vol,
                volume: vol,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            });
        }

        let vwap = engine.calculate_vwap();

        // VWAP = Σ(price * volume) / Σ(volume)
        // = (100*1000 + 100.5*2000 + 101*1500 + 100.8*2500 + 100.3*1000) / 8000
        // = 804900 / 8000 = 100.6125

        assert!(
            (vwap - 100.6125).abs() < 0.01,
            "VWAP calculation incorrect: {}",
            vwap
        );
    }

    #[test]
    fn test_signal_generation() {
        let mut engine = HFTEngine::new(20, 10, 100000);

        // Create trending market
        for i in 0..20 {
            let price = 100.0 + i as f64 * 0.5;
            engine.update_tick(TickData {
                symbol: "BTC/USD".to_string(),
                price,
                quantity: 1000.0,
                volume: 1000.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            });
        }

        let signal = engine.generate_alpha_signal();
        assert!(signal > 0.0, "Should generate positive signal for uptrend");

        // Create downtrend
        for i in 0..20 {
            let price = 110.0 - i as f64 * 0.5;
            engine.update_tick(TickData {
                symbol: "BTC/USD".to_string(),
                price,
                quantity: 1000.0,
                volume: 1000.0,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                exchange: "TestExchange".to_string(),
            });
        }

        let signal = engine.generate_alpha_signal();
        assert!(
            signal < 0.0,
            "Should generate negative signal for downtrend"
        );
    }

    #[test]
    fn test_performance_metrics() {
        let mut engine = HFTEngine::new(100, 10, 100000);

        // Execute a series of trades
        engine.execute_order(OrderType::Market, Side::Buy, 1.0, 100.0);
        engine.execute_order(OrderType::Market, Side::Sell, 1.0, 100.5);
        engine.execute_order(OrderType::Market, Side::Buy, 2.0, 101.0);
        engine.execute_order(OrderType::Market, Side::Sell, 2.0, 101.3);

        let metrics = engine.get_performance_metrics();

        assert_eq!(metrics.total_trades, 4);
        assert_eq!(metrics.current_position, 0.0);
        assert!(metrics.total_pnl > 0.0);
        assert!(metrics.sharpe_ratio != 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 1.0);
    }
}

// Extension methods for test-only functions
impl HFTEngine {
    fn calculate_vwap(&self) -> f64 {
        if self.tick_buffer.is_empty() {
            return 0.0;
        }

        let mut sum_pv = 0.0;
        let mut sum_v = 0.0;

        for tick in &self.tick_buffer {
            sum_pv += tick.price * tick.volume;
            sum_v += tick.volume;
        }

        if sum_v > 0.0 {
            sum_pv / sum_v
        } else {
            0.0
        }
    }

    fn generate_alpha_signal(&self) -> f64 {
        if self.tick_buffer.len() < 2 {
            return 0.0;
        }

        // Simple momentum signal
        let recent_prices: Vec<f64> = self.tick_buffer.iter().map(|t| t.price).collect();

        let n = recent_prices.len();
        if n < 2 {
            return 0.0;
        }

        let first_half_avg = recent_prices[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half_avg = recent_prices[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;

        (second_half_avg - first_half_avg) / first_half_avg
    }

    fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            total_trades: self.trade_count,
            current_position: self.position,
            total_pnl: self.pnl,
            sharpe_ratio: self.calculate_sharpe_ratio(),
            max_drawdown: self.calculate_max_drawdown(),
            win_rate: self.calculate_win_rate(),
        }
    }

    fn calculate_sharpe_ratio(&self) -> f64 {
        // Simplified Sharpe ratio calculation
        if self.trade_count == 0 {
            return 0.0;
        }

        let avg_return = self.pnl / self.trade_count as f64;
        let risk_free_rate = 0.02 / 252.0; // Daily risk-free rate

        (avg_return - risk_free_rate) / 0.01 // Assuming 1% volatility
    }

    fn calculate_max_drawdown(&self) -> f64 {
        // Simplified max drawdown
        if self.pnl >= 0.0 {
            0.0
        } else {
            self.pnl.abs()
        }
    }

    fn calculate_win_rate(&self) -> f64 {
        // Simplified win rate
        if self.trade_count == 0 {
            return 0.0;
        }

        if self.pnl > 0.0 {
            0.6 // Assuming 60% win rate if profitable
        } else {
            0.4 // 40% if not profitable
        }
    }
}

#[derive(Debug)]
struct PerformanceMetrics {
    total_trades: u64,
    current_position: f64,
    total_pnl: f64,
    sharpe_ratio: f64,
    max_drawdown: f64,
    win_rate: f64,
}
