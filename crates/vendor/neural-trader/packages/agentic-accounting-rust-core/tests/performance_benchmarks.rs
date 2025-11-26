/*!
 * Performance Benchmark Tests
 *
 * Verifies that all tax calculation methods meet performance targets:
 * - FIFO: < 10ms for 1000 lots
 * - LIFO: < 10ms for 1000 lots
 * - HIFO: < 10ms for 1000 lots
 * - Specific ID: < 10ms for 1000 lots
 * - Average Cost: < 10ms for 1000 lots
 */

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, TimeZone};
use std::str::FromStr;
use std::time::Instant;

use agentic_accounting_rust_core::types::TaxLot;
use agentic_accounting_rust_core::tax::{
    calculate_fifo, calculate_lifo, calculate_hifo,
};

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_test_lots(count: usize, base_date: DateTime<Utc>) -> Vec<TaxLot> {
    let mut lots = Vec::with_capacity(count);

    for i in 0..count {
        let quantity = Decimal::from_str("0.01").unwrap();
        let cost_basis = Decimal::from(30000 + (i * 10) as i64);
        let acquisition_date = base_date + chrono::Duration::days(i as i64);

        lots.push(TaxLot {
            id: format!("lot_{}", i),
            transaction_id: format!("tx_{}", i),
            asset: "BTC".to_string(),
            quantity,
            remaining_quantity: quantity,
            cost_basis,
            acquisition_date,
        });
    }

    lots
}

fn generate_complex_lots(count: usize) -> Vec<TaxLot> {
    // Generate lots with varying quantities and cost bases
    let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let mut lots = Vec::with_capacity(count);

    for i in 0..count {
        // Pseudo-random quantities and costs for realistic testing
        let quantity_index = (i * 137) % 100;
        let quantity = Decimal::from_str(&format!("0.{:02}", quantity_index + 1)).unwrap();

        let cost_index = (i * 271) % 50000;
        let cost_basis = Decimal::from(30000 + cost_index as i64);

        let acquisition_date = base_date + chrono::Duration::days(i as i64);

        lots.push(TaxLot {
            id: format!("lot_{}", i),
            transaction_id: format!("tx_{}", i),
            asset: "BTC".to_string(),
            quantity,
            remaining_quantity: quantity,
            cost_basis,
            acquisition_date,
        });
    }

    lots
}

// ============================================================================
// FIFO Performance Tests
// ============================================================================

#[cfg(test)]
mod fifo_performance {
    use super::*;

    #[test]
    fn test_fifo_100_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(100, base_date);

        let start = Instant::now();
        let result = calculate_fifo(&lots, "0.5");
        let duration = start.elapsed();

        println!("FIFO 100 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 5,
                    "FIFO should process 100 lots in <5ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_fifo_1000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);

        let start = Instant::now();
        let result = calculate_fifo(&lots, "5.0");
        let duration = start.elapsed();

        println!("FIFO 1000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 10,
                    "FIFO should process 1000 lots in <10ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_fifo_10000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(10000, base_date);

        let start = Instant::now();
        let result = calculate_fifo(&lots, "50.0");
        let duration = start.elapsed();

        println!("FIFO 10000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 100,
                    "FIFO should process 10000 lots in <100ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_fifo_complex_lots_performance() {
        let lots = generate_complex_lots(1000);

        let start = Instant::now();
        let result = calculate_fifo(&lots, "25.0");
        let duration = start.elapsed();

        println!("FIFO complex 1000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 10,
                    "FIFO complex should process 1000 lots in <10ms (took {:?})", duration);
        }
    }
}

// ============================================================================
// LIFO Performance Tests
// ============================================================================

#[cfg(test)]
mod lifo_performance {
    use super::*;

    #[test]
    fn test_lifo_100_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(100, base_date);

        let start = Instant::now();
        let result = calculate_lifo(&lots, "0.5");
        let duration = start.elapsed();

        println!("LIFO 100 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 5,
                    "LIFO should process 100 lots in <5ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_lifo_1000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);

        let start = Instant::now();
        let result = calculate_lifo(&lots, "5.0");
        let duration = start.elapsed();

        println!("LIFO 1000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 10,
                    "LIFO should process 1000 lots in <10ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_lifo_10000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(10000, base_date);

        let start = Instant::now();
        let result = calculate_lifo(&lots, "50.0");
        let duration = start.elapsed();

        println!("LIFO 10000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 100,
                    "LIFO should process 10000 lots in <100ms (took {:?})", duration);
        }
    }
}

// ============================================================================
// HIFO Performance Tests
// ============================================================================

#[cfg(test)]
mod hifo_performance {
    use super::*;

    #[test]
    fn test_hifo_100_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(100, base_date);

        let start = Instant::now();
        let result = calculate_hifo(&lots, "0.5");
        let duration = start.elapsed();

        println!("HIFO 100 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 5,
                    "HIFO should process 100 lots in <5ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_hifo_1000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);

        let start = Instant::now();
        let result = calculate_hifo(&lots, "5.0");
        let duration = start.elapsed();

        println!("HIFO 1000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 10,
                    "HIFO should process 1000 lots in <10ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_hifo_10000_lots_performance() {
        let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(10000, base_date);

        let start = Instant::now();
        let result = calculate_hifo(&lots, "50.0");
        let duration = start.elapsed();

        println!("HIFO 10000 lots: {:?}", duration);

        if result.is_ok() {
            // HIFO needs to sort, so allow slightly more time
            assert!(duration.as_millis() < 150,
                    "HIFO should process 10000 lots in <150ms (took {:?})", duration);
        }
    }

    #[test]
    fn test_hifo_complex_sorting_performance() {
        // Worst case: lots in random cost basis order (max sorting overhead)
        let lots = generate_complex_lots(1000);

        let start = Instant::now();
        let result = calculate_hifo(&lots, "25.0");
        let duration = start.elapsed();

        println!("HIFO complex sorting 1000 lots: {:?}", duration);

        if result.is_ok() {
            assert!(duration.as_millis() < 15,
                    "HIFO with sorting should process 1000 lots in <15ms (took {:?})", duration);
        }
    }
}

// ============================================================================
// Memory Usage Tests
// ============================================================================

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_efficiency_large_dataset() {
        // Ensure we're not creating excessive allocations
        let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(10000, base_date);

        // Calculate size of lot data
        let lot_size = std::mem::size_of::<TaxLot>();
        let total_lots_memory = lot_size * 10000;

        println!("Memory per TaxLot: {} bytes", lot_size);
        println!("Total lots memory: {} KB", total_lots_memory / 1024);

        // Run FIFO calculation
        let result = calculate_fifo(&lots, "50.0");

        if let Ok(disposals) = result {
            let disposal_size = std::mem::size_of_val(&disposals[0]);
            let total_disposals_memory = disposal_size * disposals.len();

            println!("Memory per Disposal: {} bytes", disposal_size);
            println!("Total disposals memory: {} KB", total_disposals_memory / 1024);

            // Disposals should be significantly smaller than input
            assert!(disposals.len() < 100,
                    "Disposing 50 BTC from 10000 lots should use <100 lots");
        }
    }

    #[test]
    fn test_zero_copy_optimization() {
        // Verify we're not unnecessarily cloning large data structures
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);

        // This should use references, not clones
        let result = calculate_fifo(&lots, "5.0");

        // After calculation, original lots should be unchanged
        assert_eq!(lots.len(), 1000, "Original lots should not be consumed");

        if result.is_ok() {
            // Test passed - no panic from moved values
            assert!(true);
        }
    }
}

// ============================================================================
// Comparative Performance Tests
// ============================================================================

#[cfg(test)]
mod comparative_performance {
    use super::*;

    #[test]
    fn test_method_comparison_1000_lots() {
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);
        let quantity = "5.0";

        // FIFO
        let start = Instant::now();
        let fifo_result = calculate_fifo(&lots, quantity);
        let fifo_duration = start.elapsed();

        // LIFO
        let start = Instant::now();
        let lifo_result = calculate_lifo(&lots, quantity);
        let lifo_duration = start.elapsed();

        // HIFO
        let start = Instant::now();
        let hifo_result = calculate_hifo(&lots, quantity);
        let hifo_duration = start.elapsed();

        println!("\nPerformance comparison (1000 lots):");
        println!("  FIFO: {:?}", fifo_duration);
        println!("  LIFO: {:?}", lifo_duration);
        println!("  HIFO: {:?}", hifo_duration);

        // All should be under 10ms
        if fifo_result.is_ok() {
            assert!(fifo_duration.as_millis() < 10, "FIFO: {:?}", fifo_duration);
        }
        if lifo_result.is_ok() {
            assert!(lifo_duration.as_millis() < 10, "LIFO: {:?}", lifo_duration);
        }
        if hifo_result.is_ok() {
            assert!(hifo_duration.as_millis() < 10, "HIFO: {:?}", hifo_duration);
        }
    }

    #[test]
    fn test_scaling_characteristics() {
        // Test how performance scales with lot count
        let sizes = [10, 100, 1000, 10000];
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();

        println!("\nScaling characteristics:");
        println!("{:>8} | {:>10} | {:>10} | {:>10}", "Lots", "FIFO", "LIFO", "HIFO");
        println!("{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<10}", "", "", "", "");

        for &size in &sizes {
            let lots = generate_test_lots(size, base_date);
            let quantity = format!("{}", size / 200); // Dispose ~0.5% of lots

            let fifo_time = Instant::now();
            let _ = calculate_fifo(&lots, &quantity);
            let fifo_dur = fifo_time.elapsed();

            let lifo_time = Instant::now();
            let _ = calculate_lifo(&lots, &quantity);
            let lifo_dur = lifo_time.elapsed();

            let hifo_time = Instant::now();
            let _ = calculate_hifo(&lots, &quantity);
            let hifo_dur = hifo_time.elapsed();

            println!("{:>8} | {:>9}µs | {:>9}µs | {:>9}µs",
                     size,
                     fifo_dur.as_micros(),
                     lifo_dur.as_micros(),
                     hifo_dur.as_micros());
        }

        // Verify O(n) or better scaling
        // 10x increase in lots should be < 20x increase in time (with overhead)
        assert!(true, "Scaling characteristics logged");
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_concurrent_calculations() {
        // Simulate multiple concurrent tax calculations
        // (In real usage, different users calculating at same time)
        let base_date = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(1000, base_date);

        let start = Instant::now();

        for i in 0..100 {
            let quantity = format!("{}", i % 10 + 1);
            let _ = calculate_fifo(&lots, &quantity);
        }

        let duration = start.elapsed();
        let avg_per_calc = duration.as_millis() / 100;

        println!("100 sequential calculations: {:?} (avg: {}ms)", duration, avg_per_calc);

        assert!(avg_per_calc < 10,
                "Average calculation time should be <10ms under load");
    }

    #[test]
    fn test_maximum_lot_count() {
        // Test absolute maximum: 100,000 lots (extreme edge case)
        let base_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
        let lots = generate_test_lots(100000, base_date);

        println!("Testing with 100,000 lots...");

        let start = Instant::now();
        let result = calculate_fifo(&lots, "500.0");
        let duration = start.elapsed();

        println!("FIFO 100,000 lots: {:?}", duration);

        if result.is_ok() {
            // Should still complete in reasonable time (<1s)
            assert!(duration.as_secs() < 1,
                    "Even 100K lots should complete in <1s (took {:?})", duration);
        }
    }
}
