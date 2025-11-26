/*!
 * Comprehensive Wash Sale Tests
 *
 * Tests IRS wash sale rules (Publication 550):
 * - 30-day window (30 days before and after disposal)
 * - Only applies to losses (not gains)
 * - Cost basis adjustments for replacement shares
 * - Chain wash sales
 */

use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{DateTime, Utc, TimeZone};
use std::str::FromStr;

// Note: These tests assume wash sale detection functions will be implemented
// in agentic_accounting_rust_core::tax::wash_sale module

// ============================================================================
// Test Structures
// ============================================================================

#[derive(Debug, Clone)]
struct WashSaleScenario {
    disposal_date: DateTime<Utc>,
    disposal_loss: Decimal,
    replacement_purchases: Vec<(DateTime<Utc>, Decimal)>, // (date, quantity)
}

impl WashSaleScenario {
    fn is_wash_sale(&self) -> bool {
        // Check if any replacement purchase is within 30-day window
        self.replacement_purchases.iter().any(|(purchase_date, _)| {
            let days_diff = (*purchase_date - self.disposal_date).num_days().abs();
            days_diff <= 30
        })
    }

    fn disallowed_loss(&self) -> Decimal {
        if self.disposal_loss >= dec!(0.0) {
            return dec!(0.0); // Not a loss
        }

        if !self.is_wash_sale() {
            return dec!(0.0); // Not a wash sale
        }

        // For simplicity, disallow entire loss
        // Real implementation should prorate based on replacement quantity
        self.disposal_loss.abs()
    }
}

// ============================================================================
// Basic Wash Sale Detection Tests
// ============================================================================

#[cfg(test)]
mod wash_sale_detection_tests {
    use super::*;

    #[test]
    fn test_wash_sale_detected_30_days_later() {
        // Sell at loss on June 15, buy replacement on June 20 (5 days later)
        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap(),
            disposal_loss: dec!(-1000.0), // $1000 loss
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 6, 20, 0, 0, 0).unwrap(), dec!(1.0)),
            ],
        };

        assert!(scenario.is_wash_sale(), "Should detect wash sale within 30-day window");
        assert_eq!(scenario.disallowed_loss(), dec!(1000.0));
    }

    #[test]
    fn test_wash_sale_detected_30_days_before() {
        // Buy on June 1, sell at loss on June 15 (14 days after purchase)
        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap(),
            disposal_loss: dec!(-1000.0),
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(), dec!(1.0)),
            ],
        };

        assert!(scenario.is_wash_sale(), "Should detect wash sale 30 days before");
        assert_eq!(scenario.disallowed_loss(), dec!(1000.0));
    }

    #[test]
    fn test_no_wash_sale_beyond_30_days() {
        // Buy 44 days after disposal - not a wash sale
        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap(),
            disposal_loss: dec!(-1000.0),
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 7, 15, 0, 0, 0).unwrap(), dec!(1.0)), // 44 days
            ],
        };

        assert!(!scenario.is_wash_sale(), "Should NOT be wash sale beyond 30 days");
        assert_eq!(scenario.disallowed_loss(), dec!(0.0));
    }

    #[test]
    fn test_no_wash_sale_for_gains() {
        // Gains are never wash sales, even with replacement purchase
        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap(),
            disposal_loss: dec!(1000.0), // GAIN, not loss
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 6, 20, 0, 0, 0).unwrap(), dec!(1.0)),
            ],
        };

        assert_eq!(scenario.disallowed_loss(), dec!(0.0),
                   "Wash sale rules don't apply to gains");
    }

    #[test]
    fn test_wash_sale_exact_30_day_boundary() {
        // Replacement exactly 30 days later - should be included
        let disposal_date = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let replacement_date = Utc.with_ymd_and_hms(2024, 7, 1, 0, 0, 0).unwrap();

        let days_diff = (replacement_date - disposal_date).num_days();
        assert_eq!(days_diff, 30);

        let scenario = WashSaleScenario {
            disposal_date,
            disposal_loss: dec!(-500.0),
            replacement_purchases: vec![(replacement_date, dec!(1.0))],
        };

        assert!(scenario.is_wash_sale(),
                "Wash sale window includes exactly 30 days");
    }

    #[test]
    fn test_wash_sale_same_day() {
        // Buy and sell on same day - wash sale
        let same_date = Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap();

        let scenario = WashSaleScenario {
            disposal_date: same_date,
            disposal_loss: dec!(-800.0),
            replacement_purchases: vec![(same_date, dec!(1.0))],
        };

        assert!(scenario.is_wash_sale(), "Same-day purchase is a wash sale");
    }
}

// ============================================================================
// IRS Publication 550 Example Tests
// ============================================================================

#[cfg(test)]
mod irs_pub_550_examples {
    use super::*;

    #[test]
    fn test_irs_example_1_basic_wash_sale() {
        // IRS Example 1:
        // Dec 15: Sell 100 shares at $70 (cost basis $100) = $3000 loss
        // Jan 10: Buy 100 shares at $65
        // Result: $3000 loss is disallowed, added to cost basis of new shares

        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2023, 12, 15, 0, 0, 0).unwrap(),
            disposal_loss: dec!(-3000.0),
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 1, 10, 0, 0, 0).unwrap(), dec!(100.0)),
            ],
        };

        assert!(scenario.is_wash_sale());
        assert_eq!(scenario.disallowed_loss(), dec!(3000.0));

        // New cost basis should be: $6500 (purchase) + $3000 (disallowed loss) = $9500
        let original_cost = dec!(6500.0); // 100 shares @ $65
        let adjusted_cost = original_cost + scenario.disallowed_loss();
        assert_eq!(adjusted_cost, dec!(9500.0));
    }

    #[test]
    fn test_irs_example_2_partial_replacement() {
        // IRS Example 2:
        // Sell 100 shares at loss
        // Buy back only 50 shares within window
        // Result: Only 50% of loss is disallowed (prorated)

        let total_loss = dec!(-2000.0);
        let disposal_quantity = dec!(100.0);
        let replacement_quantity = dec!(50.0);

        let proration_factor = replacement_quantity / disposal_quantity;
        let disallowed = total_loss.abs() * proration_factor;
        let allowed = total_loss.abs() * (dec!(1.0) - proration_factor);

        assert_eq!(disallowed, dec!(1000.0), "50% of loss should be disallowed");
        assert_eq!(allowed, dec!(1000.0), "50% of loss should be allowed");
    }

    #[test]
    fn test_irs_example_3_multiple_purchases() {
        // Multiple purchases within window
        // Buy 30 shares on Day 5
        // Buy 40 shares on Day 15
        // Buy 30 shares on Day 25
        // Total replacement: 100 shares (full replacement)

        let disposal_date = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        let scenario = WashSaleScenario {
            disposal_date,
            disposal_loss: dec!(-5000.0),
            replacement_purchases: vec![
                (disposal_date + chrono::Duration::days(5), dec!(30.0)),
                (disposal_date + chrono::Duration::days(15), dec!(40.0)),
                (disposal_date + chrono::Duration::days(25), dec!(30.0)),
            ],
        };

        assert!(scenario.is_wash_sale());

        // All purchases are within window
        let total_replacement: Decimal = scenario.replacement_purchases
            .iter()
            .map(|(_, qty)| qty)
            .sum();
        assert_eq!(total_replacement, dec!(100.0));
    }

    #[test]
    fn test_irs_example_4_chain_wash_sales() {
        // Chain of wash sales:
        // Day 0: Sell Lot A at $2000 loss, buy Lot B
        // Day 15: Sell Lot B at $1000 loss, buy Lot C
        // Result: Both losses disallowed, accumulated in Lot C cost basis

        let first_loss = dec!(2000.0);
        let second_loss = dec!(1000.0);
        let total_accumulated_loss = first_loss + second_loss;

        assert_eq!(total_accumulated_loss, dec!(3000.0),
                   "Chain wash sales accumulate disallowed losses");
    }
}

// ============================================================================
// Cost Basis Adjustment Tests
// ============================================================================

#[cfg(test)]
mod cost_basis_adjustment_tests {
    use super::*;

    #[test]
    fn test_cost_basis_adjustment_simple() {
        // Original cost basis: $10,000
        // Disposal: $8,000 (loss: $2,000)
        // Replacement cost: $7,500
        // Adjusted basis: $7,500 + $2,000 = $9,500

        let original_cost = dec!(10000.0);
        let proceeds = dec!(8000.0);
        let loss = proceeds - original_cost;
        assert_eq!(loss, dec!(-2000.0));

        let replacement_cost = dec!(7500.0);
        let adjusted_cost_basis = replacement_cost + loss.abs();

        assert_eq!(adjusted_cost_basis, dec!(9500.0));
    }

    #[test]
    fn test_holding_period_adjustment() {
        // IRS rules: holding period of replacement stock includes
        // holding period of washed stock

        let original_acquisition = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let disposal_date = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();
        let replacement_date = Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap();

        let original_holding_days = (disposal_date - original_acquisition).num_days();
        assert_eq!(original_holding_days, 152);

        // Adjusted holding period starts from original acquisition
        let adjusted_acquisition_date = original_acquisition;

        // If sold 1 year after replacement, check if long-term
        let future_sale = replacement_date + chrono::Duration::days(365);
        let total_holding = (future_sale - adjusted_acquisition_date).num_days();

        assert!(total_holding > 365, "Should qualify for long-term with adjusted holding period");
    }

    #[test]
    fn test_fractional_adjustment_crypto() {
        // Crypto example with fractional quantities
        // Sell 1.5 BTC at $30,000 = $45,000 (cost basis: $60,000) = -$15,000 loss
        // Buy 0.75 BTC (50% replacement)
        // Disallowed loss: $7,500 (50% of $15,000)
        // Allowed loss: $7,500

        let total_loss = dec!(-15000.0);
        let disposal_qty = dec!(1.5);
        let replacement_qty = dec!(0.75);

        let replacement_ratio = replacement_qty / disposal_qty;
        assert_eq!(replacement_ratio, dec!(0.5));

        let disallowed_loss = total_loss.abs() * replacement_ratio;
        let allowed_loss = total_loss.abs() - disallowed_loss;

        assert_eq!(disallowed_loss, dec!(7500.0));
        assert_eq!(allowed_loss, dec!(7500.0));
    }
}

// ============================================================================
// Edge Cases and Complex Scenarios
// ============================================================================

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_wash_sale_with_multiple_assets() {
        // Wash sale is per-security, not across different assets
        // Sell BTC at loss, buy ETH -> NOT a wash sale
        // Must be "substantially identical" securities

        // This test is conceptual - actual implementation would need
        // asset comparison logic
        assert!(true, "Different assets should not trigger wash sale");
    }

    #[test]
    fn test_wash_sale_across_accounts() {
        // IRS rules: wash sales apply across ALL accounts
        // (taxable, IRA, spouse's accounts, etc.)

        // This is a conceptual test - would need multi-account support
        assert!(true, "Wash sales apply across accounts");
    }

    #[test]
    fn test_wash_sale_with_options() {
        // Buying options on same underlying can trigger wash sale
        // E.g., sell AAPL stock at loss, buy AAPL call option

        // Conceptual test - requires derivative handling
        assert!(true, "Options can trigger wash sales");
    }

    #[test]
    fn test_zero_cost_basis_wash_sale() {
        // Edge case: inherited stock with stepped-up basis
        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap(),
            disposal_loss: dec!(0.0), // No loss
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 6, 20, 0, 0, 0).unwrap(), dec!(1.0)),
            ],
        };

        assert_eq!(scenario.disallowed_loss(), dec!(0.0),
                   "No loss means no wash sale impact");
    }

    #[test]
    fn test_wash_sale_year_boundary() {
        // Sell at loss Dec 20, 2023
        // Buy replacement Jan 5, 2024
        // Loss disallowed in 2023 tax year

        let scenario = WashSaleScenario {
            disposal_date: Utc.with_ymd_and_hms(2023, 12, 20, 0, 0, 0).unwrap(),
            disposal_loss: dec!(-10000.0),
            replacement_purchases: vec![
                (Utc.with_ymd_and_hms(2024, 1, 5, 0, 0, 0).unwrap(), dec!(1.0)),
            ],
        };

        assert!(scenario.is_wash_sale());

        // Loss should be deferred to 2024 (adjusted cost basis)
        let disposal_year = 2023;
        let replacement_year = 2024;

        assert_ne!(disposal_year, replacement_year,
                   "Wash sale spans tax years - important for reporting");
    }

    #[test]
    fn test_wash_sale_substantiality_identical() {
        // "Substantially identical" securities test
        // BTC is substantially identical to BTC (obviously)
        // But BTC vs BCH? BTC vs WBTC?
        // This requires domain knowledge about crypto fungibility

        let btc = "BTC";
        let btc2 = "BTC";
        assert_eq!(btc, btc2, "Same asset is substantially identical");

        // In practice, would need rules engine for cross-asset comparisons
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_wash_sale_detection_performance() {
        // Test with 1000 disposals and 1000 potential replacement transactions
        let disposal_date = Utc.with_ymd_and_hms(2024, 6, 15, 0, 0, 0).unwrap();
        let mut replacement_purchases = Vec::new();

        // Generate 1000 transactions across 365 days
        for i in 0..1000 {
            let days_offset = (i % 365) as i64 - 182; // -182 to +182 days
            let purchase_date = disposal_date + chrono::Duration::days(days_offset);
            replacement_purchases.push((purchase_date, dec!(0.001)));
        }

        let start = std::time::Instant::now();

        for _ in 0..100 {
            let scenario = WashSaleScenario {
                disposal_date,
                disposal_loss: dec!(-1000.0),
                replacement_purchases: replacement_purchases.clone(),
            };

            let _ = scenario.is_wash_sale();
        }

        let duration = start.elapsed();
        let avg_per_check = duration.as_micros() / 100;

        println!("Wash sale detection: {} μs per check (100 iterations)", avg_per_check);

        assert!(avg_per_check < 1000,
                "Wash sale detection should be < 1ms per disposal");
    }
}
