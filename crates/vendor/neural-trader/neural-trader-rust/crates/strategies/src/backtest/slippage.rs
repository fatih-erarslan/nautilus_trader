//! Slippage Models for Realistic Backtesting
//!
//! Models the impact of order execution on price:
//! - Fixed slippage
//! - Percentage-based slippage
//! - Volume-based slippage (market impact)

use crate::Direction;
use rust_decimal::Decimal;
use rust_decimal::prelude::FromPrimitive;
use serde::{Serialize, Deserialize};

/// Slippage model for backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage (unrealistic)
    None,
    /// Fixed dollar amount per share
    Fixed { amount: f64 },
    /// Percentage of price
    Percentage { rate: f64 },
    /// Volume-based market impact
    VolumeBased {
        participation_rate: f64,
        impact_coefficient: f64,
    },
    /// Combined model
    Combined {
        fixed: f64,
        percentage: f64,
        volume_impact: f64,
    },
}

impl SlippageModel {
    /// Apply slippage to execution price
    pub fn apply_slippage(
        &self,
        price: Decimal,
        direction: Direction,
        quantity: u64,
        volume: u64,
    ) -> Decimal {
        let slippage = self.calculate_slippage(price, quantity, volume);

        match direction {
            Direction::Long => price + slippage, // Pay more for buys
            Direction::Short | Direction::Close => price - slippage, // Receive less for sells
        }
    }

    /// Calculate slippage amount
    fn calculate_slippage(&self, price: Decimal, quantity: u64, volume: u64) -> Decimal {
        match self {
            SlippageModel::None => Decimal::ZERO,

            SlippageModel::Fixed { amount } => {
                Decimal::from_f64_retain(*amount).unwrap()
            }

            SlippageModel::Percentage { rate } => {
                price * Decimal::from_f64_retain(*rate).unwrap()
            }

            SlippageModel::VolumeBased {
                participation_rate,
                impact_coefficient,
            } => {
                if volume == 0 {
                    return Decimal::ZERO;
                }

                // Market impact increases with order size relative to volume
                let order_fraction = quantity as f64 / volume as f64;
                let impact = if order_fraction > *participation_rate {
                    // Larger orders have square-root market impact
                    impact_coefficient * (order_fraction / participation_rate).sqrt()
                } else {
                    0.0
                };

                price * Decimal::from_f64_retain(impact).unwrap()
            }

            SlippageModel::Combined {
                fixed,
                percentage,
                volume_impact,
            } => {
                let fixed_slip = Decimal::from_f64_retain(*fixed).unwrap();
                let pct_slip = price * Decimal::from_f64_retain(*percentage).unwrap();

                let volume_slip = if volume > 0 {
                    let order_fraction = quantity as f64 / volume as f64;
                    price * Decimal::from_f64_retain(volume_impact * order_fraction.sqrt()).unwrap()
                } else {
                    Decimal::ZERO
                };

                fixed_slip + pct_slip + volume_slip
            }
        }
    }
}

impl Default for SlippageModel {
    fn default() -> Self {
        // Realistic default: 0.05% + volume impact
        SlippageModel::Combined {
            fixed: 0.0,
            percentage: 0.0005, // 5 basis points
            volume_impact: 0.001, // 10 basis points max
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_slippage() {
        let model = SlippageModel::None;
        let price = Decimal::from(100);

        let buy_price = model.apply_slippage(price, Direction::Long, 100, 1000000);
        assert_eq!(buy_price, price);

        let sell_price = model.apply_slippage(price, Direction::Short, 100, 1000000);
        assert_eq!(sell_price, price);
    }

    #[test]
    fn test_fixed_slippage() {
        let model = SlippageModel::Fixed { amount: 0.10 };
        let price = Decimal::from(100);

        let buy_price = model.apply_slippage(price, Direction::Long, 100, 1000000);
        assert_eq!(buy_price, Decimal::from_f64_retain(100.10).unwrap());

        let sell_price = model.apply_slippage(price, Direction::Short, 100, 1000000);
        assert_eq!(sell_price, Decimal::from_f64_retain(99.90).unwrap());
    }

    #[test]
    fn test_percentage_slippage() {
        let model = SlippageModel::Percentage { rate: 0.001 }; // 0.1%
        let price = Decimal::from(100);

        let buy_price = model.apply_slippage(price, Direction::Long, 100, 1000000);
        assert!(buy_price > price);
        assert!(buy_price < Decimal::from_f64_retain(100.15).unwrap());
    }

    #[test]
    fn test_volume_based_slippage() {
        let model = SlippageModel::VolumeBased {
            participation_rate: 0.01, // 1% of volume
            impact_coefficient: 0.005, // 0.5% max impact
        };

        let price = Decimal::from(100);

        // Small order (0.01% of volume) - minimal slippage
        let small_order = model.apply_slippage(price, Direction::Long, 100, 1000000);
        assert!(small_order > price);
        assert!(small_order < Decimal::from_f64_retain(100.05).unwrap());

        // Large order (1% of volume) - more slippage
        let large_order = model.apply_slippage(price, Direction::Long, 10000, 1000000);
        assert!(large_order > small_order);
    }

    #[test]
    fn test_combined_slippage() {
        let model = SlippageModel::Combined {
            fixed: 0.05,
            percentage: 0.001,
            volume_impact: 0.001,
        };

        let price = Decimal::from(100);
        let slipped = model.apply_slippage(price, Direction::Long, 1000, 1000000);

        // Should have all three components
        assert!(slipped > price);
        assert!(slipped > Decimal::from_f64_retain(100.05).unwrap()); // At least fixed component
    }

    #[test]
    fn test_direction_impact() {
        let model = SlippageModel::Fixed { amount: 0.10 };
        let price = Decimal::from(100);

        let buy_price = model.apply_slippage(price, Direction::Long, 100, 1000000);
        let sell_price = model.apply_slippage(price, Direction::Short, 100, 1000000);

        // Buy should be higher, sell should be lower
        assert!(buy_price > price);
        assert!(sell_price < price);
        assert_eq!(buy_price - price, price - sell_price); // Symmetric
    }
}
