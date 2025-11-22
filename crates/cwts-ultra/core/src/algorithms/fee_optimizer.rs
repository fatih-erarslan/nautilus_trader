use crate::common_types::OrderType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FeeError {
    #[error("Exchange not found: {0}")]
    ExchangeNotFound(String),
    #[error("Invalid trading pair: {0}")]
    InvalidTradingPair(String),
    #[error("Invalid order size: {0}")]
    InvalidOrderSize(f64),
    #[error("Fee calculation failed: {0}")]
    CalculationFailed(String),
    #[error("Insufficient volume data for tier calculation")]
    InsufficientVolumeData,
}

// OrderType and TradeSide now imported from common_types module

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTier {
    pub tier_name: String,
    pub volume_threshold: f64,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFeeStructure {
    pub exchange_name: String,
    pub base_maker_fee: f64,
    pub base_taker_fee: f64,
    pub tiers: Vec<FeeTier>,
    pub token_discount_rate: f64, // Discount for using exchange token
    pub minimum_fee: f64,
    pub maximum_fee_cap: f64,
    pub withdrawal_fees: HashMap<String, f64>,
    pub deposit_fees: HashMap<String, f64>,
    pub special_pairs: HashMap<String, (f64, f64)>, // (maker, taker) for specific pairs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeCalculation {
    pub exchange: String,
    pub symbol: String,
    pub order_size: f64,
    pub order_price: f64,
    pub notional_value: f64,
    pub applicable_tier: String,
    pub maker_fee_rate: f64,
    pub taker_fee_rate: f64,
    pub maker_fee_amount: f64,
    pub taker_fee_amount: f64,
    pub token_discount_applied: bool,
    pub token_discount_amount: f64,
    pub net_maker_fee: f64,
    pub net_taker_fee: f64,
    pub break_even_price_maker: f64,
    pub break_even_price_taker: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiExchangeComparison {
    pub symbol: String,
    pub order_size: f64,
    pub order_price: f64,
    pub calculations: Vec<FeeCalculation>,
    pub best_maker_exchange: String,
    pub best_taker_exchange: String,
    pub maker_savings: f64,
    pub taker_savings: f64,
    pub recommendation: ExchangeRecommendation,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRecommendation {
    pub recommended_exchange: String,
    pub recommended_order_type: OrderType,
    pub expected_fee: f64,
    pub savings_vs_worst: f64,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeData {
    pub exchange: String,
    pub thirty_day_volume: f64,
    pub seven_day_volume: f64,
    pub daily_volume: f64,
    pub timestamp: u64,
}

pub struct FeeOptimizer {
    exchanges: HashMap<String, ExchangeFeeStructure>,
    user_volumes: HashMap<String, VolumeData>,
    fee_history: HashMap<String, Vec<FeeCalculation>>,
}

impl Default for FeeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl FeeOptimizer {
    pub fn new() -> Self {
        let mut optimizer = Self {
            exchanges: HashMap::new(),
            user_volumes: HashMap::new(),
            fee_history: HashMap::new(),
        };

        // Initialize with major exchange fee structures
        optimizer.add_default_exchanges();
        optimizer
    }

    /// Add exchange fee structure
    pub fn add_exchange(&mut self, fee_structure: ExchangeFeeStructure) {
        self.exchanges
            .insert(fee_structure.exchange_name.clone(), fee_structure);
    }

    /// Update user's trading volume for an exchange
    pub fn update_user_volume(&mut self, volume_data: VolumeData) {
        self.user_volumes
            .insert(volume_data.exchange.clone(), volume_data);
    }

    /// Calculate trading fees for a specific exchange
    pub fn calculate_fees(
        &self,
        exchange: &str,
        symbol: &str,
        order_size: f64,
        order_price: f64,
        use_token_discount: bool,
    ) -> Result<FeeCalculation, FeeError> {
        if order_size <= 0.0 || order_price <= 0.0 {
            return Err(FeeError::InvalidOrderSize(order_size));
        }

        let fee_structure = self
            .exchanges
            .get(exchange)
            .ok_or_else(|| FeeError::ExchangeNotFound(exchange.to_string()))?;

        let notional_value = order_size * order_price;

        // Determine applicable tier based on user's volume
        let applicable_tier = self.get_applicable_tier(exchange, fee_structure)?;

        // Get fee rates (check for special pair rates first)
        let (maker_rate, taker_rate) =
            fee_structure.special_pairs.get(symbol).copied().unwrap_or((
                applicable_tier.maker_fee_rate,
                applicable_tier.taker_fee_rate,
            ));

        // Calculate base fees
        let maker_fee_amount = notional_value * maker_rate;
        let taker_fee_amount = notional_value * taker_rate;

        // Apply token discount if applicable
        let (token_discount_applied, token_discount_amount) = if use_token_discount {
            let maker_discount = maker_fee_amount * fee_structure.token_discount_rate;
            let taker_discount = taker_fee_amount * fee_structure.token_discount_rate;
            (true, maker_discount.max(taker_discount))
        } else {
            (false, 0.0)
        };

        // Calculate net fees after discounts
        let net_maker_fee = if use_token_discount {
            (maker_fee_amount * (1.0 - fee_structure.token_discount_rate))
                .max(fee_structure.minimum_fee)
                .min(fee_structure.maximum_fee_cap)
        } else {
            maker_fee_amount
                .max(fee_structure.minimum_fee)
                .min(fee_structure.maximum_fee_cap)
        };

        let net_taker_fee = if use_token_discount {
            (taker_fee_amount * (1.0 - fee_structure.token_discount_rate))
                .max(fee_structure.minimum_fee)
                .min(fee_structure.maximum_fee_cap)
        } else {
            taker_fee_amount
                .max(fee_structure.minimum_fee)
                .min(fee_structure.maximum_fee_cap)
        };

        // Calculate break-even prices
        let break_even_price_maker = if order_size != 0.0 {
            order_price + (net_maker_fee / order_size)
        } else {
            order_price
        };

        let break_even_price_taker = if order_size != 0.0 {
            order_price + (net_taker_fee / order_size)
        } else {
            order_price
        };

        Ok(FeeCalculation {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            order_size,
            order_price,
            notional_value,
            applicable_tier: applicable_tier.tier_name.clone(),
            maker_fee_rate: maker_rate,
            taker_fee_rate: taker_rate,
            maker_fee_amount,
            taker_fee_amount,
            token_discount_applied,
            token_discount_amount,
            net_maker_fee,
            net_taker_fee,
            break_even_price_maker,
            break_even_price_taker,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Compare fees across multiple exchanges
    pub fn compare_exchanges(
        &self,
        symbol: &str,
        order_size: f64,
        order_price: f64,
        exchanges: Option<Vec<String>>,
        use_token_discount: bool,
    ) -> Result<MultiExchangeComparison, FeeError> {
        let exchanges_to_compare =
            exchanges.unwrap_or_else(|| self.exchanges.keys().cloned().collect());

        let mut calculations = Vec::new();
        let mut errors = Vec::new();

        for exchange in &exchanges_to_compare {
            match self.calculate_fees(
                exchange,
                symbol,
                order_size,
                order_price,
                use_token_discount,
            ) {
                Ok(calc) => calculations.push(calc),
                Err(e) => errors.push((exchange.clone(), e)),
            }
        }

        if calculations.is_empty() {
            return Err(FeeError::CalculationFailed(format!(
                "No successful calculations. Errors: {:?}",
                errors
            )));
        }

        // Find best exchanges for maker and taker
        let best_maker = calculations
            .iter()
            .min_by(|a, b| a.net_maker_fee.partial_cmp(&b.net_maker_fee).unwrap())
            .unwrap();

        let best_taker = calculations
            .iter()
            .min_by(|a, b| a.net_taker_fee.partial_cmp(&b.net_taker_fee).unwrap())
            .unwrap();

        let worst_maker = calculations
            .iter()
            .max_by(|a, b| a.net_maker_fee.partial_cmp(&b.net_maker_fee).unwrap())
            .unwrap();

        let worst_taker = calculations
            .iter()
            .max_by(|a, b| a.net_taker_fee.partial_cmp(&b.net_taker_fee).unwrap())
            .unwrap();

        let maker_savings = worst_maker.net_maker_fee - best_maker.net_maker_fee;
        let taker_savings = worst_taker.net_taker_fee - best_taker.net_taker_fee;

        // Clone necessary data before the move
        let best_maker_exchange = best_maker.exchange.clone();
        let best_taker_exchange = best_taker.exchange.clone();

        // Generate recommendation
        let recommendation = self.generate_recommendation(
            &calculations,
            best_maker,
            best_taker,
            maker_savings,
            taker_savings,
        );

        Ok(MultiExchangeComparison {
            symbol: symbol.to_string(),
            order_size,
            order_price,
            calculations,
            best_maker_exchange,
            best_taker_exchange,
            maker_savings,
            taker_savings,
            recommendation,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }

    /// Calculate optimal order splitting across exchanges
    pub fn calculate_optimal_splits(
        &self,
        symbol: &str,
        total_order_size: f64,
        order_price: f64,
        exchanges: Vec<String>,
        target_order_type: OrderType,
    ) -> Result<Vec<(String, f64, f64)>, FeeError> {
        let mut exchange_calculations = Vec::new();

        // Calculate fees for each exchange
        for exchange in &exchanges {
            let calc =
                self.calculate_fees(exchange, symbol, total_order_size, order_price, true)?;

            let fee_rate = match target_order_type {
                OrderType::Market => calc.taker_fee_rate,
                OrderType::Limit => calc.maker_fee_rate,
                _ => calc.taker_fee_rate, // Default to taker for stop orders
            };

            exchange_calculations.push((exchange.clone(), fee_rate, calc));
        }

        // Sort by fee rate (lowest first)
        exchange_calculations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Simple equal split strategy for now
        // In practice, this could be more sophisticated based on liquidity, slippage, etc.
        let split_size = total_order_size / exchanges.len() as f64;
        let splits: Vec<(String, f64, f64)> = exchange_calculations
            .into_iter()
            .map(|(exchange, fee_rate, _)| (exchange, split_size, fee_rate))
            .collect();

        Ok(splits)
    }

    /// Calculate net profit after fees
    pub fn calculate_net_profit(
        &self,
        entry_calc: &FeeCalculation,
        exit_price: f64,
        is_long: bool,
        use_maker_exit: bool,
    ) -> Result<f64, FeeError> {
        let entry_fee = if use_maker_exit {
            entry_calc.net_maker_fee
        } else {
            entry_calc.net_taker_fee
        };

        // Calculate exit fees
        let exit_calc = self.calculate_fees(
            &entry_calc.exchange,
            &entry_calc.symbol,
            entry_calc.order_size,
            exit_price,
            entry_calc.token_discount_applied,
        )?;

        let exit_fee = if use_maker_exit {
            exit_calc.net_maker_fee
        } else {
            exit_calc.net_taker_fee
        };

        // Calculate gross profit/loss
        let gross_pnl = if is_long {
            entry_calc.order_size * (exit_price - entry_calc.order_price)
        } else {
            entry_calc.order_size * (entry_calc.order_price - exit_price)
        };

        // Net profit after fees
        let net_profit = gross_pnl - entry_fee - exit_fee;

        Ok(net_profit)
    }

    /// Get volume-based tier for user
    fn get_applicable_tier(
        &self,
        exchange: &str,
        fee_structure: &ExchangeFeeStructure,
    ) -> Result<FeeTier, FeeError> {
        let user_volume = self
            .user_volumes
            .get(exchange)
            .map(|v| v.thirty_day_volume)
            .unwrap_or(0.0);

        // Find the highest tier the user qualifies for
        let mut applicable_tier = FeeTier {
            tier_name: "Base".to_string(),
            volume_threshold: 0.0,
            maker_fee_rate: fee_structure.base_maker_fee,
            taker_fee_rate: fee_structure.base_taker_fee,
            description: "Base tier".to_string(),
        };

        for tier in &fee_structure.tiers {
            if user_volume >= tier.volume_threshold {
                applicable_tier = tier.clone();
            } else {
                break; // Tiers should be sorted by volume threshold
            }
        }

        Ok(applicable_tier)
    }

    /// Generate trading recommendation
    fn generate_recommendation(
        &self,
        _calculations: &[FeeCalculation],
        best_maker: &FeeCalculation,
        best_taker: &FeeCalculation,
        maker_savings: f64,
        taker_savings: f64,
    ) -> ExchangeRecommendation {
        let (recommended_exchange, recommended_order_type, expected_fee, savings, reasoning) =
            if maker_savings > taker_savings * 1.5 {
                // Significant maker savings
                (
                    best_maker.exchange.clone(),
                    OrderType::Limit,
                    best_maker.net_maker_fee,
                    maker_savings,
                    format!(
                        "Use limit orders on {} to save {:.4} in fees vs market orders",
                        best_maker.exchange, maker_savings
                    ),
                )
            } else if taker_savings > 0.0 {
                // Taker savings available
                (
                    best_taker.exchange.clone(),
                    OrderType::Market,
                    best_taker.net_taker_fee,
                    taker_savings,
                    format!(
                        "Use {} for market orders to save {:.4} in fees",
                        best_taker.exchange, taker_savings
                    ),
                )
            } else {
                // No clear advantage, default to best maker
                (
                    best_maker.exchange.clone(),
                    OrderType::Limit,
                    best_maker.net_maker_fee,
                    0.0,
                    "No significant fee advantage between exchanges".to_string(),
                )
            };

        ExchangeRecommendation {
            recommended_exchange,
            recommended_order_type,
            expected_fee,
            savings_vs_worst: savings,
            reasoning,
        }
    }

    /// Add default exchange fee structures
    fn add_default_exchanges(&mut self) {
        // Binance fee structure
        let binance = ExchangeFeeStructure {
            exchange_name: "Binance".to_string(),
            base_maker_fee: 0.001, // 0.1%
            base_taker_fee: 0.001, // 0.1%
            tiers: vec![
                FeeTier {
                    tier_name: "VIP 1".to_string(),
                    volume_threshold: 100_000.0,
                    maker_fee_rate: 0.0009,
                    taker_fee_rate: 0.001,
                    description: "≥100K USDT 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "VIP 2".to_string(),
                    volume_threshold: 500_000.0,
                    maker_fee_rate: 0.0008,
                    taker_fee_rate: 0.001,
                    description: "≥500K USDT 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "VIP 3".to_string(),
                    volume_threshold: 1_000_000.0,
                    maker_fee_rate: 0.0007,
                    taker_fee_rate: 0.0009,
                    description: "≥1M USDT 30-day volume".to_string(),
                },
            ],
            token_discount_rate: 0.25, // 25% discount with BNB
            minimum_fee: 0.0001,
            maximum_fee_cap: 100.0,
            withdrawal_fees: HashMap::from([
                ("BTC".to_string(), 0.0005),
                ("ETH".to_string(), 0.005),
                ("USDT".to_string(), 1.0),
            ]),
            deposit_fees: HashMap::new(),
            special_pairs: HashMap::new(),
        };

        // Coinbase Pro fee structure
        let coinbase_pro = ExchangeFeeStructure {
            exchange_name: "Coinbase Pro".to_string(),
            base_maker_fee: 0.005, // 0.5%
            base_taker_fee: 0.005, // 0.5%
            tiers: vec![
                FeeTier {
                    tier_name: "Tier 1".to_string(),
                    volume_threshold: 10_000.0,
                    maker_fee_rate: 0.004,
                    taker_fee_rate: 0.005,
                    description: "≥$10K 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "Tier 2".to_string(),
                    volume_threshold: 50_000.0,
                    maker_fee_rate: 0.0035,
                    taker_fee_rate: 0.004,
                    description: "≥$50K 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "Tier 3".to_string(),
                    volume_threshold: 100_000.0,
                    maker_fee_rate: 0.003,
                    taker_fee_rate: 0.0035,
                    description: "≥$100K 30-day volume".to_string(),
                },
            ],
            token_discount_rate: 0.0, // No token discount
            minimum_fee: 0.01,
            maximum_fee_cap: 10000.0,
            withdrawal_fees: HashMap::from([
                ("BTC".to_string(), 0.0),
                ("ETH".to_string(), 0.0),
                ("USDT".to_string(), 2.5),
            ]),
            deposit_fees: HashMap::new(),
            special_pairs: HashMap::new(),
        };

        // Kraken fee structure
        let kraken = ExchangeFeeStructure {
            exchange_name: "Kraken".to_string(),
            base_maker_fee: 0.0016, // 0.16%
            base_taker_fee: 0.0026, // 0.26%
            tiers: vec![
                FeeTier {
                    tier_name: "Starter".to_string(),
                    volume_threshold: 50_000.0,
                    maker_fee_rate: 0.0014,
                    taker_fee_rate: 0.0024,
                    description: "≥$50K 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "Intermediate".to_string(),
                    volume_threshold: 100_000.0,
                    maker_fee_rate: 0.0012,
                    taker_fee_rate: 0.0022,
                    description: "≥$100K 30-day volume".to_string(),
                },
                FeeTier {
                    tier_name: "Pro".to_string(),
                    volume_threshold: 250_000.0,
                    maker_fee_rate: 0.001,
                    taker_fee_rate: 0.002,
                    description: "≥$250K 30-day volume".to_string(),
                },
            ],
            token_discount_rate: 0.0,
            minimum_fee: 0.0001,
            maximum_fee_cap: 1000.0,
            withdrawal_fees: HashMap::from([
                ("BTC".to_string(), 0.00015),
                ("ETH".to_string(), 0.0025),
                ("USDT".to_string(), 2.5),
            ]),
            deposit_fees: HashMap::new(),
            special_pairs: HashMap::new(),
        };

        self.add_exchange(binance);
        self.add_exchange(coinbase_pro);
        self.add_exchange(kraken);
    }

    /// Get fee history for analysis
    pub fn get_fee_history(&self, exchange: &str) -> Option<&Vec<FeeCalculation>> {
        self.fee_history.get(exchange)
    }

    /// Record fee calculation for history
    pub fn record_fee_calculation(&mut self, calculation: FeeCalculation) {
        let history = self
            .fee_history
            .entry(calculation.exchange.clone())
            .or_default();

        history.push(calculation);

        // Keep only recent history (last 1000 calculations)
        if history.len() > 1000 {
            history.remove(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_calculation() {
        let optimizer = FeeOptimizer::new();

        let calc = optimizer
            .calculate_fees(
                "Binance", "BTCUSDT", 1.0, 50000.0, false, // No token discount
            )
            .unwrap();

        assert_eq!(calc.notional_value, 50000.0);
        assert!(calc.maker_fee_amount > 0.0);
        assert!(calc.taker_fee_amount > 0.0);
    }

    #[test]
    fn test_token_discount() {
        let optimizer = FeeOptimizer::new();

        let without_discount = optimizer
            .calculate_fees("Binance", "BTCUSDT", 1.0, 50000.0, false)
            .unwrap();

        let with_discount = optimizer
            .calculate_fees("Binance", "BTCUSDT", 1.0, 50000.0, true)
            .unwrap();

        assert!(with_discount.net_maker_fee < without_discount.net_maker_fee);
        assert!(with_discount.token_discount_applied);
    }

    #[test]
    fn test_exchange_comparison() {
        let optimizer = FeeOptimizer::new();

        let comparison = optimizer
            .compare_exchanges("BTCUSDT", 1.0, 50000.0, None, false)
            .unwrap();

        assert!(!comparison.calculations.is_empty());
        assert!(!comparison.best_maker_exchange.is_empty());
        assert!(!comparison.best_taker_exchange.is_empty());
    }

    #[test]
    fn test_volume_tier_calculation() {
        let mut optimizer = FeeOptimizer::new();

        // Add high volume for user
        optimizer.update_user_volume(VolumeData {
            exchange: "Binance".to_string(),
            thirty_day_volume: 150_000.0,
            seven_day_volume: 35_000.0,
            daily_volume: 5_000.0,
            timestamp: 0,
        });

        let calc = optimizer
            .calculate_fees("Binance", "BTCUSDT", 1.0, 50000.0, false)
            .unwrap();

        // Should get VIP 1 tier with lower maker fee
        assert_eq!(calc.applicable_tier, "VIP 1");
        assert!(calc.maker_fee_rate < 0.001); // Lower than base 0.1%
    }

    #[test]
    fn test_net_profit_calculation() {
        let optimizer = FeeOptimizer::new();

        let entry_calc = optimizer
            .calculate_fees("Binance", "BTCUSDT", 1.0, 50000.0, false)
            .unwrap();

        let net_profit = optimizer
            .calculate_net_profit(
                &entry_calc,
                55000.0, // Exit price
                true,    // Long position
                false,   // Use taker for exit
            )
            .unwrap();

        // Profit should be close to 5000 minus fees
        assert!(net_profit > 4900.0 && net_profit < 5000.0);
    }
}
