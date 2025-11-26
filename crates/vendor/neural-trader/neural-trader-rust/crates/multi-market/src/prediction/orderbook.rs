//! Order Book Analysis for Prediction Markets

use crate::error::{MultiMarketError, Result};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};

/// Order book depth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookDepth {
    /// Total bid volume
    pub bid_volume: Decimal,
    /// Total ask volume
    pub ask_volume: Decimal,
    /// Bid-ask spread
    pub spread: Decimal,
    /// Mid price
    pub mid_price: Decimal,
    /// Depth imbalance (-1 to 1)
    pub imbalance: Decimal,
}

/// Liquidity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityMetrics {
    /// Available liquidity
    pub available_liquidity: Decimal,
    /// Average order size
    pub avg_order_size: Decimal,
    /// Order book density
    pub density: Decimal,
    /// Market impact estimate
    pub market_impact: Decimal,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderLevel {
    pub price: Decimal,
    pub size: Decimal,
}

/// Order book analyzer
pub struct OrderbookAnalyzer;

impl OrderbookAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze order book depth
    pub fn analyze_depth(&self, bids: &[OrderLevel], asks: &[OrderLevel]) -> Result<OrderbookDepth> {
        if bids.is_empty() || asks.is_empty() {
            return Err(MultiMarketError::MarketDataError(
                "Empty order book".to_string(),
            ));
        }

        let bid_volume: Decimal = bids.iter().map(|b| b.size).sum();
        let ask_volume: Decimal = asks.iter().map(|a| a.size).sum();

        let best_bid = bids.first().unwrap().price;
        let best_ask = asks.first().unwrap().price;

        let spread = best_ask - best_bid;
        let mid_price = (best_bid + best_ask) / Decimal::from(2);

        let imbalance = if bid_volume + ask_volume > Decimal::ZERO {
            (bid_volume - ask_volume) / (bid_volume + ask_volume)
        } else {
            Decimal::ZERO
        };

        Ok(OrderbookDepth {
            bid_volume,
            ask_volume,
            spread,
            mid_price,
            imbalance,
        })
    }

    /// Calculate liquidity metrics
    pub fn calculate_liquidity(&self, bids: &[OrderLevel], asks: &[OrderLevel]) -> Result<LiquidityMetrics> {
        let total_orders = bids.len() + asks.len();
        if total_orders == 0 {
            return Err(MultiMarketError::MarketDataError(
                "No orders in book".to_string(),
            ));
        }

        let total_volume: Decimal = bids.iter().chain(asks.iter()).map(|o| o.size).sum();
        let available_liquidity = total_volume;

        let avg_order_size = total_volume / Decimal::from(total_orders);

        // Density: orders per price level
        let price_levels: std::collections::HashSet<_> = bids
            .iter()
            .chain(asks.iter())
            .map(|o| o.price)
            .collect();
        let density = Decimal::from(total_orders) / Decimal::from(price_levels.len());

        // Market impact: estimate slippage for $1000 trade
        let market_impact = self.estimate_market_impact(bids, asks, dec!(1000));

        Ok(LiquidityMetrics {
            available_liquidity,
            avg_order_size,
            density,
            market_impact,
        })
    }

    /// Estimate market impact for a given order size
    pub fn estimate_market_impact(&self, _bids: &[OrderLevel], asks: &[OrderLevel], size: Decimal) -> Decimal {
        let mut remaining = size;
        let mut total_cost = Decimal::ZERO;

        // Walk through ask levels
        for level in asks {
            if remaining <= Decimal::ZERO {
                break;
            }

            let fill_size = remaining.min(level.size);
            total_cost += fill_size * level.price;
            remaining -= fill_size;
        }

        if size > Decimal::ZERO {
            let avg_fill_price = total_cost / (size - remaining);
            let best_ask = asks.first().map(|a| a.price).unwrap_or(Decimal::ZERO);

            if best_ask > Decimal::ZERO {
                (avg_fill_price - best_ask) / best_ask
            } else {
                Decimal::ZERO
            }
        } else {
            Decimal::ZERO
        }
    }

    /// Detect order book anomalies
    pub fn detect_anomalies(&self, depth: &OrderbookDepth) -> Vec<String> {
        let mut anomalies = Vec::new();

        // Wide spread
        if depth.spread > dec!(0.1) {
            anomalies.push("Wide bid-ask spread detected".to_string());
        }

        // Extreme imbalance
        if depth.imbalance.abs() > dec!(0.8) {
            anomalies.push("Extreme order book imbalance".to_string());
        }

        // Low liquidity
        if depth.bid_volume + depth.ask_volume < dec!(1000) {
            anomalies.push("Low order book liquidity".to_string());
        }

        anomalies
    }
}

impl Default for OrderbookAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_orders() -> (Vec<OrderLevel>, Vec<OrderLevel>) {
        let bids = vec![
            OrderLevel { price: dec!(0.55), size: dec!(100) },
            OrderLevel { price: dec!(0.54), size: dec!(150) },
            OrderLevel { price: dec!(0.53), size: dec!(200) },
        ];

        let asks = vec![
            OrderLevel { price: dec!(0.56), size: dec!(120) },
            OrderLevel { price: dec!(0.57), size: dec!(130) },
            OrderLevel { price: dec!(0.58), size: dec!(180) },
        ];

        (bids, asks)
    }

    #[test]
    fn test_analyze_depth() {
        let analyzer = OrderbookAnalyzer::new();
        let (bids, asks) = create_test_orders();

        let depth = analyzer.analyze_depth(&bids, &asks).unwrap();

        assert_eq!(depth.bid_volume, dec!(450));
        assert_eq!(depth.ask_volume, dec!(430));
        assert_eq!(depth.spread, dec!(0.01));
    }

    #[test]
    fn test_liquidity_metrics() {
        let analyzer = OrderbookAnalyzer::new();
        let (bids, asks) = create_test_orders();

        let metrics = analyzer.calculate_liquidity(&bids, &asks).unwrap();

        assert_eq!(metrics.available_liquidity, dec!(880));
        assert!(metrics.avg_order_size > Decimal::ZERO);
    }
}
